"""Numba-fused host executor for the compiled G1 flat-walk task.

This executor is deliberately task-owned.  It shares the *cold-path* task
schema with :mod:`managed_reference`, but it never instantiates, imports, or
delegates to the reference kernel at runtime.  In particular, the hot path
only accepts a typed :class:`~unilab.base.backend.batch.StateBatch` and
runtime-owned arrays.  It has no backend, environment, asset, model, selector,
or registry reference to fall back to.

The executor keeps the canonical lifecycle in ``ManagedReferenceRuntime``.
"Reference" in that runtime name describes the lifecycle owner, not the math
implementation: this module supplies an independent executor key and a
separate Numba implementation of action, reward, termination, and observation
math.  Unsupported profiles and unavailable Numba fail during construction or
binding; no exception is converted into a NumPy/reference fallback.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

try:  # pragma: no cover - the negative path is exercised by fault tests.
    from numba import njit

    NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover - optional-import boundary.
    njit = None  # type: ignore[assignment]
    NUMBA_AVAILABLE = False

from unilab.base.backend import (
    BoundMutationPlan,
    BoundMutationValueBuffers,
    BufferLifetime,
    ControlSpec,
    ExecutionProfile,
    RowSelection,
    SimulationStateMutationBatch,
    StateBatch,
    TypedBackendMutationBatch,
)
from unilab.base.backend.base import SimBackend
from unilab.dtype_config import get_global_dtype
from unilab.manager import (
    BackendEntityResolver,
    ManagedKernelBinding,
    ManagedMetric,
    ManagedReferenceRuntime,
    ManagedResetRequest,
    ManagedRuntimeBuffer,
    ManagerContractError,
    PolicySpec,
    TaskCompiler,
    TaskSpec,
    TensorSpec,
    TermDefinition,
    TermInvocation,
    TermPhase,
    TermRegistry,
    TermRole,
)
from unilab.manager.plan import CompiledTaskPlan
from unilab.utils.rotation import np_quat_mul, np_yaw_to_quat

from .joystick import G1WalkEnvCfg
from .managed_reference import (
    _ACTOR_WIDTH,
    _CRITIC_WIDTH,
    _RESET_TERM,
    _ROOT_RESET_SUFFIXES,
    _STATE_KEYS,
    G1ManagedReferenceError,
    _action_scale,
    _g1_reset_templates,
    _g1_selectors,
    _g1_state_requirements,
    _G1KernelConfig,
    _G1ResetSample,
    _G1StateViews,
    _kernel_config,
    _manager_buffer,
    _reset_suffix_for_dof,
    _reset_term_key,
    _validate_reference_profile,
)

G1_MANAGED_FUSED_EXECUTOR_KEY = "host.numba.g1-walk-flat.v1"
"""Explicit executor identity; it is never inferred from a Python object."""


class G1ManagedFusedError(ManagerContractError):
    """Raised when the strict G1 fused-host profile cannot be dispatched."""


# The values are frozen into a compact, task-owned array in the cold path.  A
# numeric term table lets the Numba kernel remain independent from registry or
# string dispatch in the hot path.
_TERM_TRACKING_LIN_VEL = 0
_TERM_TRACKING_ANG_VEL = 1
_TERM_FORWARD_PROGRESS = 2
_TERM_UNDER_SPEED = 3
_TERM_LIN_VEL_Z = 4
_TERM_ORIENTATION = 5
_TERM_ANG_VEL_XY = 6
_TERM_ACTION_RATE = 7
_TERM_BASE_HEIGHT = 8
_TERM_POSE = 9
_TERM_UPPER_BODY_POSE = 10
_TERM_CLOSE_FEET_XY = 11
_TERM_FEET_PHASE = 12
_TERM_FEET_PHASE_CONTRAST = 13
_TERM_ALIVE = 14

_TERM_CODES = {
    "tracking_lin_vel": _TERM_TRACKING_LIN_VEL,
    "tracking_ang_vel": _TERM_TRACKING_ANG_VEL,
    "forward_progress": _TERM_FORWARD_PROGRESS,
    "under_speed": _TERM_UNDER_SPEED,
    "lin_vel_z": _TERM_LIN_VEL_Z,
    "orientation": _TERM_ORIENTATION,
    "penalty_orientation": _TERM_ORIENTATION,
    "ang_vel_xy": _TERM_ANG_VEL_XY,
    "penalty_ang_vel_xy": _TERM_ANG_VEL_XY,
    "action_rate": _TERM_ACTION_RATE,
    "penalty_action_rate": _TERM_ACTION_RATE,
    "base_height": _TERM_BASE_HEIGHT,
    "pose": _TERM_POSE,
    "upper_body_pose": _TERM_UPPER_BODY_POSE,
    "penalty_close_feet_xy": _TERM_CLOSE_FEET_XY,
    "feet_phase": _TERM_FEET_PHASE,
    "feet_phase_contrast": _TERM_FEET_PHASE_CONTRAST,
    "alive": _TERM_ALIVE,
}


if NUMBA_AVAILABLE:

    # These kernels deliberately use serial Numba loops rather than ``prange``.
    # The Phase-0 frozen host uses sixteen Numba threads but its available TBB
    # runtime cannot service the parallel scheduler efficiently; entering five
    # tiny parallel regions per control step dominated the full lifecycle at
    # batches 128--4096.  Serial kernels retain nopython/GIL-free math while
    # avoiding that scheduler cost.  Any future parallel strategy must be
    # introduced as an explicit cold-bound dispatch mode and re-pass the
    # independent-process Phase-4 host gate; it must not silently replace this
    # measured default.

    @njit(inline="always", cache=True, nogil=True)  # type: ignore[misc]
    def _positive(value):
        return value if value > 0.0 else 0.0

    @njit(inline="always", cache=True, nogil=True)  # type: ignore[misc]
    def _bezier_height(phase, swing_height):
        two_pi = 2.0 * math.pi
        normalized = np.fmod(phase + math.pi, two_pi) - math.pi
        x = (normalized + math.pi) / two_pi
        if x <= 0.5:
            t = 2.0 * x
            bezier = t * t * t + 3.0 * (t * t * (1.0 - t))
            return swing_height * bezier
        t = 2.0 * x - 1.0
        bezier = t * t * t + 3.0 * (t * t * (1.0 - t))
        return swing_height * (1.0 - bezier)

    @njit(cache=True, nogil=True)  # type: ignore[misc]
    def _apply_action_kernel(
        actions,
        action_scale,
        default_angles,
        gait_phase_delta,
        current_actions,
        last_actions,
        gait_phase,
        control_out,
    ):
        two_pi = 2.0 * math.pi
        for row in range(actions.shape[0]):
            for action_index in range(actions.shape[1]):
                value = actions[row, action_index]
                last_actions[row, action_index] = current_actions[row, action_index]
                current_actions[row, action_index] = value
                control_out[row, action_index] = (
                    value * action_scale[action_index] + default_angles[action_index]
                )
            for phase_index in range(2):
                value = gait_phase[row, phase_index] + gait_phase_delta
                gait_phase[row, phase_index] = value - math.floor(value / two_pi) * two_pi

    @njit(inline="always", cache=True, nogil=True)  # type: ignore[misc]
    def _write_observation_row(
        row,
        target_row,
        pelvis_local_linear_velocity,
        torso_gyro,
        torso_upvector,
        dof_position,
        dof_angular_velocity,
        commands,
        current_actions,
        gait_phase,
        default_angles,
        gyro_scale,
        dof_velocity_scale,
        linear_velocity_scale,
        actor_out,
        critic_out,
    ):
        action_dim = dof_position.shape[1]
        cursor = 0
        for component in range(3):
            actor_out[target_row, cursor + component] = torso_gyro[row, component] * gyro_scale
        cursor += 3
        for component in range(3):
            actor_out[target_row, cursor + component] = -torso_upvector[row, component]
        cursor += 3
        for action_index in range(action_dim):
            actor_out[target_row, cursor + action_index] = (
                dof_position[row, action_index] - default_angles[action_index]
            )
        cursor += action_dim
        for action_index in range(action_dim):
            actor_out[target_row, cursor + action_index] = (
                dof_angular_velocity[row, action_index] * dof_velocity_scale
            )
        cursor += action_dim
        for action_index in range(action_dim):
            actor_out[target_row, cursor + action_index] = current_actions[row, action_index]
        cursor += action_dim
        for component in range(3):
            actor_out[target_row, cursor + component] = commands[row, component]
        cursor += 3
        actor_out[target_row, cursor] = gait_phase[row, 0]
        actor_out[target_row, cursor + 1] = gait_phase[row, 1]

        cursor = 0
        for component in range(3):
            critic_out[target_row, cursor + component] = torso_gyro[row, component] * gyro_scale
        cursor += 3
        for component in range(3):
            critic_out[target_row, cursor + component] = -torso_upvector[row, component]
        cursor += 3
        for action_index in range(action_dim):
            critic_out[target_row, cursor + action_index] = (
                dof_position[row, action_index] - default_angles[action_index]
            )
        cursor += action_dim
        for action_index in range(action_dim):
            critic_out[target_row, cursor + action_index] = (
                dof_angular_velocity[row, action_index] * dof_velocity_scale
            )
        cursor += action_dim
        for action_index in range(action_dim):
            critic_out[target_row, cursor + action_index] = current_actions[row, action_index]
        cursor += action_dim
        for component in range(3):
            critic_out[target_row, cursor + component] = commands[row, component]
        cursor += 3
        critic_out[target_row, cursor] = gait_phase[row, 0]
        critic_out[target_row, cursor + 1] = gait_phase[row, 1]
        cursor += 2
        for component in range(3):
            critic_out[target_row, cursor + component] = (
                pelvis_local_linear_velocity[row, component] * linear_velocity_scale
            )

    @njit(cache=True, nogil=True)  # type: ignore[misc]
    def _write_observations_kernel(
        row_indices,
        pelvis_local_linear_velocity,
        torso_gyro,
        torso_upvector,
        dof_position,
        dof_angular_velocity,
        commands,
        current_actions,
        gait_phase,
        default_angles,
        gyro_scale,
        dof_velocity_scale,
        linear_velocity_scale,
        actor_out,
        critic_out,
    ):
        for row in range(row_indices.shape[0]):
            _write_observation_row(
                row,
                row_indices[row],
                pelvis_local_linear_velocity,
                torso_gyro,
                torso_upvector,
                dof_position,
                dof_angular_velocity,
                commands,
                current_actions,
                gait_phase,
                default_angles,
                gyro_scale,
                dof_velocity_scale,
                linear_velocity_scale,
                actor_out,
                critic_out,
            )

    @njit(cache=True, nogil=True)  # type: ignore[misc]
    def _compute_terminal_kernel(
        root_position,
        torso_upvector,
        pelvis_local_linear_velocity,
        torso_gyro,
        left_foot_position,
        right_foot_position,
        dof_position,
        dof_angular_velocity,
        commands,
        current_actions,
        last_actions,
        gait_phase,
        default_angles,
        pose_weights,
        upper_body_pose_weights,
        term_codes,
        term_scales,
        ctrl_dt,
        tracking_sigma,
        base_height_target,
        min_base_height,
        max_tilt_rad,
        feet_phase_swing_height,
        feet_phase_tracking_sigma,
        min_forward_speed_for_gait_reward,
        close_feet_threshold,
        gyro_scale,
        dof_velocity_scale,
        linear_velocity_scale,
        reward_out,
        terminated_out,
        weighted_terms_out,
        actor_out,
        critic_out,
    ):
        action_dim = dof_position.shape[1]
        for row in range(root_position.shape[0]):
            up_z = torso_upvector[row, 2]
            if up_z < -1.0:
                up_z = -1.0
            elif up_z > 1.0:
                up_z = 1.0
            terminated_out[row] = (
                math.acos(up_z) > max_tilt_rad or root_position[row, 2] < min_base_height
            )

            total = 0.0
            for term_index in range(term_codes.shape[0]):
                code = term_codes[term_index]
                value = 0.0
                if code == _TERM_TRACKING_LIN_VEL:
                    dx = commands[row, 0] - pelvis_local_linear_velocity[row, 0]
                    dy = commands[row, 1] - pelvis_local_linear_velocity[row, 1]
                    value = math.exp(-(dx * dx + dy * dy) / tracking_sigma)
                elif code == _TERM_TRACKING_ANG_VEL:
                    dz = commands[row, 2] - torso_gyro[row, 2]
                    value = math.exp(-(dz * dz) / tracking_sigma)
                elif code == _TERM_FORWARD_PROGRESS:
                    speed = _positive(pelvis_local_linear_velocity[row, 0])
                    command = commands[row, 0]
                    if command < 1.0e-6:
                        command = 1.0e-6
                    value = speed / command
                    if value > 1.0:
                        value = 1.0
                elif code == _TERM_UNDER_SPEED:
                    command = commands[row, 0]
                    if command < 1.0e-6:
                        command = 1.0e-6
                    gap = commands[row, 0] - _positive(pelvis_local_linear_velocity[row, 0])
                    value = _positive(gap) / command
                elif code == _TERM_LIN_VEL_Z:
                    value = (
                        pelvis_local_linear_velocity[row, 2] * pelvis_local_linear_velocity[row, 2]
                    )
                elif code == _TERM_ORIENTATION:
                    value = (
                        torso_upvector[row, 0] * torso_upvector[row, 0]
                        + torso_upvector[row, 1] * torso_upvector[row, 1]
                    )
                elif code == _TERM_ANG_VEL_XY:
                    value = (
                        torso_gyro[row, 0] * torso_gyro[row, 0]
                        + torso_gyro[row, 1] * torso_gyro[row, 1]
                    )
                elif code == _TERM_ACTION_RATE:
                    for action_index in range(action_dim):
                        delta = current_actions[row, action_index] - last_actions[row, action_index]
                        value += delta * delta
                elif code == _TERM_BASE_HEIGHT:
                    delta = root_position[row, 2] - base_height_target
                    value = delta * delta
                elif code == _TERM_POSE:
                    for action_index in range(action_dim):
                        delta = dof_position[row, action_index] - default_angles[action_index]
                        value += pose_weights[action_index] * delta * delta
                elif code == _TERM_UPPER_BODY_POSE:
                    for action_index in range(action_dim):
                        delta = dof_position[row, action_index] - default_angles[action_index]
                        value += upper_body_pose_weights[action_index] * delta * delta
                elif code == _TERM_CLOSE_FEET_XY:
                    dx = left_foot_position[row, 0] - right_foot_position[row, 0]
                    dy = left_foot_position[row, 1] - right_foot_position[row, 1]
                    distance = math.sqrt(dx * dx + dy * dy)
                    if distance < close_feet_threshold:
                        delta = distance - close_feet_threshold
                        value = delta * delta
                elif code == _TERM_FEET_PHASE or code == _TERM_FEET_PHASE_CONTRAST:
                    left_target = _bezier_height(gait_phase[row, 0], feet_phase_swing_height)
                    right_target = _bezier_height(gait_phase[row, 1], feet_phase_swing_height)
                    if code == _TERM_FEET_PHASE:
                        left_error = left_foot_position[row, 2] - left_target
                        right_error = right_foot_position[row, 2] - right_target
                        value = math.exp(
                            -(left_error * left_error + right_error * right_error)
                            / feet_phase_tracking_sigma
                        )
                    else:
                        error = (left_foot_position[row, 2] - right_foot_position[row, 2]) - (
                            left_target - right_target
                        )
                        value = math.exp(-(error * error) / feet_phase_tracking_sigma)
                    if (
                        _positive(pelvis_local_linear_velocity[row, 0])
                        < min_forward_speed_for_gait_reward
                    ):
                        value = 0.0
                elif code == _TERM_ALIVE:
                    value = 1.0
                weighted = value * term_scales[term_index]
                weighted_terms_out[row, term_index] = weighted
                total += weighted
            reward_out[row] = total * ctrl_dt
            _write_observation_row(
                row,
                row,
                pelvis_local_linear_velocity,
                torso_gyro,
                torso_upvector,
                dof_position,
                dof_angular_velocity,
                commands,
                current_actions,
                gait_phase,
                default_angles,
                gyro_scale,
                dof_velocity_scale,
                linear_velocity_scale,
                actor_out,
                critic_out,
            )

    @njit(cache=True, nogil=True)  # type: ignore[misc]
    def _copy_rows_kernel(row_indices, source, target):
        for row in range(row_indices.shape[0]):
            target[row_indices[row], :] = source[row, :]

    @njit(cache=True, nogil=True)  # type: ignore[misc]
    def _gather_task_rows_kernel(
        row_indices,
        commands,
        current_actions,
        gait_phase,
        commands_out,
        current_actions_out,
        gait_phase_out,
    ):
        for row in range(row_indices.shape[0]):
            source_row = row_indices[row]
            for component in range(3):
                commands_out[row, component] = commands[source_row, component]
            for action_index in range(current_actions.shape[1]):
                current_actions_out[row, action_index] = current_actions[source_row, action_index]
            for phase_index in range(2):
                gait_phase_out[row, phase_index] = gait_phase[source_row, phase_index]

    @njit(cache=True, nogil=True)  # type: ignore[misc]
    def _complete_reset_task_state_kernel(
        row_indices,
        reset_commands,
        reset_gait_phase,
        commands,
        current_actions,
        last_actions,
        gait_phase,
        steps,
    ):
        for row in range(row_indices.shape[0]):
            target_row = row_indices[row]
            for component in range(3):
                commands[target_row, component] = reset_commands[row, component]
            for action_index in range(current_actions.shape[1]):
                current_actions[target_row, action_index] = 0.0
                last_actions[target_row, action_index] = 0.0
            for phase_index in range(2):
                gait_phase[target_row, phase_index] = reset_gait_phase[row, phase_index]
            steps[target_row] = 0


@dataclass
class _G1ManagedFusedTaskState:
    """All mutable task-owned buffers used by the fused host executor."""

    commands: np.ndarray
    current_actions: np.ndarray
    last_actions: np.ndarray
    gait_phase: np.ndarray
    steps: np.ndarray
    reset_qpos: np.ndarray
    reset_qvel: np.ndarray
    reset_commands: np.ndarray
    reset_gait_phase: np.ndarray
    reset_value_buffers: tuple[np.ndarray, ...]
    reset_value_buffer_set: BoundMutationValueBuffers
    reset_rng: np.random.RandomState
    observation_noise_rng: np.random.Generator | None
    reward_means: np.ndarray
    logged_reward_means: np.ndarray
    has_logged_reward: np.ndarray
    reward_scratch: np.ndarray
    weighted_term_scratch: np.ndarray
    actor_scratch: np.ndarray
    critic_scratch: np.ndarray
    row_scratch: np.ndarray
    command_row_scratch: np.ndarray
    action_row_scratch: np.ndarray
    gait_phase_row_scratch: np.ndarray
    noise_uniform_vector_scratch: np.ndarray
    noise_value_vector_scratch: np.ndarray
    noise_uniform_action_scratch: np.ndarray
    noise_value_action_scratch: np.ndarray
    terminal_state_token: int | None = None
    # A terminal ``StateBatch`` is consumed first by termination/reward and
    # then by terminal-observation materialization.  Keep a *strong* object
    # identity cache for that one borrowed view so the second consumer does
    # not repeat field-by-field validation.  ``_state_views`` still calls
    # ``assert_valid`` before consulting the cache, so a reset/step barrier
    # cannot make a stale lease usable.  Do not cache ``id(state)`` alone:
    # Python may reuse an object id after the borrowed batch is released.
    cached_state: StateBatch | None = field(default=None, repr=False, compare=False)
    cached_state_views: _G1StateViews | None = field(default=None, repr=False, compare=False)


def _require_numba() -> None:
    if not NUMBA_AVAILABLE:
        raise G1ManagedFusedError(
            "G1 managed fused executor requires numba; no host-reference fallback is available"
        )


def _term_code_array(
    config: _G1KernelConfig, dtype: np.dtype[Any]
) -> tuple[np.ndarray, np.ndarray]:
    codes: list[int] = []
    scales: list[float] = []
    for name, scale in config.reward_terms:
        try:
            code = _TERM_CODES[name]
        except KeyError as exc:  # _validate_reference_profile normally catches this first.
            raise G1ManagedFusedError(
                f"G1 fused executor does not implement reward term {name!r}"
            ) from exc
        if not np.isfinite(scale):
            raise G1ManagedFusedError("G1 fused executor reward scales must be finite")
        codes.append(code)
        scales.append(float(scale))
    return np.asarray(codes, dtype=np.int64), np.asarray(scales, dtype=dtype)


def compile_g1_managed_fused_task(*, backend: SimBackend, cfg: G1WalkEnvCfg) -> CompiledTaskPlan:
    """Compile the explicit fused executor plan through public backend APIs.

    This function is intentionally separate from the reference compiler so
    that executor identity participates in the immutable task fingerprint.
    It performs only cold-path metadata resolution and leaves the resulting
    plan free of a concrete backend instance.
    """

    _require_numba()
    if not isinstance(backend, SimBackend):
        raise G1ManagedFusedError("G1 managed fused executor requires a SimBackend")
    try:
        reward = _validate_reference_profile(cfg)
    except G1ManagedReferenceError as exc:
        raise G1ManagedFusedError(str(exc).replace("managed reference", "fused executor")) from exc
    actuator_names = backend.get_actuator_names()
    if not actuator_names:
        raise G1ManagedFusedError("G1 managed fused executor requires named actuators")
    if len(set(actuator_names)) != len(actuator_names):
        raise G1ManagedFusedError("G1 managed fused executor actuator names must be unique")
    action_dim = len(actuator_names)
    _action_scale(cfg, action_dim)

    root, dofs, reset_position, reset_velocity = _g1_selectors(actuator_names)
    state_requirements = _g1_state_requirements(root=root, dofs=dofs, action_dim=action_dim)
    reset_templates = _g1_reset_templates(
        root=root,
        reset_position=reset_position,
        reset_velocity=reset_velocity,
    )
    registry = TermRegistry()
    registry.register(
        TermDefinition(
            key="g1.fused.reset",
            version="1",
            phase=TermPhase.RESET,
            role=TermRole.EVENT,
            mutation_templates=reset_templates,
        )
    )
    registry.register(
        TermDefinition(
            key="g1.fused.termination",
            version="1",
            phase=TermPhase.TERMINATION,
            role=TermRole.TERMINATION,
            state_requirements=tuple(
                requirement
                for requirement in state_requirements
                if requirement.semantic_key in {"g1.root.position", "g1.sensor.torso_upvector"}
            ),
        )
    )
    registry.register(
        TermDefinition(
            key="g1.fused.reward",
            version="1",
            phase=TermPhase.REWARD,
            role=TermRole.REWARD,
            state_requirements=state_requirements,
        )
    )
    registry.register(
        TermDefinition(
            key="g1.fused.actor_observation",
            version="1",
            phase=TermPhase.TERMINAL_OBSERVATION,
            role=TermRole.OBSERVATION,
            state_requirements=state_requirements,
            output=TensorSpec((_ACTOR_WIDTH,), np.dtype(get_global_dtype()).name),
        )
    )
    registry.register(
        TermDefinition(
            key="g1.fused.critic_observation",
            version="1",
            phase=TermPhase.TERMINAL_OBSERVATION,
            role=TermRole.OBSERVATION,
            state_requirements=state_requirements,
            output=TensorSpec((_CRITIC_WIDTH,), np.dtype(get_global_dtype()).name),
        )
    )
    task = TaskSpec.create(
        key="g1_walk_flat.managed_fused",
        terms=(
            TermInvocation.create(key=_RESET_TERM, definition_key="g1.fused.reset"),
            TermInvocation.create(
                key="g1_termination",
                definition_key="g1.fused.termination",
                dependencies=(_RESET_TERM,),
            ),
            TermInvocation.create(
                key="g1_reward",
                definition_key="g1.fused.reward",
                dependencies=("g1_termination",),
            ),
            TermInvocation.create(
                key="g1_actor_observation",
                definition_key="g1.fused.actor_observation",
                dependencies=("g1_reward",),
                observation_group="obs",
            ),
            TermInvocation.create(
                key="g1_critic_observation",
                definition_key="g1.fused.critic_observation",
                dependencies=("g1_reward",),
                observation_group="critic",
            ),
        ),
        control=ControlSpec(
            semantic_key="g1.joint.position_target",
            buffer=_manager_buffer(
                row_shape=(action_dim,),
                lifetime=BufferLifetime.UNTIL_STEP_COMPLETE,
            ),
            physics_substeps_per_control=cfg.sim_substeps,
        ),
        execution_profile=ExecutionProfile.HOST_NUMPY,
        executor_key=G1_MANAGED_FUSED_EXECUTOR_KEY,
        policy=PolicySpec(
            ("obs", "critic"),
            tuple(float(value) for value in _action_scale(cfg, action_dim)),
        ),
    )
    capabilities = frozenset(
        {
            "state.root.position",
            "state.root.orientation",
            "state.root.linear_velocity",
            "state.root.angular_velocity",
            "state.dof.position",
            "state.dof.angular_velocity",
            "state.sensor.value",
        }
    )
    plan = TaskCompiler(registry).compile(
        task,
        resolver=BackendEntityResolver(backend),
        capabilities=capabilities,
    )
    if plan.policy_abi.observation_groups[0].width != _ACTOR_WIDTH or (
        plan.policy_abi.observation_groups[1].width != _CRITIC_WIDTH
    ):
        raise G1ManagedFusedError("compiled G1 fused policy ABI has an unexpected width")
    if reward is not cfg.reward_config:  # pragma: no cover - type narrowing invariant.
        raise G1ManagedFusedError("G1 reward configuration changed during compilation")
    return plan


class G1ManagedFusedKernel:
    """Pure Numba host task math over a cold-bound typed G1 state plan."""

    executor_key = G1_MANAGED_FUSED_EXECUTOR_KEY

    def __init__(self, config: _G1KernelConfig, *, expected_plan_fingerprint: str) -> None:
        _require_numba()
        if not isinstance(config, _G1KernelConfig):
            raise G1ManagedFusedError("G1 managed fused kernel requires frozen config")
        if not isinstance(expected_plan_fingerprint, str) or not expected_plan_fingerprint.strip():
            raise G1ManagedFusedError(
                "G1 fused kernel requires a canonical expected plan fingerprint"
            )
        self._config = config
        self._action_dim = int(config.default_angles.size)
        self._expected_plan_fingerprint = expected_plan_fingerprint
        self._binding: ManagedKernelBinding | None = None
        self._state_indices: tuple[int, ...] | None = None
        self._obs_buffer_indices: tuple[int, int] | None = None
        self._mutation_plan: BoundMutationPlan | None = None
        self._root_reset_indices: tuple[int, int, int, int] | None = None
        self._dof_position_reset_indices: tuple[int, ...] | None = None
        self._dof_velocity_reset_indices: tuple[int, ...] | None = None
        dtype = np.dtype(get_global_dtype())
        self._term_codes, self._term_scales = _term_code_array(config, dtype)
        scalar = dtype.type
        self._action_scale = np.ascontiguousarray(config.action_scale, dtype=dtype)
        self._default_angles = np.ascontiguousarray(config.default_angles, dtype=dtype)
        self._pose_weights = np.ascontiguousarray(config.pose_weights, dtype=dtype)
        self._upper_body_pose_weights = np.ascontiguousarray(
            config.upper_body_pose_weights, dtype=dtype
        )
        self._gait_phase_delta = scalar(config.gait_phase_delta)
        self._ctrl_dt = scalar(config.ctrl_dt)
        self._tracking_sigma = scalar(config.tracking_sigma)
        self._base_height_target = scalar(config.base_height_target)
        self._min_base_height = scalar(config.min_base_height)
        self._max_tilt_rad = scalar(config.max_tilt_rad)
        self._feet_phase_swing_height = scalar(config.feet_phase_swing_height)
        self._feet_phase_tracking_sigma = scalar(config.feet_phase_tracking_sigma)
        self._min_forward_speed_for_gait_reward = scalar(config.min_forward_speed_for_gait_reward)
        self._close_feet_threshold = scalar(config.close_feet_threshold)
        self._gyro_scale = scalar(0.25 if config.walk_observation_profile else 1.0)
        self._dof_velocity_scale = scalar(0.05 if config.walk_observation_profile else 1.0)
        self._linear_velocity_scale = scalar(2.0 if config.walk_observation_profile else 1.0)
        self._observation_noise_level = scalar(config.observation_noise_level)
        self._observation_noise_joint_angle_scale = scalar(
            config.observation_noise_scale_joint_angle
        )
        self._observation_noise_joint_velocity_scale = scalar(
            config.observation_noise_scale_joint_vel
        )
        self._observation_noise_gyro_scale = scalar(config.observation_noise_scale_gyro)
        self._observation_noise_gravity_scale = scalar(config.observation_noise_scale_gravity)

    def bind(self, *, binding: ManagedKernelBinding) -> None:
        _require_numba()
        if self._binding is not None:
            raise G1ManagedFusedError("G1 managed fused kernel may only bind once")
        if not isinstance(binding, ManagedKernelBinding):
            raise G1ManagedFusedError("G1 managed fused executor requires a ManagedKernelBinding")
        if binding.task_fingerprint != self._expected_plan_fingerprint:
            raise G1ManagedFusedError(
                "G1 managed fused executor received a stale or foreign compiled plan fingerprint"
            )
        if binding.execution_profile is not ExecutionProfile.HOST_NUMPY:
            raise G1ManagedFusedError("G1 managed fused executor only supports host_numpy")
        if binding.dtype != np.dtype(get_global_dtype()).name:
            raise G1ManagedFusedError(
                "G1 managed fused executor dtype must match the repository global dtype"
            )
        state_index_by_key = dict(binding.state_field_indices)
        missing_state = tuple(key for key in _STATE_KEYS if key not in state_index_by_key)
        if missing_state:
            raise G1ManagedFusedError(
                "G1 managed fused plan is missing state fields: " + ", ".join(missing_state)
            )
        observation_index_by_key = dict(binding.observation_buffer_indices)
        try:
            observation_indices = (
                observation_index_by_key["obs"],
                observation_index_by_key["critic"],
            )
        except KeyError as exc:
            raise G1ManagedFusedError(
                "G1 managed fused executor requires obs and critic output buffers"
            ) from exc
        mutation_plan = binding.mutation_plan
        if mutation_plan is None:
            raise G1ManagedFusedError("G1 managed fused executor requires typed reset mutations")
        mutation_indices = {spec.term_key: index for index, spec in enumerate(mutation_plan.specs)}
        root_keys = tuple(_reset_term_key(suffix=item[0]) for item in _ROOT_RESET_SUFFIXES)
        position_keys = tuple(
            _reset_term_key(suffix=_reset_suffix_for_dof(kind="position", index=index))
            for index in range(self._action_dim)
        )
        velocity_keys = tuple(
            _reset_term_key(suffix=_reset_suffix_for_dof(kind="velocity", index=index))
            for index in range(self._action_dim)
        )
        required_mutations = (*root_keys, *position_keys, *velocity_keys)
        missing_mutations = tuple(key for key in required_mutations if key not in mutation_indices)
        if missing_mutations:
            raise G1ManagedFusedError(
                "G1 managed fused plan is missing reset mutations: " + ", ".join(missing_mutations)
            )
        self._binding = binding
        self._state_indices = tuple(state_index_by_key[key] for key in _STATE_KEYS)
        self._obs_buffer_indices = observation_indices
        self._mutation_plan = mutation_plan
        self._root_reset_indices = tuple(mutation_indices[key] for key in root_keys)  # type: ignore[assignment]
        self._dof_position_reset_indices = tuple(mutation_indices[key] for key in position_keys)
        self._dof_velocity_reset_indices = tuple(mutation_indices[key] for key in velocity_keys)
        self._warm_numba(dtype=np.dtype(binding.dtype))

    def _warm_numba(self, *, dtype: np.dtype[Any]) -> None:
        """Compile every hot kernel on the cold bind path or fail explicitly."""

        _require_numba()
        if dtype != np.dtype(get_global_dtype()):
            raise G1ManagedFusedError("G1 fused warmup dtype differs from global dtype")
        action_dim = self._action_dim
        one = 1
        vector3 = np.zeros((one, 3), dtype=dtype)
        dofs = np.zeros((one, action_dim), dtype=dtype)
        state_vector3 = vector3.view()
        state_dofs = dofs.view()
        state_vector3.flags.writeable = False
        state_dofs.flags.writeable = False
        commands = np.zeros((one, 3), dtype=dtype)
        phase = np.zeros((one, 2), dtype=dtype)
        actor = np.zeros((one, _ACTOR_WIDTH), dtype=dtype)
        critic = np.zeros((one, _CRITIC_WIDTH), dtype=dtype)
        reward = np.zeros((one,), dtype=dtype)
        terminated = np.zeros((one,), dtype=bool)
        weighted = np.zeros((one, len(self._term_codes)), dtype=dtype)
        rows = np.zeros((one,), dtype=np.intp)
        _apply_action_kernel(
            dofs,
            self._action_scale,
            self._default_angles,
            self._gait_phase_delta,
            dofs.copy(),
            dofs.copy(),
            phase.copy(),
            dofs.copy(),
        )
        _write_observations_kernel(
            rows,
            state_vector3,
            state_vector3,
            state_vector3,
            state_dofs,
            state_dofs,
            commands,
            dofs,
            phase,
            self._default_angles,
            self._gyro_scale,
            self._dof_velocity_scale,
            self._linear_velocity_scale,
            actor,
            critic,
        )
        _compute_terminal_kernel(
            state_vector3,
            state_vector3,
            state_vector3,
            state_vector3,
            state_vector3,
            state_vector3,
            state_dofs,
            state_dofs,
            commands,
            dofs,
            dofs,
            phase,
            self._default_angles,
            self._pose_weights,
            self._upper_body_pose_weights,
            self._term_codes,
            self._term_scales,
            self._ctrl_dt,
            self._tracking_sigma,
            self._base_height_target,
            self._min_base_height,
            self._max_tilt_rad,
            self._feet_phase_swing_height,
            self._feet_phase_tracking_sigma,
            self._min_forward_speed_for_gait_reward,
            self._close_feet_threshold,
            self._gyro_scale,
            self._dof_velocity_scale,
            self._linear_velocity_scale,
            reward,
            terminated,
            weighted,
            actor,
            critic,
        )
        _copy_rows_kernel(rows, actor, actor)
        _gather_task_rows_kernel(
            rows,
            commands,
            dofs,
            phase,
            commands.copy(),
            dofs.copy(),
            phase.copy(),
        )
        _complete_reset_task_state_kernel(
            rows,
            commands,
            phase,
            commands.copy(),
            dofs.copy(),
            dofs.copy(),
            phase.copy(),
            np.zeros((one,), dtype=np.uint32),
        )

    def _require_binding(self) -> ManagedKernelBinding:
        if self._binding is None:
            raise G1ManagedFusedError("G1 managed fused kernel has not been bound")
        return self._binding

    def _require_state_indices(self) -> tuple[int, ...]:
        if self._state_indices is None:
            raise G1ManagedFusedError("G1 managed fused state fields are not bound")
        return self._state_indices

    def _require_observation_indices(self) -> tuple[int, int]:
        if self._obs_buffer_indices is None:
            raise G1ManagedFusedError("G1 managed fused observations are not bound")
        return self._obs_buffer_indices

    def _require_reset_indices(
        self,
    ) -> tuple[tuple[int, int, int, int], tuple[int, ...], tuple[int, ...]]:
        if (
            self._root_reset_indices is None
            or self._dof_position_reset_indices is None
            or self._dof_velocity_reset_indices is None
        ):
            raise G1ManagedFusedError("G1 managed fused reset fields are not bound")
        return (
            self._root_reset_indices,
            self._dof_position_reset_indices,
            self._dof_velocity_reset_indices,
        )

    @staticmethod
    def _require_task_state(task_state: object) -> _G1ManagedFusedTaskState:
        if not isinstance(task_state, _G1ManagedFusedTaskState):
            raise G1ManagedFusedError("G1 managed fused executor received foreign task state")
        return task_state

    def _state_views(
        self,
        state: StateBatch,
        task: _G1ManagedFusedTaskState,
    ) -> _G1StateViews:
        """Validate and map typed fields without compiler-order assumptions."""

        state.assert_valid()
        if task.cached_state is state:
            cached = task.cached_state_views
            if cached is None:  # pragma: no cover - task-state invariant guard.
                raise G1ManagedFusedError("G1 fused state-view cache is incomplete")
            return cached
        expected_dtype = np.dtype(self._require_binding().dtype)
        expected_shapes = (
            (self._action_dim,),
            (self._action_dim,),
            (3,),
            (3,),
            (4,),
            (3,),
            (3,),
            (3,),
            (3,),
            (3,),
            (3,),
        )
        values: list[np.ndarray] = []
        for key, index, row_shape in zip(
            _STATE_KEYS, self._require_state_indices(), expected_shapes, strict=True
        ):
            handle = state.buffer_at(index).handle
            if not isinstance(handle, np.ndarray):
                raise G1ManagedFusedError(f"G1 typed state {key} must be a numpy array")
            expected_shape = (state.rows.count, *row_shape)
            if handle.shape != expected_shape:
                raise G1ManagedFusedError(
                    f"G1 typed state {key} must have shape {expected_shape}, got {handle.shape}"
                )
            if handle.dtype != expected_dtype:
                raise G1ManagedFusedError(
                    f"G1 typed state {key} must have dtype {expected_dtype.name}, "
                    f"got {handle.dtype.name}"
                )
            if not handle.flags.c_contiguous:
                raise G1ManagedFusedError(f"G1 typed state {key} must be C-contiguous")
            if not np.isfinite(handle).all():
                raise G1ManagedFusedError(
                    f"G1 fused executor rejects non-finite typed state {key} before math dispatch"
                )
            values.append(handle)
        views = _G1StateViews(
            dof_angular_velocity=values[0],
            dof_position=values[1],
            root_angular_velocity=values[2],
            root_linear_velocity=values[3],
            root_orientation=values[4],
            root_position=values[5],
            left_foot_position=values[6],
            pelvis_local_linear_velocity=values[7],
            right_foot_position=values[8],
            torso_gyro=values[9],
            torso_upvector=values[10],
        )
        task.cached_state = state
        task.cached_state_views = views
        return views

    def create_task_state(self, *, num_envs: int, dtype: np.dtype[Any]) -> object:
        binding = self._require_binding()
        if num_envs != binding.num_envs or dtype != np.dtype(binding.dtype):
            raise G1ManagedFusedError("G1 fused task state disagrees with its kernel binding")
        if dtype != np.dtype(get_global_dtype()):
            raise G1ManagedFusedError("G1 fused task state dtype differs from global dtype")
        if self._mutation_plan is None:
            raise G1ManagedFusedError("G1 fused task state requires a bound mutation plan")
        reset_value_buffers = tuple(
            np.empty((num_envs, *spec.value_buffer.row_shape), dtype=dtype)
            for spec in self._mutation_plan.specs
        )
        reset_value_buffer_set = BoundMutationValueBuffers(
            plan=self._mutation_plan,
            buffers=reset_value_buffers,
        )
        noise_rng = (
            None
            if self._config.observation_noise_level <= 0.0
            else np.random.default_rng(self._config.observation_noise_seed)
        )
        return _G1ManagedFusedTaskState(
            commands=np.zeros((num_envs, 3), dtype=dtype),
            current_actions=np.zeros((num_envs, self._action_dim), dtype=dtype),
            last_actions=np.zeros((num_envs, self._action_dim), dtype=dtype),
            gait_phase=np.zeros((num_envs, 2), dtype=dtype),
            steps=np.zeros((num_envs,), dtype=np.uint32),
            reset_qpos=np.empty((num_envs, 7 + self._action_dim), dtype=dtype),
            reset_qvel=np.empty((num_envs, 6 + self._action_dim), dtype=dtype),
            reset_commands=np.empty((num_envs, 3), dtype=dtype),
            reset_gait_phase=np.empty((num_envs, 2), dtype=dtype),
            reset_value_buffers=reset_value_buffers,
            reset_value_buffer_set=reset_value_buffer_set,
            reset_rng=np.random.RandomState(self._config.reset_seed),
            observation_noise_rng=noise_rng,
            reward_means=np.zeros((len(self._config.reward_terms),), dtype=dtype),
            logged_reward_means=np.zeros((len(self._config.reward_terms),), dtype=dtype),
            has_logged_reward=np.zeros((len(self._config.reward_terms),), dtype=bool),
            reward_scratch=np.empty((num_envs,), dtype=dtype),
            weighted_term_scratch=np.empty((num_envs, len(self._config.reward_terms)), dtype=dtype),
            actor_scratch=np.empty((num_envs, _ACTOR_WIDTH), dtype=dtype),
            critic_scratch=np.empty((num_envs, _CRITIC_WIDTH), dtype=dtype),
            row_scratch=np.arange(num_envs, dtype=np.intp),
            command_row_scratch=np.empty((num_envs, 3), dtype=dtype),
            action_row_scratch=np.empty((num_envs, self._action_dim), dtype=dtype),
            gait_phase_row_scratch=np.empty((num_envs, 2), dtype=dtype),
            noise_uniform_vector_scratch=np.empty((num_envs, 3), dtype=np.float64),
            noise_value_vector_scratch=np.empty((num_envs, 3), dtype=dtype),
            noise_uniform_action_scratch=np.empty((num_envs, self._action_dim), dtype=np.float64),
            noise_value_action_scratch=np.empty((num_envs, self._action_dim), dtype=dtype),
        )

    def managed_runtime_buffers(self, *, task_state: object) -> tuple[ManagedRuntimeBuffer, ...]:
        """Register every fused task-owned numeric buffer for warm auditing.

        The method is called only when the caller explicitly enables runtime
        stability instrumentation.  It never receives a backend/state/model
        object and returns descriptor wrappers around already allocated task
        arrays; it does not allocate numeric storage on the hot path.
        """

        task = self._require_task_state(task_state)
        candidates: list[tuple[str, np.ndarray]] = [
            ("kernel.action_scale", self._action_scale),
            ("kernel.default_angles", self._default_angles),
            ("kernel.pose_weights", self._pose_weights),
            ("kernel.upper_body_pose_weights", self._upper_body_pose_weights),
            ("kernel.term_codes", self._term_codes),
            ("kernel.term_scales", self._term_scales),
            ("kernel.initial_qpos", self._config.initial_qpos),
            ("kernel.initial_qvel", self._config.initial_qvel),
            ("kernel.command_low", self._config.command_low),
            ("kernel.command_high", self._config.command_high),
            ("task.commands", task.commands),
            ("task.current_actions", task.current_actions),
            ("task.last_actions", task.last_actions),
            ("task.gait_phase", task.gait_phase),
            ("task.steps", task.steps),
            ("task.reset_qpos", task.reset_qpos),
            ("task.reset_qvel", task.reset_qvel),
            ("task.reset_commands", task.reset_commands),
            ("task.reset_gait_phase", task.reset_gait_phase),
            ("task.reward_means", task.reward_means),
            ("task.logged_reward_means", task.logged_reward_means),
            ("task.has_logged_reward", task.has_logged_reward),
            ("task.reward_scratch", task.reward_scratch),
            ("task.weighted_term_scratch", task.weighted_term_scratch),
            ("task.actor_scratch", task.actor_scratch),
            ("task.critic_scratch", task.critic_scratch),
            ("task.row_scratch", task.row_scratch),
            ("task.command_row_scratch", task.command_row_scratch),
            ("task.action_row_scratch", task.action_row_scratch),
            ("task.gait_phase_row_scratch", task.gait_phase_row_scratch),
            ("task.noise_uniform_vector_scratch", task.noise_uniform_vector_scratch),
            ("task.noise_value_vector_scratch", task.noise_value_vector_scratch),
            ("task.noise_uniform_action_scratch", task.noise_uniform_action_scratch),
            ("task.noise_value_action_scratch", task.noise_value_action_scratch),
        ]
        candidates.extend(
            (f"task.reset_value_buffers.{index}", buffer)
            for index, buffer in enumerate(task.reset_value_buffers)
        )
        return tuple(
            ManagedRuntimeBuffer(name=name, array=array)
            for name, array in candidates
            if array.size > 0
        )

    def apply_action(
        self,
        *,
        actions: np.ndarray,
        task_state: object,
        control_out: np.ndarray,
    ) -> None:
        task = self._require_task_state(task_state)
        if (
            actions.shape != task.current_actions.shape
            or actions.dtype != task.current_actions.dtype
        ):
            raise G1ManagedFusedError("G1 fused actions do not match task action state")
        if control_out.shape != actions.shape or control_out.dtype != actions.dtype:
            raise G1ManagedFusedError("G1 fused control output does not match actions")
        if not np.isfinite(actions).all():
            raise G1ManagedFusedError(
                "G1 fused executor rejects non-finite actions before dispatch"
            )
        _apply_action_kernel(
            actions,
            self._action_scale,
            self._default_angles,
            self._gait_phase_delta,
            task.current_actions,
            task.last_actions,
            task.gait_phase,
            control_out,
        )

    def build_pre_physics_mutation(self, *, task_state: object):
        self._require_task_state(task_state)
        return None

    def evaluate_termination(
        self,
        *,
        state: StateBatch,
        task_state: object,
        terminated_out: np.ndarray,
    ) -> None:
        task = self._require_task_state(task_state)
        views = self._state_views(state, task)
        binding = self._require_binding()
        if not state.rows.is_all or state.rows.count != binding.num_envs:
            raise G1ManagedFusedError(
                "G1 fused terminal dispatch requires the complete state batch"
            )
        if terminated_out.shape != task.steps.shape or terminated_out.dtype != np.dtype(bool):
            raise G1ManagedFusedError("G1 fused terminated output has an invalid shape or dtype")
        _compute_terminal_kernel(
            views.root_position,
            views.torso_upvector,
            views.pelvis_local_linear_velocity,
            views.torso_gyro,
            views.left_foot_position,
            views.right_foot_position,
            views.dof_position,
            views.dof_angular_velocity,
            task.commands,
            task.current_actions,
            task.last_actions,
            task.gait_phase,
            self._default_angles,
            self._pose_weights,
            self._upper_body_pose_weights,
            self._term_codes,
            self._term_scales,
            self._ctrl_dt,
            self._tracking_sigma,
            self._base_height_target,
            self._min_base_height,
            self._max_tilt_rad,
            self._feet_phase_swing_height,
            self._feet_phase_tracking_sigma,
            self._min_forward_speed_for_gait_reward,
            self._close_feet_threshold,
            self._gyro_scale,
            self._dof_velocity_scale,
            self._linear_velocity_scale,
            task.reward_scratch,
            terminated_out,
            task.weighted_term_scratch,
            task.actor_scratch,
            task.critic_scratch,
        )
        for index in range(len(self._config.reward_terms)):
            task.reward_means[index] = np.mean(task.weighted_term_scratch[:, index])
        task.terminal_state_token = id(state)

    def evaluate_reward(
        self,
        *,
        state: StateBatch,
        task_state: object,
        reward_out: np.ndarray,
    ) -> None:
        task = self._require_task_state(task_state)
        if task.terminal_state_token != id(state):
            raise G1ManagedFusedError(
                "G1 fused reward dispatch requires the terminal state previously evaluated for termination"
            )
        if (
            reward_out.shape != task.reward_scratch.shape
            or reward_out.dtype != task.reward_scratch.dtype
        ):
            raise G1ManagedFusedError("G1 fused reward output has an invalid shape or dtype")
        np.copyto(reward_out, task.reward_scratch)
        if not np.isfinite(reward_out).all():
            raise G1ManagedFusedError("G1 fused reward math produced non-finite values")

    def evaluate_metrics(
        self,
        *,
        state: StateBatch,
        task_state: object,
        terminated: np.ndarray,
    ) -> tuple[ManagedMetric, ...]:
        del state, terminated
        task = self._require_task_state(task_state)
        if int(task.steps[0]) % 4 == 0:
            for index, (_, scale) in enumerate(self._config.reward_terms):
                if scale != 0.0:
                    task.logged_reward_means[index] = task.reward_means[index]
                    task.has_logged_reward[index] = True
        return tuple(
            ManagedMetric(f"reward/{name}", float(task.logged_reward_means[index]))
            for index, (name, scale) in enumerate(self._config.reward_terms)
            if scale != 0.0 and task.has_logged_reward[index]
        )

    @staticmethod
    def _rows_array(task: _G1ManagedFusedTaskState, rows: RowSelection) -> np.ndarray:
        count = rows.count
        target = task.row_scratch[:count]
        if rows.is_all:
            # A preceding sparse reset may have populated this reusable scratch
            # with non-identity row ids.  Restore the canonical full-batch map
            # rather than treating stale indices as a valid terminal mapping.
            for index in range(count):
                target[index] = index
            return target
        assert rows.indices is not None
        for index, row in enumerate(rows.indices):
            target[index] = row
        return target

    def write_observations(
        self,
        *,
        state: StateBatch,
        task_state: object,
        observation_buffers: tuple[np.ndarray, ...],
    ) -> None:
        task = self._require_task_state(task_state)
        views = self._state_views(state, task)
        actor_index, critic_index = self._require_observation_indices()
        try:
            actor_all = observation_buffers[actor_index]
            critic_all = observation_buffers[critic_index]
        except IndexError as exc:
            raise G1ManagedFusedError(
                "G1 fused runtime observation buffers are incomplete"
            ) from exc
        binding = self._require_binding()
        if actor_all.shape != (binding.num_envs, _ACTOR_WIDTH) or critic_all.shape != (
            binding.num_envs,
            _CRITIC_WIDTH,
        ):
            raise G1ManagedFusedError("G1 fused runtime observation buffers have invalid widths")
        if (
            actor_all.dtype != task.current_actions.dtype
            or critic_all.dtype != task.current_actions.dtype
        ):
            raise G1ManagedFusedError("G1 fused runtime observation buffers have an invalid dtype")
        row_indices = self._rows_array(task, state.rows)
        if state.phase.value == "terminal":
            if task.terminal_state_token != id(state):
                raise G1ManagedFusedError(
                    "G1 fused terminal observation requires prior fused terminal math dispatch"
                )
            _copy_rows_kernel(row_indices, task.actor_scratch[: state.rows.count], actor_all)
            _copy_rows_kernel(row_indices, task.critic_scratch[: state.rows.count], critic_all)
        else:
            count = state.rows.count
            _gather_task_rows_kernel(
                row_indices,
                task.commands,
                task.current_actions,
                task.gait_phase,
                task.command_row_scratch[:count],
                task.action_row_scratch[:count],
                task.gait_phase_row_scratch[:count],
            )
            _write_observations_kernel(
                row_indices,
                views.pelvis_local_linear_velocity,
                views.torso_gyro,
                views.torso_upvector,
                views.dof_position,
                views.dof_angular_velocity,
                task.command_row_scratch[:count],
                task.action_row_scratch[:count],
                task.gait_phase_row_scratch[:count],
                self._default_angles,
                self._gyro_scale,
                self._dof_velocity_scale,
                self._linear_velocity_scale,
                actor_all,
                critic_all,
            )
        # Noise is applied separately below.  This first implementation keeps
        # the exact reference RNG sequence; the Phase 4 stability slice will
        # instrument its allocation/addresses before making performance claims.
        if self._observation_noise_level > 0.0:
            self._write_observations_with_noise(
                task=task,
                views=views,
                rows=state.rows,
                actor_all=actor_all,
            )
        if state.phase.value == "terminal":
            if state.rows.is_all:
                task.steps += 1
            else:
                task.steps[row_indices] += 1

    def _write_observations_with_noise(
        self,
        *,
        task: _G1ManagedFusedTaskState,
        views: _G1StateViews,
        rows: RowSelection,
        actor_all: np.ndarray,
    ) -> None:
        """Write the actor's four noisy slices in the reference RNG order."""

        rng = task.observation_noise_rng
        if rng is None:
            raise G1ManagedFusedError("G1 fused observation noise RNG was not initialized")
        row_indices = self._rows_array(task, rows)
        count = rows.count
        level = self._observation_noise_level

        def sample_vector() -> np.ndarray:
            uniforms = task.noise_uniform_vector_scratch[:count]
            rng.random(out=uniforms)
            np.multiply(uniforms, 2.0, out=uniforms)
            np.subtract(uniforms, 1.0, out=uniforms)
            values = task.noise_value_vector_scratch[:count]
            np.copyto(values, uniforms, casting="unsafe")
            return values

        def sample_action() -> np.ndarray:
            uniforms = task.noise_uniform_action_scratch[:count]
            rng.random(out=uniforms)
            np.multiply(uniforms, 2.0, out=uniforms)
            np.subtract(uniforms, 1.0, out=uniforms)
            values = task.noise_value_action_scratch[:count]
            np.copyto(values, uniforms, casting="unsafe")
            return values

        gyro_noise = sample_vector()
        for local_row, target_row in enumerate(row_indices):
            actor_all[target_row, 0:3] = (
                views.torso_gyro[local_row]
                + gyro_noise[local_row] * level * self._observation_noise_gyro_scale
            ) * self._gyro_scale
        gravity_noise = sample_vector()
        for local_row, target_row in enumerate(row_indices):
            actor_all[target_row, 3:6] = -(
                views.torso_upvector[local_row]
                + gravity_noise[local_row] * level * self._observation_noise_gravity_scale
            )
        angle_noise = sample_action()
        for local_row, target_row in enumerate(row_indices):
            actor_all[target_row, 6 : 6 + self._action_dim] = (
                views.dof_position[local_row]
                - self._default_angles
                + angle_noise[local_row] * level * self._observation_noise_joint_angle_scale
            )
        velocity_noise = sample_action()
        for local_row, target_row in enumerate(row_indices):
            velocity_start = 6 + self._action_dim
            actor_all[target_row, velocity_start : velocity_start + self._action_dim] = (
                views.dof_angular_velocity[local_row]
                + velocity_noise[local_row] * level * self._observation_noise_joint_velocity_scale
            ) * self._dof_velocity_scale

    def _sample_commands(self, task: _G1ManagedFusedTaskState, count: int) -> None:
        values = np.asarray(
            task.reset_rng.uniform(
                low=self._config.command_low,
                high=self._config.command_high,
                size=(count, 3),
            ),
            dtype=task.commands.dtype,
        )
        moving = np.linalg.norm(values[:, :2], axis=1) > 0.2
        values[:, :2] *= moving[:, None]
        if self._config.standing_probability > 0.0:
            standing = task.reset_rng.uniform(size=(count,)) < self._config.standing_probability
            values[standing] = 0.0
        np.copyto(task.reset_commands[:count], values)

    def _sample_gait_phase(self, task: _G1ManagedFusedTaskState, count: int) -> None:
        if self._config.gait_phase_init_mode == "independent":
            task.reset_gait_phase[:count, 0] = task.reset_rng.uniform(
                0.0, 2.0 * math.pi, size=(count,)
            )
            task.reset_gait_phase[:count, 1] = task.reset_rng.uniform(
                0.0, 2.0 * math.pi, size=(count,)
            )
            return
        phase = task.reset_rng.uniform(0.0, 2.0 * math.pi, size=(count,))
        task.reset_gait_phase[:count, 0] = phase
        task.reset_gait_phase[:count, 1] = phase + math.pi

    def _prepare_reset_values(self, task: _G1ManagedFusedTaskState, rows: RowSelection) -> None:
        count = rows.count
        qpos = task.reset_qpos[:count]
        qvel = task.reset_qvel[:count]
        qpos[...] = self._config.initial_qpos
        qvel[...] = self._config.initial_qvel
        qpos[:, :2] += task.reset_rng.uniform(-0.5, 0.5, (count, 2))
        yaw = task.reset_rng.uniform(-math.pi, math.pi, (count,))
        qpos[:, 3:7] = np_quat_mul(qpos[:, 3:7], np_yaw_to_quat(yaw))
        qvel[:, :6] = np.asarray(
            task.reset_rng.uniform(
                -self._config.reset_base_qvel_limit,
                self._config.reset_base_qvel_limit,
                size=(count, 6),
            ),
            dtype=qvel.dtype,
        )
        self._sample_commands(task, count)
        self._sample_gait_phase(task, count)

        root_indices, position_indices, velocity_indices = self._require_reset_indices()
        root_values = (qpos[:, :3], qpos[:, 3:7], qvel[:, :3], qvel[:, 3:6])
        for index, values in zip(root_indices, root_values, strict=True):
            task.reset_value_buffers[index][:count, 0, :] = values
        for dof_index, mutation_index in enumerate(position_indices):
            task.reset_value_buffers[mutation_index][:count, 0, 0] = qpos[:, 7 + dof_index]
        for dof_index, mutation_index in enumerate(velocity_indices):
            task.reset_value_buffers[mutation_index][:count, 0, 0] = qvel[:, 6 + dof_index]

    def prepare_reset(self, *, rows: RowSelection, task_state: object) -> ManagedResetRequest:
        task = self._require_task_state(task_state)
        binding = self._require_binding()
        if rows.universe_size != binding.num_envs:
            raise G1ManagedFusedError("G1 fused reset rows differ from task row universe")
        if self._mutation_plan is None:
            raise G1ManagedFusedError("G1 fused reset requires a bound mutation plan")
        self._prepare_reset_values(task, rows)
        return ManagedResetRequest(
            rows=rows,
            mutation_batch=TypedBackendMutationBatch(
                plan=self._mutation_plan,
                rows=rows,
                state=SimulationStateMutationBatch(
                    bound_buffer_window=task.reset_value_buffer_set.window(rows)
                ),
            ),
            kernel_state=_G1ResetSample(rows=rows),
        )

    def complete_reset(
        self,
        *,
        request: ManagedResetRequest,
        state: StateBatch,
        task_state: object,
    ) -> None:
        task = self._require_task_state(task_state)
        if not isinstance(request.kernel_state, _G1ResetSample):
            raise G1ManagedFusedError("G1 fused reset request carries foreign task state")
        sample = request.kernel_state
        if sample.rows != request.rows or state.rows != request.rows:
            raise G1ManagedFusedError("G1 fused reset sample rows do not match reset state")
        state.assert_valid()
        count = request.rows.count
        _complete_reset_task_state_kernel(
            self._rows_array(task, request.rows),
            task.reset_commands[:count],
            task.reset_gait_phase[:count],
            task.commands,
            task.current_actions,
            task.last_actions,
            task.gait_phase,
            task.steps,
        )
        task.terminal_state_token = None


def create_g1_managed_fused_runtime(
    *,
    backend: SimBackend,
    cfg: G1WalkEnvCfg,
    reset_seed: int = 0,
    observation_noise_seed: int | None = None,
    autoreset: bool = True,
    record_lifecycle: bool = False,
    enable_stability_instrumentation: bool = False,
) -> ManagedReferenceRuntime:
    """Create a cold-bound fused G1 runtime with no reference fallback path."""

    _require_numba()
    if not isinstance(enable_stability_instrumentation, bool):
        raise G1ManagedFusedError("enable_stability_instrumentation must be a bool")
    # ``_kernel_config`` validates every unsupported legacy feature before its
    # first public backend metadata query.  It is a cold-path schema helper,
    # not a reference executor invocation.
    try:
        config = _kernel_config(
            backend=backend,
            cfg=cfg,
            reset_seed=reset_seed,
            observation_noise_seed=observation_noise_seed,
        )
    except G1ManagedReferenceError as exc:
        raise G1ManagedFusedError(str(exc).replace("managed reference", "fused executor")) from exc
    plan = compile_g1_managed_fused_task(backend=backend, cfg=cfg)
    kernel = G1ManagedFusedKernel(config, expected_plan_fingerprint=plan.fingerprint)
    backend.materialize()
    return ManagedReferenceRuntime(
        backend=backend,
        plan=plan,
        kernel=kernel,
        max_episode_steps=cfg.max_episode_steps,
        autoreset=autoreset,
        record_lifecycle=record_lifecycle,
        stability_buffer_provider=kernel if enable_stability_instrumentation else None,
        require_complete_backend_instrumentation=enable_stability_instrumentation,
    )


__all__ = [
    "G1_MANAGED_FUSED_EXECUTOR_KEY",
    "G1ManagedFusedError",
    "G1ManagedFusedKernel",
    "NUMBA_AVAILABLE",
    "compile_g1_managed_fused_task",
    "create_g1_managed_fused_runtime",
]
