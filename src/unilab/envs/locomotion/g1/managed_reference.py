"""Compiled host-reference G1 walking task for the manager runtime.

This module deliberately lives with the G1 task owner rather than in
``unilab.manager``.  The cold path lowers the existing flat-walk semantics to
an immutable :class:`~unilab.manager.CompiledTaskPlan`; the hot kernel consumes
only ``StateBatch`` views and runtime-owned task buffers.  In particular it
does not retain a backend, environment, model, selector, registry, or asset
object after construction.

The first slice is intentionally narrow and fail-closed: it mirrors the
``g1_walk_flat`` host NumPy reward/observation/reset profile, not arbitrary G1
domain randomization or terrain behaviour.  More capable DR/Event support is
owned by the later typed-mutation phase of issue #705.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from unilab.base.backend import (
    BoundMutationPlan,
    BufferContract,
    BufferLayout,
    BufferLifetime,
    BufferMutability,
    BufferOwner,
    BufferPlacement,
    BufferView,
    ControlSpec,
    ExecutionProfile,
    MutationBaseline,
    MutationCommitPhase,
    MutationFieldKind,
    MutationOperation,
    MutationPersistence,
    MutationRecomputeLevel,
    MutationTargetKind,
    MutationTrigger,
    MutationValueBatch,
    PhysicalUnit,
    ReferenceFrame,
    RowSelection,
    SimulationStateMutationBatch,
    StateBatch,
    StateFieldKind,
    TypedBackendMutationBatch,
)
from unilab.base.backend.base import SimBackend
from unilab.dtype_config import get_global_dtype
from unilab.manager import (
    BackendEntityResolver,
    EntityKind,
    EntitySelector,
    ManagedKernelBinding,
    ManagedMetric,
    ManagedReferenceRuntime,
    ManagedResetRequest,
    ManagerContractError,
    MutationTemplate,
    PolicySpec,
    QuaternionOrder,
    StateRequirement,
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

from .joystick import (
    G1WalkEnvCfg,
    G1WalkRewardConfig,
    build_upper_body_pose_weights,
    compute_feet_phase_height_targets,
    compute_forward_speed_gate,
)

G1_MANAGED_REFERENCE_EXECUTOR_KEY = "reference.numpy.g1-walk-flat.v1"
"""The explicit host-only executor identity used in the compiled plan."""

_ROOT_NAME = "pelvis"
_ACTOR_WIDTH = 98
_CRITIC_WIDTH = 101
_RESET_TERM = "g1_reset_state"
_ROOT_RESET_SUFFIXES = (
    ("root_position", "state.root.position", MutationFieldKind.POSITION, (3,)),
    (
        "root_orientation",
        "state.root.orientation",
        MutationFieldKind.ORIENTATION,
        (4,),
    ),
    (
        "root_linear_velocity",
        "state.root.linear_velocity",
        MutationFieldKind.LINEAR_VELOCITY,
        (3,),
    ),
    (
        "root_angular_velocity",
        "state.root.angular_velocity",
        MutationFieldKind.ANGULAR_VELOCITY,
        (3,),
    ),
)
_STATE_KEYS = (
    "g1.dof.angular_velocity",
    "g1.dof.position",
    "g1.root.angular_velocity",
    "g1.root.linear_velocity",
    "g1.root.orientation",
    "g1.root.position",
    "g1.sensor.left_foot_pos",
    "g1.sensor.pelvis_local_linvel",
    "g1.sensor.right_foot_pos",
    "g1.sensor.torso_gyro",
    "g1.sensor.torso_upvector",
)
_SUPPORTED_REWARD_TERMS = frozenset(
    {
        "action_rate",
        "alive",
        "ang_vel_xy",
        "base_height",
        "feet_phase",
        "feet_phase_contrast",
        "forward_progress",
        "lin_vel_z",
        "orientation",
        "penalty_action_rate",
        "penalty_ang_vel_xy",
        "penalty_close_feet_xy",
        "penalty_orientation",
        "pose",
        "tracking_ang_vel",
        "tracking_lin_vel",
        "under_speed",
        "upper_body_pose",
    }
)


class G1ManagedReferenceError(ManagerContractError):
    """Raised when a requested G1 managed-reference profile is unsupported."""


def _manager_buffer(*, row_shape: tuple[int, ...], lifetime: BufferLifetime) -> BufferContract:
    return BufferContract(
        row_shape=row_shape,
        dtype=np.dtype(get_global_dtype()).name,
        layout=BufferLayout.C_CONTIGUOUS,
        placement=BufferPlacement.host(),
        owner=BufferOwner.MANAGER,
        mutability=BufferMutability.READ_ONLY,
        lifetime=lifetime,
        dlpack_exportable=False,
    )


def _state_requirement(
    *,
    key: str,
    selector: EntitySelector,
    field_kind: StateFieldKind,
    shape: tuple[int, ...],
    frame: ReferenceFrame,
    unit: PhysicalUnit,
    quaternion_order: QuaternionOrder = QuaternionOrder.NONE,
    entity_axis: int | None = None,
) -> StateRequirement:
    return StateRequirement(
        semantic_key=key,
        selector=selector,
        field_kind=field_kind,
        tensor=TensorSpec(
            shape,
            np.dtype(get_global_dtype()).name,
            frame=frame,
            unit=unit,
            quaternion_order=quaternion_order,
        ),
        entity_axis=entity_axis,
    )


def _validate_reference_profile(
    cfg: G1WalkEnvCfg,
    *,
    allow_pd_randomization: bool = False,
    allow_dof_armature_randomization: bool = False,
) -> G1WalkRewardConfig:
    """Reject legacy features whose effects are not in this compiled slice."""

    if not isinstance(allow_pd_randomization, bool):
        raise G1ManagedReferenceError("allow_pd_randomization must be a bool")
    if not isinstance(allow_dof_armature_randomization, bool):
        raise G1ManagedReferenceError("allow_dof_armature_randomization must be a bool")

    reward = cfg.reward_config
    if not isinstance(reward, G1WalkRewardConfig):
        raise G1ManagedReferenceError("G1 managed reference requires a G1WalkRewardConfig")
    unsupported_rewards = tuple(sorted(set(reward.scales) - _SUPPORTED_REWARD_TERMS))
    if unsupported_rewards:
        raise G1ManagedReferenceError(
            "G1 managed reference does not implement reward terms: "
            + ", ".join(unsupported_rewards)
        )
    if any(not np.isfinite(float(scale)) for scale in reward.scales.values()):
        raise G1ManagedReferenceError("G1 managed reference reward scales must be finite")
    if cfg.curriculum.enabled:
        raise G1ManagedReferenceError(
            "G1 managed reference does not implement the legacy penalty curriculum"
        )
    if cfg.numba_acceleration:
        raise G1ManagedReferenceError(
            "G1 managed reference does not select the legacy Numba executor"
        )
    if cfg.commands.heading_command:
        raise G1ManagedReferenceError(
            "G1 managed reference does not implement heading-command task state"
        )
    if cfg.commands.resampling_time != 0.0:
        raise G1ManagedReferenceError(
            "G1 managed reference only supports reset-sampled velocity commands"
        )
    if cfg.gait_phase_init_mode not in {"offset_phase", "independent"}:
        raise G1ManagedReferenceError(
            "G1 managed reference requires gait_phase_init_mode='offset_phase' or 'independent'"
        )

    dr = cfg.domain_rand
    enabled_dr = tuple(
        name
        for name, enabled in (
            ("randomize_base_mass", dr.randomize_base_mass),
            ("randomize_body_mass", dr.randomize_body_mass),
            ("random_com", dr.random_com),
            ("randomize_gravity", dr.randomize_gravity),
            ("randomize_ground_friction", dr.randomize_ground_friction),
            (
                "randomize_dof_armature",
                dr.randomize_dof_armature and not allow_dof_armature_randomization,
            ),
            ("push_robots", dr.push_robots),
            ("randomize_kp", dr.randomize_kp and not allow_pd_randomization),
            ("randomize_kd", dr.randomize_kd and not allow_pd_randomization),
        )
        if enabled
    )
    if enabled_dr:
        raise G1ManagedReferenceError(
            "G1 managed reference has no typed DR/Event implementation for: "
            + ", ".join(enabled_dr)
        )
    return reward


def _action_scale(cfg: G1WalkEnvCfg, action_dim: int) -> np.ndarray:
    raw = np.asarray(cfg.control_config.action_scale, dtype=get_global_dtype())
    if raw.ndim == 0:
        values = np.full((action_dim,), raw.item(), dtype=get_global_dtype())
    elif raw.shape == (action_dim,):
        values = np.asarray(raw, dtype=get_global_dtype())
    else:
        raise G1ManagedReferenceError(
            f"G1 action_scale must be scalar or shape ({action_dim},), got {raw.shape}"
        )
    if not np.isfinite(values).all() or np.any(values <= 0.0):
        raise G1ManagedReferenceError("G1 action_scale must contain finite positive values")
    return values


def _walk_observation_profile(reward: G1WalkRewardConfig, cfg: G1WalkEnvCfg) -> bool:
    scales = reward.scales
    if any(
        key in scales
        for key in (
            "penalty_orientation",
            "penalty_ang_vel_xy",
            "penalty_action_rate",
            "alive",
        )
    ):
        return True
    if any(key in scales for key in ("orientation", "ang_vel_xy", "action_rate")):
        return False
    return cfg.curriculum.enabled


def _reset_suffix_for_dof(*, kind: str, index: int) -> str:
    return f"dof_{kind}_{index:02d}"


def _reset_term_key(*, suffix: str) -> str:
    return f"{_RESET_TERM}.{suffix}"


def _g1_selectors(
    actuator_names: tuple[str, ...],
) -> tuple[
    EntitySelector,
    EntitySelector,
    tuple[EntitySelector, ...],
    tuple[EntitySelector, ...],
]:
    root = EntitySelector(
        key="g1.root",
        entity="g1",
        kind=EntityKind.ROOT,
        expressions=(_ROOT_NAME,),
    )
    dofs = EntitySelector(
        key="g1.actuated_dofs",
        entity="g1",
        kind=EntityKind.DOF,
        expressions=actuator_names,
    )
    reset_position = tuple(
        EntitySelector(
            key=f"g1.reset.dof_position.{index:02d}",
            entity="g1",
            kind=EntityKind.DOF,
            expressions=(name,),
        )
        for index, name in enumerate(actuator_names)
    )
    reset_velocity = tuple(
        EntitySelector(
            key=f"g1.reset.dof_velocity.{index:02d}",
            entity="g1",
            kind=EntityKind.DOF,
            expressions=(name,),
        )
        for index, name in enumerate(actuator_names)
    )
    return root, dofs, reset_position, reset_velocity


def _g1_state_requirements(
    *, root: EntitySelector, dofs: EntitySelector, action_dim: int
) -> tuple[StateRequirement, ...]:
    sensors = (
        (
            "pelvis_local_linvel",
            ReferenceFrame.SENSOR,
            PhysicalUnit.METER_PER_SECOND,
        ),
        ("torso_gyro", ReferenceFrame.SENSOR, PhysicalUnit.RADIAN_PER_SECOND),
        ("torso_upvector", ReferenceFrame.WORLD, PhysicalUnit.UNITLESS),
        ("left_foot_pos", ReferenceFrame.WORLD, PhysicalUnit.METER),
        ("right_foot_pos", ReferenceFrame.WORLD, PhysicalUnit.METER),
    )
    requirements = [
        _state_requirement(
            key="g1.root.position",
            selector=root,
            field_kind=StateFieldKind.POSITION,
            shape=(3,),
            frame=ReferenceFrame.WORLD,
            unit=PhysicalUnit.METER,
        ),
        _state_requirement(
            key="g1.root.orientation",
            selector=root,
            field_kind=StateFieldKind.ORIENTATION,
            shape=(4,),
            frame=ReferenceFrame.WORLD,
            unit=PhysicalUnit.QUATERNION,
            quaternion_order=QuaternionOrder.WXYZ,
        ),
        _state_requirement(
            key="g1.root.linear_velocity",
            selector=root,
            field_kind=StateFieldKind.LINEAR_VELOCITY,
            shape=(3,),
            frame=ReferenceFrame.WORLD,
            unit=PhysicalUnit.METER_PER_SECOND,
        ),
        _state_requirement(
            key="g1.root.angular_velocity",
            selector=root,
            field_kind=StateFieldKind.ANGULAR_VELOCITY,
            shape=(3,),
            frame=ReferenceFrame.WORLD,
            unit=PhysicalUnit.RADIAN_PER_SECOND,
        ),
        _state_requirement(
            key="g1.dof.position",
            selector=dofs,
            field_kind=StateFieldKind.POSITION,
            shape=(action_dim,),
            frame=ReferenceFrame.JOINT,
            unit=PhysicalUnit.RADIAN,
            entity_axis=0,
        ),
        _state_requirement(
            key="g1.dof.angular_velocity",
            selector=dofs,
            field_kind=StateFieldKind.ANGULAR_VELOCITY,
            shape=(action_dim,),
            frame=ReferenceFrame.JOINT,
            unit=PhysicalUnit.RADIAN_PER_SECOND,
            entity_axis=0,
        ),
    ]
    for name, frame, unit in sensors:
        requirements.append(
            _state_requirement(
                key=f"g1.sensor.{name}",
                selector=EntitySelector(
                    key=f"g1.sensor.{name}",
                    entity="g1",
                    kind=EntityKind.SENSOR,
                    expressions=(name,),
                ),
                field_kind=StateFieldKind.VALUE,
                shape=(3,),
                frame=frame,
                unit=unit,
            )
        )
    return tuple(requirements)


def _g1_reset_templates(
    *,
    root: EntitySelector,
    reset_position: tuple[EntitySelector, ...],
    reset_velocity: tuple[EntitySelector, ...],
) -> tuple[MutationTemplate, ...]:
    templates: list[MutationTemplate] = []
    for suffix, target_key, field_kind, row_shape in _ROOT_RESET_SUFFIXES:
        templates.append(
            MutationTemplate(
                key_suffix=suffix,
                target_key=target_key,
                target_kind=MutationTargetKind.SIMULATION_STATE,
                selector=root,
                field_kind=field_kind,
                trigger=MutationTrigger.RESET,
                commit_phase=MutationCommitPhase.RESET,
                operation=MutationOperation.SET,
                baseline=MutationBaseline.DEFAULT,
                persistence=MutationPersistence.EPISODE,
                recompute=MutationRecomputeLevel.KINEMATICS,
                value_template=_manager_buffer(
                    row_shape=row_shape, lifetime=BufferLifetime.UNTIL_COMMIT
                ),
            )
        )
    for index, selector in enumerate(reset_position):
        templates.append(
            MutationTemplate(
                key_suffix=_reset_suffix_for_dof(kind="position", index=index),
                target_key="state.dof.position",
                target_kind=MutationTargetKind.SIMULATION_STATE,
                selector=selector,
                field_kind=MutationFieldKind.POSITION,
                trigger=MutationTrigger.RESET,
                commit_phase=MutationCommitPhase.RESET,
                operation=MutationOperation.SET,
                baseline=MutationBaseline.DEFAULT,
                persistence=MutationPersistence.EPISODE,
                recompute=MutationRecomputeLevel.KINEMATICS,
                value_template=_manager_buffer(
                    row_shape=(1,), lifetime=BufferLifetime.UNTIL_COMMIT
                ),
            )
        )
    for index, selector in enumerate(reset_velocity):
        templates.append(
            MutationTemplate(
                key_suffix=_reset_suffix_for_dof(kind="velocity", index=index),
                target_key="state.dof.angular_velocity",
                target_kind=MutationTargetKind.SIMULATION_STATE,
                selector=selector,
                field_kind=MutationFieldKind.ANGULAR_VELOCITY,
                trigger=MutationTrigger.RESET,
                commit_phase=MutationCommitPhase.RESET,
                operation=MutationOperation.SET,
                baseline=MutationBaseline.DEFAULT,
                persistence=MutationPersistence.EPISODE,
                recompute=MutationRecomputeLevel.KINEMATICS,
                value_template=_manager_buffer(
                    row_shape=(1,), lifetime=BufferLifetime.UNTIL_COMMIT
                ),
            )
        )
    return tuple(templates)


def compile_g1_managed_reference_task(
    *, backend: SimBackend, cfg: G1WalkEnvCfg
) -> CompiledTaskPlan:
    """Compile the static G1 host-reference plan through public backend APIs.

    This function only executes cold-path metadata queries.  Its returned plan
    contains bound IDs and mutation selector metadata, but no backend object.
    """

    if not isinstance(backend, SimBackend):
        raise G1ManagedReferenceError("G1 managed reference requires a SimBackend")
    reward = _validate_reference_profile(cfg)
    actuator_names = backend.get_actuator_names()
    if not actuator_names:
        raise G1ManagedReferenceError("G1 managed reference requires named actuators")
    if len(set(actuator_names)) != len(actuator_names):
        raise G1ManagedReferenceError("G1 managed reference actuator names must be unique")
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
            key="g1.reference.reset",
            version="1",
            phase=TermPhase.RESET,
            role=TermRole.EVENT,
            mutation_templates=reset_templates,
        )
    )
    registry.register(
        TermDefinition(
            key="g1.reference.termination",
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
            key="g1.reference.reward",
            version="1",
            phase=TermPhase.REWARD,
            role=TermRole.REWARD,
            state_requirements=state_requirements,
        )
    )
    registry.register(
        TermDefinition(
            key="g1.reference.actor_observation",
            version="1",
            phase=TermPhase.TERMINAL_OBSERVATION,
            role=TermRole.OBSERVATION,
            state_requirements=state_requirements,
            output=TensorSpec((_ACTOR_WIDTH,), np.dtype(get_global_dtype()).name),
        )
    )
    registry.register(
        TermDefinition(
            key="g1.reference.critic_observation",
            version="1",
            phase=TermPhase.TERMINAL_OBSERVATION,
            role=TermRole.OBSERVATION,
            state_requirements=state_requirements,
            output=TensorSpec((_CRITIC_WIDTH,), np.dtype(get_global_dtype()).name),
        )
    )
    task = TaskSpec.create(
        key="g1_walk_flat.managed_reference",
        terms=(
            TermInvocation.create(key=_RESET_TERM, definition_key="g1.reference.reset"),
            TermInvocation.create(
                key="g1_termination",
                definition_key="g1.reference.termination",
                dependencies=(_RESET_TERM,),
            ),
            TermInvocation.create(
                key="g1_reward",
                definition_key="g1.reference.reward",
                dependencies=("g1_termination",),
            ),
            TermInvocation.create(
                key="g1_actor_observation",
                definition_key="g1.reference.actor_observation",
                dependencies=("g1_reward",),
                observation_group="obs",
            ),
            TermInvocation.create(
                key="g1_critic_observation",
                definition_key="g1.reference.critic_observation",
                dependencies=("g1_reward",),
                observation_group="critic",
            ),
        ),
        control=ControlSpec(
            semantic_key="g1.joint.position_target",
            buffer=_manager_buffer(
                row_shape=(action_dim,), lifetime=BufferLifetime.UNTIL_STEP_COMPLETE
            ),
            physics_substeps_per_control=cfg.sim_substeps,
        ),
        execution_profile=ExecutionProfile.HOST_NUMPY,
        executor_key=G1_MANAGED_REFERENCE_EXECUTOR_KEY,
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
        raise G1ManagedReferenceError("compiled G1 policy ABI has an unexpected observation width")
    if reward is not cfg.reward_config:  # pragma: no cover - type-narrowing invariant
        raise G1ManagedReferenceError("G1 reward configuration changed during compilation")
    return plan


@dataclass(frozen=True)
class _G1KernelConfig:
    action_scale: np.ndarray
    default_angles: np.ndarray
    initial_qpos: np.ndarray
    initial_qvel: np.ndarray
    reward_terms: tuple[tuple[str, float], ...]
    ctrl_dt: float
    tracking_sigma: float
    gait_phase_delta: float
    gait_phase_init_mode: str
    reset_base_qvel_limit: float
    command_low: np.ndarray
    command_high: np.ndarray
    standing_probability: float
    base_height_target: float
    min_base_height: float
    max_tilt_rad: float
    feet_phase_swing_height: float
    feet_phase_tracking_sigma: float
    min_forward_speed_for_gait_reward: float
    close_feet_threshold: float
    pose_weights: np.ndarray
    upper_body_pose_weights: np.ndarray
    walk_observation_profile: bool
    observation_noise_level: float
    observation_noise_scale_joint_angle: float
    observation_noise_scale_joint_vel: float
    observation_noise_scale_gyro: float
    observation_noise_scale_gravity: float
    reset_seed: int
    observation_noise_seed: int | None


@dataclass(frozen=True)
class _G1StateViews:
    """Named, validated views of the canonical compiled G1 state fields.

    ``TaskCompiler`` deliberately orders fields by semantic key, rather than
    by a task's preferred math order.  Keeping the mapping here explicit
    prevents a future compiler ordering change from silently feeding the
    reward or observation kernel a semantically different vector.
    """

    dof_angular_velocity: np.ndarray
    dof_position: np.ndarray
    root_angular_velocity: np.ndarray
    root_linear_velocity: np.ndarray
    root_orientation: np.ndarray
    root_position: np.ndarray
    left_foot_position: np.ndarray
    pelvis_local_linear_velocity: np.ndarray
    right_foot_position: np.ndarray
    torso_gyro: np.ndarray
    torso_upvector: np.ndarray


@dataclass
class _G1ManagedTaskState:
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
    reset_rng: np.random.RandomState
    observation_noise_rng: np.random.Generator | None
    reward_means: np.ndarray
    logged_reward_means: np.ndarray
    has_logged_reward: np.ndarray


@dataclass(frozen=True)
class _G1ResetSample:
    rows: RowSelection


class G1ManagedReferenceKernel:
    """Pure host-Numpy G1 task kernel over the bound typed state plan."""

    executor_key = G1_MANAGED_REFERENCE_EXECUTOR_KEY

    def __init__(self, config: _G1KernelConfig) -> None:
        if not isinstance(config, _G1KernelConfig):
            raise G1ManagedReferenceError("G1 managed reference kernel requires frozen config")
        self._config = config
        self._action_dim = int(config.default_angles.size)
        self._binding: ManagedKernelBinding | None = None
        self._state_indices: tuple[int, ...] | None = None
        self._obs_buffer_indices: tuple[int, int] | None = None
        self._mutation_plan: BoundMutationPlan | None = None
        self._root_reset_indices: tuple[int, int, int, int] | None = None
        self._dof_position_reset_indices: tuple[int, ...] | None = None
        self._dof_velocity_reset_indices: tuple[int, ...] | None = None

    def bind(self, *, binding: ManagedKernelBinding) -> None:
        if self._binding is not None:
            raise G1ManagedReferenceError("G1 managed reference kernel may only bind once")
        if not isinstance(binding, ManagedKernelBinding):
            raise G1ManagedReferenceError("G1 managed reference requires a ManagedKernelBinding")
        if binding.execution_profile is not ExecutionProfile.HOST_NUMPY:
            raise G1ManagedReferenceError("G1 managed reference only supports host_numpy")
        if binding.dtype != np.dtype(get_global_dtype()).name:
            raise G1ManagedReferenceError(
                "G1 managed reference dtype must match the repository global dtype"
            )
        state_index_by_key = dict(binding.state_field_indices)
        missing_state = tuple(key for key in _STATE_KEYS if key not in state_index_by_key)
        if missing_state:
            raise G1ManagedReferenceError(
                "G1 managed reference plan is missing state fields: " + ", ".join(missing_state)
            )
        self._state_indices = tuple(state_index_by_key[key] for key in _STATE_KEYS)
        obs_index_by_key = dict(binding.observation_buffer_indices)
        try:
            self._obs_buffer_indices = (obs_index_by_key["obs"], obs_index_by_key["critic"])
        except KeyError as exc:
            raise G1ManagedReferenceError(
                "G1 managed reference requires obs and critic output buffers"
            ) from exc

        mutation_plan = binding.mutation_plan
        if mutation_plan is None:
            raise G1ManagedReferenceError("G1 managed reference requires typed reset mutations")
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
            raise G1ManagedReferenceError(
                "G1 managed reference plan is missing reset mutations: "
                + ", ".join(missing_mutations)
            )
        self._binding = binding
        self._mutation_plan = mutation_plan
        self._root_reset_indices = (
            mutation_indices[root_keys[0]],
            mutation_indices[root_keys[1]],
            mutation_indices[root_keys[2]],
            mutation_indices[root_keys[3]],
        )
        self._dof_position_reset_indices = tuple(mutation_indices[key] for key in position_keys)
        self._dof_velocity_reset_indices = tuple(mutation_indices[key] for key in velocity_keys)

    def _require_binding(self) -> ManagedKernelBinding:
        if self._binding is None:
            raise G1ManagedReferenceError("G1 managed reference kernel has not been bound")
        return self._binding

    def _require_state_indices(self) -> tuple[int, ...]:
        if self._state_indices is None:
            raise G1ManagedReferenceError("G1 managed reference state fields are not bound")
        return self._state_indices

    def _require_observation_indices(self) -> tuple[int, int]:
        if self._obs_buffer_indices is None:
            raise G1ManagedReferenceError("G1 managed reference observations are not bound")
        return self._obs_buffer_indices

    def _require_reset_indices(
        self,
    ) -> tuple[tuple[int, int, int, int], tuple[int, ...], tuple[int, ...]]:
        if (
            self._root_reset_indices is None
            or self._dof_position_reset_indices is None
            or self._dof_velocity_reset_indices is None
        ):
            raise G1ManagedReferenceError("G1 managed reference reset fields are not bound")
        return (
            self._root_reset_indices,
            self._dof_position_reset_indices,
            self._dof_velocity_reset_indices,
        )

    @staticmethod
    def _require_task_state(task_state: object) -> _G1ManagedTaskState:
        if not isinstance(task_state, _G1ManagedTaskState):
            raise G1ManagedReferenceError("G1 managed reference received foreign task state")
        return task_state

    @staticmethod
    def _host_array(value: object, *, name: str) -> np.ndarray:
        if not isinstance(value, np.ndarray):
            raise G1ManagedReferenceError(f"G1 typed state {name} must be a numpy array")
        return value

    def _state_views(self, state: StateBatch) -> _G1StateViews:
        """Return semantic state views without relying on compiler field order."""

        state.assert_valid()
        binding = self._require_binding()
        expected_dtype = np.dtype(binding.dtype)
        expected_shapes = {
            "g1.dof.angular_velocity": (self._action_dim,),
            "g1.dof.position": (self._action_dim,),
            "g1.root.angular_velocity": (3,),
            "g1.root.linear_velocity": (3,),
            "g1.root.orientation": (4,),
            "g1.root.position": (3,),
            "g1.sensor.left_foot_pos": (3,),
            "g1.sensor.pelvis_local_linvel": (3,),
            "g1.sensor.right_foot_pos": (3,),
            "g1.sensor.torso_gyro": (3,),
            "g1.sensor.torso_upvector": (3,),
        }
        arrays: dict[str, np.ndarray] = {}
        for key, index in zip(_STATE_KEYS, self._require_state_indices(), strict=True):
            array = self._host_array(state.buffer_at(index).handle, name=key)
            expected_shape = (state.rows.count, *expected_shapes[key])
            if array.shape != expected_shape:
                raise G1ManagedReferenceError(
                    f"G1 typed state {key} must have shape {expected_shape}, got {array.shape}"
                )
            if array.dtype != expected_dtype:
                raise G1ManagedReferenceError(
                    f"G1 typed state {key} must have dtype {expected_dtype.name}, "
                    f"got {array.dtype.name}"
                )
            arrays[key] = array
        return _G1StateViews(
            dof_angular_velocity=arrays["g1.dof.angular_velocity"],
            dof_position=arrays["g1.dof.position"],
            root_angular_velocity=arrays["g1.root.angular_velocity"],
            root_linear_velocity=arrays["g1.root.linear_velocity"],
            root_orientation=arrays["g1.root.orientation"],
            root_position=arrays["g1.root.position"],
            left_foot_position=arrays["g1.sensor.left_foot_pos"],
            pelvis_local_linear_velocity=arrays["g1.sensor.pelvis_local_linvel"],
            right_foot_position=arrays["g1.sensor.right_foot_pos"],
            torso_gyro=arrays["g1.sensor.torso_gyro"],
            torso_upvector=arrays["g1.sensor.torso_upvector"],
        )

    def create_task_state(self, *, num_envs: int, dtype: np.dtype[Any]) -> object:
        binding = self._require_binding()
        if num_envs != binding.num_envs:
            raise G1ManagedReferenceError("G1 task state row universe differs from kernel binding")
        if dtype != np.dtype(binding.dtype):
            raise G1ManagedReferenceError("G1 task state dtype differs from kernel binding")
        if dtype != np.dtype(get_global_dtype()):
            raise G1ManagedReferenceError("G1 task state dtype differs from global dtype")
        if self._mutation_plan is None:
            raise G1ManagedReferenceError("G1 task state requires a bound mutation plan")
        reset_value_buffers = tuple(
            np.empty((num_envs, *spec.value_buffer.row_shape), dtype=dtype)
            for spec in self._mutation_plan.specs
        )
        noise_rng = (
            None
            if self._config.observation_noise_level <= 0.0
            else np.random.default_rng(self._config.observation_noise_seed)
        )
        return _G1ManagedTaskState(
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
            reset_rng=np.random.RandomState(self._config.reset_seed),
            observation_noise_rng=noise_rng,
            reward_means=np.zeros((len(self._config.reward_terms),), dtype=dtype),
            logged_reward_means=np.zeros((len(self._config.reward_terms),), dtype=dtype),
            has_logged_reward=np.zeros((len(self._config.reward_terms),), dtype=bool),
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
            raise G1ManagedReferenceError("G1 managed actions do not match task action state")
        if control_out.shape != actions.shape or control_out.dtype != actions.dtype:
            raise G1ManagedReferenceError("G1 managed control output does not match actions")
        np.copyto(task.last_actions, task.current_actions)
        np.copyto(task.current_actions, actions)
        task.gait_phase += self._config.gait_phase_delta
        np.remainder(task.gait_phase, 2.0 * math.pi, out=task.gait_phase)
        np.multiply(actions, self._config.action_scale, out=control_out)
        control_out += self._config.default_angles

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
        views = self._state_views(state)
        if terminated_out.shape != task.steps.shape or terminated_out.dtype != np.dtype(bool):
            raise G1ManagedReferenceError("G1 terminated output has an invalid shape or dtype")
        tilt = np.arccos(np.clip(views.torso_upvector[:, 2], -1.0, 1.0))
        np.logical_or(
            tilt > self._config.max_tilt_rad,
            views.root_position[:, 2] < self._config.min_base_height,
            out=terminated_out,
        )

    def _reward_value(
        self,
        *,
        name: str,
        task: _G1ManagedTaskState,
        root_position: np.ndarray,
        dof_position: np.ndarray,
        linear_velocity: np.ndarray,
        gyro: np.ndarray,
        upvector: np.ndarray,
        left_foot_position: np.ndarray,
        right_foot_position: np.ndarray,
    ) -> np.ndarray:
        if name == "tracking_lin_vel":
            error = np.sum(np.square(task.commands[:, :2] - linear_velocity[:, :2]), axis=1)
            return np.exp(-error / self._config.tracking_sigma)
        if name == "tracking_ang_vel":
            error = np.square(task.commands[:, 2] - gyro[:, 2])
            return np.exp(-error / self._config.tracking_sigma)
        if name == "forward_progress":
            commanded_speed = np.maximum(task.commands[:, 0], 1.0e-6)
            return np.minimum(np.maximum(linear_velocity[:, 0], 0.0) / commanded_speed, 1.0)
        if name == "under_speed":
            commanded_speed = np.maximum(task.commands[:, 0], 1.0e-6)
            gap = np.maximum(task.commands[:, 0] - np.maximum(linear_velocity[:, 0], 0.0), 0.0)
            return gap / commanded_speed
        if name == "lin_vel_z":
            return np.square(linear_velocity[:, 2])
        if name in {"orientation", "penalty_orientation"}:
            return np.square(upvector[:, 0]) + np.square(upvector[:, 1])
        if name in {"ang_vel_xy", "penalty_ang_vel_xy"}:
            return np.sum(np.square(gyro[:, :2]), axis=1)
        if name in {"action_rate", "penalty_action_rate"}:
            return np.sum(np.square(task.current_actions - task.last_actions), axis=1)
        if name == "base_height":
            return np.square(root_position[:, 2] - self._config.base_height_target)
        if name == "pose":
            diff = dof_position - self._config.default_angles
            return np.asarray(
                np.sum(self._config.pose_weights * np.square(diff), axis=1),
                dtype=dof_position.dtype,
            )
        if name == "upper_body_pose":
            diff = dof_position - self._config.default_angles
            return np.asarray(
                np.sum(self._config.upper_body_pose_weights * np.square(diff), axis=1),
                dtype=dof_position.dtype,
            )
        if name == "penalty_close_feet_xy":
            feet_distance = np.linalg.norm(
                left_foot_position[:, :2] - right_foot_position[:, :2], axis=1
            )
            return np.where(
                feet_distance < self._config.close_feet_threshold,
                np.square(feet_distance - self._config.close_feet_threshold),
                0.0,
            )
        if name in {"feet_phase", "feet_phase_contrast"}:
            left_target, right_target = compute_feet_phase_height_targets(
                task.gait_phase, self._config.feet_phase_swing_height
            )
            gate = compute_forward_speed_gate(
                linear_velocity, self._config.min_forward_speed_for_gait_reward
            )
            if name == "feet_phase":
                error = np.square(left_foot_position[:, 2] - left_target) + np.square(
                    right_foot_position[:, 2] - right_target
                )
            else:
                error = np.square(
                    (left_foot_position[:, 2] - right_foot_position[:, 2])
                    - (left_target - right_target)
                )
            return np.asarray(
                np.exp(-error / self._config.feet_phase_tracking_sigma) * gate,
                dtype=linear_velocity.dtype,
            )
        if name == "alive":
            return np.ones((linear_velocity.shape[0],), dtype=linear_velocity.dtype)
        raise G1ManagedReferenceError(f"unsupported bound G1 reward term {name!r}")

    def evaluate_reward(
        self,
        *,
        state: StateBatch,
        task_state: object,
        reward_out: np.ndarray,
    ) -> None:
        task = self._require_task_state(task_state)
        views = self._state_views(state)
        if reward_out.shape != task.steps.shape or reward_out.dtype != task.current_actions.dtype:
            raise G1ManagedReferenceError("G1 reward output has an invalid shape or dtype")
        reward_out.fill(0.0)
        for index, (name, scale) in enumerate(self._config.reward_terms):
            if scale == 0.0:
                task.reward_means[index] = 0.0
                continue
            value = self._reward_value(
                name=name,
                task=task,
                root_position=views.root_position,
                dof_position=views.dof_position,
                linear_velocity=views.pelvis_local_linear_velocity,
                gyro=views.torso_gyro,
                upvector=views.torso_upvector,
                left_foot_position=views.left_foot_position,
                right_foot_position=views.right_foot_position,
            )
            weighted = value * scale
            reward_out += weighted
            task.reward_means[index] = np.mean(weighted)
        reward_out *= np.asarray(self._config.ctrl_dt, dtype=reward_out.dtype)

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

    def _observation_noise(
        self, task: _G1ManagedTaskState, values: np.ndarray, scale: float
    ) -> np.ndarray:
        if self._config.observation_noise_level <= 0.0:
            return values
        rng = task.observation_noise_rng
        if rng is None:  # pragma: no cover - create_task_state invariant
            raise G1ManagedReferenceError("G1 observation noise RNG was not initialized")
        noise = rng.uniform(-1.0, 1.0, values.shape).astype(values.dtype)
        return values + noise * self._config.observation_noise_level * scale

    def write_observations(
        self,
        *,
        state: StateBatch,
        task_state: object,
        observation_buffers: tuple[np.ndarray, ...],
    ) -> None:
        task = self._require_task_state(task_state)
        views = self._state_views(state)
        actor_index, critic_index = self._require_observation_indices()
        try:
            actor_all = observation_buffers[actor_index]
            critic_all = observation_buffers[critic_index]
        except IndexError as exc:
            raise G1ManagedReferenceError("G1 runtime observation buffers are incomplete") from exc
        binding = self._require_binding()
        if actor_all.shape != (binding.num_envs, _ACTOR_WIDTH) or critic_all.shape != (
            binding.num_envs,
            _CRITIC_WIDTH,
        ):
            raise G1ManagedReferenceError("G1 runtime observation buffers have invalid widths")
        if (
            actor_all.dtype != task.current_actions.dtype
            or critic_all.dtype != task.current_actions.dtype
        ):
            raise G1ManagedReferenceError("G1 runtime observation buffers have an invalid dtype")
        if state.rows.is_all:
            target_rows: slice | np.ndarray = slice(None)
            commands = task.commands
            current_actions = task.current_actions
            gait_phase = task.gait_phase
        else:
            assert state.rows.indices is not None
            indices = np.asarray(state.rows.indices, dtype=np.intp)
            target_rows = indices
            commands = task.commands[indices]
            current_actions = task.current_actions[indices]
            gait_phase = task.gait_phase[indices]
        diff = views.dof_position - self._config.default_angles
        gyro_scale = 0.25 if self._config.walk_observation_profile else 1.0
        dof_velocity_scale = 0.05 if self._config.walk_observation_profile else 1.0
        linvel_scale = 2.0 if self._config.walk_observation_profile else 1.0

        cursor = 0
        actor_all[target_rows, cursor : cursor + 3] = (
            self._observation_noise(
                task, views.torso_gyro, self._config.observation_noise_scale_gyro
            )
            * gyro_scale
        )
        cursor += 3
        actor_all[target_rows, cursor : cursor + 3] = -self._observation_noise(
            task, views.torso_upvector, self._config.observation_noise_scale_gravity
        )
        cursor += 3
        actor_all[target_rows, cursor : cursor + self._action_dim] = self._observation_noise(
            task, diff, self._config.observation_noise_scale_joint_angle
        )
        cursor += self._action_dim
        actor_all[target_rows, cursor : cursor + self._action_dim] = (
            self._observation_noise(
                task, views.dof_angular_velocity, self._config.observation_noise_scale_joint_vel
            )
            * dof_velocity_scale
        )
        cursor += self._action_dim
        actor_all[target_rows, cursor : cursor + self._action_dim] = current_actions
        cursor += self._action_dim
        actor_all[target_rows, cursor : cursor + 3] = commands
        cursor += 3
        actor_all[target_rows, cursor : cursor + 2] = gait_phase
        cursor += 2
        if cursor != _ACTOR_WIDTH:  # pragma: no cover - static layout assertion
            raise G1ManagedReferenceError("G1 actor observation layout is inconsistent")

        cursor = 0
        critic_all[target_rows, cursor : cursor + 3] = views.torso_gyro * gyro_scale
        cursor += 3
        critic_all[target_rows, cursor : cursor + 3] = -views.torso_upvector
        cursor += 3
        critic_all[target_rows, cursor : cursor + self._action_dim] = diff
        cursor += self._action_dim
        critic_all[target_rows, cursor : cursor + self._action_dim] = (
            views.dof_angular_velocity * dof_velocity_scale
        )
        cursor += self._action_dim
        critic_all[target_rows, cursor : cursor + self._action_dim] = current_actions
        cursor += self._action_dim
        critic_all[target_rows, cursor : cursor + 3] = commands
        cursor += 3
        critic_all[target_rows, cursor : cursor + 2] = gait_phase
        cursor += 2
        critic_all[target_rows, cursor : cursor + 3] = (
            views.pelvis_local_linear_velocity * linvel_scale
        )
        cursor += 3
        if cursor != _CRITIC_WIDTH:  # pragma: no cover - static layout assertion
            raise G1ManagedReferenceError("G1 critic observation layout is inconsistent")
        if state.phase.value == "terminal":
            if state.rows.is_all:
                task.steps += 1
            else:  # pragma: no cover - runtime only terminal-materializes all rows
                assert state.rows.indices is not None
                task.steps[np.asarray(state.rows.indices, dtype=np.intp)] += 1

    def _sample_commands(self, task: _G1ManagedTaskState, count: int) -> None:
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

    def _sample_gait_phase(self, task: _G1ManagedTaskState, count: int) -> None:
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

    def _prepare_reset_values(self, task: _G1ManagedTaskState, rows: RowSelection) -> None:
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
        root_values = (
            qpos[:, :3],
            qpos[:, 3:7],
            qvel[:, :3],
            qvel[:, 3:6],
        )
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
            raise G1ManagedReferenceError("G1 reset rows differ from task row universe")
        if self._mutation_plan is None:
            raise G1ManagedReferenceError("G1 reset requires a bound mutation plan")
        self._prepare_reset_values(task, rows)
        values = tuple(
            MutationValueBatch(
                plan=self._mutation_plan,
                field_index=index,
                rows=rows,
                buffer=BufferView(
                    handle=buffer[: rows.count],
                    shape=(rows.count, *buffer.shape[1:]),
                    contract=self._mutation_plan.specs[index].value_buffer,
                ),
            )
            for index, buffer in enumerate(task.reset_value_buffers)
        )
        mutation = TypedBackendMutationBatch(
            plan=self._mutation_plan,
            rows=rows,
            state=SimulationStateMutationBatch(values=values),
        )
        return ManagedResetRequest(
            rows=rows,
            mutation_batch=mutation,
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
            raise G1ManagedReferenceError("G1 reset request carries foreign task state")
        sample = request.kernel_state
        if sample.rows != request.rows or state.rows != request.rows:
            raise G1ManagedReferenceError("G1 reset sample rows do not match reset state")
        state.assert_valid()
        count = request.rows.count
        if request.rows.is_all:
            target = slice(None)
        else:
            assert request.rows.indices is not None
            target = np.asarray(request.rows.indices, dtype=np.intp)
        task.commands[target] = task.reset_commands[:count]
        task.current_actions[target] = 0.0
        task.last_actions[target] = 0.0
        task.gait_phase[target] = task.reset_gait_phase[:count]
        task.steps[target] = 0


def _kernel_config(
    *,
    backend: SimBackend,
    cfg: G1WalkEnvCfg,
    reset_seed: int,
    observation_noise_seed: int | None,
    allow_pd_randomization: bool = False,
    allow_dof_armature_randomization: bool = False,
) -> _G1KernelConfig:
    reward = _validate_reference_profile(
        cfg,
        allow_pd_randomization=allow_pd_randomization,
        allow_dof_armature_randomization=allow_dof_armature_randomization,
    )
    if not np.isfinite(float(cfg.sim_dt)) or float(cfg.sim_dt) <= 0.0:
        raise G1ManagedReferenceError("G1 sim_dt must be finite and positive")
    if not np.isfinite(float(cfg.ctrl_dt)) or float(cfg.ctrl_dt) <= 0.0:
        raise G1ManagedReferenceError("G1 ctrl_dt must be finite and positive")
    if cfg.sim_substeps <= 0:
        raise G1ManagedReferenceError("G1 managed reference requires positive sim_substeps")
    if isinstance(reset_seed, bool) or not isinstance(reset_seed, int) or reset_seed < 0:
        raise G1ManagedReferenceError("G1 reset_seed must be a non-negative integer")
    if observation_noise_seed is None:
        observation_noise_seed = cfg.noise_config.seed
    if observation_noise_seed is not None and (
        isinstance(observation_noise_seed, bool)
        or not isinstance(observation_noise_seed, int)
        or observation_noise_seed < 0
    ):
        raise G1ManagedReferenceError("G1 observation_noise_seed must be non-negative or None")
    if cfg.noise_config.level > 0.0 and observation_noise_seed is None:
        raise G1ManagedReferenceError(
            "G1 managed reference requires an explicit observation noise seed when noise is enabled"
        )
    actuator_names = backend.get_actuator_names()
    action_dim = len(actuator_names)
    default_qpos = np.asarray(backend.get_keyframe_qpos("stand"), dtype=get_global_dtype())
    initial_qvel = np.asarray(backend.get_init_qvel(), dtype=get_global_dtype())
    if default_qpos.shape != (7 + action_dim,) or initial_qvel.shape != (6 + action_dim,):
        raise G1ManagedReferenceError(
            "G1 managed reference requires one floating root and one coordinate per actuator"
        )
    if not np.isfinite(default_qpos).all() or not np.isfinite(initial_qvel).all():
        raise G1ManagedReferenceError("G1 initial state must be finite")
    command_low = np.asarray(cfg.commands.vel_limit[0], dtype=get_global_dtype())
    command_high = np.asarray(cfg.commands.vel_limit[1], dtype=get_global_dtype())
    if command_low.shape != (3,) or command_high.shape != (3,):
        raise G1ManagedReferenceError("G1 command limits must have shape (2, 3)")
    if (
        not np.isfinite(command_low).all()
        or not np.isfinite(command_high).all()
        or np.any(command_high < command_low)
    ):
        raise G1ManagedReferenceError("G1 command maximum must be >= minimum")
    standing_probability = float(cfg.commands.rel_standing_envs)
    if not np.isfinite(standing_probability) or standing_probability < 0.0:
        raise G1ManagedReferenceError("G1 standing probability must be finite and non-negative")
    pose_weights = np.asarray(reward.pose_weights, dtype=get_global_dtype())
    if pose_weights.shape != (action_dim,) or not np.isfinite(pose_weights).all():
        raise G1ManagedReferenceError(
            f"G1 pose_weights must be finite with shape ({action_dim},), got {pose_weights.shape}"
        )
    finite_positive = {
        "tracking_sigma": reward.tracking_sigma,
        "feet_phase_tracking_sigma": reward.feet_phase_tracking_sigma,
    }
    for name, value in finite_positive.items():
        if not np.isfinite(float(value)) or float(value) <= 0.0:
            raise G1ManagedReferenceError(f"G1 {name} must be finite and positive")
    finite_non_negative = {
        "reset_base_qvel_limit": cfg.reset_base_qvel_limit,
        "feet_phase_swing_height": reward.feet_phase_swing_height,
        "min_forward_speed_for_gait_reward": reward.min_forward_speed_for_gait_reward,
        "close_feet_threshold": reward.close_feet_threshold,
        "noise_config.level": cfg.noise_config.level,
    }
    for name, value in finite_non_negative.items():
        if not np.isfinite(float(value)) or float(value) < 0.0:
            raise G1ManagedReferenceError(f"G1 {name} must be finite and non-negative")
    finite_values = {
        "gait_frequency": reward.gait_frequency,
        "base_height_target": reward.base_height_target,
        "min_base_height": reward.min_base_height,
        "max_tilt_deg": reward.max_tilt_deg,
        "noise_config.scale_joint_angle": cfg.noise_config.scale_joint_angle,
        "noise_config.scale_joint_vel": cfg.noise_config.scale_joint_vel,
        "noise_config.scale_gyro": cfg.noise_config.scale_gyro,
        "noise_config.scale_gravity": cfg.noise_config.scale_gravity,
    }
    for name, value in finite_values.items():
        if not np.isfinite(float(value)):
            raise G1ManagedReferenceError(f"G1 {name} must be finite")
    return _G1KernelConfig(
        action_scale=np.asarray(_action_scale(cfg, action_dim), dtype=get_global_dtype()),
        default_angles=np.asarray(default_qpos[-action_dim:], dtype=get_global_dtype()),
        initial_qpos=np.asarray(default_qpos, dtype=get_global_dtype()),
        initial_qvel=np.asarray(initial_qvel, dtype=get_global_dtype()),
        reward_terms=tuple((name, float(scale)) for name, scale in reward.scales.items()),
        ctrl_dt=float(cfg.ctrl_dt),
        tracking_sigma=float(reward.tracking_sigma),
        gait_phase_delta=float(2.0 * math.pi * reward.gait_frequency * cfg.ctrl_dt),
        gait_phase_init_mode=cfg.gait_phase_init_mode,
        reset_base_qvel_limit=float(cfg.reset_base_qvel_limit),
        command_low=command_low,
        command_high=command_high,
        standing_probability=min(standing_probability, 1.0),
        base_height_target=float(reward.base_height_target),
        min_base_height=float(reward.min_base_height),
        max_tilt_rad=float(np.deg2rad(reward.max_tilt_deg)),
        feet_phase_swing_height=float(reward.feet_phase_swing_height),
        feet_phase_tracking_sigma=float(reward.feet_phase_tracking_sigma),
        min_forward_speed_for_gait_reward=float(reward.min_forward_speed_for_gait_reward),
        close_feet_threshold=float(reward.close_feet_threshold),
        pose_weights=pose_weights,
        upper_body_pose_weights=build_upper_body_pose_weights(pose_weights.tolist()),
        walk_observation_profile=_walk_observation_profile(reward, cfg),
        observation_noise_level=float(cfg.noise_config.level),
        observation_noise_scale_joint_angle=float(cfg.noise_config.scale_joint_angle),
        observation_noise_scale_joint_vel=float(cfg.noise_config.scale_joint_vel),
        observation_noise_scale_gyro=float(cfg.noise_config.scale_gyro),
        observation_noise_scale_gravity=float(cfg.noise_config.scale_gravity),
        reset_seed=reset_seed,
        observation_noise_seed=observation_noise_seed,
    )


def create_g1_managed_reference_runtime(
    *,
    backend: SimBackend,
    cfg: G1WalkEnvCfg,
    reset_seed: int = 0,
    observation_noise_seed: int | None = None,
    autoreset: bool = True,
    record_lifecycle: bool = False,
) -> ManagedReferenceRuntime:
    """Create the cold-bound G1 host reference runtime.

    ``backend.materialize`` is intentionally part of this factory's cold path;
    the runtime itself only binds typed state/mutation plans and executes the
    public batch lifecycle.
    """

    kernel = G1ManagedReferenceKernel(
        _kernel_config(
            backend=backend,
            cfg=cfg,
            reset_seed=reset_seed,
            observation_noise_seed=observation_noise_seed,
        )
    )
    plan = compile_g1_managed_reference_task(backend=backend, cfg=cfg)
    backend.materialize()
    return ManagedReferenceRuntime(
        backend=backend,
        plan=plan,
        kernel=kernel,
        max_episode_steps=cfg.max_episode_steps,
        autoreset=autoreset,
        record_lifecycle=record_lifecycle,
    )


__all__ = [
    "G1_MANAGED_REFERENCE_EXECUTOR_KEY",
    "G1ManagedReferenceError",
    "G1ManagedReferenceKernel",
    "compile_g1_managed_reference_task",
    "create_g1_managed_reference_runtime",
]
