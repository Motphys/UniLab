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
    BufferLifetime,
    BufferView,
    ControlSpec,
    ExecutionProfile,
    MutationValueBatch,
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
from .managed_reward_terms import (
    NUMPY_G1_REWARD_MATH,
    G1RewardContext,
    G1RewardScratch,
    bind_g1_reward_terms,
)
from .managed_schema import (
    G1_ACTOR_OBSERVATION_WIDTH,
    G1_CRITIC_OBSERVATION_WIDTH,
    G1_RESET_TERM,
    G1_ROOT_RESET_SPECS,
    G1_STATE_KEYS,
    G1KernelConfig,
    G1ResetSample,
    G1StateViews,
    build_g1_kernel_config,
    g1_action_scale,
    g1_reset_templates,
    g1_selectors,
    g1_state_requirements,
    manager_buffer_contract,
    reset_suffix_for_dof,
    reset_term_key,
    validate_g1_managed_profile,
)

G1_MANAGED_REFERENCE_EXECUTOR_KEY = "reference.numpy.g1-walk-flat.v1"
"""The explicit host-only executor identity used in the compiled plan."""


class G1ManagedReferenceError(ManagerContractError):
    """Raised when a requested G1 managed-reference profile is unsupported."""


def compile_g1_managed_reference_task(
    *, backend: SimBackend, cfg: G1WalkEnvCfg
) -> CompiledTaskPlan:
    """Compile the static G1 host-reference plan through public backend APIs.

    This function only executes cold-path metadata queries.  Its returned plan
    contains bound IDs and mutation selector metadata, but no backend object.
    """

    if not isinstance(backend, SimBackend):
        raise G1ManagedReferenceError("G1 managed reference requires a SimBackend")
    reward = validate_g1_managed_profile(
        cfg,
        profile_name="managed reference",
        error_type=G1ManagedReferenceError,
    )
    actuator_names = backend.get_actuator_names()
    if not actuator_names:
        raise G1ManagedReferenceError("G1 managed reference requires named actuators")
    if len(set(actuator_names)) != len(actuator_names):
        raise G1ManagedReferenceError("G1 managed reference actuator names must be unique")
    action_dim = len(actuator_names)
    g1_action_scale(cfg, action_dim, error_type=G1ManagedReferenceError)

    root, dofs, reset_position, reset_velocity = g1_selectors(actuator_names)
    state_requirements = g1_state_requirements(root=root, dofs=dofs, action_dim=action_dim)
    reset_templates = g1_reset_templates(
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
            output=TensorSpec((G1_ACTOR_OBSERVATION_WIDTH,), np.dtype(get_global_dtype()).name),
        )
    )
    registry.register(
        TermDefinition(
            key="g1.reference.critic_observation",
            version="1",
            phase=TermPhase.TERMINAL_OBSERVATION,
            role=TermRole.OBSERVATION,
            state_requirements=state_requirements,
            output=TensorSpec((G1_CRITIC_OBSERVATION_WIDTH,), np.dtype(get_global_dtype()).name),
        )
    )
    task = TaskSpec.create(
        key="g1_walk_flat.managed_reference",
        terms=(
            TermInvocation.create(key=G1_RESET_TERM, definition_key="g1.reference.reset"),
            TermInvocation.create(
                key="g1_termination",
                definition_key="g1.reference.termination",
                dependencies=(G1_RESET_TERM,),
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
            buffer=manager_buffer_contract(
                row_shape=(action_dim,), lifetime=BufferLifetime.UNTIL_STEP_COMPLETE
            ),
            physics_substeps_per_control=cfg.sim_substeps,
        ),
        execution_profile=ExecutionProfile.HOST_NUMPY,
        executor_key=G1_MANAGED_REFERENCE_EXECUTOR_KEY,
        policy=PolicySpec(
            ("obs", "critic"),
            tuple(
                float(value)
                for value in g1_action_scale(cfg, action_dim, error_type=G1ManagedReferenceError)
            ),
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
    if plan.policy_abi.observation_groups[0].width != G1_ACTOR_OBSERVATION_WIDTH or (
        plan.policy_abi.observation_groups[1].width != G1_CRITIC_OBSERVATION_WIDTH
    ):
        raise G1ManagedReferenceError("compiled G1 policy ABI has an unexpected observation width")
    if reward is not cfg.reward_config:  # pragma: no cover - type-narrowing invariant
        raise G1ManagedReferenceError("G1 reward configuration changed during compilation")
    return plan


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
    reward_value: np.ndarray
    weighted_reward: np.ndarray
    reward_scratch: G1RewardScratch


class G1ManagedReferenceKernel:
    """Pure host-Numpy G1 task kernel over the bound typed state plan."""

    executor_key = G1_MANAGED_REFERENCE_EXECUTOR_KEY

    def __init__(self, config: G1KernelConfig) -> None:
        if not isinstance(config, G1KernelConfig):
            raise G1ManagedReferenceError("G1 managed reference kernel requires frozen config")
        self._config = config
        try:
            self._reward_terms = bind_g1_reward_terms(config.reward_terms)
        except ValueError as exc:  # pragma: no cover - cold profile validation owns this.
            raise G1ManagedReferenceError(str(exc)) from exc
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
        missing_state = tuple(key for key in G1_STATE_KEYS if key not in state_index_by_key)
        if missing_state:
            raise G1ManagedReferenceError(
                "G1 managed reference plan is missing state fields: " + ", ".join(missing_state)
            )
        self._state_indices = tuple(state_index_by_key[key] for key in G1_STATE_KEYS)
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
        root_keys = tuple(reset_term_key(suffix=item[0]) for item in G1_ROOT_RESET_SPECS)
        position_keys = tuple(
            reset_term_key(suffix=reset_suffix_for_dof(kind="position", index=index))
            for index in range(self._action_dim)
        )
        velocity_keys = tuple(
            reset_term_key(suffix=reset_suffix_for_dof(kind="velocity", index=index))
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

    def _state_views(self, state: StateBatch) -> G1StateViews:
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
        for key, index in zip(G1_STATE_KEYS, self._require_state_indices(), strict=True):
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
        return G1StateViews(
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
            reward_value=np.empty((num_envs,), dtype=dtype),
            weighted_reward=np.empty((num_envs,), dtype=dtype),
            reward_scratch=G1RewardScratch(
                bool_a=np.empty((num_envs,), dtype=bool),
                bool_b=np.empty((num_envs,), dtype=bool),
                scalar_b=np.empty((num_envs,), dtype=dtype),
                scalar_c=np.empty((num_envs,), dtype=dtype),
                scalar_d=np.empty((num_envs,), dtype=dtype),
                vector2=np.empty((num_envs, 2), dtype=dtype),
                action=np.empty((num_envs, self._action_dim), dtype=dtype),
                left_height=np.empty((num_envs,), dtype=dtype),
                right_height=np.empty((num_envs,), dtype=dtype),
            ),
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
        context = G1RewardContext(
            commands=task.commands,
            current_actions=task.current_actions,
            last_actions=task.last_actions,
            gait_phase=task.gait_phase,
            root_position=views.root_position,
            dof_position=views.dof_position,
            linear_velocity=views.pelvis_local_linear_velocity,
            gyro=views.torso_gyro,
            upvector=views.torso_upvector,
            left_foot_position=views.left_foot_position,
            right_foot_position=views.right_foot_position,
            default_angles=self._config.default_angles,
            pose_weights=self._config.pose_weights,
            upper_body_pose_weights=self._config.upper_body_pose_weights,
            tracking_sigma=self._config.tracking_sigma,
            base_height_target=self._config.base_height_target,
            feet_phase_swing_height=self._config.feet_phase_swing_height,
            feet_phase_tracking_sigma=self._config.feet_phase_tracking_sigma,
            min_forward_speed_for_gait_reward=(self._config.min_forward_speed_for_gait_reward),
            close_feet_threshold=self._config.close_feet_threshold,
            scratch=task.reward_scratch,
        )
        reward_out.fill(0.0)
        for index, (term, scale) in enumerate(self._reward_terms):
            if scale == 0.0:
                task.reward_means[index] = 0.0
                continue
            term.evaluate(NUMPY_G1_REWARD_MATH, context, out=task.reward_value)
            np.multiply(task.reward_value, scale, out=task.weighted_reward)
            np.add(reward_out, task.weighted_reward, out=reward_out)
            task.reward_means[index] = np.mean(task.weighted_reward)
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
        if actor_all.shape != (
            binding.num_envs,
            G1_ACTOR_OBSERVATION_WIDTH,
        ) or critic_all.shape != (
            binding.num_envs,
            G1_CRITIC_OBSERVATION_WIDTH,
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
        if cursor != G1_ACTOR_OBSERVATION_WIDTH:  # pragma: no cover - static layout assertion
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
        if cursor != G1_CRITIC_OBSERVATION_WIDTH:  # pragma: no cover - static layout assertion
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
            kernel_state=G1ResetSample(rows=rows),
        )

    def complete_reset(
        self,
        *,
        request: ManagedResetRequest,
        state: StateBatch,
        task_state: object,
    ) -> None:
        task = self._require_task_state(task_state)
        if not isinstance(request.kernel_state, G1ResetSample):
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
        build_g1_kernel_config(
            backend=backend,
            cfg=cfg,
            reset_seed=reset_seed,
            observation_noise_seed=observation_noise_seed,
            profile_name="managed reference",
            error_type=G1ManagedReferenceError,
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
