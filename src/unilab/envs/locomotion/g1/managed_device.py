"""Torch CUDA executor for the compiled G1 flat-walk manager task.

The executor consumes only immutable plan metadata, public device
``StateBatch`` views, and task-owned CUDA buffers.  It never retains a backend,
environment, model, asset, selector, or registry.  Backend construction and
selector resolution remain in the factory's cold path; action scaling,
reward, termination, observations, task state, and reset staging execute on
the runtime-selected CUDA stream.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cached_property
from typing import Any

import numpy as np
import torch

from unilab.base.backend import (
    BoundMutationPlan,
    BufferContract,
    BufferLayout,
    BufferLifetime,
    BufferMutability,
    BufferOwner,
    BufferPlacement,
    ControlSpec,
    DeviceTensorView,
    ExecutionProfile,
    MutationBaseline,
    MutationCommitPhase,
    MutationFieldKind,
    MutationOperation,
    MutationPersistence,
    MutationRecomputeLevel,
    MutationTargetKind,
    MutationTrigger,
    StateBatch,
)
from unilab.base.backend.base import SimBackend
from unilab.dr.keyed_rng import RandomCorrelation, RandomDistribution
from unilab.dtype_config import get_global_dtype
from unilab.manager import (
    BackendEntityResolver,
    DeviceManagedRuntime,
    DeviceResetPayload,
    DeviceResetValue,
    DeviceRuntimeBuffer,
    EntityKind,
    EntitySelector,
    ManagedKernelBinding,
    ManagerContractError,
    MutationRandomization,
    MutationTemplate,
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

from .joystick import G1WalkEnvCfg, G1WalkRewardConfig
from .managed_reward_terms import (
    TORCH_G1_REWARD_MATH,
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
    build_g1_kernel_config,
    g1_action_scale,
    g1_selectors,
    g1_state_requirements,
    reset_term_key,
    validate_g1_managed_profile,
)

G1_MANAGED_DEVICE_EXECUTOR_KEY = "device.torch.g1-walk-flat.v1"
_KP_EVENT_TERM = "g1_randomize_kp"
_KD_EVENT_TERM = "g1_randomize_kd"
_ARMATURE_EVENT_TERM = "g1_randomize_dof_armature"
_GRAVCOMP_EVENT_TERM = "g1_randomize_body_gravity_compensation"


class G1ManagedDeviceError(ManagerContractError):
    """Raised when the strict G1 CUDA executor cannot honor a requested profile."""


def _device_manager_buffer(
    *,
    placement: BufferPlacement,
    row_shape: tuple[int, ...],
    lifetime: BufferLifetime,
    owner: BufferOwner = BufferOwner.MANAGER,
) -> BufferContract:
    dtype = np.dtype(get_global_dtype()).name
    if dtype != "float32":
        raise G1ManagedDeviceError("G1 device executor currently requires global float32")
    return BufferContract(
        row_shape=row_shape,
        dtype=dtype,
        layout=BufferLayout.C_CONTIGUOUS,
        placement=placement,
        owner=owner,
        mutability=BufferMutability.READ_ONLY,
        lifetime=lifetime,
        dlpack_exportable=True,
        address_stable=True,
    )


def _device_reset_templates(
    *,
    placement: BufferPlacement,
    root: Any,
    dofs: Any,
) -> tuple[MutationTemplate, ...]:
    templates: list[MutationTemplate] = []
    for suffix, target_key, field_kind, row_shape in G1_ROOT_RESET_SPECS:
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
                value_template=_device_manager_buffer(
                    placement=placement,
                    row_shape=row_shape,
                    lifetime=BufferLifetime.UNTIL_COMMIT,
                ),
            )
        )
    # The bound contract expands this scalar template to one row per selected
    # DoF, preserving selector order while reducing DoF index-copy dispatches
    # from 58 to 2 per reset barrier.
    for key_suffix, target_key, field_kind in (
        ("dof_position", "state.dof.position", MutationFieldKind.POSITION),
        (
            "dof_angular_velocity",
            "state.dof.angular_velocity",
            MutationFieldKind.ANGULAR_VELOCITY,
        ),
    ):
        templates.append(
            MutationTemplate(
                key_suffix=key_suffix,
                target_key=target_key,
                target_kind=MutationTargetKind.SIMULATION_STATE,
                selector=dofs,
                field_kind=field_kind,
                trigger=MutationTrigger.RESET,
                commit_phase=MutationCommitPhase.RESET,
                operation=MutationOperation.SET,
                baseline=MutationBaseline.DEFAULT,
                persistence=MutationPersistence.EPISODE,
                recompute=MutationRecomputeLevel.KINEMATICS,
                value_template=_device_manager_buffer(
                    placement=placement,
                    row_shape=(1,),
                    lifetime=BufferLifetime.UNTIL_COMMIT,
                ),
            )
        )
    return tuple(templates)


def _device_pd_event_template(
    *,
    placement: BufferPlacement,
    actuators: EntitySelector,
    target_key: str,
    field_kind: MutationFieldKind,
    multiplier_range: list[float],
) -> MutationTemplate:
    if len(multiplier_range) != 2:
        raise G1ManagedDeviceError("G1 PD multiplier range must contain two values")
    parameters = (float(multiplier_range[0]), float(multiplier_range[1]))
    return MutationTemplate(
        key_suffix="",
        target_key=target_key,
        target_kind=MutationTargetKind.MODEL_PARAMETER,
        selector=actuators,
        field_kind=field_kind,
        trigger=MutationTrigger.RESET,
        commit_phase=MutationCommitPhase.RESET,
        operation=MutationOperation.SCALE,
        baseline=MutationBaseline.DEFAULT,
        persistence=MutationPersistence.EPISODE,
        recompute=MutationRecomputeLevel.NONE,
        value_template=_device_manager_buffer(
            placement=placement,
            row_shape=(1,),
            lifetime=BufferLifetime.UNTIL_COMMIT,
        ),
        randomization=MutationRandomization(
            distribution=RandomDistribution.UNIFORM,
            parameters=parameters,
            correlation=RandomCorrelation.PER_ENV,
        ),
    )


def _device_armature_event_template(
    *,
    placement: BufferPlacement,
    dofs: EntitySelector,
    multiplier_range: list[float],
) -> MutationTemplate:
    if len(multiplier_range) != 2:
        raise G1ManagedDeviceError("G1 armature multiplier range must contain two values")
    parameters = (float(multiplier_range[0]), float(multiplier_range[1]))
    return MutationTemplate(
        key_suffix="",
        target_key="joint.armature",
        target_kind=MutationTargetKind.MODEL_PARAMETER,
        selector=dofs,
        field_kind=MutationFieldKind.ARMATURE,
        trigger=MutationTrigger.RESET,
        commit_phase=MutationCommitPhase.RESET,
        operation=MutationOperation.SCALE,
        baseline=MutationBaseline.DEFAULT,
        persistence=MutationPersistence.EPISODE,
        recompute=MutationRecomputeLevel.DYNAMICS,
        value_template=_device_manager_buffer(
            placement=placement,
            row_shape=(1,),
            lifetime=BufferLifetime.UNTIL_COMMIT,
        ),
        randomization=MutationRandomization(
            distribution=RandomDistribution.UNIFORM,
            parameters=parameters,
            correlation=RandomCorrelation.PER_ENV,
        ),
    )


def _device_gravity_compensation_event_template(
    *,
    placement: BufferPlacement,
    bodies: EntitySelector,
    value_range: list[float],
) -> MutationTemplate:
    if len(value_range) != 2:
        raise G1ManagedDeviceError("G1 body gravity-compensation range must contain two values")
    parameters = (float(value_range[0]), float(value_range[1]))
    if not all(math.isfinite(value) for value in parameters) or parameters[0] > parameters[1]:
        raise G1ManagedDeviceError("G1 body gravity-compensation range must be finite and ordered")
    return MutationTemplate(
        key_suffix="",
        target_key="body.gravity_compensation",
        target_kind=MutationTargetKind.MODEL_PARAMETER,
        selector=bodies,
        field_kind=MutationFieldKind.GRAVITY_COMPENSATION,
        trigger=MutationTrigger.RESET,
        commit_phase=MutationCommitPhase.RESET,
        operation=MutationOperation.SET,
        baseline=MutationBaseline.DEFAULT,
        persistence=MutationPersistence.EPISODE,
        recompute=MutationRecomputeLevel.KINEMATICS,
        value_template=_device_manager_buffer(
            placement=placement,
            row_shape=(1,),
            lifetime=BufferLifetime.UNTIL_COMMIT,
        ),
        randomization=MutationRandomization(
            distribution=RandomDistribution.UNIFORM,
            parameters=parameters,
            correlation=RandomCorrelation.PER_ENV,
        ),
    )


def _validate_device_profile(cfg: G1WalkEnvCfg) -> G1WalkRewardConfig:
    reward = validate_g1_managed_profile(
        cfg,
        profile_name="device executor",
        error_type=G1ManagedDeviceError,
        allow_pd_randomization=True,
        allow_dof_armature_randomization=True,
        allow_body_gravity_compensation_randomization=True,
    )
    if float(cfg.noise_config.level) != 0.0:
        raise G1ManagedDeviceError(
            "G1 device executor observation noise is not implemented; disable it explicitly"
        )
    return reward


def compile_g1_managed_device_task(
    *,
    backend: SimBackend,
    cfg: G1WalkEnvCfg,
    device_index: int | None = None,
) -> CompiledTaskPlan:
    """Compile the G1 device plan through public cold-path backend metadata."""

    if not isinstance(backend, SimBackend):
        raise G1ManagedDeviceError("G1 device executor requires a SimBackend")
    if backend.backend_type != "mjwarp":
        raise G1ManagedDeviceError(
            "G1 device executor currently requires the independent mjwarp backend"
        )
    reward = _validate_device_profile(cfg)
    if not torch.cuda.is_available():
        raise G1ManagedDeviceError("G1 device executor requires an available CUDA device")
    if device_index is None:
        device_index = int(torch.cuda.current_device())
    if isinstance(device_index, bool) or not isinstance(device_index, int) or device_index < 0:
        raise G1ManagedDeviceError("G1 device executor device_index must be non-negative")
    placement = BufferPlacement.device("cuda", device_index)

    actuator_names = backend.get_actuator_names()
    if not actuator_names or len(set(actuator_names)) != len(actuator_names):
        raise G1ManagedDeviceError("G1 device executor requires unique named actuators")
    action_dim = len(actuator_names)
    g1_action_scale(cfg, action_dim, error_type=G1ManagedDeviceError)
    root, dofs, _, _ = g1_selectors(actuator_names)
    actuators = EntitySelector(
        key="g1.position_actuators",
        entity="g1",
        kind=EntityKind.ACTUATOR,
        expressions=actuator_names,
    )
    gravity_compensation_bodies: EntitySelector | None = None
    if cfg.domain_rand.randomize_body_gravity_compensation:
        body_names = tuple(cfg.domain_rand.body_gravity_compensation_bodies)
        if (
            not body_names
            or any(not isinstance(name, str) or not name.strip() for name in body_names)
            or len(set(body_names)) != len(body_names)
        ):
            raise G1ManagedDeviceError(
                "G1 body gravity-compensation selector requires unique non-empty body names"
            )
        gravity_compensation_bodies = EntitySelector(
            key="g1.gravity_compensation_bodies",
            entity="g1",
            kind=EntityKind.BODY,
            expressions=body_names,
        )
    state_requirements = g1_state_requirements(root=root, dofs=dofs, action_dim=action_dim)
    reset_templates = _device_reset_templates(
        placement=placement,
        root=root,
        dofs=dofs,
    )

    registry = TermRegistry()
    registry.register(
        TermDefinition(
            key="g1.device.reset",
            version="1",
            phase=TermPhase.RESET,
            role=TermRole.EVENT,
            mutation_templates=reset_templates,
        )
    )
    if cfg.domain_rand.randomize_kp:
        registry.register(
            TermDefinition(
                key="g1.device.randomize_kp",
                version="1",
                phase=TermPhase.RESET,
                role=TermRole.EVENT,
                mutation_templates=(
                    _device_pd_event_template(
                        placement=placement,
                        actuators=actuators,
                        target_key="actuator.pd_stiffness",
                        field_kind=MutationFieldKind.STIFFNESS,
                        multiplier_range=cfg.domain_rand.kp_multiplier_range,
                    ),
                ),
            )
        )
    if cfg.domain_rand.randomize_kd:
        registry.register(
            TermDefinition(
                key="g1.device.randomize_kd",
                version="1",
                phase=TermPhase.RESET,
                role=TermRole.EVENT,
                mutation_templates=(
                    _device_pd_event_template(
                        placement=placement,
                        actuators=actuators,
                        target_key="actuator.pd_damping",
                        field_kind=MutationFieldKind.DAMPING,
                        multiplier_range=cfg.domain_rand.kd_multiplier_range,
                    ),
                ),
            )
        )
    if cfg.domain_rand.randomize_dof_armature:
        registry.register(
            TermDefinition(
                key="g1.device.randomize_dof_armature",
                version="1",
                phase=TermPhase.RESET,
                role=TermRole.EVENT,
                mutation_templates=(
                    _device_armature_event_template(
                        placement=placement,
                        dofs=dofs,
                        multiplier_range=cfg.domain_rand.dof_armature_multiplier_range,
                    ),
                ),
            )
        )
    if cfg.domain_rand.randomize_body_gravity_compensation:
        assert gravity_compensation_bodies is not None
        registry.register(
            TermDefinition(
                key="g1.device.randomize_body_gravity_compensation",
                version="1",
                phase=TermPhase.RESET,
                role=TermRole.EVENT,
                mutation_templates=(
                    _device_gravity_compensation_event_template(
                        placement=placement,
                        bodies=gravity_compensation_bodies,
                        value_range=cfg.domain_rand.body_gravity_compensation_range,
                    ),
                ),
            )
        )
    registry.register(
        TermDefinition(
            key="g1.device.termination",
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
            key="g1.device.reward",
            version="1",
            phase=TermPhase.REWARD,
            role=TermRole.REWARD,
            state_requirements=state_requirements,
        )
    )
    registry.register(
        TermDefinition(
            key="g1.device.actor_observation",
            version="1",
            phase=TermPhase.TERMINAL_OBSERVATION,
            role=TermRole.OBSERVATION,
            state_requirements=state_requirements,
            output=TensorSpec((G1_ACTOR_OBSERVATION_WIDTH,), "float32"),
        )
    )
    registry.register(
        TermDefinition(
            key="g1.device.critic_observation",
            version="1",
            phase=TermPhase.TERMINAL_OBSERVATION,
            role=TermRole.OBSERVATION,
            state_requirements=state_requirements,
            output=TensorSpec((G1_CRITIC_OBSERVATION_WIDTH,), "float32"),
        )
    )
    reset_terms = [TermInvocation.create(key=G1_RESET_TERM, definition_key="g1.device.reset")]
    if cfg.domain_rand.randomize_kp:
        reset_terms.append(
            TermInvocation.create(
                key=_KP_EVENT_TERM,
                definition_key="g1.device.randomize_kp",
                dependencies=(G1_RESET_TERM,),
            )
        )
    if cfg.domain_rand.randomize_kd:
        reset_terms.append(
            TermInvocation.create(
                key=_KD_EVENT_TERM,
                definition_key="g1.device.randomize_kd",
                dependencies=(G1_RESET_TERM,),
            )
        )
    if cfg.domain_rand.randomize_dof_armature:
        reset_terms.append(
            TermInvocation.create(
                key=_ARMATURE_EVENT_TERM,
                definition_key="g1.device.randomize_dof_armature",
                dependencies=(G1_RESET_TERM,),
            )
        )
    if cfg.domain_rand.randomize_body_gravity_compensation:
        reset_terms.append(
            TermInvocation.create(
                key=_GRAVCOMP_EVENT_TERM,
                definition_key="g1.device.randomize_body_gravity_compensation",
                dependencies=(G1_RESET_TERM,),
            )
        )
    reset_dependencies = tuple(term.key for term in reset_terms)
    task = TaskSpec.create(
        key="g1_walk_flat.managed_device",
        terms=(
            *reset_terms,
            TermInvocation.create(
                key="g1_termination",
                definition_key="g1.device.termination",
                dependencies=reset_dependencies,
            ),
            TermInvocation.create(
                key="g1_reward",
                definition_key="g1.device.reward",
                dependencies=("g1_termination",),
            ),
            TermInvocation.create(
                key="g1_actor_observation",
                definition_key="g1.device.actor_observation",
                dependencies=("g1_reward",),
                observation_group="obs",
            ),
            TermInvocation.create(
                key="g1_critic_observation",
                definition_key="g1.device.critic_observation",
                dependencies=("g1_reward",),
                observation_group="critic",
            ),
        ),
        control=ControlSpec(
            semantic_key="g1.joint.position_target",
            buffer=_device_manager_buffer(
                placement=placement,
                row_shape=(action_dim,),
                lifetime=BufferLifetime.UNTIL_STEP_COMPLETE,
                owner=BufferOwner.RUNNER,
            ),
            physics_substeps_per_control=cfg.sim_substeps,
        ),
        execution_profile=ExecutionProfile.DEVICE_RESIDENT,
        executor_key=G1_MANAGED_DEVICE_EXECUTOR_KEY,
        policy=PolicySpec(
            ("obs", "critic"),
            tuple(
                float(value)
                for value in g1_action_scale(cfg, action_dim, error_type=G1ManagedDeviceError)
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
            *(("actuator.pd_stiffness",) if cfg.domain_rand.randomize_kp else ()),
            *(("actuator.pd_damping",) if cfg.domain_rand.randomize_kd else ()),
            *(("joint.armature",) if cfg.domain_rand.randomize_dof_armature else ()),
            *(
                ("body.gravity_compensation",)
                if cfg.domain_rand.randomize_body_gravity_compensation
                else ()
            ),
        }
    )
    plan = TaskCompiler(registry).compile(
        task,
        resolver=BackendEntityResolver(backend),
        capabilities=capabilities,
    )
    widths = tuple(group.width for group in plan.policy_abi.observation_groups)
    if widths != (G1_ACTOR_OBSERVATION_WIDTH, G1_CRITIC_OBSERVATION_WIDTH):
        raise G1ManagedDeviceError("compiled G1 device policy ABI has unexpected widths")
    if reward is not cfg.reward_config:  # pragma: no cover - narrowing invariant.
        raise G1ManagedDeviceError("G1 reward configuration changed during device compilation")
    return plan


@dataclass(frozen=True)
class _G1DeviceStateViews:
    dof_angular_velocity: torch.Tensor
    dof_position: torch.Tensor
    root_angular_velocity: torch.Tensor
    root_linear_velocity: torch.Tensor
    root_orientation: torch.Tensor
    root_position: torch.Tensor
    left_foot_position: torch.Tensor
    pelvis_local_linear_velocity: torch.Tensor
    right_foot_position: torch.Tensor
    torso_gyro: torch.Tensor
    torso_upvector: torch.Tensor


@dataclass
class _G1DeviceTaskState:
    commands: torch.Tensor
    current_actions: torch.Tensor
    last_actions: torch.Tensor
    gait_phase: torch.Tensor
    reset_qpos: torch.Tensor
    reset_qvel: torch.Tensor
    reset_commands: torch.Tensor
    reset_gait_phase: torch.Tensor
    reset_values: tuple[DeviceResetValue, ...]
    reset_mask: torch.Tensor
    reset_yaw: torch.Tensor
    reset_bool_scratch: torch.Tensor
    bool_scratch_b: torch.Tensor
    reset_scalar_scratch: torch.Tensor
    scalar_scratch_b: torch.Tensor
    scalar_scratch_c: torch.Tensor
    scalar_scratch_d: torch.Tensor
    vector2_scratch: torch.Tensor
    action_scratch: torch.Tensor
    left_height_scratch: torch.Tensor
    right_height_scratch: torch.Tensor
    zero_actions: torch.Tensor
    generator: torch.Generator

    @cached_property
    def reward_scratch(self) -> G1RewardScratch:
        return G1RewardScratch(
            bool_a=self.reset_bool_scratch,
            bool_b=self.bool_scratch_b,
            scalar_b=self.scalar_scratch_b,
            scalar_c=self.scalar_scratch_c,
            scalar_d=self.scalar_scratch_d,
            vector2=self.vector2_scratch,
            action=self.action_scratch,
            left_height=self.left_height_scratch,
            right_height=self.right_height_scratch,
        )


class G1ManagedDeviceKernel:
    """G1 task math implemented directly with Torch CUDA operations."""

    executor_key = G1_MANAGED_DEVICE_EXECUTOR_KEY

    def __init__(
        self,
        config: G1KernelConfig,
        *,
        expected_plan_fingerprint: str,
        placement: BufferPlacement,
    ) -> None:
        if not isinstance(config, G1KernelConfig):
            raise G1ManagedDeviceError("G1 device kernel requires a frozen G1 config")
        if not isinstance(expected_plan_fingerprint, str) or not expected_plan_fingerprint.strip():
            raise G1ManagedDeviceError("G1 device kernel requires an expected plan fingerprint")
        if placement.device_type != "cuda" or placement.device_index is None:
            raise G1ManagedDeviceError("G1 device kernel requires CUDA placement")
        self._config = config
        try:
            bound_reward_terms = bind_g1_reward_terms(config.reward_terms)
        except ValueError as exc:  # pragma: no cover - cold profile validation owns this.
            raise G1ManagedDeviceError(str(exc)) from exc
        self._expected_plan_fingerprint = expected_plan_fingerprint
        self._device = torch.device(f"cuda:{placement.device_index}")
        self._action_dim = int(config.default_angles.size)
        self._binding: ManagedKernelBinding | None = None
        self._state_indices: tuple[int, ...] | None = None
        self._observation_indices: tuple[int, int] | None = None
        self._mutation_plan: BoundMutationPlan | None = None
        self._deterministic_reset_indices: tuple[int, ...] | None = None
        self._root_reset_offsets: tuple[int, int, int, int] | None = None
        self._position_reset_offset: int | None = None
        self._velocity_reset_offset: int | None = None
        self._action_scale = torch.as_tensor(
            config.action_scale, dtype=torch.float32, device=self._device
        )
        self._default_angles = torch.as_tensor(
            config.default_angles, dtype=torch.float32, device=self._device
        )
        self._initial_qpos = torch.as_tensor(
            config.initial_qpos, dtype=torch.float32, device=self._device
        )
        self._initial_qvel = torch.as_tensor(
            config.initial_qvel, dtype=torch.float32, device=self._device
        )
        self._initial_quaternion = tuple(float(value) for value in config.initial_qpos[3:7])
        self._command_low = torch.as_tensor(
            config.command_low, dtype=torch.float32, device=self._device
        )
        self._command_span = torch.as_tensor(
            config.command_high - config.command_low,
            dtype=torch.float32,
            device=self._device,
        )
        self._pose_weights = torch.as_tensor(
            config.pose_weights, dtype=torch.float32, device=self._device
        )
        self._upper_body_pose_weights = torch.as_tensor(
            config.upper_body_pose_weights, dtype=torch.float32, device=self._device
        )
        self._reward_terms = tuple(
            (term, float(scale)) for term, scale in bound_reward_terms if scale != 0.0
        )
        self.metric_keys = tuple(f"reward/{term.key}" for term, _ in self._reward_terms)

    def bind(self, *, binding: ManagedKernelBinding) -> None:
        if self._binding is not None:
            raise G1ManagedDeviceError("G1 device kernel may only bind once")
        if not isinstance(binding, ManagedKernelBinding):
            raise G1ManagedDeviceError("G1 device kernel requires ManagedKernelBinding")
        if binding.task_fingerprint != self._expected_plan_fingerprint:
            raise G1ManagedDeviceError("G1 device kernel received a stale compiled plan")
        if binding.execution_profile is not ExecutionProfile.DEVICE_RESIDENT:
            raise G1ManagedDeviceError("G1 device kernel only supports device_resident plans")
        if binding.dtype != "float32":
            raise G1ManagedDeviceError("G1 device kernel requires float32 binding")
        state_indices = dict(binding.state_field_indices)
        missing_state = tuple(key for key in G1_STATE_KEYS if key not in state_indices)
        if missing_state:
            raise G1ManagedDeviceError(
                "G1 device plan is missing state fields: " + ", ".join(missing_state)
            )
        observations = dict(binding.observation_buffer_indices)
        try:
            observation_indices = (observations["obs"], observations["critic"])
        except KeyError as exc:
            raise G1ManagedDeviceError("G1 device plan requires obs and critic channels") from exc
        mutation_plan = binding.mutation_plan
        if mutation_plan is None:
            raise G1ManagedDeviceError("G1 device plan requires typed reset mutations")
        mutation_indices = {spec.term_key: index for index, spec in enumerate(mutation_plan.specs)}
        root_keys = tuple(reset_term_key(suffix=item[0]) for item in G1_ROOT_RESET_SPECS)
        position_key = reset_term_key(suffix="dof_position")
        velocity_key = reset_term_key(suffix="dof_angular_velocity")
        missing_mutations = tuple(
            key for key in (*root_keys, position_key, velocity_key) if key not in mutation_indices
        )
        if missing_mutations:
            raise G1ManagedDeviceError(
                "G1 device plan is missing reset mutations: " + ", ".join(missing_mutations)
            )
        event_indices = binding.event_mutation_indices
        event_keys = tuple(mutation_plan.specs[index].term_key for index in event_indices)
        if any(
            key
            not in {
                _KP_EVENT_TERM,
                _KD_EVENT_TERM,
                _ARMATURE_EVENT_TERM,
                _GRAVCOMP_EVENT_TERM,
            }
            for key in event_keys
        ):
            raise G1ManagedDeviceError("G1 device plan contains an unsupported random Event")
        deterministic_indices = tuple(
            index
            for index in range(len(mutation_plan.specs))
            if index not in frozenset(event_indices)
        )
        required_indices = tuple(
            sorted(mutation_indices[key] for key in (*root_keys, position_key, velocity_key))
        )
        if deterministic_indices != required_indices:
            raise G1ManagedDeviceError(
                "G1 device deterministic reset plan has unexpected mutation fields"
            )
        offsets = {index: offset for offset, index in enumerate(deterministic_indices)}
        self._binding = binding
        self._state_indices = tuple(state_indices[key] for key in G1_STATE_KEYS)
        self._observation_indices = observation_indices
        self._mutation_plan = mutation_plan
        self._deterministic_reset_indices = deterministic_indices
        self._root_reset_offsets = tuple(offsets[mutation_indices[key]] for key in root_keys)  # type: ignore[assignment]
        self._position_reset_offset = offsets[mutation_indices[position_key]]
        self._velocity_reset_offset = offsets[mutation_indices[velocity_key]]

    def _require_binding(self) -> ManagedKernelBinding:
        if self._binding is None:
            raise G1ManagedDeviceError("G1 device kernel has not been bound")
        return self._binding

    def _require_reset_offsets(
        self,
    ) -> tuple[tuple[int, int, int, int], int, int]:
        if (
            self._root_reset_offsets is None
            or self._position_reset_offset is None
            or self._velocity_reset_offset is None
        ):
            raise G1ManagedDeviceError("G1 device reset offsets are not bound")
        return (
            self._root_reset_offsets,
            self._position_reset_offset,
            self._velocity_reset_offset,
        )

    @staticmethod
    def _task(task_state: object) -> _G1DeviceTaskState:
        if not isinstance(task_state, _G1DeviceTaskState):
            raise G1ManagedDeviceError("G1 device kernel received foreign task state")
        return task_state

    def create_task_state(
        self,
        *,
        num_envs: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> object:
        binding = self._require_binding()
        if num_envs != binding.num_envs or dtype is not torch.float32 or device != self._device:
            raise G1ManagedDeviceError("G1 device task-state placement differs from its binding")
        mutation_plan = self._mutation_plan
        deterministic_indices = self._deterministic_reset_indices
        if mutation_plan is None or deterministic_indices is None:  # pragma: no cover
            raise G1ManagedDeviceError("G1 device task state lacks mutation plan")
        reset_values = tuple(
            DeviceResetValue(
                field_index=field_index,
                tensor=torch.empty(
                    (num_envs, *mutation_plan.specs[field_index].value_buffer.row_shape),
                    dtype=dtype,
                    device=device,
                ),
            )
            for field_index in deterministic_indices
        )
        generator = torch.Generator(device=device)
        generator.manual_seed(self._config.reset_seed)
        return _G1DeviceTaskState(
            commands=torch.zeros((num_envs, 3), dtype=dtype, device=device),
            current_actions=torch.zeros((num_envs, self._action_dim), dtype=dtype, device=device),
            last_actions=torch.zeros((num_envs, self._action_dim), dtype=dtype, device=device),
            gait_phase=torch.zeros((num_envs, 2), dtype=dtype, device=device),
            reset_qpos=torch.empty((num_envs, 7 + self._action_dim), dtype=dtype, device=device),
            reset_qvel=torch.empty((num_envs, 6 + self._action_dim), dtype=dtype, device=device),
            reset_commands=torch.empty((num_envs, 3), dtype=dtype, device=device),
            reset_gait_phase=torch.empty((num_envs, 2), dtype=dtype, device=device),
            reset_values=reset_values,
            reset_mask=torch.empty((num_envs,), dtype=torch.bool, device=device),
            reset_yaw=torch.empty((num_envs,), dtype=dtype, device=device),
            reset_bool_scratch=torch.empty((num_envs,), dtype=torch.bool, device=device),
            bool_scratch_b=torch.empty((num_envs,), dtype=torch.bool, device=device),
            reset_scalar_scratch=torch.empty((num_envs,), dtype=dtype, device=device),
            scalar_scratch_b=torch.empty((num_envs,), dtype=dtype, device=device),
            scalar_scratch_c=torch.empty((num_envs,), dtype=dtype, device=device),
            scalar_scratch_d=torch.empty((num_envs,), dtype=dtype, device=device),
            vector2_scratch=torch.empty((num_envs, 2), dtype=dtype, device=device),
            action_scratch=torch.empty((num_envs, self._action_dim), dtype=dtype, device=device),
            left_height_scratch=torch.empty((num_envs,), dtype=dtype, device=device),
            right_height_scratch=torch.empty((num_envs,), dtype=dtype, device=device),
            zero_actions=torch.zeros((num_envs, self._action_dim), dtype=dtype, device=device),
            generator=generator,
        )

    def device_runtime_buffers(self, *, task_state: object) -> tuple[DeviceRuntimeBuffer, ...]:
        """Register all executor-owned CUDA storage for opt-in warm auditing."""

        task = self._task(task_state)
        candidates: list[tuple[str, torch.Tensor]] = [
            ("kernel.action_scale", self._action_scale),
            ("kernel.default_angles", self._default_angles),
            ("kernel.initial_qpos", self._initial_qpos),
            ("kernel.initial_qvel", self._initial_qvel),
            ("kernel.command_low", self._command_low),
            ("kernel.command_span", self._command_span),
            ("kernel.pose_weights", self._pose_weights),
            ("kernel.upper_body_pose_weights", self._upper_body_pose_weights),
            ("task.commands", task.commands),
            ("task.current_actions", task.current_actions),
            ("task.last_actions", task.last_actions),
            ("task.gait_phase", task.gait_phase),
            ("task.reset_qpos", task.reset_qpos),
            ("task.reset_qvel", task.reset_qvel),
            ("task.reset_commands", task.reset_commands),
            ("task.reset_gait_phase", task.reset_gait_phase),
            ("task.reset_mask", task.reset_mask),
            ("task.reset_yaw", task.reset_yaw),
            ("task.bool_scratch_a", task.reset_bool_scratch),
            ("task.bool_scratch_b", task.bool_scratch_b),
            ("task.scalar_scratch_a", task.reset_scalar_scratch),
            ("task.scalar_scratch_b", task.scalar_scratch_b),
            ("task.scalar_scratch_c", task.scalar_scratch_c),
            ("task.scalar_scratch_d", task.scalar_scratch_d),
            ("task.vector2_scratch", task.vector2_scratch),
            ("task.action_scratch", task.action_scratch),
            ("task.left_height_scratch", task.left_height_scratch),
            ("task.right_height_scratch", task.right_height_scratch),
            ("task.zero_actions", task.zero_actions),
        ]
        candidates.extend(
            (f"task.reset_values.{value.field_index}", value.tensor) for value in task.reset_values
        )
        return tuple(DeviceRuntimeBuffer(name=name, tensor=tensor) for name, tensor in candidates)

    def _state_views(self, state: StateBatch) -> _G1DeviceStateViews:
        state.assert_valid()
        if self._state_indices is None:
            raise G1ManagedDeviceError("G1 device state indices are not bound")
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
        values: list[torch.Tensor] = []
        for key, field_index, row_shape in zip(
            G1_STATE_KEYS, self._state_indices, expected_shapes, strict=True
        ):
            handle = state.buffer_at(field_index).handle
            if not isinstance(handle, DeviceTensorView):
                raise G1ManagedDeviceError(f"G1 device state {key} is not DeviceTensorView")
            tensor = handle.torch()
            expected = (state.rows.count, *row_shape)
            if (
                tuple(tensor.shape) != expected
                or tensor.dtype is not torch.float32
                or tensor.device != self._device
                or not tensor.is_contiguous()
            ):
                raise G1ManagedDeviceError(
                    f"G1 device state {key} differs from its cold-bound tensor contract"
                )
            values.append(tensor)
        return _G1DeviceStateViews(*values)

    def apply_action(
        self,
        *,
        actions: torch.Tensor,
        task_state: object,
        control_out: torch.Tensor,
    ) -> None:
        task = self._task(task_state)
        expected = (self._require_binding().num_envs, self._action_dim)
        if (
            tuple(actions.shape) != expected
            or tuple(control_out.shape) != expected
            or actions.dtype is not torch.float32
            or control_out.dtype is not torch.float32
            or actions.device != self._device
            or control_out.device != self._device
        ):
            raise G1ManagedDeviceError("G1 device action/control tensors are incompatible")
        task.last_actions.copy_(task.current_actions, non_blocking=True)
        task.current_actions.copy_(actions, non_blocking=True)
        task.gait_phase.add_(self._config.gait_phase_delta)
        task.gait_phase.remainder_(2.0 * math.pi)
        torch.mul(actions, self._action_scale, out=control_out)
        control_out.add_(self._default_angles)

    def _reward_context(
        self,
        *,
        views: _G1DeviceStateViews,
        task: _G1DeviceTaskState,
    ) -> G1RewardContext:
        return G1RewardContext(
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
            default_angles=self._default_angles,
            pose_weights=self._pose_weights,
            upper_body_pose_weights=self._upper_body_pose_weights,
            tracking_sigma=self._config.tracking_sigma,
            base_height_target=self._config.base_height_target,
            feet_phase_swing_height=self._config.feet_phase_swing_height,
            feet_phase_tracking_sigma=self._config.feet_phase_tracking_sigma,
            min_forward_speed_for_gait_reward=(self._config.min_forward_speed_for_gait_reward),
            close_feet_threshold=self._config.close_feet_threshold,
            scratch=task.reward_scratch,
        )

    def _write_observations(
        self,
        *,
        views: _G1DeviceStateViews,
        task: _G1DeviceTaskState,
        observation_buffers: tuple[torch.Tensor, ...],
    ) -> None:
        if self._observation_indices is None:
            raise G1ManagedDeviceError("G1 device observation indices are not bound")
        try:
            actor = observation_buffers[self._observation_indices[0]]
            critic = observation_buffers[self._observation_indices[1]]
        except IndexError as exc:
            raise G1ManagedDeviceError("G1 device observation buffers are incomplete") from exc
        expected_actor = (self._require_binding().num_envs, G1_ACTOR_OBSERVATION_WIDTH)
        expected_critic = (self._require_binding().num_envs, G1_CRITIC_OBSERVATION_WIDTH)
        if tuple(actor.shape) != expected_actor or tuple(critic.shape) != expected_critic:
            raise G1ManagedDeviceError("G1 device observation buffer widths are incompatible")
        gyro_scale = 0.25 if self._config.walk_observation_profile else 1.0
        dof_velocity_scale = 0.05 if self._config.walk_observation_profile else 1.0
        linear_velocity_scale = 2.0 if self._config.walk_observation_profile else 1.0

        cursor = 0
        torch.mul(views.torso_gyro, gyro_scale, out=actor[:, cursor : cursor + 3])
        cursor += 3
        torch.neg(views.torso_upvector, out=actor[:, cursor : cursor + 3])
        cursor += 3
        torch.sub(
            views.dof_position,
            self._default_angles,
            out=actor[:, cursor : cursor + self._action_dim],
        )
        cursor += self._action_dim
        torch.mul(
            views.dof_angular_velocity,
            dof_velocity_scale,
            out=actor[:, cursor : cursor + self._action_dim],
        )
        cursor += self._action_dim
        actor[:, cursor : cursor + self._action_dim].copy_(task.current_actions, non_blocking=True)
        cursor += self._action_dim
        actor[:, cursor : cursor + 3].copy_(task.commands, non_blocking=True)
        cursor += 3
        actor[:, cursor : cursor + 2].copy_(task.gait_phase, non_blocking=True)
        cursor += 2
        if cursor != G1_ACTOR_OBSERVATION_WIDTH:  # pragma: no cover - static layout.
            raise G1ManagedDeviceError("G1 device actor observation layout is inconsistent")

        cursor = 0
        torch.mul(views.torso_gyro, gyro_scale, out=critic[:, cursor : cursor + 3])
        cursor += 3
        torch.neg(views.torso_upvector, out=critic[:, cursor : cursor + 3])
        cursor += 3
        torch.sub(
            views.dof_position,
            self._default_angles,
            out=critic[:, cursor : cursor + self._action_dim],
        )
        cursor += self._action_dim
        torch.mul(
            views.dof_angular_velocity,
            dof_velocity_scale,
            out=critic[:, cursor : cursor + self._action_dim],
        )
        cursor += self._action_dim
        critic[:, cursor : cursor + self._action_dim].copy_(task.current_actions, non_blocking=True)
        cursor += self._action_dim
        critic[:, cursor : cursor + 3].copy_(task.commands, non_blocking=True)
        cursor += 3
        critic[:, cursor : cursor + 2].copy_(task.gait_phase, non_blocking=True)
        cursor += 2
        torch.mul(
            views.pelvis_local_linear_velocity,
            linear_velocity_scale,
            out=critic[:, cursor : cursor + 3],
        )
        cursor += 3
        if cursor != G1_CRITIC_OBSERVATION_WIDTH:  # pragma: no cover - static layout.
            raise G1ManagedDeviceError("G1 device critic observation layout is inconsistent")

    def evaluate_terminal(
        self,
        *,
        state: StateBatch,
        task_state: object,
        reward_out: torch.Tensor,
        metric_buffers: tuple[torch.Tensor, ...],
        terminated_out: torch.Tensor,
        terminal_observation_buffers: tuple[torch.Tensor, ...],
    ) -> None:
        task = self._task(task_state)
        views = self._state_views(state)
        expected_shape = (self._require_binding().num_envs,)
        if len(metric_buffers) != len(self.metric_keys) or any(
            tuple(metric.shape) != expected_shape
            or metric.dtype is not torch.float32
            or metric.device != self._device
            or not metric.is_contiguous()
            for metric in metric_buffers
        ):
            raise G1ManagedDeviceError(
                "G1 device metric buffers differ from the cold-bound metric layout"
            )
        torch.clamp(
            views.torso_upvector[:, 2],
            -1.0,
            1.0,
            out=task.reset_scalar_scratch,
        )
        torch.acos(task.reset_scalar_scratch, out=task.reset_scalar_scratch)
        torch.gt(
            task.reset_scalar_scratch,
            self._config.max_tilt_rad,
            out=task.reset_bool_scratch,
        )
        torch.lt(
            views.root_position[:, 2],
            self._config.min_base_height,
            out=task.bool_scratch_b,
        )
        torch.logical_or(
            task.reset_bool_scratch,
            task.bool_scratch_b,
            out=terminated_out,
        )
        reward_context = self._reward_context(views=views, task=task)
        reward_out.zero_()
        for (term, scale), metric in zip(self._reward_terms, metric_buffers, strict=True):
            term.evaluate(TORCH_G1_REWARD_MATH, reward_context, out=metric)
            metric.mul_(scale)
            reward_out.add_(metric)
        reward_out.mul_(self._config.ctrl_dt)
        self._write_observations(
            views=views,
            task=task,
            observation_buffers=terminal_observation_buffers,
        )

    def _sample_reset(self, task: _G1DeviceTaskState) -> None:
        qpos = task.reset_qpos
        qvel = task.reset_qvel
        qpos.copy_(self._initial_qpos.expand_as(qpos), non_blocking=True)
        qvel.copy_(self._initial_qvel.expand_as(qvel), non_blocking=True)
        qpos[:, :2].uniform_(-0.5, 0.5, generator=task.generator)
        qpos[:, :2].add_(self._initial_qpos[:2])
        task.reset_yaw.uniform_(-math.pi, math.pi, generator=task.generator)
        torch.mul(task.reset_yaw, 0.5, out=task.reset_scalar_scratch)
        torch.cos(task.reset_scalar_scratch, out=task.scalar_scratch_b)
        torch.sin(task.reset_scalar_scratch, out=task.scalar_scratch_c)
        cos_yaw = task.scalar_scratch_b
        sin_yaw = task.scalar_scratch_c
        qw, qx, qy, qz = self._initial_quaternion
        torch.mul(cos_yaw, qw, out=qpos[:, 3])
        qpos[:, 3].add_(sin_yaw, alpha=-qz)
        torch.mul(cos_yaw, qx, out=qpos[:, 4])
        qpos[:, 4].add_(sin_yaw, alpha=qy)
        torch.mul(cos_yaw, qy, out=qpos[:, 5])
        qpos[:, 5].add_(sin_yaw, alpha=-qx)
        torch.mul(cos_yaw, qz, out=qpos[:, 6])
        qpos[:, 6].add_(sin_yaw, alpha=qw)
        qvel[:, :6].uniform_(
            -self._config.reset_base_qvel_limit,
            self._config.reset_base_qvel_limit,
            generator=task.generator,
        )

        task.reset_commands.uniform_(0.0, 1.0, generator=task.generator)
        task.reset_commands.mul_(self._command_span).add_(self._command_low)
        torch.linalg.vector_norm(task.reset_commands[:, :2], dim=1, out=task.reset_scalar_scratch)
        torch.le(task.reset_scalar_scratch, 0.2, out=task.reset_bool_scratch)
        task.reset_commands[:, 0].masked_fill_(task.reset_bool_scratch, 0.0)
        task.reset_commands[:, 1].masked_fill_(task.reset_bool_scratch, 0.0)
        if self._config.standing_probability > 0.0:
            task.reset_scalar_scratch.uniform_(0.0, 1.0, generator=task.generator)
            torch.lt(
                task.reset_scalar_scratch,
                self._config.standing_probability,
                out=task.reset_bool_scratch,
            )
            task.reset_commands.masked_fill_(task.reset_bool_scratch[:, None], 0.0)

        if self._config.gait_phase_init_mode == "independent":
            task.reset_gait_phase.uniform_(0.0, 2.0 * math.pi, generator=task.generator)
        else:
            task.reset_gait_phase[:, 0].uniform_(0.0, 2.0 * math.pi, generator=task.generator)
            task.reset_gait_phase[:, 1].copy_(task.reset_gait_phase[:, 0], non_blocking=True)
            task.reset_gait_phase[:, 1].add_(math.pi)

    def prepare_reset(
        self,
        *,
        active_mask: torch.Tensor,
        task_state: object,
    ) -> DeviceResetPayload:
        task = self._task(task_state)
        if (
            tuple(active_mask.shape) != (self._require_binding().num_envs,)
            or active_mask.dtype is not torch.bool
            or active_mask.device != self._device
        ):
            raise G1ManagedDeviceError("G1 device reset mask is incompatible")
        self._sample_reset(task)
        task.reset_mask.copy_(active_mask, non_blocking=True)
        root_offsets, position_offset, velocity_offset = self._require_reset_offsets()
        root_values = (
            task.reset_qpos[:, :3],
            task.reset_qpos[:, 3:7],
            task.reset_qvel[:, :3],
            task.reset_qvel[:, 3:6],
        )
        for value_offset, source in zip(root_offsets, root_values, strict=True):
            task.reset_values[value_offset].tensor[:, 0, :].copy_(source, non_blocking=True)
        task.reset_values[position_offset].tensor[:, :, 0].copy_(
            task.reset_qpos[:, 7:], non_blocking=True
        )
        task.reset_values[velocity_offset].tensor[:, :, 0].copy_(
            task.reset_qvel[:, 6:], non_blocking=True
        )
        return DeviceResetPayload(active_mask=task.reset_mask, values=task.reset_values)

    def complete_reset(
        self,
        *,
        active_mask: torch.Tensor,
        state: StateBatch,
        task_state: object,
        observation_buffers: tuple[torch.Tensor, ...],
    ) -> None:
        task = self._task(task_state)
        mask = active_mask[:, None]
        torch.where(mask, task.reset_commands, task.commands, out=task.commands)
        torch.where(mask, task.zero_actions, task.current_actions, out=task.current_actions)
        torch.where(mask, task.zero_actions, task.last_actions, out=task.last_actions)
        torch.where(mask, task.reset_gait_phase, task.gait_phase, out=task.gait_phase)
        self._write_observations(
            views=self._state_views(state),
            task=task,
            observation_buffers=observation_buffers,
        )


def create_g1_managed_device_runtime(
    *,
    backend: SimBackend,
    cfg: G1WalkEnvCfg,
    reset_seed: int = 0,
    max_episode_steps: int | None = None,
    record_lifecycle: bool = False,
    enable_stability_diagnostics: bool = False,
) -> DeviceManagedRuntime:
    """Create the strict mjwarp G1 device runtime on the cold path."""

    if backend.backend_type != "mjwarp":
        raise G1ManagedDeviceError("G1 device runtime requires the independent mjwarp backend")
    if not isinstance(enable_stability_diagnostics, bool):
        raise G1ManagedDeviceError("enable_stability_diagnostics must be a bool")
    _validate_device_profile(cfg)
    config = build_g1_kernel_config(
        backend=backend,
        cfg=cfg,
        reset_seed=reset_seed,
        observation_noise_seed=None,
        profile_name="device executor",
        error_type=G1ManagedDeviceError,
        allow_pd_randomization=True,
        allow_dof_armature_randomization=True,
        allow_body_gravity_compensation_randomization=True,
    )
    plan = compile_g1_managed_device_task(backend=backend, cfg=cfg)
    placement = plan.backend_io.control.buffer.placement
    kernel = G1ManagedDeviceKernel(
        config,
        expected_plan_fingerprint=plan.fingerprint,
        placement=placement,
    )
    backend.materialize()
    return DeviceManagedRuntime(
        backend=backend,
        plan=plan,
        kernel=kernel,
        max_episode_steps=cfg.max_episode_steps if max_episode_steps is None else max_episode_steps,
        run_seed=reset_seed,
        record_lifecycle=record_lifecycle,
        stability_buffer_provider=kernel if enable_stability_diagnostics else None,
    )


__all__ = [
    "G1_MANAGED_DEVICE_EXECUTOR_KEY",
    "G1ManagedDeviceError",
    "G1ManagedDeviceKernel",
    "compile_g1_managed_device_task",
    "create_g1_managed_device_runtime",
]
