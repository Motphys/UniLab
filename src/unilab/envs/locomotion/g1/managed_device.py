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
from unilab.dtype_config import get_global_dtype
from unilab.manager import (
    BackendEntityResolver,
    DeviceManagedRuntime,
    DeviceResetPayload,
    ManagedKernelBinding,
    ManagerContractError,
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
from .managed_reference import (
    _ACTOR_WIDTH,
    _CRITIC_WIDTH,
    _RESET_TERM,
    _ROOT_RESET_SUFFIXES,
    _STATE_KEYS,
    G1ManagedReferenceError,
    _action_scale,
    _g1_selectors,
    _g1_state_requirements,
    _G1KernelConfig,
    _kernel_config,
    _reset_suffix_for_dof,
    _reset_term_key,
    _validate_reference_profile,
)

G1_MANAGED_DEVICE_EXECUTOR_KEY = "device.torch.g1-walk-flat.v1"


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
    reset_position: tuple[Any, ...],
    reset_velocity: tuple[Any, ...],
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
                value_template=_device_manager_buffer(
                    placement=placement,
                    row_shape=row_shape,
                    lifetime=BufferLifetime.UNTIL_COMMIT,
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
                value_template=_device_manager_buffer(
                    placement=placement,
                    row_shape=(1,),
                    lifetime=BufferLifetime.UNTIL_COMMIT,
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
                value_template=_device_manager_buffer(
                    placement=placement,
                    row_shape=(1,),
                    lifetime=BufferLifetime.UNTIL_COMMIT,
                ),
            )
        )
    return tuple(templates)


def _validate_device_profile(cfg: G1WalkEnvCfg) -> G1WalkRewardConfig:
    try:
        reward = _validate_reference_profile(cfg)
    except G1ManagedReferenceError as exc:
        raise G1ManagedDeviceError(
            str(exc).replace("managed reference", "device executor")
        ) from exc
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
    _action_scale(cfg, action_dim)
    root, dofs, reset_position, reset_velocity = _g1_selectors(actuator_names)
    state_requirements = _g1_state_requirements(root=root, dofs=dofs, action_dim=action_dim)
    reset_templates = _device_reset_templates(
        placement=placement,
        root=root,
        reset_position=reset_position,
        reset_velocity=reset_velocity,
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
            output=TensorSpec((_ACTOR_WIDTH,), "float32"),
        )
    )
    registry.register(
        TermDefinition(
            key="g1.device.critic_observation",
            version="1",
            phase=TermPhase.TERMINAL_OBSERVATION,
            role=TermRole.OBSERVATION,
            state_requirements=state_requirements,
            output=TensorSpec((_CRITIC_WIDTH,), "float32"),
        )
    )
    task = TaskSpec.create(
        key="g1_walk_flat.managed_device",
        terms=(
            TermInvocation.create(key=_RESET_TERM, definition_key="g1.device.reset"),
            TermInvocation.create(
                key="g1_termination",
                definition_key="g1.device.termination",
                dependencies=(_RESET_TERM,),
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
    widths = tuple(group.width for group in plan.policy_abi.observation_groups)
    if widths != (_ACTOR_WIDTH, _CRITIC_WIDTH):
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
    reset_values: tuple[torch.Tensor, ...]
    reset_mask: torch.Tensor
    reset_yaw: torch.Tensor
    reset_bool_scratch: torch.Tensor
    reset_scalar_scratch: torch.Tensor
    zero_actions: torch.Tensor
    generator: torch.Generator


class G1ManagedDeviceKernel:
    """G1 task math implemented directly with Torch CUDA operations."""

    executor_key = G1_MANAGED_DEVICE_EXECUTOR_KEY

    def __init__(
        self,
        config: _G1KernelConfig,
        *,
        expected_plan_fingerprint: str,
        placement: BufferPlacement,
    ) -> None:
        if not isinstance(config, _G1KernelConfig):
            raise G1ManagedDeviceError("G1 device kernel requires a frozen G1 config")
        if not isinstance(expected_plan_fingerprint, str) or not expected_plan_fingerprint.strip():
            raise G1ManagedDeviceError("G1 device kernel requires an expected plan fingerprint")
        if placement.device_type != "cuda" or placement.device_index is None:
            raise G1ManagedDeviceError("G1 device kernel requires CUDA placement")
        self._config = config
        self._expected_plan_fingerprint = expected_plan_fingerprint
        self._device = torch.device(f"cuda:{placement.device_index}")
        self._action_dim = int(config.default_angles.size)
        self._binding: ManagedKernelBinding | None = None
        self._state_indices: tuple[int, ...] | None = None
        self._observation_indices: tuple[int, int] | None = None
        self._mutation_plan: BoundMutationPlan | None = None
        self._root_reset_indices: tuple[int, int, int, int] | None = None
        self._position_reset_indices: tuple[int, ...] | None = None
        self._velocity_reset_indices: tuple[int, ...] | None = None
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
        self._reward_scales = {name: float(scale) for name, scale in config.reward_terms}

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
        missing_state = tuple(key for key in _STATE_KEYS if key not in state_indices)
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
        root_keys = tuple(_reset_term_key(suffix=item[0]) for item in _ROOT_RESET_SUFFIXES)
        position_keys = tuple(
            _reset_term_key(suffix=_reset_suffix_for_dof(kind="position", index=index))
            for index in range(self._action_dim)
        )
        velocity_keys = tuple(
            _reset_term_key(suffix=_reset_suffix_for_dof(kind="velocity", index=index))
            for index in range(self._action_dim)
        )
        missing_mutations = tuple(
            key
            for key in (*root_keys, *position_keys, *velocity_keys)
            if key not in mutation_indices
        )
        if missing_mutations:
            raise G1ManagedDeviceError(
                "G1 device plan is missing reset mutations: " + ", ".join(missing_mutations)
            )
        self._binding = binding
        self._state_indices = tuple(state_indices[key] for key in _STATE_KEYS)
        self._observation_indices = observation_indices
        self._mutation_plan = mutation_plan
        self._root_reset_indices = tuple(mutation_indices[key] for key in root_keys)  # type: ignore[assignment]
        self._position_reset_indices = tuple(mutation_indices[key] for key in position_keys)
        self._velocity_reset_indices = tuple(mutation_indices[key] for key in velocity_keys)

    def _require_binding(self) -> ManagedKernelBinding:
        if self._binding is None:
            raise G1ManagedDeviceError("G1 device kernel has not been bound")
        return self._binding

    def _require_reset_indices(
        self,
    ) -> tuple[tuple[int, int, int, int], tuple[int, ...], tuple[int, ...]]:
        if (
            self._root_reset_indices is None
            or self._position_reset_indices is None
            or self._velocity_reset_indices is None
        ):
            raise G1ManagedDeviceError("G1 device reset indices are not bound")
        return (
            self._root_reset_indices,
            self._position_reset_indices,
            self._velocity_reset_indices,
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
        if mutation_plan is None:  # pragma: no cover - bind invariant.
            raise G1ManagedDeviceError("G1 device task state lacks mutation plan")
        reset_values = tuple(
            torch.empty(
                (num_envs, *spec.value_buffer.row_shape),
                dtype=dtype,
                device=device,
            )
            for spec in mutation_plan.specs
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
            reset_scalar_scratch=torch.empty((num_envs,), dtype=dtype, device=device),
            zero_actions=torch.zeros((num_envs, self._action_dim), dtype=dtype, device=device),
            generator=generator,
        )

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
            _STATE_KEYS, self._state_indices, expected_shapes, strict=True
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

    @staticmethod
    def _bezier_height(phase: torch.Tensor, swing_height: float) -> torch.Tensor:
        normalized = torch.remainder(phase + math.pi, 2.0 * math.pi) - math.pi
        x = (normalized + math.pi) / (2.0 * math.pi)
        rise_t = 2.0 * x
        fall_t = 2.0 * x - 1.0
        rise = rise_t**3 + 3.0 * rise_t**2 * (1.0 - rise_t)
        fall = fall_t**3 + 3.0 * fall_t**2 * (1.0 - fall_t)
        return torch.where(x <= 0.5, swing_height * rise, swing_height * (1.0 - fall))

    def _reward_value(
        self,
        name: str,
        *,
        views: _G1DeviceStateViews,
        task: _G1DeviceTaskState,
    ) -> torch.Tensor:
        if name == "tracking_lin_vel":
            error = torch.sum(
                (task.commands[:, :2] - views.pelvis_local_linear_velocity[:, :2]) ** 2, dim=1
            )
            return torch.exp(-error / self._config.tracking_sigma)
        if name == "tracking_ang_vel":
            error = (task.commands[:, 2] - views.torso_gyro[:, 2]) ** 2
            return torch.exp(-error / self._config.tracking_sigma)
        if name == "forward_progress":
            speed = torch.clamp_min(views.pelvis_local_linear_velocity[:, 0], 0.0)
            command = torch.clamp_min(task.commands[:, 0], 1.0e-6)
            return torch.clamp_max(speed / command, 1.0)
        if name == "under_speed":
            command = torch.clamp_min(task.commands[:, 0], 1.0e-6)
            speed = torch.clamp_min(views.pelvis_local_linear_velocity[:, 0], 0.0)
            return torch.clamp_min(task.commands[:, 0] - speed, 0.0) / command
        if name == "lin_vel_z":
            return views.pelvis_local_linear_velocity[:, 2] ** 2
        if name in {"orientation", "penalty_orientation"}:
            return torch.sum(views.torso_upvector[:, :2] ** 2, dim=1)
        if name in {"ang_vel_xy", "penalty_ang_vel_xy"}:
            return torch.sum(views.torso_gyro[:, :2] ** 2, dim=1)
        if name in {"action_rate", "penalty_action_rate"}:
            return torch.sum((task.current_actions - task.last_actions) ** 2, dim=1)
        if name == "base_height":
            return (views.root_position[:, 2] - self._config.base_height_target) ** 2
        if name == "pose":
            return torch.sum(
                self._pose_weights * (views.dof_position - self._default_angles) ** 2,
                dim=1,
            )
        if name == "upper_body_pose":
            return torch.sum(
                self._upper_body_pose_weights * (views.dof_position - self._default_angles) ** 2,
                dim=1,
            )
        if name == "penalty_close_feet_xy":
            distance = torch.linalg.vector_norm(
                views.left_foot_position[:, :2] - views.right_foot_position[:, :2],
                dim=1,
            )
            return torch.where(
                distance < self._config.close_feet_threshold,
                (distance - self._config.close_feet_threshold) ** 2,
                torch.zeros_like(distance),
            )
        if name in {"feet_phase", "feet_phase_contrast"}:
            left = self._bezier_height(task.gait_phase[:, 0], self._config.feet_phase_swing_height)
            right = self._bezier_height(task.gait_phase[:, 1], self._config.feet_phase_swing_height)
            gate = (
                torch.clamp_min(views.pelvis_local_linear_velocity[:, 0], 0.0)
                >= self._config.min_forward_speed_for_gait_reward
            ).to(dtype=torch.float32)
            if name == "feet_phase":
                error = (views.left_foot_position[:, 2] - left) ** 2 + (
                    views.right_foot_position[:, 2] - right
                ) ** 2
            else:
                error = (
                    views.left_foot_position[:, 2]
                    - views.right_foot_position[:, 2]
                    - (left - right)
                ) ** 2
            return torch.exp(-error / self._config.feet_phase_tracking_sigma) * gate
        if name == "alive":
            return torch.ones_like(views.root_position[:, 2])
        raise G1ManagedDeviceError(f"unsupported G1 device reward term {name!r}")

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
        expected_actor = (self._require_binding().num_envs, _ACTOR_WIDTH)
        expected_critic = (self._require_binding().num_envs, _CRITIC_WIDTH)
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
        if cursor != _ACTOR_WIDTH:  # pragma: no cover - static layout.
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
        if cursor != _CRITIC_WIDTH:  # pragma: no cover - static layout.
            raise G1ManagedDeviceError("G1 device critic observation layout is inconsistent")

    def evaluate_terminal(
        self,
        *,
        state: StateBatch,
        task_state: object,
        reward_out: torch.Tensor,
        terminated_out: torch.Tensor,
        terminal_observation_buffers: tuple[torch.Tensor, ...],
    ) -> None:
        task = self._task(task_state)
        views = self._state_views(state)
        tilt = torch.acos(torch.clamp(views.torso_upvector[:, 2], -1.0, 1.0))
        torch.logical_or(
            tilt > self._config.max_tilt_rad,
            views.root_position[:, 2] < self._config.min_base_height,
            out=terminated_out,
        )
        reward_out.zero_()
        for name, scale in self._reward_scales.items():
            if scale != 0.0:
                reward_out.add_(self._reward_value(name, views=views, task=task), alpha=scale)
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
        half_yaw = task.reset_yaw * 0.5
        cos_yaw = torch.cos(half_yaw)
        sin_yaw = torch.sin(half_yaw)
        qw, qx, qy, qz = self._initial_qpos[3:7]
        qpos[:, 3] = qw * cos_yaw - qz * sin_yaw
        qpos[:, 4] = qx * cos_yaw + qy * sin_yaw
        qpos[:, 5] = qy * cos_yaw - qx * sin_yaw
        qpos[:, 6] = qz * cos_yaw + qw * sin_yaw
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
        root_indices, position_indices, velocity_indices = self._require_reset_indices()
        root_values = (
            task.reset_qpos[:, :3],
            task.reset_qpos[:, 3:7],
            task.reset_qvel[:, :3],
            task.reset_qvel[:, 3:6],
        )
        for mutation_index, source in zip(root_indices, root_values, strict=True):
            task.reset_values[mutation_index][:, 0, :].copy_(source, non_blocking=True)
        for dof_index, mutation_index in enumerate(position_indices):
            task.reset_values[mutation_index][:, 0, 0].copy_(
                task.reset_qpos[:, 7 + dof_index], non_blocking=True
            )
        for dof_index, mutation_index in enumerate(velocity_indices):
            task.reset_values[mutation_index][:, 0, 0].copy_(
                task.reset_qvel[:, 6 + dof_index], non_blocking=True
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
) -> DeviceManagedRuntime:
    """Create the strict mjwarp G1 device runtime on the cold path."""

    if backend.backend_type != "mjwarp":
        raise G1ManagedDeviceError("G1 device runtime requires the independent mjwarp backend")
    _validate_device_profile(cfg)
    try:
        config = _kernel_config(
            backend=backend,
            cfg=cfg,
            reset_seed=reset_seed,
            observation_noise_seed=None,
        )
    except G1ManagedReferenceError as exc:
        raise G1ManagedDeviceError(
            str(exc).replace("managed reference", "device executor")
        ) from exc
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
        record_lifecycle=record_lifecycle,
    )


__all__ = [
    "G1_MANAGED_DEVICE_EXECUTOR_KEY",
    "G1ManagedDeviceError",
    "G1ManagedDeviceKernel",
    "compile_g1_managed_device_task",
    "create_g1_managed_device_runtime",
]
