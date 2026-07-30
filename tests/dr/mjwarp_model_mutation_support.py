"""Shared production-CUDA fixtures for mjwarp Model mutation tests."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator, cast

import numpy as np
import torch
from tests.training.device_runtime_harness import require_cuda

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base.backend import (
    BackendIORequirements,
    BoundBackendPlan,
    BoundFieldIdentity,
    BoundMutationPlan,
    BufferContract,
    BufferLayout,
    BufferLifetime,
    BufferMutability,
    BufferOwner,
    BufferPlacement,
    BufferView,
    ControlBatch,
    ControlSpec,
    DeviceBufferLease,
    DeviceCompletion,
    DeviceResetMutationBatch,
    DeviceTensorView,
    ExecutionProfile,
    ModelParameterMutationBatch,
    MutationBaseline,
    MutationCommitPhase,
    MutationEntityKind,
    MutationFieldKind,
    MutationOperation,
    MutationPersistence,
    MutationRecomputeLevel,
    MutationSpec,
    MutationTargetKind,
    MutationTargetSpec,
    MutationTrigger,
    MutationValueBatch,
    PhysicalUnit,
    ReferenceFrame,
    RowSelection,
    SimulationStateMutationBatch,
    StateEntityKind,
    StateFieldKind,
    StateFieldSpec,
    TypedBackendMutationBatch,
)
from unilab.base.backend.mjwarp.backend import MjwarpBackend
from unilab.base.scene import SceneCfg

ACTUATOR_NAME = "left_hip_pitch_joint"
JOINT_NAME = ACTUATOR_NAME
BASE_NAME = "pelvis"


@dataclass(frozen=True)
class PlanKey:
    target_key: str
    operation: MutationOperation
    mixed_state: bool = False


@dataclass
class ModelMutationRuntime:
    backend: MjwarpBackend
    plan: BoundBackendPlan
    placement: BufferPlacement
    device: torch.device
    num_envs: int
    actuator_id: int
    dof_position_index: int
    dof_velocity_index: int
    raw_qpos_index: int
    raw_qvel_index: int
    mutation_plans: dict[PlanKey, BoundMutationPlan]

    @property
    def gain(self) -> torch.Tensor:
        return self.backend._get_model_field_bridge("actuator_gainprm")

    @property
    def bias(self) -> torch.Tensor:
        return self.backend._get_model_field_bridge("actuator_biasprm")

    @property
    def default_gain(self) -> torch.Tensor:
        return self.backend._get_model_default_bridge("actuator_gainprm")

    @property
    def default_bias(self) -> torch.Tensor:
        return self.backend._get_model_default_bridge("actuator_biasprm")

    def restore_compiled_model_defaults(self) -> None:
        self.gain.copy_(self.default_gain.expand_as(self.gain), non_blocking=True)
        self.bias.copy_(self.default_bias.expand_as(self.bias), non_blocking=True)
        torch.cuda.current_stream(self.device).synchronize()

    def set_uniform_state(
        self,
        *,
        target_position: float | None = None,
        target_velocity: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        qpos = np.tile(self.backend.get_keyframe_qpos("stand"), (self.num_envs, 1)).astype(
            np.float32
        )
        qvel = np.zeros((self.num_envs, self.backend.get_init_qvel().size), dtype=np.float32)
        if target_position is not None:
            qpos[:, self.raw_qpos_index] = target_position
        qvel[:, self.raw_qvel_index] = target_velocity
        rows = np.arange(self.num_envs, dtype=np.int32)
        self.backend.set_state(rows, qpos, qvel)
        initial = self.backend.read_state_batch(
            self.plan,
            RowSelection.all(self.num_envs),
        )
        wait_result(initial)
        return qpos, qvel

    def position_hold_control(self, qpos: np.ndarray) -> torch.Tensor:
        values = np.zeros((self.num_envs, self.backend.num_actuators), dtype=np.float32)
        model = self.backend._cpu_model
        for actuator_id in self.backend._position_actuator_ids:
            joint_id = int(model.actuator_trnid[actuator_id, 0])
            qpos_index = int(model.jnt_qposadr[joint_id])
            values[:, actuator_id] = qpos[:, qpos_index]
        return torch.as_tensor(values, dtype=torch.float32, device=self.device)


def _state_contract(placement: BufferPlacement, row_shape: tuple[int, ...]) -> BufferContract:
    return BufferContract(
        row_shape=row_shape,
        dtype="float32",
        layout=BufferLayout.C_CONTIGUOUS,
        placement=placement,
        owner=BufferOwner.BACKEND,
        mutability=BufferMutability.READ_ONLY,
        lifetime=BufferLifetime.BORROWED_UNTIL_MUTATION,
        dlpack_exportable=True,
    )


def _bind_io(backend: MjwarpBackend, placement: BufferPlacement) -> BoundBackendPlan:
    all_dofs = tuple(range(backend.num_dof_vel))
    fields = (
        StateFieldSpec(
            semantic_key="dof.position",
            identity=BoundFieldIdentity(
                StateEntityKind.DOF,
                StateFieldKind.POSITION,
                all_dofs,
            ),
            frame=ReferenceFrame.JOINT,
            unit=PhysicalUnit.RADIAN,
            buffer=_state_contract(placement, (len(all_dofs),)),
        ),
        StateFieldSpec(
            semantic_key="dof.angular_velocity",
            identity=BoundFieldIdentity(
                StateEntityKind.DOF,
                StateFieldKind.ANGULAR_VELOCITY,
                all_dofs,
            ),
            frame=ReferenceFrame.JOINT,
            unit=PhysicalUnit.RADIAN_PER_SECOND,
            buffer=_state_contract(placement, (len(all_dofs),)),
        ),
    )
    control = BufferContract(
        row_shape=(backend.num_actuators,),
        dtype="float32",
        layout=BufferLayout.C_CONTIGUOUS,
        placement=placement,
        owner=BufferOwner.RUNNER,
        mutability=BufferMutability.READ_ONLY,
        lifetime=BufferLifetime.UNTIL_STEP_COMPLETE,
        dlpack_exportable=True,
    )
    return backend.bind_task_io(
        BackendIORequirements(
            state_fields=fields,
            control=ControlSpec(
                semantic_key="joint.position_target",
                buffer=control,
                physics_substeps_per_control=1,
            ),
            execution_profile=ExecutionProfile.DEVICE_RESIDENT,
        )
    )


def _mutation_spec(
    runtime: ModelMutationRuntime,
    *,
    target_key: str,
    operation: MutationOperation,
    term_key: str,
) -> MutationSpec:
    capabilities = {
        capability.target_key: capability
        for capability in runtime.backend.get_mutation_capability_manifest(
            ExecutionProfile.DEVICE_RESIDENT
        ).capabilities
    }
    capability = capabilities[target_key]
    return MutationSpec(
        term_key=term_key,
        target=MutationTargetSpec(
            target_key=target_key,
            target_kind=capability.target_kind,
            entity_kind=capability.entity_kind,
            field_kind=capability.field_kind,
            selector=ACTUATOR_NAME
            if capability.entity_kind is MutationEntityKind.ACTUATOR
            else JOINT_NAME,
        ),
        trigger=MutationTrigger.RESET,
        commit_phase=MutationCommitPhase.RESET,
        operation=operation,
        baseline=MutationBaseline.DEFAULT,
        persistence=MutationPersistence.EPISODE,
        recompute=(
            MutationRecomputeLevel.NONE
            if capability.target_kind is MutationTargetKind.MODEL_PARAMETER
            else MutationRecomputeLevel.KINEMATICS
        ),
        value_template=capability.value_template,
    )


def bind_model_plan(runtime: ModelMutationRuntime, key: PlanKey) -> BoundMutationPlan:
    specs = [
        _mutation_spec(
            runtime,
            target_key=key.target_key,
            operation=key.operation,
            term_key="model.value",
        )
    ]
    if key.mixed_state:
        specs.extend(
            (
                _mutation_spec(
                    runtime,
                    target_key="state.dof.position",
                    operation=MutationOperation.SET,
                    term_key="state.position",
                ),
                _mutation_spec(
                    runtime,
                    target_key="state.dof.angular_velocity",
                    operation=MutationOperation.SET,
                    term_key="state.velocity",
                ),
            )
        )
    return runtime.backend.bind_mutation_plan(tuple(specs))


def bind_combined_pd_plan(
    runtime: ModelMutationRuntime,
    *,
    mixed_state: bool,
) -> BoundMutationPlan:
    specs = [
        _mutation_spec(
            runtime,
            target_key=target_key,
            operation=MutationOperation.SET,
            term_key=f"model.{field_name}",
        )
        for target_key, field_name in (
            ("actuator.pd_stiffness", "stiffness"),
            ("actuator.pd_damping", "damping"),
        )
    ]
    if mixed_state:
        specs.extend(
            (
                _mutation_spec(
                    runtime,
                    target_key="state.dof.position",
                    operation=MutationOperation.SET,
                    term_key="state.position",
                ),
                _mutation_spec(
                    runtime,
                    target_key="state.dof.angular_velocity",
                    operation=MutationOperation.SET,
                    term_key="state.velocity",
                ),
            )
        )
    return runtime.backend.bind_mutation_plan(tuple(specs))


@contextmanager
def model_mutation_runtime(
    *,
    num_envs: int,
    plan_keys: tuple[PlanKey, ...],
) -> Iterator[ModelMutationRuntime]:
    require_cuda()
    backend = MjwarpBackend(
        SceneCfg(model_file=str(ASSETS_ROOT_PATH / "robots" / "g1" / "scene_flat.xml")),
        num_envs,
        0.02 / 3.0,
        base_name=BASE_NAME,
    )
    try:
        bridge = backend._ensure_device_bridge()
        device_index = bridge.qpos.device.index
        assert device_index is not None
        placement = BufferPlacement.device("cuda", int(device_index))
        plan = _bind_io(backend, placement)
        actuator_id = backend.get_actuator_names().index(ACTUATOR_NAME)
        joint_id = int(backend._cpu_model.actuator_trnid[actuator_id, 0])
        runtime = ModelMutationRuntime(
            backend=backend,
            plan=plan,
            placement=placement,
            device=bridge.qpos.device,
            num_envs=num_envs,
            actuator_id=actuator_id,
            dof_position_index=int(backend.get_joint_dof_pos_indices((JOINT_NAME,))[0]),
            dof_velocity_index=int(backend.get_joint_dof_vel_indices((JOINT_NAME,))[0]),
            raw_qpos_index=int(backend._cpu_model.jnt_qposadr[joint_id]),
            raw_qvel_index=int(backend._cpu_model.jnt_dofadr[joint_id]),
            mutation_plans={},
        )
        for key in plan_keys:
            runtime.mutation_plans[key] = bind_model_plan(runtime, key)
        yield runtime
    finally:
        backend.cleanup_scene_assets()


class ResetBatchBuffers:
    """Stable manager buffers that can publish a fresh epoch without reallocating tensors."""

    def __init__(self, runtime: ModelMutationRuntime, plan: BoundMutationPlan) -> None:
        self.runtime = runtime
        self.plan = plan
        self.rows = RowSelection.all(runtime.num_envs)
        self.values = {
            spec.term_key: torch.zeros(
                (runtime.num_envs, *spec.value_buffer.row_shape),
                dtype=torch.float32,
                device=runtime.device,
            )
            for spec in plan.specs
        }
        self.active_mask = torch.zeros(
            (runtime.num_envs,),
            dtype=torch.bool,
            device=runtime.device,
        )
        self.lease = DeviceBufferLease(f"mjwarp-model-reset-{id(self):x}")
        self.event = cast(torch.cuda.Event, torch.cuda.Event(enable_timing=False))
        self._published = False

    @property
    def numeric_addresses(self) -> tuple[int, ...]:
        return (
            *(int(value.data_ptr()) for value in self.values.values()),
            int(self.active_mask.data_ptr()),
        )

    def publish(self) -> DeviceResetMutationBatch:
        if self._published:
            self.lease.invalidate()
        self._published = True
        completion = DeviceCompletion.record(
            placement=self.runtime.placement,
            owner_id=self.lease.owner_id,
            epoch=self.lease.epoch,
            event=self.event,
        )
        model_values: list[MutationValueBatch] = []
        state_values: list[MutationValueBatch] = []
        for field_index, spec in enumerate(self.plan.specs):
            tensor = self.values[spec.term_key]
            view = DeviceTensorView(
                tensor_handle=tensor,
                contract=spec.value_buffer,
                lease=self.lease,
                completion=completion,
            )
            value = MutationValueBatch(
                plan=self.plan,
                field_index=field_index,
                rows=self.rows,
                buffer=BufferView(view, tuple(tensor.shape), spec.value_buffer),
            )
            if spec.target.target_kind is MutationTargetKind.MODEL_PARAMETER:
                model_values.append(value)
            else:
                state_values.append(value)
        mask_contract = BufferContract(
            row_shape=(),
            dtype="bool",
            layout=BufferLayout.C_CONTIGUOUS,
            placement=self.runtime.placement,
            owner=BufferOwner.MANAGER,
            mutability=BufferMutability.READ_ONLY,
            lifetime=BufferLifetime.UNTIL_COMMIT,
            dlpack_exportable=True,
        )
        mask_view = DeviceTensorView(
            tensor_handle=self.active_mask,
            contract=mask_contract,
            lease=self.lease,
            completion=completion,
        )
        mutation = TypedBackendMutationBatch(
            plan=self.plan,
            rows=self.rows,
            model=ModelParameterMutationBatch(tuple(model_values)),
            state=SimulationStateMutationBatch(tuple(state_values)),
        )
        return DeviceResetMutationBatch(
            plan=self.plan,
            rows=self.rows,
            mutation=mutation,
            active_mask=BufferView(
                mask_view,
                tuple(self.active_mask.shape),
                mask_contract,
            ),
        )


def control_batch(
    runtime: ModelMutationRuntime, values: torch.Tensor, *, owner: str
) -> ControlBatch:
    contract = runtime.plan.control.buffer
    lease = DeviceBufferLease(owner)
    completion = DeviceCompletion.record(
        placement=runtime.placement,
        owner_id=lease.owner_id,
        epoch=lease.epoch,
    )
    view = DeviceTensorView(
        tensor_handle=values,
        contract=contract,
        lease=lease,
        completion=completion,
    )
    return ControlBatch(
        plan=runtime.plan,
        rows=RowSelection.all(runtime.num_envs),
        buffer=BufferView(view, tuple(values.shape), contract),
    )


def state_tensor(state: Any, key: str) -> torch.Tensor:
    view = state.buffer(key).handle
    assert isinstance(view, DeviceTensorView)
    view.wait()
    return view.torch()


def wait_result(result: Any) -> None:
    completion = result.diagnostics.completion_event
    assert completion is not None
    completion.handle.event.synchronize()


__all__ = [
    "ACTUATOR_NAME",
    "ModelMutationRuntime",
    "PlanKey",
    "ResetBatchBuffers",
    "bind_combined_pd_plan",
    "bind_model_plan",
    "control_batch",
    "model_mutation_runtime",
    "state_tensor",
    "wait_result",
]
