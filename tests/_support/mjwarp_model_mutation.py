"""Shared production-CUDA fixtures for mjwarp Model mutation tests."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator, cast

import numpy as np
import torch

from tests._support.device_runtime import require_cuda
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
    MutationSelectorMode,
    MutationSelectorSpec,
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
    base_name: str
    body_name: str
    joint_name: str
    actuator_name: str
    keyframe_name: str | None
    body_id: int
    joint_id: int
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

    def model_field(self, field_name: str) -> torch.Tensor:
        return self.backend._get_model_field_bridge(field_name)

    def model_default(self, field_name: str) -> torch.Tensor:
        return self.backend._get_model_default_bridge(field_name)

    def restore_compiled_model_defaults(self) -> None:
        receipt = self.backend._model_materialization_receipt
        if receipt is None:
            return
        for field in receipt.fields:
            if field.role.value != "direct":
                continue
            target = self.model_field(field.field_name)
            default = self.model_default(field.field_name)
            target.copy_(default.expand_as(target), non_blocking=True)
        torch.cuda.current_stream(self.device).synchronize()

    def set_uniform_state(
        self,
        *,
        target_position: float | None = None,
        target_velocity: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        default_qpos = (
            self.backend.get_keyframe_qpos(self.keyframe_name)
            if self.keyframe_name is not None
            else self.backend.get_default_qpos()
        )
        qpos = np.tile(default_qpos, (self.num_envs, 1)).astype(np.float32)
        qvel = np.zeros((self.num_envs, self.backend.get_init_qvel().size), dtype=np.float32)
        if target_position is not None:
            qpos[:, self.raw_qpos_index] = target_position
        qvel[:, self.raw_qvel_index] = target_velocity
        reset_device_state(
            backend=self.backend,
            plan=self.plan,
            placement=self.placement,
            base_name=self.base_name,
            qpos=qpos,
            qvel=qvel,
        )
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


def _bind_io(
    backend: MjwarpBackend,
    placement: BufferPlacement,
) -> tuple[BoundBackendPlan, tuple[int, ...], tuple[int, ...]]:
    hinge_type = int(backend._mujoco.mjtJoint.mjJNT_HINGE)
    hinge_joint_ids = tuple(
        joint_id
        for joint_id in range(backend._cpu_model.njnt)
        if int(backend._cpu_model.jnt_type[joint_id]) == hinge_type
    )
    position_dofs = tuple(
        int(backend._cpu_model.jnt_qposadr[joint_id]) - backend._root_qpos_dim
        for joint_id in hinge_joint_ids
    )
    velocity_dofs = tuple(
        int(backend._cpu_model.jnt_dofadr[joint_id]) - backend._root_qvel_dim
        for joint_id in hinge_joint_ids
    )
    fields = (
        StateFieldSpec(
            semantic_key="dof.position",
            identity=BoundFieldIdentity(
                StateEntityKind.DOF,
                StateFieldKind.POSITION,
                position_dofs,
            ),
            frame=ReferenceFrame.JOINT,
            unit=PhysicalUnit.RADIAN,
            buffer=_state_contract(placement, (len(position_dofs),)),
        ),
        StateFieldSpec(
            semantic_key="dof.angular_velocity",
            identity=BoundFieldIdentity(
                StateEntityKind.DOF,
                StateFieldKind.ANGULAR_VELOCITY,
                velocity_dofs,
            ),
            frame=ReferenceFrame.JOINT,
            unit=PhysicalUnit.RADIAN_PER_SECOND,
            buffer=_state_contract(placement, (len(velocity_dofs),)),
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
    plan = backend.bind_task_io(
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
    return plan, position_dofs, velocity_dofs


def model_mutation_spec(
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
    if len(capability.recompute_levels) != 1:
        raise AssertionError(f"test fixture requires one recompute level for {target_key!r}")
    if capability.entity_kind is MutationEntityKind.ACTUATOR:
        selector = runtime.actuator_name
    elif capability.entity_kind is MutationEntityKind.BODY:
        selector = runtime.body_name
    else:
        selector = runtime.joint_name
    return MutationSpec(
        term_key=term_key,
        target=MutationTargetSpec(
            target_key=target_key,
            target_kind=capability.target_kind,
            entity_kind=capability.entity_kind,
            field_kind=capability.field_kind,
            selector=selector,
        ),
        trigger=MutationTrigger.RESET,
        commit_phase=MutationCommitPhase.RESET,
        operation=operation,
        baseline=MutationBaseline.DEFAULT,
        persistence=MutationPersistence.EPISODE,
        recompute=next(iter(capability.recompute_levels)),
        value_template=capability.value_template,
    )


def bind_model_plan(runtime: ModelMutationRuntime, key: PlanKey) -> BoundMutationPlan:
    specs = [
        model_mutation_spec(
            runtime,
            target_key=key.target_key,
            operation=key.operation,
            term_key="model.value",
        )
    ]
    if key.mixed_state:
        specs.extend(
            (
                model_mutation_spec(
                    runtime,
                    target_key="state.dof.position",
                    operation=MutationOperation.SET,
                    term_key="state.position",
                ),
                model_mutation_spec(
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
        model_mutation_spec(
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
                model_mutation_spec(
                    runtime,
                    target_key="state.dof.position",
                    operation=MutationOperation.SET,
                    term_key="state.position",
                ),
                model_mutation_spec(
                    runtime,
                    target_key="state.dof.angular_velocity",
                    operation=MutationOperation.SET,
                    term_key="state.velocity",
                ),
            )
        )
    return runtime.backend.bind_mutation_plan(tuple(specs))


def bind_combined_model_plan(
    runtime: ModelMutationRuntime,
    *,
    targets: tuple[tuple[str, MutationOperation, str], ...],
    mixed_state: bool,
) -> BoundMutationPlan:
    specs = [
        model_mutation_spec(
            runtime,
            target_key=target_key,
            operation=operation,
            term_key=term_key,
        )
        for target_key, operation, term_key in targets
    ]
    if mixed_state:
        specs.extend(
            (
                model_mutation_spec(
                    runtime,
                    target_key="state.dof.position",
                    operation=MutationOperation.SET,
                    term_key="state.position",
                ),
                model_mutation_spec(
                    runtime,
                    target_key="state.dof.angular_velocity",
                    operation=MutationOperation.SET,
                    term_key="state.velocity",
                ),
            )
        )
    return runtime.backend.bind_mutation_plan(tuple(specs))


def bind_state_reset_plan(runtime: ModelMutationRuntime) -> BoundMutationPlan:
    return runtime.backend.bind_mutation_plan(
        (
            model_mutation_spec(
                runtime,
                target_key="state.dof.position",
                operation=MutationOperation.SET,
                term_key="state.position",
            ),
            model_mutation_spec(
                runtime,
                target_key="state.dof.angular_velocity",
                operation=MutationOperation.SET,
                term_key="state.velocity",
            ),
        )
    )


def reset_device_state(
    *,
    backend: MjwarpBackend,
    plan: BoundBackendPlan,
    placement: BufferPlacement,
    base_name: str,
    qpos: np.ndarray,
    qvel: np.ndarray,
) -> None:
    """Reset the floating root and hinge DoFs through the typed device contract."""

    model = backend._cpu_model
    mujoco = backend._mujoco
    hinge_type = int(mujoco.mjtJoint.mjJNT_HINGE)
    hinge_joint_ids = tuple(
        joint_id for joint_id in range(model.njnt) if int(model.jnt_type[joint_id]) == hinge_type
    )
    hinge_names = tuple(
        str(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id))
        for joint_id in hinge_joint_ids
    )
    qpos_coordinates = tuple(int(model.jnt_qposadr[joint_id]) for joint_id in hinge_joint_ids)
    qvel_coordinates = tuple(int(model.jnt_dofadr[joint_id]) for joint_id in hinge_joint_ids)
    if qpos.shape != (backend.num_envs, int(model.nq)) or qvel.shape != (
        backend.num_envs,
        int(model.nv),
    ):
        raise AssertionError("device reset fixture does not match the mjwarp state layout")

    capabilities = {
        capability.target_key: capability
        for capability in backend.get_mutation_capability_manifest(
            ExecutionProfile.DEVICE_RESIDENT
        ).capabilities
    }
    all_hinges = MutationSelectorSpec(
        semantic_key="all_hinge_joints",
        mode=MutationSelectorMode.EXACT,
        expressions=hinge_names,
    )
    spec_entries: list[tuple[str, str, MutationSelectorSpec | str]] = [
        ("reset.dof.position", "state.dof.position", all_hinges),
        ("reset.dof.angular_velocity", "state.dof.angular_velocity", all_hinges),
    ]
    root_entries = (
        ("reset.root.position", "state.root.position", base_name),
        ("reset.root.orientation", "state.root.orientation", base_name),
        ("reset.root.linear_velocity", "state.root.linear_velocity", base_name),
        ("reset.root.angular_velocity", "state.root.angular_velocity", base_name),
    )
    if all(target_key in capabilities for _, target_key, _ in root_entries):
        spec_entries[0:0] = root_entries

    def state_spec(
        *,
        term_key: str,
        target_key: str,
        selector: MutationSelectorSpec | str,
    ) -> MutationSpec:
        capability = capabilities[target_key]
        return MutationSpec(
            term_key=term_key,
            target=MutationTargetSpec(
                target_key=target_key,
                target_kind=capability.target_kind,
                entity_kind=capability.entity_kind,
                field_kind=capability.field_kind,
                selector=selector,
            ),
            trigger=MutationTrigger.RESET,
            commit_phase=MutationCommitPhase.RESET,
            operation=MutationOperation.SET,
            baseline=MutationBaseline.DEFAULT,
            persistence=MutationPersistence.EPISODE,
            recompute=next(iter(capability.recompute_levels)),
            value_template=capability.value_template,
        )

    mutation_plan = backend.bind_mutation_plan(
        tuple(
            state_spec(term_key=term_key, target_key=target_key, selector=selector)
            for term_key, target_key, selector in spec_entries
        )
    )
    device = torch.device(f"cuda:{placement.device_index}")
    qpos_device = torch.as_tensor(qpos, dtype=torch.float32, device=device)
    qvel_device = torch.as_tensor(qvel, dtype=torch.float32, device=device)
    tensors: dict[str, torch.Tensor] = {
        "reset.dof.position": qpos_device[:, qpos_coordinates, None].contiguous(),
        "reset.dof.angular_velocity": qvel_device[:, qvel_coordinates, None].contiguous(),
    }
    if len(spec_entries) == 6:
        tensors.update(
            {
                "reset.root.position": qpos_device[:, None, 0:3].contiguous(),
                "reset.root.orientation": qpos_device[:, None, 3:7].contiguous(),
                "reset.root.linear_velocity": qvel_device[:, None, 0:3].contiguous(),
                "reset.root.angular_velocity": qvel_device[:, None, 3:6].contiguous(),
            }
        )
    active_mask = torch.ones((backend.num_envs,), dtype=torch.bool, device=device)
    rows = RowSelection.all(backend.num_envs)
    lease = DeviceBufferLease(f"complete-device-state-reset-{id(tensors):x}")
    completion = DeviceCompletion.record(
        placement=placement,
        owner_id=lease.owner_id,
        epoch=lease.epoch,
    )
    values: list[MutationValueBatch] = []
    for field_index, spec in enumerate(mutation_plan.specs):
        tensor = tensors[spec.term_key]
        view = DeviceTensorView(
            tensor_handle=tensor,
            contract=spec.value_buffer,
            lease=lease,
            completion=completion,
        )
        values.append(
            MutationValueBatch(
                plan=mutation_plan,
                field_index=field_index,
                rows=rows,
                buffer=BufferView(view, tuple(tensor.shape), spec.value_buffer),
            )
        )
    mask_contract = BufferContract(
        row_shape=(),
        dtype="bool",
        layout=BufferLayout.C_CONTIGUOUS,
        placement=placement,
        owner=BufferOwner.MANAGER,
        mutability=BufferMutability.READ_ONLY,
        lifetime=BufferLifetime.UNTIL_COMMIT,
        dlpack_exportable=True,
    )
    mask_view = DeviceTensorView(
        tensor_handle=active_mask,
        contract=mask_contract,
        lease=lease,
        completion=completion,
    )
    mutation = TypedBackendMutationBatch(
        plan=mutation_plan,
        rows=rows,
        state=SimulationStateMutationBatch(tuple(values)),
    )
    result = backend.reset_batch(
        plan,
        rows,
        mutation_batch=DeviceResetMutationBatch(
            plan=mutation_plan,
            rows=rows,
            mutation=mutation,
            active_mask=BufferView(mask_view, tuple(active_mask.shape), mask_contract),
        ),
    )
    wait_result(result)


@contextmanager
def model_mutation_runtime(
    *,
    num_envs: int,
    plan_keys: tuple[PlanKey, ...],
    model_file: str | None = None,
    base_name: str = BASE_NAME,
    body_name: str = BASE_NAME,
    joint_name: str = JOINT_NAME,
    actuator_name: str = ACTUATOR_NAME,
    keyframe_name: str | None = "stand",
) -> Iterator[ModelMutationRuntime]:
    require_cuda()
    if model_file is None:
        model_file = str(ASSETS_ROOT_PATH / "robots" / "g1" / "scene_flat.xml")
    backend = MjwarpBackend(
        SceneCfg(model_file=model_file),
        num_envs,
        0.02 / 3.0,
        base_name=base_name,
    )
    try:
        bridge = backend._ensure_device_bridge()
        device_index = bridge.qpos.device.index
        assert device_index is not None
        placement = BufferPlacement.device("cuda", int(device_index))
        plan, position_dofs, velocity_dofs = _bind_io(backend, placement)
        actuator_id = backend.get_actuator_names().index(actuator_name)
        joint_id = int(
            backend._mujoco.mj_name2id(
                backend._cpu_model,
                backend._mujoco.mjtObj.mjOBJ_JOINT,
                joint_name,
            )
        )
        body_id = int(
            backend._mujoco.mj_name2id(
                backend._cpu_model,
                backend._mujoco.mjtObj.mjOBJ_BODY,
                body_name,
            )
        )
        assert joint_id >= 0
        assert body_id > 0
        logical_position_index = int(backend.get_joint_dof_pos_indices((joint_name,))[0])
        logical_velocity_index = int(backend.get_joint_dof_vel_indices((joint_name,))[0])
        runtime = ModelMutationRuntime(
            backend=backend,
            plan=plan,
            placement=placement,
            device=bridge.qpos.device,
            num_envs=num_envs,
            base_name=base_name,
            body_name=body_name,
            joint_name=joint_name,
            actuator_name=actuator_name,
            keyframe_name=keyframe_name,
            body_id=body_id,
            joint_id=joint_id,
            actuator_id=actuator_id,
            dof_position_index=position_dofs.index(logical_position_index),
            dof_velocity_index=velocity_dofs.index(logical_velocity_index),
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
    "bind_combined_model_plan",
    "bind_combined_pd_plan",
    "bind_model_plan",
    "bind_state_reset_plan",
    "control_batch",
    "model_mutation_runtime",
    "model_mutation_spec",
    "reset_device_state",
    "state_tensor",
    "wait_result",
]
