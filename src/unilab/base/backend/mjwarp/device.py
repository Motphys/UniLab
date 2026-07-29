"""CUDA-resident typed batch binding for the independent ``mjwarp`` backend.

This module deliberately does not adapt the host cache.  It cold-binds Warp
arrays through zero-copy Torch aliases, packs the declared semantic fields into
stable backend-owned CUDA buffers, and exposes only lease-guarded device views.
The physics data, state packs, stream and events remain owned by ``mjwarp``;
manager/runner callers receive no raw Warp model or data object.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from unilab.base.backend.device import DeviceBufferLease, DeviceCompletion, DeviceTensorView

from ..batch import (
    BACKEND_BATCH_CONTRACT_VERSION,
    BackendBatchContractError,
    BackendBatchCounters,
    BackendBatchDiagnostics,
    BackendCompletionEvent,
    BackendIORequirements,
    BackendReadResult,
    BackendTiming,
    BoundBackendPlan,
    BoundStatePlan,
    BufferLayout,
    BufferLifetime,
    BufferMutability,
    BufferOwner,
    BufferPlacement,
    BufferView,
    ExecutionProfile,
    MemorySpace,
    PhysicalUnit,
    ReferenceFrame,
    RowSelection,
    StateBatch,
    StateBatchLease,
    StateBatchPhase,
    StateEntityKind,
    StateFieldKind,
    StateFieldSpec,
)

if TYPE_CHECKING:
    from .backend import MjwarpBackend


_STATE_FINGERPRINT_PREFIX = "mjwarp-device-state-v1"
_PLAN_FINGERPRINT_PREFIX = "mjwarp-device-batch-v1"


@dataclass
class _MjwarpDeviceStateSource:
    """One raw device source and its cold-allocated contiguous state pack."""

    spec: StateFieldSpec
    source: torch.Tensor = field(repr=False)
    packed: torch.Tensor = field(repr=False)
    reset_staging: torch.Tensor = field(repr=False)

    def __post_init__(self) -> None:
        if (
            self.source.device != self.packed.device
            or self.source.device != self.reset_staging.device
        ):
            raise BackendBatchContractError("mjwarp device state source and pack differ in device")
        if self.source.dtype != self.packed.dtype or self.source.dtype != self.reset_staging.dtype:
            raise BackendBatchContractError("mjwarp device state source and pack differ in dtype")
        if self.source.shape != self.packed.shape or self.source.shape != self.reset_staging.shape:
            raise BackendBatchContractError("mjwarp device state source and pack differ in shape")
        if not self.packed.is_contiguous() or not self.reset_staging.is_contiguous():
            raise BackendBatchContractError("mjwarp device state pack must be C-contiguous")

    def refresh(self) -> None:
        """Enqueue one D2D pack copy on the caller-selected physics stream."""

        self.packed.copy_(self.source, non_blocking=True)

    def refresh_masked(self, active_mask: torch.Tensor) -> None:
        """Commit reset rows while preserving the preceding terminal pack."""

        if (
            active_mask.device != self.packed.device
            or active_mask.dtype is not torch.bool
            or tuple(active_mask.shape) != (self.packed.shape[0],)
            or not active_mask.is_contiguous()
        ):
            raise BackendBatchContractError("mjwarp device state reset mask is incompatible")
        self.reset_staging.copy_(self.source, non_blocking=True)
        broadcast_shape = (self.packed.shape[0],) + (1,) * (self.packed.ndim - 1)
        torch.where(
            active_mask.view(broadcast_shape),
            self.reset_staging,
            self.packed,
            out=self.packed,
        )


@dataclass
class MjwarpDeviceBatchPlan:
    """Runtime companion for one immutable device-resident batch plan."""

    public_plan: BoundBackendPlan
    sources: tuple[_MjwarpDeviceStateSource, ...]
    state_lease: StateBatchLease
    device_lease: DeviceBufferLease
    placement: BufferPlacement
    owner_id: str
    _refreshes: int = field(default=0, init=False, repr=False)

    def refresh(self) -> None:
        for source in self.sources:
            source.refresh()
        self._refreshes += 1

    def refresh_masked(self, active_mask: torch.Tensor) -> None:
        """Refresh reset rows without changing terminal values for their complement."""

        for source in self.sources:
            source.refresh_masked(active_mask)
        self._refreshes += 1

    def invalidate(self) -> None:
        self.state_lease.invalidate()
        self.device_lease.invalidate()

    def materialize(
        self,
        *,
        rows: RowSelection,
        phase: StateBatchPhase,
        completion_event: torch.cuda.Event,
    ) -> BackendReadResult:
        if not rows.is_all:
            raise BackendBatchContractError(
                "mjwarp device-resident StateBatch currently requires all rows; "
                "selected device rows must remain inside a typed device mutation envelope"
            )
        if not isinstance(completion_event, torch.cuda.Event):
            raise BackendBatchContractError("mjwarp device completion event is invalid")
        self.invalidate()
        completion = DeviceCompletion(
            placement=self.placement,
            owner_id=self.owner_id,
            epoch=self.device_lease.epoch,
            event=completion_event,
        )
        start = time.perf_counter()
        descriptors = tuple(
            BufferView(
                handle=DeviceTensorView(
                    tensor_handle=source.packed,
                    contract=source.spec.buffer,
                    lease=self.device_lease,
                    completion=completion,
                ),
                shape=tuple(int(dim) for dim in source.packed.shape),
                contract=source.spec.buffer,
            )
            for source in self.sources
        )
        state = StateBatch(
            plan=self.public_plan,
            rows=rows,
            phase=phase,
            descriptors=descriptors,
            lease=self.state_lease,
        )
        return BackendReadResult(
            state=state,
            diagnostics=BackendBatchDiagnostics(
                counters=BackendBatchCounters(
                    state_materializations=1,
                    instrumentation_complete=True,
                ),
                timings=(
                    BackendTiming(
                        "device_state_descriptor", (time.perf_counter() - start) * 1000.0
                    ),
                ),
                completion_event=BackendCompletionEvent(
                    backend_type=self.public_plan.backend_type,
                    placement=self.placement,
                    handle=completion,
                ),
            ),
        )


def _buffer_payload(buffer: Any) -> dict[str, Any]:
    return {
        "row_shape": buffer.row_shape,
        "dtype": buffer.dtype,
        "layout": buffer.layout.value,
        "placement": {
            "memory_space": buffer.placement.memory_space.value,
            "device_type": buffer.placement.device_type,
            "device_index": buffer.placement.device_index,
        },
        "owner": buffer.owner.value,
        "mutability": buffer.mutability.value,
        "lifetime": buffer.lifetime.value,
        "dlpack_exportable": buffer.dlpack_exportable,
        "address_stable": buffer.address_stable,
    }


def _payload_digest(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _device_placement(backend: MjwarpBackend) -> BufferPlacement:
    bridge = backend._ensure_device_bridge()
    index = bridge.qpos.device.index
    if index is None:
        raise BackendBatchContractError("mjwarp device bridge has no CUDA device index")
    return BufferPlacement.device("cuda", int(index))


def _require_device_field_contract(
    backend: MjwarpBackend,
    spec: StateFieldSpec,
    *,
    row_shape: tuple[int, ...],
    frame: ReferenceFrame | None = None,
    unit: PhysicalUnit | None = None,
) -> None:
    expected_placement = _device_placement(backend)
    if spec.buffer.row_shape != row_shape:
        raise BackendBatchContractError(
            f"mjwarp device field {spec.key!r} requires row_shape {row_shape}, "
            f"got {spec.buffer.row_shape}"
        )
    if spec.buffer.dtype != "float32":
        raise BackendBatchContractError(
            f"mjwarp device field {spec.key!r} requires dtype float32, got {spec.buffer.dtype}"
        )
    if spec.buffer.layout is not BufferLayout.C_CONTIGUOUS:
        raise BackendBatchContractError(
            f"mjwarp device field {spec.key!r} requires c_contiguous layout"
        )
    if spec.buffer.placement != expected_placement:
        raise BackendBatchContractError(
            f"mjwarp device field {spec.key!r} must use placement {expected_placement}"
        )
    if spec.buffer.owner is not BufferOwner.BACKEND:
        raise BackendBatchContractError(f"mjwarp device field {spec.key!r} must be backend-owned")
    if spec.buffer.mutability is not BufferMutability.READ_ONLY:
        raise BackendBatchContractError(f"mjwarp device field {spec.key!r} must be read-only")
    if spec.buffer.lifetime is not BufferLifetime.BORROWED_UNTIL_MUTATION:
        raise BackendBatchContractError(
            f"mjwarp device field {spec.key!r} requires borrowed_until_mutation lifetime"
        )
    if not spec.buffer.dlpack_exportable or not spec.buffer.address_stable:
        raise BackendBatchContractError(
            f"mjwarp device field {spec.key!r} requires stable DLPack-exportable storage"
        )
    if frame is not None and spec.frame is not frame:
        raise BackendBatchContractError(
            f"mjwarp device field {spec.key!r} requires frame {frame.value}, got {spec.frame.value}"
        )
    if unit is not None and spec.unit is not unit:
        raise BackendBatchContractError(
            f"mjwarp device field {spec.key!r} requires unit {unit.value}, got {spec.unit.value}"
        )


def _allocate_pack(source: torch.Tensor, spec: StateFieldSpec) -> torch.Tensor:
    expected = (int(source.shape[0]), *spec.buffer.row_shape)
    if tuple(int(dim) for dim in source.shape) != expected:
        raise BackendBatchContractError(
            f"mjwarp device source for {spec.key!r} has shape {tuple(source.shape)}, expected {expected}"
        )
    return torch.empty(expected, dtype=source.dtype, device=source.device)


def _root_source(backend: MjwarpBackend, spec: StateFieldSpec) -> _MjwarpDeviceStateSource:
    bridge = backend._ensure_device_bridge()
    if (backend._root_qpos_dim, backend._root_qvel_dim) != (7, 6):
        raise BackendBatchContractError("mjwarp device root fields require a first free joint")
    if backend._base_body_id is None or spec.identity.entity_ids != (backend._base_body_id,):
        raise BackendBatchContractError(
            "mjwarp device root field must bind the configured base body"
        )
    sources = {
        StateFieldKind.POSITION: (
            bridge.qpos[:, 0:3],
            (3,),
            ReferenceFrame.WORLD,
            PhysicalUnit.METER,
        ),
        StateFieldKind.ORIENTATION: (
            bridge.qpos[:, 3:7],
            (4,),
            ReferenceFrame.WORLD,
            PhysicalUnit.QUATERNION,
        ),
        StateFieldKind.LINEAR_VELOCITY: (
            bridge.qvel[:, 0:3],
            (3,),
            ReferenceFrame.WORLD,
            PhysicalUnit.METER_PER_SECOND,
        ),
        StateFieldKind.ANGULAR_VELOCITY: (
            bridge.qvel[:, 3:6],
            (3,),
            ReferenceFrame.WORLD,
            PhysicalUnit.RADIAN_PER_SECOND,
        ),
    }
    try:
        source, shape, frame, unit = sources[spec.identity.field_kind]
    except KeyError as exc:
        raise BackendBatchContractError(
            f"unsupported mjwarp device root field kind {spec.identity.field_kind.value!r}"
        ) from exc
    _require_device_field_contract(backend, spec, row_shape=shape, frame=frame, unit=unit)
    return _MjwarpDeviceStateSource(
        spec=spec,
        source=source,
        packed=_allocate_pack(source, spec),
        reset_staging=_allocate_pack(source, spec),
    )


def _contiguous_columns(indices: tuple[int, ...], *, context: str) -> slice:
    if not indices:
        raise BackendBatchContractError(f"{context} requires at least one bound coordinate")
    expected = tuple(range(indices[0], indices[0] + len(indices)))
    if indices != expected:
        raise BackendBatchContractError(
            f"{context} requires contiguous IDs for zero-copy device source binding"
        )
    return slice(indices[0], indices[-1] + 1)


def _dof_unit(
    backend: MjwarpBackend, spec: StateFieldSpec, indices: tuple[int, ...]
) -> PhysicalUnit:
    model = backend._cpu_model
    mujoco = backend._mujoco
    identity = spec.identity
    if identity.field_kind is StateFieldKind.POSITION:
        coordinate_types: list[int | None] = [None] * backend._num_dof_pos
        for joint_id in range(int(model.njnt)):
            joint_type = int(model.jnt_type[joint_id])
            if joint_type == int(mujoco.mjtJoint.mjJNT_FREE):
                continue
            start = int(model.jnt_qposadr[joint_id]) - backend._root_qpos_dim
            width = 4 if joint_type == int(mujoco.mjtJoint.mjJNT_BALL) else 1
            for offset in range(width):
                if 0 <= start + offset < len(coordinate_types):
                    coordinate_types[start + offset] = joint_type
        selected = {coordinate_types[index] for index in indices}
        if selected == {int(mujoco.mjtJoint.mjJNT_HINGE)}:
            return PhysicalUnit.RADIAN
        if selected == {int(mujoco.mjtJoint.mjJNT_SLIDE)}:
            return PhysicalUnit.METER
        raise BackendBatchContractError(
            f"mjwarp device DOF position field {spec.key!r} must select homogeneous hinge or slide coordinates"
        )
    if identity.field_kind in {StateFieldKind.ANGULAR_VELOCITY, StateFieldKind.LINEAR_VELOCITY}:
        coordinate_types = [None] * backend._num_dof_vel
        for joint_id in range(int(model.njnt)):
            joint_type = int(model.jnt_type[joint_id])
            if joint_type == int(mujoco.mjtJoint.mjJNT_FREE):
                continue
            start = int(model.jnt_dofadr[joint_id]) - backend._root_qvel_dim
            width = 3 if joint_type == int(mujoco.mjtJoint.mjJNT_BALL) else 1
            for offset in range(width):
                if 0 <= start + offset < len(coordinate_types):
                    coordinate_types[start + offset] = joint_type
        selected = {coordinate_types[index] for index in indices}
        angular = {int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_BALL)}
        if (
            identity.field_kind is StateFieldKind.ANGULAR_VELOCITY
            and selected
            and selected.issubset(angular)
        ):
            return PhysicalUnit.RADIAN_PER_SECOND
        if identity.field_kind is StateFieldKind.LINEAR_VELOCITY and selected == {
            int(mujoco.mjtJoint.mjJNT_SLIDE)
        }:
            return PhysicalUnit.METER_PER_SECOND
    raise BackendBatchContractError(
        f"unsupported mjwarp device DOF field kind {identity.field_kind.value!r}"
    )


def _dof_source(backend: MjwarpBackend, spec: StateFieldSpec) -> _MjwarpDeviceStateSource:
    bridge = backend._ensure_device_bridge()
    indices = tuple(int(index) for index in spec.identity.entity_ids)
    source_slice = _contiguous_columns(indices, context=f"mjwarp device DOF field {spec.key!r}")
    if spec.identity.field_kind is StateFieldKind.POSITION:
        if indices[-1] >= backend._num_dof_pos:
            raise BackendBatchContractError(f"mjwarp device DOF field {spec.key!r} is out of range")
        source = bridge.qpos[
            :,
            backend._root_qpos_dim + source_slice.start : backend._root_qpos_dim
            + source_slice.stop,
        ]
    else:
        if indices[-1] >= backend._num_dof_vel:
            raise BackendBatchContractError(f"mjwarp device DOF field {spec.key!r} is out of range")
        source = bridge.qvel[
            :,
            backend._root_qvel_dim + source_slice.start : backend._root_qvel_dim
            + source_slice.stop,
        ]
    _require_device_field_contract(
        backend,
        spec,
        row_shape=(len(indices),),
        frame=ReferenceFrame.JOINT,
        unit=_dof_unit(backend, spec, indices),
    )
    return _MjwarpDeviceStateSource(
        spec=spec,
        source=source,
        packed=_allocate_pack(source, spec),
        reset_staging=_allocate_pack(source, spec),
    )


def _sensor_frame_unit(
    backend: MjwarpBackend, sensor_id: int
) -> tuple[ReferenceFrame, PhysicalUnit]:
    model = backend._cpu_model
    mujoco = backend._mujoco
    sensor_type = int(model.sensor_type[sensor_id])
    direct = {
        int(mujoco.mjtSensor.mjSENS_VELOCIMETER): (
            ReferenceFrame.SENSOR,
            PhysicalUnit.METER_PER_SECOND,
        ),
        int(mujoco.mjtSensor.mjSENS_GYRO): (ReferenceFrame.SENSOR, PhysicalUnit.RADIAN_PER_SECOND),
        int(mujoco.mjtSensor.mjSENS_CONTACT): (ReferenceFrame.SENSOR, PhysicalUnit.NEWTON),
    }
    if sensor_type in direct:
        return direct[sensor_type]
    frame_sensor_units = {
        int(mujoco.mjtSensor.mjSENS_FRAMEZAXIS): PhysicalUnit.UNITLESS,
        int(mujoco.mjtSensor.mjSENS_FRAMEPOS): PhysicalUnit.METER,
        int(mujoco.mjtSensor.mjSENS_FRAMEQUAT): PhysicalUnit.QUATERNION,
    }
    try:
        unit = frame_sensor_units[sensor_type]
    except KeyError as exc:
        raise BackendBatchContractError(
            f"mjwarp device sensor type {sensor_type} has no typed state contract"
        ) from exc
    reference_id = int(model.sensor_refid[sensor_id])
    if reference_id < 0:
        return ReferenceFrame.WORLD, unit
    reference_type = int(model.sensor_reftype[sensor_id])
    body_types = {int(mujoco.mjtObj.mjOBJ_BODY), int(mujoco.mjtObj.mjOBJ_XBODY)}
    if (
        backend._base_body_id is None
        or reference_type not in body_types
        or reference_id != backend._base_body_id
    ):
        raise BackendBatchContractError("mjwarp device sensor has an unsupported reference frame")
    return ReferenceFrame.BASE, unit


def _sensor_source(backend: MjwarpBackend, spec: StateFieldSpec) -> _MjwarpDeviceStateSource:
    if spec.identity.field_kind is not StateFieldKind.VALUE:
        raise BackendBatchContractError("mjwarp device sensor fields only support value semantics")
    bridge = backend._ensure_device_bridge()
    model = backend._cpu_model
    sensor_ids = tuple(int(value) for value in spec.identity.entity_ids)
    if not sensor_ids:
        raise BackendBatchContractError("mjwarp device sensor field requires at least one ID")
    columns: list[int] = []
    contracts: set[tuple[ReferenceFrame, PhysicalUnit]] = set()
    for sensor_id in sensor_ids:
        if sensor_id < 0 or sensor_id >= int(model.nsensor):
            raise BackendBatchContractError(
                f"mjwarp device sensor field {spec.key!r} is out of range"
            )
        start = int(model.sensor_adr[sensor_id])
        columns.extend(range(start, start + int(model.sensor_dim[sensor_id])))
        contracts.add(_sensor_frame_unit(backend, sensor_id))
    column_slice = _contiguous_columns(
        tuple(columns), context=f"mjwarp device sensor field {spec.key!r}"
    )
    if contracts != {(spec.frame, spec.unit)}:
        raise BackendBatchContractError(
            f"mjwarp device sensor field {spec.key!r} has incompatible frame/unit metadata"
        )
    source = bridge.sensordata[:, column_slice]
    _require_device_field_contract(backend, spec, row_shape=(len(columns),))
    return _MjwarpDeviceStateSource(
        spec=spec,
        source=source,
        packed=_allocate_pack(source, spec),
        reset_staging=_allocate_pack(source, spec),
    )


def _bind_source(backend: MjwarpBackend, spec: StateFieldSpec) -> _MjwarpDeviceStateSource:
    binders = {
        StateEntityKind.ROOT: _root_source,
        StateEntityKind.DOF: _dof_source,
        StateEntityKind.SENSOR: _sensor_source,
    }
    try:
        return binders[spec.identity.entity_kind](backend, spec)
    except KeyError as exc:
        raise BackendBatchContractError(
            f"mjwarp device-resident does not support state entity kind "
            f"{spec.identity.entity_kind.value!r}"
        ) from exc


def _require_device_control_contract(
    backend: MjwarpBackend, requirements: BackendIORequirements
) -> None:
    control = requirements.control
    placement = _device_placement(backend)
    if control.buffer.row_shape != (backend._nu,):
        raise BackendBatchContractError(
            f"mjwarp device control requires row_shape {(backend._nu,)}, got {control.buffer.row_shape}"
        )
    if control.buffer.dtype != "float32" or control.buffer.layout is not BufferLayout.C_CONTIGUOUS:
        raise BackendBatchContractError("mjwarp device control requires C-contiguous float32")
    if control.buffer.placement != placement:
        raise BackendBatchContractError(
            "mjwarp device control placement must match backend CUDA device"
        )
    if control.buffer.owner is not BufferOwner.RUNNER:
        raise BackendBatchContractError("mjwarp device control must be runner-owned")
    if control.buffer.mutability is not BufferMutability.READ_ONLY:
        raise BackendBatchContractError("mjwarp device control must be read-only to the backend")
    if control.buffer.lifetime is not BufferLifetime.UNTIL_STEP_COMPLETE:
        raise BackendBatchContractError(
            "mjwarp device control requires until_step_complete lifetime"
        )
    if not control.buffer.dlpack_exportable or not control.buffer.address_stable:
        raise BackendBatchContractError(
            "mjwarp device control requires stable DLPack-exportable storage"
        )


def _binding_payload(backend: MjwarpBackend, requirements: BackendIORequirements) -> dict[str, Any]:
    bridge = backend._ensure_device_bridge()
    model = backend._cpu_model
    return {
        "contract_version": requirements.contract_version,
        "backend_type": backend.backend_type,
        "execution_profile": requirements.execution_profile.value,
        "cuda_device": str(bridge.qpos.device),
        "model_dims": {
            "nq": int(model.nq),
            "nv": int(model.nv),
            "nu": int(model.nu),
            "nsensor": int(model.nsensor),
            "nsensordata": int(model.nsensordata),
        },
        "fields": tuple(
            {
                "semantic_key": spec.key,
                "entity_kind": spec.identity.entity_kind.value,
                "field_kind": spec.identity.field_kind.value,
                "entity_ids": spec.identity.entity_ids,
                "frame": spec.frame.value,
                "unit": spec.unit.value,
                "buffer": _buffer_payload(spec.buffer),
            }
            for spec in requirements.state_fields
        ),
        "control": {
            "semantic_key": requirements.control.semantic_key,
            "buffer": _buffer_payload(requirements.control.buffer),
            "cadence": requirements.control.physics_substeps_per_control,
        },
        "hot_path_budget": (
            None
            if requirements.hot_path_budget is None
            else dict(requirements.hot_path_budget.items())
        ),
        "reset_hot_path_budget": (
            None
            if requirements.reset_hot_path_budget is None
            else dict(requirements.reset_hot_path_budget.items())
        ),
    }


def bind_mjwarp_device_batch(
    backend: MjwarpBackend,
    requirements: BackendIORequirements,
    *,
    backend_instance_id: str,
) -> MjwarpDeviceBatchPlan:
    """Cold-bind a real CUDA device-resident state/control plan."""

    if not isinstance(requirements, BackendIORequirements):
        raise BackendBatchContractError(
            "mjwarp device batch requirements must be BackendIORequirements"
        )
    if requirements.execution_profile is not ExecutionProfile.DEVICE_RESIDENT:
        raise BackendBatchContractError(
            "mjwarp device batch requires execution_profile=device_resident"
        )
    if backend._pre_step_control_fn is not None:
        raise BackendBatchContractError("mjwarp device batches reject host pre-step callbacks")
    _require_device_control_contract(backend, requirements)
    sources = tuple(_bind_source(backend, spec) for spec in requirements.state_fields)
    payload = _binding_payload(backend, requirements)
    state_payload = {key: value for key, value in payload.items() if key != "control"}
    state = BoundStatePlan(
        backend_type=backend.backend_type,
        backend_instance_id=backend_instance_id,
        num_envs=backend.num_envs,
        fields=requirements.state_fields,
        execution_profile=ExecutionProfile.DEVICE_RESIDENT,
        fingerprint=f"{_STATE_FINGERPRINT_PREFIX}:{_payload_digest(state_payload)}",
    )
    public_plan = BoundBackendPlan(
        state=state,
        control=requirements.control,
        execution_profile=ExecutionProfile.DEVICE_RESIDENT,
        fingerprint=f"{_PLAN_FINGERPRINT_PREFIX}:{_payload_digest(payload)}",
        hot_path_budget=requirements.hot_path_budget,
        reset_hot_path_budget=requirements.reset_hot_path_budget,
        contract_version=BACKEND_BATCH_CONTRACT_VERSION,
    )
    return MjwarpDeviceBatchPlan(
        public_plan=public_plan,
        sources=sources,
        state_lease=StateBatchLease(backend_instance_id),
        device_lease=DeviceBufferLease(backend_instance_id),
        placement=_device_placement(backend),
        owner_id=backend_instance_id,
    )


__all__ = ["MjwarpDeviceBatchPlan", "bind_mjwarp_device_batch"]
