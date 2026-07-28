"""Backend-owned typed ``host_numpy`` adapter for independent ``mjwarp``.

The adapter binds semantic state fields to the stable host cache allocated by
``MjwarpBackend``.  It deliberately does not import or share any runtime object
from the sibling MuJoCo backend.  Selector/model inspection happens while a
plan is bound; materialization only performs fixed-index copies into
cold-allocated scratch buffers.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from ..batch import (
    BACKEND_BATCH_CONTRACT_VERSION,
    BackendBatchContractError,
    BackendBatchCounters,
    BackendBatchDiagnostics,
    BackendIORequirements,
    BackendReadResult,
    BackendTiming,
    BoundBackendPlan,
    BoundStatePlan,
    BufferLayout,
    BufferView,
    ExecutionProfile,
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
from ..telemetry import BackendTransferCounters

if TYPE_CHECKING:
    from .backend import MjwarpBackend

_STATE_FINGERPRINT_PREFIX = "mjwarp-host-state-v1"
_PLAN_FINGERPRINT_PREFIX = "mjwarp-host-batch-v1"
_HOST_CACHE_DOWNLOAD_ALLOCATIONS = 3


def _readonly_view(array: np.ndarray) -> np.ndarray:
    view = array.view()
    view.flags.writeable = False
    return view


@dataclass
class _MjwarpStateSource:
    """One bound cache source plus fixed all-row and selected-row storage."""

    spec: StateFieldSpec
    source: np.ndarray
    gather_indices: np.ndarray | None = None
    gather_axis: int = 1
    _full: np.ndarray = field(init=False, repr=False)
    _copy_source: bool = field(init=False, repr=False)
    _all_view: np.ndarray = field(init=False, repr=False)
    _selected: np.ndarray = field(init=False, repr=False)
    _selected_views: dict[int, np.ndarray] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        num_envs = int(self.source.shape[0])
        expected_shape = (num_envs, *self.spec.buffer.row_shape)
        expected_dtype = np.dtype(self.spec.buffer.dtype)
        if self.source.dtype != expected_dtype:
            raise BackendBatchContractError(
                f"mjwarp source for {self.spec.key!r} has dtype {self.source.dtype.name}, "
                f"expected {expected_dtype.name}"
            )
        if self.gather_indices is None:
            if self.source.shape != expected_shape:
                raise BackendBatchContractError(
                    f"mjwarp source for {self.spec.key!r} has shape {self.source.shape}, "
                    f"expected {expected_shape}"
                )
            self._copy_source = not self.source.flags.c_contiguous
            self._full = (
                np.empty(expected_shape, dtype=expected_dtype) if self._copy_source else self.source
            )
        else:
            self._copy_source = False
            self._full = np.empty(expected_shape, dtype=expected_dtype)
        self._all_view = _readonly_view(self._full)
        self._selected = np.empty(expected_shape, dtype=expected_dtype)

    def materialize(
        self,
        rows: RowSelection,
        row_ids: np.ndarray | None,
    ) -> BufferView:
        if self.gather_indices is not None:
            np.take(
                self.source,
                self.gather_indices,
                axis=self.gather_axis,
                out=self._full,
            )
        elif self._copy_source:
            np.copyto(self._full, self.source)

        if rows.is_all:
            handle = self._all_view
        else:
            assert row_ids is not None
            np.take(self._full, row_ids, axis=0, out=self._selected[: rows.count])
            handle = self._selected_views.get(rows.count)
            if handle is None:
                handle = _readonly_view(self._selected[: rows.count])
                self._selected_views[rows.count] = handle
        return BufferView(
            handle=handle,
            shape=(rows.count, *self.spec.buffer.row_shape),
            contract=self.spec.buffer,
        )


@dataclass
class MjwarpHostBatchPlan:
    """Runtime companion for one public, immutable mjwarp batch plan."""

    public_plan: BoundBackendPlan
    sources: tuple[_MjwarpStateSource, ...]
    lease: StateBatchLease
    _row_ids: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._row_ids = np.empty(self.public_plan.num_envs, dtype=np.intp)

    @property
    def step_allocations(self) -> int:
        """Host arrays returned by three Warp ``numpy()`` downloads per barrier."""
        return _HOST_CACHE_DOWNLOAD_ALLOCATIONS

    def materialize(self, rows: RowSelection, phase: StateBatchPhase) -> BackendReadResult:
        self.lease.invalidate()
        start = time.perf_counter()
        row_ids = None
        if not rows.is_all:
            assert rows.indices is not None
            for offset, row_id in enumerate(rows.indices):
                self._row_ids[offset] = row_id
            row_ids = self._row_ids[: rows.count]
        descriptors = tuple(source.materialize(rows, row_ids) for source in self.sources)
        state = StateBatch(
            plan=self.public_plan,
            rows=rows,
            phase=phase,
            descriptors=descriptors,
            lease=self.lease,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return BackendReadResult(
            state=state,
            diagnostics=BackendBatchDiagnostics(
                counters=BackendBatchCounters(
                    state_materializations=1,
                    instrumentation_complete=True,
                ),
                timings=(BackendTiming("state_materialize", elapsed_ms),),
            ),
        )


def transfer_delta_to_batch_counters(
    transfer: BackendTransferCounters,
    *,
    allocations: int,
    state_materializations: int,
) -> BackendBatchCounters:
    """Translate public transfer telemetry into one managed-barrier diagnostic."""
    return BackendBatchCounters(
        host_to_device_transfers=transfer.host_to_device_transfers,
        device_to_host_transfers=transfer.device_to_host_transfers,
        host_to_device_bytes=transfer.host_to_device_bytes,
        device_to_host_bytes=transfer.device_to_host_bytes,
        global_synchronizations=transfer.global_synchronizations,
        allocations=allocations,
        state_materializations=state_materializations,
        instrumentation_complete=True,
    )


def _require_field_contract(
    backend: MjwarpBackend,
    spec: StateFieldSpec,
    *,
    row_shape: tuple[int, ...],
    frame: ReferenceFrame | None = None,
    unit: PhysicalUnit | None = None,
) -> None:
    expected_dtype = np.dtype(backend._qpos_cache.dtype).name
    if spec.buffer.row_shape != row_shape:
        raise BackendBatchContractError(
            f"mjwarp field {spec.key!r} requires row_shape {row_shape}, got {spec.buffer.row_shape}"
        )
    if spec.buffer.dtype != expected_dtype:
        raise BackendBatchContractError(
            f"mjwarp field {spec.key!r} requires dtype {expected_dtype}, got {spec.buffer.dtype}"
        )
    if spec.buffer.layout is not BufferLayout.C_CONTIGUOUS:
        raise BackendBatchContractError(
            f"mjwarp host field {spec.key!r} requires c_contiguous layout"
        )
    if frame is not None and spec.frame is not frame:
        raise BackendBatchContractError(
            f"mjwarp field {spec.key!r} requires frame {frame.value}, got {spec.frame.value}"
        )
    if unit is not None and spec.unit is not unit:
        raise BackendBatchContractError(
            f"mjwarp field {spec.key!r} requires unit {unit.value}, got {spec.unit.value}"
        )


def _bind_root_source(backend: MjwarpBackend, spec: StateFieldSpec) -> _MjwarpStateSource:
    if (backend._root_qpos_dim, backend._root_qvel_dim) != (7, 6):
        raise BackendBatchContractError("mjwarp typed root fields require a first free joint")
    if backend._base_body_id is None:
        raise BackendBatchContractError(
            "mjwarp typed root fields require base_name during backend construction"
        )
    if spec.identity.entity_ids != (backend._base_body_id,):
        raise BackendBatchContractError(
            f"mjwarp root field {spec.key!r} must bind base body id {backend._base_body_id}"
        )
    sources = {
        StateFieldKind.POSITION: (
            backend._qpos_cache[:, 0:3],
            (3,),
            ReferenceFrame.WORLD,
            PhysicalUnit.METER,
        ),
        StateFieldKind.ORIENTATION: (
            backend._qpos_cache[:, 3:7],
            (4,),
            ReferenceFrame.WORLD,
            PhysicalUnit.QUATERNION,
        ),
        StateFieldKind.LINEAR_VELOCITY: (
            backend._qvel_cache[:, 0:3],
            (3,),
            ReferenceFrame.WORLD,
            PhysicalUnit.METER_PER_SECOND,
        ),
        StateFieldKind.ANGULAR_VELOCITY: (
            backend._qvel_cache[:, 3:6],
            (3,),
            ReferenceFrame.WORLD,
            PhysicalUnit.RADIAN_PER_SECOND,
        ),
    }
    try:
        source, row_shape, frame, unit = sources[spec.identity.field_kind]
    except KeyError as exc:
        raise BackendBatchContractError(
            f"unsupported mjwarp root field kind {spec.identity.field_kind.value!r}"
        ) from exc
    _require_field_contract(
        backend,
        spec,
        row_shape=row_shape,
        frame=frame,
        unit=unit,
    )
    return _MjwarpStateSource(spec=spec, source=source)


def _bind_dof_source(backend: MjwarpBackend, spec: StateFieldSpec) -> _MjwarpStateSource:
    identity = spec.identity
    indices = np.asarray(identity.entity_ids, dtype=np.intp)
    model = backend._cpu_model
    mujoco = backend._mujoco
    if identity.field_kind is StateFieldKind.POSITION:
        source = backend._qpos_cache[:, backend._root_qpos_dim :]
        if np.any(indices >= source.shape[1]):
            raise BackendBatchContractError(
                f"mjwarp DOF field {spec.key!r} contains an out-of-range id"
            )
        coordinate_types: list[int | None] = [None] * source.shape[1]
        for joint_id in range(int(model.njnt)):
            joint_type = int(model.jnt_type[joint_id])
            if joint_type == int(mujoco.mjtJoint.mjJNT_FREE):
                continue
            start = int(model.jnt_qposadr[joint_id]) - backend._root_qpos_dim
            width = 4 if joint_type == int(mujoco.mjtJoint.mjJNT_BALL) else 1
            for offset in range(width):
                if 0 <= start + offset < len(coordinate_types):
                    coordinate_types[start + offset] = joint_type
        selected_types = {coordinate_types[int(index)] for index in indices}
        if selected_types == {int(mujoco.mjtJoint.mjJNT_HINGE)}:
            unit = PhysicalUnit.RADIAN
        elif selected_types == {int(mujoco.mjtJoint.mjJNT_SLIDE)}:
            unit = PhysicalUnit.METER
        else:
            raise BackendBatchContractError(
                f"mjwarp DOF position field {spec.key!r} must select homogeneous "
                "hinge or slide coordinates"
            )
    elif identity.field_kind in {
        StateFieldKind.ANGULAR_VELOCITY,
        StateFieldKind.LINEAR_VELOCITY,
    }:
        source = backend._qvel_cache[:, backend._root_qvel_dim :]
        if np.any(indices >= source.shape[1]):
            raise BackendBatchContractError(
                f"mjwarp DOF field {spec.key!r} contains an out-of-range id"
            )
        coordinate_types = [None] * source.shape[1]
        for joint_id in range(int(model.njnt)):
            joint_type = int(model.jnt_type[joint_id])
            if joint_type == int(mujoco.mjtJoint.mjJNT_FREE):
                continue
            start = int(model.jnt_dofadr[joint_id]) - backend._root_qvel_dim
            width = 3 if joint_type == int(mujoco.mjtJoint.mjJNT_BALL) else 1
            for offset in range(width):
                if 0 <= start + offset < len(coordinate_types):
                    coordinate_types[start + offset] = joint_type
        selected_types = {coordinate_types[int(index)] for index in indices}
        angular_types = {
            int(mujoco.mjtJoint.mjJNT_HINGE),
            int(mujoco.mjtJoint.mjJNT_BALL),
        }
        if identity.field_kind is StateFieldKind.ANGULAR_VELOCITY:
            if not selected_types or not selected_types.issubset(angular_types):
                raise BackendBatchContractError(
                    f"mjwarp angular DOF field {spec.key!r} contains a non-angular coordinate"
                )
            unit = PhysicalUnit.RADIAN_PER_SECOND
        else:
            if selected_types != {int(mujoco.mjtJoint.mjJNT_SLIDE)}:
                raise BackendBatchContractError(
                    f"mjwarp linear DOF field {spec.key!r} contains a non-linear coordinate"
                )
            unit = PhysicalUnit.METER_PER_SECOND
    else:
        raise BackendBatchContractError(
            f"unsupported mjwarp DOF field kind {identity.field_kind.value!r}"
        )
    _require_field_contract(
        backend,
        spec,
        row_shape=(len(indices),),
        frame=ReferenceFrame.JOINT,
        unit=unit,
    )
    return _MjwarpStateSource(
        spec=spec,
        source=source,
        gather_indices=indices,
    )


def _bind_sensor_source(backend: MjwarpBackend, spec: StateFieldSpec) -> _MjwarpStateSource:
    identity = spec.identity
    if identity.field_kind is not StateFieldKind.VALUE:
        raise BackendBatchContractError("mjwarp sensor fields only support value semantics")
    model = backend._cpu_model
    mujoco = backend._mujoco
    sensor_contracts = {
        int(mujoco.mjtSensor.mjSENS_VELOCIMETER): (
            ReferenceFrame.SENSOR,
            PhysicalUnit.METER_PER_SECOND,
        ),
        int(mujoco.mjtSensor.mjSENS_GYRO): (
            ReferenceFrame.SENSOR,
            PhysicalUnit.RADIAN_PER_SECOND,
        ),
        int(mujoco.mjtSensor.mjSENS_CONTACT): (
            ReferenceFrame.SENSOR,
            PhysicalUnit.NEWTON,
        ),
    }
    frame_sensor_units = {
        int(mujoco.mjtSensor.mjSENS_FRAMEZAXIS): PhysicalUnit.UNITLESS,
        int(mujoco.mjtSensor.mjSENS_FRAMEPOS): PhysicalUnit.METER,
        int(mujoco.mjtSensor.mjSENS_FRAMEQUAT): PhysicalUnit.QUATERNION,
    }
    expected_contracts: set[tuple[ReferenceFrame, PhysicalUnit]] = set()
    sensor_columns: list[int] = []
    for sensor_id in identity.entity_ids:
        if sensor_id >= int(model.nsensor):
            raise BackendBatchContractError(
                f"mjwarp sensor field {spec.key!r} contains an out-of-range id"
            )
        start = int(model.sensor_adr[sensor_id])
        dim = int(model.sensor_dim[sensor_id])
        sensor_columns.extend(range(start, start + dim))
        sensor_type = int(model.sensor_type[sensor_id])
        if sensor_type in frame_sensor_units:
            reference_id = int(model.sensor_refid[sensor_id])
            if reference_id < 0:
                reference_frame = ReferenceFrame.WORLD
            else:
                reference_type = int(model.sensor_reftype[sensor_id])
                body_types = {
                    int(mujoco.mjtObj.mjOBJ_BODY),
                    int(mujoco.mjtObj.mjOBJ_XBODY),
                }
                if (
                    backend._base_body_id is None
                    or reference_type not in body_types
                    or reference_id != backend._base_body_id
                ):
                    raise BackendBatchContractError(
                        f"mjwarp sensor field {spec.key!r} has an unsupported reference frame"
                    )
                reference_frame = ReferenceFrame.BASE
            expected_contracts.add((reference_frame, frame_sensor_units[sensor_type]))
            continue
        try:
            expected_contracts.add(sensor_contracts[sensor_type])
        except KeyError as exc:
            raise BackendBatchContractError(
                f"mjwarp sensor type {sensor_type} has no typed state contract"
            ) from exc
    if expected_contracts != {(spec.frame, spec.unit)}:
        expected = ", ".join(
            f"{frame.value}/{unit.value}"
            for frame, unit in sorted(
                expected_contracts,
                key=lambda item: (item[0].value, item[1].value),
            )
        )
        raise BackendBatchContractError(
            f"mjwarp sensor field {spec.key!r} requires homogeneous frame/unit {expected}"
        )
    indices = np.asarray(sensor_columns, dtype=np.intp)
    _require_field_contract(backend, spec, row_shape=(len(indices),))
    return _MjwarpStateSource(
        spec=spec,
        source=backend._sensor_cache,
        gather_indices=indices,
    )


def _bind_state_source(backend: MjwarpBackend, spec: StateFieldSpec) -> _MjwarpStateSource:
    binders = {
        StateEntityKind.ROOT: _bind_root_source,
        StateEntityKind.DOF: _bind_dof_source,
        StateEntityKind.SENSOR: _bind_sensor_source,
    }
    try:
        binder = binders[spec.identity.entity_kind]
    except KeyError as exc:
        raise BackendBatchContractError(
            f"mjwarp host_numpy does not support state entity kind "
            f"{spec.identity.entity_kind.value!r}"
        ) from exc
    return binder(backend, spec)


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


def _names(backend: MjwarpBackend, object_type: Any, count: int) -> tuple[str, ...]:
    return tuple(
        backend._mujoco.mj_id2name(backend._cpu_model, object_type, index) or f"#{index}"
        for index in range(count)
    )


def _binding_payloads(
    backend: MjwarpBackend,
    requirements: BackendIORequirements,
) -> tuple[dict[str, Any], dict[str, Any]]:
    model = backend._cpu_model
    mujoco = backend._mujoco
    fields = tuple(
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
    )
    state_payload = {
        "contract_version": requirements.contract_version,
        "backend_type": backend.backend_type,
        "model_dims": {
            "nq": int(model.nq),
            "nv": int(model.nv),
            "nu": int(model.nu),
            "nsensor": int(model.nsensor),
            "nsensordata": int(model.nsensordata),
        },
        "model_semantics": {
            "base_body_id": backend._base_body_id,
            "body_names": _names(backend, mujoco.mjtObj.mjOBJ_BODY, int(model.nbody)),
            "joint_names": _names(backend, mujoco.mjtObj.mjOBJ_JOINT, int(model.njnt)),
            "actuator_names": _names(backend, mujoco.mjtObj.mjOBJ_ACTUATOR, int(model.nu)),
            "sensor_names": _names(backend, mujoco.mjtObj.mjOBJ_SENSOR, int(model.nsensor)),
            "joint_type": tuple(int(value) for value in model.jnt_type),
            "joint_qposadr": tuple(int(value) for value in model.jnt_qposadr),
            "joint_dofadr": tuple(int(value) for value in model.jnt_dofadr),
            "sensor_type": tuple(int(value) for value in model.sensor_type),
            "sensor_dim": tuple(int(value) for value in model.sensor_dim),
            "sensor_adr": tuple(int(value) for value in model.sensor_adr),
            "sensor_objtype": tuple(int(value) for value in model.sensor_objtype),
            "sensor_objid": tuple(int(value) for value in model.sensor_objid),
            "sensor_reftype": tuple(int(value) for value in model.sensor_reftype),
            "sensor_refid": tuple(int(value) for value in model.sensor_refid),
            "timestep": float(model.opt.timestep),
            "dtype": np.dtype(backend._qpos_cache.dtype).name,
        },
        "execution_profile": requirements.execution_profile.value,
        "fields": fields,
    }
    plan_payload = {
        "state": state_payload,
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
    }
    return state_payload, plan_payload


def _payload_digest(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _worst_step_counters(backend: MjwarpBackend) -> BackendBatchCounters:
    return BackendBatchCounters(
        host_to_device_transfers=1,
        device_to_host_transfers=3,
        host_to_device_bytes=int(backend._ctrl_staging.nbytes),
        device_to_host_bytes=int(
            backend._qpos_cache.nbytes + backend._qvel_cache.nbytes + backend._sensor_cache.nbytes
        ),
        global_synchronizations=1,
        allocations=_HOST_CACHE_DOWNLOAD_ALLOCATIONS,
        state_materializations=1,
        instrumentation_complete=True,
    )


def bind_mjwarp_host_batch(
    backend: MjwarpBackend,
    requirements: BackendIORequirements,
    *,
    backend_instance_id: str,
) -> MjwarpHostBatchPlan:
    """Bind one typed plan without touching device runtime state."""
    if not isinstance(requirements, BackendIORequirements):
        raise BackendBatchContractError("mjwarp batch requirements must be BackendIORequirements")
    if requirements.execution_profile is not ExecutionProfile.HOST_NUMPY:
        raise BackendBatchContractError("mjwarp reference batches only support host_numpy")
    if backend._pre_step_control_fn is not None:
        raise BackendBatchContractError(
            "mjwarp managed host batches do not support pre-step control callbacks"
        )
    control = requirements.control
    expected_dtype = np.dtype(backend._ctrl_staging.dtype).name
    if control.buffer.row_shape != (backend._nu,):
        raise BackendBatchContractError(
            f"mjwarp control requires row_shape {(backend._nu,)}, got {control.buffer.row_shape}"
        )
    if control.buffer.dtype != expected_dtype:
        raise BackendBatchContractError(
            f"mjwarp control requires dtype {expected_dtype}, got {control.buffer.dtype}"
        )
    if control.buffer.layout is not BufferLayout.C_CONTIGUOUS:
        raise BackendBatchContractError("mjwarp host control requires c_contiguous layout")
    if requirements.hot_path_budget is not None:
        _worst_step_counters(backend).require_within(requirements.hot_path_budget)

    sources = tuple(_bind_state_source(backend, spec) for spec in requirements.state_fields)
    state_payload, plan_payload = _binding_payloads(backend, requirements)
    state_plan = BoundStatePlan(
        backend_type=backend.backend_type,
        backend_instance_id=backend_instance_id,
        num_envs=backend.num_envs,
        fields=requirements.state_fields,
        execution_profile=requirements.execution_profile,
        fingerprint=f"{_STATE_FINGERPRINT_PREFIX}:{_payload_digest(state_payload)}",
    )
    public_plan = BoundBackendPlan(
        state=state_plan,
        control=control,
        execution_profile=requirements.execution_profile,
        fingerprint=f"{_PLAN_FINGERPRINT_PREFIX}:{_payload_digest(plan_payload)}",
        hot_path_budget=requirements.hot_path_budget,
        contract_version=BACKEND_BATCH_CONTRACT_VERSION,
    )
    return MjwarpHostBatchPlan(
        public_plan=public_plan,
        sources=sources,
        lease=StateBatchLease(backend_instance_id),
    )


__all__ = [
    "MjwarpHostBatchPlan",
    "bind_mjwarp_host_batch",
    "transfer_delta_to_batch_counters",
]
