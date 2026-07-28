"""MuJoCo-owned host reference implementation of the typed batch contract."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import mujoco
import numpy as np

from unilab.base.backend.batch import (
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

if TYPE_CHECKING:
    from .backend import MuJoCoBackend

_STATE_FINGERPRINT_PREFIX = "mujoco-host-state-v1"
_PLAN_FINGERPRINT_PREFIX = "mujoco-host-batch-v1"


def _readonly_view(array: np.ndarray) -> np.ndarray:
    view = array.view()
    view.flags.writeable = False
    return view


@dataclass
class _MuJoCoStateSource:
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
        if self.gather_indices is None:
            if self.source.shape != expected_shape:
                raise BackendBatchContractError(
                    f"MuJoCo source for {self.spec.key!r} has shape {self.source.shape}, "
                    f"expected {expected_shape}"
                )
            self._copy_source = not self.source.flags.c_contiguous
            self._full = (
                np.empty(expected_shape, dtype=self.spec.buffer.dtype)
                if self._copy_source
                else self.source
            )
        else:
            self._copy_source = False
            self._full = np.empty(expected_shape, dtype=self.spec.buffer.dtype)
        self._all_view = _readonly_view(self._full)
        self._selected = np.empty(expected_shape, dtype=self.spec.buffer.dtype)

    def materialize(self, rows: RowSelection) -> BufferView:
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
            assert rows.indices is not None
            row_ids = np.asarray(rows.indices, dtype=np.intp)
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
class _MuJoCoHostBatchPlan:
    public_plan: BoundBackendPlan
    sources: tuple[_MuJoCoStateSource, ...]
    lease: StateBatchLease

    def materialize(self, rows: RowSelection, phase: StateBatchPhase) -> BackendReadResult:
        self.lease.invalidate()
        start = time.perf_counter()
        descriptors = tuple(source.materialize(rows) for source in self.sources)
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
                    instrumentation_complete=False,
                ),
                timings=(BackendTiming("state_materialize", elapsed_ms),),
            ),
        )


def _require_field_contract(
    backend: MuJoCoBackend,
    spec: StateFieldSpec,
    *,
    row_shape: tuple[int, ...],
    frame: ReferenceFrame | None = None,
    unit: PhysicalUnit | None = None,
) -> None:
    expected_dtype = np.dtype(backend._np_dtype).name
    if spec.buffer.row_shape != row_shape:
        raise BackendBatchContractError(
            f"MuJoCo field {spec.key!r} requires row_shape {row_shape}, got {spec.buffer.row_shape}"
        )
    if spec.buffer.dtype != expected_dtype:
        raise BackendBatchContractError(
            f"MuJoCo field {spec.key!r} requires dtype {expected_dtype}, got {spec.buffer.dtype}"
        )
    if spec.buffer.layout is not BufferLayout.C_CONTIGUOUS:
        raise BackendBatchContractError(
            f"MuJoCo host field {spec.key!r} requires c_contiguous layout"
        )
    if frame is not None and spec.frame is not frame:
        raise BackendBatchContractError(
            f"MuJoCo field {spec.key!r} requires frame {frame.value}, got {spec.frame.value}"
        )
    if unit is not None and spec.unit is not unit:
        raise BackendBatchContractError(
            f"MuJoCo field {spec.key!r} requires unit {unit.value}, got {spec.unit.value}"
        )


def _bind_root_source(backend: MuJoCoBackend, spec: StateFieldSpec) -> _MuJoCoStateSource:
    if (backend._root_qpos_dim, backend._root_qvel_dim) != (7, 6):
        raise BackendBatchContractError(
            "MuJoCo typed root fields require a free-joint root cache; "
            "fixed-base body state must use a bound body field"
        )
    identity = spec.identity
    if identity.entity_ids != (backend._base_body_id,):
        raise BackendBatchContractError(
            f"MuJoCo root field {spec.key!r} must bind base body id {backend._base_body_id}"
        )
    sources = {
        StateFieldKind.POSITION: (
            backend._base_pos_view,
            (3,),
            ReferenceFrame.WORLD,
            PhysicalUnit.METER,
        ),
        StateFieldKind.ORIENTATION: (
            backend._base_quat_view,
            (4,),
            ReferenceFrame.WORLD,
            PhysicalUnit.QUATERNION,
        ),
        StateFieldKind.LINEAR_VELOCITY: (
            backend._base_lin_vel_view,
            (3,),
            ReferenceFrame.WORLD,
            PhysicalUnit.METER_PER_SECOND,
        ),
        StateFieldKind.ANGULAR_VELOCITY: (
            backend._base_ang_vel_view,
            (3,),
            ReferenceFrame.WORLD,
            PhysicalUnit.RADIAN_PER_SECOND,
        ),
    }
    try:
        source, row_shape, frame, unit = sources[identity.field_kind]
    except KeyError as exc:
        raise BackendBatchContractError(
            f"unsupported MuJoCo root field kind {identity.field_kind.value!r}"
        ) from exc
    _require_field_contract(
        backend,
        spec,
        row_shape=row_shape,
        frame=frame,
        unit=unit,
    )
    return _MuJoCoStateSource(spec=spec, source=source)


def _bind_dof_source(backend: MuJoCoBackend, spec: StateFieldSpec) -> _MuJoCoStateSource:
    identity = spec.identity
    indices = np.asarray(identity.entity_ids, dtype=np.intp)
    if identity.field_kind is StateFieldKind.POSITION:
        source = backend._dof_pos_view
        if np.any(indices >= source.shape[1]):
            raise BackendBatchContractError(
                f"MuJoCo DOF field {spec.key!r} contains an out-of-range id"
            )
        position_types: list[int | None] = [None] * source.shape[1]
        for joint_id in range(int(backend._model.njnt)):
            joint_type = int(backend._model.jnt_type[joint_id])
            if joint_type == int(mujoco.mjtJoint.mjJNT_FREE):
                continue
            start = int(backend._model.jnt_qposadr[joint_id]) - backend._root_qpos_dim
            width = 4 if joint_type == int(mujoco.mjtJoint.mjJNT_BALL) else 1
            for offset in range(width):
                if 0 <= start + offset < len(position_types):
                    position_types[start + offset] = joint_type
        selected_types = {position_types[int(index)] for index in indices}
        if selected_types == {int(mujoco.mjtJoint.mjJNT_HINGE)}:
            unit = PhysicalUnit.RADIAN
        elif selected_types == {int(mujoco.mjtJoint.mjJNT_SLIDE)}:
            unit = PhysicalUnit.METER
        else:
            raise BackendBatchContractError(
                f"MuJoCo DOF position field {spec.key!r} must select homogeneous "
                "hinge or slide coordinates; ball/mixed fields require a separate semantic"
            )
    elif identity.field_kind in {
        StateFieldKind.ANGULAR_VELOCITY,
        StateFieldKind.LINEAR_VELOCITY,
    }:
        source = backend._dof_vel_view
        if np.any(indices >= source.shape[1]):
            raise BackendBatchContractError(
                f"MuJoCo DOF field {spec.key!r} contains an out-of-range id"
            )
        velocity_types: list[int | None] = [None] * source.shape[1]
        for joint_id in range(int(backend._model.njnt)):
            joint_type = int(backend._model.jnt_type[joint_id])
            if joint_type == int(mujoco.mjtJoint.mjJNT_FREE):
                continue
            start = int(backend._model.jnt_dofadr[joint_id]) - backend._root_qvel_dim
            width = 3 if joint_type == int(mujoco.mjtJoint.mjJNT_BALL) else 1
            for offset in range(width):
                if 0 <= start + offset < len(velocity_types):
                    velocity_types[start + offset] = joint_type
        selected_types = {velocity_types[int(index)] for index in indices}
        angular_types = {
            int(mujoco.mjtJoint.mjJNT_HINGE),
            int(mujoco.mjtJoint.mjJNT_BALL),
        }
        if identity.field_kind is StateFieldKind.ANGULAR_VELOCITY:
            if not selected_types or not selected_types.issubset(angular_types):
                raise BackendBatchContractError(
                    f"MuJoCo angular DOF field {spec.key!r} contains a non-angular coordinate"
                )
            unit = PhysicalUnit.RADIAN_PER_SECOND
        else:
            if selected_types != {int(mujoco.mjtJoint.mjJNT_SLIDE)}:
                raise BackendBatchContractError(
                    f"MuJoCo linear DOF field {spec.key!r} contains a non-linear coordinate"
                )
            unit = PhysicalUnit.METER_PER_SECOND
    else:
        raise BackendBatchContractError(
            f"unsupported MuJoCo DOF field kind {identity.field_kind.value!r}"
        )
    _require_field_contract(
        backend,
        spec,
        row_shape=(len(indices),),
        frame=ReferenceFrame.JOINT,
        unit=unit,
    )
    return _MuJoCoStateSource(
        spec=spec,
        source=source,
        gather_indices=indices,
    )


def _bind_sensor_source(backend: MuJoCoBackend, spec: StateFieldSpec) -> _MuJoCoStateSource:
    identity = spec.identity
    if identity.field_kind is not StateFieldKind.VALUE:
        raise BackendBatchContractError("MuJoCo sensor fields only support value semantics")
    sensor_columns: list[int] = []
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
    for sensor_id in identity.entity_ids:
        if sensor_id >= int(backend._model.nsensor):
            raise BackendBatchContractError(
                f"MuJoCo sensor field {spec.key!r} contains an out-of-range id"
            )
        start = int(backend._model.sensor_adr[sensor_id])
        dim = int(backend._model.sensor_dim[sensor_id])
        sensor_columns.extend(range(start, start + dim))
        sensor_type = int(backend._model.sensor_type[sensor_id])
        if sensor_type in frame_sensor_units:
            reference_id = int(backend._model.sensor_refid[sensor_id])
            if reference_id < 0:
                reference_frame = ReferenceFrame.WORLD
            else:
                reference_type = int(backend._model.sensor_reftype[sensor_id])
                body_reference_types = {
                    int(mujoco.mjtObj.mjOBJ_BODY),
                    int(mujoco.mjtObj.mjOBJ_XBODY),
                }
                if (
                    reference_type not in body_reference_types
                    or reference_id != backend._base_body_id
                ):
                    raise BackendBatchContractError(
                        f"MuJoCo sensor field {spec.key!r} has an unsupported reference frame"
                    )
                reference_frame = ReferenceFrame.BASE
            expected_contracts.add((reference_frame, frame_sensor_units[sensor_type]))
            continue
        try:
            expected_contracts.add(sensor_contracts[sensor_type])
        except KeyError as exc:
            raise BackendBatchContractError(
                f"MuJoCo sensor type {sensor_type} has no typed state contract"
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
            f"MuJoCo sensor field {spec.key!r} requires homogeneous frame/unit {expected}"
        )
    indices = np.asarray(sensor_columns, dtype=np.intp)
    _require_field_contract(backend, spec, row_shape=(len(indices),))
    return _MuJoCoStateSource(
        spec=spec,
        source=backend._sensor_data,
        gather_indices=indices,
    )


def _bind_body_source(backend: MuJoCoBackend, spec: StateFieldSpec) -> _MuJoCoStateSource:
    if not backend.add_body_sensors:
        raise BackendBatchContractError(
            "MuJoCo body batch fields require cold-path add_body_sensors materialization"
        )
    identity = spec.identity
    body_ids = np.asarray(identity.entity_ids, dtype=np.intp)
    if np.any(body_ids >= int(backend._model.nbody)):
        raise BackendBatchContractError(
            f"MuJoCo body field {spec.key!r} contains an out-of-range id"
        )
    mapped = backend._body_id_to_tracked_idx[body_ids]
    if np.any(mapped < 0):
        raise BackendBatchContractError(
            f"MuJoCo body field {spec.key!r} references a body without a bound tracking cache"
        )
    sources: dict[
        tuple[StateFieldKind, ReferenceFrame],
        tuple[np.ndarray, int, PhysicalUnit],
    ] = {
        (StateFieldKind.POSITION, ReferenceFrame.WORLD): (
            backend._tracked_pos_w_all,
            3,
            PhysicalUnit.METER,
        ),
        (StateFieldKind.ORIENTATION, ReferenceFrame.WORLD): (
            backend._tracked_quat_w_all,
            4,
            PhysicalUnit.QUATERNION,
        ),
        (StateFieldKind.LINEAR_VELOCITY, ReferenceFrame.WORLD): (
            backend._tracked_linvel_w_all,
            3,
            PhysicalUnit.METER_PER_SECOND,
        ),
        (StateFieldKind.ANGULAR_VELOCITY, ReferenceFrame.WORLD): (
            backend._tracked_angvel_w_all,
            3,
            PhysicalUnit.RADIAN_PER_SECOND,
        ),
        (StateFieldKind.POSITION, ReferenceFrame.BASE): (
            backend._tracked_pos_b_all,
            3,
            PhysicalUnit.METER,
        ),
        (StateFieldKind.ORIENTATION, ReferenceFrame.BASE): (
            backend._tracked_quat_b_all,
            4,
            PhysicalUnit.QUATERNION,
        ),
        (StateFieldKind.LINEAR_VELOCITY, ReferenceFrame.BASE): (
            backend._tracked_linvel_b_all,
            3,
            PhysicalUnit.METER_PER_SECOND,
        ),
        (StateFieldKind.ANGULAR_VELOCITY, ReferenceFrame.BASE): (
            backend._tracked_angvel_b_all,
            3,
            PhysicalUnit.RADIAN_PER_SECOND,
        ),
    }
    try:
        source, width, unit = sources[(identity.field_kind, spec.frame)]
    except KeyError as exc:
        raise BackendBatchContractError(
            f"unsupported MuJoCo body field/frame combination for {spec.key!r}"
        ) from exc
    _require_field_contract(
        backend,
        spec,
        row_shape=(len(mapped), width),
        frame=spec.frame,
        unit=unit,
    )
    return _MuJoCoStateSource(
        spec=spec,
        source=source,
        gather_indices=np.asarray(mapped, dtype=np.intp),
    )


def _bind_state_source(backend: MuJoCoBackend, spec: StateFieldSpec) -> _MuJoCoStateSource:
    binders = {
        StateEntityKind.ROOT: _bind_root_source,
        StateEntityKind.DOF: _bind_dof_source,
        StateEntityKind.SENSOR: _bind_sensor_source,
        StateEntityKind.BODY: _bind_body_source,
    }
    return binders[spec.identity.entity_kind](backend, spec)


def _binding_payloads(
    backend: MuJoCoBackend,
    requirements: BackendIORequirements,
) -> tuple[dict[str, Any], dict[str, Any]]:
    def _names(object_type: mujoco.mjtObj, count: int) -> tuple[str, ...]:
        return tuple(
            mujoco.mj_id2name(backend._model, object_type, index) or f"#{index}"
            for index in range(count)
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

    fields = []
    for spec in requirements.state_fields:
        fields.append(
            {
                "semantic_key": spec.key,
                "entity_kind": spec.identity.entity_kind.value,
                "field_kind": spec.identity.field_kind.value,
                "entity_ids": spec.identity.entity_ids,
                "frame": spec.frame.value,
                "unit": spec.unit.value,
                "buffer": _buffer_payload(spec.buffer),
            }
        )
    state_payload = {
        "contract_version": requirements.contract_version,
        "backend_type": backend.backend_type,
        "model_dims": {
            "nq": int(backend._model.nq),
            "nv": int(backend._model.nv),
            "nu": int(backend._model.nu),
            "nsensor": int(backend._model.nsensor),
        },
        "model_semantics": {
            "body_names": _names(mujoco.mjtObj.mjOBJ_BODY, int(backend._model.nbody)),
            "joint_names": _names(mujoco.mjtObj.mjOBJ_JOINT, int(backend._model.njnt)),
            "actuator_names": _names(
                mujoco.mjtObj.mjOBJ_ACTUATOR,
                int(backend._model.nu),
            ),
            "sensor_names": _names(mujoco.mjtObj.mjOBJ_SENSOR, int(backend._model.nsensor)),
            "joint_type": tuple(int(value) for value in backend._model.jnt_type),
            "joint_qposadr": tuple(int(value) for value in backend._model.jnt_qposadr),
            "joint_dofadr": tuple(int(value) for value in backend._model.jnt_dofadr),
            "sensor_type": tuple(int(value) for value in backend._model.sensor_type),
            "sensor_dim": tuple(int(value) for value in backend._model.sensor_dim),
            "sensor_adr": tuple(int(value) for value in backend._model.sensor_adr),
            "sensor_objtype": tuple(int(value) for value in backend._model.sensor_objtype),
            "sensor_objid": tuple(int(value) for value in backend._model.sensor_objid),
            "sensor_reftype": tuple(int(value) for value in backend._model.sensor_reftype),
            "sensor_refid": tuple(int(value) for value in backend._model.sensor_refid),
            "timestep": float(backend._model.opt.timestep),
            "dtype": np.dtype(backend._np_dtype).name,
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
    }
    return state_payload, plan_payload


def _payload_digest(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _bind_mujoco_host_batch(
    backend: MuJoCoBackend,
    requirements: BackendIORequirements,
    *,
    backend_instance_id: str,
) -> _MuJoCoHostBatchPlan:
    if requirements.execution_profile is not ExecutionProfile.HOST_NUMPY:
        raise BackendBatchContractError("MuJoCo reference batches only support host_numpy")
    expected_dtype = np.dtype(backend._np_dtype).name
    control = requirements.control
    if control.buffer.row_shape != (int(backend._model.nu),):
        raise BackendBatchContractError(
            f"MuJoCo control requires row_shape {(int(backend._model.nu),)}, "
            f"got {control.buffer.row_shape}"
        )
    if control.buffer.dtype != expected_dtype:
        raise BackendBatchContractError(
            f"MuJoCo control requires dtype {expected_dtype}, got {control.buffer.dtype}"
        )
    if control.buffer.layout is not BufferLayout.C_CONTIGUOUS:
        raise BackendBatchContractError("MuJoCo host control requires c_contiguous layout")

    sources = tuple(_bind_state_source(backend, spec) for spec in requirements.state_fields)
    state_payload, plan_payload = _binding_payloads(backend, requirements)
    state_digest = _payload_digest(state_payload)
    plan_digest = _payload_digest(plan_payload)
    state_plan = BoundStatePlan(
        backend_type=backend.backend_type,
        backend_instance_id=backend_instance_id,
        num_envs=backend.num_envs,
        fields=requirements.state_fields,
        execution_profile=requirements.execution_profile,
        fingerprint=f"{_STATE_FINGERPRINT_PREFIX}:{state_digest}",
    )
    public_plan = BoundBackendPlan(
        state=state_plan,
        control=control,
        execution_profile=requirements.execution_profile,
        fingerprint=f"{_PLAN_FINGERPRINT_PREFIX}:{plan_digest}",
        contract_version=BACKEND_BATCH_CONTRACT_VERSION,
    )
    return _MuJoCoHostBatchPlan(
        public_plan=public_plan,
        sources=sources,
        lease=StateBatchLease(backend_instance_id),
    )


def _legacy_timings(payload: dict | None) -> tuple[BackendTiming, ...]:
    if not payload:
        return ()
    timing = payload.get("timing")
    if not isinstance(timing, dict):
        return ()
    return tuple(BackendTiming(str(name), float(value)) for name, value in timing.items())
