from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import numpy as np
import pytest

from unilab.base.backend.base import SimBackend
from unilab.base.backend.batch import (
    BACKEND_BATCH_CONTRACT_VERSION,
    BackendBatchContractError,
    BackendBatchCounters,
    BackendBatchDiagnostics,
    BackendCompletionEvent,
    BackendIORequirements,
    BackendResetResult,
    BackendStepResult,
    BackendTiming,
    BoundBackendPlan,
    BoundFieldIdentity,
    BoundStatePlan,
    BufferContract,
    BufferLayout,
    BufferLifetime,
    BufferMutability,
    BufferOwner,
    BufferPlacement,
    BufferView,
    ControlBatch,
    ControlSpec,
    ExecutionProfile,
    MemorySpace,
    PhysicalUnit,
    ReferenceFrame,
    RowSelection,
    StaleStateBatchError,
    StateBatch,
    StateBatchLease,
    StateBatchPhase,
    StateEntityKind,
    StateFieldKind,
    StateFieldSpec,
)
from unilab.base.scene import SceneCfg


def _state_buffer(
    *,
    row_shape: tuple[int, ...] = (3,),
    dtype: str = "float32",
    layout: BufferLayout = BufferLayout.C_CONTIGUOUS,
    placement: BufferPlacement | None = None,
    owner: BufferOwner = BufferOwner.BACKEND,
    mutability: BufferMutability = BufferMutability.READ_ONLY,
    lifetime: BufferLifetime = BufferLifetime.BORROWED_UNTIL_MUTATION,
    dlpack_exportable: bool = False,
    address_stable: bool = True,
) -> BufferContract:
    return BufferContract(
        row_shape=row_shape,
        dtype=dtype,
        layout=layout,
        placement=placement or BufferPlacement.host(),
        owner=owner,
        mutability=mutability,
        lifetime=lifetime,
        dlpack_exportable=dlpack_exportable,
        address_stable=address_stable,
    )


def _control_buffer(*, placement: BufferPlacement | None = None) -> BufferContract:
    return BufferContract(
        row_shape=(2,),
        dtype="float32",
        layout=BufferLayout.C_CONTIGUOUS,
        placement=placement or BufferPlacement.host(),
        owner=BufferOwner.MANAGER,
        mutability=BufferMutability.READ_ONLY,
        lifetime=BufferLifetime.UNTIL_STEP_COMPLETE,
        dlpack_exportable=False,
    )


def _field(
    semantic_key: str = "base.position",
    *,
    entity_ids: tuple[int, ...] = (0,),
    frame: ReferenceFrame = ReferenceFrame.WORLD,
    unit: PhysicalUnit = PhysicalUnit.METER,
    buffer: BufferContract | None = None,
) -> StateFieldSpec:
    return StateFieldSpec(
        semantic_key=semantic_key,
        identity=BoundFieldIdentity(
            entity_kind=StateEntityKind.BODY,
            field_kind=StateFieldKind.POSITION,
            entity_ids=entity_ids,
        ),
        frame=frame,
        unit=unit,
        buffer=buffer or _state_buffer(),
    )


def _requirements(
    *,
    fields: tuple[StateFieldSpec, ...] | None = None,
    profile: ExecutionProfile = ExecutionProfile.HOST_NUMPY,
    control: ControlSpec | None = None,
) -> BackendIORequirements:
    return BackendIORequirements(
        state_fields=fields or (_field(),),
        control=control or ControlSpec("joint.command", _control_buffer()),
        execution_profile=profile,
    )


class _FakeBatchBackend:
    backend_type = "fake"
    backend_instance_id = "fake:0"

    def __init__(
        self,
        requirements: BackendIORequirements | None = None,
        *,
        num_envs: int = 4,
    ) -> None:
        requirements = requirements or _requirements()
        state_plan = BoundStatePlan(
            backend_type=self.backend_type,
            backend_instance_id=self.backend_instance_id,
            num_envs=num_envs,
            fields=requirements.state_fields,
            execution_profile=requirements.execution_profile,
            fingerprint="fake-state-v1",
        )
        self.plan = BoundBackendPlan(
            state=state_plan,
            control=requirements.control,
            execution_profile=requirements.execution_profile,
            fingerprint=BACKEND_BATCH_CONTRACT_VERSION,
        )
        self.lease = StateBatchLease(self.backend_instance_id)

    def state(
        self,
        rows: RowSelection,
        *,
        phase: StateBatchPhase = StateBatchPhase.CURRENT,
        buffers: tuple[BufferView, ...] | None = None,
    ) -> StateBatch:
        if buffers is None:
            built = []
            for spec in self.plan.state.fields:
                shape = (rows.count, *spec.buffer.row_shape)
                handle = np.zeros(shape, dtype=spec.buffer.dtype)
                handle.flags.writeable = False
                built.append(BufferView(handle=handle, shape=shape, contract=spec.buffer))
            buffers = tuple(built)
        return StateBatch(
            plan=self.plan,
            rows=rows,
            phase=phase,
            descriptors=buffers,
            lease=self.lease,
        )

    def control(self, rows: RowSelection, *, view: BufferView | None = None) -> ControlBatch:
        if view is None:
            shape = (rows.count, *self.plan.control.buffer.row_shape)
            view = BufferView(
                handle=np.zeros(shape, dtype=self.plan.control.buffer.dtype),
                shape=shape,
                contract=self.plan.control.buffer,
            )
        return ControlBatch(plan=self.plan, rows=rows, buffer=view)


def _valid_all_rows() -> None:
    backend = _FakeBatchBackend()
    state = backend.state(RowSelection.all(4))
    assert state.rows.count == 4
    assert state.buffer_at(0).shape == (4, 3)
    assert state.buffer("base.position").contract.dtype == "float32"
    assert cast(np.ndarray, state.buffer_at(0).handle).flags.writeable is False


def _valid_selected_row_order() -> None:
    backend = _FakeBatchBackend()
    rows = RowSelection.selected(4, (3, 1))
    state = backend.state(rows)
    assert state.rows.indices == (3, 1)
    assert state.buffer_at(0).shape == (2, 3)


def _valid_frozen_batch_sizes() -> None:
    for num_envs in (1, 17):
        backend = _FakeBatchBackend(num_envs=num_envs)
        state = backend.state(RowSelection.all(num_envs))
        assert state.buffer_at(0).shape == (num_envs, 3)
        assert state.plan.fingerprint == BACKEND_BATCH_CONTRACT_VERSION


def _valid_same_entity_in_distinct_frames() -> None:
    world = _field("body.position.world")
    base = replace(world, semantic_key="body.position.base", frame=ReferenceFrame.BASE)
    backend = _FakeBatchBackend(_requirements(fields=(world, base)))
    state = backend.state(RowSelection.all(4))
    assert state.buffer("body.position.world").shape == (4, 3)
    assert state.buffer("body.position.base").shape == (4, 3)


def _invalid_shape() -> None:
    backend = _FakeBatchBackend()
    spec = backend.plan.state.fields[0]
    view = BufferView(np.zeros((4, 2)), (4, 2), spec.buffer)
    with pytest.raises(BackendBatchContractError, match="requires shape"):
        backend.state(RowSelection.all(4), buffers=(view,))


def _invalid_frame() -> None:
    with pytest.raises(BackendBatchContractError, match="frame must be one of"):
        _field(frame=cast(Any, "galactic"))


def _invalid_unit() -> None:
    with pytest.raises(BackendBatchContractError, match="unit must be one of"):
        _field(unit=cast(Any, "parsec"))


def _invalid_dtype() -> None:
    with pytest.raises(BackendBatchContractError, match="invalid dtype"):
        _state_buffer(dtype="float23")


def _invalid_layout() -> None:
    with pytest.raises(BackendBatchContractError, match="layout must be one of"):
        _state_buffer(layout=cast(Any, "sometimes-contiguous"))


def _invalid_duplicate_rows() -> None:
    with pytest.raises(BackendBatchContractError, match="must be unique"):
        RowSelection.selected(4, (1, 1))


def _invalid_row_bounds() -> None:
    with pytest.raises(BackendBatchContractError, match="outside the row universe"):
        RowSelection.selected(4, (4,))


def _invalid_row_universe() -> None:
    backend = _FakeBatchBackend()
    with pytest.raises(BackendBatchContractError, match="row universe"):
        backend.state(RowSelection.all(3))


def _invalid_placement() -> None:
    device_state = _field(buffer=_state_buffer(placement=BufferPlacement.device("cuda", 0)))
    with pytest.raises(BackendBatchContractError, match="host_numpy requires"):
        _requirements(fields=(device_state,))


def _invalid_owner() -> None:
    with pytest.raises(BackendBatchContractError, match="backend-owned"):
        _field(buffer=_state_buffer(owner=BufferOwner.RUNTIME))


def _invalid_mutability() -> None:
    with pytest.raises(BackendBatchContractError, match="read-only"):
        _field(buffer=_state_buffer(mutability=BufferMutability.READ_WRITE))


def _invalid_lifetime() -> None:
    with pytest.raises(BackendBatchContractError, match="borrowed_until_mutation"):
        _field(buffer=_state_buffer(lifetime=BufferLifetime.PLAN))


def _invalid_dlpack_metadata() -> None:
    backend = _FakeBatchBackend()
    spec = backend.plan.state.fields[0]
    wrong = replace(spec.buffer, dlpack_exportable=True)
    view = BufferView(np.zeros((4, 3)), (4, 3), wrong)
    with pytest.raises(BackendBatchContractError, match="metadata does not match"):
        backend.state(RowSelection.all(4), buffers=(view,))


def _invalid_unbound_identity() -> None:
    with pytest.raises(BackendBatchContractError, match="non-negative integers"):
        _field(entity_ids=(-1,))


def _invalid_empty_identity() -> None:
    with pytest.raises(BackendBatchContractError, match="at least one bound id"):
        _field(entity_ids=())


def _invalid_duplicate_field() -> None:
    duplicate = _field()
    with pytest.raises(BackendBatchContractError, match="semantic keys must be unique"):
        _requirements(fields=(duplicate, duplicate))


def _invalid_duplicate_bound_identity() -> None:
    first = _field("base.position")
    alias = replace(first, semantic_key="robot.root.position")
    with pytest.raises(BackendBatchContractError, match="identities and frames must be unique"):
        _requirements(fields=(first, alias))


def _invalid_plan_owner() -> None:
    backend = _FakeBatchBackend()
    with pytest.raises(BackendBatchContractError, match="different backend"):
        backend.plan.require_owner(backend_type="other", backend_instance_id="fake:0")


def _invalid_plan_fingerprint() -> None:
    backend = _FakeBatchBackend()
    other = replace(backend.plan, fingerprint="other-plan-v1")
    with pytest.raises(BackendBatchContractError, match="different backend plan or fingerprint"):
        backend.plan.require_compatible(other)


def _invalid_plan_metadata_with_reused_fingerprint() -> None:
    backend = _FakeBatchBackend()
    tampered = replace(
        backend.plan,
        control=replace(backend.plan.control, semantic_key="other.command"),
    )
    with pytest.raises(BackendBatchContractError, match="different backend plan or fingerprint"):
        backend.plan.require_compatible(tampered)


_CONTRACT_CASES: tuple[Callable[[], None], ...] = (
    _valid_all_rows,
    _valid_selected_row_order,
    _valid_frozen_batch_sizes,
    _valid_same_entity_in_distinct_frames,
    _invalid_shape,
    _invalid_frame,
    _invalid_unit,
    _invalid_dtype,
    _invalid_layout,
    _invalid_duplicate_rows,
    _invalid_row_bounds,
    _invalid_row_universe,
    _invalid_placement,
    _invalid_owner,
    _invalid_mutability,
    _invalid_lifetime,
    _invalid_dlpack_metadata,
    _invalid_unbound_identity,
    _invalid_empty_identity,
    _invalid_duplicate_field,
    _invalid_duplicate_bound_identity,
    _invalid_plan_owner,
    _invalid_plan_fingerprint,
    _invalid_plan_metadata_with_reused_fingerprint,
)


@pytest.mark.parametrize("case", _CONTRACT_CASES, ids=lambda case: case.__name__)
def test_bound_state_batch_contract(case: Callable[[], None]) -> None:
    case()


def test_borrowed_state_and_field_views_expire_at_mutation_barrier() -> None:
    backend = _FakeBatchBackend()
    state = backend.state(RowSelection.all(4))
    borrowed = state.buffer_at(0)
    assert not hasattr(state, "descriptors")
    assert not hasattr(state, "lease")

    backend.lease.invalidate()

    with pytest.raises(StaleStateBatchError, match="mutation barrier"):
        state.assert_valid()
    with pytest.raises(StaleStateBatchError, match="mutation barrier"):
        _ = borrowed.handle


def test_state_field_indices_are_non_negative_and_bound() -> None:
    state = _FakeBatchBackend().state(RowSelection.all(4))
    with pytest.raises(BackendBatchContractError, match="not bound"):
        state.buffer_at(-1)


def test_control_batch_matches_bound_plan_metadata_and_rows() -> None:
    backend = _FakeBatchBackend()
    rows = RowSelection.selected(4, (2, 0))
    control = backend.control(rows)
    assert control.buffer.shape == (2, 2)

    wrong_contract = replace(backend.plan.control.buffer, owner=BufferOwner.RUNNER)
    wrong_view = BufferView(np.zeros((2, 2)), (2, 2), wrong_contract)
    with pytest.raises(BackendBatchContractError, match="metadata does not match"):
        backend.control(rows, view=wrong_view)


def test_batch_results_preserve_terminal_reset_and_diagnostic_semantics() -> None:
    backend = _FakeBatchBackend()
    rows = RowSelection.all(4)
    diagnostics = BackendBatchDiagnostics(
        counters=BackendBatchCounters(state_materializations=1),
        timings=(BackendTiming("physics", 0.25),),
    )
    step_result = BackendStepResult(
        terminal_state=backend.state(rows, phase=StateBatchPhase.TERMINAL),
        diagnostics=diagnostics,
    )
    assert step_result.diagnostics.counters.state_materializations == 1

    backend.lease.invalidate()
    reset_result = BackendResetResult(
        reset_state=backend.state(rows, phase=StateBatchPhase.RESET),
        diagnostics=diagnostics,
    )
    assert reset_result.reset_state.phase is StateBatchPhase.RESET

    with pytest.raises(BackendBatchContractError, match="terminal state semantics"):
        BackendStepResult(terminal_state=reset_result.reset_state)


def test_completion_event_is_explicitly_device_owned() -> None:
    event = BackendCompletionEvent(
        backend_type="fake",
        placement=BufferPlacement.device("cuda", 0),
        handle=object(),
    )
    diagnostics = BackendBatchDiagnostics(completion_event=event)
    assert diagnostics.completion_event is event

    with pytest.raises(BackendBatchContractError, match="require device placement"):
        BackendCompletionEvent(
            backend_type="fake",
            placement=BufferPlacement.host(),
            handle=object(),
        )

    placement = BufferPlacement.device("cuda", 0)
    requirements = _requirements(
        fields=(_field(buffer=_state_buffer(placement=placement, dlpack_exportable=True)),),
        profile=ExecutionProfile.DEVICE_RESIDENT,
        control=ControlSpec("joint.command", _control_buffer(placement=placement)),
    )
    backend = _FakeBatchBackend(requirements)
    terminal = backend.state(RowSelection.all(4), phase=StateBatchPhase.TERMINAL)
    BackendStepResult(terminal, diagnostics)

    wrong_event = replace(event, backend_type="other")
    with pytest.raises(BackendBatchContractError, match="different backend type"):
        BackendStepResult(
            terminal,
            BackendBatchDiagnostics(completion_event=wrong_event),
        )


def test_batch_contract_version_and_cross_device_placement_fail_closed() -> None:
    with pytest.raises(BackendBatchContractError, match="unsupported backend batch contract"):
        replace(_requirements(), contract_version="backend-batch-contract-v0")

    cuda0 = BufferPlacement.device("cuda", 0)
    cuda1 = BufferPlacement.device("cuda", 1)
    field = _field(buffer=_state_buffer(placement=cuda0, dlpack_exportable=True))
    control = ControlSpec("joint.command", _control_buffer(placement=cuda1))
    with pytest.raises(BackendBatchContractError, match="one shared state/control placement"):
        _requirements(
            fields=(field,),
            profile=ExecutionProfile.DEVICE_RESIDENT,
            control=control,
        )


def test_device_profile_requires_explicit_device_placement() -> None:
    placement = BufferPlacement.device("cuda", 0)
    field = _field(buffer=_state_buffer(placement=placement, dlpack_exportable=True))
    control = ControlSpec("joint.command", _control_buffer(placement=placement))
    requirements = _requirements(
        fields=(field,),
        profile=ExecutionProfile.DEVICE_RESIDENT,
        control=control,
    )
    assert requirements.state_fields[0].buffer.placement.memory_space is MemorySpace.DEVICE


def test_sim_backend_batch_extensions_are_additive_and_fail_closed() -> None:
    backend = cast(Any, object())
    fake = _FakeBatchBackend()
    rows = RowSelection.all(4)
    control = fake.control(rows)

    assert "bind_task_io" not in SimBackend.__abstractmethods__
    assert "read_state_batch" not in SimBackend.__abstractmethods__
    assert "step_batch" not in SimBackend.__abstractmethods__
    assert "reset_batch" not in SimBackend.__abstractmethods__
    with pytest.raises(NotImplementedError, match="typed backend batches"):
        SimBackend.bind_task_io(backend, _requirements())
    with pytest.raises(NotImplementedError, match="typed backend batches"):
        SimBackend.step_batch(backend, fake.plan, control)
    with pytest.raises(NotImplementedError, match="typed backend batches"):
        SimBackend.read_state_batch(backend, fake.plan, rows)
    with pytest.raises(NotImplementedError, match="typed backend batches"):
        SimBackend.reset_batch(backend, fake.plan, rows)


def _g1_scene_path() -> Path:
    from unilab.assets import ASSETS_ROOT_PATH

    return ASSETS_ROOT_PATH / "robots" / "g1" / "scene_flat.xml"


def _mujoco_state_buffer(row_shape: tuple[int, ...]) -> BufferContract:
    return _state_buffer(row_shape=row_shape, dtype="float64")


def _mujoco_field(
    semantic_key: str,
    *,
    entity_kind: StateEntityKind,
    field_kind: StateFieldKind,
    entity_ids: tuple[int, ...],
    row_shape: tuple[int, ...],
    frame: ReferenceFrame,
    unit: PhysicalUnit,
) -> StateFieldSpec:
    return StateFieldSpec(
        semantic_key=semantic_key,
        identity=BoundFieldIdentity(
            entity_kind=entity_kind,
            field_kind=field_kind,
            entity_ids=entity_ids,
        ),
        frame=frame,
        unit=unit,
        buffer=_mujoco_state_buffer(row_shape),
    )


def _mujoco_requirements(backend: Any) -> tuple[BackendIORequirements, list[Callable[[], Any]]]:
    import mujoco

    base_id = backend.get_body_id("pelvis")
    dof_pos_ids = tuple(range(backend.get_dof_pos().shape[1]))
    dof_vel_ids = tuple(range(backend.get_dof_vel().shape[1]))
    fields: list[StateFieldSpec] = [
        _mujoco_field(
            "root.position",
            entity_kind=StateEntityKind.ROOT,
            field_kind=StateFieldKind.POSITION,
            entity_ids=(base_id,),
            row_shape=(3,),
            frame=ReferenceFrame.WORLD,
            unit=PhysicalUnit.METER,
        ),
        _mujoco_field(
            "root.orientation",
            entity_kind=StateEntityKind.ROOT,
            field_kind=StateFieldKind.ORIENTATION,
            entity_ids=(base_id,),
            row_shape=(4,),
            frame=ReferenceFrame.WORLD,
            unit=PhysicalUnit.QUATERNION,
        ),
        _mujoco_field(
            "root.linear_velocity",
            entity_kind=StateEntityKind.ROOT,
            field_kind=StateFieldKind.LINEAR_VELOCITY,
            entity_ids=(base_id,),
            row_shape=(3,),
            frame=ReferenceFrame.WORLD,
            unit=PhysicalUnit.METER_PER_SECOND,
        ),
        _mujoco_field(
            "root.angular_velocity",
            entity_kind=StateEntityKind.ROOT,
            field_kind=StateFieldKind.ANGULAR_VELOCITY,
            entity_ids=(base_id,),
            row_shape=(3,),
            frame=ReferenceFrame.WORLD,
            unit=PhysicalUnit.RADIAN_PER_SECOND,
        ),
        _mujoco_field(
            "dof.position",
            entity_kind=StateEntityKind.DOF,
            field_kind=StateFieldKind.POSITION,
            entity_ids=dof_pos_ids,
            row_shape=(len(dof_pos_ids),),
            frame=ReferenceFrame.JOINT,
            unit=PhysicalUnit.RADIAN,
        ),
        _mujoco_field(
            "dof.velocity",
            entity_kind=StateEntityKind.DOF,
            field_kind=StateFieldKind.ANGULAR_VELOCITY,
            entity_ids=dof_vel_ids,
            row_shape=(len(dof_vel_ids),),
            frame=ReferenceFrame.JOINT,
            unit=PhysicalUnit.RADIAN_PER_SECOND,
        ),
    ]
    getters: list[Callable[[], Any]] = [
        backend.get_base_pos,
        backend.get_base_quat,
        backend.get_base_lin_vel,
        backend.get_base_ang_vel,
        backend.get_dof_pos,
        backend.get_dof_vel,
    ]
    sensor_specs = (
        ("pelvis_local_linvel", ReferenceFrame.SENSOR, PhysicalUnit.METER_PER_SECOND),
        ("torso_gyro", ReferenceFrame.SENSOR, PhysicalUnit.RADIAN_PER_SECOND),
        ("torso_upvector", ReferenceFrame.WORLD, PhysicalUnit.UNITLESS),
        ("left_foot_pos", ReferenceFrame.WORLD, PhysicalUnit.METER),
        ("left_foot_quat", ReferenceFrame.WORLD, PhysicalUnit.QUATERNION),
        ("right_foot_pos", ReferenceFrame.WORLD, PhysicalUnit.METER),
        ("right_foot_quat", ReferenceFrame.WORLD, PhysicalUnit.QUATERNION),
        *(
            (f"{side}_foot_contact_{index}", ReferenceFrame.SENSOR, PhysicalUnit.NEWTON)
            for side in ("left", "right")
            for index in range(4)
        ),
    )
    for sensor_name, frame, unit in sensor_specs:
        sensor_id = mujoco.mj_name2id(
            backend.model,
            mujoco.mjtObj.mjOBJ_SENSOR,
            sensor_name,
        )
        sensor_dim = int(backend.model.sensor_dim[sensor_id])
        fields.append(
            _mujoco_field(
                f"sensor.{sensor_name}",
                entity_kind=StateEntityKind.SENSOR,
                field_kind=StateFieldKind.VALUE,
                entity_ids=(int(sensor_id),),
                row_shape=(sensor_dim,),
                frame=frame,
                unit=unit,
            )
        )
        getters.append(lambda name=sensor_name: backend.get_sensor_data(name))

    control_buffer = BufferContract(
        row_shape=(backend.num_actuators,),
        dtype="float64",
        layout=BufferLayout.C_CONTIGUOUS,
        placement=BufferPlacement.host(),
        owner=BufferOwner.MANAGER,
        mutability=BufferMutability.READ_ONLY,
        lifetime=BufferLifetime.UNTIL_STEP_COMPLETE,
        dlpack_exportable=False,
    )
    requirements = BackendIORequirements(
        state_fields=tuple(fields),
        control=ControlSpec(
            "joint.position_target",
            control_buffer,
            physics_substeps_per_control=2,
        ),
        execution_profile=ExecutionProfile.HOST_NUMPY,
    )
    return requirements, getters


def _random_g1_state(backend: Any, rng: np.random.Generator, count: int) -> tuple[Any, Any]:
    qpos = np.broadcast_to(backend.model.qpos0, (count, backend.model.nq)).copy()
    qvel = rng.normal(0.0, 0.1, size=(count, backend.model.nv))
    qpos[:, :3] += rng.uniform(-0.1, 0.1, size=(count, 3))
    quaternion = rng.normal(size=(count, 4))
    quaternion /= np.linalg.norm(quaternion, axis=1, keepdims=True)
    qpos[:, 3:7] = quaternion
    qpos[:, 7:] += rng.uniform(-0.05, 0.05, size=(count, backend.model.nq - 7))
    return qpos, qvel


def _assert_batch_matches(
    result: Any,
    references: list[np.ndarray],
    rows: tuple[int, ...] | None,
) -> None:
    assert result.diagnostics.counters.state_materializations == 1
    assert not result.diagnostics.counters.instrumentation_complete
    for index, reference in enumerate(references):
        expected = reference if rows is None else reference[np.asarray(rows, dtype=np.intp)]
        actual = cast(np.ndarray, result.state.buffer_at(index).handle)
        assert not actual.flags.writeable
        assert actual.flags.c_contiguous
        np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-8)


@pytest.mark.parametrize("num_envs", [1, 32])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_mujoco_batch_matches_getter_reference(seed: int, num_envs: int) -> None:
    from unilab.base.backend.mujoco.backend import MuJoCoBackend

    rng = np.random.default_rng(seed)
    backend = MuJoCoBackend(
        SceneCfg(model_file=str(_g1_scene_path())),
        num_envs,
        0.005,
        base_name="pelvis",
        np_dtype=np.float64,
        chunk_size=max(1, num_envs),
        bench_nsteps=2,
    )
    try:
        backend.materialize()
        requirements, getters = _mujoco_requirements(backend)
        plan = backend.bind_task_io(requirements)
        assert plan.fingerprint.startswith("mujoco-host-batch-v1:")

        pre_reset = backend.read_state_batch(plan, RowSelection.all(num_envs))
        pre_reset_view = pre_reset.state.buffer_at(0)

        reset_count = max(1, num_envs // 2)
        reset_rows = tuple(int(row) for row in rng.permutation(num_envs)[:reset_count])
        qpos, qvel = _random_g1_state(backend, rng, reset_count)
        backend.set_state(np.asarray(reset_rows, dtype=np.int32), qpos, qvel)
        with pytest.raises(StaleStateBatchError, match="mutation barrier"):
            _ = pre_reset_view.handle
        references = [np.asarray(getter()).copy() for getter in getters]

        with (
            patch.object(backend, "get_base_pos", side_effect=AssertionError("getter fallback")),
            patch.object(backend, "get_base_quat", side_effect=AssertionError("getter fallback")),
            patch.object(
                backend,
                "get_base_lin_vel",
                side_effect=AssertionError("getter fallback"),
            ),
            patch.object(
                backend,
                "get_base_ang_vel",
                side_effect=AssertionError("getter fallback"),
            ),
            patch.object(backend, "get_dof_pos", side_effect=AssertionError("getter fallback")),
            patch.object(backend, "get_dof_vel", side_effect=AssertionError("getter fallback")),
            patch.object(
                backend,
                "get_sensor_data",
                side_effect=AssertionError("getter fallback"),
            ),
        ):
            selected = backend.read_state_batch(
                plan,
                RowSelection.selected(num_envs, reset_rows),
                phase=StateBatchPhase.RESET,
            )
        _assert_batch_matches(selected, references, reset_rows)
        selected_view = selected.state.buffer_at(0)

        all_rows = backend.read_state_batch(plan, RowSelection.all(num_envs))
        _assert_batch_matches(all_rows, references, None)
        with pytest.raises(StaleStateBatchError, match="mutation barrier"):
            _ = selected_view.handle
        all_view = all_rows.state.buffer_at(0)

        control = np.ascontiguousarray(
            rng.uniform(-0.1, 0.1, size=(num_envs, backend.num_actuators)),
            dtype=np.float64,
        )
        control_batch = ControlBatch(
            plan=plan,
            rows=RowSelection.all(num_envs),
            buffer=BufferView(control, control.shape, plan.control.buffer),
        )
        stepped = backend.step_batch(plan, control_batch, nsteps=2)
        with pytest.raises(StaleStateBatchError, match="mutation barrier"):
            _ = all_view.handle
        stepped_references = [np.asarray(getter()).copy() for getter in getters]
        assert stepped.diagnostics.counters.state_materializations == 1
        for index, expected in enumerate(stepped_references):
            np.testing.assert_allclose(
                cast(np.ndarray, stepped.terminal_state.buffer_at(index).handle),
                expected,
                atol=1e-10,
                rtol=1e-8,
            )
    finally:
        if backend._pool is not None:
            backend._pool.close()


def test_mujoco_batch_contract_faults_fail_closed() -> None:
    import mujoco

    from unilab.base.backend.mujoco.backend import MuJoCoBackend

    backend = MuJoCoBackend(
        SceneCfg(model_file=str(_g1_scene_path())),
        2,
        0.005,
        base_name="pelvis",
        np_dtype=np.float64,
        add_body_sensors=True,
        chunk_size=2,
    )
    try:
        with pytest.raises(BackendBatchContractError, match="materialized backend pool"):
            backend.bind_task_io(_requirements())
        backend.materialize()
        requirements, _ = _mujoco_requirements(backend)

        body_id = backend.get_body_id("left_ankle_roll_link")
        body_field = _mujoco_field(
            "body.left_foot.position.base",
            entity_kind=StateEntityKind.BODY,
            field_kind=StateFieldKind.POSITION,
            entity_ids=(body_id,),
            row_shape=(1, 3),
            frame=ReferenceFrame.BASE,
            unit=PhysicalUnit.METER,
        )
        body_plan = backend.bind_task_io(replace(requirements, state_fields=(body_field,)))
        body_state = backend.read_state_batch(body_plan, RowSelection.all(2))
        np.testing.assert_allclose(
            cast(np.ndarray, body_state.state.buffer_at(0).handle),
            backend.get_body_pos_b(np.asarray([body_id], dtype=np.int32)),
        )

        body_world_field = replace(
            body_field,
            semantic_key="body.left_foot.position.world",
            frame=ReferenceFrame.WORLD,
        )
        dual_frame_plan = backend.bind_task_io(
            replace(requirements, state_fields=(body_field, body_world_field))
        )
        dual_frame_state = backend.read_state_batch(dual_frame_plan, RowSelection.all(2))
        np.testing.assert_allclose(
            cast(np.ndarray, dual_frame_state.state.buffer_at(0).handle),
            backend.get_body_pos_b(np.asarray([body_id], dtype=np.int32)),
        )
        np.testing.assert_allclose(
            cast(np.ndarray, dual_frame_state.state.buffer_at(1).handle),
            backend.get_body_pos_w(np.asarray([body_id], dtype=np.int32)),
        )

        tracked_sensor_name = "track_pos_b_left_ankle_roll_link"
        tracked_sensor_id = mujoco.mj_name2id(
            backend.model,
            mujoco.mjtObj.mjOBJ_SENSOR,
            tracked_sensor_name,
        )
        tracked_sensor = _mujoco_field(
            "sensor.left_foot.position.base",
            entity_kind=StateEntityKind.SENSOR,
            field_kind=StateFieldKind.VALUE,
            entity_ids=(int(tracked_sensor_id),),
            row_shape=(3,),
            frame=ReferenceFrame.BASE,
            unit=PhysicalUnit.METER,
        )
        tracked_sensor_plan = backend.bind_task_io(
            replace(requirements, state_fields=(tracked_sensor,))
        )
        tracked_sensor_state = backend.read_state_batch(
            tracked_sensor_plan,
            RowSelection.all(2),
        )
        np.testing.assert_allclose(
            cast(np.ndarray, tracked_sensor_state.state.buffer_at(0).handle),
            backend.get_sensor_data(tracked_sensor_name),
        )
        with pytest.raises(BackendBatchContractError, match="homogeneous frame/unit"):
            backend.bind_task_io(
                replace(
                    requirements,
                    state_fields=(replace(tracked_sensor, frame=ReferenceFrame.WORLD),),
                )
            )

        root = requirements.state_fields[0]
        sensor = requirements.state_fields[6]
        dof = requirements.state_fields[4]
        with (
            patch.object(backend, "_root_qpos_dim", 0),
            patch.object(backend, "_root_qvel_dim", 0),
            pytest.raises(BackendBatchContractError, match="free-joint root cache"),
        ):
            backend.bind_task_io(replace(requirements, state_fields=(root,)))

        invalid_specs = (
            replace(root, buffer=replace(root.buffer, row_shape=(2,))),
            replace(root, frame=ReferenceFrame.BASE),
            replace(root, unit=PhysicalUnit.RADIAN),
            replace(root, identity=replace(root.identity, entity_ids=(backend.model.nbody,))),
            replace(
                root,
                identity=replace(root.identity, field_kind=StateFieldKind.VALUE),
            ),
            replace(sensor, unit=PhysicalUnit.RADIAN),
            replace(
                dof,
                identity=replace(dof.identity, entity_ids=(backend.get_dof_pos().shape[1],)),
                buffer=replace(dof.buffer, row_shape=(1,)),
            ),
        )
        for invalid in invalid_specs:
            invalid_requirements = replace(requirements, state_fields=(invalid,))
            with pytest.raises(BackendBatchContractError):
                backend.bind_task_io(invalid_requirements)

        invalid_control = replace(
            requirements.control,
            buffer=replace(requirements.control.buffer, row_shape=(1,)),
        )
        with pytest.raises(BackendBatchContractError, match="control requires row_shape"):
            backend.bind_task_io(replace(requirements, control=invalid_control))

        device_placement = BufferPlacement.device("cuda", 0)
        device_fields = tuple(
            replace(spec, buffer=replace(spec.buffer, placement=device_placement))
            for spec in requirements.state_fields
        )
        device_control = replace(
            requirements.control,
            buffer=replace(requirements.control.buffer, placement=device_placement),
        )
        with pytest.raises(BackendBatchContractError, match="only support host_numpy"):
            backend.bind_task_io(
                replace(
                    requirements,
                    state_fields=device_fields,
                    control=device_control,
                    execution_profile=ExecutionProfile.DEVICE_RESIDENT,
                )
            )

        plan = backend.bind_task_io(requirements)
        assert backend.bind_task_io(replace(requirements)) is plan

        alternate_cadence_plan = backend.bind_task_io(
            replace(
                requirements,
                control=replace(
                    requirements.control,
                    physics_substeps_per_control=1,
                ),
            )
        )
        assert alternate_cadence_plan.fingerprint != plan.fingerprint
        assert alternate_cadence_plan.state.fingerprint == plan.state.fingerprint

        root_only_plan = backend.bind_task_io(
            replace(requirements, state_fields=requirements.state_fields[:1])
        )
        assert root_only_plan.fingerprint != plan.fingerprint
        assert root_only_plan.state.fingerprint != plan.state.fingerprint

        wrong_owner_plan = replace(
            plan,
            state=replace(plan.state, backend_instance_id="mujoco:other"),
        )
        with pytest.raises(BackendBatchContractError, match="different backend"):
            backend.read_state_batch(wrong_owner_plan, RowSelection.all(2))
        wrong_dtype = np.zeros((2, backend.num_actuators), dtype=np.float32)
        control_batch = ControlBatch(
            plan,
            RowSelection.all(2),
            BufferView(wrong_dtype, wrong_dtype.shape, plan.control.buffer),
        )
        with pytest.raises(BackendBatchContractError, match="handle dtype"):
            backend.step_batch(plan, control_batch, nsteps=2)

        valid_control = np.zeros((2, backend.num_actuators), dtype=np.float64)
        valid_batch = ControlBatch(
            plan,
            RowSelection.all(2),
            BufferView(valid_control, valid_control.shape, plan.control.buffer),
        )
        with pytest.raises(BackendBatchContractError, match="control cadence"):
            backend.step_batch(plan, valid_batch, nsteps=1)
        with pytest.raises(BackendBatchContractError, match="mutation batches"):
            backend.step_batch(plan, valid_batch, mutation_batch=cast(Any, object()), nsteps=2)
        with pytest.raises(BackendBatchContractError, match="controls for all rows"):
            partial = ControlBatch(
                plan,
                RowSelection.selected(2, (1,)),
                BufferView(
                    np.zeros((1, backend.num_actuators), dtype=np.float64),
                    (1, backend.num_actuators),
                    plan.control.buffer,
                ),
            )
            backend.step_batch(plan, partial, nsteps=2)
    finally:
        if backend._pool is not None:
            backend._pool.close()
