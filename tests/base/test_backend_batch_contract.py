from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Any, cast

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
    StateFieldSpec,
)


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
            entity_kind="body",
            field_kind="position",
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
    with pytest.raises(BackendBatchContractError, match="identities must be unique"):
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


_CONTRACT_CASES: tuple[Callable[[], None], ...] = (
    _valid_all_rows,
    _valid_selected_row_order,
    _valid_frozen_batch_sizes,
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
    assert "step_batch" not in SimBackend.__abstractmethods__
    assert "reset_batch" not in SimBackend.__abstractmethods__
    with pytest.raises(NotImplementedError, match="typed backend batches"):
        SimBackend.bind_task_io(backend, _requirements())
    with pytest.raises(NotImplementedError, match="typed backend batches"):
        SimBackend.step_batch(backend, fake.plan, control)
    with pytest.raises(NotImplementedError, match="typed backend batches"):
        SimBackend.reset_batch(backend, fake.plan, rows)
