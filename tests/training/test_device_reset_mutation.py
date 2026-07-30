"""Real-CUDA tests for the all-world masked ``mjwarp`` device reset ABI.

The production envelope intentionally keeps the done-row set on CUDA: the
typed descriptor is always ``RowSelection.all`` and a manager-owned CUDA bool
mask selects which worlds receive reset values.  These tests make the only
D2H copies at an explicit test oracle boundary after waiting for the backend
completion event.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import numpy as np
import pytest
import torch

from unilab.base.backend import (
    BackendBatchContractError,
    BackendIORequirements,
    BoundFieldIdentity,
    BoundMutationValueBuffers,
    BufferContract,
    BufferLayout,
    BufferLifetime,
    BufferMutability,
    BufferOwner,
    BufferPlacement,
    BufferView,
    ControlSpec,
    DeviceBufferContractError,
    DeviceBufferLease,
    DeviceCompletion,
    DeviceResetMutationBatch,
    DeviceTensorView,
    ExecutionProfile,
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
    StateBatchPhase,
    StateEntityKind,
    StateFieldKind,
    StateFieldSpec,
    TypedBackendMutationBatch,
    create_backend,
)
from unilab.base.backend.mjwarp.dependencies import load_mjwarp_dependencies
from unilab.base.scene import SceneCfg

pytestmark = pytest.mark.slow

_BASE = "pelvis"
_HINGE = "left_hip_pitch_joint"


@dataclass(frozen=True)
class _DeviceResetFixture:
    backend: Any
    plan: Any
    mutation_plan: Any
    placement: BufferPlacement
    position_index: int
    velocity_index: int


def _require_cuda() -> None:
    dependencies = load_mjwarp_dependencies()
    if not bool(dependencies.warp.get_device().is_cuda):
        pytest.fail("mjwarp device reset tests require an active CUDA Warp device")


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


def _control_contract(placement: BufferPlacement, num_actuators: int) -> BufferContract:
    return BufferContract(
        row_shape=(num_actuators,),
        dtype="float32",
        layout=BufferLayout.C_CONTIGUOUS,
        placement=placement,
        owner=BufferOwner.RUNNER,
        mutability=BufferMutability.READ_ONLY,
        lifetime=BufferLifetime.UNTIL_STEP_COMPLETE,
        dlpack_exportable=True,
    )


def _value_contract(placement: BufferPlacement, row_shape: tuple[int, ...]) -> BufferContract:
    return BufferContract(
        row_shape=row_shape,
        dtype="float32",
        layout=BufferLayout.C_CONTIGUOUS,
        placement=placement,
        owner=BufferOwner.MANAGER,
        mutability=BufferMutability.READ_ONLY,
        lifetime=BufferLifetime.UNTIL_COMMIT,
        dlpack_exportable=True,
    )


def _mask_contract(placement: BufferPlacement) -> BufferContract:
    return BufferContract(
        row_shape=(),
        dtype="bool",
        layout=BufferLayout.C_CONTIGUOUS,
        placement=placement,
        owner=BufferOwner.MANAGER,
        mutability=BufferMutability.READ_ONLY,
        lifetime=BufferLifetime.UNTIL_COMMIT,
        dlpack_exportable=True,
    )


def _field(
    key: str,
    *,
    entity_kind: StateEntityKind,
    field_kind: StateFieldKind,
    entity_ids: tuple[int, ...],
    row_shape: tuple[int, ...],
    frame: ReferenceFrame,
    unit: PhysicalUnit,
    placement: BufferPlacement,
) -> StateFieldSpec:
    return StateFieldSpec(
        semantic_key=key,
        identity=BoundFieldIdentity(entity_kind, field_kind, entity_ids),
        frame=frame,
        unit=unit,
        buffer=_state_contract(placement, row_shape),
    )


def _reset_spec(
    *,
    term_key: str,
    target_key: str,
    entity_kind: MutationEntityKind,
    field_kind: MutationFieldKind,
    selector: str,
    row_shape: tuple[int, ...],
    placement: BufferPlacement,
) -> MutationSpec:
    return MutationSpec(
        term_key=term_key,
        target=MutationTargetSpec(
            target_key=target_key,
            target_kind=MutationTargetKind.SIMULATION_STATE,
            entity_kind=entity_kind,
            field_kind=field_kind,
            selector=selector,
        ),
        trigger=MutationTrigger.RESET,
        commit_phase=MutationCommitPhase.RESET,
        operation=MutationOperation.SET,
        baseline=MutationBaseline.DEFAULT,
        persistence=MutationPersistence.EPISODE,
        recompute=MutationRecomputeLevel.KINEMATICS,
        value_template=_value_contract(placement, row_shape),
    )


def _fixture(num_envs: int) -> _DeviceResetFixture:
    _require_cuda()
    from unilab.assets import ASSETS_ROOT_PATH

    device_index = int(torch.cuda.current_device())
    placement = BufferPlacement.device("cuda", device_index)
    backend = create_backend(
        "mjwarp",
        SceneCfg(model_file=str(ASSETS_ROOT_PATH / "robots" / "g1" / "scene_flat.xml")),
        num_envs,
        0.02 / 3.0,
        base_name=_BASE,
    )
    base_id = int(backend.get_body_ids((_BASE,))[0])
    position_index = int(backend.get_joint_dof_pos_indices((_HINGE,))[0])
    velocity_index = int(backend.get_joint_dof_vel_indices((_HINGE,))[0])
    all_dofs = tuple(range(backend.num_dof_vel))
    fields = (
        _field(
            "root.position",
            entity_kind=StateEntityKind.ROOT,
            field_kind=StateFieldKind.POSITION,
            entity_ids=(base_id,),
            row_shape=(3,),
            frame=ReferenceFrame.WORLD,
            unit=PhysicalUnit.METER,
            placement=placement,
        ),
        _field(
            "root.orientation",
            entity_kind=StateEntityKind.ROOT,
            field_kind=StateFieldKind.ORIENTATION,
            entity_ids=(base_id,),
            row_shape=(4,),
            frame=ReferenceFrame.WORLD,
            unit=PhysicalUnit.QUATERNION,
            placement=placement,
        ),
        _field(
            "root.linear_velocity",
            entity_kind=StateEntityKind.ROOT,
            field_kind=StateFieldKind.LINEAR_VELOCITY,
            entity_ids=(base_id,),
            row_shape=(3,),
            frame=ReferenceFrame.WORLD,
            unit=PhysicalUnit.METER_PER_SECOND,
            placement=placement,
        ),
        _field(
            "root.angular_velocity",
            entity_kind=StateEntityKind.ROOT,
            field_kind=StateFieldKind.ANGULAR_VELOCITY,
            entity_ids=(base_id,),
            row_shape=(3,),
            frame=ReferenceFrame.WORLD,
            unit=PhysicalUnit.RADIAN_PER_SECOND,
            placement=placement,
        ),
        _field(
            "dof.position",
            entity_kind=StateEntityKind.DOF,
            field_kind=StateFieldKind.POSITION,
            entity_ids=all_dofs,
            row_shape=(backend.num_dof_vel,),
            frame=ReferenceFrame.JOINT,
            unit=PhysicalUnit.RADIAN,
            placement=placement,
        ),
        _field(
            "dof.angular_velocity",
            entity_kind=StateEntityKind.DOF,
            field_kind=StateFieldKind.ANGULAR_VELOCITY,
            entity_ids=all_dofs,
            row_shape=(backend.num_dof_vel,),
            frame=ReferenceFrame.JOINT,
            unit=PhysicalUnit.RADIAN_PER_SECOND,
            placement=placement,
        ),
    )
    plan = backend.bind_task_io(
        BackendIORequirements(
            state_fields=fields,
            control=ControlSpec(
                "joint.position_target",
                _control_contract(placement, backend.num_actuators),
                1,
            ),
            execution_profile=ExecutionProfile.DEVICE_RESIDENT,
        )
    )
    specs = (
        _reset_spec(
            term_key="reset.root.position",
            target_key="state.root.position",
            entity_kind=MutationEntityKind.BODY,
            field_kind=MutationFieldKind.POSITION,
            selector=_BASE,
            row_shape=(3,),
            placement=placement,
        ),
        _reset_spec(
            term_key="reset.root.orientation",
            target_key="state.root.orientation",
            entity_kind=MutationEntityKind.BODY,
            field_kind=MutationFieldKind.ORIENTATION,
            selector=_BASE,
            row_shape=(4,),
            placement=placement,
        ),
        _reset_spec(
            term_key="reset.root.linear_velocity",
            target_key="state.root.linear_velocity",
            entity_kind=MutationEntityKind.BODY,
            field_kind=MutationFieldKind.LINEAR_VELOCITY,
            selector=_BASE,
            row_shape=(3,),
            placement=placement,
        ),
        _reset_spec(
            term_key="reset.root.angular_velocity",
            target_key="state.root.angular_velocity",
            entity_kind=MutationEntityKind.BODY,
            field_kind=MutationFieldKind.ANGULAR_VELOCITY,
            selector=_BASE,
            row_shape=(3,),
            placement=placement,
        ),
        _reset_spec(
            term_key="reset.hinge.position",
            target_key="state.dof.position",
            entity_kind=MutationEntityKind.DOF,
            field_kind=MutationFieldKind.POSITION,
            selector=_HINGE,
            row_shape=(1,),
            placement=placement,
        ),
        _reset_spec(
            term_key="reset.hinge.angular_velocity",
            target_key="state.dof.angular_velocity",
            entity_kind=MutationEntityKind.DOF,
            field_kind=MutationFieldKind.ANGULAR_VELOCITY,
            selector=_HINGE,
            row_shape=(1,),
            placement=placement,
        ),
    )
    return _DeviceResetFixture(
        backend=backend,
        plan=plan,
        mutation_plan=backend.bind_mutation_plan(specs),
        placement=placement,
        position_index=position_index,
        velocity_index=velocity_index,
    )


def _copy_state_to_oracle(state: Any, *, stream: torch.cuda.Stream) -> dict[str, torch.Tensor]:
    copies: dict[str, torch.Tensor] = {}
    with torch.cuda.stream(stream):
        for key in (
            "root.position",
            "root.orientation",
            "root.linear_velocity",
            "root.angular_velocity",
            "dof.position",
            "dof.angular_velocity",
        ):
            view = state.buffer(key).handle
            assert isinstance(view, DeviceTensorView)
            view.wait(stream)
            copies[key] = view.torch().clone()
    stream.synchronize()  # Explicit test-only oracle boundary.
    return copies


def _device_reset_batch(
    fixture: _DeviceResetFixture,
    *,
    active_mask: torch.Tensor,
    values: dict[str, torch.Tensor],
    lease: DeviceBufferLease,
    completion: DeviceCompletion,
) -> DeviceResetMutationBatch:
    rows = RowSelection.all(fixture.backend.num_envs)
    mask_contract = _mask_contract(fixture.placement)
    mask_view = DeviceTensorView(
        tensor_handle=active_mask,
        contract=mask_contract,
        lease=lease,
        completion=completion,
    )
    entries: list[MutationValueBatch] = []
    for field_index, spec in enumerate(fixture.mutation_plan.specs):
        tensor = values[spec.term_key]
        view = DeviceTensorView(
            tensor_handle=tensor,
            contract=spec.value_buffer,
            lease=lease,
            completion=completion,
        )
        entries.append(
            MutationValueBatch(
                plan=fixture.mutation_plan,
                field_index=field_index,
                rows=rows,
                buffer=BufferView(
                    handle=view,
                    shape=tuple(int(dim) for dim in tensor.shape),
                    contract=spec.value_buffer,
                ),
            )
        )
    mutation = TypedBackendMutationBatch(
        plan=fixture.mutation_plan,
        rows=rows,
        state=SimulationStateMutationBatch(tuple(entries)),
    )
    return DeviceResetMutationBatch(
        plan=fixture.mutation_plan,
        rows=rows,
        mutation=mutation,
        active_mask=BufferView(
            handle=mask_view,
            shape=tuple(int(dim) for dim in active_mask.shape),
            contract=mask_contract,
        ),
    )


def _reset_values(fixture: _DeviceResetFixture) -> dict[str, torch.Tensor]:
    device = torch.device(f"cuda:{fixture.placement.device_index}")
    count = fixture.backend.num_envs
    values: dict[str, torch.Tensor] = {}
    for spec in fixture.mutation_plan.specs:
        shape = (count, *spec.value_buffer.row_shape)
        values[spec.term_key] = torch.zeros(shape, dtype=torch.float32, device=device)
    values["reset.root.position"][:, 0, :] = torch.tensor(
        (0.11, -0.07, 0.92), dtype=torch.float32, device=device
    )
    values["reset.root.orientation"][:, 0, 0] = 1.0
    values["reset.root.linear_velocity"][:, 0, :] = torch.tensor(
        (0.03, -0.02, 0.01), dtype=torch.float32, device=device
    )
    values["reset.root.angular_velocity"][:, 0, :] = torch.tensor(
        (-0.04, 0.05, -0.06), dtype=torch.float32, device=device
    )
    values["reset.hinge.position"][:, 0, 0] = 0.17
    values["reset.hinge.angular_velocity"][:, 0, 0] = -0.23
    return values


@pytest.mark.parametrize("num_envs", (1, 32))
def test_device_masked_reset_preserves_complement_without_host_roundtrip(num_envs: int) -> None:
    """Initial/all/partial reset is CUDA-only and stale terminal views expire."""

    fixture = _fixture(num_envs)
    initial = fixture.backend.read_state_batch(
        fixture.plan, RowSelection.all(num_envs), phase=StateBatchPhase.CURRENT
    ).state
    stale_view = initial.buffer("root.position").handle
    assert isinstance(stale_view, DeviceTensorView)
    oracle_stream = cast(
        torch.cuda.Stream,
        torch.cuda.Stream(device=f"cuda:{fixture.placement.device_index}"),
    )
    before = _copy_state_to_oracle(initial, stream=oracle_stream)

    device = torch.device(f"cuda:{fixture.placement.device_index}")
    mask = torch.zeros((num_envs,), dtype=torch.bool, device=device)
    mask[::2] = True
    values = _reset_values(fixture)
    lease = DeviceBufferLease("device-reset-manager")
    producer_stream = cast(torch.cuda.Stream, torch.cuda.Stream(device=device))
    producer_stream.wait_stream(cast(torch.cuda.Stream, torch.cuda.current_stream(device)))
    with torch.cuda.stream(producer_stream):
        # Values were prepared on the current stream.  The reset producer
        # stream takes that explicit dependency before publishing one event.
        completion = DeviceCompletion.record(
            placement=fixture.placement,
            owner_id=lease.owner_id,
            epoch=lease.epoch,
            stream=producer_stream,
        )
    reset = _device_reset_batch(
        fixture,
        active_mask=mask,
        values=values,
        lease=lease,
        completion=completion,
    )

    with (
        patch("torch.cuda.synchronize", side_effect=AssertionError("global sync is forbidden")),
        patch.object(fixture.backend, "set_state", side_effect=AssertionError("legacy reset")),
        patch.object(fixture.backend, "get_base_pos", side_effect=AssertionError("getter")),
        patch.object(fixture.backend, "get_base_quat", side_effect=AssertionError("getter")),
        patch.object(fixture.backend, "get_base_lin_vel", side_effect=AssertionError("getter")),
        patch.object(fixture.backend, "get_base_ang_vel", side_effect=AssertionError("getter")),
        patch.object(fixture.backend, "get_dof_pos", side_effect=AssertionError("getter")),
        patch.object(fixture.backend, "get_dof_vel", side_effect=AssertionError("getter")),
        patch.object(
            fixture.backend,
            "_resolve_mjwarp_typed_mutation_selector",
            side_effect=AssertionError("selector"),
        ),
        patch.object(Path, "read_text", side_effect=AssertionError("asset")),
        patch.object(Path, "read_bytes", side_effect=AssertionError("asset")),
    ):
        result = fixture.backend.reset_batch(
            fixture.plan,
            RowSelection.all(num_envs),
            mutation_batch=reset,
        )

    assert result.reset_state.phase is StateBatchPhase.RESET
    with pytest.raises(DeviceBufferContractError, match="stale"):
        stale_view.torch()
    counters = result.diagnostics.counters
    assert counters.host_to_device_transfers == 0
    assert counters.device_to_host_transfers == 0
    assert counters.global_synchronizations == 0
    assert counters.instrumentation_complete

    after = _copy_state_to_oracle(result.reset_state, stream=oracle_stream)
    expected = {key: value.clone() for key, value in before.items()}
    expected["root.position"] = torch.where(
        mask[:, None], values["reset.root.position"][:, 0, :], before["root.position"]
    )
    expected["root.orientation"] = torch.where(
        mask[:, None], values["reset.root.orientation"][:, 0, :], before["root.orientation"]
    )
    expected["root.linear_velocity"] = torch.where(
        mask[:, None],
        values["reset.root.linear_velocity"][:, 0, :],
        before["root.linear_velocity"],
    )
    expected["root.angular_velocity"] = torch.where(
        mask[:, None],
        values["reset.root.angular_velocity"][:, 0, :],
        before["root.angular_velocity"],
    )
    expected["dof.position"][:, fixture.position_index] = torch.where(
        mask,
        values["reset.hinge.position"][:, 0, 0],
        before["dof.position"][:, fixture.position_index],
    )
    expected["dof.angular_velocity"][:, fixture.velocity_index] = torch.where(
        mask,
        values["reset.hinge.angular_velocity"][:, 0, 0],
        before["dof.angular_velocity"][:, fixture.velocity_index],
    )
    for key in expected:
        torch.testing.assert_close(after[key].cpu(), expected[key].cpu(), atol=2.0e-5, rtol=2.0e-5)


def test_device_reset_rejects_host_or_foreign_envelopes_before_physics() -> None:
    fixture = _fixture(4)
    device = torch.device(f"cuda:{fixture.placement.device_index}")
    values = _reset_values(fixture)
    mask = torch.ones((4,), dtype=torch.bool, device=device)
    lease = DeviceBufferLease("device-reset-manager")
    completion = DeviceCompletion.record(
        placement=fixture.placement,
        owner_id=lease.owner_id,
        epoch=lease.epoch,
    )
    valid = _device_reset_batch(
        fixture,
        active_mask=mask,
        values=values,
        lease=lease,
        completion=completion,
    )
    host_buffers = BoundMutationValueBuffers(
        plan=fixture.mutation_plan,
        buffers=tuple(
            np.zeros(
                (fixture.mutation_plan.num_envs, *spec.value_buffer.row_shape),
                dtype=spec.value_buffer.dtype,
            )
            for spec in fixture.mutation_plan.specs
        ),
    )
    host_window_mutation = TypedBackendMutationBatch(
        plan=fixture.mutation_plan,
        rows=RowSelection.all(4),
        state=SimulationStateMutationBatch(
            bound_buffer_window=host_buffers.window(RowSelection.all(4))
        ),
    )
    with pytest.raises(
        BackendBatchContractError,
        match="does not support cold-bound host state buffers",
    ):
        DeviceResetMutationBatch(
            plan=fixture.mutation_plan,
            rows=RowSelection.all(4),
            mutation=host_window_mutation,
            active_mask=valid.active_mask,
        )

    with pytest.raises(BackendBatchContractError, match="DeviceResetMutationBatch"):
        fixture.backend.reset_batch(
            fixture.plan,
            RowSelection.all(4),
            mutation_batch=valid.mutation,
        )
    with pytest.raises(BackendBatchContractError, match="RowSelection.all"):
        fixture.backend.reset_batch(
            fixture.plan,
            RowSelection.selected(4, (1,)),
            mutation_batch=valid,
        )
    fixture.backend.reset_batch(
        fixture.plan,
        RowSelection.all(4),
        mutation_batch=valid,
    )
    with pytest.raises(DeviceBufferContractError, match="already committed"):
        fixture.backend.reset_batch(
            fixture.plan,
            RowSelection.all(4),
            mutation_batch=valid,
        )
    foreign = _fixture(4)
    with pytest.raises(
        (BackendBatchContractError, DeviceBufferContractError), match="different|bound"
    ):
        foreign.backend.reset_batch(
            foreign.plan,
            RowSelection.all(4),
            mutation_batch=valid,
        )
