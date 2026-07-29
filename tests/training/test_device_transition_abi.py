"""Real-CUDA ABI tests for ``mjwarp`` device-resident typed batches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
import torch
from torch.utils.dlpack import from_dlpack

from unilab.base.backend import (
    BackendBatchContractError,
    BackendIORequirements,
    BoundFieldIdentity,
    BufferContract,
    BufferLayout,
    BufferLifetime,
    BufferMutability,
    BufferOwner,
    BufferPlacement,
    BufferView,
    ControlBatch,
    ControlSpec,
    DeviceBufferContractError,
    DeviceBufferLease,
    DeviceCompletion,
    DeviceTensorView,
    ExecutionProfile,
    PhysicalUnit,
    ReferenceFrame,
    RowSelection,
    StateEntityKind,
    StateFieldKind,
    StateFieldSpec,
    create_backend,
)
from unilab.base.backend.mjwarp.dependencies import load_mjwarp_dependencies
from unilab.base.scene import SceneCfg

pytestmark = pytest.mark.slow


@dataclass(frozen=True)
class _DevicePlanFixture:
    backend: Any
    plan: Any
    control_contract: BufferContract
    placement: BufferPlacement


def _require_cuda() -> None:
    dependencies = load_mjwarp_dependencies()
    if not bool(dependencies.warp.get_device().is_cuda):
        pytest.fail("device transition ABI requires an active CUDA Warp device")


def _backend(num_envs: int) -> Any:
    _require_cuda()
    from unilab.assets import ASSETS_ROOT_PATH

    return create_backend(
        "mjwarp",
        SceneCfg(model_file=str(ASSETS_ROOT_PATH / "robots" / "g1" / "scene_flat.xml")),
        num_envs,
        0.02 / 3.0,
        base_name="pelvis",
    )


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


def _control_contract(placement: BufferPlacement) -> BufferContract:
    return BufferContract(
        row_shape=(29,),
        dtype="float32",
        layout=BufferLayout.C_CONTIGUOUS,
        placement=placement,
        owner=BufferOwner.RUNNER,
        mutability=BufferMutability.READ_ONLY,
        lifetime=BufferLifetime.UNTIL_STEP_COMPLETE,
        dlpack_exportable=True,
    )


def _fixture(num_envs: int) -> _DevicePlanFixture:
    backend = _backend(num_envs)
    placement = BufferPlacement.device("cuda", 0)
    base_id = int(backend.get_body_ids(("pelvis",))[0])
    fields = (
        StateFieldSpec(
            semantic_key="root.position",
            identity=BoundFieldIdentity(
                StateEntityKind.ROOT,
                StateFieldKind.POSITION,
                (base_id,),
            ),
            frame=ReferenceFrame.WORLD,
            unit=PhysicalUnit.METER,
            buffer=_state_contract(placement, (3,)),
        ),
        StateFieldSpec(
            semantic_key="root.orientation",
            identity=BoundFieldIdentity(
                StateEntityKind.ROOT,
                StateFieldKind.ORIENTATION,
                (base_id,),
            ),
            frame=ReferenceFrame.WORLD,
            unit=PhysicalUnit.QUATERNION,
            buffer=_state_contract(placement, (4,)),
        ),
        StateFieldSpec(
            semantic_key="dof.position",
            identity=BoundFieldIdentity(
                StateEntityKind.DOF,
                StateFieldKind.POSITION,
                tuple(range(29)),
            ),
            frame=ReferenceFrame.JOINT,
            unit=PhysicalUnit.RADIAN,
            buffer=_state_contract(placement, (29,)),
        ),
        StateFieldSpec(
            semantic_key="sensor.torso_gyro",
            identity=BoundFieldIdentity(
                StateEntityKind.SENSOR,
                StateFieldKind.VALUE,
                tuple(int(value) for value in backend.get_sensor_ids(("torso_gyro",))),
            ),
            frame=ReferenceFrame.SENSOR,
            unit=PhysicalUnit.RADIAN_PER_SECOND,
            buffer=_state_contract(placement, (3,)),
        ),
    )
    control_contract = _control_contract(placement)
    plan = backend.bind_task_io(
        BackendIORequirements(
            state_fields=fields,
            control=ControlSpec("joint.position_target", control_contract, 1),
            execution_profile=ExecutionProfile.DEVICE_RESIDENT,
        )
    )
    return _DevicePlanFixture(
        backend=backend,
        plan=plan,
        control_contract=control_contract,
        placement=placement,
    )


def _control_batch(
    fixture: _DevicePlanFixture,
    *,
    completion: DeviceCompletion | None,
    record_completion: bool = False,
) -> tuple[ControlBatch, DeviceBufferLease]:
    if completion is not None and record_completion:
        raise ValueError("test control batch cannot receive and record a completion event")
    action = torch.zeros(
        (fixture.backend.num_envs, 29),
        dtype=torch.float32,
        device="cuda:0",
    )
    lease = DeviceBufferLease("test-runner-action")
    if record_completion:
        completion = DeviceCompletion.record(
            placement=fixture.placement,
            owner_id=lease.owner_id,
            epoch=lease.epoch,
        )
    view = DeviceTensorView(
        tensor_handle=action,
        contract=fixture.control_contract,
        lease=lease,
        completion=completion,
    )
    return (
        ControlBatch(
            plan=fixture.plan,
            rows=RowSelection.all(fixture.backend.num_envs),
            buffer=BufferView(
                handle=view,
                shape=tuple(int(dim) for dim in action.shape),
                contract=fixture.control_contract,
            ),
        ),
        lease,
    )


@pytest.mark.parametrize("num_envs", [1, 128, 4096])
def test_dlpack_pointer_shape_dtype_and_lifetime(num_envs: int) -> None:
    """Device StateBatch is stable, zero-copy and invalidated at the next barrier."""

    fixture = _fixture(num_envs)
    first = fixture.backend.read_state_batch(fixture.plan, RowSelection.all(num_envs)).state
    view = first.buffer("root.position").handle
    assert isinstance(view, DeviceTensorView)
    assert view.shape == (num_envs, 3)
    assert view.dtype == "float32"
    assert view.contract.placement == fixture.placement
    assert view.contract.owner is BufferOwner.BACKEND

    # A DLPack consumer must see the exact backend-owned state-pack address.
    view.wait()
    imported = from_dlpack(view)
    assert imported.device == torch.device("cuda:0")
    assert imported.dtype is torch.float32
    assert imported.data_ptr() == view.data_ptr

    # A raw capsule is single-consumption by the DLPack runtime.  The view
    # produces a fresh capsule per explicit export, never a reusable one.
    capsule = view.__dlpack__()
    again = from_dlpack(capsule)
    assert again.data_ptr() == view.data_ptr
    with pytest.raises(RuntimeError):
        from_dlpack(capsule)

    assert view.owner_id == fixture.plan.backend_instance_id
    control, _ = _control_batch(fixture, completion=None, record_completion=True)
    fixture.backend.step_batch(fixture.plan, control, nsteps=1)

    # Both StateBatch and the exported DLPack path fail after the backend's
    # next mutation barrier; no stale pointer is accepted as current state.
    with pytest.raises(DeviceBufferContractError, match="stale"):
        view.torch()
    with pytest.raises(DeviceBufferContractError, match="stale"):
        view.__dlpack__()


def test_device_action_requires_explicit_producer_completion() -> None:
    fixture = _fixture(128)
    control, _ = _control_batch(fixture, completion=None)
    with pytest.raises(DeviceBufferContractError, match="no producer completion"):
        fixture.backend.step_batch(fixture.plan, control, nsteps=1)


def test_stale_completion_epoch_is_rejected_before_physics() -> None:
    fixture = _fixture(1)
    action = torch.zeros((1, 29), dtype=torch.float32, device="cuda:0")
    lease = DeviceBufferLease("stale-action")
    completion = DeviceCompletion.record(
        placement=fixture.placement,
        owner_id="stale-action",
        epoch=lease.epoch,
    )
    lease.invalidate()
    with pytest.raises(DeviceBufferContractError, match="completion epoch"):
        DeviceTensorView(
            tensor_handle=action,
            contract=fixture.control_contract,
            lease=lease,
            completion=completion,
        )


@pytest.mark.parametrize(
    ("dtype", "shape", "device", "message"),
    (
        (torch.float64, (1, 29), "cuda:0", "dtype"),
        (torch.float32, (1, 28), "cuda:0", "row shape"),
        (torch.float32, (1, 29), "cpu", "on cpu"),
    ),
)
def test_device_view_rejects_wrong_dtype_shape_or_placement(
    dtype: torch.dtype,
    shape: tuple[int, int],
    device: str,
    message: str,
) -> None:
    fixture = _fixture(1)
    action = torch.zeros(shape, dtype=dtype, device=device)
    lease = DeviceBufferLease("invalid-device-view")
    completion = DeviceCompletion.record(
        placement=fixture.placement,
        owner_id=lease.owner_id,
        epoch=lease.epoch,
    )
    with pytest.raises(DeviceBufferContractError, match=message):
        DeviceTensorView(
            tensor_handle=action,
            contract=fixture.control_contract,
            lease=lease,
            completion=completion,
        )


def test_completion_owner_must_match_the_device_buffer_lease() -> None:
    fixture = _fixture(1)
    action = torch.zeros((1, 29), dtype=torch.float32, device="cuda:0")
    lease = DeviceBufferLease("runner-a")
    completion = DeviceCompletion.record(
        placement=fixture.placement,
        owner_id="runner-b",
        epoch=lease.epoch,
    )
    with pytest.raises(DeviceBufferContractError, match="completion owner"):
        DeviceTensorView(
            tensor_handle=action,
            contract=fixture.control_contract,
            lease=lease,
            completion=completion,
        )


def test_device_control_epoch_cannot_be_reused_after_a_physics_barrier() -> None:
    fixture = _fixture(1)
    control, lease = _control_batch(fixture, completion=None, record_completion=True)
    old_view = control.buffer.handle
    assert isinstance(old_view, DeviceTensorView)
    action = old_view.torch()

    fixture.backend.step_batch(fixture.plan, control, nsteps=1)
    with pytest.raises(DeviceBufferContractError, match="already consumed"):
        fixture.backend.step_batch(fixture.plan, control, nsteps=1)

    # The runner owns the lease and explicitly creates the next policy write.
    lease.invalidate()
    completion = DeviceCompletion.record(
        placement=fixture.placement,
        owner_id=lease.owner_id,
        epoch=lease.epoch,
    )
    renewed_view = DeviceTensorView(
        tensor_handle=action,
        contract=fixture.control_contract,
        lease=lease,
        completion=completion,
    )
    renewed_control = ControlBatch(
        plan=fixture.plan,
        rows=RowSelection.all(1),
        buffer=BufferView(
            handle=renewed_view,
            shape=(1, 29),
            contract=fixture.control_contract,
        ),
    )
    fixture.backend.step_batch(fixture.plan, renewed_control, nsteps=1)


def test_device_control_from_another_backend_plan_is_rejected_before_physics() -> None:
    source = _fixture(1)
    target = _fixture(1)
    control, _ = _control_batch(source, completion=None, record_completion=True)
    with pytest.raises(BackendBatchContractError, match="different backend plan"):
        target.backend.step_batch(target.plan, control, nsteps=1)
