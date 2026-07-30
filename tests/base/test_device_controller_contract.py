"""Real-CUDA acceptance for the mjwarp device substep controller contract."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import mujoco
import numpy as np
import pytest
import torch
from numpy.testing import assert_allclose

from unilab.base.backend import (
    BackendBatchContractError,
    BackendBatchCounterBudget,
    BackendIORequirements,
    BoundBackendPlan,
    BoundFieldIdentity,
    BufferContract,
    BufferLayout,
    BufferLifetime,
    BufferMutability,
    BufferOwner,
    BufferPlacement,
    BufferView,
    ControlBatch,
    ControlImplementation,
    ControllerParameter,
    ControllerStateRead,
    ControlSpec,
    DeviceBufferLease,
    DeviceCompletion,
    DeviceControllerSpec,
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
from unilab.base.backend.device import DeviceCompletion as DeviceCompletionContract
from unilab.base.backend.mjwarp.backend import MjwarpBackend
from unilab.base.backend.mjwarp.controller import MJWARP_JOINT_PD_TORQUE_CONTROLLER
from unilab.base.backend.mjwarp.dependencies import load_mjwarp_dependencies
from unilab.base.scene import SceneCfg

pytestmark = pytest.mark.slow

_MUJOCO: Any = mujoco

_NUM_WORLDS = 32
_SIM_DT = 0.002
_CADENCES = (2, 4)
_SEEDS = (0, 1)
_ATOL = 3.0e-4
_RTOL = 2.0e-3


@dataclass(frozen=True)
class _ControllerCase:
    plan: BoundBackendPlan
    command: torch.Tensor
    lease: DeviceBufferLease
    event: torch.cuda.Event
    producer_stream: torch.cuda.Stream


def _scene_path() -> Path:
    from unilab.assets import ASSETS_ROOT_PATH

    return Path(ASSETS_ROOT_PATH) / "robots" / "go2w" / "scene_flat.xml"


def _require_cuda() -> None:
    dependencies = load_mjwarp_dependencies()
    if not bool(dependencies.warp.get_device().is_cuda):
        pytest.fail("device controller acceptance requires an active CUDA Warp device")


def _state_buffer(placement: BufferPlacement, width: int) -> BufferContract:
    return BufferContract(
        row_shape=(width,),
        dtype="float32",
        layout=BufferLayout.C_CONTIGUOUS,
        placement=placement,
        owner=BufferOwner.BACKEND,
        mutability=BufferMutability.READ_ONLY,
        lifetime=BufferLifetime.BORROWED_UNTIL_MUTATION,
        dlpack_exportable=True,
        address_stable=True,
    )


def _control_buffer(placement: BufferPlacement, width: int) -> BufferContract:
    return BufferContract(
        row_shape=(width,),
        dtype="float32",
        layout=BufferLayout.C_CONTIGUOUS,
        placement=placement,
        owner=BufferOwner.RUNNER,
        mutability=BufferMutability.READ_ONLY,
        lifetime=BufferLifetime.UNTIL_STEP_COMPLETE,
        dlpack_exportable=True,
        address_stable=True,
    )


def _fields(backend: Any, placement: BufferPlacement) -> tuple[StateFieldSpec, ...]:
    width = int(backend.num_dof_vel)
    ids = tuple(range(width))
    return (
        StateFieldSpec(
            semantic_key="dof.position",
            identity=BoundFieldIdentity(
                StateEntityKind.DOF,
                StateFieldKind.POSITION,
                ids,
            ),
            frame=ReferenceFrame.JOINT,
            unit=PhysicalUnit.RADIAN,
            buffer=_state_buffer(placement, width),
        ),
        StateFieldSpec(
            semantic_key="dof.velocity",
            identity=BoundFieldIdentity(
                StateEntityKind.DOF,
                StateFieldKind.ANGULAR_VELOCITY,
                ids,
            ),
            frame=ReferenceFrame.JOINT,
            unit=PhysicalUnit.RADIAN_PER_SECOND,
            buffer=_state_buffer(placement, width),
        ),
    )


def _parameters(width: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.arange(width, dtype=np.float32)
    stiffness = 18.0 + 0.25 * indices
    damping = 0.6 + 0.015 * indices
    effort_limit = np.full((width,), 10.0, dtype=np.float32)
    return stiffness, damping, effort_limit


def _requirements(
    backend: Any,
    placement: BufferPlacement,
    *,
    cadence: int,
    controller: DeviceControllerSpec | None,
    fields: tuple[StateFieldSpec, ...] | None = None,
) -> BackendIORequirements:
    implementation = (
        ControlImplementation.CONTROL_STEP_CONSTANT
        if controller is None
        else ControlImplementation.DEVICE_SUBSTEP_CONTROLLER
    )
    return BackendIORequirements(
        state_fields=_fields(backend, placement) if fields is None else fields,
        control=ControlSpec(
            semantic_key="joint.command",
            buffer=_control_buffer(placement, int(backend.num_actuators)),
            physics_substeps_per_control=cadence,
            implementation=implementation,
            controller=controller,
        ),
        execution_profile=ExecutionProfile.DEVICE_RESIDENT,
        hot_path_budget=BackendBatchCounterBudget(state_materializations=1),
    )


def _descriptor(backend: Any) -> DeviceControllerSpec:
    stiffness, damping, effort_limit = _parameters(int(backend.num_actuators))
    return DeviceControllerSpec(
        implementation_key=MJWARP_JOINT_PD_TORQUE_CONTROLLER,
        state_reads=(
            ControllerStateRead("dof.position"),
            ControllerStateRead("dof.velocity"),
        ),
        parameters=(
            ControllerParameter("damping", tuple(float(value) for value in damping)),
            ControllerParameter("effort_limit", tuple(float(value) for value in effort_limit)),
            ControllerParameter("stiffness", tuple(float(value) for value in stiffness)),
        ),
    )


def _initial_state(model: Any, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    key_id = _MUJOCO.mj_name2id(model, _MUJOCO.mjtObj.mjOBJ_KEY, "home")
    assert key_id >= 0
    home = np.asarray(model.key_qpos[key_id], dtype=np.float64)
    qpos = np.broadcast_to(home, (_NUM_WORLDS, model.nq)).copy()
    qpos[:, 2] = 1.0
    qvel = np.zeros((_NUM_WORLDS, model.nv), dtype=np.float64)
    qpos[:, 7:] += rng.uniform(-0.035, 0.035, size=(_NUM_WORLDS, model.nq - 7))
    qvel[:, :6] = rng.uniform(-0.015, 0.015, size=(_NUM_WORLDS, 6))
    qvel[:, 6:] = rng.uniform(-0.08, 0.08, size=(_NUM_WORLDS, model.nv - 6))

    actuator_ids = np.arange(model.nu, dtype=np.float64)
    world_ids = np.arange(_NUM_WORLDS, dtype=np.float64)[:, None]
    joint_ids = np.asarray(model.actuator_trnid[:, 0], dtype=np.intp)
    qpos_ids = np.asarray(model.jnt_qposadr[joint_ids], dtype=np.intp)
    command = home[qpos_ids][None, :] + 0.08 * np.sin(
        0.19 * actuator_ids[None, :] + 0.07 * world_ids + 0.11 * seed
    )
    return qpos.astype(np.float32), qvel.astype(np.float32), command.astype(np.float32)


def _reference(
    model_path: Path,
    qpos: np.ndarray,
    qvel: np.ndarray,
    command: np.ndarray,
    *,
    cadence: int,
    stale_control: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model = _MUJOCO.MjModel.from_xml_path(str(model_path))
    model.opt.timestep = _SIM_DT
    stiffness, damping, effort_limit = _parameters(model.nu)
    joint_ids = np.asarray(model.actuator_trnid[:, 0], dtype=np.intp)
    qpos_ids = np.asarray(model.jnt_qposadr[joint_ids], dtype=np.intp)
    qvel_ids = np.asarray(model.jnt_dofadr[joint_ids], dtype=np.intp)
    final_qpos = np.empty_like(qpos, dtype=np.float64)
    final_qvel = np.empty_like(qvel, dtype=np.float64)
    final_ctrl = np.empty_like(command, dtype=np.float64)

    for world in range(qpos.shape[0]):
        data = _MUJOCO.MjData(model)
        data.qpos[:] = qpos[world]
        data.qvel[:] = qvel[world]
        _MUJOCO.mj_forward(model, data)
        held = np.clip(
            stiffness * (command[world] - data.qpos[qpos_ids]) - damping * data.qvel[qvel_ids],
            -effort_limit,
            effort_limit,
        )
        for _ in range(cadence):
            if stale_control:
                data.ctrl[:] = held
            else:
                data.ctrl[:] = np.clip(
                    stiffness * (command[world] - data.qpos[qpos_ids])
                    - damping * data.qvel[qvel_ids],
                    -effort_limit,
                    effort_limit,
                )
            _MUJOCO.mj_step(model, data)
        final_qpos[world] = data.qpos
        final_qvel[world] = data.qvel
        final_ctrl[world] = data.ctrl
    return final_qpos, final_qvel, final_ctrl


def _make_case(
    backend: Any,
    plan: BoundBackendPlan,
    command: np.ndarray,
    device: torch.device,
) -> _ControllerCase:
    producer_stream = cast(torch.cuda.Stream, torch.cuda.Stream(device=device))
    return _ControllerCase(
        plan=plan,
        command=torch.as_tensor(command, dtype=torch.float32, device=device).clone(),
        lease=DeviceBufferLease(f"controller-command:{plan.fingerprint}"),
        event=cast(torch.cuda.Event, torch.cuda.Event(enable_timing=False)),
        producer_stream=producer_stream,
    )


def _step(backend: Any, case: _ControllerCase) -> Any:
    case.lease.invalidate()
    with torch.cuda.stream(case.producer_stream):
        completion = DeviceCompletion.record(
            placement=case.plan.control.buffer.placement,
            owner_id=case.lease.owner_id,
            epoch=case.lease.epoch,
            stream=case.producer_stream,
            event=case.event,
        )
    view = DeviceTensorView(
        tensor_handle=case.command,
        contract=case.plan.control.buffer,
        lease=case.lease,
        completion=completion,
    )
    return backend.step_batch(
        case.plan,
        ControlBatch(
            plan=case.plan,
            rows=RowSelection.all(_NUM_WORLDS),
            buffer=BufferView(
                handle=view,
                shape=tuple(int(dim) for dim in case.command.shape),
                contract=case.plan.control.buffer,
            ),
        ),
        nsteps=case.plan.control.physics_substeps_per_control,
    )


def _device_snapshot(backend: Any, result: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    completion_event = result.diagnostics.completion_event
    assert completion_event is not None
    completion = completion_event.handle
    assert isinstance(completion, DeviceCompletion)
    completion.event.synchronize()
    bridge = backend._ensure_device_bridge()
    return (
        bridge.qpos.detach().cpu().numpy().copy(),
        bridge.qvel.detach().cpu().numpy().copy(),
        bridge.ctrl.detach().cpu().numpy().copy(),
    )


def _assert_reference(
    actual: tuple[np.ndarray, np.ndarray, np.ndarray],
    expected: tuple[np.ndarray, np.ndarray, np.ndarray],
    *,
    context: str,
) -> None:
    for name, candidate, reference in zip(("qpos", "qvel", "ctrl"), actual, expected):
        assert_allclose(
            candidate,
            reference,
            atol=_ATOL,
            rtol=_RTOL,
            err_msg=f"{context} {name}",
        )


def test_device_controller_cadence_reads_and_host_rejection() -> None:
    """Prove fresh-state PD cadence, graph-only execution and fail-closed binding."""

    _require_cuda()
    scene_path = _scene_path()
    cpu_model = _MUJOCO.MjModel.from_xml_path(str(scene_path))
    cpu_model.opt.timestep = _SIM_DT
    backend = cast(
        MjwarpBackend,
        create_backend(
            "mjwarp",
            SceneCfg(model_file=str(scene_path)),
            _NUM_WORLDS,
            _SIM_DT,
            base_name="base_link",
        ),
    )
    try:
        bridge = backend._ensure_device_bridge()
        device = bridge.qpos.device
        device_index = device.index
        assert device_index is not None
        placement = BufferPlacement.device("cuda", int(device_index))
        descriptor = _descriptor(backend)
        controller_plans: dict[int, BoundBackendPlan] = {}
        actuator_joint_ids = np.asarray(cpu_model.actuator_trnid[:, 0], dtype=np.intp)
        expected_position_indices = (
            np.asarray(cpu_model.jnt_qposadr[actuator_joint_ids], dtype=np.int64)
            - backend._root_qpos_dim
        )
        expected_velocity_indices = (
            np.asarray(cpu_model.jnt_dofadr[actuator_joint_ids], dtype=np.int64)
            - backend._root_qvel_dim
        )
        identity = np.arange(cpu_model.nu, dtype=np.int64)
        assert not np.array_equal(expected_position_indices, identity)
        assert not np.array_equal(expected_velocity_indices, identity)

        for cadence in (1, *_CADENCES):
            plan = backend.bind_task_io(
                _requirements(
                    backend,
                    placement,
                    cadence=cadence,
                    controller=descriptor,
                )
            )
            controller_plans[cadence] = plan
            owner = backend._device_batch_plans[plan.fingerprint]
            assert owner.controller is not None
            assert owner.controller.native_ctrl.data_ptr() == bridge.ctrl.data_ptr()
            np.testing.assert_array_equal(
                owner.controller.position_indices.cpu().numpy(),
                expected_position_indices,
            )
            np.testing.assert_array_equal(
                owner.controller.velocity_indices.cpu().numpy(),
                expected_velocity_indices,
            )

        diagnostics = backend.get_device_graph_diagnostics(verify_storage=True)
        controller_buffers = tuple(
            item for item in diagnostics.storage_buffers if item.name.startswith("controller[")
        )
        assert controller_buffers
        assert len({item.name for item in controller_buffers}) == len(controller_buffers)

        for seed in _SEEDS:
            for cadence in _CADENCES:
                qpos, qvel, command = _initial_state(cpu_model, seed)
                backend.set_state(
                    np.arange(_NUM_WORLDS, dtype=np.int32),
                    qpos,
                    qvel,
                )
                expected = _reference(
                    scene_path,
                    qpos,
                    qvel,
                    command,
                    cadence=cadence,
                )
                stale = _reference(
                    scene_path,
                    qpos,
                    qvel,
                    command,
                    cadence=cadence,
                    stale_control=True,
                )
                case = _make_case(backend, controller_plans[cadence], command, device)
                actual = _device_snapshot(backend, _step(backend, case))
                _assert_reference(actual, expected, context=f"seed={seed} cadence={cadence}")
                assert float(np.max(np.abs(expected[1] - stale[1]))) > 1.0e-5
                assert float(np.max(np.abs(actual[1] - stale[1]))) > 1.0e-5

        qpos, qvel, command = _initial_state(cpu_model, 17)
        one_step_ctrl = _reference(
            scene_path,
            qpos,
            qvel,
            command,
            cadence=1,
        )[2].astype(np.float32)
        backend.set_state(np.arange(_NUM_WORLDS, dtype=np.int32), qpos, qvel)
        controller_case = _make_case(backend, controller_plans[1], command, device)
        controller_result = _device_snapshot(backend, _step(backend, controller_case))

        constant_plan = backend.bind_task_io(
            _requirements(
                backend,
                placement,
                cadence=1,
                controller=None,
            )
        )
        backend.set_state(np.arange(_NUM_WORLDS, dtype=np.int32), qpos, qvel)
        constant_case = _make_case(backend, constant_plan, one_step_ctrl, device)
        constant_result = _device_snapshot(backend, _step(backend, constant_case))
        _assert_reference(
            controller_result,
            constant_result,
            context="cadence=1 controller/constant parity",
        )

        warm_plan = controller_plans[_CADENCES[-1]]
        warm_owner = backend._device_batch_plans[warm_plan.fingerprint]
        assert warm_owner.controller is not None
        warm_case = _make_case(backend, warm_plan, command, device)
        warm_result = _step(backend, warm_case)
        warm_completion = warm_result.diagnostics.completion_event
        assert warm_completion is not None
        assert isinstance(warm_completion.handle, DeviceCompletion)
        warm_completion.handle.event.synchronize()
        addresses = warm_owner.controller.numeric_buffer_addresses
        graph_before = backend.get_device_graph_diagnostics(verify_storage=True)
        transfers_before = backend.get_transfer_counters()
        allocations_before = torch.cuda.memory_stats(device)["allocation.all.allocated"]
        waits = 0
        original_wait = DeviceCompletionContract.wait

        def counted_wait(
            completion: DeviceCompletionContract,
            stream: torch.cuda.Stream | None = None,
        ) -> None:
            nonlocal waits
            waits += 1
            original_wait(completion, stream)

        with (
            patch.object(DeviceCompletionContract, "wait", new=counted_wait),
            patch("torch.cuda.synchronize", side_effect=AssertionError("global synchronize")),
            patch.object(
                backend,
                "_upload",
                side_effect=AssertionError("warm controller H2D fallback"),
            ),
            patch.object(
                backend,
                "_download",
                side_effect=AssertionError("warm controller D2H fallback"),
            ),
            patch.object(
                backend,
                "_synchronize",
                side_effect=AssertionError("warm controller global sync"),
            ),
        ):
            _step(backend, warm_case)
            warm_result = _step(backend, warm_case)
        warm_completion = warm_result.diagnostics.completion_event
        assert warm_completion is not None
        assert isinstance(warm_completion.handle, DeviceCompletion)
        warm_completion.handle.event.synchronize()
        graph_after = backend.get_device_graph_diagnostics(verify_storage=True)
        assert waits == 2
        assert graph_after.launch_count == graph_before.launch_count + 2
        assert graph_after.capture_count == graph_before.capture_count
        assert graph_after.storage_buffers == graph_before.storage_buffers
        assert backend.get_transfer_counters() == transfers_before
        assert warm_owner.controller.numeric_buffer_addresses == addresses
        assert torch.cuda.memory_stats(device)["allocation.all.allocated"] == allocations_before

        graph_state = backend.get_device_graph_diagnostics(verify_storage=True)
        unsupported = replace(descriptor, implementation_key="mjwarp.unknown.v9")
        with pytest.raises(BackendBatchContractError, match="does not support device controller"):
            backend.bind_task_io(
                _requirements(
                    backend,
                    placement,
                    cadence=3,
                    controller=unsupported,
                )
            )
        invalid_limit = replace(
            descriptor,
            parameters=(
                descriptor.parameter("damping"),
                ControllerParameter("effort_limit", (1000.0,)),
                descriptor.parameter("stiffness"),
            ),
        )
        with pytest.raises(BackendBatchContractError, match="within actuator ctrl ranges"):
            backend.bind_task_io(
                _requirements(
                    backend,
                    placement,
                    cadence=3,
                    controller=invalid_limit,
                )
            )
        short_fields = tuple(
            replace(
                field,
                identity=replace(
                    field.identity,
                    entity_ids=field.identity.entity_ids[:-1],
                ),
                buffer=replace(field.buffer, row_shape=(field.buffer.row_shape[0] - 1,)),
            )
            for field in _fields(backend, placement)
        )
        with pytest.raises(BackendBatchContractError, match="do not cover every actuator"):
            backend.bind_task_io(
                _requirements(
                    backend,
                    placement,
                    cadence=3,
                    controller=descriptor,
                    fields=short_fields,
                )
            )
        assert backend.get_device_graph_diagnostics() == graph_state

        changed_descriptor = replace(
            descriptor,
            parameters=(
                descriptor.parameter("damping"),
                descriptor.parameter("effort_limit"),
                ControllerParameter("stiffness", (7.0,)),
            ),
        )
        with (
            patch.object(
                backend,
                "_capture_device_graph_bundle",
                side_effect=BackendBatchContractError("injected controller capture failure"),
            ),
            pytest.raises(BackendBatchContractError, match="injected controller capture failure"),
        ):
            backend.bind_task_io(
                _requirements(
                    backend,
                    placement,
                    cadence=3,
                    controller=changed_descriptor,
                )
            )
        assert backend.get_device_graph_diagnostics() == graph_state

        def host_callback(_data: Any, _substep: int) -> np.ndarray:
            return np.empty((_NUM_WORLDS, backend.num_actuators), dtype=np.float32)

        with pytest.raises(NotImplementedError, match="rejects host pre-step callbacks"):
            backend.set_pre_step_control(host_callback)
    finally:
        backend.cleanup_scene_assets()
