"""Real-CUDA contract tests for the production mjwarp typed host adapter."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, cast
from unittest.mock import patch

import numpy as np
import pytest
from numpy.testing import assert_allclose

from unilab.base.backend import (
    BackendBatchContractError,
    BackendBatchCounterBudget,
    BackendHotPathViolationError,
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
    ExecutionProfile,
    PhysicalUnit,
    ReferenceFrame,
    RowSelection,
    StaleStateBatchError,
    StateBatchPhase,
    StateEntityKind,
    StateFieldKind,
    StateFieldSpec,
    create_backend,
)
from unilab.base.backend.mjwarp.dependencies import load_mjwarp_dependencies
from unilab.base.scene import SceneCfg

pytestmark = pytest.mark.slow


def _backend(num_envs: int) -> Any:
    dependencies = load_mjwarp_dependencies()
    if not bool(dependencies.warp.get_device().is_cuda):
        pytest.fail("mjwarp typed batch tests require an active CUDA Warp device")
    from unilab.assets import ASSETS_ROOT_PATH

    return create_backend(
        "mjwarp",
        SceneCfg(model_file=str(ASSETS_ROOT_PATH / "robots" / "g1" / "scene_flat.xml")),
        num_envs,
        0.02 / 3.0,
        base_name="pelvis",
    )


def _state_buffer(row_shape: tuple[int, ...], dtype: str = "float32") -> BufferContract:
    return BufferContract(
        row_shape=row_shape,
        dtype=dtype,
        layout=BufferLayout.C_CONTIGUOUS,
        placement=BufferPlacement.host(),
        owner=BufferOwner.BACKEND,
        mutability=BufferMutability.READ_ONLY,
        lifetime=BufferLifetime.BORROWED_UNTIL_MUTATION,
        dlpack_exportable=False,
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
    dtype: str = "float32",
) -> StateFieldSpec:
    return StateFieldSpec(
        semantic_key=key,
        identity=BoundFieldIdentity(entity_kind, field_kind, entity_ids),
        frame=frame,
        unit=unit,
        buffer=_state_buffer(row_shape, dtype),
    )


def _requirements(
    backend: Any,
    *,
    fields: tuple[StateFieldSpec, ...] | None = None,
    control_dtype: str = "float32",
    cadence: int = 2,
) -> tuple[BackendIORequirements, tuple[Callable[[], np.ndarray], ...]]:
    base_id = int(backend.get_body_ids(("pelvis",))[0])
    state_fields: list[StateFieldSpec] = [
        _field(
            "root.position",
            entity_kind=StateEntityKind.ROOT,
            field_kind=StateFieldKind.POSITION,
            entity_ids=(base_id,),
            row_shape=(3,),
            frame=ReferenceFrame.WORLD,
            unit=PhysicalUnit.METER,
        ),
        _field(
            "root.orientation",
            entity_kind=StateEntityKind.ROOT,
            field_kind=StateFieldKind.ORIENTATION,
            entity_ids=(base_id,),
            row_shape=(4,),
            frame=ReferenceFrame.WORLD,
            unit=PhysicalUnit.QUATERNION,
        ),
        _field(
            "root.linear_velocity",
            entity_kind=StateEntityKind.ROOT,
            field_kind=StateFieldKind.LINEAR_VELOCITY,
            entity_ids=(base_id,),
            row_shape=(3,),
            frame=ReferenceFrame.WORLD,
            unit=PhysicalUnit.METER_PER_SECOND,
        ),
        _field(
            "root.angular_velocity",
            entity_kind=StateEntityKind.ROOT,
            field_kind=StateFieldKind.ANGULAR_VELOCITY,
            entity_ids=(base_id,),
            row_shape=(3,),
            frame=ReferenceFrame.WORLD,
            unit=PhysicalUnit.RADIAN_PER_SECOND,
        ),
        _field(
            "dof.position",
            entity_kind=StateEntityKind.DOF,
            field_kind=StateFieldKind.POSITION,
            entity_ids=tuple(range(backend.num_dof_vel)),
            row_shape=(backend.num_dof_vel,),
            frame=ReferenceFrame.JOINT,
            unit=PhysicalUnit.RADIAN,
        ),
        _field(
            "dof.angular_velocity",
            entity_kind=StateEntityKind.DOF,
            field_kind=StateFieldKind.ANGULAR_VELOCITY,
            entity_ids=tuple(range(backend.num_dof_vel)),
            row_shape=(backend.num_dof_vel,),
            frame=ReferenceFrame.JOINT,
            unit=PhysicalUnit.RADIAN_PER_SECOND,
        ),
    ]
    getters: list[Callable[[], np.ndarray]] = [
        backend.get_base_pos,
        backend.get_base_quat,
        backend.get_base_lin_vel,
        backend.get_base_ang_vel,
        backend.get_dof_pos,
        backend.get_dof_vel,
    ]
    sensor_contracts = (
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
    for sensor_name, frame, unit in sensor_contracts:
        # Entity IDs are cold-path model bindings.  Expected values below still
        # come exclusively from the public cache getter, never raw Warp storage.
        sensor_id = int(
            backend._mujoco.mj_name2id(
                backend._cpu_model,
                backend._mujoco.mjtObj.mjOBJ_SENSOR,
                sensor_name,
            )
        )
        sensor_dim = int(backend._cpu_model.sensor_dim[sensor_id])
        state_fields.append(
            _field(
                f"sensor.{sensor_name}",
                entity_kind=StateEntityKind.SENSOR,
                field_kind=StateFieldKind.VALUE,
                entity_ids=(sensor_id,),
                row_shape=(sensor_dim,),
                frame=frame,
                unit=unit,
            )
        )
        getters.append(lambda name=sensor_name: backend.get_sensor_data(name))

    transfer_bytes = {
        "control": backend._ctrl_staging.nbytes,
        "reset_mask": backend._reset_mask_host.nbytes,
        "qpos": backend._qpos_cache.nbytes,
        "qvel": backend._qvel_cache.nbytes,
        "sensordata": backend._sensor_cache.nbytes,
    }
    control = ControlSpec(
        "joint.position_target",
        BufferContract(
            row_shape=(backend.num_actuators,),
            dtype=control_dtype,
            layout=BufferLayout.C_CONTIGUOUS,
            placement=BufferPlacement.host(),
            owner=BufferOwner.MANAGER,
            mutability=BufferMutability.READ_ONLY,
            lifetime=BufferLifetime.UNTIL_STEP_COMPLETE,
            dlpack_exportable=False,
        ),
        physics_substeps_per_control=cadence,
    )
    requirements = BackendIORequirements(
        state_fields=tuple(state_fields) if fields is None else fields,
        control=control,
        execution_profile=ExecutionProfile.HOST_NUMPY,
        hot_path_budget=BackendBatchCounterBudget(
            host_to_device_transfers=1,
            device_to_host_transfers=3,
            host_to_device_bytes=transfer_bytes["control"],
            device_to_host_bytes=sum(
                transfer_bytes[name] for name in ("qpos", "qvel", "sensordata")
            ),
            global_synchronizations=1,
            allocations=3,
            state_materializations=1,
        ),
        reset_hot_path_budget=BackendBatchCounterBudget(
            host_to_device_transfers=3,
            device_to_host_transfers=3,
            host_to_device_bytes=sum(
                transfer_bytes[name] for name in ("reset_mask", "qpos", "qvel")
            ),
            device_to_host_bytes=sum(
                transfer_bytes[name] for name in ("qpos", "qvel", "sensordata")
            ),
            global_synchronizations=1,
            allocations=3,
            state_materializations=1,
        ),
    )
    return requirements, tuple(getters)


def _random_state(
    backend: Any,
    rng: np.random.Generator,
    count: int,
) -> tuple[np.ndarray, np.ndarray]:
    qpos = np.tile(backend.get_keyframe_qpos("stand"), (count, 1)).astype(np.float32)
    qvel = rng.normal(0.0, 0.05, size=(count, backend.get_init_qvel().size)).astype(np.float32)
    qpos[:, :3] += rng.uniform(-0.05, 0.05, size=(count, 3)).astype(np.float32)
    quaternion = rng.normal(size=(count, 4)).astype(np.float32)
    quaternion /= np.linalg.norm(quaternion, axis=1, keepdims=True)
    qpos[:, 3:7] = quaternion
    qpos[:, 7:] += rng.uniform(-0.02, 0.02, size=qpos[:, 7:].shape).astype(np.float32)
    return qpos, qvel


def _assert_state_matches(
    state: Any,
    expected: tuple[np.ndarray, ...],
    rows: tuple[int, ...] | None = None,
) -> None:
    for field_index, reference in enumerate(expected):
        actual = cast(np.ndarray, state.buffer_at(field_index).handle)
        selected = reference if rows is None else reference[np.asarray(rows, dtype=np.intp)]
        assert actual.flags.c_contiguous
        assert not actual.flags.writeable
        assert_allclose(actual, selected, atol=1.0e-6, rtol=1.0e-5)


@pytest.mark.parametrize("num_envs", [1, 32])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_mjwarp_typed_batch_matches_getter_and_transfer_contract(
    seed: int,
    num_envs: int,
) -> None:
    rng = np.random.default_rng(seed)
    backend = _backend(num_envs)
    requirements, getters = _requirements(backend)
    plan = backend.bind_task_io(requirements)
    assert backend.bind_task_io(replace(requirements)) is plan
    assert plan.fingerprint.startswith("mjwarp-host-batch-v1:")
    assert plan.state.fingerprint.startswith("mjwarp-host-state-v1:")

    pre_reset = backend.read_state_batch(plan, RowSelection.all(num_envs)).state
    pre_reset_view = pre_reset.buffer_at(0)
    qpos, qvel = _random_state(backend, rng, num_envs)
    backend.set_state(np.arange(num_envs, dtype=np.int32), qpos, qvel)
    with pytest.raises(StaleStateBatchError, match="mutation barrier"):
        _ = pre_reset_view.handle
    expected = tuple(np.asarray(getter()).copy() for getter in getters)

    selected_rows = tuple(int(row) for row in rng.permutation(num_envs)[: max(1, num_envs // 2)])
    with (
        patch.object(backend, "get_base_pos", side_effect=AssertionError("getter fallback")),
        patch.object(backend, "get_base_quat", side_effect=AssertionError("getter fallback")),
        patch.object(backend, "get_base_lin_vel", side_effect=AssertionError("getter fallback")),
        patch.object(backend, "get_base_ang_vel", side_effect=AssertionError("getter fallback")),
        patch.object(backend, "get_dof_pos", side_effect=AssertionError("getter fallback")),
        patch.object(backend, "get_dof_vel", side_effect=AssertionError("getter fallback")),
        patch.object(backend, "get_sensor_data", side_effect=AssertionError("getter fallback")),
        patch.object(backend, "get_body_ids", side_effect=AssertionError("selector fallback")),
        patch.object(backend, "step", side_effect=AssertionError("legacy step fallback")),
        patch.object(Path, "read_bytes", side_effect=AssertionError("asset fallback")),
        patch.object(Path, "read_text", side_effect=AssertionError("asset fallback")),
    ):
        selected = backend.read_state_batch(
            plan,
            RowSelection.selected(num_envs, selected_rows),
            phase=StateBatchPhase.RESET,
        ).state
        selected_view = selected.buffer_at(0)
        selected_address = cast(np.ndarray, selected_view.handle).ctypes.data
        _assert_state_matches(selected, expected, selected_rows)

        selected_again = backend.read_state_batch(
            plan,
            RowSelection.selected(num_envs, selected_rows),
            phase=StateBatchPhase.RESET,
        ).state
        with pytest.raises(StaleStateBatchError, match="mutation barrier"):
            _ = selected_view.handle
        assert cast(np.ndarray, selected_again.buffer_at(0).handle).ctypes.data == selected_address
        _assert_state_matches(selected_again, expected, selected_rows)

        control_array = rng.uniform(
            -0.1,
            0.1,
            size=(num_envs, backend.num_actuators),
        ).astype(np.float32)
        control = ControlBatch(
            plan=plan,
            rows=RowSelection.all(num_envs),
            buffer=BufferView(control_array, control_array.shape, plan.control.buffer),
        )
        terminal = backend.step_batch(plan, control, nsteps=2)
        with pytest.raises(StaleStateBatchError, match="mutation barrier"):
            selected_again.assert_valid()

    terminal_expected = tuple(np.asarray(getter()).copy() for getter in getters)
    _assert_state_matches(terminal.terminal_state, terminal_expected)
    assert terminal.terminal_state.phase is StateBatchPhase.TERMINAL
    counters = terminal.diagnostics.counters
    assert counters.host_to_device_transfers == 1
    assert counters.device_to_host_transfers == 3
    assert counters.host_to_device_bytes == backend._ctrl_staging.nbytes
    assert counters.device_to_host_bytes == (
        backend._qpos_cache.nbytes + backend._qvel_cache.nbytes + backend._sensor_cache.nbytes
    )
    assert counters.global_synchronizations == 1
    assert counters.allocations == 3
    assert counters.state_materializations == 1
    assert counters.dynamic_getter_calls == 0
    assert counters.selector_resolutions == 0
    assert counters.asset_metadata_reads == 0
    assert counters.registry_lookups == 0
    assert counters.instrumentation_complete


def test_mjwarp_typed_batch_fails_closed_before_physics() -> None:
    backend = _backend(2)
    requirements, _ = _requirements(backend)
    plan = backend.bind_task_io(requirements)
    state = backend.read_state_batch(plan, RowSelection.all(2)).state

    with pytest.raises(BackendHotPathViolationError, match="host_to_device_transfers"):
        backend.bind_task_io(replace(requirements, reset_hot_path_budget=None))

    foreign = replace(plan, state=replace(plan.state, backend_instance_id="mjwarp:foreign"))
    with pytest.raises(BackendBatchContractError, match="different backend"):
        backend.read_state_batch(foreign, RowSelection.all(2))

    root = requirements.state_fields[0]
    with pytest.raises(BackendBatchContractError, match="requires frame world"):
        backend.bind_task_io(
            replace(requirements, state_fields=(replace(root, frame=ReferenceFrame.BASE),))
        )
    with pytest.raises(BackendBatchContractError, match="requires dtype float32"):
        backend.bind_task_io(
            replace(
                requirements,
                state_fields=(replace(root, buffer=_state_buffer((3,), "float64")),),
            )
        )
    unsupported = _field(
        "body.position",
        entity_kind=StateEntityKind.BODY,
        field_kind=StateFieldKind.POSITION,
        entity_ids=(1,),
        row_shape=(1, 3),
        frame=ReferenceFrame.WORLD,
        unit=PhysicalUnit.METER,
    )
    with pytest.raises(BackendBatchContractError, match="does not support state entity kind"):
        backend.bind_task_io(replace(requirements, state_fields=(unsupported,)))

    valid = np.zeros((2, backend.num_actuators), dtype=np.float32)
    valid_control = ControlBatch(
        plan,
        RowSelection.all(2),
        BufferView(valid, valid.shape, plan.control.buffer),
    )
    wrong_dtype = np.zeros_like(valid, dtype=np.float64)
    wrong_dtype_control = ControlBatch(
        plan,
        RowSelection.all(2),
        BufferView(wrong_dtype, valid.shape, plan.control.buffer),
    )
    with pytest.raises(BackendBatchContractError, match="handle dtype"):
        backend.step_batch(plan, wrong_dtype_control, nsteps=2)
    wrong_shape = np.zeros((1, backend.num_actuators), dtype=np.float32)
    lying_shape_control = ControlBatch(
        plan,
        RowSelection.all(2),
        BufferView(wrong_shape, valid.shape, plan.control.buffer),
    )
    with pytest.raises(BackendBatchContractError, match="handle shape"):
        backend.step_batch(plan, lying_shape_control, nsteps=2)
    partial = ControlBatch(
        plan,
        RowSelection.selected(2, (1,)),
        BufferView(wrong_shape, wrong_shape.shape, plan.control.buffer),
    )
    with pytest.raises(BackendBatchContractError, match="controls for all rows"):
        backend.step_batch(plan, partial, nsteps=2)
    with pytest.raises(BackendBatchContractError, match="control cadence"):
        backend.step_batch(plan, valid_control, nsteps=1)
    with pytest.raises(BackendBatchContractError, match="mutation batches"):
        backend.step_batch(
            plan,
            valid_control,
            mutation_batch=cast(Any, object()),
            nsteps=2,
        )
    with pytest.raises(BackendBatchContractError, match="identity reset"):
        backend.reset_batch(
            plan,
            RowSelection.selected(2, (0,)),
            mutation_batch=cast(Any, object()),
        )

    state.assert_valid()

    backend.step(valid, nsteps=1)
    with pytest.raises(StaleStateBatchError, match="mutation barrier"):
        state.assert_valid()
    state = backend.read_state_batch(plan, RowSelection.all(2)).state
    qpos, qvel = _random_state(backend, np.random.default_rng(7), 1)
    backend.set_state(np.asarray([1], dtype=np.int32), qpos, qvel)
    with pytest.raises(StaleStateBatchError, match="mutation barrier"):
        state.assert_valid()
