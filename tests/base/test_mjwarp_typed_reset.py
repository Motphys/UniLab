"""Real-CUDA tests for ``mjwarp`` typed host identity reset commits."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, cast

import numpy as np
import pytest
from numpy.testing import assert_allclose

from unilab.base.backend import (
    BackendBatchContractError,
    BackendBatchCounterBudget,
    BackendIORequirements,
    BoundFieldIdentity,
    BufferContract,
    BufferLayout,
    BufferLifetime,
    BufferMutability,
    BufferOwner,
    BufferPlacement,
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

_HINGE = "left_hip_pitch_joint"
_BASE = "pelvis"


def _backend(num_envs: int) -> Any:
    dependencies = load_mjwarp_dependencies()
    if not bool(dependencies.warp.get_device().is_cuda):
        pytest.fail("mjwarp typed reset tests require an active CUDA Warp device")
    from unilab.assets import ASSETS_ROOT_PATH

    return create_backend(
        "mjwarp",
        SceneCfg(model_file=str(ASSETS_ROOT_PATH / "robots" / "g1" / "scene_flat.xml")),
        num_envs,
        0.02 / 3.0,
        base_name=_BASE,
    )


def _state_buffer(row_shape: tuple[int, ...]) -> BufferContract:
    return BufferContract(
        row_shape=row_shape,
        dtype="float32",
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
) -> StateFieldSpec:
    return StateFieldSpec(
        semantic_key=key,
        identity=BoundFieldIdentity(entity_kind, field_kind, entity_ids),
        frame=frame,
        unit=unit,
        buffer=_state_buffer(row_shape),
    )


def _hinge_coordinates(backend: Any) -> tuple[int, int]:
    """Resolve coordinates only while constructing this cold-path test fixture."""

    joint_id = int(
        backend._mujoco.mj_name2id(
            backend._cpu_model,
            backend._mujoco.mjtObj.mjOBJ_JOINT,
            _HINGE,
        )
    )
    return (
        int(backend._cpu_model.jnt_qposadr[joint_id]) - 7,
        int(backend._cpu_model.jnt_dofadr[joint_id]) - 6,
    )


def _requirements(backend: Any) -> BackendIORequirements:
    base_id = int(backend.get_body_ids((_BASE,))[0])
    qpos_coordinate, qvel_coordinate = _hinge_coordinates(backend)
    fields = (
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
            "hinge.position",
            entity_kind=StateEntityKind.DOF,
            field_kind=StateFieldKind.POSITION,
            entity_ids=(qpos_coordinate,),
            row_shape=(1,),
            frame=ReferenceFrame.JOINT,
            unit=PhysicalUnit.RADIAN,
        ),
        _field(
            "hinge.angular_velocity",
            entity_kind=StateEntityKind.DOF,
            field_kind=StateFieldKind.ANGULAR_VELOCITY,
            entity_ids=(qvel_coordinate,),
            row_shape=(1,),
            frame=ReferenceFrame.JOINT,
            unit=PhysicalUnit.RADIAN_PER_SECOND,
        ),
    )
    control = BufferContract(
        row_shape=(backend.num_actuators,),
        dtype="float32",
        layout=BufferLayout.C_CONTIGUOUS,
        placement=BufferPlacement.host(),
        owner=BufferOwner.MANAGER,
        mutability=BufferMutability.READ_ONLY,
        lifetime=BufferLifetime.UNTIL_STEP_COMPLETE,
        dlpack_exportable=False,
    )
    transfer_bytes = {
        "control": backend._ctrl_staging.nbytes,
        "reset_mask": backend._reset_mask_host.nbytes,
        "qpos": backend._qpos_cache.nbytes,
        "qvel": backend._qvel_cache.nbytes,
        "sensordata": backend._sensor_cache.nbytes,
    }
    return BackendIORequirements(
        state_fields=fields,
        control=ControlSpec("joint.position_target", control, physics_substeps_per_control=1),
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


@pytest.mark.parametrize("num_envs", [1, 4])
def test_mjwarp_typed_identity_reset_restores_defaults(num_envs: int) -> None:
    rng = np.random.default_rng(0)
    backend = _backend(num_envs)
    requirements = _requirements(backend)
    plan = backend.bind_task_io(requirements)
    assert backend.bind_task_io(replace(requirements)) is plan

    qpos, qvel = _random_state(backend, rng, num_envs)
    backend.set_state(np.arange(num_envs, dtype=np.int32), qpos, qvel)
    pre_reset = backend.read_state_batch(plan, RowSelection.all(num_envs)).state
    pre_reset_view = pre_reset.buffer_at(0)

    rows = RowSelection.all(num_envs) if num_envs == 1 else RowSelection.selected(num_envs, (3, 1))
    result = backend.reset_batch(plan, rows)
    with pytest.raises(StaleStateBatchError, match="mutation barrier"):
        _ = pre_reset_view.handle

    assert result.reset_state.phase is StateBatchPhase.RESET
    default_qpos = backend.get_default_qpos()
    reset_state = result.reset_state
    assert_allclose(
        cast(np.ndarray, reset_state.buffer("root.position").handle),
        np.broadcast_to(default_qpos[0:3], (rows.count, 3)),
        atol=1.0e-6,
    )
    assert_allclose(
        cast(np.ndarray, reset_state.buffer("root.orientation").handle),
        np.broadcast_to(default_qpos[3:7], (rows.count, 4)),
        atol=1.0e-6,
    )
    assert_allclose(
        cast(np.ndarray, reset_state.buffer("root.linear_velocity").handle),
        np.zeros((rows.count, 3), dtype=np.float32),
        atol=1.0e-6,
    )
    assert_allclose(
        cast(np.ndarray, reset_state.buffer("hinge.position").handle),
        np.broadcast_to(default_qpos[7 + _hinge_coordinates(backend)[0]], (rows.count, 1)),
        atol=1.0e-6,
    )
    assert_allclose(
        cast(np.ndarray, reset_state.buffer("hinge.angular_velocity").handle),
        np.zeros((rows.count, 1), dtype=np.float32),
        atol=1.0e-6,
    )

    counters = result.diagnostics.counters
    assert counters.host_to_device_transfers == 3
    assert counters.device_to_host_transfers == 3
    assert counters.global_synchronizations == 1
    assert counters.allocations == 3
    assert counters.state_materializations == 1
    assert counters.instrumentation_complete

    if not rows.is_all:
        complement = np.asarray([0, 2], dtype=np.int32)
        assert_allclose(backend.get_base_pos()[complement], qpos[complement, 0:3], atol=1.0e-6)
        assert_allclose(
            backend.get_base_lin_vel()[complement],
            qvel[complement, 0:3],
            atol=1.0e-6,
        )


def test_mjwarp_typed_reset_fails_closed_before_physics() -> None:
    backend = _backend(2)
    plan = backend.bind_task_io(_requirements(backend))
    state = backend.read_state_batch(plan, RowSelection.all(2)).state

    with pytest.raises(BackendBatchContractError, match="identity reset"):
        backend.reset_batch(
            plan,
            RowSelection.selected(2, (0,)),
            mutation_batch=cast(Any, object()),
        )
    foreign = replace(plan, state=replace(plan.state, backend_instance_id="mjwarp:foreign"))
    with pytest.raises(BackendBatchContractError, match="different backend"):
        backend.reset_batch(foreign, RowSelection.all(2))
    with pytest.raises(BackendBatchContractError, match="row universe"):
        backend.reset_batch(plan, RowSelection.all(3))

    state.assert_valid()
