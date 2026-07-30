"""Real-CUDA tests for ``mjwarp`` typed simulation-state reset commits."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import patch

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
    BufferView,
    ControlBatch,
    ControlSpec,
    ExecutionProfile,
    MutationBaseline,
    MutationCommitPhase,
    MutationContractError,
    MutationEntityKind,
    MutationFieldKind,
    MutationOperation,
    MutationPersistence,
    MutationRecomputeLevel,
    MutationSelectorMode,
    MutationSelectorSpec,
    MutationSpec,
    MutationTargetKind,
    MutationTargetSpec,
    MutationTrigger,
    MutationValueBatch,
    PhysicalUnit,
    ReferenceFrame,
    RowSelection,
    SimulationStateMutationBatch,
    StaleStateBatchError,
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


def _value_buffer(row_shape: tuple[int, ...]) -> BufferContract:
    return BufferContract(
        row_shape=row_shape,
        dtype="float32",
        layout=BufferLayout.C_CONTIGUOUS,
        placement=BufferPlacement.host(),
        owner=BufferOwner.MANAGER,
        mutability=BufferMutability.READ_ONLY,
        lifetime=BufferLifetime.UNTIL_COMMIT,
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
    transfer_bytes = {item.name: item.nbytes for item in backend.get_transfer_buffers()}
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


def _reset_spec(
    *,
    term_key: str,
    target_key: str,
    entity_kind: MutationEntityKind,
    field_kind: MutationFieldKind,
    selector: MutationSelectorSpec | str,
    row_shape: tuple[int, ...],
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
        value_template=_value_buffer(row_shape),
    )


def _reset_specs() -> tuple[MutationSpec, ...]:
    return (
        _reset_spec(
            term_key="reset.root.position",
            target_key="state.root.position",
            entity_kind=MutationEntityKind.BODY,
            field_kind=MutationFieldKind.POSITION,
            selector=_BASE,
            row_shape=(3,),
        ),
        _reset_spec(
            term_key="reset.root.orientation",
            target_key="state.root.orientation",
            entity_kind=MutationEntityKind.BODY,
            field_kind=MutationFieldKind.ORIENTATION,
            selector=_BASE,
            row_shape=(4,),
        ),
        _reset_spec(
            term_key="reset.root.linear_velocity",
            target_key="state.root.linear_velocity",
            entity_kind=MutationEntityKind.BODY,
            field_kind=MutationFieldKind.LINEAR_VELOCITY,
            selector=_BASE,
            row_shape=(3,),
        ),
        _reset_spec(
            term_key="reset.root.angular_velocity",
            target_key="state.root.angular_velocity",
            entity_kind=MutationEntityKind.BODY,
            field_kind=MutationFieldKind.ANGULAR_VELOCITY,
            selector=_BASE,
            row_shape=(3,),
        ),
        _reset_spec(
            term_key="reset.hinge.position",
            target_key="state.dof.position",
            entity_kind=MutationEntityKind.DOF,
            field_kind=MutationFieldKind.POSITION,
            selector=_HINGE,
            row_shape=(1,),
        ),
        _reset_spec(
            term_key="reset.hinge.angular_velocity",
            target_key="state.dof.angular_velocity",
            entity_kind=MutationEntityKind.DOF,
            field_kind=MutationFieldKind.ANGULAR_VELOCITY,
            selector=_HINGE,
            row_shape=(1,),
        ),
    )


def _value(
    mutation_plan: Any,
    term_key: str,
    rows: RowSelection,
    values: np.ndarray,
) -> MutationValueBatch:
    field_index = mutation_plan.spec_index(term_key)
    contract = mutation_plan.specs[field_index].value_buffer
    handle = np.ascontiguousarray(values, dtype=contract.dtype)
    return MutationValueBatch(
        plan=mutation_plan,
        field_index=field_index,
        rows=rows,
        buffer=BufferView(handle, handle.shape, contract),
    )


def _reset_batch(
    mutation_plan: Any,
    rows: RowSelection,
    values: dict[str, np.ndarray],
) -> TypedBackendMutationBatch:
    return TypedBackendMutationBatch(
        plan=mutation_plan,
        rows=rows,
        state=SimulationStateMutationBatch(
            tuple(
                _value(mutation_plan, term_key, rows, value) for term_key, value in values.items()
            )
        ),
    )


def _random_state(backend: Any, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    qpos = np.tile(backend.get_keyframe_qpos("stand"), (backend.num_envs, 1)).astype(np.float32)
    qvel = rng.normal(0.0, 0.05, size=(backend.num_envs, backend.get_init_qvel().size)).astype(
        np.float32
    )
    qpos[:, :3] += rng.uniform(-0.02, 0.02, size=(backend.num_envs, 3)).astype(np.float32)
    quat = rng.normal(size=(backend.num_envs, 4)).astype(np.float32)
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    qpos[:, 3:7] = quat
    qpos[:, 7:] += rng.uniform(-0.01, 0.01, size=qpos[:, 7:].shape).astype(np.float32)
    return qpos, qvel


def _public_state(backend: Any) -> dict[str, np.ndarray]:
    return {
        "root.position": np.asarray(backend.get_base_pos()).copy(),
        "root.orientation": np.asarray(backend.get_base_quat()).copy(),
        "root.linear_velocity": np.asarray(backend.get_base_lin_vel()).copy(),
        "root.angular_velocity": np.asarray(backend.get_base_ang_vel()).copy(),
        "hinge.position": np.asarray(backend.get_dof_pos()[:, :1]).copy(),
        "hinge.angular_velocity": np.asarray(backend.get_dof_vel()[:, :1]).copy(),
    }


def _assert_state_matches(
    state: Any,
    expected: dict[str, np.ndarray],
    rows: RowSelection,
    *,
    atol: float = 2.0e-5,
) -> None:
    indices = None if rows.is_all else np.asarray(rows.indices, dtype=np.intp)
    for key, reference in expected.items():
        actual = np.asarray(state.buffer(key).handle)
        target = reference if indices is None else reference[indices]
        assert actual.flags.c_contiguous
        assert not actual.flags.writeable
        assert_allclose(actual, target, atol=atol, rtol=atol)


def _zero_control(plan: Any, num_envs: int) -> ControlBatch:
    control = np.zeros((num_envs, *plan.control.buffer.row_shape), dtype=np.float32)
    return ControlBatch(
        plan=plan,
        rows=RowSelection.all(num_envs),
        buffer=BufferView(control, control.shape, plan.control.buffer),
    )


@pytest.mark.parametrize("seed", (0, 1, 2))
@pytest.mark.parametrize("num_envs", (1, 32))
@pytest.mark.parametrize("row_mode", ("all", "selected"))
def test_mjwarp_typed_reset_matches_independent_legacy_oracle(
    seed: int,
    num_envs: int,
    row_mode: str,
) -> None:
    rng = np.random.default_rng(seed)
    backend = _backend(num_envs)
    reference = _backend(num_envs)
    requirements = _requirements(backend)
    reference_requirements = _requirements(reference)
    plan = backend.bind_task_io(requirements)
    reference.bind_task_io(reference_requirements)
    mutation_plan = backend.bind_mutation_plan(_reset_specs())

    qpos, qvel = _random_state(backend, rng)
    all_rows = np.arange(num_envs, dtype=np.int32)
    backend.set_state(all_rows, qpos, qvel)
    reference.set_state(all_rows, qpos, qvel)
    if row_mode == "all":
        rows = RowSelection.all(num_envs)
        selected = all_rows
    else:
        selected = rng.permutation(num_envs)[: max(1, num_envs // 2)].astype(np.int32)
        rows = RowSelection.selected(num_envs, tuple(int(row) for row in selected))

    root_position = rng.uniform(-0.1, 0.1, size=(rows.count, 1, 3)).astype(np.float32)
    root_position[..., 2] += 0.85
    root_orientation = rng.normal(size=(rows.count, 1, 4)).astype(np.float32)
    root_orientation /= np.linalg.norm(root_orientation, axis=-1, keepdims=True)
    values = {
        "reset.root.position": root_position,
        "reset.root.orientation": root_orientation,
        "reset.root.linear_velocity": rng.normal(0.0, 0.2, size=(rows.count, 1, 3)).astype(
            np.float32
        ),
        "reset.root.angular_velocity": rng.normal(0.0, 0.3, size=(rows.count, 1, 3)).astype(
            np.float32
        ),
        "reset.hinge.position": rng.uniform(-0.3, 0.3, size=(rows.count, 1, 1)).astype(np.float32),
        "reset.hinge.angular_velocity": rng.normal(0.0, 0.5, size=(rows.count, 1, 1)).astype(
            np.float32
        ),
    }
    mutation = _reset_batch(mutation_plan, rows, values)

    reference_qpos = qpos.copy()
    reference_qvel = qvel.copy()
    reference_qpos[selected, :3] = values["reset.root.position"][:, 0, :]
    reference_qpos[selected, 3:7] = values["reset.root.orientation"][:, 0, :]
    reference_qvel[selected, :3] = values["reset.root.linear_velocity"][:, 0, :]
    reference_qvel[selected, 3:6] = values["reset.root.angular_velocity"][:, 0, :]
    reference_qpos[selected, 7] = values["reset.hinge.position"][:, 0, 0]
    reference_qvel[selected, 6] = values["reset.hinge.angular_velocity"][:, 0, 0]
    reference.set_state(selected, reference_qpos[selected], reference_qvel[selected])
    expected_reset = _public_state(reference)

    terminal = backend.read_state_batch(plan, RowSelection.all(num_envs)).state
    terminal_view = terminal.buffer("root.position")
    backend.reset_transfer_telemetry()
    before_transfer = backend.get_transfer_counters()
    with (
        patch.object(backend, "set_state", side_effect=AssertionError("legacy reset fallback")),
        patch.object(backend, "get_base_pos", side_effect=AssertionError("getter fallback")),
        patch.object(backend, "get_base_quat", side_effect=AssertionError("getter fallback")),
        patch.object(backend, "get_base_lin_vel", side_effect=AssertionError("getter fallback")),
        patch.object(backend, "get_base_ang_vel", side_effect=AssertionError("getter fallback")),
        patch.object(backend, "get_dof_pos", side_effect=AssertionError("getter fallback")),
        patch.object(backend, "get_dof_vel", side_effect=AssertionError("getter fallback")),
        patch.object(
            backend,
            "_resolve_mjwarp_typed_mutation_selector",
            side_effect=AssertionError("selector fallback"),
        ),
        patch.object(Path, "read_bytes", side_effect=AssertionError("asset fallback")),
        patch.object(Path, "read_text", side_effect=AssertionError("asset fallback")),
    ):
        result = backend.reset_batch(plan, rows, mutation_batch=mutation)
    transfer_delta = backend.get_transfer_counters().delta(before_transfer)

    assert result.reset_state.phase is StateBatchPhase.RESET
    assert result.reset_state.rows == rows
    with pytest.raises(StaleStateBatchError, match="mutation barrier"):
        _ = terminal_view.handle
    _assert_state_matches(result.reset_state, expected_reset, rows)
    reset_address = np.asarray(result.reset_state.buffer("root.position").handle).ctypes.data
    _assert_state_matches(
        backend.read_state_batch(plan, RowSelection.all(num_envs)).state,
        expected_reset,
        RowSelection.all(num_envs),
    )

    buffers = {item.name: item.nbytes for item in backend.get_transfer_buffers()}
    counters = result.diagnostics.counters
    assert counters.host_to_device_transfers == transfer_delta.host_to_device_transfers == 3
    assert counters.device_to_host_transfers == transfer_delta.device_to_host_transfers == 3
    assert (
        counters.host_to_device_bytes
        == transfer_delta.host_to_device_bytes
        == sum(buffers[name] for name in ("reset_mask", "qpos", "qvel"))
    )
    assert (
        counters.device_to_host_bytes
        == transfer_delta.device_to_host_bytes
        == sum(buffers[name] for name in ("qpos", "qvel", "sensordata"))
    )
    assert counters.global_synchronizations == transfer_delta.global_synchronizations == 1
    assert counters.allocations == 3
    assert counters.state_materializations == 1
    assert counters.dynamic_getter_calls == 0
    assert counters.selector_resolutions == 0
    assert counters.asset_metadata_reads == 0
    assert counters.registry_lookups == 0
    assert counters.instrumentation_complete
    trace = backend.get_transfer_trace()
    assert trace.counters() == transfer_delta
    assert {event.barrier for event in trace.events} == {"reset"}

    candidate_terminal = backend.step_batch(plan, _zero_control(plan, num_envs), nsteps=1)
    reference.step(np.zeros((num_envs, reference.num_actuators), dtype=np.float32), nsteps=1)
    _assert_state_matches(
        candidate_terminal.terminal_state,
        _public_state(reference),
        RowSelection.all(num_envs),
        atol=1.0e-4,
    )

    repeat = backend.reset_batch(plan, rows, mutation_batch=mutation)
    assert (
        np.asarray(repeat.reset_state.buffer("root.position").handle).ctypes.data == reset_address
    )


def test_mjwarp_typed_reset_binding_and_commit_faults_fail_closed() -> None:
    backend = _backend(4)
    plan = backend.bind_task_io(_requirements(backend))
    specs = _reset_specs()
    mutation_plan = backend.bind_mutation_plan(specs)
    rows = RowSelection.selected(backend.num_envs, (3, 1))
    values = {
        "reset.root.position": np.zeros((rows.count, 1, 3), dtype=np.float32),
        "reset.root.orientation": np.tile(
            np.asarray((1.0, 0.0, 0.0, 0.0), dtype=np.float32),
            (rows.count, 1, 1),
        ),
        "reset.root.linear_velocity": np.zeros((rows.count, 1, 3), dtype=np.float32),
        "reset.root.angular_velocity": np.zeros((rows.count, 1, 3), dtype=np.float32),
        "reset.hinge.position": np.zeros((rows.count, 1, 1), dtype=np.float32),
        "reset.hinge.angular_velocity": np.zeros((rows.count, 1, 1), dtype=np.float32),
    }
    valid = _reset_batch(mutation_plan, rows, values)

    compiled_hinge_selector = MutationSelectorSpec(
        semantic_key="robot.managed_hinge",
        mode=MutationSelectorMode.EXACT,
        expressions=(_HINGE,),
        entity_ids=(17,),
    )
    structured = replace(
        specs[4],
        target=replace(specs[4].target, selector=compiled_hinge_selector),
    )
    structured_plan = backend.bind_mutation_plan((structured,))
    assert (
        structured_plan.specs[0].target.entity_ids
        == mutation_plan.specs[mutation_plan.spec_index("reset.hinge.position")].target.entity_ids
    )

    grouped_hinges = (_HINGE, "right_hip_pitch_joint")
    grouped_selector = MutationSelectorSpec(
        semantic_key="robot.managed_hinges",
        mode=MutationSelectorMode.EXACT,
        expressions=grouped_hinges,
        entity_ids=(17, 23),
    )
    grouped = replace(
        specs[4],
        target=replace(specs[4].target, selector=grouped_selector),
    )
    grouped_plan = backend.bind_mutation_plan((grouped,))
    expected_coordinates = tuple(
        int(value) for value in backend.get_joint_dof_pos_indices(grouped_hinges)
    )
    assert grouped_plan.specs[0].target.entity_ids == expected_coordinates
    assert grouped_plan.specs[0].value_buffer.row_shape == (2, 1)

    compiled_root_selector = MutationSelectorSpec(
        semantic_key="robot.managed_root",
        mode=MutationSelectorMode.EXACT,
        expressions=(_BASE,),
        entity_ids=(1,),
    )
    structured_root = replace(
        specs[0],
        target=replace(specs[0].target, selector=compiled_root_selector),
    )
    structured_root_plan = backend.bind_mutation_plan((structured_root,))
    assert (
        structured_root_plan.specs[0].target.entity_ids
        == mutation_plan.specs[mutation_plan.spec_index("reset.root.position")].target.entity_ids
    )

    unsupported_specs = (
        replace(specs[0], operation=MutationOperation.ADD),
        replace(specs[0], commit_phase=MutationCommitPhase.PRE_PHYSICS),
        replace(specs[0], persistence=MutationPersistence.ONE_STEP),
        replace(specs[0], target=replace(specs[0].target, selector="left_hip_pitch_link")),
        replace(specs[4], target=replace(specs[4].target, selector="floating_base_joint")),
        replace(specs[4], target=replace(specs[4].target, selector="missing_joint")),
        replace(
            specs[4],
            target=replace(
                specs[4].target,
                selector=MutationSelectorSpec(
                    semantic_key="robot.regex_hinges",
                    mode=MutationSelectorMode.REGEX,
                    expressions=(".*hip.*",),
                    entity_ids=(17,),
                ),
            ),
        ),
    )
    for spec in unsupported_specs:
        with pytest.raises(MutationContractError):
            backend.bind_mutation_plan((spec,))

    state = backend.read_state_batch(plan, RowSelection.all(backend.num_envs)).state
    telemetry = backend.get_transfer_counters()
    with pytest.raises(BackendBatchContractError, match="TypedBackendMutationBatch"):
        backend.reset_batch(plan, rows)
    with pytest.raises(BackendBatchContractError, match="rows must match"):
        backend.reset_batch(
            plan,
            rows,
            mutation_batch=_reset_batch(
                mutation_plan,
                RowSelection.selected(backend.num_envs, (1, 3)),
                {key: value.copy() for key, value in values.items()},
            ),
        )
    with pytest.raises(BackendBatchContractError, match="different backend"):
        backend.reset_batch(
            plan,
            rows,
            mutation_batch=TypedBackendMutationBatch(
                plan=replace(mutation_plan, backend_instance_id="mjwarp:foreign"),
                rows=rows,
            ),
        )
    with pytest.raises(BackendBatchContractError, match="different backend"):
        backend.reset_batch(
            replace(plan, state=replace(plan.state, backend_instance_id="mjwarp:foreign")),
            rows,
            mutation_batch=valid,
        )
    with pytest.raises(BackendBatchContractError, match="at least one state value"):
        backend.reset_batch(
            plan,
            rows,
            mutation_batch=TypedBackendMutationBatch(plan=mutation_plan, rows=rows),
        )

    field_index = mutation_plan.spec_index("reset.root.position")
    contract = mutation_plan.specs[field_index].value_buffer
    malformed_handles = (
        np.zeros((rows.count, 1, 3), dtype=np.float64),
        np.zeros((rows.count, 1, 2), dtype=np.float32),
        np.zeros((rows.count, 1, 6), dtype=np.float32)[:, :, ::2],
    )
    for handle in malformed_handles:
        malformed_value = MutationValueBatch(
            plan=mutation_plan,
            field_index=field_index,
            rows=rows,
            buffer=BufferView(handle, (rows.count, 1, 3), contract),
        )
        malformed = TypedBackendMutationBatch(
            plan=mutation_plan,
            rows=rows,
            state=SimulationStateMutationBatch((malformed_value,)),
        )
        with pytest.raises(BackendBatchContractError, match="value handle|C-contiguous"):
            backend.reset_batch(plan, rows, mutation_batch=malformed)

    runtime_plan = backend._host_mutation_plans[mutation_plan.fingerprint]
    paired = runtime_plan._registered_batch_plans.pop(plan.fingerprint)
    try:
        with pytest.raises(BackendBatchContractError, match="not cold-path paired"):
            backend.reset_batch(plan, rows, mutation_batch=valid)
    finally:
        runtime_plan._registered_batch_plans[plan.fingerprint] = paired

    assert backend.get_transfer_counters() == telemetry
    state.assert_valid()
