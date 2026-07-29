from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

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
    ExternalWrenchMutationBatch,
    MutationBaseline,
    MutationCommitPhase,
    MutationContractError,
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
    StaleStateBatchError,
    StateEntityKind,
    StateFieldKind,
    StateFieldSpec,
    TypedBackendMutationBatch,
)
from unilab.base.backend.mujoco.backend import MuJoCoBackend
from unilab.base.scene import SceneCfg


def _write_free_body_model(tmp_path: Path) -> Path:
    model_file = tmp_path / "typed_wrench_free_body.xml"
    model_file.parent.mkdir(parents=True, exist_ok=True)
    model_file.write_text(
        """
<mujoco model="typed_wrench_free_body">
  <option timestep="0.01" gravity="0 0 0"/>
  <worldbody>
    <body name="payload">
      <freejoint name="payload_free"/>
      <geom name="payload_geom" type="sphere" size="0.1" mass="1"/>
    </body>
  </worldbody>
  <actuator>
    <motor name="payload_motor" joint="payload_free" gear="1 0 0 0 0 0"/>
  </actuator>
</mujoco>
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return model_file


def _backend(
    tmp_path: Path,
    *,
    num_envs: int = 4,
    np_dtype: type[np.floating] = np.float64,
    materialize: bool = True,
) -> MuJoCoBackend:
    backend = MuJoCoBackend(
        SceneCfg(model_file=str(_write_free_body_model(tmp_path))),
        num_envs=num_envs,
        sim_dt=0.01,
        base_name="payload",
        np_dtype=np_dtype,
        chunk_size=num_envs,
        bench_nsteps=1,
    )
    if materialize:
        backend.materialize()
    return backend


def _state_buffer(dtype: np.dtype[np.floating]) -> BufferContract:
    return BufferContract(
        row_shape=(3,),
        dtype=dtype.name,
        layout=BufferLayout.C_CONTIGUOUS,
        placement=BufferPlacement.host(),
        owner=BufferOwner.BACKEND,
        mutability=BufferMutability.READ_ONLY,
        lifetime=BufferLifetime.BORROWED_UNTIL_MUTATION,
        dlpack_exportable=False,
    )


def _value_buffer(dtype: np.dtype[np.floating]) -> BufferContract:
    return BufferContract(
        row_shape=(3,),
        dtype=dtype.name,
        layout=BufferLayout.C_CONTIGUOUS,
        placement=BufferPlacement.host(),
        owner=BufferOwner.MANAGER,
        mutability=BufferMutability.READ_ONLY,
        lifetime=BufferLifetime.UNTIL_COMMIT,
        dlpack_exportable=False,
    )


def _requirements(backend: MuJoCoBackend) -> BackendIORequirements:
    dtype = np.dtype(backend.get_init_qvel().dtype)
    payload_id = backend.get_body_id("payload")
    field = StateFieldSpec(
        semantic_key="payload.linear_velocity",
        identity=BoundFieldIdentity(
            entity_kind=StateEntityKind.ROOT,
            field_kind=StateFieldKind.LINEAR_VELOCITY,
            entity_ids=(payload_id,),
        ),
        frame=ReferenceFrame.WORLD,
        unit=PhysicalUnit.METER_PER_SECOND,
        buffer=_state_buffer(dtype),
    )
    control_buffer = BufferContract(
        row_shape=(backend.num_actuators,),
        dtype=dtype.name,
        layout=BufferLayout.C_CONTIGUOUS,
        placement=BufferPlacement.host(),
        owner=BufferOwner.MANAGER,
        mutability=BufferMutability.READ_ONLY,
        lifetime=BufferLifetime.UNTIL_STEP_COMPLETE,
        dlpack_exportable=False,
    )
    return BackendIORequirements(
        state_fields=(field,),
        control=ControlSpec("payload.control", control_buffer),
        execution_profile=ExecutionProfile.HOST_NUMPY,
        hot_path_budget=BackendBatchCounterBudget(
            allocations=2 if dtype == np.dtype(np.float64) else 3,
            state_materializations=1,
        ),
    )


def _force_spec(dtype: np.dtype[np.floating]) -> MutationSpec:
    return MutationSpec(
        term_key="push.payload",
        target=MutationTargetSpec(
            target_key="wrench.body.force",
            target_kind=MutationTargetKind.EXTERNAL_WRENCH,
            entity_kind=MutationEntityKind.BODY,
            field_kind=MutationFieldKind.FORCE,
            selector="payload",
        ),
        trigger=MutationTrigger.INTERVAL,
        commit_phase=MutationCommitPhase.PRE_PHYSICS,
        operation=MutationOperation.SET,
        baseline=MutationBaseline.CURRENT,
        persistence=MutationPersistence.ONE_STEP,
        recompute=MutationRecomputeLevel.NONE,
        value_template=_value_buffer(dtype),
    )


def _zero_control(plan, num_envs: int) -> ControlBatch:
    control = np.zeros(
        (num_envs, *plan.control.buffer.row_shape),
        dtype=plan.control.buffer.dtype,
    )
    return ControlBatch(
        plan=plan,
        rows=RowSelection.all(num_envs),
        buffer=BufferView(control, control.shape, plan.control.buffer),
    )


def _reset_to_default(backend: MuJoCoBackend) -> None:
    default_qpos = backend.get_default_qpos()
    init_qvel = backend.get_init_qvel()
    qpos = np.broadcast_to(default_qpos, (backend.num_envs, len(default_qpos))).copy()
    qvel = np.broadcast_to(init_qvel, (backend.num_envs, len(init_qvel))).copy()
    backend.set_state(np.arange(backend.num_envs, dtype=np.int32), qpos, qvel)


def _force_batch(
    mutation_plan,
    rows: RowSelection,
    values: np.ndarray,
) -> TypedBackendMutationBatch:
    field_index = mutation_plan.spec_index("push.payload")
    contract = mutation_plan.specs[field_index].value_buffer
    force_values = np.ascontiguousarray(values, dtype=contract.dtype)
    value = MutationValueBatch(
        plan=mutation_plan,
        field_index=field_index,
        rows=rows,
        buffer=BufferView(force_values, force_values.shape, contract),
    )
    return TypedBackendMutationBatch(
        plan=mutation_plan,
        rows=rows,
        wrench=ExternalWrenchMutationBatch((value,)),
    )


def _velocity(result) -> np.ndarray:
    return np.asarray(result.terminal_state.buffer_at(0).handle).copy()


def _close(backend: MuJoCoBackend) -> None:
    if backend._pool is not None:
        backend._pool.close()


@pytest.mark.parametrize(
    ("np_dtype", "atol"),
    ((np.float32, 2e-6), (np.float64, 1e-12)),
)
def test_mujoco_typed_wrench_commits_to_next_step_with_selected_row_isolation(
    tmp_path: Path, np_dtype: type[np.floating], atol: float
) -> None:
    backend = _backend(tmp_path, np_dtype=np_dtype)
    try:
        plan = backend.bind_task_io(_requirements(backend))
        mutation_plan = backend.bind_mutation_plan((_force_spec(np.dtype(np_dtype)),))
        _reset_to_default(backend)
        control = _zero_control(plan, backend.num_envs)
        pre_step = backend.read_state_batch(plan, RowSelection.all(backend.num_envs))
        pre_step_velocity = pre_step.state.buffer_at(0)
        initial_velocity = pre_step_velocity.handle
        assert isinstance(initial_velocity, np.ndarray)
        np.testing.assert_allclose(initial_velocity, 0.0, atol=atol)

        rows = RowSelection.selected(backend.num_envs, (3, 1))
        mutation = _force_batch(
            mutation_plan,
            rows,
            np.array(
                [
                    [[10.0, 0.0, 0.0]],
                    [[20.0, 0.0, 0.0]],
                ],
                dtype=np_dtype,
            ),
        )
        with (
            patch.object(backend, "get_body_ids", side_effect=AssertionError("selector fallback")),
            patch.object(Path, "read_bytes", side_effect=AssertionError("asset fallback")),
            patch.object(Path, "read_text", side_effect=AssertionError("asset fallback")),
        ):
            first_step = backend.step_batch(plan, control, mutation_batch=mutation)
        with pytest.raises(StaleStateBatchError, match="mutation barrier"):
            _ = pre_step_velocity.handle

        first_velocity = _velocity(first_step)
        np.testing.assert_allclose(
            first_velocity,
            np.array(
                [
                    [0.0, 0.0, 0.0],
                    [0.2, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.1, 0.0, 0.0],
                ]
            ),
            atol=atol,
            rtol=atol,
        )

        second_step = backend.step_batch(plan, control)
        np.testing.assert_allclose(
            _velocity(second_step),
            first_velocity,
            atol=atol,
            rtol=atol,
        )
    finally:
        _close(backend)


def test_mujoco_typed_wrench_binding_and_commit_faults_fail_closed(tmp_path: Path) -> None:
    unmaterialized = _backend(tmp_path / "unmaterialized", materialize=False)
    with pytest.raises(BackendBatchContractError, match="materialized backend pool"):
        unmaterialized.bind_mutation_plan((_force_spec(np.dtype(np.float64)),))

    backend = _backend(tmp_path / "materialized")
    try:
        plan = backend.bind_task_io(_requirements(backend))
        mutation_plan = backend.bind_mutation_plan((_force_spec(np.dtype(np.float64)),))
        control = _zero_control(plan, backend.num_envs)
        rows = RowSelection.selected(backend.num_envs, (2, 0))
        valid_values = np.ones((rows.count, 1, 3), dtype=np.float64)
        valid_mutation = _force_batch(mutation_plan, rows, valid_values)

        unsupported_specs = (
            replace(
                _force_spec(np.dtype(np.float64)),
                target=replace(
                    _force_spec(np.dtype(np.float64)).target,
                    field_kind=MutationFieldKind.TORQUE,
                ),
            ),
            replace(
                _force_spec(np.dtype(np.float64)),
                commit_phase=MutationCommitPhase.POST_PHYSICS,
            ),
            replace(_force_spec(np.dtype(np.float64)), operation=MutationOperation.ADD),
            replace(_force_spec(np.dtype(np.float64)), persistence=MutationPersistence.EPISODE),
        )
        for spec in unsupported_specs:
            with pytest.raises(MutationContractError):
                backend.bind_mutation_plan((spec,))

        wrong_owner_plan = replace(mutation_plan, backend_instance_id="mujoco:other")
        with pytest.raises(BackendBatchContractError, match="different backend"):
            backend.step_batch(
                plan,
                control,
                mutation_batch=TypedBackendMutationBatch(plan=wrong_owner_plan, rows=rows),
            )

        field_index = mutation_plan.spec_index("push.payload")
        contract = mutation_plan.specs[field_index].value_buffer
        malformed_handles = (
            np.zeros((rows.count, 1, 3), dtype=np.float32),
            np.zeros((rows.count, 1, 4), dtype=np.float64),
            np.zeros((rows.count, 2, 3), dtype=np.float64)[:, :1, :],
        )
        for handle in malformed_handles:
            value = MutationValueBatch(
                plan=mutation_plan,
                field_index=field_index,
                rows=rows,
                buffer=BufferView(handle, (rows.count, 1, 3), contract),
            )
            malformed = TypedBackendMutationBatch(
                plan=mutation_plan,
                rows=rows,
                wrench=ExternalWrenchMutationBatch((value,)),
            )
            with pytest.raises(BackendBatchContractError, match="value handle|C-contiguous"):
                backend.step_batch(plan, control, mutation_batch=malformed)

        valid_value = valid_mutation.wrench.values[0]
        with pytest.raises(MutationContractError, match="wrong typed sub-batch"):
            TypedBackendMutationBatch(
                plan=mutation_plan,
                rows=rows,
                state=SimulationStateMutationBatch((valid_value,)),
            )

        partial_control = ControlBatch(
            plan=plan,
            rows=rows,
            buffer=BufferView(
                np.zeros((rows.count, backend.num_actuators), dtype=np.float64),
                (rows.count, backend.num_actuators),
                plan.control.buffer,
            ),
        )
        with pytest.raises(BackendBatchContractError, match="controls for all rows"):
            backend.step_batch(plan, partial_control, mutation_batch=valid_mutation)

        backend.apply_body_force(
            np.asarray([backend.get_body_id("payload")], dtype=np.int32),
            np.ones((backend.num_envs, 1, 3), dtype=np.float64),
        )
        with pytest.raises(BackendBatchContractError, match="out-of-band external wrench"):
            backend.step_batch(plan, control, mutation_batch=valid_mutation)
    finally:
        _close(backend)
