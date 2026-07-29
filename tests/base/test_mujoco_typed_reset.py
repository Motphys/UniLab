from __future__ import annotations

import gc
import weakref
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
    BoundMutationValueBufferGroup,
    BoundMutationValueBuffers,
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
)
from unilab.base.backend.mujoco.backend import MuJoCoBackend
from unilab.base.scene import SceneCfg


def _write_reset_model(tmp_path: Path) -> Path:
    """Create a free-root fixture with offset qpos/qvel hinge coordinates.

    The ball joint preceding ``hinge`` deliberately makes its qpos and qvel
    coordinate IDs differ.  This proves that a cold-path selector cache is
    target-specific rather than accidentally sharing qpos IDs with qvel.
    """

    model_file = tmp_path / "typed_reset_free_hinge.xml"
    model_file.parent.mkdir(parents=True, exist_ok=True)
    model_file.write_text(
        """
<mujoco model="typed_reset_free_hinge">
  <option timestep="0.01" gravity="0 0 0"/>
  <worldbody>
    <body name="payload">
      <freejoint name="payload_free"/>
      <geom name="payload_geom" type="sphere" size="0.1" mass="1"/>
      <body name="ball_link" pos="0 0 0.15">
        <joint name="prefix_ball" type="ball"/>
        <geom name="ball_geom" type="sphere" size="0.04" mass="0.1"/>
        <body name="hinge_link" pos="0 0 0.12">
          <joint name="hinge" type="hinge" axis="0 1 0"/>
          <geom name="hinge_geom" type="capsule" fromto="0 0 0 0 0 0.15" size="0.02" mass="0.1"/>
        </body>
      </body>
      <body name="slide_link" pos="0.2 0 0">
        <joint name="slide" type="slide" axis="1 0 0"/>
        <geom name="slide_geom" type="sphere" size="0.03" mass="0.1"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="hinge_motor" joint="hinge" gear="1"/>
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
    base_name: str | None = "payload",
) -> MuJoCoBackend:
    backend = MuJoCoBackend(
        SceneCfg(model_file=str(_write_reset_model(tmp_path))),
        num_envs=num_envs,
        sim_dt=0.01,
        base_name=base_name,
        np_dtype=np_dtype,
        chunk_size=num_envs,
        bench_nsteps=1,
    )
    if materialize:
        backend.materialize()
    return backend


def _state_buffer(
    dtype: np.dtype[np.floating],
    *,
    row_shape: tuple[int, ...] = (1,),
) -> BufferContract:
    return BufferContract(
        row_shape=row_shape,
        dtype=dtype.name,
        layout=BufferLayout.C_CONTIGUOUS,
        placement=BufferPlacement.host(),
        owner=BufferOwner.BACKEND,
        mutability=BufferMutability.READ_ONLY,
        lifetime=BufferLifetime.BORROWED_UNTIL_MUTATION,
        dlpack_exportable=False,
    )


def _value_buffer(
    dtype: np.dtype[np.floating],
    *,
    row_shape: tuple[int, ...] = (1,),
) -> BufferContract:
    return BufferContract(
        row_shape=row_shape,
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
    position_id = tuple(int(value) for value in backend.get_joint_dof_pos_indices(("hinge",)))
    velocity_id = tuple(int(value) for value in backend.get_joint_dof_vel_indices(("hinge",)))
    state_buffer = _state_buffer(dtype)
    fields = (
        StateFieldSpec(
            semantic_key="hinge.position",
            identity=BoundFieldIdentity(
                entity_kind=StateEntityKind.DOF,
                field_kind=StateFieldKind.POSITION,
                entity_ids=position_id,
            ),
            frame=ReferenceFrame.JOINT,
            unit=PhysicalUnit.RADIAN,
            buffer=state_buffer,
        ),
        StateFieldSpec(
            semantic_key="hinge.angular_velocity",
            identity=BoundFieldIdentity(
                entity_kind=StateEntityKind.DOF,
                field_kind=StateFieldKind.ANGULAR_VELOCITY,
                entity_ids=velocity_id,
            ),
            frame=ReferenceFrame.JOINT,
            unit=PhysicalUnit.RADIAN_PER_SECOND,
            buffer=state_buffer,
        ),
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
        state_fields=fields,
        control=ControlSpec("hinge.control", control_buffer),
        execution_profile=ExecutionProfile.HOST_NUMPY,
        hot_path_budget=BackendBatchCounterBudget(
            allocations=2 if dtype == np.dtype(np.float64) else 3,
            state_materializations=1,
        ),
    )


def _full_reset_requirements(backend: MuJoCoBackend) -> BackendIORequirements:
    """Bind the public root and hinge fields used by a complete G1 reset slice."""

    dtype = np.dtype(backend.get_init_qvel().dtype)
    base_id = int(backend.get_body_ids(("payload",))[0])
    position_id = tuple(int(value) for value in backend.get_joint_dof_pos_indices(("hinge",)))
    velocity_id = tuple(int(value) for value in backend.get_joint_dof_vel_indices(("hinge",)))
    fields = (
        StateFieldSpec(
            semantic_key="root.position",
            identity=BoundFieldIdentity(
                entity_kind=StateEntityKind.ROOT,
                field_kind=StateFieldKind.POSITION,
                entity_ids=(base_id,),
            ),
            frame=ReferenceFrame.WORLD,
            unit=PhysicalUnit.METER,
            buffer=_state_buffer(dtype, row_shape=(3,)),
        ),
        StateFieldSpec(
            semantic_key="root.orientation",
            identity=BoundFieldIdentity(
                entity_kind=StateEntityKind.ROOT,
                field_kind=StateFieldKind.ORIENTATION,
                entity_ids=(base_id,),
            ),
            frame=ReferenceFrame.WORLD,
            unit=PhysicalUnit.QUATERNION,
            buffer=_state_buffer(dtype, row_shape=(4,)),
        ),
        StateFieldSpec(
            semantic_key="root.linear_velocity",
            identity=BoundFieldIdentity(
                entity_kind=StateEntityKind.ROOT,
                field_kind=StateFieldKind.LINEAR_VELOCITY,
                entity_ids=(base_id,),
            ),
            frame=ReferenceFrame.WORLD,
            unit=PhysicalUnit.METER_PER_SECOND,
            buffer=_state_buffer(dtype, row_shape=(3,)),
        ),
        StateFieldSpec(
            semantic_key="root.angular_velocity",
            identity=BoundFieldIdentity(
                entity_kind=StateEntityKind.ROOT,
                field_kind=StateFieldKind.ANGULAR_VELOCITY,
                entity_ids=(base_id,),
            ),
            frame=ReferenceFrame.WORLD,
            unit=PhysicalUnit.RADIAN_PER_SECOND,
            buffer=_state_buffer(dtype, row_shape=(3,)),
        ),
        StateFieldSpec(
            semantic_key="hinge.position",
            identity=BoundFieldIdentity(
                entity_kind=StateEntityKind.DOF,
                field_kind=StateFieldKind.POSITION,
                entity_ids=position_id,
            ),
            frame=ReferenceFrame.JOINT,
            unit=PhysicalUnit.RADIAN,
            buffer=_state_buffer(dtype),
        ),
        StateFieldSpec(
            semantic_key="hinge.angular_velocity",
            identity=BoundFieldIdentity(
                entity_kind=StateEntityKind.DOF,
                field_kind=StateFieldKind.ANGULAR_VELOCITY,
                entity_ids=velocity_id,
            ),
            frame=ReferenceFrame.JOINT,
            unit=PhysicalUnit.RADIAN_PER_SECOND,
            buffer=_state_buffer(dtype),
        ),
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
        state_fields=fields,
        control=ControlSpec("hinge.control", control_buffer),
        execution_profile=ExecutionProfile.HOST_NUMPY,
        hot_path_budget=BackendBatchCounterBudget(
            allocations=2 if dtype == np.dtype(np.float64) else 3,
            state_materializations=1,
        ),
    )


def _reset_spec(
    dtype: np.dtype[np.floating],
    *,
    term_key: str,
    target_key: str,
    field_kind: MutationFieldKind,
    entity_kind: MutationEntityKind = MutationEntityKind.DOF,
    selector: MutationSelectorSpec | str = "hinge",
    row_shape: tuple[int, ...] = (1,),
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
        value_template=_value_buffer(dtype, row_shape=row_shape),
    )


def _position_spec(dtype: np.dtype[np.floating], *, selector: str = "hinge") -> MutationSpec:
    return _reset_spec(
        dtype,
        term_key="reset.hinge.position",
        target_key="state.dof.position",
        field_kind=MutationFieldKind.POSITION,
        selector=selector,
    )


def _velocity_spec(dtype: np.dtype[np.floating], *, selector: str = "hinge") -> MutationSpec:
    return _reset_spec(
        dtype,
        term_key="reset.hinge.velocity",
        target_key="state.dof.angular_velocity",
        field_kind=MutationFieldKind.ANGULAR_VELOCITY,
        selector=selector,
    )


def _full_reset_specs(dtype: np.dtype[np.floating]) -> tuple[MutationSpec, ...]:
    return (
        _reset_spec(
            dtype,
            term_key="reset.root.position",
            target_key="state.root.position",
            entity_kind=MutationEntityKind.BODY,
            field_kind=MutationFieldKind.POSITION,
            selector="payload",
            row_shape=(3,),
        ),
        _reset_spec(
            dtype,
            term_key="reset.root.orientation",
            target_key="state.root.orientation",
            entity_kind=MutationEntityKind.BODY,
            field_kind=MutationFieldKind.ORIENTATION,
            selector="payload",
            row_shape=(4,),
        ),
        _reset_spec(
            dtype,
            term_key="reset.root.linear_velocity",
            target_key="state.root.linear_velocity",
            entity_kind=MutationEntityKind.BODY,
            field_kind=MutationFieldKind.LINEAR_VELOCITY,
            selector="payload",
            row_shape=(3,),
        ),
        _reset_spec(
            dtype,
            term_key="reset.root.angular_velocity",
            target_key="state.root.angular_velocity",
            entity_kind=MutationEntityKind.BODY,
            field_kind=MutationFieldKind.ANGULAR_VELOCITY,
            selector="payload",
            row_shape=(3,),
        ),
        _position_spec(dtype),
        _velocity_spec(dtype),
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
        value_template=BufferContract(
            row_shape=(3,),
            dtype=dtype.name,
            layout=BufferLayout.C_CONTIGUOUS,
            placement=BufferPlacement.host(),
            owner=BufferOwner.MANAGER,
            mutability=BufferMutability.READ_ONLY,
            lifetime=BufferLifetime.UNTIL_COMMIT,
            dlpack_exportable=False,
        ),
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


def _state_value(
    mutation_plan,
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
    mutation_plan,
    rows: RowSelection,
    position: np.ndarray,
    velocity: np.ndarray,
) -> TypedBackendMutationBatch:
    return TypedBackendMutationBatch(
        plan=mutation_plan,
        rows=rows,
        state=SimulationStateMutationBatch(
            (
                _state_value(mutation_plan, "reset.hinge.position", rows, position),
                _state_value(mutation_plan, "reset.hinge.velocity", rows, velocity),
            )
        ),
    )


def _full_reset_batch(
    mutation_plan,
    rows: RowSelection,
    values: dict[str, np.ndarray],
) -> TypedBackendMutationBatch:
    return TypedBackendMutationBatch(
        plan=mutation_plan,
        rows=rows,
        state=SimulationStateMutationBatch(
            tuple(
                _state_value(mutation_plan, term_key, rows, value)
                for term_key, value in values.items()
            )
        ),
    )


def _prepared_full_reset_batch(
    mutation_plan,
    rows: RowSelection,
    values: dict[str, np.ndarray],
    *,
    group_hinge: bool = False,
) -> TypedBackendMutationBatch:
    """Build the cold-bound complete-value path used by fused reset kernels."""

    buffers_list = [
        np.empty(
            (mutation_plan.num_envs, *spec.value_buffer.row_shape),
            dtype=spec.value_buffer.dtype,
        )
        for spec in mutation_plan.specs
    ]
    groups: tuple[BoundMutationValueBufferGroup, ...] = ()
    if group_hinge:
        group_indices = (
            mutation_plan.spec_index("reset.hinge.position"),
            mutation_plan.spec_index("reset.hinge.velocity"),
        )
        group_buffer = np.empty(
            (len(group_indices), mutation_plan.num_envs, 1, 1),
            dtype=mutation_plan.specs[group_indices[0]].value_buffer.dtype,
        )
        for group_offset, field_index in enumerate(group_indices):
            buffers_list[field_index] = group_buffer[group_offset]
        groups = (
            BoundMutationValueBufferGroup(
                field_indices=group_indices,
                buffer=group_buffer,
            ),
        )
    buffers = tuple(buffers_list)
    for index, spec in enumerate(mutation_plan.specs):
        np.copyto(buffers[index][: rows.count], values[spec.term_key], casting="unsafe")
    bound_buffers = BoundMutationValueBuffers(
        plan=mutation_plan,
        buffers=buffers,
        groups=groups,
    )
    return TypedBackendMutationBatch(
        plan=mutation_plan,
        rows=rows,
        state=SimulationStateMutationBatch(bound_buffer_window=bound_buffers.window(rows)),
    )


def _state_arrays(state) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray(state.buffer("hinge.position").handle).copy(),
        np.asarray(state.buffer("hinge.angular_velocity").handle).copy(),
    )


def _full_state_arrays(state) -> dict[str, np.ndarray]:
    return {
        key: np.asarray(state.buffer(key).handle).copy()
        for key in (
            "root.position",
            "root.orientation",
            "root.linear_velocity",
            "root.angular_velocity",
            "hinge.position",
            "hinge.angular_velocity",
        )
    }


def _set_distinct_hinge_state(backend: MuJoCoBackend) -> tuple[np.ndarray, np.ndarray]:
    """Arrange a public legacy state only as an independent reset oracle."""

    qpos = np.broadcast_to(
        backend.get_default_qpos(),
        (backend.num_envs, len(backend.get_default_qpos())),
    ).copy()
    qvel = np.broadcast_to(
        backend.get_init_qvel(),
        (backend.num_envs, len(backend.get_init_qvel())),
    ).copy()
    position_id = int(backend.get_joint_dof_pos_indices(("hinge",))[0])
    velocity_id = int(backend.get_joint_dof_vel_indices(("hinge",))[0])
    # The fixture has one free root, whose public MuJoCo state layout is 7/6.
    qpos[:, 7 + position_id] = np.asarray((0.1, 0.2, 0.3, 0.4), dtype=qpos.dtype)
    qvel[:, 6 + velocity_id] = np.asarray((0.5, 1.0, 1.5, 2.0), dtype=qvel.dtype)
    backend.set_state(np.arange(backend.num_envs, dtype=np.int32), qpos, qvel)
    return qpos, qvel


def _close(backend: MuJoCoBackend) -> None:
    if backend._pool is not None:
        backend._pool.close()


@pytest.mark.parametrize(
    ("np_dtype", "atol"),
    ((np.float32, 3e-6), (np.float64, 1e-12)),
)
def test_mujoco_typed_reset_commits_selected_hinge_state_and_exposes_reset_oracle(
    tmp_path: Path,
    np_dtype: type[np.floating],
    atol: float,
) -> None:
    backend = _backend(tmp_path, np_dtype=np_dtype)
    reference = _backend(tmp_path / "legacy_reference", np_dtype=np_dtype)
    try:
        plan = backend.bind_task_io(_requirements(backend))
        reference_plan = reference.bind_task_io(_requirements(reference))
        mutation_plan = backend.bind_mutation_plan(
            (_position_spec(np.dtype(np_dtype)), _velocity_spec(np.dtype(np_dtype)))
        )
        # The preceding ball joint gives the same hinge distinct qpos/qvel IDs.
        assert mutation_plan.specs[0].target.entity_ids == (4,)
        assert mutation_plan.specs[1].target.entity_ids == (3,)

        _set_distinct_hinge_state(backend)
        reference_qpos, reference_qvel = _set_distinct_hinge_state(reference)
        before = backend.read_state_batch(plan, RowSelection.all(backend.num_envs))
        before_position_view = before.state.buffer("hinge.position")
        before_position, before_velocity = _state_arrays(before.state)
        rows = RowSelection.selected(backend.num_envs, (3, 1))
        position = np.asarray([[[1.25]], [[-0.75]]], dtype=np_dtype)
        velocity = np.asarray([[[2.5]], [[-3.0]]], dtype=np_dtype)
        mutation = _reset_batch(mutation_plan, rows, position, velocity)
        selected = list(rows.indices or ())
        reference_position_id = int(reference.get_joint_dof_pos_indices(("hinge",))[0])
        reference_velocity_id = int(reference.get_joint_dof_vel_indices(("hinge",))[0])
        reference_qpos[selected, 7 + reference_position_id] = position[:, 0, 0]
        reference_qvel[selected, 6 + reference_velocity_id] = velocity[:, 0, 0]
        reference.set_state(
            np.asarray(selected, dtype=np.int32),
            reference_qpos[selected],
            reference_qvel[selected],
        )

        with (
            patch.object(
                backend,
                "get_joint_dof_pos_indices",
                side_effect=AssertionError("getter fallback"),
            ),
            patch.object(
                backend,
                "get_joint_dof_vel_indices",
                side_effect=AssertionError("getter fallback"),
            ),
            patch.object(backend, "get_body_id", side_effect=AssertionError("selector fallback")),
            patch.object(Path, "read_bytes", side_effect=AssertionError("asset fallback")),
            patch.object(Path, "read_text", side_effect=AssertionError("asset fallback")),
        ):
            result = backend.reset_batch(plan, rows, mutation_batch=mutation)

        assert result.reset_state.phase is StateBatchPhase.RESET
        assert result.reset_state.rows == rows
        with pytest.raises(StaleStateBatchError, match="mutation barrier"):
            _ = before_position_view.handle
        reset_position, reset_velocity = _state_arrays(result.reset_state)
        np.testing.assert_allclose(reset_position[:, 0], position[:, 0, 0], atol=atol, rtol=atol)
        np.testing.assert_allclose(reset_velocity[:, 0], velocity[:, 0, 0], atol=atol, rtol=atol)

        after = backend.read_state_batch(plan, RowSelection.all(backend.num_envs))
        after_position, after_velocity = _state_arrays(after.state)
        np.testing.assert_allclose(
            after_position[[0, 2]], before_position[[0, 2]], atol=atol, rtol=atol
        )
        np.testing.assert_allclose(
            after_velocity[[0, 2]], before_velocity[[0, 2]], atol=atol, rtol=atol
        )
        np.testing.assert_allclose(
            after_position[list(rows.indices or ())], position[:, 0, :], atol=atol, rtol=atol
        )
        np.testing.assert_allclose(
            after_velocity[list(rows.indices or ())], velocity[:, 0, :], atol=atol, rtol=atol
        )

        terminal = backend.step_batch(plan, _zero_control(plan, backend.num_envs))
        expected = reference.step_batch(
            reference_plan,
            _zero_control(reference_plan, reference.num_envs),
        )
        terminal_position, terminal_velocity = _state_arrays(terminal.terminal_state)
        expected_position, expected_velocity = _state_arrays(expected.terminal_state)
        np.testing.assert_allclose(
            terminal_position,
            expected_position,
            atol=20 * atol,
            rtol=20 * atol,
        )
        np.testing.assert_allclose(
            terminal_velocity, expected_velocity, atol=20 * atol, rtol=20 * atol
        )
    finally:
        _close(backend)
        _close(reference)


@pytest.mark.parametrize(
    ("np_dtype", "atol"),
    ((np.float32, 3e-6), (np.float64, 1e-12)),
)
def test_mujoco_typed_reset_commits_full_floating_root_and_hinge_slice(
    tmp_path: Path,
    np_dtype: type[np.floating],
    atol: float,
) -> None:
    """A full managed reset envelope stays backend-owned and row-isolated.

    The independent reference uses only the legacy public setup path before
    the typed commit.  Assertions after the commit inspect public typed state
    views, so this test detects an implementation that writes staging memory
    but fails to forward the actual MuJoCo physics state.
    """

    backend = _backend(tmp_path, np_dtype=np_dtype)
    reference = _backend(tmp_path / "legacy_reference", np_dtype=np_dtype)
    try:
        plan = backend.bind_task_io(_full_reset_requirements(backend))
        reference_plan = reference.bind_task_io(_full_reset_requirements(reference))
        mutation_plan = backend.bind_mutation_plan(_full_reset_specs(np.dtype(np_dtype)))
        bound_specs = {spec.term_key: spec for spec in mutation_plan.specs}
        assert {spec.target.target_key for spec in mutation_plan.specs} == {
            "state.root.position",
            "state.root.orientation",
            "state.root.linear_velocity",
            "state.root.angular_velocity",
            "state.dof.position",
            "state.dof.angular_velocity",
        }
        assert bound_specs["reset.root.position"].target.entity_ids == (1,)
        assert bound_specs["reset.hinge.position"].target.entity_ids == (4,)
        assert bound_specs["reset.hinge.velocity"].target.entity_ids == (3,)

        qpos, qvel = _set_distinct_hinge_state(backend)
        reference_qpos, reference_qvel = _set_distinct_hinge_state(reference)
        qpos[:, :3] = np.asarray(
            ((0.0, 0.1, 0.2), (0.1, 0.2, 0.3), (0.2, 0.3, 0.4), (0.3, 0.4, 0.5)),
            dtype=np_dtype,
        )
        qvel[:, :6] = np.asarray(
            (
                (0.01, 0.02, 0.03, 0.04, 0.05, 0.06),
                (0.11, 0.12, 0.13, 0.14, 0.15, 0.16),
                (0.21, 0.22, 0.23, 0.24, 0.25, 0.26),
                (0.31, 0.32, 0.33, 0.34, 0.35, 0.36),
            ),
            dtype=np_dtype,
        )
        reference_qpos[...] = qpos
        reference_qvel[...] = qvel
        all_rows = np.arange(backend.num_envs, dtype=np.int32)
        backend.set_state(all_rows, qpos, qvel)
        reference.set_state(all_rows, reference_qpos, reference_qvel)

        before = backend.read_state_batch(plan, RowSelection.all(backend.num_envs))
        before_root_view = before.state.buffer("root.position")
        before_values = _full_state_arrays(before.state)
        rows = RowSelection.selected(backend.num_envs, (3, 1))
        values = {
            "reset.root.position": np.asarray(
                [[[1.25, -0.75, 0.55]], [[-0.5, 0.75, 1.1]]], dtype=np_dtype
            ),
            "reset.root.orientation": np.asarray(
                [
                    [[0.5, 0.5, 0.5, 0.5]],
                    [[0.0, 0.0, 0.0, 1.0]],
                ],
                dtype=np_dtype,
            ),
            "reset.root.linear_velocity": np.asarray(
                [[[2.5, -3.0, 1.0]], [[-1.5, 0.25, 0.75]]], dtype=np_dtype
            ),
            "reset.root.angular_velocity": np.asarray(
                [[[0.2, 0.4, 0.6]], [[-0.3, -0.2, -0.1]]], dtype=np_dtype
            ),
            "reset.hinge.position": np.asarray([[[1.25]], [[-0.75]]], dtype=np_dtype),
            "reset.hinge.velocity": np.asarray([[[2.5]], [[-3.0]]], dtype=np_dtype),
        }
        mutation = _full_reset_batch(mutation_plan, rows, values)
        selected = np.asarray(rows.indices, dtype=np.intp)
        reference_position_id = int(reference.get_joint_dof_pos_indices(("hinge",))[0])
        reference_velocity_id = int(reference.get_joint_dof_vel_indices(("hinge",))[0])
        reference_qpos[selected, :3] = values["reset.root.position"][:, 0, :]
        reference_qpos[selected, 3:7] = values["reset.root.orientation"][:, 0, :]
        reference_qvel[selected, :3] = values["reset.root.linear_velocity"][:, 0, :]
        reference_qvel[selected, 3:6] = values["reset.root.angular_velocity"][:, 0, :]
        reference_qpos[selected, 7 + reference_position_id] = values["reset.hinge.position"][
            :, 0, 0
        ]
        reference_qvel[selected, 6 + reference_velocity_id] = values["reset.hinge.velocity"][
            :, 0, 0
        ]
        reference.set_state(
            selected.astype(np.int32),
            reference_qpos[selected],
            reference_qvel[selected],
        )

        with (
            patch.object(backend, "get_body_ids", side_effect=AssertionError("getter fallback")),
            patch.object(backend, "get_body_id", side_effect=AssertionError("selector fallback")),
            patch.object(
                backend,
                "get_joint_dof_pos_indices",
                side_effect=AssertionError("getter fallback"),
            ),
            patch.object(
                backend,
                "get_joint_dof_vel_indices",
                side_effect=AssertionError("getter fallback"),
            ),
            patch.object(backend, "set_state", side_effect=AssertionError("legacy reset fallback")),
            patch.object(Path, "read_bytes", side_effect=AssertionError("asset fallback")),
            patch.object(Path, "read_text", side_effect=AssertionError("asset fallback")),
        ):
            result = backend.reset_batch(plan, rows, mutation_batch=mutation)

        assert result.reset_state.phase is StateBatchPhase.RESET
        assert result.reset_state.rows == rows
        with pytest.raises(StaleStateBatchError, match="mutation barrier"):
            _ = before_root_view.handle
        reset_values = _full_state_arrays(result.reset_state)
        expected_by_state_key = {
            "root.position": values["reset.root.position"][:, 0, :],
            "root.orientation": values["reset.root.orientation"][:, 0, :],
            "root.linear_velocity": values["reset.root.linear_velocity"][:, 0, :],
            "root.angular_velocity": values["reset.root.angular_velocity"][:, 0, :],
            "hinge.position": values["reset.hinge.position"][:, 0, :],
            "hinge.angular_velocity": values["reset.hinge.velocity"][:, 0, :],
        }
        for key, expected in expected_by_state_key.items():
            np.testing.assert_allclose(reset_values[key], expected, atol=atol, rtol=atol)

        after = backend.read_state_batch(plan, RowSelection.all(backend.num_envs))
        after_values = _full_state_arrays(after.state)
        complement = np.asarray((0, 2), dtype=np.intp)
        for key, expected in expected_by_state_key.items():
            np.testing.assert_allclose(
                after_values[key][complement],
                before_values[key][complement],
                atol=atol,
                rtol=atol,
            )
            np.testing.assert_allclose(after_values[key][selected], expected, atol=atol, rtol=atol)

        terminal = backend.step_batch(plan, _zero_control(plan, backend.num_envs))
        expected_terminal = reference.step_batch(
            reference_plan,
            _zero_control(reference_plan, reference.num_envs),
        )
        terminal_values = _full_state_arrays(terminal.terminal_state)
        reference_values = _full_state_arrays(expected_terminal.terminal_state)
        for key in terminal_values:
            np.testing.assert_allclose(
                terminal_values[key],
                reference_values[key],
                atol=20 * atol,
                rtol=20 * atol,
            )
    finally:
        _close(backend)
        _close(reference)


@pytest.mark.parametrize(
    ("np_dtype", "atol"),
    ((np.float32, 3e-6), (np.float64, 1e-12)),
)
def test_mujoco_cold_bound_reset_buffers_commit_complete_state_without_value_wrappers(
    tmp_path: Path,
    np_dtype: type[np.floating],
    atol: float,
) -> None:
    """A complete cold-bound window is equivalent to the public reset envelope.

    The test intentionally supplies no ``MutationValueBatch`` descriptors.
    If the backend regresses to the generic descriptor path it must fail
    closed instead of silently reading a manager-private reset layout.
    """

    backend = _backend(tmp_path, np_dtype=np_dtype)
    try:
        plan = backend.bind_task_io(_full_reset_requirements(backend))
        mutation_plan = backend.bind_mutation_plan(_full_reset_specs(np.dtype(np_dtype)))
        rows = RowSelection.selected(backend.num_envs, (3, 1))
        values = {
            "reset.root.position": np.asarray(
                [[[1.25, -0.75, 0.55]], [[-0.5, 0.75, 1.1]]], dtype=np_dtype
            ),
            "reset.root.orientation": np.asarray(
                [[[0.5, 0.5, 0.5, 0.5]], [[0.0, 0.0, 0.0, 1.0]]], dtype=np_dtype
            ),
            "reset.root.linear_velocity": np.asarray(
                [[[2.5, -3.0, 1.0]], [[-1.5, 0.25, 0.75]]], dtype=np_dtype
            ),
            "reset.root.angular_velocity": np.asarray(
                [[[0.2, 0.4, 0.6]], [[-0.3, -0.2, -0.1]]], dtype=np_dtype
            ),
            "reset.hinge.position": np.asarray([[[1.25]], [[-0.75]]], dtype=np_dtype),
            "reset.hinge.velocity": np.asarray([[[2.5]], [[-3.0]]], dtype=np_dtype),
        }
        mutation = _prepared_full_reset_batch(
            mutation_plan,
            rows,
            values,
            group_hinge=True,
        )
        assert mutation.state.values == ()
        assert mutation.state.bound_buffer_window is not None
        # The public host cache intentionally follows ``np_dtype``, whereas
        # ``BatchEnvPool.reset`` consumes native float64 input.  The typed
        # backend owns a cold-allocated native staging buffer so a float32
        # manager profile does not ask the pool to allocate/cast every reset.
        mutation_runtime = backend._host_mutation_plans[mutation_plan.fingerprint]
        assert mutation_runtime._reset_state.dtype == np.dtype(np.float64)
        assert mutation_runtime._reset_state.flags.c_contiguous

        with (
            patch.object(backend, "get_body_ids", side_effect=AssertionError("getter fallback")),
            patch.object(
                backend, "get_joint_dof_pos_indices", side_effect=AssertionError("getter fallback")
            ),
            patch.object(
                backend, "get_joint_dof_vel_indices", side_effect=AssertionError("getter fallback")
            ),
            patch.object(backend, "set_state", side_effect=AssertionError("legacy reset fallback")),
            patch.object(Path, "read_bytes", side_effect=AssertionError("asset fallback")),
            patch.object(Path, "read_text", side_effect=AssertionError("asset fallback")),
        ):
            result = backend.reset_batch(plan, rows, mutation_batch=mutation)

        window = mutation.state.bound_buffer_window
        assert window is not None
        prepared = mutation_runtime._prepared_buffer_sets[id(window.buffers)]
        assert prepared.owner is window.buffers
        assert len(prepared.groups) == 1
        assert len(prepared.individual) == 4

        reset_values = _full_state_arrays(result.reset_state)
        expected_by_state_key = {
            "root.position": values["reset.root.position"][:, 0, :],
            "root.orientation": values["reset.root.orientation"][:, 0, :],
            "root.linear_velocity": values["reset.root.linear_velocity"][:, 0, :],
            "root.angular_velocity": values["reset.root.angular_velocity"][:, 0, :],
            "hinge.position": values["reset.hinge.position"][:, 0, :],
            "hinge.angular_velocity": values["reset.hinge.velocity"][:, 0, :],
        }
        for key, expected in expected_by_state_key.items():
            np.testing.assert_allclose(reset_values[key], expected, atol=atol, rtol=atol)

        with pytest.raises(MutationContractError, match="rows do not match"):
            TypedBackendMutationBatch(
                plan=mutation_plan,
                rows=RowSelection.all(backend.num_envs),
                state=SimulationStateMutationBatch(
                    bound_buffer_window=mutation.state.bound_buffer_window
                ),
            )

        buffer_set_id = id(window.buffers)
        owner_ref = weakref.ref(window.buffers)
        del prepared
        del window
        del mutation
        gc.collect()
        assert owner_ref() is None
        assert buffer_set_id not in mutation_runtime._prepared_buffer_sets
    finally:
        _close(backend)


def test_cold_bound_reset_buffer_groups_fail_closed_on_invalid_aliasing(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    try:
        mutation_plan = backend.bind_mutation_plan(_full_reset_specs(np.dtype(np.float64)))
        position_index = mutation_plan.spec_index("reset.hinge.position")
        velocity_index = mutation_plan.spec_index("reset.hinge.velocity")
        group_indices = (position_index, velocity_index)
        group_buffer = np.empty((2, mutation_plan.num_envs, 1, 1), dtype=np.float64)
        canonical = tuple(
            np.empty(
                (mutation_plan.num_envs, *spec.value_buffer.row_shape),
                dtype=spec.value_buffer.dtype,
            )
            for spec in mutation_plan.specs
        )
        group = BoundMutationValueBufferGroup(
            field_indices=group_indices,
            buffer=group_buffer,
        )

        with pytest.raises(MutationContractError, match="field slice does not match"):
            BoundMutationValueBuffers(
                plan=mutation_plan,
                buffers=canonical,
                groups=(group,),
            )

        aliased = list(canonical)
        aliased[position_index] = group_buffer[0]
        aliased[velocity_index] = group_buffer[1]
        with pytest.raises(MutationContractError, match="overlap canonical fields"):
            BoundMutationValueBuffers(
                plan=mutation_plan,
                buffers=tuple(aliased),
                groups=(group, group),
            )

        root_position_index = mutation_plan.spec_index("reset.root.position")
        heterogeneous = BoundMutationValueBufferGroup(
            field_indices=(root_position_index, position_index),
            buffer=np.empty((2, mutation_plan.num_envs, 1, 3), dtype=np.float64),
        )
        with pytest.raises(MutationContractError, match="not homogeneous"):
            BoundMutationValueBuffers(
                plan=mutation_plan,
                buffers=canonical,
                groups=(heterogeneous,),
            )
    finally:
        _close(backend)


def test_mujoco_typed_reset_binding_and_commit_faults_fail_closed(tmp_path: Path) -> None:
    unmaterialized = _backend(tmp_path / "unmaterialized", materialize=False)
    with pytest.raises(BackendBatchContractError, match="materialized backend pool"):
        unmaterialized.bind_task_io(_requirements(unmaterialized))
    with pytest.raises(BackendBatchContractError, match="materialized backend pool"):
        unmaterialized.bind_mutation_plan((_position_spec(np.dtype(np.float64)),))

    misbound_root = _backend(tmp_path / "misbound_root", base_name="ball_link")
    try:
        with pytest.raises(MutationContractError, match="not supported by the backend"):
            misbound_root.bind_mutation_plan((_full_reset_specs(np.dtype(np.float64))[0],))
    finally:
        _close(misbound_root)

    backend = _backend(tmp_path / "materialized")
    try:
        plan = backend.bind_task_io(_requirements(backend))
        position_spec = _position_spec(np.dtype(np.float64))
        velocity_spec = _velocity_spec(np.dtype(np.float64))
        mutation_plan = backend.bind_mutation_plan((position_spec, velocity_spec))
        rows = RowSelection.selected(backend.num_envs, (2, 0))
        valid = _reset_batch(
            mutation_plan,
            rows,
            np.asarray([[[0.25]], [[-0.5]]], dtype=np.float64),
            np.asarray([[[1.5]], [[-2.0]]], dtype=np.float64),
        )

        compiled_selector = MutationSelectorSpec(
            semantic_key="robot.managed_hinge",
            mode=MutationSelectorMode.EXACT,
            expressions=("hinge",),
            entity_ids=(37,),
        )
        structured = _position_spec(np.dtype(np.float64), selector=compiled_selector)
        structured_plan = backend.bind_mutation_plan((structured,))
        assert (
            structured_plan.specs[0].target.entity_ids == mutation_plan.specs[0].target.entity_ids
        )

        root_specs = _full_reset_specs(np.dtype(np.float64))
        root_plan = backend.bind_mutation_plan(root_specs)
        root_bound = {spec.term_key: spec for spec in root_plan.specs}
        assert root_bound["reset.root.position"].target.entity_ids == (1,)
        structured_root = replace(
            root_specs[0],
            target=replace(
                root_specs[0].target,
                selector=MutationSelectorSpec(
                    semantic_key="robot.managed_root",
                    mode=MutationSelectorMode.EXACT,
                    expressions=("payload",),
                    entity_ids=(1,),
                ),
            ),
        )
        assert backend.bind_mutation_plan((structured_root,)).specs[0].target.entity_ids == (1,)

        unsupported_specs = (
            replace(
                position_spec,
                target=replace(position_spec.target, field_kind=MutationFieldKind.LINEAR_VELOCITY),
            ),
            replace(position_spec, commit_phase=MutationCommitPhase.PRE_PHYSICS),
            replace(position_spec, operation=MutationOperation.ADD),
            replace(position_spec, persistence=MutationPersistence.ONE_STEP),
            _position_spec(np.dtype(np.float64), selector="payload_free"),
            _position_spec(np.dtype(np.float64), selector="prefix_ball"),
            _position_spec(np.dtype(np.float64), selector="slide"),
            _position_spec(
                np.dtype(np.float64),
                selector=MutationSelectorSpec(
                    semantic_key="robot.regex_hinges",
                    mode=MutationSelectorMode.REGEX,
                    expressions=(".*hinge.*",),
                    entity_ids=(37,),
                ),
            ),
            _position_spec(
                np.dtype(np.float64),
                selector=MutationSelectorSpec(
                    semantic_key="robot.two_hinges",
                    mode=MutationSelectorMode.EXACT,
                    expressions=("hinge", "slide"),
                    entity_ids=(37, 41),
                ),
            ),
            replace(
                root_specs[0],
                target=replace(root_specs[0].target, selector="ball_link"),
            ),
            replace(
                root_specs[0],
                target=replace(root_specs[0].target, selector="payload_free"),
            ),
            replace(
                root_specs[0],
                target=replace(
                    root_specs[0].target,
                    field_kind=MutationFieldKind.ORIENTATION,
                ),
            ),
        )
        for spec in unsupported_specs:
            with pytest.raises(MutationContractError):
                backend.bind_mutation_plan((spec,))

        with pytest.raises(BackendBatchContractError, match="TypedBackendMutationBatch"):
            backend.reset_batch(plan, rows)
        with pytest.raises(BackendBatchContractError, match="rows must match"):
            backend.reset_batch(
                plan,
                rows,
                mutation_batch=_reset_batch(
                    mutation_plan,
                    RowSelection.selected(backend.num_envs, (0, 2)),
                    np.asarray([[[0.25]], [[-0.5]]], dtype=np.float64),
                    np.asarray([[[1.5]], [[-2.0]]], dtype=np.float64),
                ),
            )
        with pytest.raises(BackendBatchContractError, match="different backend"):
            backend.reset_batch(
                plan,
                rows,
                mutation_batch=TypedBackendMutationBatch(
                    plan=replace(mutation_plan, backend_instance_id="mujoco:other"),
                    rows=rows,
                ),
            )
        with pytest.raises(BackendBatchContractError, match="not bound by this backend"):
            backend.reset_batch(
                replace(plan, fingerprint="backend-batch-contract-v1:other"),
                rows,
                mutation_batch=valid,
            )
        with pytest.raises(BackendBatchContractError, match="at least one state value"):
            backend.reset_batch(
                plan,
                rows,
                mutation_batch=TypedBackendMutationBatch(plan=mutation_plan, rows=rows),
            )

        field_index = mutation_plan.spec_index("reset.hinge.position")
        contract = mutation_plan.specs[field_index].value_buffer
        malformed_handles = (
            np.zeros((rows.count, 1, 1), dtype=np.float32),
            np.zeros((rows.count, 2, 1), dtype=np.float64),
            np.zeros((rows.count, 2, 1), dtype=np.float64)[:, :1, :],
        )
        for handle in malformed_handles:
            malformed_value = MutationValueBatch(
                plan=mutation_plan,
                field_index=field_index,
                rows=rows,
                buffer=BufferView(handle, (rows.count, 1, 1), contract),
            )
            malformed = TypedBackendMutationBatch(
                plan=mutation_plan,
                rows=rows,
                state=SimulationStateMutationBatch((malformed_value,)),
            )
            with pytest.raises(BackendBatchContractError, match="value handle|C-contiguous"):
                backend.reset_batch(plan, rows, mutation_batch=malformed)

        mixed_plan = backend.bind_mutation_plan((position_spec, _force_spec(np.dtype(np.float64))))
        force_index = mixed_plan.spec_index("push.payload")
        force_contract = mixed_plan.specs[force_index].value_buffer
        force = MutationValueBatch(
            plan=mixed_plan,
            field_index=force_index,
            rows=rows,
            buffer=BufferView(
                np.ones((rows.count, 1, 3), dtype=np.float64),
                (rows.count, 1, 3),
                force_contract,
            ),
        )
        with pytest.raises(BackendBatchContractError, match="only supports simulation-state"):
            backend.reset_batch(
                plan,
                rows,
                mutation_batch=TypedBackendMutationBatch(
                    plan=mixed_plan,
                    rows=rows,
                    wrench=ExternalWrenchMutationBatch((force,)),
                ),
            )

        backend.apply_body_force(
            np.asarray([backend.get_body_id("payload")], dtype=np.int32),
            np.ones((backend.num_envs, 1, 3), dtype=np.float64),
        )
        with pytest.raises(BackendBatchContractError, match="out-of-band external wrench"):
            backend.reset_batch(plan, rows, mutation_batch=valid)
    finally:
        _close(backend)
