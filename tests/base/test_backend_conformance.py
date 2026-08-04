"""Cross-backend conformance suite driven only through the public contract.

Every interaction goes through ``create_backend`` and the public ``SimBackend``
surface.  MuJoCo must pass the full legacy and typed flow; mjwarp runs when a
CUDA Warp device is available (slow lane); motrix/drake must at least fail
closed with ``NotImplementedError`` on the typed batch contract.
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base.backend import (
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
    StateEntityKind,
    StateFieldKind,
    StateFieldSpec,
    TypedBackendMutationBatch,
    create_backend,
)
from unilab.base.scene import SceneCfg
from unilab.tools.backend_isolation import audit_backend_isolation

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
_FACTORY_FILE = SRC_ROOT / "unilab" / "base" / "backend" / "__init__.py"
_BACKEND_CLASS_NAMES = frozenset(
    {"MuJoCoBackend", "MotrixBackend", "DrakeBackend", "MjwarpBackend"}
)

NUM_ENVS = 2
SIM_DT = 0.005
_G1_SCENE = str(ASSETS_ROOT_PATH / "robots" / "g1" / "scene_flat.xml")


def _module_available(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def _mjwarp_cuda_available() -> bool:
    from unilab.base.backend.mjwarp.dependencies import (
        load_mjwarp_dependencies,
        mjwarp_dependencies_available,
    )

    if not mjwarp_dependencies_available():
        return False
    try:
        return bool(load_mjwarp_dependencies().warp.get_device().is_cuda)
    except Exception:
        return False


def _drake_batch_available() -> bool:
    try:
        from unilab.base.backend.drake.backend import ensure_drake_batch_available
    except ImportError:
        return False
    available, _ = ensure_drake_batch_available()
    return bool(available)


def _require_backend(backend_type: str) -> None:
    if backend_type == "mujoco":
        pytest.importorskip("mujoco", reason="mujoco not installed")
    elif backend_type == "motrix":
        pytest.importorskip("motrixsim", reason="motrixsim not installed")
    elif backend_type == "mjwarp":
        if not _mjwarp_cuda_available():
            pytest.skip("mjwarp requires an active CUDA Warp device")
    elif backend_type == "drake":
        if not _drake_batch_available():
            pytest.skip("drake batch extension not available")


_BACKEND_PARAMS = [
    pytest.param("mujoco", id="mujoco"),
    pytest.param("motrix", id="motrix"),
    pytest.param("drake", id="drake"),
    pytest.param("mjwarp", id="mjwarp", marks=pytest.mark.slow),
]


def test_backend_classes_are_only_instantiated_through_create_backend() -> None:
    offenders: list[str] = []
    for path in sorted(SRC_ROOT.rglob("*.py")):
        if path == _FACTORY_FILE or "__pycache__" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = None
            if isinstance(func, ast.Name):
                name = func.id
            elif isinstance(func, ast.Attribute):
                name = func.attr
            if name in _BACKEND_CLASS_NAMES:
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}: {name}(")
    assert not offenders, "direct backend instantiation outside create_backend:\n" + "\n".join(
        offenders
    )


def test_real_repo_backend_isolation_audit_passes() -> None:
    report = audit_backend_isolation(REPO_ROOT)
    assert report.ok, "\n".join(violation.format() for violation in report.violations)


@pytest.mark.parametrize("backend_type", _BACKEND_PARAMS)
def test_legacy_contract_step_set_state_and_state_reads(backend_type: str) -> None:
    _require_backend(backend_type)

    backend = create_backend(
        backend_type,
        SceneCfg(model_file=_G1_SCENE),
        NUM_ENVS,
        SIM_DT,
        base_name="pelvis",
    )
    backend.materialize()

    assert backend.num_envs == NUM_ENVS
    assert backend.num_actuators > 0

    backend.step(np.zeros((NUM_ENVS, backend.num_actuators)), nsteps=2)

    default_qpos = np.asarray(backend.get_default_qpos())
    qpos = np.broadcast_to(default_qpos, (NUM_ENVS, default_qpos.shape[0])).copy()
    target_xyz = np.array([1.0, 2.0, 0.8])
    qpos[:, :3] = target_xyz
    qvel = np.zeros((NUM_ENVS, len(backend.get_init_qvel())))
    backend.set_state(np.arange(NUM_ENVS, dtype=np.int32), qpos, qvel)

    np.testing.assert_allclose(
        backend.get_base_pos(), np.tile(target_xyz, (NUM_ENVS, 1)), atol=1e-4
    )
    np.testing.assert_allclose(
        np.linalg.norm(backend.get_base_quat(), axis=-1), 1.0, atol=1e-5
    )


def _write_free_hinge_model(tmp_path: Path) -> Path:
    model_file = tmp_path / "conformance_free_hinge.xml"
    model_file.write_text(
        """
<mujoco model="conformance_free_hinge">
  <option timestep="0.01" gravity="0 0 0"/>
  <worldbody>
    <body name="payload">
      <freejoint name="payload_free"/>
      <geom name="payload_geom" type="sphere" size="0.1" mass="1"/>
      <body name="hinge_link" pos="0 0 0.15">
        <joint name="hinge" type="hinge" axis="0 1 0"/>
        <geom name="hinge_geom" type="capsule" fromto="0 0 0 0 0 0.15" size="0.02" mass="0.1"/>
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


def _state_buffer(dtype: np.dtype) -> BufferContract:
    return BufferContract(
        row_shape=(1,),
        dtype=dtype.name,
        layout=BufferLayout.C_CONTIGUOUS,
        placement=BufferPlacement.host(),
        owner=BufferOwner.BACKEND,
        mutability=BufferMutability.READ_ONLY,
        lifetime=BufferLifetime.BORROWED_UNTIL_MUTATION,
        dlpack_exportable=False,
    )


def _typed_requirements(backend) -> BackendIORequirements:
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
        hot_path_budget=BackendBatchCounterBudget(allocations=8, state_materializations=2),
    )


def _reset_spec(dtype: np.dtype, *, term_key: str, target_key: str, field_kind) -> MutationSpec:
    return MutationSpec(
        term_key=term_key,
        target=MutationTargetSpec(
            target_key=target_key,
            target_kind=MutationTargetKind.SIMULATION_STATE,
            entity_kind=MutationEntityKind.DOF,
            field_kind=field_kind,
            selector="hinge",
        ),
        trigger=MutationTrigger.RESET,
        commit_phase=MutationCommitPhase.RESET,
        operation=MutationOperation.SET,
        baseline=MutationBaseline.DEFAULT,
        persistence=MutationPersistence.EPISODE,
        recompute=MutationRecomputeLevel.KINEMATICS,
        value_template=BufferContract(
            row_shape=(1,),
            dtype=dtype.name,
            layout=BufferLayout.C_CONTIGUOUS,
            placement=BufferPlacement.host(),
            owner=BufferOwner.MANAGER,
            mutability=BufferMutability.READ_ONLY,
            lifetime=BufferLifetime.UNTIL_COMMIT,
            dlpack_exportable=False,
        ),
    )


def _state_value(mutation_plan, term_key: str, rows: RowSelection, values: np.ndarray):
    field_index = mutation_plan.spec_index(term_key)
    contract = mutation_plan.specs[field_index].value_buffer
    handle = np.ascontiguousarray(values, dtype=contract.dtype)
    return MutationValueBatch(
        plan=mutation_plan,
        field_index=field_index,
        rows=rows,
        buffer=BufferView(handle, handle.shape, contract),
    )


@pytest.mark.parametrize("backend_type", _BACKEND_PARAMS)
def test_typed_contract_bind_step_read_reset(backend_type: str, tmp_path: Path) -> None:
    _require_backend(backend_type)

    kwargs = {"chunk_size": NUM_ENVS, "bench_nsteps": 1} if backend_type == "mujoco" else {}
    backend = create_backend(
        backend_type,
        SceneCfg(model_file=str(_write_free_hinge_model(tmp_path))),
        NUM_ENVS,
        0.01,
        base_name="payload",
        **kwargs,
    )
    backend.materialize()
    requirements = _typed_requirements(backend)

    if backend_type in ("motrix", "drake"):
        # Fail-closed on the typed contract is conformant for backends that do
        # not implement typed batches yet.
        with pytest.raises(NotImplementedError, match="does not support typed backend batches"):
            backend.bind_task_io(requirements)
        return

    dtype = np.dtype(backend.get_init_qvel().dtype)
    plan = backend.bind_task_io(requirements)
    rows = RowSelection.all(NUM_ENVS)

    before = backend.read_state_batch(plan, rows)
    before_position = np.asarray(before.state.buffer("hinge.position").handle).copy()
    assert before_position.shape == (NUM_ENVS, 1)

    mutation_plan = backend.bind_mutation_plan(
        (
            _reset_spec(
                dtype,
                term_key="reset.hinge.position",
                target_key="state.dof.position",
                field_kind=MutationFieldKind.POSITION,
            ),
            _reset_spec(
                dtype,
                term_key="reset.hinge.velocity",
                target_key="state.dof.angular_velocity",
                field_kind=MutationFieldKind.ANGULAR_VELOCITY,
            ),
        )
    )
    position = np.asarray([[[0.5]], [[-0.25]]], dtype=dtype)
    velocity = np.asarray([[[1.0]], [[-2.0]]], dtype=dtype)
    mutation = TypedBackendMutationBatch(
        plan=mutation_plan,
        rows=rows,
        state=SimulationStateMutationBatch(
            (
                _state_value(mutation_plan, "reset.hinge.position", rows, position),
                _state_value(mutation_plan, "reset.hinge.velocity", rows, velocity),
            )
        ),
    )
    reset_result = backend.reset_batch(plan, rows, mutation_batch=mutation)
    reset_position = np.asarray(reset_result.reset_state.buffer("hinge.position").handle)
    reset_velocity = np.asarray(reset_result.reset_state.buffer("hinge.angular_velocity").handle)
    np.testing.assert_allclose(reset_position, position[:, 0, :], atol=1e-5)
    np.testing.assert_allclose(reset_velocity, velocity[:, 0, :], atol=1e-5)

    control = np.zeros((NUM_ENVS, *plan.control.buffer.row_shape), dtype=plan.control.buffer.dtype)
    terminal = backend.step_batch(
        plan,
        ControlBatch(
            plan=plan,
            rows=rows,
            buffer=BufferView(control, control.shape, plan.control.buffer),
        ),
    )
    terminal_position = np.asarray(terminal.terminal_state.buffer("hinge.position").handle)
    assert terminal_position.shape == (NUM_ENVS, 1)
    assert np.all(np.isfinite(terminal_position))
