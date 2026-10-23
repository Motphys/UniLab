"""Cross-backend conformance suite driven only through the public contract.

Every interaction goes through ``create_backend`` and the public ``SimBackend``
surface. mjwarp runs when a CUDA Warp device is available (slow lane).
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch
from unisim.backend.base import (
    RootStateLayout,
    SimBackend,
    TerrainSpawnData,
)

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base.backend_factory import create_backend
from unilab.base.reset_state import ResetStateTransaction
from unilab.base.scene import SceneCfg

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
_FACTORY_FILE = SRC_ROOT / "unilab" / "base" / "backend" / "__init__.py"
_BACKEND_CLASS_NAMES = frozenset(
    {
        "MuJoCoBackend",
        "MotrixBackend",
        "DrakeBackend",
        "MjwarpBackend",
        "IsaacGymBackend",
        "GenesisBackend",
        "IsaacSimBackend",
        "NewtonBackend",
    }
)
_TASK_SOURCE_ROOTS = (
    SRC_ROOT / "unilab" / "envs",
    SRC_ROOT / "unilab" / "tasks",
)
_TERRAIN_CONSUMER_PATHS: tuple[Path, ...] = ()

NUM_ENVS = 2
SIM_DT = 0.005
_G1_SCENE = str(ASSETS_ROOT_PATH / "robots" / "g1" / "scene_flat.xml")


def _module_available(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def _mjwarp_cuda_available() -> bool:
    from unisim.backend.mjwarp.dependencies import (
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
        from unisim.backend.drake.backend import ensure_drake_batch_available
    except ImportError:
        return False
    available, _ = ensure_drake_batch_available()
    return bool(available)


def _isaacgym_runtime_available() -> bool:
    from unisim.backend.isaacgym.dependencies import isaacgym_runtime_available

    return isaacgym_runtime_available()


def _genesis_runtime_available() -> bool:
    from unisim.backend.genesis.dependencies import genesis_dependencies_available

    if not genesis_dependencies_available():
        return False
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _isaacsim_runtime_available() -> bool:
    from unisim.backend.isaacsim.dependencies import isaacsim_runtime_available

    return isaacsim_runtime_available()


def _newton_runtime_available() -> bool:
    try:
        from unisim.backend.newton.dependencies import load_newton_dependencies

        dependencies = load_newton_dependencies()
        return bool(dependencies.warp.get_device().is_cuda)
    except Exception:
        return False


def _require_backend(backend_type: str) -> None:
    if backend_type == "mujoco":
        pytest.importorskip("mujoco", reason="mujoco not installed")
        pytest.importorskip(
            "unisim.backend.mujoco.backend",
            reason="unisim-core MuJoCo adapter (mjbatch build) not available",
        )
    elif backend_type == "motrix":
        pytest.importorskip("motrixsim", reason="motrixsim not installed")
    elif backend_type == "mjwarp":
        if not _mjwarp_cuda_available():
            pytest.skip("mjwarp requires an active CUDA Warp device")
    elif backend_type == "drake":
        if not _drake_batch_available():
            pytest.skip("drake batch extension not available")
    elif backend_type == "isaacgym":
        if not _isaacgym_runtime_available():
            pytest.skip("isaacgym requires the Python 3.8 worker runtime")
    elif backend_type == "genesis":
        if not _genesis_runtime_available():
            pytest.skip("genesis requires the genesis-world extra and a CUDA device")
    elif backend_type == "isaacsim":
        if not _isaacsim_runtime_available():
            pytest.skip("isaacsim requires the Python 3.11 IsaacSim/IsaacLab worker runtime")
        try:
            import torch

            if not torch.cuda.is_available():
                pytest.skip("isaacsim requires a CUDA-enabled NVIDIA device")
        except ImportError:
            pytest.skip("isaacsim conformance requires host torch to check CUDA visibility")
    elif backend_type == "newton":
        if not _newton_runtime_available():
            pytest.skip("newton requires the pinned runtime and an active CUDA Warp device")


_BACKEND_PARAMS = [
    pytest.param("mujoco", id="mujoco"),
    pytest.param("motrix", id="motrix"),
    pytest.param("drake", id="drake"),
    pytest.param("mjwarp", id="mjwarp", marks=pytest.mark.slow),
    pytest.param("isaacgym", id="isaacgym", marks=pytest.mark.slow),
    pytest.param("genesis", id="genesis", marks=pytest.mark.slow),
    pytest.param("isaacsim", id="isaacsim", marks=pytest.mark.slow),
    pytest.param("newton", id="newton", marks=pytest.mark.slow),
]


def _create_backend_for_test(backend_type: str, *args, **kwargs):
    if backend_type == "newton":
        kwargs.setdefault("newton_device", "cuda:0")
        kwargs.setdefault("newton_nconmax", 128)
        kwargs.setdefault("newton_njmax", 256)
    return create_backend(backend_type, *args, **kwargs)


def _skip_raw_scalar_isaacsim_fixture() -> None:
    pytest.skip("IsaacSim tensor lifecycle requires its mapped G1 owner fixture")


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


def test_terrain_spawn_contract_is_read_only_and_defaults_to_unsupported() -> None:
    origins = np.arange(12, dtype=np.float64).reshape(2, 2, 3)

    def sample_height(xy: np.ndarray) -> np.ndarray:
        return np.zeros(np.asarray(xy).shape[:-1], dtype=np.float64)

    data = TerrainSpawnData(origins, sample_height=sample_height)

    assert SimBackend.get_terrain_spawn_data(object()) is None  # type: ignore[arg-type]
    assert data.sample_height is sample_height
    assert not data.terrain_origins.flags.writeable
    assert not np.shares_memory(data.terrain_origins, origins)
    origins.fill(-1.0)
    assert np.all(data.terrain_origins >= 0.0)

    with pytest.raises(ValueError, match="terrain_origins must have shape"):
        TerrainSpawnData(np.zeros((2, 3), dtype=np.float64))
    with pytest.raises(TypeError, match="sample_height must be callable"):
        TerrainSpawnData(
            np.zeros((1, 1, 3), dtype=np.float64),
            sample_height=object(),  # type: ignore[arg-type]
        )


def test_actuation_metadata_defaults_fail_closed() -> None:
    with pytest.raises(NotImplementedError, match="actuator target joints"):
        SimBackend.get_actuator_joint_names(object())  # type: ignore[arg-type]
    with pytest.raises(NotImplementedError, match="default DoF positions"):
        SimBackend.get_default_dof_pos(object())  # type: ignore[arg-type]
    with pytest.raises(NotImplementedError, match="get_joint_state_qpos_indices"):
        SimBackend.get_joint_state_qpos_indices(object(), ("joint",))  # type: ignore[arg-type]
    with pytest.raises(NotImplementedError, match="get_joint_state_qvel_indices"):
        SimBackend.get_joint_state_qvel_indices(object(), ("joint",))  # type: ignore[arg-type]
    with pytest.raises(NotImplementedError, match="root-state layout"):
        SimBackend.get_root_state_layout(object(), "root")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("qpos_indices", "qvel_indices", "error", "match"),
    [
        (list(range(7)), tuple(range(6)), TypeError, "qpos_indices must be a tuple"),
        (tuple(range(6)), tuple(range(6)), ValueError, "qpos_indices must contain 7"),
        (tuple(range(7)), tuple(range(5)), ValueError, "qvel_indices must contain 6"),
        ((0, 1, 2, 3, 4, 5, True), tuple(range(6)), TypeError, "integer columns"),
        ((0, 1, 2, 3, 4, 5, -1), tuple(range(6)), ValueError, "negative columns"),
        ((0, 1, 2, 3, 4, 5, 5), tuple(range(6)), ValueError, "unique columns"),
    ],
)
def test_root_state_layout_metadata_fails_closed(
    qpos_indices,
    qvel_indices,
    error,
    match: str,
) -> None:
    with pytest.raises(error, match=match):
        RootStateLayout(qpos_indices, qvel_indices)


@pytest.mark.parametrize("backend_type", _BACKEND_PARAMS)
def test_actuation_metadata_contract(backend_type: str) -> None:
    _require_backend(backend_type)
    if backend_type == "isaacsim":
        _skip_raw_scalar_isaacsim_fixture()

    backend = _create_backend_for_test(
        backend_type,
        SceneCfg(model_file=_G1_SCENE),
        NUM_ENVS,
        SIM_DT,
        base_name="pelvis",
    )
    backend.materialize()
    tensor_device = torch.device("cuda") if backend_type == "isaacgym" else torch.device("cpu")

    actuator_names = backend.get_actuator_names()
    target_joint_names = backend.get_actuator_joint_names()
    default_dof_pos = backend.get_default_dof_pos()
    reset_qpos_ids = backend.get_joint_state_qpos_indices(target_joint_names)
    reset_qvel_ids = backend.get_joint_state_qvel_indices(target_joint_names)
    current_qpos = backend.get_state_views(("qpos",))["qpos"].detach().cpu().numpy()

    assert len(actuator_names) == backend.num_actuators
    assert len(set(actuator_names)) == len(actuator_names)
    assert all(actuator_names)
    assert len(target_joint_names) == backend.num_actuators
    assert all(target_joint_names)
    assert default_dof_pos.shape == current_qpos[:, reset_qpos_ids].shape[1:]
    assert np.issubdtype(default_dof_pos.dtype, np.floating)
    assert np.isfinite(default_dof_pos).all()
    assert reset_qpos_ids.shape == (backend.num_actuators,)
    assert reset_qvel_ids.shape == (backend.num_actuators,)
    assert np.issubdtype(reset_qpos_ids.dtype, np.integer)
    assert np.issubdtype(reset_qvel_ids.dtype, np.integer)
    assert np.unique(reset_qpos_ids).size == reset_qpos_ids.size
    assert np.unique(reset_qvel_ids).size == reset_qvel_ids.size
    assert np.all((reset_qpos_ids >= 0) & (reset_qpos_ids < backend.get_default_qpos().size))
    assert np.all((reset_qvel_ids >= 0) & (reset_qvel_ids < backend.get_init_qvel().size))
    np.testing.assert_allclose(default_dof_pos, current_qpos[0, reset_qpos_ids], atol=1e-6)

    qpos = np.broadcast_to(
        backend.get_default_qpos(), (NUM_ENVS, backend.get_default_qpos().size)
    ).copy()
    qvel = np.broadcast_to(backend.get_init_qvel(), (NUM_ENVS, backend.get_init_qvel().size)).copy()
    expected_joint_pos = np.broadcast_to(default_dof_pos + 0.01, (NUM_ENVS, default_dof_pos.size))
    qpos[:, reset_qpos_ids] = expected_joint_pos
    backend.set_state_tensor(
        torch.arange(NUM_ENVS, dtype=torch.int64, device=tensor_device),
        torch.from_numpy(qpos.astype(np.float32, copy=False)).to(tensor_device),
        torch.from_numpy(qvel.astype(np.float32, copy=False)).to(tensor_device),
    )
    joint_pos = backend.get_state_views(("qpos",))["qpos"].detach().cpu().numpy()
    np.testing.assert_allclose(joint_pos[:, reset_qpos_ids], expected_joint_pos, atol=1e-5)

    detached = default_dof_pos.copy()
    default_dof_pos[:] = np.nan
    np.testing.assert_array_equal(backend.get_default_dof_pos(), detached)


@pytest.mark.parametrize(
    "backend_type",
    [
        pytest.param("mujoco", id="mujoco"),
        pytest.param(
            "motrix",
            id="motrix",
            marks=pytest.mark.xfail(
                strict=True,
                reason="unisim-core 1.7.24 leaks a native RuntimeError for missing sensors (unisim#384)",
            ),
        ),
        pytest.param("drake", id="drake"),
        pytest.param("mjwarp", id="mjwarp", marks=pytest.mark.slow),
        pytest.param("isaacgym", id="isaacgym", marks=pytest.mark.slow),
        pytest.param("genesis", id="genesis", marks=pytest.mark.slow),
        pytest.param("isaacsim", id="isaacsim", marks=pytest.mark.slow),
        pytest.param("newton", id="newton", marks=pytest.mark.slow),
    ],
)
def test_named_sensor_view_contract(backend_type: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """All available adapters expose the same ordered, finite sensor view."""
    _require_backend(backend_type)
    if backend_type == "isaacsim":
        _skip_raw_scalar_isaacsim_fixture()

    backend = _create_backend_for_test(
        backend_type,
        SceneCfg(model_file=_G1_SCENE),
        NUM_ENVS,
        SIM_DT,
        base_name="pelvis",
    )
    backend.materialize()
    names = (
        ("torso_gyro", "pelvis_local_linvel")
        if backend_type == "isaacgym"
        else ("pelvis_gyro", "pelvis_local_linvel")
    )

    view = backend.bind_sensor_data(names)
    assert view.backend_type == backend.backend_type
    assert view.names == names
    assert view.dimensions == (3, 3)
    assert view.width == 6
    values = view.read()
    assert values.shape == (NUM_ENVS, 6)
    assert np.isfinite(values).all()
    sensor_values = [backend.get_sensor_view(name).detach().cpu().numpy() for name in names]
    np.testing.assert_allclose(values, np.concatenate(sensor_values, axis=1))

    def fail_if_public_batch_getter_is_used(*_args, **_kwargs):
        raise AssertionError("materialized sensor view called the public batch getter")

    monkeypatch.setattr(backend, "get_sensor_view", fail_if_public_batch_getter_is_used)
    np.testing.assert_allclose(view.read(), values)

    with pytest.raises((KeyError, ValueError), match="missing_sensor|Missing|missing"):
        backend.bind_sensor_data(("missing_sensor",))


@pytest.mark.parametrize("backend_type", _BACKEND_PARAMS)
def test_root_state_layout_contract(backend_type: str) -> None:
    _require_backend(backend_type)

    backend = _create_backend_for_test(
        backend_type,
        SceneCfg(model_file=_G1_SCENE),
        NUM_ENVS,
        SIM_DT,
        base_name="pelvis",
    )
    backend.materialize()

    if backend_type == "drake":
        with pytest.raises(
            NotImplementedError,
            match="DrakeBackend does not expose root-state layout.*pelvis",
        ):
            backend.get_root_state_layout("pelvis")
        return

    layout = backend.get_root_state_layout("pelvis")
    assert isinstance(layout, RootStateLayout)
    qpos_indices = np.asarray(layout.qpos_indices)
    qvel_indices = np.asarray(layout.qvel_indices)
    assert qpos_indices.shape == (7,)
    assert qvel_indices.shape == (6,)
    assert np.all(qpos_indices < backend.get_default_qpos().size)
    assert np.all(qvel_indices < backend.get_init_qvel().size)
    root_pose = backend.get_default_qpos()[qpos_indices]
    np.testing.assert_allclose(np.linalg.norm(root_pose[3:7]), 1.0, atol=1e-6)


@pytest.mark.parametrize("backend_type", ["mujoco", "motrix"])
def test_root_qvel_body_angular_contract_reads_back_world_velocity(backend_type: str) -> None:
    _require_backend(backend_type)
    backend = _create_backend_for_test(
        backend_type,
        SceneCfg(model_file=_G1_SCENE),
        1,
        SIM_DT,
        base_name="pelvis",
        add_body_sensors=True,
    )
    backend.materialize()
    layout = backend.get_root_state_layout("pelvis")
    qpos_indices = np.asarray(layout.qpos_indices)
    qvel_indices = np.asarray(layout.qvel_indices)
    qpos = backend.get_default_qpos()[None].copy()
    qvel = backend.get_init_qvel()[None].copy()
    half_sqrt = np.sqrt(0.5)
    qpos[0, qpos_indices[3:7]] = [half_sqrt, 0.0, 0.0, half_sqrt]
    qvel[0, qvel_indices[3:6]] = [0.0, -1.0, 0.0]

    backend.set_state_tensor(
        torch.tensor([0], dtype=torch.int64),
        torch.from_numpy(qpos.astype(np.float32, copy=False)),
        torch.from_numpy(qvel.astype(np.float32, copy=False)),
    )

    angular_velocity = backend.get_sensor_view("track_angvel_w_pelvis").detach().cpu().numpy()
    np.testing.assert_allclose(angular_velocity[0], [1.0, 0.0, 0.0], atol=2e-6)


@pytest.mark.parametrize("backend_type", ["mujoco", "motrix"])
def test_fused_tensor_root_reset_reads_back_world_angular_velocity(
    backend_type: str,
) -> None:
    _require_backend(backend_type)
    backend = _create_backend_for_test(
        backend_type,
        SceneCfg(model_file=_G1_SCENE),
        1,
        SIM_DT,
        base_name="pelvis",
        add_body_sensors=True,
    )
    backend.materialize()
    try:
        layout = backend.get_root_state_layout("pelvis")
        qvel_indices = np.asarray(layout.qvel_indices)
        reset = ResetStateTransaction(backend)
        reset.declare_packed_reset_device(torch.device("cpu"))
        root_state = torch.zeros((1, 13), dtype=torch.float32)
        half_sqrt = float(np.sqrt(0.5))
        root_state[:, 3:7] = torch.tensor([half_sqrt, 0.0, 0.0, half_sqrt])
        root_state[:, 10:13] = torch.tensor([[1.0, 0.0, 0.0]])
        reset_mask = torch.ones(1, dtype=torch.bool)

        with reset.scoped_device_event_tensor(reset_mask):
            reset.write_root_state_tensor(
                reset_mask,
                layout,
                root_state,
                term_name="fused_root_reset_contract",
            )

        generalized_qvel = backend.get_state_views(("qvel",))["qvel"].detach().cpu().numpy()
        np.testing.assert_allclose(
            generalized_qvel[0, qvel_indices[3:6]],
            [0.0, -1.0, 0.0],
            atol=2e-6,
        )
        world_angular = backend.get_sensor_view("track_angvel_w_pelvis").detach().cpu().numpy()
        np.testing.assert_allclose(world_angular[0], [1.0, 0.0, 0.0], atol=2e-6)
    finally:
        backend.close()


def test_mujoco_root_layout_resolves_a_nonfirst_free_joint(tmp_path: Path) -> None:
    pytest.importorskip(
        "unisim.backend.mujoco.backend",
        reason="unisim-core MuJoCo adapter (mjbatch build) not available",
    )
    source = tmp_path / "nonfirst-root.xml"
    source.write_text(
        """
        <mujoco>
          <worldbody>
            <body name="hinged">
              <joint name="hinge" type="hinge"/>
              <geom type="sphere" size="0.1" mass="1"/>
            </body>
            <body name="floating">
              <freejoint name="floating_free"/>
              <geom type="sphere" size="0.1" mass="1"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )
    backend = create_backend(
        "mujoco", SceneCfg(model_file=str(source)), 1, SIM_DT, base_name="floating"
    )
    try:
        layout = backend.get_root_state_layout("floating")
        assert layout.qpos_indices == tuple(range(1, 8))
        assert layout.qvel_indices == tuple(range(1, 7))
        with pytest.raises(NotImplementedError, match="hinged.*exactly one free joint"):
            backend.get_root_state_layout("hinged")
    finally:
        backend.close()


def test_drake_root_layout_is_explicitly_unsupported_without_runtime_metadata() -> None:
    from unisim.backend.drake.backend import DrakeBackend

    backend = object.__new__(DrakeBackend)
    with pytest.raises(
        NotImplementedError,
        match="DrakeBackend does not expose root-state layout.*trunk",
    ):
        backend.get_root_state_layout("trunk")


def test_terrain_spawn_consumers_do_not_probe_private_backend_capabilities() -> None:
    forbidden_names = {"terrain_origins", "terrain_surface_sampler", "sample_height"}
    offenders: list[str] = []
    for relative_path in _TERRAIN_CONSUMER_PATHS:
        owner_paths = tuple(
            root / relative_path for root in _TASK_SOURCE_ROOTS if (root / relative_path).is_file()
        )
        assert owner_paths, f"terrain consumer source not found: {relative_path}"
        for path in owner_paths:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                    continue
                if node.func.id not in {"getattr", "hasattr"}:
                    continue
                for arg in node.args[1:]:
                    if isinstance(arg, ast.Constant) and arg.value in forbidden_names:
                        offenders.append(
                            f"{path.relative_to(REPO_ROOT)}:{node.lineno}: {arg.value}"
                        )
    assert not offenders, "private terrain capability probes:\n" + "\n".join(offenders)


@pytest.mark.parametrize("backend_type", _BACKEND_PARAMS)
def test_tensor_contract_step_set_state_and_state_reads(backend_type: str) -> None:
    _require_backend(backend_type)
    if backend_type == "isaacsim":
        _skip_raw_scalar_isaacsim_fixture()

    backend = _create_backend_for_test(
        backend_type,
        SceneCfg(model_file=_G1_SCENE),
        NUM_ENVS,
        SIM_DT,
        base_name="pelvis",
        add_body_sensors=True,
    )
    backend.materialize()
    tensor_device = torch.device("cuda") if backend_type == "isaacgym" else torch.device("cpu")

    assert backend.num_envs == NUM_ENVS
    assert backend.num_actuators > 0
    assert backend.get_terrain_spawn_data() is None

    backend.step_tensor(
        torch.zeros((NUM_ENVS, backend.num_actuators), dtype=torch.float32, device=tensor_device),
        nsteps=2,
    )

    default_qpos = np.asarray(backend.get_default_qpos())
    qpos = np.broadcast_to(default_qpos, (NUM_ENVS, default_qpos.shape[0])).copy()
    target_xyz = np.array([1.0, 2.0, 0.8])
    qpos[:, :3] = target_xyz
    qvel = np.zeros((NUM_ENVS, len(backend.get_init_qvel())))
    backend.set_state_tensor(
        torch.arange(NUM_ENVS, dtype=torch.int64, device=tensor_device),
        torch.from_numpy(qpos.astype(np.float32, copy=False)).to(tensor_device),
        torch.from_numpy(qvel.astype(np.float32, copy=False)).to(tensor_device),
    )

    positions = backend.get_sensor_view("track_pos_w_pelvis").detach().cpu().numpy()
    quaternions = backend.get_sensor_view("track_quat_w_pelvis").detach().cpu().numpy()
    np.testing.assert_allclose(positions, np.tile(target_xyz, (NUM_ENVS, 1)), atol=1e-4)
    np.testing.assert_allclose(np.linalg.norm(quaternions, axis=-1), 1.0, atol=1e-5)


@pytest.mark.slow
def test_isaacgym_position_hold_is_stable() -> None:
    """Holding the keyframe pose via position targets must not destabilize.

    Regression guard for two real-runtime failures: ctrl must carry
    *position targets* (PhysX DOF_MODE_POS with the MJCF kp/kv/forcerange),
    and actor self-collision must stay disabled — the G1 collision capsules
    overlap at the default pose (MuJoCo excludes those pairs), so with
    self-collision on, the wrist/hip joints get pushed away within a few
    substeps even when the drive target equals the current position.
    """
    if not _isaacgym_runtime_available():
        pytest.skip("isaacgym requires the Python 3.8 worker runtime")

    backend = create_backend(
        "isaacgym",
        SceneCfg(model_file=_G1_SCENE),
        NUM_ENVS,
        1.0 / 150.0,
        base_name="pelvis",
    )
    backend.materialize()
    try:
        default_qpos = backend.get_default_qpos()
        default_dof_pos = backend.get_default_dof_pos()
        qpos = np.broadcast_to(default_qpos, (NUM_ENVS, default_qpos.shape[0])).copy()
        qvel = np.zeros((NUM_ENVS, len(backend.get_init_qvel())), dtype=np.float32)
        rows = torch.arange(NUM_ENVS, dtype=torch.int64, device="cuda")
        backend.set_state_tensor(
            rows,
            torch.from_numpy(qpos).to(device=rows.device),
            torch.from_numpy(qvel).to(device=rows.device),
        )
        ctrl = np.broadcast_to(default_dof_pos, (NUM_ENVS, backend.num_actuators)).copy()
        # 0.2s of sim time (ctrl_dt=0.02, 3 substeps per control step). The
        # failure signature is immediate and large (wrist dof at -0.85 rad
        # within 3 substeps when self-collision is on); normal PD gravity sag
        # stays below ~0.15 rad in this window.
        joint_qpos_ids = backend.get_joint_state_qpos_indices(backend.get_actuator_joint_names())
        for _ in range(10):
            backend.step_tensor(
                torch.from_numpy(ctrl.astype(np.float32)).to(device=rows.device),
                nsteps=3,
            )
        current_qpos = backend.get_state_views(("qpos",))["qpos"].detach().cpu().numpy()
        drift = np.abs(current_qpos[:, joint_qpos_ids] - default_dof_pos).max()
        assert drift < 0.3, f"dof drift under position hold: {drift}"
        base_z = current_qpos[:, 2]
        np.testing.assert_allclose(
            base_z,
            default_qpos[2],
            atol=0.08,
            err_msg="base height collapsed under position hold",
        )
    finally:
        backend.close()


@pytest.mark.slow
def test_isaacgym_native_camera_capture_renders_scene() -> None:
    """Real-runtime guard for the native capture path used by record playback.

    The camera sensor tracks env 0's root; after a physics step the frame must
    be a real render (correct shape, non-uniform pixels — the ground plane and
    the robot differ in color).
    """
    if not _isaacgym_runtime_available():
        pytest.skip("isaacgym requires the Python 3.8 worker runtime")

    backend = create_backend(
        "isaacgym",
        SceneCfg(model_file=_G1_SCENE),
        NUM_ENVS,
        SIM_DT,
        base_name="pelvis",
    )
    backend.materialize()
    try:
        backend.bind_renderer(headless=True, capture=True, width=320, height=240)
        backend.step_tensor(
            torch.zeros((NUM_ENVS, backend.num_actuators), dtype=torch.float32, device="cuda")
        )
        frame = backend.capture_video_frame()
        assert frame.shape == (240, 320, 3)
        assert frame.dtype == np.uint8
        assert np.unique(frame).size > 8, "camera frame looks blank"
        # A second init with the same config is a no-op and keeps capturing.
        backend.bind_renderer(headless=True, capture=True, width=320, height=240)
        assert backend.capture_video_frame().shape == (240, 320, 3)
    finally:
        backend.close()


@pytest.mark.slow
def test_genesis_native_camera_capture_renders_scene() -> None:
    """Real-runtime guard for the native capture path used by record playback.

    The offscreen camera attaches post-build; after a physics step the frame
    must be a real render (correct shape, non-uniform pixels — the ground
    plane and the robot differ in color). The backend session is deliberately
    left open: one gs.init per process and the other genesis lanes in this
    pytest session share it.
    """
    if not _genesis_runtime_available():
        pytest.skip("genesis requires the genesis-world extra and a CUDA device")

    backend = create_backend(
        "genesis",
        SceneCfg(model_file=_G1_SCENE),
        NUM_ENVS,
        SIM_DT,
        base_name="pelvis",
    )
    backend.materialize()
    backend.bind_renderer(headless=True, capture=True, width=320, height=240)
    backend.step(np.zeros((NUM_ENVS, backend.num_actuators), dtype=np.float32))
    frame = backend.capture_video_frame()
    assert frame.shape == (240, 320, 3)
    assert frame.dtype == np.uint8
    assert np.unique(frame).size > 8, "camera frame looks blank"
    # A second init with the same config is a no-op and keeps capturing.
    backend.bind_renderer(headless=True, capture=True, width=320, height=240)
    assert backend.capture_video_frame().shape == (240, 320, 3)
