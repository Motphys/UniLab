"""Host-free tests for the ``genesis`` backend using a fake Genesis runtime.

These tests run on machines WITHOUT the genesis-world extra (and without
CUDA): the fake runtime in ``genesis_fake_runtime.py`` mirrors the 1.3.3 API
surface the adapter consumes, while the adapter's real cold-path MJCF scan
(the ``mujoco`` package) runs against a tiny inline scene document.  The
genesis parameterization of ``test_backend_conformance.py`` and unisim's
Genesis adapter suite cover the broader contract and real-runtime lifecycle.
"""

from __future__ import annotations

import sys
from importlib import metadata as importlib_metadata
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch
import unisim.backend.genesis.dependencies as genesis_dependencies
import unisim.backend.genesis.materialization as genesis_materialization
from unisim.backend.base import RenderClosedError
from unisim.backend.genesis.backend import GenesisBackend
from unisim.backend.genesis.dependencies import (
    GenesisDependencies,
    GenesisDependencyError,
    genesis_dependencies_available,
    load_genesis_dependencies,
)
from unisim.backend.genesis.materialization import preserve_torch_globals
from unisim.dr.types import (
    INTERVAL_TERM_BODY_FORCE,
    INTERVAL_TERM_BODY_LINEAR_VELOCITY_DELTA,
    INTERVAL_TERM_BODY_TORQUE,
    INTERVAL_TERM_PUSH,
    IntervalRandomizationPlan,
    IntervalTermOp,
    TensorResetRandomizationPayload,
)

import unilab.base.backend_factory as backend_factory
from unilab.base.backend_factory import create_backend, env_backend_kwargs
from unilab.base.base import EnvCfg
from unilab.base.scene import SceneCfg


@pytest.fixture(autouse=True)
def _bypass_cuda_only_platform_gate(monkeypatch: pytest.MonkeyPatch):
    """Exercise the adapter below the factory's CUDA-only platform gate.

    The fail-closed gate (issue #1811 platform matrix) has dedicated coverage
    in ``tests/base/test_cuda_backend_platform_preflight.py``.  These
    fake-runtime suites intentionally run on CUDA-less hosts, so the gate is
    bypassed here to keep the adapter behavior under test reachable.
    """
    monkeypatch.setattr(
        backend_factory, "_validate_cuda_only_backend_platform", lambda *args, **kwargs: None
    )


from .genesis_fake_runtime import N_LINKS, NQ, NV, make_fake_genesis

# Must stay in sync with the model spec in genesis_fake_runtime.py.
TINY_MODEL_XML = """
<mujoco model="tiny">
  <compiler angle="radian"/>
  <option integrator="implicitfast" timestep="0.005"/>
  <worldbody>
    <geom name="floor" type="plane" size="0 0 0.05" friction="1.0"/>
    <body name="pelvis" pos="0 0 0.8">
      <freejoint name="floating_base_joint"/>
      <site name="imu_site" pos="0.05 0 -0.08"/>
      <geom name="pelvis_geom" type="capsule" size="0.05 0.1" mass="5.0" friction="0.9"/>
      <body name="thigh" pos="0 0 -0.3">
        <joint name="joint_a" type="hinge" axis="0 1 0" range="-0.5 0.5"/>
        <geom name="thigh_geom" type="capsule" size="0.04 0.1" mass="2.0" friction="0.8"/>
        <body name="shank" pos="0 0 -0.3">
          <joint name="joint_b" type="hinge" axis="0 1 0" range="-0.6 0.6"/>
          <geom name="foot_geom" type="sphere" size="0.05" mass="1.0" friction="0.6"/>
          <site name="foot_site" pos="0.02 0 -0.03"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="act_a" joint="joint_a" kp="10" kv="2" ctrlrange="-1 1"/>
    <position name="act_b" joint="joint_b" kp="20" kv="3" ctrlrange="-2 2"/>
  </actuator>
  <sensor>
    <velocimeter site="imu_site" name="base_linvel"/>
    <gyro site="imu_site" name="base_gyro"/>
    <accelerometer site="imu_site" name="base_acc"/>
    <framezaxis objtype="site" objname="imu_site" name="base_up"/>
    <framepos objtype="site" objname="foot_site" name="foot_pos"/>
    <framequat objtype="site" objname="foot_site" name="foot_quat"/>
    <contact name="foot_contact" geom1="floor" geom2="foot_geom" data="found" num="1"/>
  </sensor>
  <keyframe>
    <key name="stand" qpos="0 0 0.8 1 0 0 0 0.1 -0.2"/>
  </keyframe>
</mujoco>
"""


@pytest.fixture
def tiny_model_file(tmp_path: Path) -> str:
    path = tmp_path / "tiny.xml"
    path.write_text(TINY_MODEL_XML, encoding="utf-8")
    return str(path)


@pytest.fixture
def fake_genesis(monkeypatch: pytest.MonkeyPatch):
    if not torch.cuda.is_available():
        factories = {
            name: getattr(torch, name)
            for name in ("arange", "empty", "full", "ones", "tensor", "zeros")
        }

        def cuda_free_factory(original):
            def factory(*args, **kwargs):
                device = kwargs.get("device")
                if device is not None and torch.device(device).type == "cuda":
                    kwargs["device"] = "cpu"
                return original(*args, **kwargs)

            return factory

        for name, original in factories.items():
            monkeypatch.setattr(torch, name, cuda_free_factory(original))
        monkeypatch.setattr(
            torch.cuda,
            "current_stream",
            lambda *_args, **_kwargs: SimpleNamespace(synchronize=lambda: None),
        )
        monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *_args, **_kwargs: self)
        monkeypatch.setattr(
            GenesisBackend,
            "_tensor_lifecycle_supported",
            lambda self: (
                not self._portable_mode
                and self._genesis_backend_is_exact_cuda()
                and bool(getattr(self._gs, "use_zerocopy", False))
            ),
        )

        def tensor_device(self, requested=None):
            target = self._device if requested is None else torch.device(requested)
            if target.type != "cpu":
                raise ValueError(f"fake Genesis tensor operands use CPU, requested {target}")
            return target

        monkeypatch.setattr(GenesisBackend, "_tensor_device", tensor_device)
    fake = make_fake_genesis()
    fake.backend = fake.gpu
    fake.cuda = fake.gpu
    fake.use_zerocopy = True
    original_add_entity = fake.Scene.add_entity
    device_getters = (
        "get_qpos",
        "get_dofs_velocity",
        "get_dofs_kp",
        "get_dofs_kv",
        "get_links_pos",
        "get_links_quat",
        "get_links_vel",
        "get_links_ang",
        "get_links_net_contact_force",
    )

    def add_device_entity(scene, morph):
        entity = original_add_entity(scene, morph)
        for getter_name in device_getters:
            original_getter = getattr(entity, getter_name)

            def device_getter(*args, _getter=original_getter, **kwargs):
                return _getter(*args, **kwargs).cuda()

            setattr(entity, getter_name, device_getter)
        return entity

    fake.Scene.add_entity = add_device_entity
    deps = GenesisDependencies(genesis=fake, torch=torch, mujoco=mujoco)
    monkeypatch.setattr(genesis_dependencies, "load_genesis_dependencies", lambda: deps)
    # The backend resolves the viewer class via importlib.import_module; the
    # fake module must already sit in sys.modules (genesis is not installed
    # in this lane).
    monkeypatch.setitem(sys.modules, "genesis.vis.viewer", fake.viewer_module)
    genesis_materialization._reset_session_state_for_tests()
    yield fake
    fake.Scene.add_entity = original_add_entity
    genesis_materialization._reset_session_state_for_tests()


def _backend(model_file: str, num_envs: int = 4, **kwargs):
    backend = create_backend(
        "genesis",
        SceneCfg(model_file=model_file),
        num_envs,
        0.005,
        base_name="pelvis",
        **kwargs,
    )
    backend.materialize()
    return backend


def _stand_state(backend, rows: int) -> tuple[torch.Tensor, torch.Tensor]:
    qpos = np.tile(backend.get_keyframe_qpos("stand"), (rows, 1)).astype(np.float32)
    qvel = np.zeros((rows, backend.get_init_qvel().size), dtype=np.float32)
    return torch.from_numpy(qpos).cuda(), torch.from_numpy(qvel).cuda()


def test_dependency_boundary_errors_are_actionable(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise(_name: str) -> str:
        raise importlib_metadata.PackageNotFoundError("genesis-world")

    monkeypatch.setattr(importlib_metadata, "version", _raise)
    with pytest.raises(GenesisDependencyError, match="uv sync --extra genesis"):
        load_genesis_dependencies()
    monkeypatch.setattr(importlib_metadata, "version", lambda _name: "0.0.0")
    with pytest.raises(GenesisDependencyError, match="exact genesis-world version"):
        load_genesis_dependencies()
    monkeypatch.setattr(genesis_dependencies, "find_spec", lambda name: None)
    assert genesis_dependencies_available() is False


def test_preserve_torch_globals_restores_device_dtype_and_rng() -> None:
    torch.set_default_device("cpu")
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(1234)
    rng_before = torch.get_rng_state().clone()
    with preserve_torch_globals(torch):
        # Simulate the measured gs.init pollution (REPORT §3.5 [10]).
        torch.set_default_dtype(torch.float64)
        torch.rand(8)
    assert torch.get_default_dtype() is torch.float32
    assert str(torch.get_default_device()) == "cpu"
    assert torch.equal(torch.get_rng_state(), rng_before)


def test_session_lifecycle_and_reinit_guard(fake_genesis, tiny_model_file: str) -> None:
    first = _backend(tiny_model_file)
    second = _backend(tiny_model_file)
    assert fake_genesis.init_count == 1  # backends share the process-wide session
    first.materialize()  # idempotent: explicit call after the initial build is a no-op
    assert first._scene.build_count == 1
    first.close()
    assert fake_genesis.destroy_count == 1
    with pytest.raises(RuntimeError, match="exactly one gs.init per process"):
        _backend(tiny_model_file)
    with pytest.raises(RuntimeError, match="genesis backend is closed"):
        first.step_tensor(torch.zeros((4, 2), dtype=torch.float32, device="cuda"))
    first.materialize()  # no-op when already materialized, even after close
    second.close()
    assert fake_genesis.destroy_count == 1  # idempotent: session already destroyed


def test_cold_metadata_matches_mjcf_scan(fake_genesis, tiny_model_file: str) -> None:
    backend = _backend(tiny_model_file)
    assert backend.backend_type == "genesis"
    assert backend.num_envs == 4
    assert backend.num_actuators == 2
    assert backend.num_dof_vel == 2
    assert backend.get_actuator_names() == ("act_a", "act_b")
    assert backend.get_actuator_joint_names() == ("joint_a", "joint_b")
    np.testing.assert_array_equal(
        backend.get_actuator_ctrl_range(), np.array([[-1.0, 1.0], [-2.0, 2.0]], np.float32)
    )
    kp, kd = backend.get_actuator_gains()
    np.testing.assert_allclose(kp, [10.0, 20.0])
    np.testing.assert_allclose(kd, [2.0, 3.0])
    np.testing.assert_allclose(backend.get_joint_range(), [[-0.5, 0.5], [-0.6, 0.6]])
    np.testing.assert_allclose(backend.get_gravity(), [0.0, 0.0, -9.81])
    np.testing.assert_allclose(backend.get_dof_armature(), np.zeros(NV), atol=1e-6)
    np.testing.assert_allclose(backend.get_body_mass()[1:], [5.0, 2.0, 1.0], atol=1e-4)

    layout = backend.get_root_state_layout("pelvis")
    assert layout.qpos_indices == tuple(range(7))
    assert layout.qvel_indices == tuple(range(6))
    with pytest.raises(NotImplementedError, match="exactly one free joint"):
        backend.get_root_state_layout("thigh")
    with pytest.raises(ValueError, match="not found"):
        backend.get_root_state_layout("missing")

    np.testing.assert_allclose(backend.get_keyframe_qpos("stand")[7:], [0.1, -0.2])
    with pytest.raises(ValueError, match="Keyframe 'home' not found"):
        backend.get_keyframe_qpos("home")
    np.testing.assert_allclose(backend.get_default_qpos()[:3], [0.0, 0.0, 0.8])
    np.testing.assert_allclose(backend.get_default_dof_pos(), [0.0, 0.0], atol=1e-6)
    assert backend.get_init_qvel().shape == (NV,)
    current_dof_pos = backend.get_state_views(("qpos",))["qpos"].detach().cpu().numpy()
    np.testing.assert_allclose(current_dof_pos[0, 7:], backend.get_default_dof_pos())
    np.testing.assert_array_equal(backend.get_body_ids(["pelvis", "shank"]), [1, 3])
    np.testing.assert_array_equal(backend.get_joint_dof_indices(["joint_b"]), [7])
    np.testing.assert_array_equal(backend.get_joint_dof_pos_indices(["joint_b"]), [1])
    np.testing.assert_array_equal(backend.get_joint_state_qpos_indices(["joint_a"]), [7])

    contype, conaffinity = backend.get_geom_contact_masks()
    assert contype.dtype == np.int32 and conaffinity.dtype == np.int32
    assert contype.shape == (4,) and backend.get_scene_model_file() == tiny_model_file


def test_import_cross_check_fails_closed(fake_genesis, tiny_model_file: str) -> None:
    backend = create_backend(
        "genesis", SceneCfg(model_file=tiny_model_file), 2, 0.005, base_name="pelvis"
    )
    joints = backend._entity.joints
    backend._entity.joints = [joints[0], joints[2], joints[1]]
    with pytest.raises(RuntimeError, match="import mismatch: joint names"):
        backend.materialize()


def test_unmappable_sensor_type_fails_closed_at_scan(fake_genesis) -> None:
    model = mujoco.MjModel.from_xml_string(
        "<mujoco><worldbody><body name='b' pos='0 0 1'><freejoint/><geom size='0.1' mass='1'/>"
        "</body></worldbody><sensor><subtreelinvel name='sv' body='b'/></sensor></mujoco>"
    )
    with pytest.raises(NotImplementedError, match="cannot map MJCF sensor 'sv'"):
        genesis_materialization._scan_sensor_plans(mujoco, model)


def test_control_target_held_across_substeps(fake_genesis, tiny_model_file: str) -> None:
    backend = _backend(tiny_model_file)
    entity = backend._entity
    qpos, qvel = _stand_state(backend, 4)
    backend.set_state_tensor(torch.arange(4, dtype=torch.int64, device="cuda"), qpos, qvel)

    ctrl = np.tile(np.array([[0.3, -0.3]], dtype=np.float32), (4, 1))
    ctrl_tensor = torch.from_numpy(ctrl).cuda()
    result = backend.step_tensor(ctrl_tensor, nsteps=3)
    assert set(result["timing"]) == {
        "tensor_control_upload_ms",
        "tensor_physics_ms",
        "tensor_host_cache_refresh_ms",
    }
    assert len(entity.control_calls) == 1  # target pushed once, held across substeps
    assert entity.step_count == 3
    dof_pos = backend.get_state_views(("qpos",))["qpos"][:, 7:].detach().cpu().numpy()
    assert dof_pos.shape == (4, 2) and dof_pos.dtype == np.float32
    assert np.isfinite(dof_pos).all()
    assert np.all(dof_pos[:, 0] > 0.1)  # PD pulled joint_a toward 0.3
    assert np.all(dof_pos[:, 1] < -0.2)
    np.testing.assert_allclose(entity.control_calls[0], ctrl, atol=1e-6)

    with pytest.raises(ValueError, match="ctrl must have shape"):
        backend.step_tensor(torch.zeros((4, 3), dtype=torch.float32, device="cuda"))
    with pytest.raises(ValueError, match="nsteps must be a positive integer"):
        backend.step_tensor(ctrl_tensor, nsteps=0)


def test_pre_step_control_fails_closed_for_tensor_stepping(
    fake_genesis, tiny_model_file: str
) -> None:
    backend = _backend(tiny_model_file)
    entity = backend._entity
    calls: list[np.ndarray] = []

    def convert(owner, ctrl: np.ndarray) -> np.ndarray:
        assert owner is backend
        calls.append(ctrl.copy())
        return np.zeros_like(ctrl)

    backend.set_pre_step_control(convert)
    ctrl = np.full((4, 2), 0.25, dtype=np.float32)
    with pytest.raises(
        NotImplementedError, match="tensor stepping does not support host pre-step control"
    ):
        backend.step_tensor(torch.from_numpy(ctrl).cuda(), nsteps=4)
    assert calls == []
    assert entity.control_calls == []
    assert entity.step_count == 0


def test_set_state_subset_reset_isolated(fake_genesis, tiny_model_file: str) -> None:
    backend = _backend(tiny_model_file)
    qpos, qvel = _stand_state(backend, 4)
    qpos[:, 0] = torch.tensor([-0.3, -0.1, 0.1, 0.3], device=qpos.device)
    qvel[:, 0] = torch.tensor([0.1, 0.2, 0.3, 0.4], device=qvel.device)
    backend.set_state_tensor(torch.arange(4, dtype=torch.int64, device="cuda"), qpos, qvel)
    previous = backend.get_state_views()
    previous_pos = previous["qpos"][:, :3].detach().clone()
    previous_vel = previous["qvel"][:, :3].detach().clone()

    rows = torch.tensor([3, 1], dtype=torch.int64, device="cuda")
    reset_qpos, reset_qvel = _stand_state(backend, 2)
    result = backend.set_state_tensor(rows, reset_qpos, reset_qvel)
    timing = result["timing"]
    assert timing["tensor_reset_ms"] >= 0.0

    state = backend.get_state_views()
    current_qpos = state["qpos"].detach().cpu().numpy()
    current_qvel = state["qvel"].detach().cpu().numpy()
    rows_host = rows.detach().cpu().numpy()
    np.testing.assert_allclose(current_qpos[rows_host, 2], 0.8, atol=1e-5)
    np.testing.assert_allclose(current_qpos[rows_host, 7:], [[0.1, -0.2], [0.1, -0.2]], atol=1e-6)
    complement = np.array([0, 2], dtype=np.int32)
    np.testing.assert_allclose(current_qpos[complement, :3], previous_pos[complement].cpu().numpy())
    np.testing.assert_allclose(current_qvel[complement, :3], previous_vel[complement].cpu().numpy())
    assert np.isfinite(current_qvel).all()

    with pytest.raises(ValueError, match="env_indices must be unique"):
        backend.set_state_tensor(
            torch.tensor([1, 1], dtype=torch.int64, device="cuda"), reset_qpos, reset_qvel
        )
    with pytest.raises(ValueError, match="must be in \\[0, 4\\)"):
        backend.set_state_tensor(
            torch.tensor([9], dtype=torch.int64, device="cuda"), reset_qpos[:1], reset_qvel[:1]
        )
    with pytest.raises(ValueError, match="qpos must have shape"):
        backend.set_state_tensor(rows, reset_qpos[:, :5], reset_qvel)
    empty = backend.set_state_tensor(
        torch.empty(0, dtype=torch.int64, device="cuda"),
        torch.zeros((0, NQ), dtype=torch.float32, device="cuda"),
        torch.zeros((0, NV), dtype=torch.float32, device="cuda"),
    )
    assert all(value == 0.0 for value in empty["timing"].values())


def test_base_velocity_frame_conventions(fake_genesis, tiny_model_file: str) -> None:
    backend = _backend(tiny_model_file, num_envs=1)
    qpos, qvel = _stand_state(backend, 1)
    half = np.sqrt(0.5)
    qpos[0, 3:7] = torch.tensor(
        [half, half, 0.0, 0.0], dtype=qpos.dtype, device=qpos.device
    )  # 90 deg about x (wxyz)
    qvel[0, 0:3] = torch.tensor([1.0, 2.0, 3.0], device=qvel.device)
    qvel[0, 3:6] = torch.tensor([0.0, 0.0, 1.0], device=qvel.device)  # body-frame angular velocity
    backend.set_state_tensor(torch.tensor([0], dtype=torch.int64, device="cuda"), qpos, qvel)

    lin_vel = backend.get_sensor_view("track_linvel_w_pelvis").detach().cpu().numpy()
    np.testing.assert_allclose(lin_vel[0], [1.0, 2.0, 3.0], atol=1e-6)
    # World-frame angular velocity: R_x90 @ [0,0,1] = [0,-1,0].
    ang_vel = backend.get_sensor_view("track_angvel_w_pelvis").detach().cpu().numpy()
    np.testing.assert_allclose(ang_vel[0], [0.0, -1.0, 0.0], atol=1e-6)
    # Body-frame velocities are the world-frame values rotated by R^-1:
    # R_x90^-1 @ [1,2,3] = [1,3,-2]; the angular pair round-trips the qvel
    # body-frame columns (REPORT §3.2 [2d] semantics).
    state_qvel = backend.get_state_views(("qvel",))["qvel"].detach().cpu().numpy()
    np.testing.assert_allclose(state_qvel[0, 0:3], [1.0, 2.0, 3.0], atol=1e-6)
    np.testing.assert_allclose(state_qvel[0, 3:6], [0.0, 0.0, 1.0], atol=1e-6)
    # Gyro reports body/sensor-frame components: identity site frame recovers
    # the qvel body-frame columns.
    gyro = backend.get_sensor_view("base_gyro").detach().cpu().numpy()
    np.testing.assert_allclose(gyro[0], [0.0, 0.0, 1.0], atol=1e-6)


def test_body_pose_world_frame(fake_genesis, tiny_model_file: str) -> None:
    backend = _backend(tiny_model_file, num_envs=1)
    qpos, qvel = _stand_state(backend, 1)
    backend.set_state_tensor(torch.tensor([0], dtype=torch.int64, device="cuda"), qpos, qvel)
    # Root keeps the qpos pose; the fake shank rides at the transformed world
    # [0,0,-0.6] offset, and both frames align.
    pelvis_pos = backend.get_sensor_view("track_pos_w_pelvis").detach().cpu().numpy()
    shank_pos = backend.get_sensor_view("track_pos_w_shank").detach().cpu().numpy()
    pelvis_quat = backend.get_sensor_view("track_quat_w_pelvis").detach().cpu().numpy()
    shank_quat = backend.get_sensor_view("track_quat_w_shank").detach().cpu().numpy()
    np.testing.assert_allclose(pelvis_pos[0], [0.0, 0.0, 0.8], atol=1e-6)
    np.testing.assert_allclose(shank_pos[0], [0.0, 0.0, 0.2], atol=1e-6)
    np.testing.assert_allclose(
        [pelvis_quat[0], shank_quat[0]],
        [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
        atol=1e-6,
    )


def test_sensor_equivalents_from_link_state(fake_genesis, tiny_model_file: str) -> None:
    backend = _backend(tiny_model_file, num_envs=1)
    qpos, qvel = _stand_state(backend, 1)
    qvel[0, 0:3] = torch.tensor([1.0, 2.0, 3.0], device=qvel.device)
    qvel[0, 3:6] = torch.tensor([0.1, 0.2, 0.3], device=qvel.device)
    backend.set_state_tensor(torch.tensor([0], dtype=torch.int64, device="cuda"), qpos, qvel)

    site_pos = np.array([0.05, 0.0, -0.08])

    def sensor_view(name: str) -> np.ndarray:
        return backend.get_sensor_view(name).detach().cpu().numpy()

    expected_linvel = qvel[0, 0:3].cpu().numpy() + np.cross(qvel[0, 3:6].cpu().numpy(), site_pos)
    np.testing.assert_allclose(sensor_view("base_linvel")[0], expected_linvel, atol=1e-6)
    np.testing.assert_allclose(sensor_view("base_gyro")[0], [0.1, 0.2, 0.3], atol=1e-6)
    np.testing.assert_allclose(sensor_view("base_up")[0], [0.0, 0.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(sensor_view("foot_pos")[0], [0.02, 0.0, 0.8 - 0.63], atol=1e-6)
    np.testing.assert_allclose(sensor_view("foot_quat")[0], [1.0, 0.0, 0.0, 0.0], atol=1e-6)

    # Accelerometer reads through the registered IMUSensor equivalent.
    imu = backend._imu_sensors["base_acc"]
    imu.lin_acc = torch.full((1, 3), 7.5)
    backend.step_tensor(torch.zeros((1, 2), dtype=torch.float32, device="cuda"))
    with pytest.raises(
        NotImplementedError, match="genesis tensor sensor 'base_acc' has unsupported kind"
    ):
        sensor_view("base_acc")

    # contact(found): per-link net force threshold, both directions.
    assert sensor_view("foot_contact")[0, 0] == 0.0
    entity = backend._entity
    entity.net_contact_force = torch.zeros(1, N_LINKS, 3)
    entity.net_contact_force[0, 3, 2] = 50.0
    backend.step_tensor(torch.zeros((1, 2), dtype=torch.float32, device="cuda"))
    assert sensor_view("foot_contact")[0, 0] == 1.0
    entity.net_contact_force[0, 3, 2] = 0.5
    backend.step_tensor(torch.zeros((1, 2), dtype=torch.float32, device="cuda"))
    assert sensor_view("foot_contact")[0, 0] == 0.0


def test_bind_sensor_data_view(fake_genesis, tiny_model_file: str) -> None:
    backend = _backend(tiny_model_file, num_envs=2)
    names = ("base_gyro", "base_linvel")
    view = backend.bind_sensor_data(names)
    assert view.names == names and view.dimensions == (3, 3)
    values = view.read()
    assert values.shape == (2, 6) and np.isfinite(values).all()
    sensor_values = np.concatenate(
        [backend.get_sensor_view(name).detach().cpu().numpy() for name in names], axis=1
    )
    np.testing.assert_allclose(values, sensor_values)
    with pytest.raises(KeyError, match="missing_sensor"):
        backend.bind_sensor_data(("missing_sensor",))
    with pytest.raises(NotImplementedError, match="genesis tensor sensor view is unsupported"):
        backend.get_sensor_view("does_not_exist")


def test_dr_capabilities_and_tensor_reset_randomization_boundary(
    fake_genesis, tiny_model_file: str
) -> None:
    backend = _backend(tiny_model_file)
    caps = backend.get_dr_capabilities()
    assert caps.supported_reset_terms == frozenset()
    assert caps.supported_interval_terms == frozenset()
    assert caps.supported_device_reset_terms == {"body_mass", "kp", "kd"}
    assert caps.supported_device_interval_terms == {"body_force", "body_torque"}
    assert caps.supports_interval_term(INTERVAL_TERM_BODY_FORCE, plane="device") is True
    assert caps.supports_interval_term(INTERVAL_TERM_BODY_TORQUE, plane="device") is True
    assert caps.supports_interval_term(INTERVAL_TERM_PUSH) is False
    assert caps.supports_interval_term(INTERVAL_TERM_PUSH, plane="device") is False
    assert caps.supports_interval_term(INTERVAL_TERM_BODY_LINEAR_VELOCITY_DELTA) is False
    assert caps.supports_interval_term(
        INTERVAL_TERM_BODY_LINEAR_VELOCITY_DELTA, plane="device"
    ) is (False)

    rows = torch.tensor([1, 3], dtype=torch.int64, device="cuda")
    qpos, qvel = _stand_state(backend, 2)
    backend.set_state_tensor(rows, qpos, qvel)
    with pytest.raises(
        TypeError,
        match="genesis tensor reset randomization requires a TensorResetRandomizationPayload",
    ):
        backend.set_state_tensor(
            rows,
            qpos,
            qvel,
            randomization=SimpleNamespace(
                gravity=np.zeros((2, 3)),
                kp=np.full((2, 2), 55.0),
                kd=np.full((2, 2), 6.5),
            ),
        )
    with pytest.raises(
        NotImplementedError,
        match="genesis tensor reset randomization does not support device terms: gravity",
    ):
        backend.set_state_tensor(
            rows,
            qpos,
            qvel,
            randomization=TensorResetRandomizationPayload(
                gravity=torch.zeros((2, 3), device=rows.device)
            ),
        )


def test_interval_randomization_and_body_force(fake_genesis, tiny_model_file: str) -> None:
    backend = _backend(tiny_model_file)
    solver = backend._scene.sim.rigid_solver
    device = backend._device
    force = torch.ones((4, 2, 3), dtype=torch.float32, device=device)
    torque = torch.full_like(force, 0.25)
    backend.apply_interval_randomization(
        IntervalRandomizationPlan(
            ops=(
                IntervalTermOp(
                    INTERVAL_TERM_BODY_FORCE,
                    force,
                    body_ids=torch.as_tensor([1, 3], dtype=torch.int64, device=device),
                ),
                IntervalTermOp(
                    INTERVAL_TERM_BODY_TORQUE,
                    torque,
                    body_ids=torch.as_tensor([1, 3], dtype=torch.int64, device=device),
                ),
            )
        )
    )
    # The single-entity lane submits one solver force call per selected body;
    # torque rides the same API as a zero force plus the torque payload.
    assert len(solver.external_forces) == 4
    assert len(solver.external_torques) == 0
    for offset, force_call in enumerate(solver.external_forces):
        expected_magnitude = 1.0 if offset < 2 else 0.0
        np.testing.assert_allclose(force_call[0], np.full((4, 3), expected_magnitude))
        assert force_call[1] == [1 if offset % 2 == 0 else 3]
        assert force_call[2] == [0, 1, 2, 3]

    with pytest.raises(
        NotImplementedError,
        match="genesis runtime interval randomization does not support host-plane term: body_force",
    ):
        backend.apply_interval_randomization(
            IntervalRandomizationPlan(
                ops=(
                    IntervalTermOp(
                        INTERVAL_TERM_BODY_FORCE,
                        torch.zeros((4, 1, 3), dtype=torch.float32),
                        body_ids=torch.zeros((1,), dtype=torch.int64),
                    ),
                )
            )
        )

    with pytest.raises(NotImplementedError, match="does not support device-plane term: push"):
        backend.apply_interval_randomization(
            IntervalRandomizationPlan(
                ops=(IntervalTermOp(INTERVAL_TERM_PUSH, torch.ones((1, 3), device=device)),)
            )
        )
    with pytest.raises(
        NotImplementedError, match="does not support device-plane term: body_linear_velocity_delta"
    ):
        backend.apply_interval_randomization(
            IntervalRandomizationPlan(
                ops=(
                    IntervalTermOp(
                        INTERVAL_TERM_BODY_LINEAR_VELOCITY_DELTA,
                        torch.zeros((4, 1, 3), dtype=torch.float32, device=device),
                        body_ids=torch.as_tensor([1], dtype=torch.int64, device=device),
                    ),
                )
            )
        )
    with pytest.raises(ValueError, match="body force must have shape"):
        backend.apply_interval_randomization(
            IntervalRandomizationPlan(
                ops=(
                    IntervalTermOp(
                        INTERVAL_TERM_BODY_FORCE,
                        torch.zeros((4, 2, 3), dtype=torch.float32, device=device),
                        body_ids=torch.as_tensor([1], dtype=torch.int64, device=device),
                    ),
                )
            )
        )


def test_unsupported_contract_surface_fails_closed(fake_genesis, tiny_model_file: str) -> None:
    backend = _backend(tiny_model_file)
    for call, match in (
        (lambda: backend.get_geom_id("floor"), "geom ids"),
        (lambda: backend.get_geom_size("floor"), "geom sizes"),
        (lambda: backend.get_geom_names(), "geom names"),
        (lambda: backend.get_site_ids(["imu_site"]), "get_site_ids"),
        (lambda: backend.get_site_jacobian_w(0, np.array([6])), "get_site_jacobian_w"),
        (
            lambda: backend.bind_height_scanner(
                hfield_geom_id=0, offsets=np.zeros((1, 2)), frame_body_id=1
            ),
            "height-field",
        ),
    ):
        with pytest.raises(NotImplementedError, match=match):
            call()


def test_play_capabilities_and_plan_modes(
    fake_genesis, tiny_model_file: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    backend = _backend(tiny_model_file)
    caps = backend.get_play_capabilities()
    assert caps.supports_native_interactive_renderer is True
    assert caps.supports_native_video_capture is True
    assert caps.supports_physics_state_playback is True

    plan = backend.resolve_play_render_plan(
        play_render_mode="none", play_steps=10, output_video=tmp_path / "x.mp4"
    )
    assert plan.mode == "none" and plan.num_steps is None

    plan = backend.resolve_play_render_plan(
        play_render_mode="interactive", play_steps=None, output_video=None
    )
    assert plan.mode == "interactive" and not plan.headless and not plan.record_video

    plan = backend.resolve_play_render_plan(
        play_render_mode="record", play_steps=5, output_video=tmp_path / "x.mp4"
    )
    assert plan.mode == "record" and plan.headless and plan.record_video
    assert plan.num_steps == 5
    with pytest.raises(ValueError, match="play_steps"):
        backend.resolve_play_render_plan(
            play_render_mode="record", play_steps=None, output_video=tmp_path / "x.mp4"
        )
    with pytest.raises(ValueError, match="output video path"):
        backend.resolve_play_render_plan(play_render_mode="record", play_steps=5, output_video=None)
    with pytest.raises(ValueError, match="play render mode must be one of"):
        backend.resolve_play_render_plan(play_render_mode="bogus", play_steps=5, output_video=None)

    monkeypatch.setenv("DISPLAY", ":0")
    plan = backend.resolve_play_render_plan(
        play_render_mode="auto", play_steps=5, output_video=None
    )
    assert plan.mode == "interactive"

    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    plan = backend.resolve_play_render_plan(
        play_render_mode="auto", play_steps=5, output_video=tmp_path / "x.mp4"
    )
    assert plan.mode == "record" and plan.headless


def test_renderer_init_pinning_and_frame_capture(
    fake_genesis, tiny_model_file: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DISPLAY", ":0")
    backend = _backend(tiny_model_file)
    frame = backend.capture_video_frame()  # self-initializes headless + capture
    assert frame.shape == (720, 1280, 3) and frame.dtype == np.uint8
    camera = backend._render_camera
    assert camera is not None and camera.is_built and camera.render_count == 1
    backend.bind_renderer(headless=True, capture=True)  # same config is a no-op
    with pytest.raises(RuntimeError, match="already initialized"):
        backend.bind_renderer(headless=False)
    with pytest.raises(RuntimeError, match="already initialized"):
        backend.render()


def test_interactive_render_self_init_and_close(
    fake_genesis, tiny_model_file: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DISPLAY", ":0")
    backend = _backend(tiny_model_file)
    backend.render()  # self-initializes the interactive viewer
    viewer = backend._viewer
    assert viewer is not None and viewer.built_scene is backend._scene
    assert backend._scene.visualizer._viewer is viewer
    backend.render()
    assert backend._scene.visualizer.update_count == 2
    # #1396: the viewer receives the full Z-up pose matrix (the pos/lookat
    # branch would reuse the viewer's polluted default up vector).
    pose, pos, lookat = viewer.camera_pose
    assert pos is None and lookat is None
    assert pose.shape == (4, 4)
    assert pose[2, 0] == pytest.approx(0.0, abs=1e-12)  # camera x-axis level
    viewer.alive = False
    with pytest.raises(RenderClosedError, match="viewer window was closed"):
        backend.render()
    assert backend._viewer is None


def test_step_translates_viewer_closed(
    fake_genesis, tiny_model_file: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#1393: scene.step() updates the attached viewer; a closed viewer must
    surface as RenderClosedError from backend.step, not a raw private error."""
    monkeypatch.setenv("DISPLAY", ":0")
    backend = _backend(tiny_model_file)
    backend.render()  # attaches the interactive viewer
    ctrl = torch.zeros(
        (backend.num_envs, backend.num_actuators), dtype=torch.float32, device="cuda"
    )
    backend.step_tensor(ctrl)  # live viewer: scene.step updates it fine
    backend._viewer.alive = False
    with pytest.raises(RenderClosedError, match="viewer window was closed"):
        backend.step_tensor(ctrl)
    assert backend._viewer is None
    # A partial native submission fail-closes the tensor lane instead of
    # attempting further physics after renderer failure.
    with pytest.raises(RuntimeError, match="genesis backend state is faulted"):
        backend.step_tensor(ctrl)


def test_interactive_viewer_requires_display(
    fake_genesis, tiny_model_file: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    backend = _backend(tiny_model_file)
    with pytest.raises(RuntimeError, match="requires a reachable display"):
        backend.bind_renderer(headless=False)


def test_camera_tracking_follows_root(fake_genesis, tiny_model_file: str) -> None:
    backend = _backend(tiny_model_file, num_envs=2)
    backend.bind_renderer(
        headless=True,
        capture=True,
        camera_kwargs={"cam_tracking": True, "cam_tracking_env_idx": 1},
    )
    backend.capture_video_frame()
    camera = backend._render_camera
    assert camera.poses, "tracking must re-pose the camera on every capture"
    _, lookat = camera.poses[-1]
    root_pos = backend.get_sensor_view("track_pos_w_pelvis").detach().cpu().numpy()
    np.testing.assert_allclose(lookat, root_pos[1], atol=1e-5)


def test_run_playback_record_writes_video(
    fake_genesis, tiny_model_file: str, tmp_path: Path
) -> None:
    backend = _backend(tiny_model_file, num_envs=1)
    env = SimpleNamespace(cfg=SimpleNamespace(ctrl_dt=0.02))
    steps: list[int] = []
    output = tmp_path / "play.mp4"
    result = backend.run_playback(
        env=env,
        initialize=lambda: 0,
        step=lambda obs: (steps.append(obs), obs + 1)[1],
        num_steps=4,
        output_video=output,
        record_video=True,
        headless=True,
    )
    assert result == str(output)
    assert output.is_file() and output.stat().st_size > 0
    assert steps == [0, 1, 2, 3]
    assert backend._render_camera.render_count == 4


def test_run_playback_interactive_drives_viewer_until_closed(
    fake_genesis, tiny_model_file: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DISPLAY", ":0")
    backend = _backend(tiny_model_file, num_envs=1)
    env = SimpleNamespace(cfg=SimpleNamespace(ctrl_dt=0.02))
    steps: list[int] = []

    def step(obs: int) -> int:
        steps.append(obs)
        if len(steps) == 3 and backend._viewer is not None:
            backend._viewer.alive = False  # user closes the window mid-playback
        return obs + 1

    result = backend.run_playback(
        env=env,
        initialize=lambda: 0,
        step=step,
        num_steps=None,
        headless=False,
        record_video=False,
    )
    assert result is None  # interactive close is a clean exit, not an error
    assert steps == [0, 1, 2]
    # Steps 1-2 update the live viewer; at step 3 the viewer is already dead,
    # so update raises (real genesis semantics) before it could count.
    assert backend._scene.visualizer.update_count == 2


def test_constructor_validation_and_factory_wiring(fake_genesis, tiny_model_file: str) -> None:
    scene = SceneCfg(model_file=tiny_model_file)
    with pytest.raises(NotImplementedError, match="push_body_name"):
        create_backend("genesis", scene, 1, 0.005, push_body_name="pelvis")
    with pytest.raises(TypeError, match="does not accept backend options"):
        create_backend("genesis", scene, 1, 0.005, bogus_option=1)
    with pytest.raises(ValueError, match="solver_iterations"):
        create_backend("genesis", scene, 1, 0.005, genesis_solver_iterations=0)
    with pytest.raises(ValueError, match="genesis_integrator must be one of"):
        create_backend("genesis", scene, 1, 0.005, genesis_integrator="bogus")
    with pytest.raises(ValueError, match="position_actuator_gains"):
        create_backend("genesis", scene, 1, 0.005, position_actuator_gains={"kp": 1.0})
    with pytest.raises(ValueError, match="num_envs must be a positive integer"):
        create_backend("genesis", scene, 0, 0.005)

    backend = create_backend(
        "genesis",
        scene,
        1,
        0.005,
        genesis_integrator="implicitfast",
        genesis_constraint_solver="newton",
        genesis_friction_cone="elliptic",
        genesis_solver_iterations=25,
    )
    rigid_kwargs = backend._scene.rigid_options.kwargs
    assert rigid_kwargs["integrator"] == "implicitfast"
    assert rigid_kwargs["constraint_solver"] == "Newton"
    assert rigid_kwargs["friction_cone"] == "elliptic"
    assert rigid_kwargs["iterations"] == 25
    assert rigid_kwargs["batch_links_info"] is True
    assert rigid_kwargs["batch_dofs_info"] is True
    assert backend._scene.sim_options.dt == pytest.approx(0.005)

    assert backend._scene.build_count == 0
    # First state access lazily builds the scene (isaacgym-style idempotent
    # materialize), so Entity validators that read state before the env's
    # explicit materialize hook work (#1382).
    current_dof_pos = backend.get_state_views(("qpos",))["qpos"].detach().cpu().numpy()
    np.testing.assert_allclose(current_dof_pos[0, 7:], [0.0, 0.0])
    assert backend._scene.build_count == 1
    backend.materialize()
    assert backend._scene.build_count == 1
    # Cold-path metadata stays available before materialize (manager env
    # constructors read it ahead of the materialize hook).
    assert backend.get_default_dof_pos().shape == (2,)
    assert backend.get_keyframe_qpos("stand").shape == (NQ,)


def test_env_cfg_genesis_fields_validate_and_reach_factory() -> None:
    cfg = EnvCfg(
        genesis_device_id=1,
        genesis_integrator="implicitfast",
        genesis_constraint_solver="cg",
        genesis_friction_cone="pyramidal",
        genesis_solver_iterations=30,
    )
    cfg.validate()
    kwargs = env_backend_kwargs(cfg)
    assert (
        kwargs["genesis_device_id"],
        kwargs["genesis_integrator"],
        kwargs["genesis_constraint_solver"],
        kwargs["genesis_friction_cone"],
        kwargs["genesis_solver_iterations"],
    ) == (1, "implicitfast", "cg", "pyramidal", 30)
    with pytest.raises(ValueError, match="genesis_device_id must be a non-negative integer"):
        EnvCfg(genesis_device_id=-1).validate()
    with pytest.raises(ValueError, match="genesis_integrator must be a non-empty string"):
        EnvCfg(genesis_integrator="").validate()
    with pytest.raises(ValueError, match="genesis_solver_iterations must be a positive integer"):
        EnvCfg(genesis_solver_iterations=-1).validate()
