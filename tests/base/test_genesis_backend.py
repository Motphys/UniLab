"""Host-free tests for the ``genesis`` backend using a fake Genesis runtime.

These tests run on machines WITHOUT the genesis-world extra (and without
CUDA): the fake runtime in ``genesis_fake_runtime.py`` mirrors the 1.3.3 API
surface the adapter consumes, while the adapter's real cold-path MJCF scan
(the ``mujoco`` package) runs against a tiny inline scene document.  Real
runtime coverage lives in the slow lanes (``test_genesis_runtime.py`` and the
genesis parameterization of ``test_backend_conformance.py``).
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
from unisim.backend.genesis.dependencies import (
    GenesisDependencies,
    GenesisDependencyError,
    genesis_dependencies_available,
    load_genesis_dependencies,
)
from unisim.backend.genesis.materialization import preserve_torch_globals
from unisim.dr.types import (
    GeomSizeOverride,
    InitRandomizationPlan,
    IntervalRandomizationPlan,
    ModelVariantSpec,
    ResetRandomizationPayload,
)

from unilab.base.backend_factory import create_backend, env_backend_kwargs
from unilab.base.base import EnvCfg
from unilab.base.scene import SceneCfg

from .genesis_fake_runtime import ACTUATED_DOFS, N_LINKS, NQ, NV, make_fake_genesis

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
    <contact name="foot_contact" geom1="floor" geom2="foot_geom" data="found" num="1" reduce="mindist"/>
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
    fake = make_fake_genesis()
    deps = GenesisDependencies(genesis=fake, torch=torch, mujoco=mujoco)
    monkeypatch.setattr(genesis_dependencies, "load_genesis_dependencies", lambda: deps)
    # The backend resolves the viewer class via importlib.import_module; the
    # fake module must already sit in sys.modules (genesis is not installed
    # in this lane).
    monkeypatch.setitem(sys.modules, "genesis.vis.viewer", fake.viewer_module)
    genesis_materialization._reset_session_state_for_tests()
    yield fake
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


def _stand_state(backend, rows: int) -> tuple[np.ndarray, np.ndarray]:
    qpos = np.tile(backend.get_keyframe_qpos("stand"), (rows, 1)).astype(np.float32)
    qvel = np.zeros((rows, backend.get_init_qvel().size), dtype=np.float32)
    return qpos, qvel


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
        first.step(np.zeros((4, 2), dtype=np.float32))
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
    np.testing.assert_allclose(backend.get_dof_pos()[0], backend.get_default_dof_pos())
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
    backend.set_state(np.arange(4, dtype=np.int32), qpos, qvel)

    ctrl = np.tile(np.array([[0.3, -0.3]], dtype=np.float32), (4, 1))
    result = backend.step(ctrl, nsteps=3)
    assert set(result["timing"]) == {"physics_ms", "host_cache_refresh_ms"}
    assert len(entity.control_calls) == 1  # target pushed once, held across substeps
    assert entity.step_count == 3
    dof_pos = backend.get_dof_pos()
    assert dof_pos.shape == (4, 2) and dof_pos.dtype == np.float32
    assert np.isfinite(dof_pos).all()
    assert np.all(dof_pos[:, 0] > 0.1)  # PD pulled joint_a toward 0.3
    assert np.all(dof_pos[:, 1] < -0.2)
    np.testing.assert_allclose(entity.control_calls[0], ctrl, atol=1e-6)

    with pytest.raises(ValueError, match="ctrl must have shape"):
        backend.step(np.zeros((4, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="nsteps must be a positive integer"):
        backend.step(ctrl, nsteps=0)


def test_pre_step_control_runs_per_substep(fake_genesis, tiny_model_file: str) -> None:
    backend = _backend(tiny_model_file)
    entity = backend._entity
    calls: list[np.ndarray] = []

    def convert(owner, ctrl: np.ndarray) -> np.ndarray:
        assert owner is backend
        calls.append(ctrl.copy())
        return np.zeros_like(ctrl)

    backend.set_pre_step_control(convert)
    ctrl = np.full((4, 2), 0.25, dtype=np.float32)
    backend.step(ctrl, nsteps=4)
    assert len(calls) == 4  # conversion runs before every physics substep
    assert len(entity.control_calls) == 4
    np.testing.assert_allclose(entity.control_calls[-1], np.zeros((4, 2)), atol=1e-6)

    backend.set_pre_step_control(lambda _owner, c: np.zeros((4, 3), dtype=c.dtype))
    with pytest.raises(ValueError, match="pre-step control must return shape"):
        backend.step(ctrl)
    backend.set_pre_step_control(None)
    backend.step(ctrl)
    assert len(entity.control_calls) == 5


def test_set_state_subset_reset_isolated(fake_genesis, tiny_model_file: str) -> None:
    backend = _backend(tiny_model_file)
    qpos, qvel = _stand_state(backend, 4)
    qpos[:, 0] = np.array([-0.3, -0.1, 0.1, 0.3], dtype=np.float32)
    qvel[:, 0] = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    backend.set_state(np.arange(4, dtype=np.int32), qpos, qvel)
    previous_pos = backend.get_base_pos().copy()
    previous_vel = backend.get_base_lin_vel().copy()

    rows = np.array([3, 1], dtype=np.int32)
    reset_qpos, reset_qvel = _stand_state(backend, 2)
    result = backend.set_state(rows, reset_qpos, reset_qvel)
    timing = result["timing"]
    assert timing["set_state_reset_upload_ms"] >= 0.0
    assert timing["set_state_host_cache_refresh_ms"] >= 0.0
    assert "set_state_internal_gap_ms" in timing

    np.testing.assert_allclose(backend.get_base_pos()[rows, 2], 0.8, atol=1e-5)
    np.testing.assert_allclose(backend.get_dof_pos()[rows], [[0.1, -0.2], [0.1, -0.2]], atol=1e-6)
    complement = np.array([0, 2], dtype=np.int32)
    np.testing.assert_allclose(backend.get_base_pos()[complement], previous_pos[complement])
    np.testing.assert_allclose(backend.get_base_lin_vel()[complement], previous_vel[complement])
    assert np.isfinite(backend.get_dof_vel()).all()

    with pytest.raises(ValueError, match="duplicate rows"):
        backend.set_state(np.array([1, 1], dtype=np.int32), reset_qpos, reset_qvel)
    with pytest.raises(ValueError, match="must be in \\[0, 4\\)"):
        backend.set_state(np.array([9], dtype=np.int32), reset_qpos[:1], reset_qvel[:1])
    with pytest.raises(ValueError, match="qpos must have shape"):
        backend.set_state(rows, reset_qpos[:, :5], reset_qvel)
    empty = backend.set_state(
        np.array([], dtype=np.int32),
        np.zeros((0, NQ), dtype=np.float32),
        np.zeros((0, NV), dtype=np.float32),
    )
    assert all(value == 0.0 for value in empty["timing"].values())


def test_base_velocity_frame_conventions(fake_genesis, tiny_model_file: str) -> None:
    backend = _backend(tiny_model_file, num_envs=1)
    qpos, qvel = _stand_state(backend, 1)
    half = np.sqrt(0.5)
    qpos[0, 3:7] = [half, half, 0.0, 0.0]  # 90 deg about x (wxyz)
    qvel[0, 0:3] = [1.0, 2.0, 3.0]
    qvel[0, 3:6] = [0.0, 0.0, 1.0]  # body-frame angular velocity
    backend.set_state(np.array([0], dtype=np.int32), qpos, qvel)

    np.testing.assert_allclose(backend.get_base_lin_vel()[0], [1.0, 2.0, 3.0], atol=1e-6)
    # World-frame angular velocity: R_x90 @ [0,0,1] = [0,-1,0].
    np.testing.assert_allclose(backend.get_base_ang_vel()[0], [0.0, -1.0, 0.0], atol=1e-6)
    pelvis_id = backend.get_body_ids(["pelvis"])
    np.testing.assert_allclose(
        backend.get_body_ang_vel_w(pelvis_id)[0, 0], [0.0, -1.0, 0.0], atol=1e-6
    )
    # Body-frame velocities are the world-frame values rotated by R^-1:
    # R_x90^-1 @ [1,2,3] = [1,3,-2]; the angular pair round-trips the qvel
    # body-frame columns (REPORT §3.2 [2d] semantics).
    np.testing.assert_allclose(
        backend.get_body_lin_vel_b(pelvis_id)[0, 0], [1.0, 3.0, -2.0], atol=1e-6
    )
    np.testing.assert_allclose(
        backend.get_body_ang_vel_b(pelvis_id)[0, 0], [0.0, 0.0, 1.0], atol=1e-6
    )
    # Gyro reports body/sensor-frame components: identity site frame recovers
    # the qvel body-frame columns.
    np.testing.assert_allclose(backend.get_sensor_data("base_gyro")[0], [0.0, 0.0, 1.0], atol=1e-6)


def test_body_pose_base_frame(fake_genesis, tiny_model_file: str) -> None:
    backend = _backend(tiny_model_file, num_envs=1)
    qpos, qvel = _stand_state(backend, 1)
    backend.set_state(np.array([0], dtype=np.int32), qpos, qvel)
    ids = backend.get_body_ids(["pelvis", "shank"])
    # Root sits at its own frame origin; the fake shank rides at a fixed
    # [0,0,-0.6] offset in the pelvis frame, and both frames align.
    np.testing.assert_allclose(backend.get_body_pos_b(ids)[0, 0], [0.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(backend.get_body_pos_b(ids)[0, 1], [0.0, 0.0, -0.6], atol=1e-6)
    np.testing.assert_allclose(
        backend.get_body_quat_b(ids)[0], [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], atol=1e-6
    )


def test_sensor_equivalents_from_link_state(fake_genesis, tiny_model_file: str) -> None:
    backend = _backend(tiny_model_file, num_envs=1)
    qpos, qvel = _stand_state(backend, 1)
    qvel[0, 0:3] = [1.0, 2.0, 3.0]
    qvel[0, 3:6] = [0.1, 0.2, 0.3]
    backend.set_state(np.array([0], dtype=np.int32), qpos, qvel)

    site_pos = np.array([0.05, 0.0, -0.08])
    expected_linvel = qvel[0, 0:3] + np.cross(qvel[0, 3:6], site_pos)
    np.testing.assert_allclose(
        backend.get_sensor_data("base_linvel")[0], expected_linvel, atol=1e-6
    )
    np.testing.assert_allclose(backend.get_sensor_data("base_gyro")[0], [0.1, 0.2, 0.3], atol=1e-6)
    np.testing.assert_allclose(backend.get_sensor_data("base_up")[0], [0.0, 0.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(
        backend.get_sensor_data("foot_pos")[0], [0.02, 0.0, 0.8 - 0.63], atol=1e-6
    )
    np.testing.assert_allclose(
        backend.get_sensor_data("foot_quat")[0], [1.0, 0.0, 0.0, 0.0], atol=1e-6
    )

    # Accelerometer reads through the registered IMUSensor equivalent.
    imu = backend._imu_sensors["base_acc"]
    imu.lin_acc = torch.full((1, 3), 7.5)
    backend.step(np.zeros((1, 2), dtype=np.float32))
    np.testing.assert_allclose(backend.get_sensor_data("base_acc")[0], [7.5, 7.5, 7.5])

    # contact(found): per-link net force threshold, both directions.
    assert backend.get_sensor_data("foot_contact")[0, 0] == 0.0
    entity = backend._entity
    entity.net_contact_force = torch.zeros(1, N_LINKS, 3)
    entity.net_contact_force[0, 3, 2] = 50.0
    backend.step(np.zeros((1, 2), dtype=np.float32))
    assert backend.get_sensor_data("foot_contact")[0, 0] == 1.0
    entity.net_contact_force[0, 3, 2] = 0.5
    backend.step(np.zeros((1, 2), dtype=np.float32))
    assert backend.get_sensor_data("foot_contact")[0, 0] == 0.0


def test_bind_sensor_data_view(fake_genesis, tiny_model_file: str) -> None:
    backend = _backend(tiny_model_file, num_envs=2)
    names = ("base_gyro", "base_linvel")
    view = backend.bind_sensor_data(names)
    assert view.names == names and view.dimensions == (3, 3)
    values = view.read()
    assert values.shape == (2, 6) and np.isfinite(values).all()
    np.testing.assert_allclose(values, backend.get_sensor_data_batch(names))
    with pytest.raises(ValueError, match="missing_sensor"):
        backend.bind_sensor_data(("missing_sensor",))
    with pytest.raises(ValueError, match="Sensor 'does_not_exist' not found"):
        backend.get_sensor_data("does_not_exist")


def test_dr_capabilities_and_reset_randomization(fake_genesis, tiny_model_file: str) -> None:
    backend = _backend(tiny_model_file)
    caps = backend.get_dr_capabilities()
    assert caps.supported_reset_terms == {"body_mass", "base_mass_delta", "kp", "kd"}
    assert caps.supports_interval_body_force is True
    assert caps.supports_interval_push is False
    assert caps.supports_interval_body_velocity_delta is False

    entity = backend._entity
    rows = np.array([1, 3], dtype=np.int32)
    qpos, qvel = _stand_state(backend, 2)
    body_mass = np.tile(np.array([[0.0, 6.0, 2.5, 1.5]], dtype=np.float32), (2, 1))
    payload = ResetRandomizationPayload(
        body_mass=body_mass,
        kp=np.full((2, 2), 55.0),
        kd=np.full((2, 2), 6.5),
    )
    backend.set_state(rows, qpos, qvel, randomization=payload)
    np.testing.assert_allclose(entity.link_mass[1].numpy(), [0.0, 6.0, 2.5, 1.5])
    np.testing.assert_allclose(entity.link_mass[0, 1].numpy(), 5.0)  # untouched row
    np.testing.assert_allclose(entity.dof_kp.numpy()[rows][:, ACTUATED_DOFS], 55.0)
    np.testing.assert_allclose(entity.dof_kv.numpy()[rows][:, ACTUATED_DOFS], 6.5)

    delta_payload = ResetRandomizationPayload(base_mass_delta=np.array([0.25, -0.5]))
    backend.set_state(rows, qpos, qvel, randomization=delta_payload)
    # Delta composes onto the scanned nominal base mass (5.0), matching the
    # MuJoCo backend's base-table semantics.
    np.testing.assert_allclose(entity.link_mass.numpy()[rows, 1], [5.25, 4.5])

    with pytest.raises(NotImplementedError, match="gravity"):
        backend.set_state(
            rows, qpos, qvel, randomization=ResetRandomizationPayload(gravity=np.zeros((2, 3)))
        )
    with pytest.raises(ValueError, match="kp must have shape"):
        backend.set_state(
            rows, qpos, qvel, randomization=ResetRandomizationPayload(kp=np.zeros((2, 5)))
        )


def test_interval_randomization_and_body_force(fake_genesis, tiny_model_file: str) -> None:
    backend = _backend(tiny_model_file)
    solver = backend._scene.sim.rigid_solver
    force = np.ones((4, 2, 3), dtype=np.float32)
    backend.apply_interval_randomization(
        IntervalRandomizationPlan(body_ids=np.array([1, 3], dtype=np.int32), body_force=force)
    )
    assert len(solver.external_forces) == 2
    np.testing.assert_allclose(solver.external_forces[0][0], np.ones((4, 3)))
    assert solver.external_forces[0][1] == [1]
    assert solver.external_forces[1][1] == [3]

    with pytest.raises(NotImplementedError, match="interval push"):
        backend.apply_interval_randomization(IntervalRandomizationPlan(push_perturbation_limit=5.0))
    with pytest.raises(NotImplementedError, match="body-velocity"):
        backend.apply_interval_randomization(
            IntervalRandomizationPlan(
                body_ids=np.array([1], dtype=np.int32),
                body_linear_velocity_delta=np.zeros((4, 1, 3), dtype=np.float32),
            )
        )
    with pytest.raises(ValueError, match="requires body_ids"):
        backend.apply_interval_randomization(
            IntervalRandomizationPlan(body_force=np.zeros((4, 1, 3), dtype=np.float32))
        )
    with pytest.raises(NotImplementedError, match="init-lifecycle randomization"):
        backend.apply_init_randomization(
            InitRandomizationPlan(
                model_assignments=np.zeros(4, dtype=np.int32),
                model_variants=(
                    ModelVariantSpec(geom_size_overrides=(GeomSizeOverride("foot_geom", (0.1,)),)),
                ),
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
            lambda: backend.create_hfield_scanner(
                hfield_geom_id=0, offsets=np.zeros((1, 2)), frame_body_id=1
            ),
            "height-field",
        ),
        (lambda: backend.get_physics_state(), "physics-state playback"),
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
    assert caps.supports_physics_state_playback is False

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
    backend.init_renderer(headless=True, capture=True)  # same config is a no-op
    with pytest.raises(RuntimeError, match="already initialized"):
        backend.init_renderer(headless=False)
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
    ctrl = np.zeros((backend.num_envs, backend.num_actuators), dtype=np.float32)
    backend.step(ctrl)  # live viewer: scene.step updates it fine
    backend._viewer.alive = False
    with pytest.raises(RenderClosedError, match="viewer window was closed"):
        backend.step(ctrl)
    assert backend._viewer is None
    # Dead viewer is detached: physics keeps working without a renderer.
    backend.step(ctrl)


def test_interactive_viewer_requires_display(
    fake_genesis, tiny_model_file: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    backend = _backend(tiny_model_file)
    with pytest.raises(RuntimeError, match="requires a reachable display"):
        backend.init_renderer(headless=False)


def test_camera_tracking_follows_root(fake_genesis, tiny_model_file: str) -> None:
    backend = _backend(tiny_model_file, num_envs=2)
    backend.init_renderer(
        headless=True,
        capture=True,
        camera_kwargs={"cam_tracking": True, "cam_tracking_env_idx": 1},
    )
    backend.capture_video_frame()
    camera = backend._render_camera
    assert camera.poses, "tracking must re-pose the camera on every capture"
    _, lookat = camera.poses[-1]
    np.testing.assert_allclose(lookat, backend.get_base_pos()[1], atol=1e-5)


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
    np.testing.assert_allclose(backend.get_dof_pos()[0], [0.0, 0.0])
    assert backend._scene.build_count == 1
    backend.materialize()
    assert backend._scene.build_count == 1
    # Cold-path metadata stays available before materialize (manager env
    # constructors read it ahead of the materialize hook).
    assert backend.get_default_dof_pos().shape == (2,)
    assert backend.get_keyframe_qpos("stand").shape == (NQ,)


def test_env_cfg_genesis_fields_validate_and_reach_factory() -> None:
    cfg = EnvCfg(
        genesis_integrator="implicitfast",
        genesis_constraint_solver="cg",
        genesis_friction_cone="pyramidal",
        genesis_solver_iterations=30,
    )
    cfg.validate()
    kwargs = env_backend_kwargs(cfg)
    assert (
        kwargs["genesis_integrator"],
        kwargs["genesis_constraint_solver"],
        kwargs["genesis_friction_cone"],
        kwargs["genesis_solver_iterations"],
    ) == ("implicitfast", "cg", "pyramidal", 30)
    with pytest.raises(ValueError, match="genesis_integrator must be a non-empty string"):
        EnvCfg(genesis_integrator="").validate()
    with pytest.raises(ValueError, match="genesis_solver_iterations must be a positive integer"):
        EnvCfg(genesis_solver_iterations=-1).validate()
