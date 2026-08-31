"""Real-runtime slow lane for the ``genesis`` backend (genesis-world 1.3.3).

These tests require the ``genesis`` extra and a CUDA device; the normal lane
deselects them.  They cover the #1378 acceptance smoke (g1 scene_flat.xml,
n_envs=8, keyframe reset, 10 control steps, explicit cleanup) and re-verify
on torch 2.8 the sensor combination that crashed under torch 2.7 in the
feasibility probe (REPORT #1372 §3.4/§8: IMUSensor + link net contact force
reads in one scene).
"""

from __future__ import annotations

import numpy as np
import pytest

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base.backend import create_backend
from unilab.base.scene import SceneCfg

pytestmark = pytest.mark.slow

_G1_SCENE = str(ASSETS_ROOT_PATH / "robots" / "g1" / "scene_flat.xml")
_NUM_ENVS = 8
_SIM_DT = 1.0 / 150.0


def _genesis_runtime_available() -> bool:
    from unilab.base.backend.genesis.dependencies import genesis_dependencies_available

    if not genesis_dependencies_available():
        return False
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _make_backend():
    backend = create_backend(
        "genesis",
        SceneCfg(model_file=_G1_SCENE),
        _NUM_ENVS,
        _SIM_DT,
        base_name="pelvis",
        genesis_integrator="implicitfast",
    )
    backend.materialize()
    return backend


@pytest.fixture(scope="module")
def backend():
    if not _genesis_runtime_available():
        pytest.skip("genesis requires the genesis-world extra and a CUDA device")
    instance = _make_backend()
    yield instance
    instance.close()


def _stand_state(instance, rows: int) -> tuple[np.ndarray, np.ndarray]:
    qpos = np.tile(instance.get_keyframe_qpos("stand"), (rows, 1)).astype(np.float32)
    qvel = np.zeros((rows, instance.get_init_qvel().size), dtype=np.float32)
    return qpos, qvel


def test_g1_smoke_reset_step_and_cleanup(backend) -> None:
    """Acceptance smoke for #1378: reset, 10 control steps, finite reads."""
    assert backend.num_envs == _NUM_ENVS
    assert backend.num_actuators == 29
    assert backend.num_dof_vel == 29

    qpos, qvel = _stand_state(backend, _NUM_ENVS)
    rows = np.arange(_NUM_ENVS, dtype=np.int32)
    backend.set_state(rows, qpos, qvel)
    np.testing.assert_allclose(backend.get_dof_pos(), qpos[:, -backend.num_actuators :], atol=1e-5)

    # DR round-trip on the measured per-env setters (batch_*_info build flags).
    from unilab.dr.types import ResetRandomizationPayload

    kp, kd = backend.get_actuator_gains()
    payload = ResetRandomizationPayload(kp=(kp * 1.05)[None].repeat(2, axis=0))
    backend.set_state(rows[:2], qpos[:2], qvel[:2], randomization=payload)
    np.testing.assert_allclose(
        backend._entity.get_dofs_kp().cpu().numpy()[:2, 6:],
        np.tile(kp * 1.05, (2, 1)),
        rtol=1e-4,
    )
    backend.set_state(
        rows[:2],
        qpos[:2],
        qvel[:2],
        randomization=ResetRandomizationPayload(kp=kp[None].repeat(2, axis=0)),
    )

    ctrl = np.broadcast_to(
        qpos[0, -backend.num_actuators :], (_NUM_ENVS, backend.num_actuators)
    ).copy()
    keyframe_dof = qpos[0, -backend.num_actuators :]
    for _ in range(10):
        result = backend.step(ctrl, nsteps=3)
    assert set(result["timing"]) == {"physics_ms", "host_cache_refresh_ms"}

    base_pos = backend.get_base_pos()
    base_quat = backend.get_base_quat()
    dof_pos = backend.get_dof_pos()
    assert base_pos.shape == (_NUM_ENVS, 3)
    assert np.isfinite(base_pos).all()
    assert np.isfinite(dof_pos).all()
    np.testing.assert_allclose(np.linalg.norm(base_quat, axis=-1), 1.0, atol=1e-4)
    # Position hold at the stand keyframe stays upright over 0.2 s.
    np.testing.assert_allclose(base_pos[:, 2], qpos[0, 2], atol=0.08)
    drift = np.abs(dof_pos - keyframe_dof).max()
    assert drift < 0.3, f"dof drift under position hold: {drift}"

    # Sensor reads, including the IMUSensor + net-contact-force combination
    # that crashed under torch 2.7 in the #1372 probe (torch 2.8 re-check).
    sensor_names = (
        "pelvis_gyro",
        "pelvis_local_linvel",
        "pelvis_acceleration",
        "torso_gyro",
        "left_foot_pos",
        "left_foot_quat",
        "left_foot_contact_0",
    )
    batch = backend.get_sensor_data_batch(sensor_names)
    assert batch.shape == (_NUM_ENVS, 3 + 3 + 3 + 3 + 3 + 4 + 1)
    assert np.isfinite(batch).all()
    foot_force = np.linalg.norm(
        backend._entity.get_links_net_contact_force().cpu().numpy()[:, 1:], axis=-1
    )
    assert (foot_force > 1.0).any(), "standing G1 should register foot contact force"
    assert (backend.get_sensor_data("left_foot_contact_0") > 0.5).all()
    view = backend.bind_sensor_data(("pelvis_gyro", "pelvis_local_linvel"))
    assert view.read().shape == (_NUM_ENVS, 6)

    print(
        "[genesis smoke] n_envs=8 reset+10 steps OK; "
        f"base_z={base_pos[:, 2].mean():.4f} (target {qpos[0, 2]:.3f}), "
        f"max_dof_drift={drift:.4f}, foot_force_max={foot_force.max():.1f}N"
    )


def test_body_frame_kinematics_matches_mujoco_backend(backend) -> None:
    """#1382: body-frame velocity/pose math matches MuJoCo on identical state."""
    num_envs = 2
    mujoco_backend = create_backend(
        "mujoco",
        SceneCfg(model_file=_G1_SCENE),
        num_envs,
        _SIM_DT,
        base_name="pelvis",
        add_body_sensors=True,
    )
    mujoco_backend.materialize()

    rng = np.random.default_rng(0)
    qpos = np.tile(backend.get_keyframe_qpos("stand"), (num_envs, 1)).astype(np.float32)
    qvel = rng.uniform(-0.5, 0.5, size=(num_envs, backend.get_init_qvel().size)).astype(np.float32)
    rows = np.arange(num_envs, dtype=np.int32)
    backend.set_state(rows, qpos, qvel)
    mujoco_backend.set_state(rows, qpos, qvel)

    body_names = ["pelvis", "left_hip_pitch_link", "left_knee_link"]
    gs_ids = backend.get_body_ids(body_names)
    mj_ids = mujoco_backend.get_body_ids(body_names)
    # Same document order across backends (materialize-time cross-check).
    np.testing.assert_array_equal(gs_ids, mj_ids)

    atol = 2e-3
    for getter in (
        "get_body_lin_vel_b",
        "get_body_ang_vel_b",
        "get_body_pos_b",
        "get_body_quat_b",
    ):
        actual = getattr(backend, getter)(gs_ids)[:num_envs]
        expected = getattr(mujoco_backend, getter)(mj_ids)
        np.testing.assert_allclose(actual, expected, atol=atol, err_msg=getter)


def test_reinit_after_destroy_fails_closed(backend) -> None:
    """Real-runtime lifecycle guard: one gs.init per process.

    The reset hook restores the Python-side session flag afterwards so
    later real-runtime tests in the same pytest process can re-init (the
    guard's RSS caveat is acceptable in tests; production still inits once).
    """
    from unilab.base.backend.genesis.materialization import _reset_session_state_for_tests

    backend.close()
    try:
        with pytest.raises(RuntimeError, match="exactly one gs.init per process"):
            _make_backend()
    finally:
        _reset_session_state_for_tests()
