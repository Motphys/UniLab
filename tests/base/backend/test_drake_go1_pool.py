from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pydrake", reason="Drake is not installed")

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base.backend.drake.backend import DrakeBackend
from unilab.base.backend.drake.pool import DrakeEnvPool
from unilab.base.scene import SceneCfg

MODEL_FILE = str(ASSETS_ROOT_PATH / "robots" / "go1" / "scene_flat_drake.xml")
NUM_ENVS = 2
SIM_DT = 0.01


@pytest.fixture
def backend() -> DrakeBackend:
    return DrakeBackend(
        SceneCfg(model_file=MODEL_FILE),
        num_envs=NUM_ENVS,
        sim_dt=SIM_DT,
        base_name="trunk",
    )


def _home_batch(backend: DrakeBackend) -> tuple[np.ndarray, np.ndarray]:
    qpos = np.stack([backend.get_keyframe_qpos("home") for _ in range(NUM_ENVS)])
    qvel = np.stack([backend.get_init_qvel() for _ in range(NUM_ENVS)])
    return qpos, qvel


def test_go1_backend_accepts_batched_state_and_queries(backend: DrakeBackend) -> None:
    qpos, qvel = _home_batch(backend)
    qpos[1, 0] = 1.25

    backend.set_state(np.arange(NUM_ENVS, dtype=np.int32), qpos, qvel)

    base_pos = backend.get_base_pos()
    assert base_pos.shape == (NUM_ENVS, 3)
    np.testing.assert_allclose(base_pos[:, 0], [0.0, 1.25])
    assert backend.get_dof_pos().shape == (NUM_ENVS, backend.num_actuators)
    assert backend.get_dof_vel().shape == (NUM_ENVS, backend.num_dof_vel)
    assert backend.get_sensor_data("upvector").shape == (NUM_ENVS, 3)
    assert backend.get_sensor_data("FL_pos").shape == (NUM_ENVS, 3)
    assert backend.get_sensor_data("feet_pos").shape == (NUM_ENVS, 4, 3)
    assert backend.get_sensor_data("feet_contact_force").shape == (NUM_ENVS, 4, 3)
    np.testing.assert_allclose(
        backend.get_sensor_data("FL_foot_contact"),
        np.zeros((NUM_ENVS, 3), dtype=np.float64),
    )


def test_go1_set_state_only_affects_requested_env(backend: DrakeBackend) -> None:
    qpos, qvel = _home_batch(backend)
    backend.set_state(np.arange(NUM_ENVS, dtype=np.int32), qpos, qvel)
    backend.step(backend.get_dof_pos(), nsteps=5)
    env0_before = backend.get_base_pos()[0].copy()
    env0_time_before = backend.get_physics_state()[0, 0]

    qpos_one = backend.get_keyframe_qpos("home").reshape(1, -1)
    qvel_one = backend.get_init_qvel().reshape(1, -1)
    qpos_one[0, 0] = 2.0
    backend.set_state(np.array([1], dtype=np.int32), qpos_one, qvel_one)

    base_pos = backend.get_base_pos()
    np.testing.assert_allclose(base_pos[0], env0_before)
    np.testing.assert_allclose(base_pos[1, 0], 2.0)
    np.testing.assert_allclose(backend.get_physics_state()[0, 0], env0_time_before)
    np.testing.assert_allclose(backend.get_physics_state()[1, 0], 0.0)


def test_go1_step_advances_each_context(backend: DrakeBackend) -> None:
    qpos, qvel = _home_batch(backend)
    qpos[1, 0] = 1.0
    backend.set_state(np.arange(NUM_ENVS, dtype=np.int32), qpos, qvel)
    ctrl = backend.get_dof_pos()

    backend.step(ctrl, nsteps=2)

    times = backend.get_physics_state()[:, 0]
    np.testing.assert_allclose(times, [2 * SIM_DT, 2 * SIM_DT])
    assert np.all(np.isfinite(backend.get_base_pos()))
    assert np.all(np.isfinite(backend.get_sensor_data("gyro")))


def test_go1_step_rejects_wrong_batch_shape(backend: DrakeBackend) -> None:
    with pytest.raises(ValueError, match="expected ctrl shape"):
        backend.step(np.zeros((1, backend.num_actuators), dtype=np.float64))


def test_go1_renderer_rebuild_preserves_all_env_states(backend: DrakeBackend) -> None:
    qpos, qvel = _home_batch(backend)
    qpos[1, 0] = 3.0
    backend.set_state(np.arange(NUM_ENVS, dtype=np.int32), qpos, qvel)

    backend.init_renderer(capture=True, width=64, height=48, camera_kwargs={})

    np.testing.assert_allclose(backend.get_base_pos()[:, 0], [0.0, 3.0])


def test_go1_pool_contract_returns_state_and_sensor_buffers(backend: DrakeBackend) -> None:
    pool = backend._pool
    assert isinstance(pool, DrakeEnvPool)

    state0 = backend.get_physics_state()
    result = pool.step(state0, nstep=2, control=backend.get_dof_pos())

    assert result.state.shape == state0.shape
    np.testing.assert_allclose(result.state[:, 0], [2 * SIM_DT, 2 * SIM_DT])
    assert result.sensor["gyro"].shape == (NUM_ENVS, 3)
    assert result.sensor["local_linvel"].shape == (NUM_ENVS, 3)
    assert result.sensor["upvector"].shape == (NUM_ENVS, 3)
    assert result.sensor["dof_pos"].shape == (NUM_ENVS, backend.num_actuators)
    assert result.sensor["dof_vel"].shape == (NUM_ENVS, backend.num_dof_vel)
    assert result.sensor["feet_pos"].shape == (NUM_ENVS, 4, 3)
    assert result.sensor["feet_contact_force"].shape == (NUM_ENVS, 4, 3)


def test_go1_accessors_read_cached_state_and_sensor_packet(backend: DrakeBackend) -> None:
    cached_base = np.full((NUM_ENVS, 3), 4.25, dtype=np.float64)
    cached_gyro = np.full((NUM_ENVS, 3), -0.5, dtype=np.float64)
    cached_dof = np.full((NUM_ENVS, backend.num_actuators), 0.75, dtype=np.float64)
    backend._sensor_packet["base_pos"] = cached_base
    backend._sensor_packet["gyro"] = cached_gyro
    backend._sensor_packet["dof_pos"] = cached_dof
    backend._physics_state[:, 1:4] = cached_base

    np.testing.assert_allclose(backend.get_base_pos(), cached_base)
    np.testing.assert_allclose(backend.get_sensor_data("gyro"), cached_gyro)
    np.testing.assert_allclose(backend.get_dof_pos(), cached_dof)
    np.testing.assert_allclose(backend.get_physics_state()[:, 1:4], cached_base)


def test_go1_physics_state_matches_replay_layout(backend: DrakeBackend) -> None:
    state = backend.get_physics_state()
    nq = backend.model.num_positions()
    nv = backend.model.num_velocities()
    qvel_start = 1 + nq

    assert state.shape == (NUM_ENVS, 1 + nq + nv)
    np.testing.assert_allclose(state[:, 1:4], backend.get_base_pos())
    np.testing.assert_allclose(state[:, 1 + 7 : 1 + nq], backend.get_dof_pos())
    np.testing.assert_allclose(state[:, qvel_start + 6 :], backend.get_dof_vel())
