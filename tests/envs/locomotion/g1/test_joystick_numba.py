from __future__ import annotations

import numpy as np
import pytest

from unilab.envs.locomotion.g1.joystick_numba import (
    NUMBA_AVAILABLE,
    G1WalkNumbaAccelerator,
    unsupported_terms,
)


def test_unsupported_terms_only_reports_active_unknown_terms():
    assert unsupported_terms({"tracking_lin_vel": 1.0, "custom": 0.0}) == frozenset()
    assert unsupported_terms({"tracking_lin_vel": 1.0, "custom": 1.0}) == frozenset({"custom"})


def test_numba_accelerator_requires_numba_when_declared(monkeypatch):
    import unilab.envs.locomotion.g1.joystick_numba as joystick_numba

    monkeypatch.setattr(joystick_numba, "NUMBA_AVAILABLE", False)
    with pytest.raises(RuntimeError, match="requires numba"):
        G1WalkNumbaAccelerator.from_env(_Env(8, 29))


class _Backend:
    def __init__(self, n: int):
        self.base = np.full((n, 3), [0.0, 0.0, 0.754], dtype=np.float32)
        self.sensors = {
            "left_foot_pos": np.column_stack(
                [np.full(n, -0.1), np.zeros(n), np.full(n, 0.02)]
            ).astype(np.float32),
            "right_foot_pos": np.column_stack(
                [np.full(n, 0.1), np.zeros(n), np.full(n, 0.02)]
            ).astype(np.float32),
            "left_foot_quat": np.tile(np.array([1.0, 0.01, 0.02, 0.0], dtype=np.float32), (n, 1)),
            "right_foot_quat": np.tile(np.array([1.0, 0.02, 0.01, 0.0], dtype=np.float32), (n, 1)),
        }
        for side in ("left", "right"):
            for idx in range(4):
                self.sensors[f"{side}_foot_contact_{idx}"] = np.ones((n,), dtype=np.float32)

    def get_base_pos(self):
        return self.base

    def get_sensor_data(self, name: str):
        return self.sensors[name]


class _Cfg:
    ctrl_dt = 0.02


class _RewardCfg:
    tracking_sigma = 0.25
    base_height_target = 0.754
    min_base_height = 0.3
    max_tilt_deg = 65.0
    feet_phase_swing_height = 0.09
    feet_phase_tracking_sigma = 0.04
    min_forward_speed_for_gait_reward = 0.0
    close_feet_threshold = 0.15


class _Env:
    def __init__(self, n: int, n_action: int):
        self.num_envs = n
        self._num_action = n_action
        self._cfg = _Cfg()
        self._reward_cfg = _RewardCfg()
        self.default_angles = np.zeros((n_action,), dtype=np.float32)
        self._pose_weights = np.ones((n_action,), dtype=np.float32)
        self._upper_body_pose_weights = np.ones((n_action,), dtype=np.float32)
        self._backend = _Backend(n)


@pytest.mark.skipif(not NUMBA_AVAILABLE, reason="numba is optional")
def test_g1_walk_numba_unsupported_active_term_raises():
    n = 8
    n_action = 29
    env = _Env(n, n_action)
    accel = G1WalkNumbaAccelerator.from_env(env, num_threads=1)
    info = {
        "steps": np.zeros((n,), dtype=np.uint32),
        "commands": np.zeros((n, 3), dtype=np.float32),
    }

    with pytest.raises(ValueError, match="does not support active reward terms"):
        accel.compute(
            env=env,
            info=info,
            linvel=np.zeros((n, 3), dtype=np.float32),
            gyro=np.zeros((n, 3), dtype=np.float32),
            gravity=np.tile(np.array([0.0, 0.0, 1.0], dtype=np.float32), (n, 1)),
            dof_pos=np.zeros((n, n_action), dtype=np.float32),
            dof_vel=np.zeros((n, n_action), dtype=np.float32),
            scales={"tracking_lin_vel": 1.0, "custom": 1.0},
            enable_log=False,
        )


@pytest.mark.skipif(not NUMBA_AVAILABLE, reason="numba is optional")
def test_g1_walk_numba_basic_reward_parity():
    n = 512
    n_action = 29
    rng = np.random.default_rng(0)
    env = _Env(n, n_action)
    info = {
        "steps": np.zeros((n,), dtype=np.uint32),
        "commands": rng.normal(size=(n, 3)).astype(np.float32),
        "current_actions": rng.normal(size=(n, n_action)).astype(np.float32),
        "last_actions": rng.normal(size=(n, n_action)).astype(np.float32),
        "gait_phase": rng.uniform(0.0, 2 * np.pi, size=(n, 2)).astype(np.float32),
    }
    linvel = rng.normal(size=(n, 3)).astype(np.float32)
    gyro = rng.normal(size=(n, 3)).astype(np.float32)
    gravity = np.tile(np.array([0.01, -0.02, 0.99], dtype=np.float32), (n, 1))
    dof_pos = rng.normal(scale=0.01, size=(n, n_action)).astype(np.float32)
    dof_vel = rng.normal(size=(n, n_action)).astype(np.float32)
    scales = {
        "tracking_lin_vel": 2.0,
        "tracking_ang_vel": 1.5,
        "penalty_orientation": -10.0,
        "penalty_action_rate": -4.0,
        "pose": -0.5,
        "alive": 10.0,
    }

    accel = G1WalkNumbaAccelerator.from_env(env, num_threads=2)
    assert accel is not None
    out = accel.compute(
        env=env,
        info=info,
        linvel=linvel,
        gyro=gyro,
        gravity=gravity,
        dof_pos=dof_pos,
        dof_vel=dof_vel,
        scales=scales,
        enable_log=True,
    )
    assert out is not None

    reward = np.zeros((n,), dtype=np.float32)
    err = np.sum((info["commands"][:, :2] - linvel[:, :2]) ** 2, axis=1)
    reward += np.exp(-err / _RewardCfg.tracking_sigma) * scales["tracking_lin_vel"]
    err = (info["commands"][:, 2] - gyro[:, 2]) ** 2
    reward += np.exp(-err / _RewardCfg.tracking_sigma) * scales["tracking_ang_vel"]
    reward += (gravity[:, 0] ** 2 + gravity[:, 1] ** 2) * scales["penalty_orientation"]
    reward += (
        np.sum((info["current_actions"] - info["last_actions"]) ** 2, axis=1)
        * scales["penalty_action_rate"]
    )
    reward += np.sum((dof_pos - env.default_angles) ** 2, axis=1) * scales["pose"]
    reward += np.ones((n,), dtype=np.float32) * scales["alive"]
    reward *= _Cfg.ctrl_dt

    terminated = (
        np.arccos(np.clip(gravity[:, 2], -1.0, 1.0)) > np.deg2rad(_RewardCfg.max_tilt_deg)
    ) | (env._backend.get_base_pos()[:, 2] < _RewardCfg.min_base_height)

    np.testing.assert_allclose(out.reward, reward, rtol=2e-5, atol=2e-5)
    np.testing.assert_array_equal(out.terminated, terminated)
    assert set(out.log) == {f"reward/{name}" for name in scales}


@pytest.mark.skipif(not NUMBA_AVAILABLE, reason="numba is optional")
def test_g1_walk_numba_term_py_funcs_match_numpy_math():
    from unilab.envs.locomotion.g1 import joystick_numba as T

    n = 3
    n_action = 29
    rng = np.random.default_rng(1)
    linvel = rng.normal(size=(n, 3)).astype(np.float32)
    gyro = rng.normal(size=(n, 3)).astype(np.float32)
    gravity = rng.normal(size=(n, 3)).astype(np.float32)
    commands = rng.normal(size=(n, 3)).astype(np.float32)
    dof_pos = rng.normal(size=(n, n_action)).astype(np.float32)
    current_actions = rng.normal(size=(n, n_action)).astype(np.float32)
    last_actions = rng.normal(size=(n, n_action)).astype(np.float32)
    default_angles = rng.normal(size=(n_action,)).astype(np.float32)
    weights = rng.uniform(0.1, 2.0, size=(n_action,)).astype(np.float32)
    base_height = rng.uniform(0.4, 0.9, size=(n,)).astype(np.float32)
    gait_phase = rng.uniform(0.0, 2 * np.pi, size=(n, 2)).astype(np.float32)
    left_foot_pos = rng.normal(size=(n, 3)).astype(np.float32)
    right_foot_pos = rng.normal(size=(n, 3)).astype(np.float32)
    left_foot_quat = rng.normal(size=(n, 4)).astype(np.float32)
    right_foot_quat = rng.normal(size=(n, 4)).astype(np.float32)
    left_contact = np.array([True, False, True])
    right_contact = np.array([True, True, False])
    feet_air_time = rng.uniform(0.0, 0.8, size=(n, 2)).astype(np.float32)

    i = 1
    tracking_sigma = 0.25
    base_height_target = 0.754
    swing_height = 0.09
    feet_sigma = 0.04
    min_forward_speed = 0.0
    close_feet_threshold = 0.15

    np.testing.assert_allclose(
        T.tracking_lin_vel_i.py_func(linvel, commands, tracking_sigma, i),
        np.exp(-np.sum((commands[i, :2] - linvel[i, :2]) ** 2) / tracking_sigma),
    )
    np.testing.assert_allclose(
        T.tracking_ang_vel_i.py_func(gyro, commands, tracking_sigma, i),
        np.exp(-((commands[i, 2] - gyro[i, 2]) ** 2) / tracking_sigma),
    )
    commanded_speed = max(commands[i, 0], 1.0e-6)
    forward_speed = max(linvel[i, 0], 0.0)
    assert T.forward_progress_i.py_func(linvel, commands, i) == pytest.approx(
        min(forward_speed / commanded_speed, 1.0)
    )
    assert T.under_speed_i.py_func(linvel, commands, i) == pytest.approx(
        max(commands[i, 0] - forward_speed, 0.0) / commanded_speed
    )
    assert T.lin_vel_z_i.py_func(linvel, i) == pytest.approx(linvel[i, 2] ** 2)
    assert T.orientation_i.py_func(gravity, i) == pytest.approx(
        gravity[i, 0] ** 2 + gravity[i, 1] ** 2
    )
    assert T.ang_vel_xy_i.py_func(gyro, i) == pytest.approx(gyro[i, 0] ** 2 + gyro[i, 1] ** 2)
    np.testing.assert_allclose(
        T.action_rate_i.py_func(current_actions, last_actions, n_action, i),
        np.sum((current_actions[i] - last_actions[i]) ** 2),
    )
    np.testing.assert_allclose(
        T.weighted_pose_i.py_func(dof_pos, default_angles, weights, n_action, i),
        np.sum(weights * (dof_pos[i] - default_angles) ** 2),
    )
    assert T.base_height_i.py_func(base_height, base_height_target, i) == pytest.approx(
        (base_height[i] - base_height_target) ** 2
    )

    feet_dist = np.linalg.norm(left_foot_pos[i, :2] - right_foot_pos[i, :2])
    close_feet = (
        (feet_dist - close_feet_threshold) ** 2 if feet_dist < close_feet_threshold else 0.0
    )
    assert T.close_feet_xy_i.py_func(
        left_foot_pos, right_foot_pos, close_feet_threshold, i
    ) == pytest.approx(close_feet)
    np.testing.assert_allclose(
        T.feet_ori_i.py_func(left_foot_quat, right_foot_quat, i),
        left_foot_quat[i, 1] ** 2
        + left_foot_quat[i, 2] ** 2
        + right_foot_quat[i, 1] ** 2
        + right_foot_quat[i, 2] ** 2,
    )
    assert T.feet_air_time_i.py_func(feet_air_time, i) == pytest.approx(
        np.sum((feet_air_time[i] > 0.05) & (feet_air_time[i] < 0.5))
    )
    assert T.terminated_i.py_func(gravity, base_height, np.deg2rad(65.0), 0.3, i) == pytest.approx(
        np.arccos(np.clip(gravity[i, 2], -1.0, 1.0)) > np.deg2rad(65.0) or base_height[i] < 0.3
    )

    # Phase terms share the same scalar Bezier helper; checking they return
    # finite values catches drift in the less vector-friendly gait math.
    assert np.isfinite(
        T.feet_phase_i.py_func(
            linvel,
            gait_phase,
            left_foot_pos,
            right_foot_pos,
            swing_height,
            feet_sigma,
            min_forward_speed,
            i,
        )
    )
    assert np.isfinite(
        T.feet_phase_contrast_i.py_func(
            linvel,
            gait_phase,
            left_foot_pos,
            right_foot_pos,
            swing_height,
            feet_sigma,
            min_forward_speed,
            i,
        )
    )
    assert np.isfinite(
        T.feet_phase_contact_i.py_func(
            linvel,
            gait_phase,
            left_contact,
            right_contact,
            swing_height,
            min_forward_speed,
            i,
        )
    )
    assert T.feet_double_stance_i.py_func(
        commands, left_contact, right_contact, i
    ) == pytest.approx(
        float(left_contact[i] and right_contact[i]) * float(max(commands[i, 0], 0.0) > 1.0e-6)
    )
