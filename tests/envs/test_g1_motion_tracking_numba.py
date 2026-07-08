from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
import pytest

from unilab.envs.motion_tracking.g1.motion_tracking_numba import (
    NUMBA_AVAILABLE,
    G1MotionTrackingNumbaAccelerator,
    unsupported_terms,
)
from unilab.envs.motion_tracking.g1.tracking import (
    G1MotionTrackingCfg,
    G1MotionTrackingEnv,
    RewardConfig,
)


def test_unsupported_terms_only_reports_active_unknown_terms():
    assert unsupported_terms({"motion_body_pos": 1.0, "custom": 0.0}) == frozenset()
    assert unsupported_terms({"motion_body_pos": 1.0, "custom": 1.0}) == frozenset({"custom"})


def test_numba_accelerator_requires_numba_when_declared(monkeypatch):
    import unilab.envs.motion_tracking.g1.motion_tracking_numba as motion_tracking_numba

    monkeypatch.setattr(motion_tracking_numba, "NUMBA_AVAILABLE", False)
    with pytest.raises(RuntimeError, match="requires numba"):
        G1MotionTrackingNumbaAccelerator.from_env(_make_env(8, include_undesired=False))


def test_numba_accelerator_rejects_unsupported_active_reward_terms():
    env = _make_env(8, include_undesired=False)
    env._cfg.reward_config.scales = {"motion_body_pos": 1.0, "custom": 1.0}
    if NUMBA_AVAILABLE:
        with pytest.raises(ValueError, match="does not support active reward terms"):
            G1MotionTrackingNumbaAccelerator.from_env(env)


@pytest.mark.skipif(not NUMBA_AVAILABLE, reason="numba is optional")
@pytest.mark.slow
def test_g1_motion_tracking_numba_reward_termination_parity():
    n = 512
    env = _make_env(n, include_undesired=True)
    motion_data, robot_state, info = _make_batch(n, seed=0)
    (
        robot_body_pos_w,
        robot_body_quat_w,
        robot_body_lin_vel_w,
        robot_body_ang_vel_w,
        dof_pos,
        dof_vel,
    ) = robot_state

    env._update_relative_transforms(motion_data, robot_body_pos_w, robot_body_quat_w)
    terminated_np = env._compute_terminations(motion_data, robot_body_pos_w, robot_body_quat_w)
    reward_np = env._compute_reward(
        info,
        motion_data,
        robot_body_pos_w,
        robot_body_quat_w,
        robot_body_lin_vel_w,
        robot_body_ang_vel_w,
        dof_pos,
        dof_vel,
    )
    log_np = dict(info["log"])

    accel = G1MotionTrackingNumbaAccelerator.from_env(env, num_threads=2)
    out = accel.compute(
        info={
            "steps": np.zeros((n,), dtype=np.uint32),
            "current_actions": info["current_actions"],
            "last_actions": info["last_actions"],
        },
        motion_data=motion_data,
        ref_body_pos_w=env.body_pos_relative_w,
        ref_body_quat_w=env.body_quat_relative_w,
        robot_body_pos_w=robot_body_pos_w,
        robot_body_quat_w=robot_body_quat_w,
        robot_body_lin_vel_w=robot_body_lin_vel_w,
        robot_body_ang_vel_w=robot_body_ang_vel_w,
        dof_pos=dof_pos,
        dof_vel=dof_vel,
        scales=env._cfg.reward_config.scales,
        enable_log=True,
    )

    np.testing.assert_allclose(out.reward, reward_np, rtol=1e-4, atol=1e-5)
    np.testing.assert_array_equal(out.terminated, terminated_np)
    assert set(out.log) == set(log_np)
    for key in log_np:
        assert out.log[key] == pytest.approx(log_np[key], rel=1e-3, abs=1e-6)


@pytest.mark.skipif(not NUMBA_AVAILABLE, reason="numba is optional")
@pytest.mark.slow
def test_g1_motion_tracking_numba_update_state_parity_without_noise():
    n = 512
    env = _make_env(n, include_undesired=True)
    env.default_angles = np.linspace(-0.2, 0.2, env._num_action).astype(np.float32)
    env._actor_obs_width = env._actor_obs_dim(env._num_action)
    env._critic_base_obs_width = env._critic_base_obs_dim(env._num_action)
    env._critic_obs_width = env._critic_base_obs_width + env._n_motion_bodies * 9
    env._motion_anchor_pos_b = np.empty((n, 3), dtype=np.float32)
    env._motion_anchor_ori_b = np.empty((n, 6), dtype=np.float32)
    env._joint_pos_rel = np.empty((n, env._num_action), dtype=np.float32)
    env._motion_command = np.empty((n, env._num_action * 2), dtype=np.float32)
    env._zero_actions = np.zeros((n, env._num_action), dtype=np.float32)
    env._body_vec_tmp = np.empty((n, env._n_motion_bodies, 3), dtype=np.float32)
    env._cfg.noise_config = _NoiseCfg()

    motion_data, robot_state, info = _make_batch(n, seed=2)
    (
        robot_body_pos_w,
        robot_body_quat_w,
        robot_body_lin_vel_w,
        robot_body_ang_vel_w,
        dof_pos,
        dof_vel,
    ) = robot_state
    rng = np.random.default_rng(22)
    linvel = rng.uniform(-1.0, 1.0, (n, 3)).astype(np.float32)
    gyro = rng.uniform(-1.0, 1.0, (n, 3)).astype(np.float32)

    env._update_relative_transforms(motion_data, robot_body_pos_w, robot_body_quat_w)
    reward_np = env._compute_reward(
        {**info, "log": {}},
        motion_data,
        robot_body_pos_w,
        robot_body_quat_w,
        robot_body_lin_vel_w,
        robot_body_ang_vel_w,
        dof_pos,
        dof_vel,
    )
    terminated_np = env._compute_terminations(motion_data, robot_body_pos_w, robot_body_quat_w)
    obs_np = env._compute_obs(
        info,
        motion_data,
        linvel,
        gyro,
        dof_pos,
        dof_vel,
        robot_body_pos_w,
        robot_body_quat_w,
    )

    accel = G1MotionTrackingNumbaAccelerator.from_env(env, num_threads=2)
    out = accel.compute_update_state(
        info=info,
        motion_data=motion_data,
        linvel=linvel,
        gyro=gyro,
        dof_pos=dof_pos,
        dof_vel=dof_vel,
        robot_body_pos_w=robot_body_pos_w,
        robot_body_quat_w=robot_body_quat_w,
        robot_body_lin_vel_w=robot_body_lin_vel_w,
        robot_body_ang_vel_w=robot_body_ang_vel_w,
        ref_body_pos_w=env.body_pos_relative_w,
        ref_body_quat_w=env.body_quat_relative_w,
        motion_anchor_pos_b=env._motion_anchor_pos_b,
        motion_anchor_ori_b=env._motion_anchor_ori_b,
        joint_pos_rel=env._joint_pos_rel,
        scales=env._cfg.reward_config.scales,
        enable_log=True,
        noise_level=0.0,
        noise_scale_linvel=0.0,
        noise_scale_gyro=0.0,
        noise_scale_joint_angle=0.0,
        noise_scale_joint_vel=0.0,
    )

    np.testing.assert_allclose(out.reward, reward_np, rtol=1e-4, atol=1e-5)
    np.testing.assert_array_equal(out.terminated, terminated_np)
    np.testing.assert_allclose(out.obs["obs"], obs_np["obs"], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(out.obs["critic"], obs_np["critic"], rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not NUMBA_AVAILABLE, reason="numba is optional")
def test_g1_motion_tracking_numba_term_py_funcs_match_numpy_math():
    from unilab.envs.motion_tracking.g1 import motion_tracking_numba as T

    n = 4
    env = _make_env(n, include_undesired=True)
    motion_data, robot_state, _ = _make_batch(n, seed=1)
    robot_body_pos_w, robot_body_quat_w, _, _, dof_pos, dof_vel = robot_state
    env._update_relative_transforms(motion_data, robot_body_pos_w, robot_body_quat_w)
    i = 2

    diff = motion_data.body_pos_w[i, env.anchor_body_idx] - robot_body_pos_w[i, env.anchor_body_idx]
    expected_root_pos = np.exp(-np.sum(diff * diff) / (env._cfg.reward_config.std_root_pos**2))
    assert T.motion_global_root_pos_i.py_func(
        motion_data.body_pos_w,
        robot_body_pos_w,
        env.anchor_body_idx,
        env._cfg.reward_config.std_root_pos,
        i,
    ) == pytest.approx(expected_root_pos)

    body_diff = env.body_pos_relative_w[i] - robot_body_pos_w[i]
    expected_body_pos = np.exp(
        -np.sum(body_diff * body_diff)
        / body_diff.shape[0]
        / (env._cfg.reward_config.std_body_pos**2)
    )
    assert T.motion_body_pos_i.py_func(
        env.body_pos_relative_w,
        robot_body_pos_w,
        env._n_motion_bodies,
        env._cfg.reward_config.std_body_pos,
        i,
    ) == pytest.approx(expected_body_pos)

    expected_action_rate = np.sum((dof_pos[i] - dof_vel[i]) ** 2)
    assert T.action_rate_l2_i.py_func(dof_pos, dof_vel, env._num_action, i) == pytest.approx(
        expected_action_rate
    )

    expected_undesired = np.sum(
        robot_body_pos_w[i, env.undesired_contact_body_indices, 2]
        < env._cfg.undesired_contact_z_threshold
    )
    assert T.undesired_contacts_i.py_func(
        robot_body_pos_w,
        env.undesired_contact_body_indices,
        env._cfg.undesired_contact_z_threshold,
        i,
    ) == pytest.approx(float(expected_undesired))


@dataclass
class _MotionData:
    body_pos_w: np.ndarray
    body_quat_w: np.ndarray
    body_lin_vel_w: np.ndarray
    body_ang_vel_w: np.ndarray
    joint_pos: np.ndarray
    joint_vel: np.ndarray


@dataclass
class _Cfg:
    ctrl_dt: float = 0.02
    reward_config: RewardConfig = field(default_factory=RewardConfig)
    body_names: tuple[str, ...] = G1MotionTrackingCfg.body_names
    anchor_body_name: str = G1MotionTrackingCfg.anchor_body_name
    anchor_pos_z_threshold: float = G1MotionTrackingCfg.anchor_pos_z_threshold
    anchor_ori_threshold: float = G1MotionTrackingCfg.anchor_ori_threshold
    ee_body_pos_z_threshold: float = G1MotionTrackingCfg.ee_body_pos_z_threshold
    ee_body_names: tuple[str, ...] = G1MotionTrackingCfg.ee_body_names
    undesired_contact_z_threshold: float = G1MotionTrackingCfg.undesired_contact_z_threshold
    terminate_on_undesired_contacts: bool = False


@dataclass
class _NoiseCfg:
    level: float = 0.0
    scale_linvel: float = 0.1
    scale_gyro: float = 0.1
    scale_joint_angle: float = 0.02
    scale_joint_vel: float = 0.3


def _make_env(n: int, *, include_undesired: bool) -> Any:
    env = cast(Any, object.__new__(G1MotionTrackingEnv))
    cfg = _Cfg()
    cfg.reward_config.scales = {
        "motion_global_root_pos": 0.5,
        "motion_global_root_ori": 0.5,
        "motion_body_pos": 1.0,
        "motion_body_ori": 1.0,
        "motion_body_lin_vel": 1.0,
        "motion_body_ang_vel": 1.0,
        "motion_ee_body_pos_z": 0.3,
        "motion_joint_pos": 0.4,
        "motion_joint_vel": 0.2,
        "action_rate_l2": -0.1,
        "joint_limit": -2.0,
        "undesired_contacts": -0.1 if include_undesired else 0.0,
    }
    env._cfg = cfg
    env._num_envs = n
    env._num_action = 29
    env.anchor_body_idx = cfg.body_names.index(cfg.anchor_body_name)
    env.ee_body_indices = np.array([cfg.body_names.index(name) for name in cfg.ee_body_names])
    ee_set = set(cfg.ee_body_names)
    env.undesired_contact_body_indices = np.array(
        [idx for idx, name in enumerate(cfg.body_names) if name not in ee_set], dtype=np.int32
    )
    env._has_ee_body_indices = True
    env._has_undesired_contact_body_indices = True
    env._n_motion_bodies = len(cfg.body_names)
    env._joint_range = np.column_stack(
        [np.full(env._num_action, -2.0), np.full(env._num_action, 2.0)]
    ).astype(np.float32)
    env._joint_lower = env._joint_range[:, 0]
    env._joint_upper = env._joint_range[:, 1]
    env.body_pos_relative_w = np.zeros((n, env._n_motion_bodies, 3), dtype=np.float32)
    env.body_quat_relative_w = np.zeros((n, env._n_motion_bodies, 4), dtype=np.float32)
    env.body_quat_relative_w[..., 0] = 1.0
    env._delta_pos_w = np.empty((n, 3), dtype=np.float32)
    env._delta_ori_w = np.empty((n, 4), dtype=np.float32)
    env._body_vec_error = np.empty((n, env._n_motion_bodies, 3), dtype=np.float32)
    env._env_error = np.empty((n,), dtype=np.float32)
    env._env_error2 = np.empty((n,), dtype=np.float32)
    env._reward_term = np.empty((n,), dtype=np.float32)
    env._weighted_reward = np.empty((n,), dtype=np.float32)
    env._terminated = np.empty((n,), dtype=bool)
    env._env_bool = np.empty((n,), dtype=bool)
    env._quat_error_w = np.empty((n, env._n_motion_bodies), dtype=np.float32)
    env._quat_error_x = np.empty((n, env._n_motion_bodies), dtype=np.float32)
    env._joint_error = np.empty((n, env._num_action), dtype=np.float32)
    env._joint_error_upper = np.empty((n, env._num_action), dtype=np.float32)
    env._ee_pos_error_z = np.empty((n, env.ee_body_indices.size), dtype=np.float32)
    env._ee_terminated = np.empty((n, env.ee_body_indices.size), dtype=bool)
    env._undesired_contact_mask = np.empty((n, env.undesired_contact_body_indices.size), dtype=bool)
    env._enable_reward_log = True
    env._init_reward_functions()
    env._active_reward_fns = {
        name: reward_fn
        for name, reward_fn in env._reward_fns.items()
        if env._reward_term_is_active(name)
    }
    return env


def _unit_quats(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    q = rng.standard_normal((*shape, 4)).astype(np.float32)
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    return q


def _perturb_quats(rng: np.random.Generator, q: np.ndarray, sigma: float) -> np.ndarray:
    out = q + sigma * rng.standard_normal(q.shape).astype(np.float32)
    out /= np.linalg.norm(out, axis=-1, keepdims=True)
    return out


def _make_batch(n: int, seed: int):
    rng = np.random.default_rng(seed)
    nb = len(G1MotionTrackingCfg.body_names)
    na = 29
    target_pos = rng.uniform(-1.0, 1.0, (n, nb, 3)).astype(np.float32)
    target_quat = _unit_quats(rng, (n, nb))
    target_lin_vel = rng.uniform(-2.0, 2.0, (n, nb, 3)).astype(np.float32)
    target_ang_vel = rng.uniform(-3.0, 3.0, (n, nb, 3)).astype(np.float32)
    target_joint_pos = rng.uniform(-1.0, 1.0, (n, na)).astype(np.float32)
    target_joint_vel = rng.uniform(-2.0, 2.0, (n, na)).astype(np.float32)

    motion_data = _MotionData(
        body_pos_w=np.ascontiguousarray(target_pos + 0.03 * rng.standard_normal((n, nb, 3))),
        body_quat_w=np.ascontiguousarray(_perturb_quats(rng, target_quat, 0.02)),
        body_lin_vel_w=np.ascontiguousarray(target_lin_vel + 0.1 * rng.standard_normal((n, nb, 3))),
        body_ang_vel_w=np.ascontiguousarray(target_ang_vel + 0.1 * rng.standard_normal((n, nb, 3))),
        joint_pos=np.ascontiguousarray(target_joint_pos + 0.03 * rng.standard_normal((n, na))),
        joint_vel=np.ascontiguousarray(target_joint_vel + 0.05 * rng.standard_normal((n, na))),
    )
    robot_body_pos_w = np.ascontiguousarray(
        target_pos + 0.05 * rng.standard_normal((n, nb, 3)), dtype=np.float32
    )
    robot_body_quat_w = np.ascontiguousarray(_perturb_quats(rng, target_quat, 0.03))
    robot_body_lin_vel_w = np.ascontiguousarray(
        target_lin_vel + 0.1 * rng.standard_normal((n, nb, 3)), dtype=np.float32
    )
    robot_body_ang_vel_w = np.ascontiguousarray(
        target_ang_vel + 0.1 * rng.standard_normal((n, nb, 3)), dtype=np.float32
    )
    dof_pos = np.ascontiguousarray(
        target_joint_pos + 0.05 * rng.standard_normal((n, na)), dtype=np.float32
    )
    dof_vel = np.ascontiguousarray(
        target_joint_vel + 0.1 * rng.standard_normal((n, na)), dtype=np.float32
    )
    current_actions = rng.uniform(-1.0, 1.0, (n, na)).astype(np.float32)
    last_actions = (current_actions + 0.1 * rng.standard_normal((n, na))).astype(np.float32)
    robot_body_pos_w[: max(1, n // 20), 0, 2] = 0.0
    info = {
        "steps": np.zeros((n,), dtype=np.uint32),
        "current_actions": current_actions,
        "last_actions": last_actions,
    }
    return (
        motion_data,
        (
            robot_body_pos_w,
            robot_body_quat_w,
            robot_body_lin_vel_w,
            robot_body_ang_vel_w,
            dof_pos,
            dof_vel,
        ),
        info,
    )
