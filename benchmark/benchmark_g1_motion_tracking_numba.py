#!/usr/bin/env python3
"""Benchmark the task-specific Numba backend for G1 motion tracking.

This benchmark has two scopes:

* default hot slice: reward plus termination, using deterministic synthetic
  arrays and the real ``G1MotionTrackingEnv`` reward/termination methods;
* default ``--e2e``: collector-side A/B through
  ``benchmark_offpolicy_collector_active.py`` without learner updates.

Run:
    uv run python -m benchmark.benchmark_g1_motion_tracking_numba
    uv run python benchmark/benchmark_g1_motion_tracking_numba.py
    uv run python -m benchmark.benchmark_g1_motion_tracking_numba --quick
    uv run python -m benchmark.benchmark_g1_motion_tracking_numba --no-e2e
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev
from typing import Any, cast

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from numba import get_num_threads, set_num_threads

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - plotting is optional in benchmark scripts
    plt = None

from benchmark.core.device_info import get_device_info_dict, get_device_info_line
from unilab.dtype_config import get_global_dtype
from unilab.envs.motion_tracking.g1.motion_tracking_numba import (
    G1MotionTrackingNumbaAccelerator,
)
from unilab.envs.motion_tracking.g1.tracking import (
    G1MotionTrackingCfg,
    G1MotionTrackingEnv,
    RewardConfig,
)

NUM_ACTION = 29
DEFAULT_THREADS = [2, 4, 8, 16, 32, 64]
QUICK_THREADS = [2, 4]
DEFAULT_NUM_ENVS = [512, 1024, 2048, 4096, 8192, 16384, 32768]
DEFAULT_E2E_NUM_ENVS = [1024, 2048, 4096, 8192, 16384, 32768]
DEFAULT_E2E_CASE = "sac/g1_motion_tracking/motrixsim"

PPO_SCALES = {
    "motion_global_root_pos": 1.0,
    "motion_global_root_ori": 0.5,
    "motion_body_pos": 1.0,
    "motion_body_ori": 1.0,
    "motion_body_lin_vel": 1.0,
    "motion_body_ang_vel": 1.0,
    "motion_joint_pos": 0.0,
    "motion_joint_vel": 0.0,
    "action_rate_l2": -0.05,
    "joint_limit": -10.0,
    "undesired_contacts": -0.1,
}
SAC_SCALES = {
    "motion_global_root_pos": 0.5,
    "motion_global_root_ori": 0.5,
    "motion_body_pos": 1.0,
    "motion_body_ori": 1.0,
    "motion_body_lin_vel": 1.0,
    "motion_body_ang_vel": 1.0,
    "motion_joint_pos": 0.0,
    "motion_joint_vel": 0.0,
    "action_rate_l2": -0.1,
    "joint_limit": -2.0,
    "undesired_contacts": -0.1,
}
FULL_SUPPORTED_SCALES = {
    "motion_global_root_pos": 1.0,
    "motion_global_root_ori": 0.5,
    "motion_body_pos": 1.0,
    "motion_body_ori": 1.0,
    "motion_body_lin_vel": 1.0,
    "motion_body_ang_vel": 1.0,
    "motion_ee_body_pos_z": 0.3,
    "motion_joint_pos": 0.4,
    "motion_joint_vel": 0.2,
    "action_rate_l2": -0.1,
    "joint_limit": -10.0,
    "undesired_contacts": -0.1,
}


@dataclass(frozen=True)
class ProfileSpec:
    name: str
    scales: dict[str, float]
    reward_cfg: RewardConfig


@dataclass
class BenchCase:
    profile: str
    num_envs: int
    path: str
    threads: int | None
    mean_ms: float
    min_ms: float
    std_ms: float
    env_per_s: float
    speedup_vs_numpy: float
    compile_ms: float | None = None
    parallel_speedup_vs_numba_1t: float | None = None
    parallel_efficiency: float | None = None


@dataclass
class ComponentCase:
    profile: str
    num_envs: int
    numba_threads: int
    relative_transform_ms: float
    numpy_reward_termination_ms: float
    numba_reward_termination_ms: float
    numpy_update_state_ms: float
    numba_update_state_ms: float
    numpy_total_ms: float
    numba_total_ms: float
    speedup_vs_numpy: float


@dataclass
class EndToEndCase:
    case: str
    path: str
    num_envs: int
    warmup_steps: int
    measure_steps: int
    numba_acceleration: bool
    numba_threads: int | None
    collector_active_steps_per_sec: float
    total_active_ms: float
    collector_step_ms: float
    env_step_ms: float
    physics_step_ms: float | None
    update_state_ms: float | None
    other_ms: float
    speedup_vs_numpy: float = 1.0
    env_step_speedup_vs_numpy: float | None = None
    update_state_speedup_vs_numpy: float | None = None


@dataclass
class MotionDataBatch:
    body_pos_w: np.ndarray
    body_quat_w: np.ndarray
    body_lin_vel_w: np.ndarray
    body_ang_vel_w: np.ndarray
    joint_pos: np.ndarray
    joint_vel: np.ndarray


@dataclass
class SyntheticBatch:
    env: G1MotionTrackingEnv
    info: dict[str, Any]
    motion_data: MotionDataBatch
    linvel: np.ndarray
    gyro: np.ndarray
    robot_body_pos_w: np.ndarray
    robot_body_quat_w: np.ndarray
    robot_body_lin_vel_w: np.ndarray
    robot_body_ang_vel_w: np.ndarray
    dof_pos: np.ndarray
    dof_vel: np.ndarray


@dataclass
class SyntheticCfg:
    ctrl_dt: float = 0.02
    reward_config: RewardConfig = field(default_factory=RewardConfig)
    body_names: tuple[str, ...] = G1MotionTrackingCfg.body_names
    anchor_body_name: str = G1MotionTrackingCfg.anchor_body_name
    anchor_pos_z_threshold: float = G1MotionTrackingCfg.anchor_pos_z_threshold
    anchor_ori_threshold: float = G1MotionTrackingCfg.anchor_ori_threshold
    ee_body_pos_z_threshold: float = G1MotionTrackingCfg.ee_body_pos_z_threshold
    ee_body_names: tuple[str, ...] = G1MotionTrackingCfg.ee_body_names
    undesired_contact_z_threshold: float = G1MotionTrackingCfg.undesired_contact_z_threshold
    terminate_on_undesired_contacts: bool = G1MotionTrackingCfg.terminate_on_undesired_contacts


@dataclass
class SyntheticNoiseCfg:
    level: float = 0.0
    scale_linvel: float = 0.1
    scale_gyro: float = 0.1
    scale_joint_angle: float = 0.02
    scale_joint_vel: float = 0.3


def make_profile_specs() -> dict[str, ProfileSpec]:
    return {
        "ppo_default": ProfileSpec(
            name="ppo_default",
            scales=PPO_SCALES,
            reward_cfg=RewardConfig(scales=PPO_SCALES),
        ),
        "sac_default": ProfileSpec(
            name="sac_default",
            scales=SAC_SCALES,
            reward_cfg=RewardConfig(scales=SAC_SCALES),
        ),
        "full_supported": ProfileSpec(
            name="full_supported",
            scales=FULL_SUPPORTED_SCALES,
            reward_cfg=RewardConfig(scales=FULL_SUPPORTED_SCALES),
        ),
    }


def _make_fake_env(num_envs: int, reward_cfg: RewardConfig) -> G1MotionTrackingEnv:
    env = cast(G1MotionTrackingEnv, object.__new__(G1MotionTrackingEnv))
    cfg = SyntheticCfg(reward_config=reward_cfg)
    env._cfg = cfg
    env._num_envs = num_envs
    env._num_action = NUM_ACTION
    env.anchor_body_idx = cfg.body_names.index(cfg.anchor_body_name)
    env.ee_body_indices = np.array(
        [cfg.body_names.index(name) for name in cfg.ee_body_names], dtype=np.int32
    )
    ee_set = set(cfg.ee_body_names)
    env.undesired_contact_body_indices = np.array(
        [idx for idx, name in enumerate(cfg.body_names) if name not in ee_set], dtype=np.int32
    )
    env._has_ee_body_indices = bool(env.ee_body_indices.size)
    env._has_undesired_contact_body_indices = bool(env.undesired_contact_body_indices.size)
    env._n_motion_bodies = len(cfg.body_names)
    env._joint_range = np.column_stack(
        [np.full(NUM_ACTION, -2.5), np.full(NUM_ACTION, 2.5)]
    ).astype(get_global_dtype())
    env._joint_lower = env._joint_range[:, 0]
    env._joint_upper = env._joint_range[:, 1]
    dtype = get_global_dtype()
    n_body = env._n_motion_bodies
    env.default_angles = np.linspace(-0.2, 0.2, NUM_ACTION).astype(dtype)
    env._actor_obs_width = env._actor_obs_dim(NUM_ACTION)
    env._critic_base_obs_width = env._critic_base_obs_dim(NUM_ACTION)
    env._critic_obs_width = env._critic_base_obs_width + n_body * 9
    env._cfg.noise_config = SyntheticNoiseCfg()
    env.body_pos_relative_w = np.zeros((num_envs, n_body, 3), dtype=dtype)
    env.body_quat_relative_w = np.zeros((num_envs, n_body, 4), dtype=dtype)
    env.body_quat_relative_w[:, :, 0] = 1.0
    env._delta_pos_w = np.empty((num_envs, 3), dtype=dtype)
    env._delta_ori_w = np.empty((num_envs, 4), dtype=dtype)
    env._motion_anchor_pos_b = np.empty((num_envs, 3), dtype=dtype)
    env._motion_anchor_ori_b = np.empty((num_envs, 6), dtype=dtype)
    env._motion_command = np.empty((num_envs, NUM_ACTION * 2), dtype=dtype)
    env._joint_pos_rel = np.empty((num_envs, NUM_ACTION), dtype=dtype)
    env._zero_actions = np.zeros((num_envs, NUM_ACTION), dtype=dtype)
    env._body_vec_error = np.empty((num_envs, n_body, 3), dtype=dtype)
    env._body_vec_tmp = np.empty((num_envs, n_body, 3), dtype=dtype)
    env._env_error = np.empty((num_envs,), dtype=dtype)
    env._env_error2 = np.empty((num_envs,), dtype=dtype)
    env._reward_term = np.empty((num_envs,), dtype=dtype)
    env._weighted_reward = np.empty((num_envs,), dtype=dtype)
    env._terminated = np.empty((num_envs,), dtype=bool)
    env._env_bool = np.empty((num_envs,), dtype=bool)
    env._quat_error_w = np.empty((num_envs, n_body), dtype=dtype)
    env._quat_error_x = np.empty((num_envs, n_body), dtype=dtype)
    env._joint_error = np.empty((num_envs, NUM_ACTION), dtype=dtype)
    env._joint_error_upper = np.empty((num_envs, NUM_ACTION), dtype=dtype)
    env._ee_pos_error_z = np.empty((num_envs, env.ee_body_indices.size), dtype=dtype)
    env._ee_terminated = np.empty((num_envs, env.ee_body_indices.size), dtype=bool)
    env._undesired_contact_mask = np.empty(
        (num_envs, env.undesired_contact_body_indices.size), dtype=bool
    )
    env._enable_reward_log = True
    env._init_reward_functions()
    env._active_reward_fns = {
        name: reward_fn
        for name, reward_fn in env._reward_fns.items()
        if env._reward_term_is_active(name)
    }
    return env


def _unit_quats(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    q = rng.standard_normal((*shape, 4)).astype(get_global_dtype())
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    return q


def _perturb_quats(rng: np.random.Generator, q: np.ndarray, sigma: float) -> np.ndarray:
    out = q + sigma * rng.standard_normal(q.shape).astype(q.dtype)
    out /= np.linalg.norm(out, axis=-1, keepdims=True)
    return out


def make_batch(num_envs: int, spec: ProfileSpec, seed: int) -> SyntheticBatch:
    rng = np.random.default_rng(seed)
    dtype = get_global_dtype()
    n_body = len(G1MotionTrackingCfg.body_names)

    def f32(value: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(value, dtype=dtype)

    target_pos = rng.uniform(-1.0, 1.0, (num_envs, n_body, 3)).astype(dtype)
    target_quat = _unit_quats(rng, (num_envs, n_body))
    target_lin_vel = rng.uniform(-2.0, 2.0, (num_envs, n_body, 3)).astype(dtype)
    target_ang_vel = rng.uniform(-3.0, 3.0, (num_envs, n_body, 3)).astype(dtype)
    target_joint_pos = rng.uniform(-1.0, 1.0, (num_envs, NUM_ACTION)).astype(dtype)
    target_joint_vel = rng.uniform(-2.0, 2.0, (num_envs, NUM_ACTION)).astype(dtype)

    motion_data = MotionDataBatch(
        body_pos_w=f32(target_pos + 0.04 * rng.standard_normal((num_envs, n_body, 3))),
        body_quat_w=f32(_perturb_quats(rng, target_quat, 0.02)),
        body_lin_vel_w=f32(target_lin_vel + 0.1 * rng.standard_normal((num_envs, n_body, 3))),
        body_ang_vel_w=f32(target_ang_vel + 0.1 * rng.standard_normal((num_envs, n_body, 3))),
        joint_pos=f32(target_joint_pos + 0.03 * rng.standard_normal((num_envs, NUM_ACTION))),
        joint_vel=f32(target_joint_vel + 0.05 * rng.standard_normal((num_envs, NUM_ACTION))),
    )
    robot_body_pos_w = f32(target_pos + 0.06 * rng.standard_normal((num_envs, n_body, 3)))
    robot_body_quat_w = f32(_perturb_quats(rng, target_quat, 0.03))
    robot_body_lin_vel_w = f32(
        target_lin_vel + 0.1 * rng.standard_normal((num_envs, n_body, 3))
    )
    robot_body_ang_vel_w = f32(
        target_ang_vel + 0.1 * rng.standard_normal((num_envs, n_body, 3))
    )
    linvel = f32(rng.uniform(-1.0, 1.0, (num_envs, 3)))
    gyro = f32(rng.uniform(-1.0, 1.0, (num_envs, 3)))
    dof_pos = f32(target_joint_pos + 0.05 * rng.standard_normal((num_envs, NUM_ACTION)))
    dof_vel = f32(target_joint_vel + 0.1 * rng.standard_normal((num_envs, NUM_ACTION)))
    current_actions = f32(rng.uniform(-1.0, 1.0, (num_envs, NUM_ACTION)))
    last_actions = f32(current_actions + 0.1 * rng.standard_normal((num_envs, NUM_ACTION)))

    env = _make_fake_env(num_envs, spec.reward_cfg)
    env._update_relative_transforms(motion_data, robot_body_pos_w, robot_body_quat_w)
    term_count = max(1, int(0.02 * num_envs))
    robot_body_pos_w[:term_count, env.anchor_body_idx, 2] += 0.6
    if spec.scales.get("undesired_contacts", 0.0) != 0.0:
        robot_body_pos_w[: max(1, term_count // 2), env.undesired_contact_body_indices[0], 2] = 0.0

    info = {
        "steps": np.zeros((num_envs,), dtype=np.uint32),
        "current_actions": current_actions,
        "last_actions": last_actions,
        "log": {},
    }
    return SyntheticBatch(
        env=env,
        info=info,
        motion_data=motion_data,
        linvel=linvel,
        gyro=gyro,
        robot_body_pos_w=robot_body_pos_w,
        robot_body_quat_w=robot_body_quat_w,
        robot_body_lin_vel_w=robot_body_lin_vel_w,
        robot_body_ang_vel_w=robot_body_ang_vel_w,
        dof_pos=dof_pos,
        dof_vel=dof_vel,
    )


def compute_numpy(batch: SyntheticBatch) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    env = batch.env
    info = {**batch.info, "log": {}}
    terminated = env._compute_terminations(
        batch.motion_data, batch.robot_body_pos_w, batch.robot_body_quat_w
    )
    reward = env._compute_reward(
        info,
        batch.motion_data,
        batch.robot_body_pos_w,
        batch.robot_body_quat_w,
        batch.robot_body_lin_vel_w,
        batch.robot_body_ang_vel_w,
        batch.dof_pos,
        batch.dof_vel,
    )
    return reward, terminated, info.get("log", {})


def compute_numba(
    batch: SyntheticBatch, accelerator: G1MotionTrackingNumbaAccelerator
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    info = {**batch.info, "log": {}}
    out = accelerator.compute(
        info=info,
        motion_data=batch.motion_data,
        ref_body_pos_w=batch.env.body_pos_relative_w,
        ref_body_quat_w=batch.env.body_quat_relative_w,
        robot_body_pos_w=batch.robot_body_pos_w,
        robot_body_quat_w=batch.robot_body_quat_w,
        robot_body_lin_vel_w=batch.robot_body_lin_vel_w,
        robot_body_ang_vel_w=batch.robot_body_ang_vel_w,
        dof_pos=batch.dof_pos,
        dof_vel=batch.dof_vel,
        scales=batch.env._cfg.reward_config.scales,
        enable_log=True,
    )
    return out.reward, out.terminated, out.log


def compute_relative_transforms(batch: SyntheticBatch) -> None:
    batch.env._update_relative_transforms(
        batch.motion_data, batch.robot_body_pos_w, batch.robot_body_quat_w
    )


def compute_numpy_update_state(
    batch: SyntheticBatch,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, dict[str, float]]:
    env = batch.env
    info = {**batch.info, "log": {}}
    env._update_relative_transforms(
        batch.motion_data, batch.robot_body_pos_w, batch.robot_body_quat_w
    )
    terminated = env._compute_terminations(
        batch.motion_data, batch.robot_body_pos_w, batch.robot_body_quat_w
    )
    reward = env._compute_reward(
        info,
        batch.motion_data,
        batch.robot_body_pos_w,
        batch.robot_body_quat_w,
        batch.robot_body_lin_vel_w,
        batch.robot_body_ang_vel_w,
        batch.dof_pos,
        batch.dof_vel,
    )
    obs = env._compute_obs(
        info,
        batch.motion_data,
        batch.linvel,
        batch.gyro,
        batch.dof_pos,
        batch.dof_vel,
        batch.robot_body_pos_w,
        batch.robot_body_quat_w,
    )
    return obs, reward, terminated, info.get("log", {})


def compute_numba_update_state(
    batch: SyntheticBatch, accelerator: G1MotionTrackingNumbaAccelerator
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, dict[str, float]]:
    info = {**batch.info, "log": {}}
    noise_cfg = batch.env._cfg.noise_config
    out = accelerator.compute_update_state(
        info=info,
        motion_data=batch.motion_data,
        linvel=batch.linvel,
        gyro=batch.gyro,
        dof_pos=batch.dof_pos,
        dof_vel=batch.dof_vel,
        robot_body_pos_w=batch.robot_body_pos_w,
        robot_body_quat_w=batch.robot_body_quat_w,
        robot_body_lin_vel_w=batch.robot_body_lin_vel_w,
        robot_body_ang_vel_w=batch.robot_body_ang_vel_w,
        ref_body_pos_w=batch.env.body_pos_relative_w,
        ref_body_quat_w=batch.env.body_quat_relative_w,
        motion_anchor_pos_b=batch.env._motion_anchor_pos_b,
        motion_anchor_ori_b=batch.env._motion_anchor_ori_b,
        joint_pos_rel=batch.env._joint_pos_rel,
        scales=batch.env._cfg.reward_config.scales,
        enable_log=True,
        noise_level=noise_cfg.level,
        noise_scale_linvel=noise_cfg.scale_linvel,
        noise_scale_gyro=noise_cfg.scale_gyro,
        noise_scale_joint_angle=noise_cfg.scale_joint_angle,
        noise_scale_joint_vel=noise_cfg.scale_joint_vel,
    )
    return out.obs, out.reward, out.terminated, out.log


def time_call(fn, *, iters: int, warmup: int) -> tuple[float, float, float]:
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1e3)
    return mean(samples), min(samples), stdev(samples) if len(samples) > 1 else 0.0


def check_parity(
    batch: SyntheticBatch, accelerator: G1MotionTrackingNumbaAccelerator
) -> dict[str, float]:
    reward_np, terminated_np, log_np = compute_numpy(batch)
    reward_nb, terminated_nb, log_nb = compute_numba(batch, accelerator)
    np.testing.assert_allclose(reward_nb, reward_np, rtol=1e-4, atol=1e-5)
    np.testing.assert_array_equal(terminated_nb, terminated_np)
    for key, value in log_np.items():
        if key in log_nb:
            np.testing.assert_allclose(log_nb[key], value, rtol=1e-3, atol=1e-6)
    obs_np, full_reward_np, full_terminated_np, _ = compute_numpy_update_state(batch)
    obs_nb, full_reward_nb, full_terminated_nb, _ = compute_numba_update_state(batch, accelerator)
    np.testing.assert_allclose(full_reward_nb, full_reward_np, rtol=1e-4, atol=1e-5)
    np.testing.assert_array_equal(full_terminated_nb, full_terminated_np)
    np.testing.assert_allclose(obs_nb["obs"], obs_np["obs"], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(obs_nb["critic"], obs_np["critic"], rtol=1e-5, atol=1e-5)
    return {
        "max_abs_reward_diff": float(np.max(np.abs(reward_nb - reward_np))),
        "termination_mismatch": float(np.count_nonzero(terminated_nb != terminated_np)),
        "max_abs_update_state_reward_diff": float(
            np.max(np.abs(full_reward_nb - full_reward_np))
        ),
        "update_state_termination_mismatch": float(
            np.count_nonzero(full_terminated_nb != full_terminated_np)
        ),
        "max_abs_actor_obs_diff": float(np.max(np.abs(obs_nb["obs"] - obs_np["obs"]))),
        "max_abs_critic_obs_diff": float(np.max(np.abs(obs_nb["critic"] - obs_np["critic"]))),
    }


def bench_one(
    *,
    profile: ProfileSpec,
    num_envs: int,
    thread_counts: list[int],
    iters: int,
    warmup: int,
    seed: int,
) -> tuple[list[BenchCase], ComponentCase, dict[str, float]]:
    batch = make_batch(num_envs, profile, seed)
    max_threads = get_num_threads()
    relative_mean, _, _ = time_call(
        lambda: compute_relative_transforms(batch), iters=iters, warmup=warmup
    )
    numpy_update_state_mean, _, _ = time_call(
        lambda: compute_numpy_update_state(batch), iters=iters, warmup=warmup
    )
    numpy_mean, numpy_min, numpy_std = time_call(
        lambda: compute_numpy(batch), iters=iters, warmup=warmup
    )
    records = [
        BenchCase(
            profile=profile.name,
            num_envs=num_envs,
            path="numpy_dispatch",
            threads=None,
            mean_ms=numpy_mean,
            min_ms=numpy_min,
            std_ms=numpy_std,
            env_per_s=num_envs / (numpy_mean * 1e-3),
            speedup_vs_numpy=1.0,
        )
    ]

    compile_driver = G1MotionTrackingNumbaAccelerator.from_env(batch.env, num_threads=1)
    t0 = time.perf_counter()
    compute_numba(batch, compile_driver)
    compute_numba_update_state(batch, compile_driver)
    compile_ms = (time.perf_counter() - t0) * 1e3
    parity = check_parity(batch, compile_driver)

    for threads in dict.fromkeys([1, *thread_counts]):
        if threads > max_threads:
            continue
        accelerator = G1MotionTrackingNumbaAccelerator.from_env(batch.env, num_threads=threads)
        numba_mean, numba_min, numba_std = time_call(
            lambda: compute_numba(batch, accelerator), iters=iters, warmup=warmup
        )
        records.append(
            BenchCase(
                profile=profile.name,
                num_envs=num_envs,
                path="numba_accelerator",
                threads=threads,
                mean_ms=numba_mean,
                min_ms=numba_min,
                std_ms=numba_std,
                env_per_s=num_envs / (numba_mean * 1e-3),
                speedup_vs_numpy=numpy_mean / numba_mean,
                compile_ms=compile_ms if threads == 1 else None,
            )
        )
    set_num_threads(max_threads)
    numba_1t = next(
        record
        for record in records
        if record.path == "numba_accelerator" and record.threads == 1
    )
    for record in records:
        if record.path != "numba_accelerator" or record.threads is None:
            continue
        record.parallel_speedup_vs_numba_1t = numba_1t.mean_ms / record.mean_ms
        record.parallel_efficiency = record.parallel_speedup_vs_numba_1t / record.threads

    best_numba = min(
        (record for record in records if record.path == "numba_accelerator"),
        key=lambda record: record.mean_ms,
    )
    best_update_accelerator = G1MotionTrackingNumbaAccelerator.from_env(
        batch.env, num_threads=best_numba.threads
    )
    numba_update_state_mean, _, _ = time_call(
        lambda: compute_numba_update_state(batch, best_update_accelerator),
        iters=iters,
        warmup=warmup,
    )
    numpy_total_ms = numpy_update_state_mean
    numba_total_ms = numba_update_state_mean
    component = ComponentCase(
        profile=profile.name,
        num_envs=num_envs,
        numba_threads=int(best_numba.threads or 1),
        relative_transform_ms=relative_mean,
        numpy_reward_termination_ms=numpy_mean,
        numba_reward_termination_ms=best_numba.mean_ms,
        numpy_update_state_ms=numpy_update_state_mean,
        numba_update_state_ms=numba_update_state_mean,
        numpy_total_ms=numpy_total_ms,
        numba_total_ms=numba_total_ms,
        speedup_vs_numpy=numpy_total_ms / numba_total_ms,
    )
    return records, component, parity


def _case_to_dict(case: BenchCase) -> dict[str, Any]:
    return {
        "profile": case.profile,
        "num_envs": case.num_envs,
        "path": case.path,
        "threads": case.threads,
        "mean_ms": case.mean_ms,
        "min_ms": case.min_ms,
        "std_ms": case.std_ms,
        "env_per_s": case.env_per_s,
        "speedup_vs_numpy": case.speedup_vs_numpy,
        "compile_ms": case.compile_ms,
        "parallel_speedup_vs_numba_1t": case.parallel_speedup_vs_numba_1t,
        "parallel_efficiency": case.parallel_efficiency,
    }


def _component_case_to_dict(case: ComponentCase) -> dict[str, Any]:
    return {
        "profile": case.profile,
        "num_envs": case.num_envs,
        "numba_threads": case.numba_threads,
        "relative_transform_ms": case.relative_transform_ms,
        "numpy_reward_termination_ms": case.numpy_reward_termination_ms,
        "numba_reward_termination_ms": case.numba_reward_termination_ms,
        "numpy_update_state_ms": case.numpy_update_state_ms,
        "numba_update_state_ms": case.numba_update_state_ms,
        "numpy_total_ms": case.numpy_total_ms,
        "numba_total_ms": case.numba_total_ms,
        "speedup_vs_numpy": case.speedup_vs_numpy,
    }


def _e2e_case_to_dict(case: EndToEndCase) -> dict[str, Any]:
    return {
        "case": case.case,
        "path": case.path,
        "num_envs": case.num_envs,
        "warmup_steps": case.warmup_steps,
        "measure_steps": case.measure_steps,
        "numba_acceleration": case.numba_acceleration,
        "numba_threads": case.numba_threads,
        "collector_active_steps_per_sec": case.collector_active_steps_per_sec,
        "total_active_ms": case.total_active_ms,
        "collector_step_ms": case.collector_step_ms,
        "env_step_ms": case.env_step_ms,
        "physics_step_ms": case.physics_step_ms,
        "update_state_ms": case.update_state_ms,
        "other_ms": case.other_ms,
        "speedup_vs_numpy": case.speedup_vs_numpy,
        "env_step_speedup_vs_numpy": case.env_step_speedup_vs_numpy,
        "update_state_speedup_vs_numpy": case.update_state_speedup_vs_numpy,
    }


def _timing_mean_ms(result: Any, key: str) -> float | None:
    stat = result.env_step_timing_ms_per_vector_step.get(key)
    return float(stat.mean_ms) if stat is not None else None


def _run_e2e_collector_pair(
    *,
    case_name: str,
    num_envs: int,
    warmup_steps: int,
    measure_steps: int,
    numba_threads: int | None,
) -> list[EndToEndCase]:
    from benchmark.benchmark_offpolicy_collector_active import _build_and_run_case

    common = {
        "warmup_steps": warmup_steps,
        "measure_steps": measure_steps,
        "replay_capacity_steps": max(2, measure_steps + warmup_steps + 1),
        "num_envs": num_envs,
    }
    variants = [
        ("training_collector_numpy", False, ["++env.numba_acceleration=false"]),
        (
            "training_collector_numba",
            True,
            [
                "++env.numba_acceleration=true",
                f"++env.numba_num_threads={numba_threads}" if numba_threads is not None else "",
            ],
        ),
    ]
    records: list[EndToEndCase] = []
    for path, enabled, overrides in variants:
        result = _build_and_run_case(
            case_name,
            extra_overrides=[override for override in overrides if override],
            **common,
        )
        env_step_ms = float(result.phase_ms_per_vector_step["env_step_ms"].mean_ms)
        physics_step_ms = (
            float(result.physics_ms_per_vector_step.mean_ms)
            if result.physics_ms_per_vector_step is not None
            else None
        )
        update_state_ms = _timing_mean_ms(result, "update_state_ms")
        collector_step_ms = float(result.total_active_ms) / float(result.measure_steps)
        other_ms = collector_step_ms
        if physics_step_ms is not None:
            other_ms -= physics_step_ms
        if update_state_ms is not None:
            other_ms -= update_state_ms
        records.append(
            EndToEndCase(
                case=case_name,
                path=path,
                num_envs=num_envs,
                warmup_steps=warmup_steps,
                measure_steps=measure_steps,
                numba_acceleration=enabled,
                numba_threads=numba_threads if enabled else None,
                collector_active_steps_per_sec=float(result.collector_active_steps_per_sec),
                total_active_ms=float(result.total_active_ms),
                collector_step_ms=collector_step_ms,
                env_step_ms=env_step_ms,
                physics_step_ms=physics_step_ms,
                update_state_ms=update_state_ms,
                other_ms=other_ms,
            )
        )
    baseline = next(record for record in records if not record.numba_acceleration)
    for record in records:
        record.speedup_vs_numpy = (
            record.collector_active_steps_per_sec / baseline.collector_active_steps_per_sec
        )
        record.env_step_speedup_vs_numpy = (
            baseline.env_step_ms / record.env_step_ms if record.env_step_ms > 0.0 else None
        )
        if baseline.update_state_ms is not None and record.update_state_ms:
            record.update_state_speedup_vs_numpy = baseline.update_state_ms / record.update_state_ms
    return records


def _best_numba_by_case(records: list[BenchCase]) -> dict[tuple[str, int], BenchCase]:
    best_by_case: dict[tuple[str, int], BenchCase] = {}
    for record in records:
        if record.path != "numba_accelerator":
            continue
        key = (record.profile, record.num_envs)
        if key not in best_by_case or record.mean_ms < best_by_case[key].mean_ms:
            best_by_case[key] = record
    return best_by_case


def _best_threads_for_profile(
    records: list[BenchCase], *, profile: str, num_envs: list[int]
) -> dict[int, int]:
    best_by_case = _best_numba_by_case(records)
    selected: dict[int, int] = {}
    for env_count in num_envs:
        best = best_by_case.get((profile, env_count))
        if best is not None and best.threads is not None:
            selected[env_count] = int(best.threads)
    return selected


def _run_e2e_collector_sweep(
    *,
    case_name: str,
    num_envs: list[int],
    warmup_steps: int,
    measure_steps: int,
    selected_threads: dict[int, int],
    fallback_numba_threads: int | None,
) -> list[EndToEndCase]:
    records: list[EndToEndCase] = []
    for env_count in num_envs:
        numba_threads = selected_threads.get(env_count, fallback_numba_threads)
        print(f"e2e collector case: num_envs={env_count} numba_threads={numba_threads}")
        records.extend(
            _run_e2e_collector_pair(
                case_name=case_name,
                num_envs=env_count,
                warmup_steps=warmup_steps,
                measure_steps=measure_steps,
                numba_threads=numba_threads,
            )
        )
    return records


def _format_table(records: list[BenchCase]) -> str:
    headers = [
        "profile",
        "envs",
        "path",
        "threads",
        "mean_ms",
        "min_ms",
        "vs numpy",
        "vs numba1T",
        "parallel eff",
        "M env/s",
    ]
    rows = [
        [
            r.profile,
            str(r.num_envs),
            r.path,
            "-" if r.threads is None else str(r.threads),
            f"{r.mean_ms:.3f}",
            f"{r.min_ms:.3f}",
            f"{r.speedup_vs_numpy:.2f}x",
            (
                "-"
                if r.parallel_speedup_vs_numba_1t is None
                else f"{r.parallel_speedup_vs_numba_1t:.2f}x"
            ),
            "-" if r.parallel_efficiency is None else f"{100.0 * r.parallel_efficiency:.1f}%",
            f"{r.env_per_s / 1e6:.2f}",
        ]
        for r in records
    ]
    widths = [len(h) for h in headers]
    for row in rows:
        for idx, value in enumerate(row):
            widths[idx] = max(widths[idx], len(value))

    def fmt(values: list[str]) -> str:
        return " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(values))

    return "\n".join([fmt(headers), "-+-".join("-" * width for width in widths), *map(fmt, rows)])


def _format_component_table(records: list[ComponentCase]) -> str:
    headers = [
        "profile",
        "envs",
        "threads",
        "rel ms",
        "numpy reward+term ms",
        "numba reward+term ms",
        "numpy update_state ms",
        "numba update_state ms",
        "update_state speedup",
    ]
    rows = [
        [
            record.profile,
            str(record.num_envs),
            str(record.numba_threads),
            f"{record.relative_transform_ms:.3f}",
            f"{record.numpy_reward_termination_ms:.3f}",
            f"{record.numba_reward_termination_ms:.3f}",
            f"{record.numpy_update_state_ms:.3f}",
            f"{record.numba_update_state_ms:.3f}",
            f"{record.speedup_vs_numpy:.2f}x",
        ]
        for record in records
    ]
    widths = [len(h) for h in headers]
    for row in rows:
        for idx, value in enumerate(row):
            widths[idx] = max(widths[idx], len(value))

    def fmt(values: list[str]) -> str:
        return " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(values))

    return "\n".join([fmt(headers), "-+-".join("-" * width for width in widths), *map(fmt, rows)])


def _format_hot_summary_table(records: list[BenchCase]) -> str:
    best_by_case = _best_numba_by_case(records)
    rows = []
    for profile in sorted({record.profile for record in records}):
        best_records = [
            record for key, record in best_by_case.items() if key[0] == profile
        ]
        if not best_records:
            continue
        envs = sorted(record.num_envs for record in best_records)
        speedups = [record.speedup_vs_numpy for record in best_records]
        throughputs = [record.env_per_s / 1e6 for record in best_records]
        rows.append(
            [
                profile,
                f"{envs[0]}-{envs[-1]}",
                ",".join(str(thread) for thread in sorted({record.threads for record in best_records})),
                f"{min(speedups):.2f}x-{max(speedups):.2f}x",
                f"{max(throughputs):.2f}",
            ]
        )
    headers = ["profile", "env range", "best threads", "hot speedup range", "peak M env/s"]
    widths = [len(h) for h in headers]
    for row in rows:
        for idx, value in enumerate(row):
            widths[idx] = max(widths[idx], len(value))

    def fmt(values: list[str]) -> str:
        return " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(values))

    return "\n".join([fmt(headers), "-+-".join("-" * width for width in widths), *map(fmt, rows)])


def _format_parity_summary_table(parity: dict[str, dict[str, float]]) -> str:
    if not parity:
        return ""
    max_reward_diff = max(item["max_abs_reward_diff"] for item in parity.values())
    max_update_reward_diff = max(
        item["max_abs_update_state_reward_diff"] for item in parity.values()
    )
    max_actor_obs_diff = max(item["max_abs_actor_obs_diff"] for item in parity.values())
    max_critic_obs_diff = max(item["max_abs_critic_obs_diff"] for item in parity.values())
    termination_mismatch = sum(item["termination_mismatch"] for item in parity.values())
    update_termination_mismatch = sum(
        item["update_state_termination_mismatch"] for item in parity.values()
    )
    rows = [
        ["reward max abs diff", f"{max_reward_diff:.3e}"],
        ["update_state reward max abs diff", f"{max_update_reward_diff:.3e}"],
        ["actor obs max abs diff", f"{max_actor_obs_diff:.3e}"],
        ["critic obs max abs diff", f"{max_critic_obs_diff:.3e}"],
        ["termination mismatches", f"{termination_mismatch:.0f}"],
        ["update_state termination mismatches", f"{update_termination_mismatch:.0f}"],
    ]
    headers = ["check", "value"]
    widths = [len(h) for h in headers]
    for row in rows:
        for idx, value in enumerate(row):
            widths[idx] = max(widths[idx], len(value))

    def fmt(values: list[str]) -> str:
        return " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(values))

    return "\n".join([fmt(headers), "-+-".join("-" * width for width in widths), *map(fmt, rows)])


def _format_e2e_table(records: list[EndToEndCase]) -> str:
    headers = [
        "case",
        "envs",
        "path",
        "threads",
        "collector M steps/s",
        "speedup",
        "collector step ms",
        "env_step ms",
        "physics ms",
        "update_state ms",
        "other ms",
        "env_step speedup",
        "update_state speedup",
    ]
    rows = []
    for record in records:
        rows.append(
            [
                record.case,
                str(record.num_envs),
                record.path,
                "-" if record.numba_threads is None else str(record.numba_threads),
                f"{record.collector_active_steps_per_sec / 1e6:.3f}",
                f"{record.speedup_vs_numpy:.2f}x",
                f"{record.collector_step_ms:.3f}",
                f"{record.env_step_ms:.3f}",
                "-" if record.physics_step_ms is None else f"{record.physics_step_ms:.3f}",
                "-" if record.update_state_ms is None else f"{record.update_state_ms:.3f}",
                f"{record.other_ms:.3f}",
                (
                    "-"
                    if record.env_step_speedup_vs_numpy is None
                    else f"{record.env_step_speedup_vs_numpy:.2f}x"
                ),
                (
                    "-"
                    if record.update_state_speedup_vs_numpy is None
                    else f"{record.update_state_speedup_vs_numpy:.2f}x"
                ),
            ]
        )
    widths = [len(h) for h in headers]
    for row in rows:
        for idx, value in enumerate(row):
            widths[idx] = max(widths[idx], len(value))

    def fmt(values: list[str]) -> str:
        return " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(values))

    return "\n".join([fmt(headers), "-+-".join("-" * width for width in widths), *map(fmt, rows)])


def _hot_record_for_threads(
    records: list[BenchCase], *, profile: str, num_envs: int, threads: int | None
) -> BenchCase | None:
    candidates = [
        record
        for record in records
        if record.profile == profile
        and record.num_envs == num_envs
        and record.path == "numba_accelerator"
    ]
    if not candidates:
        return None
    if threads is None:
        return min(candidates, key=lambda record: record.mean_ms)
    return next((record for record in candidates if record.threads == threads), None)


def _numpy_record(records: list[BenchCase], *, profile: str, num_envs: int) -> BenchCase | None:
    return next(
        (
            record
            for record in records
            if record.profile == profile
            and record.num_envs == num_envs
            and record.path == "numpy_dispatch"
        ),
        None,
    )


def _format_e2e_reconciliation_table(
    *,
    hot_records: list[BenchCase],
    e2e_records: list[EndToEndCase],
    profile: str = "sac_default",
) -> str:
    baseline_by_env = {
        record.num_envs: record
        for record in e2e_records
        if not record.numba_acceleration and record.update_state_ms is not None
    }
    rows = []
    for record in e2e_records:
        if not record.numba_acceleration or record.update_state_ms is None:
            continue
        baseline = baseline_by_env.get(record.num_envs)
        numpy_hot = _numpy_record(hot_records, profile=profile, num_envs=record.num_envs)
        numba_hot = _hot_record_for_threads(
            hot_records,
            profile=profile,
            num_envs=record.num_envs,
            threads=record.numba_threads,
        )
        if baseline is None or baseline.update_state_ms is None or numpy_hot is None or numba_hot is None:
            continue
        hot_saved_ms = numpy_hot.mean_ms - numba_hot.mean_ms
        update_saved_ms = baseline.update_state_ms - record.update_state_ms
        update_base_ms = baseline.update_state_ms
        rows.append(
            [
                str(record.num_envs),
                "-" if record.numba_threads is None else str(record.numba_threads),
                f"{numpy_hot.mean_ms:.3f}",
                f"{numba_hot.mean_ms:.3f}",
                f"{hot_saved_ms:.3f}",
                f"{baseline.update_state_ms:.3f}",
                f"{record.update_state_ms:.3f}",
                f"{update_saved_ms:.3f}",
                f"{100.0 * hot_saved_ms / update_base_ms:.1f}%",
                f"{100.0 * update_saved_ms / update_base_ms:.1f}%",
            ]
        )
    if not rows:
        return ""

    headers = [
        "envs",
        "threads",
        "hot numpy ms",
        "hot numba ms",
        "hot saved ms",
        "e2e update numpy ms",
        "e2e update numba ms",
        "e2e saved ms",
        "hot saved / update base",
        "e2e saved / update base",
    ]
    widths = [len(h) for h in headers]
    for row in rows:
        for idx, value in enumerate(row):
            widths[idx] = max(widths[idx], len(value))

    def fmt(values: list[str]) -> str:
        return " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(values))

    return "\n".join([fmt(headers), "-+-".join("-" * width for width in widths), *map(fmt, rows)])


def save_plots(
    records: list[BenchCase],
    component_records: list[ComponentCase],
    output_dir: Path,
    *,
    device_info: str,
) -> list[str]:
    if plt is None or not records:
        print("Plotting skipped: matplotlib is not available.")
        return []
    output_dir.mkdir(parents=True, exist_ok=True)
    profiles = sorted({record.profile for record in records})
    num_envs = sorted({record.num_envs for record in records})
    best_by_case = _best_numba_by_case(records)

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(21, 6))
    fig.suptitle(f"G1 motion tracking Numba reward+termination benchmark\n{device_info}", fontsize=13)

    ax1, ax2, ax3 = axes
    for profile in profiles:
        x, y, labels = [], [], []
        for env_count in num_envs:
            best = best_by_case.get((profile, env_count))
            if best is None:
                continue
            x.append(env_count)
            y.append(best.speedup_vs_numpy)
            labels.append(best.threads)
        if x:
            ax1.plot(x, y, marker="o", label=profile)
            for x_val, y_val, threads in zip(x, y, labels):
                ax1.annotate(
                    f"{threads}T",
                    xy=(x_val, y_val),
                    xytext=(0, 6),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8,
                )
    ax1.axhline(1.0, color="grey", linestyle=":", linewidth=0.9, label="break-even")
    ax1.set_title("Reward+termination: best vs numpy")
    ax1.set_xlabel("num_envs")
    ax1.set_ylabel("Speedup vs numpy")
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(num_envs)
    ax1.set_xticklabels([str(value) for value in num_envs])
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8)

    for profile in profiles:
        best_subset = [
            best_by_case[(profile, env_count)]
            for env_count in num_envs
            if (profile, env_count) in best_by_case
        ]
        if best_subset:
            ax2.plot(
                [r.num_envs for r in best_subset],
                [
                    0.0
                    if r.parallel_speedup_vs_numba_1t is None
                    else r.parallel_speedup_vs_numba_1t
                    for r in best_subset
                ],
                marker="s",
                linestyle="-",
                label=profile,
            )
            for record in best_subset:
                if record.parallel_efficiency is None:
                    continue
                ax2.annotate(
                    f"{100.0 * record.parallel_efficiency:.0f}%",
                    xy=(record.num_envs, record.parallel_speedup_vs_numba_1t or 0.0),
                    xytext=(0, 6),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8,
                )
    ax2.axhline(1.0, color="grey", linestyle=":", linewidth=0.9, label="1T")
    ax2.set_title("Reward+termination: parallel speedup")
    ax2.set_xlabel("num_envs")
    ax2.set_ylabel("Speedup vs numba 1T")
    ax2.set_xscale("log", base=2)
    ax2.set_xticks(num_envs)
    ax2.set_xticklabels([str(value) for value in num_envs])
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=7)

    for profile in profiles:
        component_subset = sorted(
            [record for record in component_records if record.profile == profile],
            key=lambda record: record.num_envs,
        )
        if component_subset:
            ax3.plot(
                [record.num_envs for record in component_subset],
                [record.speedup_vs_numpy for record in component_subset],
                marker="o",
                linestyle="-",
                label=profile,
            )
            for record in component_subset:
                ax3.annotate(
                    f"{record.numba_threads}T",
                    xy=(record.num_envs, record.speedup_vs_numpy),
                    xytext=(0, 6),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8,
                )
    ax3.axhline(1.0, color="grey", linestyle=":", linewidth=0.9, label="break-even")
    ax3.set_title("Full update_state array path")
    ax3.set_xlabel("num_envs")
    ax3.set_ylabel("Speedup vs numpy")
    ax3.set_xscale("log", base=2)
    ax3.set_xticks(num_envs)
    ax3.set_xticklabels([str(value) for value in num_envs])
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=7)

    fig.tight_layout(rect=(0, 0, 1, 0.9))
    summary_path = output_dir / "g1_motion_tracking_numba_summary.png"
    fig.savefig(summary_path, dpi=160)
    plt.close(fig)
    print(f"Saved summary plot: {summary_path.resolve()}")
    return [str(summary_path.resolve())]


def write_report(
    *,
    output_dir: Path,
    records: list[BenchCase],
    component_records: list[ComponentCase],
    parity: dict[str, dict[str, float]],
    e2e_records: list[EndToEndCase],
    args: argparse.Namespace,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    device_info_line = get_device_info_line()
    plot_paths = save_plots(records, component_records, output_dir, device_info=device_info_line)
    meta = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "device_info": get_device_info_dict(),
        "iters": args.iters,
        "warmup": args.warmup,
        "profiles": args.profiles,
        "num_envs": args.num_envs,
        "threads": args.threads,
        "requested_threads": args.threads,
        "measured_threads": getattr(args, "measured_threads", None),
        "skipped_threads": getattr(args, "skipped_threads", None),
        "numba_max_threads": getattr(args, "numba_max_threads", None),
        "scope": (
            "G1 motion tracking reward+termination hot slice plus full update_state "
            "array-path reconciliation; synthetic arrays"
        ),
        "e2e_enabled": args.e2e,
        "e2e_num_envs": args.e2e_num_envs,
        "e2e_case": args.e2e_case,
        "e2e_warmup_steps": args.e2e_warmup_steps,
        "e2e_measure_steps": args.e2e_measure_steps,
        "e2e_numba_threads_source": "best sac_default hot-slice thread per num_env",
    }
    payload = {
        "meta": meta,
        "results": [_case_to_dict(record) for record in records],
        "component_results": [_component_case_to_dict(record) for record in component_records],
        "end_to_end_results": [_e2e_case_to_dict(record) for record in e2e_records],
        "parity": parity,
        "plots": plot_paths,
    }
    json_path = output_dir / "results.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    summary_lines = [
        "# G1 motion tracking Numba benchmark",
        "",
        "Scope: reward plus termination for `G1MotionTrackingEnv.update_state`, using",
        "deterministic synthetic arrays. The component table additionally measures the",
        "full array portion of `update_state`: relative transforms, reward, termination,",
        "and actor/critic observation assembly. Backend state getters, motion sampling,",
        "reset/refresh, policy inference, learner work, and replay remain out of scope.",
        "",
        "## Numba-specific hot slice",
        "",
        "Definitions:",
        "",
        "- `vs numpy`: numpy reward+termination time divided by numba reward+termination time.",
        "- `vs numba1T`: numba 1-thread time divided by numba N-thread time.",
        "- `parallel eff`: `vs numba1T / N`; this is the only parallel-efficiency column.",
        "",
        "Profile meanings:",
        "",
        "- `ppo_default`: PPO G1 motion-tracking owner-config reward scales.",
        "- `sac_default`: SAC G1 motion-tracking owner-config reward scales; this chooses",
        "  Numba threads for the collector A/B run.",
        "- `full_supported`: synthetic stress profile with every reward term supported by",
        "  `motion_tracking_numba.py` enabled.",
        "",
        "```text",
        _format_hot_summary_table(records),
        "```",
        "",
        "Detailed per-env and per-thread data is stored in `results.json`.",
    ]
    if component_records:
        summary_lines.extend(
            [
                "",
                "## Component Reconciliation",
                "",
                "This table measures the synthetic full array path of `update_state`:",
                "`_update_relative_transforms`, reward, termination, motion-anchor",
                "features, joint-relative features, and actor/critic observation assembly.",
                "It still excludes backend state getters, motion sampling, reset/refresh,",
                "state replacement, collector bookkeeping, and learner work.",
                "",
                "```text",
                _format_component_table(component_records),
                "```",
            ]
        )
    if plot_paths:
        summary_lines.extend(["", "## Plots", ""])
        for path in plot_paths:
            rel_path = Path(path).name
            title = rel_path.removesuffix(".png").replace("_", " ")
            summary_lines.append(f"![{title}]({rel_path})")
            summary_lines.append("")
    if e2e_records:
        summary_lines.extend(
            [
                "",
                "## End-to-end collector comparison",
                "",
                "This section mirrors `benchmark_offpolicy_collector_active.py`: Hydra owner",
                "config, `create_env`, actor action sampling, `env.step`, terminal-observation",
                "handling, replay writes, and collector bookkeeping are included. Learner",
                "updates are not run. The Numba variant uses the best `sac_default` hot-slice",
                "thread count found above for each `num_envs`.",
                "`other_ms` is the collector active step remainder after subtracting reported",
                "`physics_ms` and `update_state_ms`; if the backend does not report physics",
                "timing, only `update_state_ms` is subtracted.",
                "",
                "```text",
                _format_e2e_table(e2e_records),
                "```",
            ]
        )
        reconciliation_table = _format_e2e_reconciliation_table(
            hot_records=records,
            e2e_records=e2e_records,
            profile="sac_default",
        )
        if reconciliation_table:
            summary_lines.extend(
                [
                    "",
                    "## E2E Reconciliation",
                    "",
                    "This compares the synthetic `sac_default` hot-slice milliseconds saved",
                    "with the collector-measured `update_state_ms` milliseconds saved at the",
                    "same `num_envs` and Numba thread count. Large hot-slice speedups can",
                    "translate to small `update_state` speedups when non-accelerated work",
                    "dominates the update-state baseline.",
                    "",
                    "```text",
                    reconciliation_table,
                    "```",
                ]
            )
    else:
        summary_lines.extend(
            [
                "",
                "## End-to-end collector comparison",
                "",
                "Not run. Pass `--e2e` to add a real off-policy collector active-window A/B",
                f"comparison for `{args.e2e_case}` with `numba_acceleration=false/true`.",
                "This collector comparison does not run learner updates.",
            ]
        )
    parity_summary = _format_parity_summary_table(parity)
    summary_lines.extend(
        [
            "",
            "## Parity Summary",
            "",
            "Reward is checked with `rtol=1e-4, atol=1e-5`; termination is exact.",
            "```text",
            parity_summary,
            "```",
            "",
            "## Interpretation",
            "",
            "- `numba 1 thread` isolates fusion/codegen benefit over numpy reward functions.",
            "- Higher thread counts add row-parallel speedup over the same fused kernel.",
            "- The collector comparison excludes learner updates but includes collector-side",
            "  Hydra setup, env construction, actor sampling, `env.step`, replay writes, and",
            "  bookkeeping inside the measured active window.",
        ]
    )
    md_path = output_dir / "report.md"
    md_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(f"Saved JSON: {json_path.resolve()}")
    print(f"Saved report: {md_path.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profiles",
        nargs="+",
        default=["ppo_default", "sac_default", "full_supported"],
        choices=sorted(make_profile_specs()),
    )
    parser.add_argument("--num-envs", nargs="+", type=int, default=DEFAULT_NUM_ENVS)
    parser.add_argument("--threads", nargs="+", type=int, default=None)
    parser.add_argument("--iters", type=int, default=80)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--e2e",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run a real off-policy collector active-window baseline vs numba comparison.",
    )
    parser.add_argument("--e2e-num-envs", nargs="+", type=int, default=DEFAULT_E2E_NUM_ENVS)
    parser.add_argument("--e2e-case", default=DEFAULT_E2E_CASE)
    parser.add_argument("--e2e-warmup-steps", type=int, default=2)
    parser.add_argument("--e2e-measure-steps", type=int, default=8)
    parser.add_argument(
        "--e2e-numba-threads",
        type=int,
        default=None,
        help="Fallback Numba thread count when a requested e2e num_env lacks hot-slice data.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark/outputs/g1_motion_tracking_numba"),
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Short smoke run: sac_default at 512 and 2048 envs.",
    )
    args = parser.parse_args()
    if args.quick:
        args.profiles = ["sac_default"]
        args.num_envs = [512, 2048]
        args.e2e_num_envs = [1024, 2048]
        if args.threads is None:
            args.threads = QUICK_THREADS
        args.iters = 10
        args.warmup = 2
    elif args.threads is None:
        args.threads = DEFAULT_THREADS
    return args


def main() -> None:
    args = parse_args()
    specs = make_profile_specs()
    all_records: list[BenchCase] = []
    all_component_records: list[ComponentCase] = []
    e2e_records: list[EndToEndCase] = []
    parity: dict[str, dict[str, float]] = {}
    max_threads = get_num_threads()
    args.numba_max_threads = max_threads
    args.measured_threads = sorted({1, *(threads for threads in args.threads if threads <= max_threads)})
    args.skipped_threads = sorted({threads for threads in args.threads if threads > max_threads})

    print("=" * 80)
    print("G1 motion tracking Numba benchmark: reward + termination")
    print("=" * 80)
    print(f"host numba threads: {max_threads}")
    print(
        f"profiles={args.profiles} num_envs={args.num_envs} "
        f"requested_threads={args.threads} measured_threads={args.measured_threads}"
    )
    if args.skipped_threads:
        print(f"skipped threads above numba max: {args.skipped_threads}")
    for profile_name in args.profiles:
        spec = specs[profile_name]
        for num_envs in args.num_envs:
            records, component, parity_result = bench_one(
                profile=spec,
                num_envs=num_envs,
                thread_counts=args.threads,
                iters=args.iters,
                warmup=args.warmup,
                seed=args.seed,
            )
            all_records.extend(records)
            all_component_records.append(component)
            parity[f"{profile_name}:{num_envs}"] = parity_result
            print()
            print(_format_table(records))
            print()
            print(_format_component_table([component]))

    if args.e2e:
        missing_e2e_hot_slice = [
            num_envs
            for num_envs in args.e2e_num_envs
            if ("sac_default", num_envs) not in _best_numba_by_case(all_records)
        ]
        if missing_e2e_hot_slice:
            print()
            print("=" * 80)
            print("Completing sac_default hot-slice data for e2e thread selection")
            print("=" * 80)
            for num_envs in missing_e2e_hot_slice:
                records, component, parity_result = bench_one(
                    profile=specs["sac_default"],
                    num_envs=num_envs,
                    thread_counts=args.threads,
                    iters=args.iters,
                    warmup=args.warmup,
                    seed=args.seed,
                )
                all_records.extend(records)
                all_component_records.append(component)
                parity[f"sac_default:{num_envs}"] = parity_result
                print()
                print(_format_table(records))
                print()
                print(_format_component_table([component]))
        selected_threads = _best_threads_for_profile(
            all_records, profile="sac_default", num_envs=args.e2e_num_envs
        )
        e2e_records = _run_e2e_collector_sweep(
            case_name=args.e2e_case,
            num_envs=args.e2e_num_envs,
            warmup_steps=args.e2e_warmup_steps,
            measure_steps=args.e2e_measure_steps,
            selected_threads=selected_threads,
            fallback_numba_threads=args.e2e_numba_threads,
        )
        print()
        print(_format_e2e_table(e2e_records))

    write_report(
        output_dir=args.output_dir,
        records=all_records,
        component_records=all_component_records,
        parity=parity,
        e2e_records=e2e_records,
        args=args,
    )


if __name__ == "__main__":
    main()
