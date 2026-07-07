"""Benchmark the task-specific Numba backend for G1 joystick rewards.

This benchmark intentionally avoids MuJoCo/Motrix construction. It measures the
hot slice owned by ``src/unilab/envs/locomotion/g1/joystick.py``:

* baseline: real ``G1WalkEnv._compute_reward`` reward dispatch + termination;
* accelerated: ``G1WalkNumbaAccelerator.compute``.

Synthetic backend arrays keep the benchmark deterministic while still using the
same reward functions, reward config fields, sensor names, and accelerator entry
point as the task.

Run:
    uv run python -m benchmark.benchmark_g1_joystick_numba
    uv run python benchmark/benchmark_g1_joystick_numba.py
    uv run python -m benchmark.benchmark_g1_joystick_numba --quick
    uv run python -m benchmark.benchmark_g1_joystick_numba --quick --e2e
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
from typing import Any

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
from unilab.envs.locomotion.g1.joystick import (
    G1RewardConfig,
    G1WalkEnv,
    build_upper_body_pose_weights,
)
from unilab.envs.locomotion.g1.joystick_numba import G1WalkNumbaAccelerator

NUM_ACTION = 29
DEFAULT_THREADS = [2, 4, 8, 16, 32, 64]
QUICK_THREADS = [2, 4]
DEFAULT_NUM_ENVS = [512, 1024, 2048, 4096, 8192, 16384, 32768]
DEFAULT_E2E_NUM_ENVS = [1024, 2048, 4096, 8192, 16384, 32768]
DEFAULT_E2E_CASE = "sac/g1_walk_flat/motrixsim"
DEFAULT_POSE_WEIGHTS = [
    0.01,
    1.0,
    5.0,
    0.01,
    5.0,
    5.0,
    0.01,
    1.0,
    5.0,
    0.01,
    5.0,
    5.0,
    50.0,
    50.0,
    50.0,
    50.0,
    50.0,
    50.0,
    50.0,
    50.0,
    50.0,
    50.0,
    50.0,
    50.0,
    50.0,
    50.0,
    50.0,
    50.0,
    50.0,
]

PPO_SCALES = {
    "tracking_lin_vel": 2.0,
    "tracking_ang_vel": 0.2,
    "feet_phase": 1.0,
    "lin_vel_z": -1.0,
    "ang_vel_xy": -0.25,
    "base_height": -500.0,
    "orientation": -5.0,
    "action_rate": -0.01,
    "pose": -0.1,
}

SAC_SCALES = {
    "tracking_lin_vel": 2.0,
    "tracking_ang_vel": 1.5,
    "penalty_ang_vel_xy": -1.0,
    "penalty_orientation": -10.0,
    "penalty_action_rate": -4.0,
    "pose": -0.5,
    "penalty_feet_ori": -20.0,
    "feet_phase": 5.0,
    "alive": 10.0,
}

FULL_SUPPORTED_SCALES = {
    "tracking_lin_vel": 2.0,
    "tracking_ang_vel": 1.5,
    "forward_progress": 0.5,
    "under_speed": -0.5,
    "lin_vel_z": -1.0,
    "orientation": -5.0,
    "penalty_orientation": -10.0,
    "ang_vel_xy": -0.25,
    "penalty_ang_vel_xy": -1.0,
    "action_rate": -0.01,
    "penalty_action_rate": -4.0,
    "base_height": -500.0,
    "pose": -0.5,
    "upper_body_pose": -0.1,
    "penalty_close_feet_xy": -2.0,
    "penalty_feet_ori": -20.0,
    "feet_phase": 5.0,
    "feet_phase_contrast": 1.0,
    "feet_phase_contact": 1.0,
    "feet_double_stance": -0.2,
    "feet_air_time": 0.5,
    "alive": 10.0,
}


@dataclass(frozen=True)
class ProfileSpec:
    name: str
    scales: dict[str, float]
    reward_cfg: G1RewardConfig


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
class SyntheticBackend:
    base_pos: np.ndarray
    sensors: dict[str, np.ndarray] = field(default_factory=dict)

    def get_base_pos(self) -> np.ndarray:
        return self.base_pos

    def get_sensor_data(self, name: str) -> np.ndarray:
        return self.sensors[name]


@dataclass
class SyntheticBatch:
    env: G1WalkEnv
    info: dict[str, Any]
    linvel: np.ndarray
    gyro: np.ndarray
    gravity: np.ndarray
    dof_pos: np.ndarray
    dof_vel: np.ndarray


class _Cfg:
    ctrl_dt = 0.02


def make_profile_specs() -> dict[str, ProfileSpec]:
    return {
        "ppo_default": ProfileSpec(
            name="ppo_default",
            scales=PPO_SCALES,
            reward_cfg=G1RewardConfig(
                scales=PPO_SCALES,
                tracking_sigma=0.25,
                gait_frequency=1.5,
                feet_phase_swing_height=0.09,
                feet_phase_tracking_sigma=0.008,
                base_height_target=0.754,
                min_base_height=0.55,
                max_tilt_deg=25.0,
                pose_weights=DEFAULT_POSE_WEIGHTS,
            ),
        ),
        "sac_default": ProfileSpec(
            name="sac_default",
            scales=SAC_SCALES,
            reward_cfg=G1RewardConfig(
                scales=SAC_SCALES,
                tracking_sigma=0.25,
                gait_frequency=1.5,
                feet_phase_swing_height=0.09,
                feet_phase_tracking_sigma=0.04,
                base_height_target=0.754,
                min_base_height=0.3,
                max_tilt_deg=65.0,
                close_feet_threshold=0.15,
                pose_weights=DEFAULT_POSE_WEIGHTS,
            ),
        ),
        "full_supported": ProfileSpec(
            name="full_supported",
            scales=FULL_SUPPORTED_SCALES,
            reward_cfg=G1RewardConfig(
                scales=FULL_SUPPORTED_SCALES,
                tracking_sigma=0.25,
                gait_frequency=1.5,
                feet_phase_swing_height=0.09,
                feet_phase_tracking_sigma=0.04,
                base_height_target=0.754,
                min_base_height=0.3,
                max_tilt_deg=65.0,
                min_forward_speed_for_gait_reward=0.0,
                close_feet_threshold=0.15,
                pose_weights=DEFAULT_POSE_WEIGHTS,
            ),
        ),
    }


def _make_fake_env(
    num_envs: int, reward_cfg: G1RewardConfig, backend: SyntheticBackend
) -> G1WalkEnv:
    env = object.__new__(G1WalkEnv)
    env._num_envs = num_envs
    env._num_action = NUM_ACTION
    env._cfg = _Cfg()
    env._reward_cfg = reward_cfg
    env._backend = backend
    env._enable_reward_log = True
    env.default_angles = np.zeros((NUM_ACTION,), dtype=get_global_dtype())
    env._pose_weights = np.asarray(reward_cfg.pose_weights, dtype=get_global_dtype())
    env._upper_body_pose_weights = build_upper_body_pose_weights(reward_cfg.pose_weights)
    env._init_reward_functions()
    return env


def make_batch(num_envs: int, spec: ProfileSpec, seed: int) -> SyntheticBatch:
    rng = np.random.default_rng(seed)
    dtype = get_global_dtype()

    linvel = rng.normal(loc=(0.8, 0.0, 0.0), scale=(0.25, 0.08, 0.08), size=(num_envs, 3)).astype(
        dtype
    )
    gyro = rng.normal(loc=(0.0, 0.0, 0.1), scale=(0.08, 0.08, 0.15), size=(num_envs, 3)).astype(
        dtype
    )
    gravity = rng.normal(loc=(0.0, 0.0, 0.98), scale=(0.02, 0.02, 0.01), size=(num_envs, 3))
    gravity[:, 2] = np.clip(gravity[:, 2], -1.0, 1.0)
    gravity = gravity.astype(dtype)
    dof_pos = rng.normal(loc=0.0, scale=0.03, size=(num_envs, NUM_ACTION)).astype(dtype)
    dof_vel = rng.normal(loc=0.0, scale=0.2, size=(num_envs, NUM_ACTION)).astype(dtype)
    current_actions = rng.normal(loc=0.0, scale=0.2, size=(num_envs, NUM_ACTION)).astype(dtype)
    last_actions = (current_actions + rng.normal(0.0, 0.05, size=current_actions.shape)).astype(
        dtype
    )
    commands = rng.normal(loc=(0.8, 0.0, 0.1), scale=(0.2, 0.05, 0.1), size=(num_envs, 3)).astype(
        dtype
    )
    commands[:, 0] = np.maximum(commands[:, 0], 0.05)
    gait_phase = rng.uniform(0.0, 2.0 * np.pi, size=(num_envs, 2)).astype(dtype)
    feet_air_time = rng.uniform(0.0, 0.8, size=(num_envs, 2)).astype(dtype)

    base_pos = np.zeros((num_envs, 3), dtype=dtype)
    base_pos[:, 2] = spec.reward_cfg.base_height_target + rng.normal(0.0, 0.015, size=num_envs)
    left_foot_pos = np.column_stack(
        [
            np.full(num_envs, -0.1, dtype=dtype),
            rng.normal(-0.06, 0.01, size=num_envs),
            rng.uniform(0.0, 0.1, size=num_envs),
        ]
    ).astype(dtype)
    right_foot_pos = np.column_stack(
        [
            np.full(num_envs, 0.1, dtype=dtype),
            rng.normal(0.06, 0.01, size=num_envs),
            rng.uniform(0.0, 0.1, size=num_envs),
        ]
    ).astype(dtype)
    left_foot_quat = np.tile(np.array([1.0, 0.01, 0.02, 0.0], dtype=dtype), (num_envs, 1))
    right_foot_quat = np.tile(np.array([1.0, 0.02, 0.01, 0.0], dtype=dtype), (num_envs, 1))

    sensors = {
        "upvector": gravity,
        "left_foot_pos": left_foot_pos,
        "right_foot_pos": right_foot_pos,
        "left_foot_quat": left_foot_quat,
        "right_foot_quat": right_foot_quat,
    }
    for side in ("left", "right"):
        for idx in range(4):
            sensors[f"{side}_foot_contact_{idx}"] = (rng.random(num_envs) > 0.45).astype(dtype)
    backend = SyntheticBackend(base_pos=base_pos, sensors=sensors)
    env = _make_fake_env(num_envs, spec.reward_cfg, backend)
    info = {
        "steps": np.zeros((num_envs,), dtype=np.uint32),
        "commands": commands,
        "current_actions": current_actions,
        "last_actions": last_actions,
        "gait_phase": gait_phase,
        "feet_air_time": feet_air_time,
        "log": {},
    }
    return SyntheticBatch(
        env=env,
        info=info,
        linvel=linvel,
        gyro=gyro,
        gravity=gravity,
        dof_pos=dof_pos,
        dof_vel=dof_vel,
    )


def compute_numpy(batch: SyntheticBatch) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    env = batch.env
    info = {**batch.info, "log": {}}
    max_tilt_rad = np.deg2rad(env._reward_cfg.max_tilt_deg)
    tilt = np.arccos(np.clip(batch.gravity[:, 2], -1.0, 1.0))
    terminated = np.logical_or(
        tilt > max_tilt_rad,
        env._backend.get_base_pos()[:, 2] < env._reward_cfg.min_base_height,
    )
    reward = env._compute_reward(
        info,
        batch.linvel,
        batch.gyro,
        batch.gravity,
        batch.dof_pos,
        batch.dof_vel,
    )
    return reward, terminated, info.get("log", {})


def compute_numba(
    batch: SyntheticBatch, accelerator: G1WalkNumbaAccelerator
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    info = {**batch.info, "log": {}}
    out = accelerator.compute(
        env=batch.env,
        info=info,
        linvel=batch.linvel,
        gyro=batch.gyro,
        gravity=batch.gravity,
        dof_pos=batch.dof_pos,
        dof_vel=batch.dof_vel,
        scales=batch.env._reward_cfg.scales,
        enable_log=True,
    )
    return out.reward, out.terminated, out.log


def time_call(fn, *, iters: int, warmup: int) -> tuple[float, float, float]:
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1e3)
    return mean(samples), min(samples), stdev(samples) if len(samples) > 1 else 0.0


def check_parity(batch: SyntheticBatch, accelerator: G1WalkNumbaAccelerator) -> dict[str, float]:
    reward_np, terminated_np, log_np = compute_numpy(batch)
    reward_nb, terminated_nb, log_nb = compute_numba(batch, accelerator)
    np.testing.assert_allclose(reward_nb, reward_np, rtol=5e-5, atol=5e-5)
    np.testing.assert_array_equal(terminated_nb, terminated_np)
    for key, value in log_np.items():
        if key in log_nb:
            np.testing.assert_allclose(log_nb[key], value, rtol=5e-5, atol=5e-5)
    return {
        "max_abs_reward_diff": float(np.max(np.abs(reward_nb - reward_np))),
        "termination_mismatch": float(np.count_nonzero(terminated_nb != terminated_np)),
    }


def bench_one(
    *,
    profile: ProfileSpec,
    num_envs: int,
    thread_counts: list[int],
    iters: int,
    warmup: int,
    seed: int,
) -> tuple[list[BenchCase], dict[str, float]]:
    batch = make_batch(num_envs, profile, seed)
    max_threads = get_num_threads()

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

    compile_driver = G1WalkNumbaAccelerator.from_env(batch.env, num_threads=1)
    t0 = time.perf_counter()
    compute_numba(batch, compile_driver)
    compile_ms = (time.perf_counter() - t0) * 1e3
    parity = check_parity(batch, compile_driver)

    for threads in [1, *thread_counts]:
        if threads > max_threads:
            continue
        accelerator = G1WalkNumbaAccelerator.from_env(batch.env, num_threads=threads)
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
    return records, parity


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
    """Run a real collector active-window A/B test using training construction paths.

    This mirrors benchmark_offpolicy_collector_active.py: Hydra owner config,
    create_env, actor action sampling, env.step, terminal-observation handling,
    replay writes, and bookkeeping are all included. Learner updates are excluded.
    It is intentionally optional because it constructs a real MuJoCo env and is
    much heavier than the synthetic reward+termination hot-slice benchmark above.
    """
    from benchmark.benchmark_offpolicy_collector_active import _build_and_run_case

    common = {
        "warmup_steps": warmup_steps,
        "measure_steps": measure_steps,
        "replay_capacity_steps": max(2, measure_steps + warmup_steps + 1),
        "num_envs": num_envs,
    }
    variants = [
        (
            "training_collector_numpy",
            False,
            [
                "++env.numba_acceleration=false",
            ],
        ),
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
    rows = []
    for r in records:
        rows.append(
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
        )
    widths = [len(h) for h in headers]
    for row in rows:
        for idx, value in enumerate(row):
            widths[idx] = max(widths[idx], len(value))

    def fmt(values: list[str]) -> str:
        return " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(values))

    lines = [fmt(headers), "-+-".join("-" * width for width in widths)]
    lines.extend(fmt(row) for row in rows)
    return "\n".join(lines)


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

    lines = [fmt(headers), "-+-".join("-" * width for width in widths)]
    lines.extend(fmt(row) for row in rows)
    return "\n".join(lines)


def _best_numba_by_case(records: list[BenchCase]) -> dict[tuple[str, int], BenchCase]:
    best_by_case: dict[tuple[str, int], BenchCase] = {}
    for record in records:
        if record.path != "numba_accelerator":
            continue
        key = (record.profile, record.num_envs)
        if key not in best_by_case or record.mean_ms < best_by_case[key].mean_ms:
            best_by_case[key] = record
    return best_by_case


def save_plots(
    records: list[BenchCase],
    e2e_records: list[EndToEndCase],
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
    fig.suptitle(f"G1 joystick Numba reward+termination benchmark\n{device_info}", fontsize=13)

    ax1 = axes[0]
    for profile in profiles:
        x = []
        y = []
        labels = []
        for env_count in num_envs:
            best = best_by_case.get((profile, env_count))
            if best is None:
                continue
            x.append(env_count)
            y.append(best.speedup_vs_numpy)
            labels.append(best.threads)
        if not x:
            continue
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

    ax2 = axes[1]
    for profile in profiles:
        best_subset = [
            best_by_case[(profile, env_count)]
            for env_count in num_envs
            if (profile, env_count) in best_by_case
        ]
        if best_subset:
            ax2.plot(
                [record.num_envs for record in best_subset],
                [
                    0.0
                    if record.parallel_speedup_vs_numba_1t is None
                    else record.parallel_speedup_vs_numba_1t
                    for record in best_subset
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

    ax3 = axes[2]
    e2e_numba = sorted(
        [
            record
            for record in e2e_records
            if record.numba_acceleration and record.update_state_speedup_vs_numpy is not None
        ],
        key=lambda record: record.num_envs,
    )
    if e2e_numba:
        ax3.plot(
            [record.num_envs for record in e2e_numba],
            [record.update_state_speedup_vs_numpy or 0.0 for record in e2e_numba],
            marker="o",
            linestyle="-",
            label="collector update_state",
        )
        for record in e2e_numba:
            ax3.annotate(
                "-" if record.numba_threads is None else f"{record.numba_threads}T",
                xy=(record.num_envs, record.update_state_speedup_vs_numpy or 0.0),
                xytext=(0, 6),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )
        ax3.set_title("Collector update_state speedup")
        ax3.set_ylabel("Speedup vs numpy collector")
    else:
        for profile in profiles:
            numpy_subset = sorted(
                [
                    record
                    for record in records
                    if record.profile == profile and record.path == "numpy_dispatch"
                ],
                key=lambda record: record.num_envs,
            )
            best_subset = [
                best_by_case[(profile, env_count)]
                for env_count in num_envs
                if (profile, env_count) in best_by_case
            ]
            if numpy_subset:
                ax3.plot(
                    [record.num_envs for record in numpy_subset],
                    [record.mean_ms for record in numpy_subset],
                    marker="o",
                    linestyle="--",
                    label=f"{profile} numpy",
                )
            if best_subset:
                ax3.plot(
                    [record.num_envs for record in best_subset],
                    [record.mean_ms for record in best_subset],
                    marker="s",
                    linestyle="-",
                    label=f"{profile} numba best",
                )
        ax3.set_title("Latency: numpy vs best numba")
        ax3.set_ylabel("Mean latency (ms)")
        ax3.set_yscale("log")
    ax3.axhline(1.0, color="grey", linestyle=":", linewidth=0.9, label="break-even")
    ax3.set_xlabel("num_envs")
    ax3.set_xscale("log", base=2)
    ax3.set_xticks(num_envs)
    ax3.set_xticklabels([str(value) for value in num_envs])
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=7)

    fig.tight_layout(rect=(0, 0, 1, 0.9))
    summary_path = output_dir / "g1_joystick_numba_summary.png"
    fig.savefig(summary_path, dpi=160)
    plt.close(fig)
    print(f"Saved summary plot: {summary_path.resolve()}")
    return [str(summary_path.resolve())]


def write_report(
    *,
    output_dir: Path,
    records: list[BenchCase],
    parity: dict[str, dict[str, float]],
    e2e_records: list[EndToEndCase],
    args: argparse.Namespace,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    device_info_line = get_device_info_line()
    plot_paths = save_plots(records, e2e_records, output_dir, device_info=device_info_line)
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
        "scope": "G1 joystick reward+termination hot slice; synthetic backend arrays",
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
        "end_to_end_results": [_e2e_case_to_dict(record) for record in e2e_records],
        "parity": parity,
        "plots": plot_paths,
    }
    json_path = output_dir / "results.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    best_by_case = _best_numba_by_case(records)

    summary_lines = [
        "# G1 joystick Numba benchmark",
        "",
        "Scope: reward dispatch plus termination for `G1WalkEnv.update_state`, using",
        "deterministic synthetic backend arrays. Physics stepping, obs assembly, reset,",
        "and policy inference are intentionally out of scope.",
        "",
        "## Numba-specific hot slice",
        "",
        "Definitions:",
        "",
        "- `vs numpy`: numpy reward+termination time divided by numba reward+termination time.",
        "- `vs numba1T`: numba 1-thread time divided by numba N-thread time.",
        "- `parallel eff`: `vs numba1T / N`; this is the only parallel-efficiency column.",
        "",
        "This section measures only the part this task-specific Numba backend accelerates:",
        "`G1WalkEnv` reward dispatch plus termination inside `update_state`. It excludes",
        "physics, observation assembly, reset/RNG, policy inference, learner work, and replay.",
        "",
        "Profile meanings:",
        "",
        "- `ppo_default`: reward terms and thresholds matching the PPO G1 walk-flat style.",
        "- `sac_default`: reward terms and thresholds matching the SAC G1 walk-flat owner config;",
        "  this is also the source used to choose Numba threads for the collector A/B run.",
        "- `full_supported`: synthetic stress profile with every reward term currently supported",
        "  by `joystick_numba.py` enabled, used to estimate the upper bound when reward work is",
        "  heavier than the default training configs.",
        "",
    ]
    for key in sorted(best_by_case):
        best = best_by_case[key]
        summary_lines.append(
            f"- `{best.profile}` / {best.num_envs} envs: best {best.mean_ms:.3f} ms "
            f"at {best.threads} threads, {best.speedup_vs_numpy:.2f}x vs numpy dispatch "
            f"({best.env_per_s / 1e6:.2f}M env/s)."
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
                "This section mirrors `benchmark_offpolicy_collector_active.py`: Hydra owner config,",
                "`create_env`, actor action sampling, `env.step`, terminal-observation handling,",
                "replay writes, and collector-side bookkeeping are included. Learner updates are",
                "not run. The Numba variant uses the best `sac_default` hot-slice thread count",
                "found above for each `num_envs`.",
                "`other_ms` is the collector active step remainder after subtracting reported",
                "`physics_ms` and `update_state_ms`; if the backend does not report physics timing,",
                "only `update_state_ms` is subtracted.",
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
                    "same `num_envs` and Numba thread count. The hot slice is only the",
                    "reward+termination work replaced by `joystick_numba.py`; collector",
                    "`update_state_ms` also includes state reads, observation assembly,",
                    "resets, state replacement, and bookkeeping around that hot slice.",
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
    summary_lines.extend(
        [
            "",
            "## Detailed Results",
            "",
            "```text",
            _format_table(records),
            "```",
            "",
            "## Parity",
            "",
            "Reward is checked with `rtol=5e-5, atol=5e-5`; termination is exact.",
            "",
            "```json",
            json.dumps(parity, indent=2),
            "```",
            "",
            "## Interpretation",
            "",
            "- `numba 1 thread` isolates fusion/codegen benefit over Python reward dispatch.",
            "- Higher thread counts add row-parallel speedup over the same fused kernel.",
            "- Hot-slice speedup is an upper bound for collector speedup because collector timing",
            "  also includes backend state reads, observation assembly, reset/RNG, policy inference,",
            "  replay, and bookkeeping.",
            "- The optional collector comparison still excludes learner updates.",
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
    parser.add_argument(
        "--num-envs",
        nargs="+",
        type=int,
        default=DEFAULT_NUM_ENVS,
    )
    parser.add_argument("--threads", nargs="+", type=int, default=None)
    parser.add_argument("--iters", type=int, default=80)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--e2e",
        action="store_true",
        help="Also run a real off-policy collector active-window baseline vs numba comparison.",
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
        default=Path("benchmark/outputs/g1_joystick_numba"),
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
    e2e_records: list[EndToEndCase] = []
    parity: dict[str, dict[str, float]] = {}
    max_threads = get_num_threads()
    args.numba_max_threads = max_threads
    args.measured_threads = sorted({1, *(threads for threads in args.threads if threads <= max_threads)})
    args.skipped_threads = sorted({threads for threads in args.threads if threads > max_threads})

    print("=" * 80)
    print("G1 joystick Numba benchmark: reward dispatch + termination")
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
            records, parity_result = bench_one(
                profile=spec,
                num_envs=num_envs,
                thread_counts=args.threads,
                iters=args.iters,
                warmup=args.warmup,
                seed=args.seed,
            )
            all_records.extend(records)
            parity[f"{profile_name}:{num_envs}"] = parity_result
            print()
            print(_format_table(records))

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
            spec = specs["sac_default"]
            for num_envs in missing_e2e_hot_slice:
                records, parity_result = bench_one(
                    profile=spec,
                    num_envs=num_envs,
                    thread_counts=args.threads,
                    iters=args.iters,
                    warmup=args.warmup,
                    seed=args.seed,
                )
                all_records.extend(records)
                parity[f"sac_default:{num_envs}"] = parity_result
                print()
                print(_format_table(records))

        selected_threads = _best_threads_for_profile(
            all_records, profile="sac_default", num_envs=args.e2e_num_envs
        )
        print()
        print("=" * 80)
        print("End-to-end collector comparison: off-policy collector active window")
        print("=" * 80)
        print(f"e2e case={args.e2e_case}")
        print(f"e2e num_envs={args.e2e_num_envs}")
        print(f"e2e numba threads from sac_default hot-slice best: {selected_threads}")
        e2e_records = _run_e2e_collector_sweep(
            case_name=args.e2e_case,
            num_envs=args.e2e_num_envs,
            warmup_steps=args.e2e_warmup_steps,
            measure_steps=args.e2e_measure_steps,
            selected_threads=selected_threads,
            fallback_numba_threads=args.e2e_numba_threads,
        )
        print(_format_e2e_table(e2e_records))

    write_report(
        output_dir=args.output_dir,
        records=all_records,
        parity=parity,
        e2e_records=e2e_records,
        args=args,
    )


if __name__ == "__main__":
    main()
