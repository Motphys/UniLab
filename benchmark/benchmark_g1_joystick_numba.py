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
    }


def _format_table(records: list[BenchCase]) -> str:
    headers = [
        "profile",
        "envs",
        "path",
        "threads",
        "mean_ms",
        "min_ms",
        "speedup",
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


def _best_numba_by_case(records: list[BenchCase]) -> dict[tuple[str, int], BenchCase]:
    best_by_case: dict[tuple[str, int], BenchCase] = {}
    for record in records:
        if record.path != "numba_accelerator":
            continue
        key = (record.profile, record.num_envs)
        if key not in best_by_case or record.mean_ms < best_by_case[key].mean_ms:
            best_by_case[key] = record
    return best_by_case


def save_plots(records: list[BenchCase], output_dir: Path, *, device_info: str) -> list[str]:
    if plt is None or not records:
        print("Plotting skipped: matplotlib is not available.")
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    profiles = sorted({record.profile for record in records})
    num_envs = sorted({record.num_envs for record in records})
    best_by_case = _best_numba_by_case(records)

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(21, 6))
    fig.suptitle(f"G1 joystick Numba reward+termination benchmark\n{device_info}", fontsize=13)

    # 1) Best speedup vs env count.
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
    ax1.set_title("Best speedup vs numpy dispatch")
    ax1.set_xlabel("num_envs")
    ax1.set_ylabel("Speedup")
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(num_envs)
    ax1.set_xticklabels([str(value) for value in num_envs])
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8)

    # 2) Latency: numpy dispatch vs best numba.
    ax2 = axes[1]
    for profile in profiles:
        numpy_subset = sorted(
            [
                record
                for record in records
                if record.profile == profile and record.path == "numpy_dispatch"
            ],
            key=lambda record: record.num_envs,
        )
        if numpy_subset:
            ax2.plot(
                [record.num_envs for record in numpy_subset],
                [record.mean_ms for record in numpy_subset],
                marker="o",
                linestyle="--",
                label=f"{profile} numpy",
            )
        best_subset = [
            best_by_case[(profile, env_count)]
            for env_count in num_envs
            if (profile, env_count) in best_by_case
        ]
        if best_subset:
            ax2.plot(
                [record.num_envs for record in best_subset],
                [record.mean_ms for record in best_subset],
                marker="s",
                linestyle="-",
                label=f"{profile} numba best",
            )
    ax2.set_title("Latency: numpy dispatch vs best numba")
    ax2.set_xlabel("num_envs")
    ax2.set_ylabel("Mean latency (ms)")
    ax2.set_xscale("log", base=2)
    ax2.set_yscale("log")
    ax2.set_xticks(num_envs)
    ax2.set_xticklabels([str(value) for value in num_envs])
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=7)

    # 3) Throughput: numpy dispatch vs best numba.
    ax3 = axes[2]
    for profile in profiles:
        numpy_subset = sorted(
            [
                record
                for record in records
                if record.profile == profile and record.path == "numpy_dispatch"
            ],
            key=lambda record: record.num_envs,
        )
        if numpy_subset:
            ax3.plot(
                [record.num_envs for record in numpy_subset],
                [record.env_per_s / 1e6 for record in numpy_subset],
                marker="o",
                linestyle="--",
                label=f"{profile} numpy",
            )
        best_subset = [
            best_by_case[(profile, env_count)]
            for env_count in num_envs
            if (profile, env_count) in best_by_case
        ]
        if best_subset:
            ax3.plot(
                [record.num_envs for record in best_subset],
                [record.env_per_s / 1e6 for record in best_subset],
                marker="s",
                linestyle="-",
                label=f"{profile} numba best",
            )
    ax3.set_title("Throughput: numpy dispatch vs best numba")
    ax3.set_xlabel("num_envs")
    ax3.set_ylabel("Throughput (M env/s)")
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
    args: argparse.Namespace,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    device_info_line = get_device_info_line()
    plot_paths = save_plots(records, output_dir, device_info=device_info_line)
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
        "scope": "G1 joystick reward+termination hot slice; synthetic backend arrays",
    }
    payload = {
        "meta": meta,
        "results": [_case_to_dict(record) for record in records],
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
        "## Summary",
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
            "- End-to-end training speedup will be lower because this benchmark excludes physics,",
            "  obs assembly, reset/RNG, policy inference, and learner work.",
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
    parser.add_argument("--num-envs", nargs="+", type=int, default=[512, 2048, 8192])
    parser.add_argument("--threads", nargs="+", type=int, default=[2, 4, 8, 16])
    parser.add_argument("--iters", type=int, default=80)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
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
        args.threads = [2, 4]
        args.iters = 10
        args.warmup = 2
    return args


def main() -> None:
    args = parse_args()
    specs = make_profile_specs()
    all_records: list[BenchCase] = []
    parity: dict[str, dict[str, float]] = {}

    print("=" * 80)
    print("G1 joystick Numba benchmark: reward dispatch + termination")
    print("=" * 80)
    print(f"host numba threads: {get_num_threads()}")
    print(f"profiles={args.profiles} num_envs={args.num_envs} threads={args.threads}")
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

    write_report(output_dir=args.output_dir, records=all_records, parity=parity, args=args)


if __name__ == "__main__":
    main()
