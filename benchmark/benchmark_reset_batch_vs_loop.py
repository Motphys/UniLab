#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import pkgutil
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from unilab.envs import registry


DEFAULT_TASKS = [
    "Go1JoystickFlatTerrain",
    "Go2JoystickFlatTerrain",
    "G1JoystickFlatTerrain",
]


@dataclass
class ResetBenchRecord:
    task: str
    env_num: int
    method: str
    masks_replayed: int
    total_reset_envs: int
    elapsed_sec: float
    ms_per_replay: float
    us_per_reset_env: float
    sampled_done_ratio: float


def ensure_locomotion_envs_registered() -> None:
    # Keep registration flow aligned with training scripts.
    package = importlib.import_module("unilab.envs.locomotion")
    if hasattr(package, "__path__"):
        for _, name, _ in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
            try:
                importlib.import_module(name)
            except Exception:
                # Skip broken optional modules to keep benchmark robust.
                continue


def parse_tasks(raw: str) -> List[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def merge_reset_info(dst: dict, new_values: dict, mask: np.ndarray) -> None:
    if not new_values:
        return
    for key, value in new_values.items():
        if key not in dst:
            dst[key] = value
            continue
        if isinstance(value, np.ndarray):
            dst[key][mask] = value
        elif isinstance(value, dict):
            if not isinstance(dst[key], dict):
                dst[key] = {}
            merge_reset_info(dst[key], value, mask)


def estimate_termination_prob(
    task: str,
    env_num: int,
    rng: np.random.Generator,
    warmup_steps: int,
    measure_steps: int,
) -> float:
    env = registry.make(task, num_envs=env_num, sim_backend="mujoco")
    try:
        if env.state is None:
            env.init_state()
        env.reset(np.arange(env_num, dtype=np.int64))

        action_low = env.action_space.low.astype(np.float32)
        action_high = env.action_space.high.astype(np.float32)
        action_dim = env.action_space.shape[0]

        term_ratio_sum = 0.0
        measured_steps = 0

        total_steps = warmup_steps + measure_steps
        for step_i in range(total_steps):
            actions = rng.uniform(action_low, action_high, size=(env_num, action_dim)).astype(np.float32, copy=False)
            state = env.step(actions)
            if step_i >= warmup_steps:
                term_ratio_sum += float(np.asarray(state.terminated, dtype=bool).mean())
                measured_steps += 1
        return term_ratio_sum / max(measured_steps, 1)
    finally:
        env.close()


def build_training_like_done_masks(
    env_num: int,
    target_masks: int,
    max_episode_steps: int,
    termination_prob: float,
    rng: np.random.Generator,
) -> tuple[list[np.ndarray], float]:
    # Match init_at_random_ep_len=True behavior in training.
    episode_steps = rng.integers(0, max_episode_steps, size=(env_num,), endpoint=False, dtype=np.int64)
    done_masks: list[np.ndarray] = []
    done_ratio_sum = 0.0
    generated_steps = 0
    max_generated_steps = max(target_masks * 50, max_episode_steps * 2)

    for _ in range(max_generated_steps):
        episode_steps += 1
        terminated = rng.random(env_num) < termination_prob
        truncated = episode_steps >= max_episode_steps
        done = np.logical_or(terminated, truncated)
        done_ratio_sum += float(done.mean())
        generated_steps += 1
        if np.any(done):
            done_masks.append(done.copy())
            episode_steps[done] = 0
            if len(done_masks) >= target_masks:
                break

    if not done_masks:
        fallback = np.zeros((env_num,), dtype=bool)
        fallback[rng.integers(0, env_num)] = True
        done_masks = [fallback]

    sampled_done_ratio = done_ratio_sum / max(generated_steps, 1)
    return done_masks, sampled_done_ratio


def setup_env_for_replay(task: str, env_num: int) -> object:
    env = registry.make(task, num_envs=env_num, sim_backend="mujoco")
    if env.state is None:
        env.init_state()
    env.reset(np.arange(env_num, dtype=np.int64))
    return env


def bench_batch_reset(env: object, done_masks: List[np.ndarray]) -> tuple[float, int, int]:
    state = env.state
    replayed = 0
    total_reset_envs = 0
    t0 = time.perf_counter()
    for done in done_masks:
        if not np.any(done):
            continue
        replayed += 1
        np.putmask(state.info["steps"], done, 0)
        idx = np.flatnonzero(done)
        new_physics, new_obs, info1 = env.reset(idx)
        state.physics_state[idx] = new_physics
        if new_obs is not None:
            state.obs[idx] = new_obs
        merge_reset_info(state.info, info1, done)
        total_reset_envs += int(idx.size)
    elapsed = time.perf_counter() - t0
    return elapsed, replayed, total_reset_envs


def bench_loop_reset(env: object, done_masks: List[np.ndarray]) -> tuple[float, int, int]:
    state = env.state
    replayed = 0
    total_reset_envs = 0
    t0 = time.perf_counter()
    for done in done_masks:
        if not np.any(done):
            continue
        replayed += 1
        np.putmask(state.info["steps"], done, 0)
        idx = np.flatnonzero(done)
        for one_idx in idx:
            single = np.array([one_idx], dtype=np.int64)
            single_mask = np.zeros_like(done, dtype=bool)
            single_mask[one_idx] = True
            new_physics, new_obs, info1 = env.reset(single)
            state.physics_state[one_idx] = new_physics[0]
            if new_obs is not None:
                state.obs[one_idx] = new_obs[0]
            merge_reset_info(state.info, info1, single_mask)
        total_reset_envs += int(idx.size)
    elapsed = time.perf_counter() - t0
    return elapsed, replayed, total_reset_envs


def run_one_setting(
    task: str,
    env_num: int,
    repeats: int,
    target_masks: int,
    max_collect_steps: int,
    warmup_steps: int,
    seed: int,
) -> List[ResetBenchRecord]:
    calib_rng = np.random.default_rng(seed + env_num + 17)
    termination_prob = estimate_termination_prob(
        task=task,
        env_num=env_num,
        rng=calib_rng,
        warmup_steps=warmup_steps,
        measure_steps=max_collect_steps,
    )

    tmp_env = setup_env_for_replay(task, env_num)
    try:
        max_episode_steps = int(tmp_env.cfg.max_episode_steps or 1000)
    finally:
        tmp_env.close()

    mask_rng = np.random.default_rng(seed + env_num + 29)
    done_masks, sampled_done_ratio = build_training_like_done_masks(
        env_num=env_num,
        target_masks=target_masks,
        max_episode_steps=max_episode_steps,
        termination_prob=termination_prob,
        rng=mask_rng,
    )
    # Reuse identical masks across methods and repeats.
    frozen_masks = [mask.copy() for mask in done_masks]

    method_to_elapsed: Dict[str, List[float]] = {"batch": [], "for_loop": []}
    method_to_replay: Dict[str, List[int]] = {"batch": [], "for_loop": []}
    method_to_total_envs: Dict[str, List[int]] = {"batch": [], "for_loop": []}

    for _ in range(repeats):
        np.random.seed(seed)
        env_batch = setup_env_for_replay(task, env_num)
        try:
            elapsed, replayed, total_reset_envs = bench_batch_reset(env_batch, frozen_masks)
        finally:
            env_batch.close()
        method_to_elapsed["batch"].append(elapsed)
        method_to_replay["batch"].append(replayed)
        method_to_total_envs["batch"].append(total_reset_envs)

        np.random.seed(seed)
        env_loop = setup_env_for_replay(task, env_num)
        try:
            elapsed, replayed, total_reset_envs = bench_loop_reset(env_loop, frozen_masks)
        finally:
            env_loop.close()
        method_to_elapsed["for_loop"].append(elapsed)
        method_to_replay["for_loop"].append(replayed)
        method_to_total_envs["for_loop"].append(total_reset_envs)

    records: List[ResetBenchRecord] = []
    for method in ("batch", "for_loop"):
        elapsed_med = float(np.median(np.asarray(method_to_elapsed[method], dtype=np.float64)))
        replay_med = int(np.median(np.asarray(method_to_replay[method], dtype=np.float64)))
        total_envs_med = int(np.median(np.asarray(method_to_total_envs[method], dtype=np.float64)))
        ms_per_replay = (elapsed_med * 1000.0 / max(replay_med, 1))
        us_per_env = (elapsed_med * 1e6 / max(total_envs_med, 1))
        records.append(
            ResetBenchRecord(
                task=task,
                env_num=env_num,
                method=method,
                masks_replayed=replay_med,
                total_reset_envs=total_envs_med,
                elapsed_sec=elapsed_med,
                ms_per_replay=ms_per_replay,
                us_per_reset_env=us_per_env,
                sampled_done_ratio=sampled_done_ratio,
            )
        )
    return records


def plot_grouped_thin_bars(records: List[ResetBenchRecord], out_png: Path) -> None:
    tasks = sorted({r.task for r in records})
    env_nums = sorted({r.env_num for r in records})
    methods = ["batch", "for_loop"]
    series_order = [(task, method) for task in tasks for method in methods]
    labels = [f"{task}-{method}" for task, method in series_order]
    color_map = {
        (tasks[0], "batch"): "#3B82F6",
        (tasks[0], "for_loop"): "#93C5FD",
        (tasks[1], "batch"): "#10B981",
        (tasks[1], "for_loop"): "#6EE7B7",
        (tasks[2], "batch"): "#F59E0B",
        (tasks[2], "for_loop"): "#FCD34D",
    }

    # Multiplicative offsets keep grouped bars readable on log-x axis.
    offset_exp = np.linspace(-0.12, 0.12, len(series_order))
    offset_scale = np.power(2.0, offset_exp)
    width_ratio = 0.035

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, (task, method) in enumerate(series_order):
        x_vals = []
        y_vals = []
        widths = []
        for env_num in env_nums:
            match = [r for r in records if r.task == task and r.method == method and r.env_num == env_num]
            if not match:
                continue
            x_center = float(env_num) * float(offset_scale[i])
            x_vals.append(x_center)
            y_vals.append(match[0].elapsed_sec * 1000.0)
            widths.append(float(env_num) * width_ratio)
        ax.bar(
            x_vals,
            y_vals,
            width=widths,
            label=labels[i],
            color=color_map[(task, method)],
            alpha=0.95,
            align="center",
            linewidth=0.4,
            edgecolor="black",
        )

    ax.set_xscale("log", base=2)
    ax.set_xticks(env_nums)
    ax.set_xticklabels([str(v) for v in env_nums])
    ax.set_xlabel("env_num")
    ax.set_yscale("log")
    ax.set_ylabel("Replay reset elapsed time (ms)")
    ax.set_title("Batch reset vs For-loop reset (3 tasks)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=3, fontsize=8, loc="upper left", bbox_to_anchor=(0.0, 1.18))
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def build_text_analysis(records: List[ResetBenchRecord]) -> Dict[str, dict]:
    tasks = sorted({r.task for r in records})
    env_nums = sorted({r.env_num for r in records})
    out: Dict[str, dict] = {}
    for task in tasks:
        out[task] = {"by_env_num": {}, "mean_speedup_loop_div_batch": 0.0}
        speedups = []
        for env_num in env_nums:
            batch = next(r for r in records if r.task == task and r.env_num == env_num and r.method == "batch")
            loop = next(r for r in records if r.task == task and r.env_num == env_num and r.method == "for_loop")
            speedup = loop.elapsed_sec / max(batch.elapsed_sec, 1e-12)
            speedups.append(speedup)
            out[task]["by_env_num"][str(env_num)] = {
                "batch_ms": batch.elapsed_sec * 1000.0,
                "for_loop_ms": loop.elapsed_sec * 1000.0,
                "loop_div_batch": speedup,
                "sampled_done_ratio": batch.sampled_done_ratio,
                "batch_us_per_reset_env": batch.us_per_reset_env,
                "for_loop_us_per_reset_env": loop.us_per_reset_env,
            }
        out[task]["mean_speedup_loop_div_batch"] = float(np.mean(np.asarray(speedups, dtype=np.float64)))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark task reset: batch vs pure for-loop")
    parser.add_argument("--tasks", type=str, default=",".join(DEFAULT_TASKS))
    parser.add_argument("--env_pow_min", type=int, default=10)
    parser.add_argument("--env_pow_max", type=int, default=14)
    parser.add_argument("--target_masks", type=int, default=80, help="Collected non-empty done masks per setup")
    parser.add_argument("--max_collect_steps", type=int, default=640)
    parser.add_argument("--warmup_steps", type=int, default=40)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output_json",
        type=str,
        default="benchmark/outputs/reset/latest_reset_batch_vs_loop.json",
    )
    parser.add_argument(
        "--output_png",
        type=str,
        default="benchmark/outputs/reset/latest_reset_batch_vs_loop.png",
    )
    args = parser.parse_args()

    ensure_locomotion_envs_registered()

    tasks = parse_tasks(args.tasks)
    env_nums = [2**p for p in range(args.env_pow_min, args.env_pow_max + 1)]
    all_records: List[ResetBenchRecord] = []

    print("Running reset benchmark with real done-mask replay")
    print(f"Tasks: {tasks}")
    print(f"env_num: {env_nums}")

    for task in tasks:
        for env_num in env_nums:
            records = run_one_setting(
                task=task,
                env_num=env_num,
                repeats=args.repeats,
                target_masks=args.target_masks,
                max_collect_steps=args.max_collect_steps,
                warmup_steps=args.warmup_steps,
                seed=args.seed,
            )
            all_records.extend(records)
            batch = next(r for r in records if r.method == "batch")
            loop = next(r for r in records if r.method == "for_loop")
            print(
                f"[{task}][{env_num}] "
                f"batch={batch.elapsed_sec * 1000.0:.2f}ms "
                f"loop={loop.elapsed_sec * 1000.0:.2f}ms "
                f"loop/batch={loop.elapsed_sec / max(batch.elapsed_sec, 1e-12):.2f}x "
                f"done_ratio={batch.sampled_done_ratio:.4f}"
            )

    output_json = Path(args.output_json)
    output_png = Path(args.output_png)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_png.parent.mkdir(parents=True, exist_ok=True)

    analysis = build_text_analysis(all_records)
    payload = {
        "meta": {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "tasks": tasks,
            "env_nums": env_nums,
            "target_masks": args.target_masks,
            "max_collect_steps": args.max_collect_steps,
            "warmup_steps": args.warmup_steps,
            "repeats": args.repeats,
            "seed": args.seed,
            "note": "done masks are sampled from real env.step random-policy trajectories, then replayed.",
        },
        "results": [asdict(r) for r in all_records],
        "analysis": analysis,
    }
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    plot_grouped_thin_bars(all_records, output_png)
    print(f"Saved JSON: {output_json}")
    print(f"Saved PNG:  {output_png}")


if __name__ == "__main__":
    main()
