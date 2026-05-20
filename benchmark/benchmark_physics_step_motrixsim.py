#!/usr/bin/env python3
"""
Benchmark MotrixSim backend physics as driven by UniLab env.step.

Sweeps batch sizes across the current locomotion owner task ids
(go1_joystick_flat / go2_joystick_flat / g1_walk_flat).  The benchmark
constructs the same task-side env/config path used by benchmark_env_step.py,
warms up with full env.step calls, then reports the Motrix backend physics
segment from state.info["timing"]["backend_physics_ms"].

Run:
    uv run benchmark/benchmark_physics_step_motrixsim.py
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from benchmark import benchmark_env_step as _benchmark_env_step
except ModuleNotFoundError:
    import benchmark_env_step as _benchmark_env_step  # type: ignore[no-redef]

try:
    from benchmark.core import device_info as _benchmark_device_info
    from benchmark.core import task_names as _benchmark_task_names
except ModuleNotFoundError:
    from core import device_info as _benchmark_device_info
    from core import task_names as _benchmark_task_names

get_device_info_dict = _benchmark_device_info.get_device_info_dict
get_device_info_line = _benchmark_device_info.get_device_info_line
canonical_locomotion_task_ids = _benchmark_task_names.canonical_locomotion_task_ids
locomotion_task_spec = _benchmark_task_names.locomotion_task_spec

_MOTRIXSIM_IMPORT_ERROR: Exception | None = None
try:
    import motrixsim as mtx
except Exception as _motrixsim_error:
    mtx = None  # type: ignore[assignment]
    _MOTRIXSIM_IMPORT_ERROR = _motrixsim_error


@dataclass
class BenchRecord:
    task: str
    backend: str
    batch_size: int
    nstep: int
    nthread: int
    avg_time_sec: float
    sps: float
    sim_substeps: int
    avg_env_step_time_sec: float
    env_step_sps: float
    median_env_step_ms: float
    median_backend_physics_ms: float
    median_backend_set_ctrl_ms: float
    median_backend_refresh_cache_ms: float


DEFAULT_TASK_IDS = ["go1_joystick_flat", "go2_joystick_flat", "g1_walk_flat"]
DEFAULT_BATCH_SIZES = [2**k for k in range(8, 15)]  # 256 .. 16384


def _normalize_task_name(task_name: str) -> str:
    normalized = task_name.strip()
    if normalized.startswith("task="):
        normalized = normalized[len("task=") :]
    if "/" in normalized:
        normalized = normalized.rsplit("/", 1)[0]
    spec = locomotion_task_spec(normalized)
    return spec.owner_task_id


def _require_motrixsim() -> None:
    if mtx is not None:
        return
    detail = (
        repr(_MOTRIXSIM_IMPORT_ERROR) if _MOTRIXSIM_IMPORT_ERROR is not None else "unknown error"
    )
    raise RuntimeError(
        "motrixsim is unavailable in the current runtime.\n"
        f"Import detail: {detail}\n"
        "Install with the project's motrix extra, e.g. `uv sync --extra motrix`."
    )


def _motrix_task_config(task_name: str) -> Any:
    task_config = _benchmark_env_step._matching_task_config(task_name)
    if task_config is None:
        env_name = locomotion_task_spec(task_name).env_task_name
        task_config = _benchmark_env_step._matching_task_config(env_name)
    if task_config is None:
        raise ValueError(f"Task '{task_name}' is not registered in benchmark_env_step.py")
    return task_config


def _build_motrix_env(task_name: str, num_envs: int) -> tuple[Any, Any]:
    task_config = _motrix_task_config(task_name)
    cfg = task_config.build_cfg("motrix")
    task_config.finalize_cfg(cfg, "motrix")
    cfg.validate()

    env_cls = task_config.env_cls_factory()
    env = env_cls(cfg, num_envs=num_envs, backend_type="motrix")
    env.init_state()
    return env, cfg


def _sample_actions(num_envs: int, num_actions: int, rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(-1, 1, size=(num_envs, num_actions)).astype(np.float32)


def _append_timing(
    timing_records: dict[str, list[float]],
    timing: dict[str, Any],
) -> None:
    for key, value in timing.items():
        timing_records.setdefault(key, []).append(float(value))


def _median_timing_ms(timing_records: dict[str, list[float]], key: str) -> float:
    values = timing_records.get(key, [])
    if not values:
        return 0.0
    return float(np.median(np.asarray(values, dtype=np.float64)))


def _sum_timing_sec(
    timing_records: dict[str, list[float]],
    key: str,
    fallback_sec: float,
    niter: int,
) -> float:
    values = timing_records.get(key, [])
    if not values:
        return fallback_sec / niter
    return float(np.asarray(values, dtype=np.float64).sum() / 1000.0 / niter)


def _run_env_step(
    env: Any,
    *,
    batch_size: int,
    nstep: int,
    niter: int,
    rng: np.random.Generator,
    collect_timing: bool,
) -> tuple[float, float, dict[str, list[float]]]:
    num_actions = int(env._backend.num_actuators)  # type: ignore[reportAttributeAccessIssue]
    timing_records: dict[str, list[float]] = {}

    t0 = time.perf_counter()
    for _ in range(niter):
        for _ in range(nstep):
            actions = _sample_actions(batch_size, num_actions, rng)
            state = env.step(actions)
            if collect_timing:
                timing = state.info.get("timing", {})
                if isinstance(timing, dict):
                    _append_timing(timing_records, timing)
    elapsed_sec = time.perf_counter() - t0

    if not collect_timing:
        avg_elapsed_sec = elapsed_sec / niter
        return avg_elapsed_sec, avg_elapsed_sec, timing_records

    avg_physics_sec = _sum_timing_sec(
        timing_records,
        "backend_physics_ms",
        fallback_sec=elapsed_sec,
        niter=niter,
    )
    avg_env_step_sec = _sum_timing_sec(
        timing_records,
        "env_step_total_ms",
        fallback_sec=elapsed_sec,
        niter=niter,
    )
    return avg_physics_sec, avg_env_step_sec, timing_records


def _bench_one_task(
    task_name: str,
    batch_sizes: List[int],
    nstep: int,
    warmup: int,
    iters: int,
    seed: int,
) -> List[BenchRecord]:
    task_key = _normalize_task_name(task_name)

    records: List[BenchRecord] = []
    for batch_size in batch_sizes:
        env = None
        try:
            env, cfg = _build_motrix_env(task_key, batch_size)
            sim_substeps = int(cfg.sim_substeps)
            rng = np.random.default_rng(seed + batch_size)

            _run_env_step(
                env,
                batch_size=batch_size,
                nstep=nstep,
                niter=warmup,
                rng=rng,
                collect_timing=False,
            )
            avg_physics_t, avg_env_step_t, timing_records = _run_env_step(
                env,
                batch_size=batch_size,
                nstep=nstep,
                niter=iters,
                rng=rng,
                collect_timing=True,
            )
        finally:
            if env is not None:
                env.close()

        physics_steps_per_iter = batch_size * nstep * sim_substeps
        env_steps_per_iter = batch_size * nstep

        records.append(
            BenchRecord(
                task=task_key,
                backend="motrixsim",
                batch_size=batch_size,
                nstep=nstep,
                nthread=0,
                avg_time_sec=avg_physics_t,
                sps=physics_steps_per_iter / avg_physics_t,
                sim_substeps=sim_substeps,
                avg_env_step_time_sec=avg_env_step_t,
                env_step_sps=env_steps_per_iter / avg_env_step_t,
                median_env_step_ms=_median_timing_ms(timing_records, "env_step_total_ms"),
                median_backend_physics_ms=_median_timing_ms(timing_records, "backend_physics_ms"),
                median_backend_set_ctrl_ms=_median_timing_ms(timing_records, "backend_set_ctrl_ms"),
                median_backend_refresh_cache_ms=_median_timing_ms(
                    timing_records, "backend_refresh_cache_ms"
                ),
            )
        )
        print(
            f"[{task_key}] batch={batch_size:5d} "
            f"motrixsim_physics={avg_physics_t * 1000:.3f}ms "
            f"({physics_steps_per_iter / avg_physics_t / 1e4:.2f}万 physics-steps/s)  "
            f"env_step={avg_env_step_t * 1000:.3f}ms "
            f"({env_steps_per_iter / avg_env_step_t / 1e4:.2f}万 env-steps/s)"
        )
    return records


def _plot_fps(
    records: List[BenchRecord], out_png: Path, batch_sizes: List[int], task_names: List[str]
):
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_title(f"Parallel Physics — Total FPS\n{get_device_info_line()}", fontsize=9)

    for task_name in task_names:
        for backend in sorted({r.backend for r in records if r.task == task_name}):
            recs = sorted(
                [r for r in records if r.task == task_name and r.backend == backend],
                key=lambda r: r.batch_size,
            )
            ax.plot(
                [r.batch_size for r in recs],
                [r.sps / 1e4 for r in recs],
                marker="o",
                linestyle="-",
                label=f"{locomotion_task_spec(task_name).display_name} ({backend})",
            )

    ax.set_xscale("log", base=2)
    ax.set_xticks(batch_sizes)
    ax.set_xticklabels([str(b) for b in batch_sizes], rotation=30, ha="right")
    ax.set_xlabel("Batch Size (Num Envs)")
    ax.set_ylabel("Total FPS (x1e4)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Saved FPS plot to {out_png}")


def _plot_per_env_sps(
    records: List[BenchRecord], out_png: Path, batch_sizes: List[int], task_names: List[str]
):
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_title(f"Parallel Physics — Per-Env SPS\n{get_device_info_line()}", fontsize=9)

    for task_name in task_names:
        for backend in sorted({r.backend for r in records if r.task == task_name}):
            recs = sorted(
                [r for r in records if r.task == task_name and r.backend == backend],
                key=lambda r: r.batch_size,
            )
            ax.plot(
                [r.batch_size for r in recs],
                [r.sps / r.batch_size for r in recs],
                marker="o",
                linestyle="-",
                label=f"{locomotion_task_spec(task_name).display_name} ({backend})",
            )

    ax.set_xscale("log", base=2)
    ax.set_xticks(batch_sizes)
    ax.set_xticklabels([str(b) for b in batch_sizes], rotation=30, ha="right")
    ax.set_xlabel("Batch Size (Num Envs)")
    ax.set_ylabel("Per-Env Steps/s")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Saved per-env SPS plot to {out_png}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark MotrixSim parallel physics execution")
    parser.add_argument("--nstep", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tasks", type=str, default=",".join(DEFAULT_TASK_IDS))
    parser.add_argument(
        "--batch-sizes", type=str, default=",".join(str(x) for x in DEFAULT_BATCH_SIZES)
    )
    parser.add_argument(
        "--out-json",
        type=str,
        default="benchmark/outputs/physics_step/motrixsim/results.json",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="benchmark/outputs/physics_step/motrixsim",
        help="Directory for output plots",
    )
    args = parser.parse_args()

    _require_motrixsim()

    task_names = [_normalize_task_name(x) for x in args.tasks.split(",") if x.strip()]
    batch_sizes = [int(x.strip()) for x in args.batch_sizes.split(",") if x.strip()]

    print("MotrixSim backend available")
    print(f"Tasks: {task_names}")
    print(f"Batch sizes: {batch_sizes}")

    records: List[BenchRecord] = []
    for task_name in task_names:
        records.extend(
            _bench_one_task(
                task_name=task_name,
                batch_sizes=batch_sizes,
                nstep=args.nstep,
                warmup=args.warmup,
                iters=args.iters,
                seed=args.seed,
            )
        )

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, object] = {
        "meta": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "device_info": get_device_info_dict(),
            "tasks": task_names,
            "batch_sizes": batch_sizes,
            "nstep": args.nstep,
            "step_unit": "env.step calls",
            "sps_unit": "physics substeps/s",
            "env_step_sps_unit": "env.step calls/s",
            "warmup": args.warmup,
            "iters": args.iters,
            "seed": args.seed,
            "backends": sorted({r.backend for r in records}),
            "measurement": "unilab_env_step_backend_physics_ms",
        },
        "results": [asdict(r) for r in records],
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved results to {out_json}")

    out_dir = Path(args.out_dir)
    _plot_fps(records, out_dir / "fps.png", batch_sizes=batch_sizes, task_names=task_names)
    _plot_per_env_sps(
        records, out_dir / "per_env_sps.png", batch_sizes=batch_sizes, task_names=task_names
    )


if __name__ == "__main__":
    main()
