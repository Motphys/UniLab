#!/usr/bin/env python3
"""
Benchmark MotrixSim parallel physics execution.

Uses the motrixsim Python API directly (no UniLab env / backend wrapper)
to mirror benchmark/benchmark_physics_step_mj_step.py:
  * sim_dt comes from the XML (`<option timestep>`), not overridden.
  * max_iterations comes from the XML (`<option iterations>`), not overridden.
  * Initial state = keyframe 0, no randomization.
  * ctrl = uniform(-1, 1), pre-generated with the shared seed used by mj_step.
  * Each measured call advances `--nstep` physics steps (default 20).
  * SPS = batch_size * nstep / wall_clock_time.

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
    from benchmark.core import device_info as _benchmark_device_info
    from benchmark.core import task_names as _benchmark_task_names
except ModuleNotFoundError:
    from core import device_info as _benchmark_device_info
    from core import task_names as _benchmark_task_names

get_device_info_dict = _benchmark_device_info.get_device_info_dict
get_device_info_line = _benchmark_device_info.get_device_info_line
canonical_locomotion_task_ids = _benchmark_task_names.canonical_locomotion_task_ids
locomotion_task_spec = _benchmark_task_names.locomotion_task_spec
locomotion_task_model_file = _benchmark_task_names.locomotion_task_model_file
normalize_locomotion_task_id = _benchmark_task_names.normalize_locomotion_task_id

_MOTRIXSIM_IMPORT_ERROR: Exception | None = None
try:
    import motrixsim as mtx
except Exception as _motrixsim_error:
    mtx = None  # type: ignore[assignment]
    _MOTRIXSIM_IMPORT_ERROR = _motrixsim_error

# Mirror UniLab MotrixBackend's production default. Training uses this value
# (EnvConfig.motrix_max_iterations defaults to None, and the backend falls back
# to DEFAULT_MOTRIX_MAX_ITERATIONS), so the benchmark reflects real workloads
# rather than the XML's <option iterations> which would otherwise leak in.
from unilab.base.backend.motrix.backend import (  # noqa: E402
    DEFAULT_MOTRIX_MAX_ITERATIONS,
)


@dataclass
class BenchRecord:
    task: str
    backend: str
    batch_size: int
    nstep: int
    nthread: int
    avg_time_sec: float
    sps: float


DEFAULT_TASK_IDS = canonical_locomotion_task_ids()
DEFAULT_BATCH_SIZES = [2**k for k in range(8, 15)]  # 256 .. 16384


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


def _load_task_model(task_name: str) -> Any:
    model_file = locomotion_task_model_file(task_name)
    model = mtx.load_model(model_file)  # type: ignore[union-attr]
    model.options.max_iterations = DEFAULT_MOTRIX_MAX_ITERATIONS
    return model


def _apply_keyframe0(model: Any, data: Any) -> None:
    if model.num_keyframes > 0:
        model.keyframes[0].apply(data)
        model.forward_kinematic(data)


def _keyframe0_ctrl(model: Any, batch_size: int) -> np.ndarray:
    """Return (batch, nu) ctrl tiled from `model.keyframes[0].ctrl`.

    Mirrors mj_step's `model.key_ctrl[0]`: PD targets that hold the robot at the
    keyframe pose. Same per-env ctrl held across all nstep substeps.
    """
    num_actuators = int(model.num_actuators)
    if num_actuators == 0:
        return np.empty((batch_size, 0), dtype=np.float32)
    if model.num_keyframes > 0:
        ctrl0 = np.asarray(model.keyframes[0].ctrl, dtype=np.float32)
    else:
        ctrl0 = np.zeros((num_actuators,), dtype=np.float32)
    return np.broadcast_to(ctrl0, (batch_size, num_actuators)).copy()


def _run_step(
    model: Any,
    data: Any,
    ctrl: np.ndarray,
    nstep: int,
    niter: int,
) -> float:
    """Run niter iterations of (apply keyframe -> step_n(nstep)). Returns avg seconds per iter."""
    t_total = 0.0
    for _ in range(niter):
        _apply_keyframe0(model, data)
        data.actuator_ctrls = np.ascontiguousarray(ctrl)
        t0 = time.perf_counter()
        model.step_n(data, nstep)
        t_total += time.perf_counter() - t0
    return t_total / niter


def _bench_one_task(
    task_name: str,
    batch_sizes: List[int],
    nstep: int,
    warmup: int,
    iters: int,
) -> List[BenchRecord]:
    task_key = normalize_locomotion_task_id(task_name)
    model = _load_task_model(task_key)

    records: List[BenchRecord] = []
    for batch_size in batch_sizes:
        data = mtx.SceneData(model, batch=[batch_size])  # type: ignore[union-attr]
        ctrl = _keyframe0_ctrl(model, batch_size)

        _run_step(model, data, ctrl, nstep=nstep, niter=warmup)
        avg_t = _run_step(model, data, ctrl, nstep=nstep, niter=iters)

        records.append(
            BenchRecord(
                task=task_key,
                backend="motrixsim",
                batch_size=batch_size,
                nstep=nstep,
                nthread=0,
                avg_time_sec=avg_t,
                sps=batch_size * nstep / avg_t,
            )
        )
        print(
            f"[{task_key}] batch={batch_size:5d} "
            f"motrixsim={avg_t * 1000:.3f}ms "
            f"({batch_size * nstep / avg_t / 1e4:.2f}万fps)"
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

    task_names = [normalize_locomotion_task_id(x) for x in args.tasks.split(",") if x.strip()]
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
            "warmup": args.warmup,
            "iters": args.iters,
            "backends": sorted({r.backend for r in records}),
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
