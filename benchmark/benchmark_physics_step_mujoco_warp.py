#!/usr/bin/env python3
"""
Benchmark MuJoCo Warp physics execution.

Benchmarks mujoco_warp across current locomotion owner task ids
(go1_joystick_flat/go2_joystick_flat/g1_walk_flat) and outputs JSON + plots
aligned with benchmark/benchmark_physics_step_mj_step.py.
Legacy env names remain accepted as aliases.

Run without changing repo dependencies:
    uv run --with mujoco-warp --with warp-lang \
        python benchmark/benchmark_physics_step_mujoco_warp.py
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import mujoco
except ImportError:
    mujoco = None

try:
    import mujoco_warp as mj_warp
except ImportError:
    mj_warp = None

try:
    import warp
except ImportError:
    warp = None

try:
    from benchmark.core import device_info as _benchmark_device_info
    from benchmark.core import task_names as _benchmark_task_names

    _device_info = _benchmark_device_info
    _task_names = _benchmark_task_names
except ModuleNotFoundError:
    from core import device_info as _core_device_info
    from core import task_names as _core_task_names

    _device_info = _core_device_info
    _task_names = _core_task_names

get_device_info_dict = _device_info.get_device_info_dict
get_device_info_line = _device_info.get_device_info_line
canonical_locomotion_task_ids = _task_names.canonical_locomotion_task_ids
locomotion_task_spec = _task_names.locomotion_task_spec
locomotion_task_model_file = _task_names.locomotion_task_model_file
normalize_locomotion_task_id = _task_names.normalize_locomotion_task_id


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
DEFAULT_NJMAX_BY_TASK = {
    "go1_joystick_flat": 100,
    "go2_joystick_flat": 100,
    "g1_walk_flat": 256,
    "sharpa_inhand": 128,
}


def _display_backend(backend: str) -> str:
    return backend


def _uv_run_hint() -> str:
    return (
        "uv run --with mujoco-warp --with warp-lang "
        "python benchmark/benchmark_physics_step_mujoco_warp.py"
    )


def _require_mujoco_warp() -> None:
    if mujoco is None:
        raise RuntimeError(
            "mujoco is unavailable in the current environment. "
            f"Run with temporary deps instead of editing pyproject.toml:\n  {_uv_run_hint()}"
        )
    if mj_warp is None:
        raise RuntimeError(
            "mujoco_warp is unavailable in the current environment. "
            f"Run with temporary deps instead of editing pyproject.toml:\n  {_uv_run_hint()}"
        )
    if warp is None:
        raise RuntimeError(
            "warp is unavailable in the current environment. "
            f"Run with temporary deps instead of editing pyproject.toml:\n  {_uv_run_hint()}"
        )


def _load_task_model(task_name: str) -> Any:
    return cast(Any, mujoco).MjModel.from_xml_path(locomotion_task_model_file(task_name))


def _task_njmax(task_name: str) -> int:
    return DEFAULT_NJMAX_BY_TASK.get(task_name, -1)


def _make_warp_data(model: Any, batch_size: int, njmax: int):
    mj_warp_mod = cast(Any, mj_warp)
    try:
        if njmax > 0:
            return mj_warp_mod.make_data(model, nworld=batch_size, njmax=njmax, nconmax=njmax)
        return mj_warp_mod.make_data(model, nworld=batch_size)
    except TypeError:
        return mj_warp_mod.make_data(model, nworld=batch_size)


def _run_warp(warp_model, warp_data, nstep: int, niter: int) -> float:
    mj_warp_mod = cast(Any, mj_warp)
    warp_mod = cast(Any, warp)
    t0 = time.perf_counter()
    for _ in range(niter):
        for _ in range(nstep):
            mj_warp_mod.step(warp_model, warp_data)
            warp_mod.synchronize()
    return (time.perf_counter() - t0) / niter


def _bench_one_task(
    task_name: str,
    batch_sizes: List[int],
    nstep: int,
    warmup: int,
    iters: int,
) -> List[BenchRecord]:
    task_key = normalize_locomotion_task_id(task_name)
    model = _load_task_model(task_key)
    warp_model = cast(Any, mj_warp).put_model(model)
    njmax = _task_njmax(task_key)

    records: List[BenchRecord] = []
    for batch_size in batch_sizes:
        warp_data = _make_warp_data(model, batch_size, njmax)

        _run_warp(warp_model, warp_data, nstep=nstep, niter=warmup)
        avg_t = _run_warp(warp_model, warp_data, nstep=nstep, niter=iters)

        records.append(
            BenchRecord(
                task=task_key,
                backend="mujoco_warp",
                batch_size=batch_size,
                nstep=nstep,
                nthread=0,
                avg_time_sec=avg_t,
                sps=batch_size * nstep / avg_t,
            )
        )
        print(
            f"[{task_key}] batch={batch_size:5d} "
            f"mujoco_warp={avg_t * 1000:.3f}ms ({batch_size * nstep / avg_t / 1e4:.2f}万fps)"
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
                label=f"{locomotion_task_spec(task_name).display_name} ({_display_backend(backend)})",
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
                label=f"{locomotion_task_spec(task_name).display_name} ({_display_backend(backend)})",
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
    parser = argparse.ArgumentParser(description="Benchmark MuJoCo Warp physics execution")
    parser.add_argument("--nstep", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument(
        "--tasks",
        type=str,
        default=",".join(DEFAULT_TASK_IDS),
    )
    parser.add_argument(
        "--batch-sizes", type=str, default=",".join(str(x) for x in DEFAULT_BATCH_SIZES)
    )
    parser.add_argument(
        "--out-json",
        type=str,
        default="benchmark/outputs/physics_step/mujoco_warp/results.json",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="benchmark/outputs/physics_step/mujoco_warp",
        help="Directory for output plots",
    )
    args = parser.parse_args()

    _require_mujoco_warp()

    task_names = [normalize_locomotion_task_id(x) for x in args.tasks.split(",") if x.strip()]
    batch_sizes = [int(x.strip()) for x in args.batch_sizes.split(",") if x.strip()]

    print("MuJoCo Warp backend available")
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
            "nthread": 0,
            "warmup": args.warmup,
            "iters": args.iters,
            "backends": sorted({r.backend for r in records}),
            "task_njmax": {task: _task_njmax(task) for task in task_names},
            "warp_available": mj_warp is not None and warp is not None,
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
