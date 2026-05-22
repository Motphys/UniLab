#!/usr/bin/env python3
"""
Benchmark Genesis physics execution.

Benchmarks Genesis across current locomotion owner task ids
(go1_joystick_flat/go2_joystick_flat/g1_walk_flat) and outputs JSON + plots
aligned with benchmark/benchmark_physics_step_mj_step.py.
Legacy env names remain accepted as aliases.

Run without changing repo dependencies:
    uv run --with genesis-world \
        python benchmark/benchmark_physics_step_genesis.py
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import genesis as _gs
except ImportError:
    gs = None
else:
    gs = _gs

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


def _display_backend(backend: str) -> str:
    return backend


def _uv_run_hint() -> str:
    return "uv run --with genesis-world python benchmark/benchmark_physics_step_genesis.py"


def _require_genesis() -> None:
    if gs is None:
        raise RuntimeError(
            "genesis is unavailable in the current environment. "
            f"Run with temporary deps instead of editing pyproject.toml:\n  {_uv_run_hint()}"
        )


def _reset_genesis() -> None:
    # Quadrants runtime pool leaks across scenes; only gs.destroy() frees it.
    gs_mod = cast(Any, gs)
    gs_mod.destroy()
    gs_mod.init(backend=gs_mod.gpu)


def _load_task_xml(task_name: str) -> str:
    return locomotion_task_model_file(task_name)


def _build_scene(xml_path: str, batch_size: int):
    gs_mod = cast(Any, gs)
    scene = gs_mod.Scene(
        show_viewer=False,
        rigid_options=gs_mod.options.RigidOptions(
            dt=0.01,
            constraint_solver=gs_mod.constraint_solver.Newton,
        ),
    )
    scene.add_entity(gs_mod.morphs.MJCF(file=xml_path))
    scene.build(n_envs=batch_size)
    return scene


def _run_scene(scene, nstep: int, niter: int) -> float:
    t0 = time.perf_counter()
    for _ in range(niter):
        for _ in range(nstep):
            scene.step()
    return (time.perf_counter() - t0) / niter


def _bench_one_task(
    task_name: str,
    batch_sizes: List[int],
    nstep: int,
    warmup: int,
    iters: int,
) -> List[BenchRecord]:
    task_key = normalize_locomotion_task_id(task_name)
    xml_path = _load_task_xml(task_key)

    records: List[BenchRecord] = []
    for batch_size in batch_sizes:
        scene = _build_scene(xml_path, batch_size)
        try:
            _run_scene(scene, nstep=nstep, niter=warmup)
            avg_t = _run_scene(scene, nstep=nstep, niter=iters)
        finally:
            scene.destroy()
            del scene
            gc.collect()
            _reset_genesis()

        records.append(
            BenchRecord(
                task=task_key,
                backend="genesis",
                batch_size=batch_size,
                nstep=nstep,
                nthread=0,
                avg_time_sec=avg_t,
                sps=batch_size * nstep / avg_t,
            )
        )
        print(
            f"[{task_key}] batch={batch_size:5d} "
            f"genesis={avg_t * 1000:.3f}ms ({batch_size * nstep / avg_t / 1e4:.2f}万fps)"
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
    parser = argparse.ArgumentParser(description="Benchmark Genesis physics execution")
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
        "--out-json", type=str, default="benchmark/outputs/physics_step/genesis/results.json"
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="benchmark/outputs/physics_step/genesis",
        help="Directory for output plots",
    )
    args = parser.parse_args()

    _require_genesis()
    _reset_genesis()

    task_names = [normalize_locomotion_task_id(x) for x in args.tasks.split(",") if x.strip()]
    batch_sizes = [int(x.strip()) for x in args.batch_sizes.split(",") if x.strip()]

    print("Genesis backend available")
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
            "genesis_available": gs is not None,
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
