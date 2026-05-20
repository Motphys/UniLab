#!/usr/bin/env python3
"""
Benchmark MuJoCo parallel physics execution.

Benchmarks mujoco.rollout with the configured thread count.

Sweeps batch sizes across current locomotion owner task ids
(go1_joystick_flat/go2_joystick_flat/g1_walk_flat).
Legacy env names remain accepted as aliases.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any, Dict, List, cast

import matplotlib
import mujoco
from mujoco import rollout as mj_rollout

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

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
normalize_locomotion_task_id = _task_names.normalize_locomotion_task_id


@dataclass
class BenchRecord:
    task: str
    backend: str  # "rollout_Nt"
    batch_size: int
    nstep: int
    nthread: int
    avg_time_sec: float
    sps: float  # steps per second = batch_size * nstep / avg_time_sec


DEFAULT_TASK_IDS = canonical_locomotion_task_ids()
DEFAULT_BATCH_SIZES = [2**k for k in range(8, 15)]  # 256 .. 16384
TASK_ALPHA = {"go1_joystick_flat": 0.75, "go2_joystick_flat": 0.9, "g1_walk_flat": 1.0}
TASK_HATCH = {"go1_joystick_flat": "//", "go2_joystick_flat": "\\\\", "g1_walk_flat": "xx"}


def _keyframe0_state_and_ctrl(model: Any) -> tuple[np.ndarray, np.ndarray]:
    mujoco_mod = cast(Any, mujoco)
    data = mujoco_mod.MjData(model)
    if model.nkey > 0:
        mujoco_mod.mj_resetDataKeyframe(model, data, 0)
    else:
        mujoco_mod.mj_resetData(model, data)
    nstate = mujoco_mod.mj_stateSize(model, mujoco_mod.mjtState.mjSTATE_FULLPHYSICS)
    state0 = np.empty((nstate,), dtype=np.float64)
    mujoco_mod.mj_getState(model, data, state0, mujoco_mod.mjtState.mjSTATE_FULLPHYSICS)
    if model.nu == 0:
        ctrl0 = np.empty((0,), dtype=np.float64)
    elif model.nkey > 0:
        ctrl0 = np.asarray(model.key_ctrl[0], dtype=np.float64).copy()
    else:
        ctrl0 = np.zeros((model.nu,), dtype=np.float64)
    return state0, ctrl0


def _run_rollout(
    runner: mj_rollout.Rollout,
    model_list,
    data_list,
    initial_state: np.ndarray,
    control: np.ndarray,
    state_buf: np.ndarray,
    sensordata_buf: np.ndarray,
    nstep: int,
    niter: int,
) -> float:
    t0 = time.perf_counter()
    for _ in range(niter):
        runner.rollout(
            model_list,
            data_list,
            initial_state,
            control,
            nstep=nstep,
            state=state_buf,
            sensordata=sensordata_buf,
        )
    return (time.perf_counter() - t0) / niter


def _load_task_model(task_name: str) -> Any:
    cfg = locomotion_task_spec(task_name).config_cls()
    return cast(Any, mujoco).MjModel.from_xml_path(cfg.scene.model_file)


def _display_backend(backend: str) -> str:
    if backend.startswith("rollout_") and backend.endswith("t"):
        return f"rollout ({backend[len('rollout_') : -1]} threads)"
    return backend


def _bench_one_task(
    task_name: str,
    batch_sizes: List[int],
    nstep: int,
    nthread: int,
    warmup: int,
    iters: int,
) -> List[BenchRecord]:
    task_key = normalize_locomotion_task_id(task_name)
    np.random.seed(42)
    model = _load_task_model(task_key)
    nstate = cast(Any, mujoco).mj_stateSize(model, cast(Any, mujoco).mjtState.mjSTATE_FULLPHYSICS)
    state0, ctrl0 = _keyframe0_state_and_ctrl(model)

    records: List[BenchRecord] = []
    for batch_size in batch_sizes:
        model_list = [model] * batch_size
        initial_state = np.empty((batch_size, nstate), dtype=np.float64)
        initial_state[:] = state0
        control = np.empty((batch_size, nstep, model.nu), dtype=np.float64)
        control[:] = ctrl0.reshape((1, 1, model.nu))
        state_buf = np.empty((batch_size, nstep, nstate), dtype=np.float64)
        sensordata_buf = np.empty((batch_size, nstep, model.nsensordata), dtype=np.float64)

        actual_nthread = nthread
        data_list = [cast(Any, mujoco).MjData(model) for _ in range(actual_nthread)]
        with mj_rollout.Rollout(nthread=actual_nthread) as runner:
            _run_rollout(
                runner,
                model_list,
                data_list,
                initial_state,
                control,
                state_buf,
                sensordata_buf,
                nstep,
                warmup,
            )
            rollout_t = _run_rollout(
                runner,
                model_list,
                data_list,
                initial_state,
                control,
                state_buf,
                sensordata_buf,
                nstep,
                iters,
            )

        records.append(
            BenchRecord(
                task=task_key,
                backend=f"rollout_{actual_nthread}t",
                batch_size=batch_size,
                nstep=nstep,
                nthread=actual_nthread,
                avg_time_sec=rollout_t,
                sps=batch_size * nstep / rollout_t,
            )
        )
        print(
            f"[{task_key}] batch={batch_size:5d} "
            f"rollout({actual_nthread}t)={rollout_t * 1000:.3f}ms "
            f"({batch_size * nstep / rollout_t / 1e4:.2f}万fps)"
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
    parser = argparse.ArgumentParser(description="Benchmark MuJoCo rollout physics stepping")
    parser.add_argument("--nstep", type=int, default=20)
    parser.add_argument("--nthread", type=int, default=cpu_count() * 2)
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
        "--out-json", type=str, default="benchmark/outputs/physics_step/mj_step/results.json"
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="benchmark/outputs/physics_step/mj_step",
        help="Directory for output plots",
    )
    args = parser.parse_args()

    task_names = [normalize_locomotion_task_id(x) for x in args.tasks.split(",") if x.strip()]
    batch_sizes = [int(x.strip()) for x in args.batch_sizes.split(",") if x.strip()]

    print(f"Using mujoco.rollout (requested nthread={args.nthread})")
    print(f"Tasks: {task_names}")
    print(f"Batch sizes: {batch_sizes}")

    records: List[BenchRecord] = []
    for task_name in task_names:
        records.extend(
            _bench_one_task(
                task_name=task_name,
                batch_sizes=batch_sizes,
                nstep=args.nstep,
                nthread=args.nthread,
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
            "nthread": args.nthread,
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
