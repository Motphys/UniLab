#!/usr/bin/env python3
"""
Benchmark MuJoCo parallel physics execution.

macOS:     compares numpy rollout vs native mlx_step.
Other:     benchmarks mujoco.rollout with the configured thread count only.

Sweeps batch sizes across current locomotion owner task ids
(go1_joystick_flat/go2_joystick_flat/g1_joystick_flat).
Legacy env names remain accepted as aliases.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from multiprocessing import cpu_count
from pathlib import Path
from typing import Dict, List

import matplotlib
import mujoco
from mujoco import rollout as mj_rollout

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

_IS_MACOS = platform.system() == "Darwin"

if _IS_MACOS:
    import mlx.core as mx

    try:
        from mujoco import mlx_step as mj_mlx_step
    except Exception:
        mj_mlx_step = None
else:
    mx = None
    mj_mlx_step = None

try:
    from benchmark.core.device_info import get_device_info_dict, get_device_info_line
    from benchmark.core.task_names import (
        canonical_locomotion_task_ids,
        locomotion_task_spec,
        normalize_locomotion_task_id,
    )
except ModuleNotFoundError:
    from core.device_info import get_device_info_dict, get_device_info_line
    from core.task_names import (
        canonical_locomotion_task_ids,
        locomotion_task_spec,
        normalize_locomotion_task_id,
    )


@dataclass
class BenchRecord:
    task: str
    backend: str  # "numpy" | "mlx_native" (macOS) | "rollout_Nt"
    batch_size: int
    nstep: int
    nthread: int
    avg_time_sec: float
    sps: float  # steps per second = batch_size * nstep / avg_time_sec


DEFAULT_TASK_IDS = canonical_locomotion_task_ids()
DEFAULT_BATCH_SIZES = [2**k for k in range(8, 15)]  # 256 .. 16384
TASK_ALPHA = {"go1_joystick_flat": 0.75, "go2_joystick_flat": 0.9, "g1_joystick_flat": 1.0}
TASK_HATCH = {"go1_joystick_flat": "//", "go2_joystick_flat": "\\\\", "g1_joystick_flat": "xx"}


def _keyframe0_state_and_ctrl(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray]:
    data = mujoco.MjData(model)
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    else:
        mujoco.mj_resetData(model, data)
    nstate = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    state0 = np.empty((nstate,), dtype=np.float64)
    mujoco.mj_getState(model, data, state0, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    if model.nu == 0:
        ctrl0 = np.empty((0,), dtype=np.float64)
    elif model.nkey > 0:
        ctrl0 = np.asarray(model.key_ctrl[0], dtype=np.float64).copy()
    else:
        ctrl0 = np.zeros((model.nu,), dtype=np.float64)
    return state0, ctrl0


def _run_numpy(
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


def _run_mlx(
    runner,
    model_list,
    data_list,
    initial_state_mx,
    control_mx,
    nstep: int,
    niter: int,
    chunk_size: int,
) -> float:
    t0 = time.perf_counter()
    for _ in range(niter):
        out = runner.step(
            model=model_list,
            data=data_list,
            initial_state=initial_state_mx,
            control=control_mx,
            nstep=nstep,
            chunk_size=chunk_size,
            out_dtype=mx.float32,
        )
        state_mx, sensor_mx = out if isinstance(out, tuple) else (out.state_mx, out.sensordata_mx)
        mx.eval(state_mx, sensor_mx)
    return (time.perf_counter() - t0) / niter


def _has_native_mujoco_mlx_step() -> bool:
    return mj_mlx_step is not None and hasattr(mj_mlx_step, "MlxStepRunner")


def _load_task_model(task_name: str) -> mujoco.MjModel:
    cfg = locomotion_task_spec(task_name).config_cls()
    return mujoco.MjModel.from_xml_path(cfg.model_file)


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
    chunk_size: int,
) -> List[BenchRecord]:
    task_key = normalize_locomotion_task_id(task_name)
    np.random.seed(42)
    model = _load_task_model(task_key)
    nstate = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
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

        if _IS_MACOS:
            actual_nthread = min(batch_size, nthread, cpu_count())
            data_list = [mujoco.MjData(model) for _ in range(actual_nthread)]
            with (
                mj_rollout.Rollout(nthread=actual_nthread) as numpy_runner,
                mj_mlx_step.MlxStepRunner(nthread=actual_nthread) as mlx_runner,
            ):
                initial_state_mx = mx.array(initial_state, dtype=mx.float32)
                control_mx = mx.array(control, dtype=mx.float32)

                _run_numpy(
                    numpy_runner,
                    model_list,
                    data_list,
                    initial_state,
                    control,
                    state_buf,
                    sensordata_buf,
                    nstep,
                    warmup,
                )
                _run_mlx(
                    mlx_runner,
                    model_list,
                    data_list,
                    initial_state_mx,
                    control_mx,
                    nstep,
                    warmup,
                    chunk_size,
                )

                numpy_t = _run_numpy(
                    numpy_runner,
                    model_list,
                    data_list,
                    initial_state,
                    control,
                    state_buf,
                    sensordata_buf,
                    nstep,
                    iters,
                )
                mlx_t = _run_mlx(
                    mlx_runner,
                    model_list,
                    data_list,
                    initial_state_mx,
                    control_mx,
                    nstep,
                    iters,
                    chunk_size,
                )

            records.append(
                BenchRecord(
                    task=task_key,
                    backend="numpy",
                    batch_size=batch_size,
                    nstep=nstep,
                    nthread=actual_nthread,
                    avg_time_sec=numpy_t,
                    sps=batch_size * nstep / numpy_t,
                )
            )
            records.append(
                BenchRecord(
                    task=task_key,
                    backend="mlx_native",
                    batch_size=batch_size,
                    nstep=nstep,
                    nthread=actual_nthread,
                    avg_time_sec=mlx_t,
                    sps=batch_size * nstep / mlx_t,
                )
            )
            print(
                f"[{task_key}] batch={batch_size:5d} "
                f"numpy={numpy_t * 1000:.3f}ms ({batch_size * nstep / numpy_t / 1e4:.2f}万fps)  "
                f"mlx={mlx_t * 1000:.3f}ms ({batch_size * nstep / mlx_t / 1e4:.2f}万fps)"
            )
        else:
            data_list_n = [mujoco.MjData(model) for _ in range(nthread)]
            with mj_rollout.Rollout(nthread=nthread) as runner_n:
                _run_numpy(
                    runner_n,
                    model_list,
                    data_list_n,
                    initial_state,
                    control,
                    state_buf,
                    sensordata_buf,
                    nstep,
                    warmup,
                )
                tn = _run_numpy(
                    runner_n,
                    model_list,
                    data_list_n,
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
                    backend=f"rollout_{nthread}t",
                    batch_size=batch_size,
                    nstep=nstep,
                    nthread=nthread,
                    avg_time_sec=tn,
                    sps=batch_size * nstep / tn,
                )
            )
            print(
                f"[{task_key}] batch={batch_size:5d} "
                f"rollout({nthread}t)={tn * 1000:.3f}ms ({batch_size * nstep / tn / 1e4:.2f}万fps)"
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
                linestyle="--" if backend == "numpy" else "-",
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
                linestyle="--" if backend == "numpy" else "-",
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
    parser = argparse.ArgumentParser(
        description="Benchmark parallel physics backends (rollout / mlx_step)"
    )
    parser.add_argument("--nstep", type=int, default=20)
    parser.add_argument("--nthread", type=int, default=cpu_count() * 2)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument(
        "--chunk-size", type=int, default=16, help="MLX step chunk size (macOS only)"
    )
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

    if _IS_MACOS:
        if not _has_native_mujoco_mlx_step():
            raise RuntimeError("Native MLX step backend unavailable. Requires mujoco.mlx_step.")
        print(f"macOS: native mlx_step available, nthread={args.nthread}")
    else:
        print(f"Non-macOS: using mujoco.rollout ({args.nthread} threads)")

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
                chunk_size=args.chunk_size,
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
            "chunk_size": args.chunk_size,
            "native_mlx_step_available": _has_native_mujoco_mlx_step(),
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
