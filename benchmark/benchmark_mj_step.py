#!/usr/bin/env python3
"""
Benchmark parallel physics backends: mujoco.rollout (numpy) and mujoco.mlx_step (macOS).

macOS:     compares numpy rollout vs native mlx_step, measuring latency and FPS.
Other:     compares mujoco.rollout with 1 thread vs N threads.

Sweeps batch sizes across go1/go2/g1 locomotion tasks and outputs JSON + plots.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from multiprocessing import cpu_count
from pathlib import Path
from typing import Dict, List

import mujoco
from mujoco import rollout as mj_rollout
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

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

from unilab.envs.locomotion.g1.joystick import G1JoystickCfg
from unilab.envs.locomotion.go1.joystick import Go1JoystickCfg
from unilab.envs.locomotion.go2.joystick import Go2JoystickCfg

try:
    from benchmark.core.device_info import get_device_info_dict, get_device_info_line
except ModuleNotFoundError:
    from core.device_info import get_device_info_dict, get_device_info_line


@dataclass
class BenchRecord:
    task: str
    backend: str  # "numpy" | "mlx_native" (macOS) | "rollout_1t" | "rollout_Nt"
    batch_size: int
    nstep: int
    nthread: int
    avg_time_sec: float
    sps: float  # steps per second = batch_size * nstep / avg_time_sec
    output_shape_mode: str = "n/a"


TASK_CONFIGS = {
    "Go1JoystickFlatTerrain": Go1JoystickCfg,
    "Go2JoystickFlatTerrain": Go2JoystickCfg,
    "G1JoystickFlatTerrain": G1JoystickCfg,
}
DEFAULT_BATCH_SIZES = [2**k for k in range(8, 15)]  # 256 .. 16384


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


def _infer_output_shape_mode(out) -> str:
    state_mx = out[0] if isinstance(out, tuple) else out.state_mx
    return "last_only" if state_mx.ndim == 2 else "full_traj"


def _has_native_mujoco_mlx_step() -> bool:
    return mj_mlx_step is not None and hasattr(mj_mlx_step, "MlxStepRunner")


def _load_task_model(task_name: str) -> mujoco.MjModel:
    cfg = TASK_CONFIGS[task_name]()
    return mujoco.MjModel.from_xml_path(cfg.model_file)


def _geomean(values: List[float]) -> float:
    vals = [v for v in values if v > 0.0]
    if not vals:
        return 0.0
    return float(math.exp(sum(math.log(v) for v in vals) / len(vals)))


def _bench_one_task(
    task_name: str,
    batch_sizes: List[int],
    nstep: int,
    nthread: int,
    warmup: int,
    iters: int,
    chunk_size: int,
) -> List[BenchRecord]:
    np.random.seed(42)
    model = _load_task_model(task_name)
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

                probe = mlx_runner.step(
                    model=model_list,
                    data=data_list,
                    initial_state=initial_state_mx,
                    control=control_mx,
                    nstep=nstep,
                    out_dtype=mx.float32,
                )
                output_shape_mode = _infer_output_shape_mode(probe)

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
                    task=task_name,
                    backend="numpy",
                    batch_size=batch_size,
                    nstep=nstep,
                    nthread=actual_nthread,
                    avg_time_sec=numpy_t,
                    sps=batch_size * nstep / numpy_t,
                    output_shape_mode="n/a",
                )
            )
            records.append(
                BenchRecord(
                    task=task_name,
                    backend="mlx_native",
                    batch_size=batch_size,
                    nstep=nstep,
                    nthread=actual_nthread,
                    avg_time_sec=mlx_t,
                    sps=batch_size * nstep / mlx_t,
                    output_shape_mode=output_shape_mode,
                )
            )
            print(
                f"[{task_name}] batch={batch_size:5d} "
                f"numpy={numpy_t * 1000:.3f}ms ({batch_size * nstep / numpy_t / 1e4:.2f}万fps)  "
                f"mlx={mlx_t * 1000:.3f}ms ({batch_size * nstep / mlx_t / 1e4:.2f}万fps)"
            )
        else:
            data_list_1 = [mujoco.MjData(model) for _ in range(1)]
            data_list_n = [mujoco.MjData(model) for _ in range(nthread)]
            with (
                mj_rollout.Rollout(nthread=1) as runner1,
                mj_rollout.Rollout(nthread=nthread) as runner_n,
            ):
                _run_numpy(
                    runner1,
                    model_list,
                    data_list_1,
                    initial_state,
                    control,
                    state_buf,
                    sensordata_buf,
                    nstep,
                    warmup,
                )
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
                t1 = _run_numpy(
                    runner1,
                    model_list,
                    data_list_1,
                    initial_state,
                    control,
                    state_buf,
                    sensordata_buf,
                    nstep,
                    iters,
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
                    task=task_name,
                    backend="rollout_1t",
                    batch_size=batch_size,
                    nstep=nstep,
                    nthread=1,
                    avg_time_sec=t1,
                    sps=batch_size * nstep / t1,
                    output_shape_mode="n/a",
                )
            )
            records.append(
                BenchRecord(
                    task=task_name,
                    backend=f"rollout_{nthread}t",
                    batch_size=batch_size,
                    nstep=nstep,
                    nthread=nthread,
                    avg_time_sec=tn,
                    sps=batch_size * nstep / tn,
                    output_shape_mode="n/a",
                )
            )
            print(
                f"[{task_name}] batch={batch_size:5d} "
                f"rollout(1t)={t1 * 1000:.3f}ms  rollout({nthread}t)={tn * 1000:.3f}ms"
            )
    return records


def _plot_latency(records: List[BenchRecord], out_png: Path, batch_sizes: List[int], nthread: int):
    out_png.parent.mkdir(parents=True, exist_ok=True)
    if _IS_MACOS:
        backend_colors = {"numpy": "#4C78A8", "mlx_native": "#F58518"}
        backend_display = {"numpy": "Backend:NumPy", "mlx_native": "Backend:MLX"}
    else:
        backend_colors = {"rollout_1t": "#4C78A8", f"rollout_{nthread}t": "#F58518"}
        backend_display = {
            "rollout_1t": "rollout (1 thread)",
            f"rollout_{nthread}t": f"rollout ({nthread} threads)",
        }

    all_backends = list(backend_colors.keys())
    task_alpha = {
        "Go1JoystickFlatTerrain": 0.75,
        "Go2JoystickFlatTerrain": 0.9,
        "G1JoystickFlatTerrain": 1.0,
    }
    task_hatch = {
        "Go1JoystickFlatTerrain": "//",
        "Go2JoystickFlatTerrain": "\\\\",
        "G1JoystickFlatTerrain": "xx",
    }
    task_names = list(TASK_CONFIGS.keys())
    x = np.arange(len(batch_sizes), dtype=np.float64)
    bar_width = 0.10
    pair_inner_gap = 0.02
    task_group_gap = 0.09
    pair_span = len(all_backends) * bar_width + (len(all_backends) - 1) * pair_inner_gap
    total_span = len(task_names) * pair_span + (len(task_names) - 1) * task_group_gap
    left_edge = -0.5 * total_span
    value_map = {(r.task, r.backend, r.batch_size): r.avg_time_sec * 1000.0 for r in records}

    fig, ax = plt.subplots(figsize=(13, 7))
    for task_idx, task_name in enumerate(task_names):
        pair_start = left_edge + task_idx * (pair_span + task_group_gap)
        for b_idx, backend in enumerate(all_backends):
            offset = pair_start + b_idx * (bar_width + pair_inner_gap) + 0.5 * bar_width
            y = [value_map.get((task_name, backend, b), np.nan) for b in batch_sizes]
            ax.bar(
                x + offset,
                y,
                width=bar_width,
                color=backend_colors[backend],
                alpha=task_alpha[task_name],
                hatch=task_hatch[task_name],
                edgecolor="black",
                linewidth=0.2,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in batch_sizes])
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Average Time per Rollout Call (ms)")
    ax.set_yscale("log")
    ax.grid(True, which="major", axis="y", alpha=0.3)
    backend_handles = [
        Patch(facecolor=backend_colors[b], edgecolor="black", label=backend_display[b])
        for b in all_backends
    ]
    task_handles = [
        Patch(
            facecolor="white",
            edgecolor="black",
            hatch=task_hatch["Go1JoystickFlatTerrain"],
            label="Task:Go1",
        ),
        Patch(
            facecolor="white",
            edgecolor="black",
            hatch=task_hatch["Go2JoystickFlatTerrain"],
            label="Task:Go2",
        ),
        Patch(
            facecolor="white",
            edgecolor="black",
            hatch=task_hatch["G1JoystickFlatTerrain"],
            label="Task:G1",
        ),
    ]
    fig.suptitle(
        f"Rollout Time by Backend and Task\n{get_device_info_line()}",
        y=0.965,
        fontsize=13,
    )
    fig.legend(
        handles=backend_handles + task_handles,
        fontsize=9,
        ncol=len(backend_handles + task_handles),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.895),
        frameon=True,
        handlelength=1.8,
        columnspacing=1.2,
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.89])
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Saved latency plot to {out_png}")


def _plot_fps(records: List[BenchRecord], out_png: Path, batch_sizes: List[int]):
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_title(f"Parallel Physics — Total FPS\n{get_device_info_line()}", fontsize=9)

    backend_linestyle = {"numpy": "--", "rollout_1t": "--"}

    for task_name in TASK_CONFIGS:
        for backend in sorted({r.backend for r in records if r.task == task_name}):
            recs = sorted(
                [r for r in records if r.task == task_name and r.backend == backend],
                key=lambda r: r.batch_size,
            )
            short = task_name.split("Joystick")[0]
            ax.plot(
                [r.batch_size for r in recs],
                [r.sps / 1e4 for r in recs],
                marker="o",
                linestyle=backend_linestyle.get(backend, "-"),
                label=f"{short} ({backend})",
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


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark parallel physics backends (rollout / mlx_step)"
    )
    parser.add_argument("--nstep", type=int, default=1)
    parser.add_argument("--nthread", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument(
        "--chunk-size", type=int, default=16, help="MLX step chunk size (macOS only)"
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="Go1JoystickFlatTerrain,Go2JoystickFlatTerrain,G1JoystickFlatTerrain",
    )
    parser.add_argument(
        "--batch-sizes", type=str, default=",".join(str(x) for x in DEFAULT_BATCH_SIZES)
    )
    parser.add_argument("--out-json", type=str, default="benchmark/outputs/mlx_step/results.json")
    parser.add_argument(
        "--out-dir",
        type=str,
        default="benchmark/outputs/mlx_step",
        help="Directory for output plots",
    )
    args = parser.parse_args()

    task_names = [x.strip() for x in args.tasks.split(",") if x.strip()]
    batch_sizes = [int(x.strip()) for x in args.batch_sizes.split(",") if x.strip()]
    for name in task_names:
        if name not in TASK_CONFIGS:
            raise ValueError(f"Unknown task '{name}'. Available: {list(TASK_CONFIGS.keys())}")

    if _IS_MACOS:
        if not _has_native_mujoco_mlx_step():
            raise RuntimeError("Native MLX step backend unavailable. Requires mujoco.mlx_step.")
        print(f"macOS: native mlx_step available")
    else:
        print(f"Non-macOS: using mujoco.rollout (1t vs {args.nthread}t)")

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

    if _IS_MACOS:
        backend_a, backend_b = "numpy", "mlx_native"
    else:
        backend_a, backend_b = "rollout_1t", f"rollout_{args.nthread}t"

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
            "mlx_output_shape_modes": sorted(
                {r.output_shape_mode for r in records if r.backend == "mlx_native"}
            )
            if _IS_MACOS
            else [],
        },
        "summary": {
            "speedup_a_over_b": {
                task: {
                    "backend_a": backend_a,
                    "backend_b": backend_b,
                    "geomean": _geomean(
                        [
                            next(
                                r.avg_time_sec
                                for r in records
                                if r.task == task and r.backend == backend_a and r.batch_size == b
                            )
                            / max(
                                next(
                                    r.avg_time_sec
                                    for r in records
                                    if r.task == task
                                    and r.backend == backend_b
                                    and r.batch_size == b
                                ),
                                1e-12,
                            )
                            for b in batch_sizes
                        ]
                    ),
                }
                for task in task_names
            }
        },
        "results": [asdict(r) for r in records],
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved results to {out_json}")

    out_dir = Path(args.out_dir)
    _plot_latency(records, out_dir / "latency.png", batch_sizes=batch_sizes, nthread=args.nthread)
    _plot_fps(records, out_dir / "fps.png", batch_sizes=batch_sizes)


if __name__ == "__main__":
    main()
