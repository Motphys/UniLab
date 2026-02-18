#!/usr/bin/env python3
"""Benchmark rollout time on Unilab locomotion tasks."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import mujoco
from mujoco import rollout as mj_rollout
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

import mlx.core as mx

try:
    from mujoco import mlx_step as mj_mlx_step
except Exception:
    mj_mlx_step = None
from unilab.envs.locomotion.g1.joystick import G1JoystickCfg
from unilab.envs.locomotion.go1.joystick import Go1JoystickCfg
from unilab.envs.locomotion.go2.joystick import Go2JoystickCfg
try:
    from benchmark.device_info import get_device_info_dict, get_device_info_line
except ModuleNotFoundError:
    from device_info import get_device_info_dict, get_device_info_line


@dataclass
class BenchRecord:
    task: str
    backend: str
    batch_size: int
    nstep: int
    nthread: int
    avg_time_sec: float


TASK_CONFIGS = {
    "Go1JoystickFlatTerrain": Go1JoystickCfg,
    "Go2JoystickFlatTerrain": Go2JoystickCfg,
    "G1JoystickFlatTerrain": G1JoystickCfg,
}
DEFAULT_BATCH_SIZES = [2**k for k in range(8, 14)]  # 2^8 ... 2^13


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
    initial_state,
    control,
    state_buf,
    sensordata_buf,
    niter: int,
) -> float:
    t0 = time.perf_counter()
    for _ in range(niter):
        runner.rollout(
            model_list,
            data_list,
            initial_state,
            control,
            nstep=1,
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
) -> float:
    t0 = time.perf_counter()
    for _ in range(niter):
        out = runner.rollout(
            model=model_list,
            data=data_list,
            initial_state=initial_state_mx,
            control=control_mx,
            nstep=nstep,
            out_dtype=mx.float32,
        )
        mx.eval(out.state_mx, out.sensordata_mx)
    return (time.perf_counter() - t0) / niter


def _has_native_mujoco_mlx_step() -> bool:
    return mj_mlx_step is not None and hasattr(mj_mlx_step, "RolloutMLX")


def _load_task_model(task_name: str) -> mujoco.MjModel:
    cfg_cls = TASK_CONFIGS[task_name]
    cfg = cfg_cls()
    return mujoco.MjModel.from_xml_path(cfg.model_file)


def _bench_one_task(
    task_name: str,
    batch_sizes: List[int],
    nstep: int,
    nthread: int,
    warmup: int,
    iters: int,
) -> List[BenchRecord]:
    np.random.seed(42)
    model = _load_task_model(task_name)
    nstate = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    state0, ctrl0 = _keyframe0_state_and_ctrl(model)

    records: List[BenchRecord] = []
    for batch_size in batch_sizes:
        model_list = [model] * batch_size
        data_list = [mujoco.MjData(model) for _ in range(nthread)]
        initial_state = np.empty((batch_size, nstate), dtype=np.float64)
        initial_state[:] = state0
        control = np.empty((batch_size, nstep, model.nu), dtype=np.float64)
        control[:] = ctrl0.reshape((1, 1, model.nu))
        state_buf = np.empty((batch_size, nstep, nstate), dtype=np.float64)
        sensordata_buf = np.empty((batch_size, nstep, model.nsensordata), dtype=np.float64)
        with mj_rollout.Rollout(nthread=nthread) as numpy_runner, mj_mlx_step.RolloutMLX(
            nthread=nthread
        ) as mlx_runner:
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
            )

            numpy_t = _run_numpy(
                numpy_runner,
                model_list,
                data_list,
                initial_state,
                control,
                state_buf,
                sensordata_buf,
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
            )

        records.append(
            BenchRecord(
                task=task_name,
                backend="numpy",
                batch_size=batch_size,
                nstep=nstep,
                nthread=nthread,
                avg_time_sec=numpy_t,
            )
        )
        records.append(
            BenchRecord(
                task=task_name,
                backend="mlx_native",
                batch_size=batch_size,
                nstep=nstep,
                nthread=nthread,
                avg_time_sec=mlx_t,
            )
        )
        print(
            f"[{task_name}] batch={batch_size:4d} "
            f"numpy={numpy_t*1000:.3f}ms "
            f"mlx(native)={mlx_t*1000:.3f}ms"
        )
    return records


def _plot(records: List[BenchRecord], out_png: Path, batch_sizes: List[int]):
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(13, 7))

    backend_colors = {
        "numpy": "#4C78A8",      # blue
        "mlx_native": "#F58518",  # orange
    }
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
    pair_inner_gap = 0.02    # 同一 task 下两根柱之间的间隙
    task_group_gap = 0.09    # 不同 task 对之间的间隙

    pair_span = 2 * bar_width + pair_inner_gap
    total_span = len(task_names) * pair_span + (len(task_names) - 1) * task_group_gap
    left_edge = -0.5 * total_span

    value_map = {
        (r.task, r.backend, r.batch_size): r.avg_time_sec * 1000.0 for r in records
    }

    for task_idx, task_name in enumerate(task_names):
        pair_start = left_edge + task_idx * (pair_span + task_group_gap)
        numpy_offset = pair_start + 0.5 * bar_width
        mlx_offset = pair_start + bar_width + pair_inner_gap + 0.5 * bar_width

        for backend, offset in (("numpy", numpy_offset), ("mlx_native", mlx_offset)):
            y = [value_map[(task_name, backend, b)] for b in batch_sizes]
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
        Patch(
            facecolor=backend_colors["numpy"],
            edgecolor="black",
            label="Backend:NumPy",
        ),
        Patch(
            facecolor=backend_colors["mlx_native"],
            edgecolor="black",
            label="Backend:MLX",
        ),
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
    all_handles = backend_handles + task_handles
    fig.suptitle(
        f"Rollout Time by Backend and Task (Grouped Bars)\n{get_device_info_line()}",
        y=0.965,
        fontsize=13,
    )
    fig.legend(
        handles=all_handles,
        fontsize=9,
        ncol=len(all_handles),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.895),
        frameon=True,
        handlelength=1.8,
        columnspacing=1.2,
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.89])
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Saved plot to {out_png}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nstep", type=int, default=1)
    parser.add_argument("--nthread", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument(
        "--tasks",
        type=str,
        default="Go1JoystickFlatTerrain,Go2JoystickFlatTerrain,G1JoystickFlatTerrain",
        help="Comma separated task names.",
    )
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default=",".join(str(x) for x in DEFAULT_BATCH_SIZES),
        help="Comma separated batch sizes (default: 2^8..2^13).",
    )
    parser.add_argument(
        "--out-json",
        type=str,
        default="benchmark/outputs/rollout/rollout_backend_results.json",
        help="Output JSON path.",
    )
    parser.add_argument(
        "--out-plot",
        type=str,
        default="benchmark/outputs/rollout/rollout_backend_time.png",
        help="Output plot path.",
    )
    args = parser.parse_args()
    if args.nstep != 1:
        raise ValueError("This benchmark is fixed to nstep=1 (same as mj_env usage).")

    task_names = [x.strip() for x in args.tasks.split(",") if x.strip()]
    batch_sizes = [int(x.strip()) for x in args.batch_sizes.split(",") if x.strip()]
    for name in task_names:
        if name not in TASK_CONFIGS:
            raise ValueError(
                f"Unknown task '{name}'. Available: {list(TASK_CONFIGS.keys())}"
            )
    if not _has_native_mujoco_mlx_step():
        raise RuntimeError(
            "Native MLX step backend is unavailable. "
            "This benchmark now requires native mujoco.mlx_step."
        )

    print(f"MLX native step available: {_has_native_mujoco_mlx_step()}")
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
            "native_mlx_step_available": _has_native_mujoco_mlx_step(),
        },
        "results": [asdict(r) for r in records],
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved results to {out_json}")

    _plot(records, Path(args.out_plot), batch_sizes=batch_sizes)


if __name__ == "__main__":
    main()
