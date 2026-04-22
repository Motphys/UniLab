#!/usr/bin/env python3
# pyright: reportAttributeAccessIssue=false
"""
Benchmark detailed MuJoCo backend step overhead.

This benchmark mirrors the hot path inside `MujocoBackend.step()` and splits it
into:
    1. control broadcast (`set_ctrl`)
    2. `BatchEnvPool.step` (`pool_step`)
    3. physics-state cast/copy (`state_copy`)
    4. `BatchEnvPool.forward` (`forward`)
    5. sensor-data cast/copy (`sensor_copy`)

It sweeps current locomotion owner tasks across MuJoCo only, with environment
counts from 2^8 to 2^13 by default, and supports multiple `nstep` values so the
per-substep cost can also be compared.

Usage:
    uv run benchmark/benchmark_mujoco_backend_step_detail.py

    uv run benchmark/benchmark_mujoco_backend_step_detail.py \
        --tasks go1_joystick_flat,go2_joystick_flat,g1_walk_flat \
        --env-nums 256,512,1024,2048,4096,8192 \
        --nsteps 1,2,3,4

    uv run benchmark/benchmark_mujoco_backend_step_detail.py \
        --tasks g1_walk_flat --env-nums 2048 --nsteps 1,2,3 --iters 20
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time
from dataclasses import asdict, dataclass
from multiprocessing import cpu_count
from pathlib import Path
from typing import Sequence

import matplotlib
import mujoco
import numpy as np
from matplotlib.patches import Rectangle
from mujoco.batch_env import BatchEnvPool

from unilab.base.dtype_config import get_global_dtype
from unilab.utils.xml_utils import create_discardvisual_xml

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parents[1]
CORE_DIR = ROOT_DIR / "benchmark" / "core"


def _load_helper_module(module_name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(module_name, CORE_DIR / relative_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load helper module {module_name} from {relative_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_DEVICE_INFO = _load_helper_module("bench_device_info", "device_info.py")
_OUTPUT = _load_helper_module("bench_output", "output.py")
_TASK_NAMES = _load_helper_module("bench_task_names", "task_names.py")

get_device_info_dict = _DEVICE_INFO.get_device_info_dict
get_device_info_line = _DEVICE_INFO.get_device_info_line
save_json = _OUTPUT.save_json
canonical_locomotion_task_ids = _TASK_NAMES.canonical_locomotion_task_ids
locomotion_task_spec = _TASK_NAMES.locomotion_task_spec
normalize_locomotion_task_id = _TASK_NAMES.normalize_locomotion_task_id


DEFAULT_TASK_IDS = canonical_locomotion_task_ids()
DEFAULT_ENV_NUMS = [2**k for k in range(8, 14)]  # 256 .. 8192
DEFAULT_NSTEPS = [1, 2, 3, 4]
DEFAULT_WARMUP = 3
DEFAULT_ITERS = 10
DEFAULT_OUTPUT_JSON = (
    ROOT_DIR / "benchmark" / "outputs" / "mujoco_backend_step_detail" / "results.json"
)

TASK_COLORS = {
    "go1_joystick_flat": "#4C78A8",
    "go2_joystick_flat": "#54A24B",
    "g1_walk_flat": "#F58518",
}
COMPONENT_COLORS = {
    "set_ctrl_ms": "#9AA0A6",
    "pool_step_ms": "#4C78A8",
    "state_copy_ms": "#9EC5FE",
    "forward_ms": "#F58518",
    "sensor_copy_ms": "#FFBF79",
}


@dataclass
class BenchRecord:
    task: str
    env_num: int
    nstep: int
    nthread: int
    warmup: int
    iters: int
    control_dim: int
    state_dim: int
    sensor_dim: int
    set_ctrl_ms: float
    pool_step_ms: float
    state_copy_ms: float
    physics_total_ms: float
    forward_ms: float
    sensor_copy_ms: float
    refresh_total_ms: float
    backend_total_ms: float
    physics_per_substep_ms: float
    refresh_per_substep_ms: float
    total_per_substep_ms: float


def _parse_csv_ints(text: str) -> list[int]:
    values = [int(part.strip()) for part in text.split(",") if part.strip()]
    if not values:
        raise ValueError("Expected at least one integer value.")
    return values


def _parse_csv_tasks(text: str) -> list[str]:
    values = [
        normalize_locomotion_task_id(part.strip()) for part in text.split(",") if part.strip()
    ]
    if not values:
        raise ValueError("Expected at least one task id.")
    return values


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


def _load_discardvisual_model(model_file: str) -> mujoco.MjModel:
    model_path = create_discardvisual_xml(model_file)
    try:
        return mujoco.MjModel.from_xml_path(model_path)
    finally:
        os.remove(model_path)


def _control_limits(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray]:
    if model.nu == 0:
        empty = np.empty((0,), dtype=np.float64)
        return empty, empty

    low = np.full((model.nu,), -1.0, dtype=np.float64)
    high = np.full((model.nu,), 1.0, dtype=np.float64)
    if hasattr(model, "actuator_ctrllimited"):
        limited = np.asarray(model.actuator_ctrllimited, dtype=bool)
        if np.any(limited):
            ctrl_range = np.asarray(model.actuator_ctrlrange, dtype=np.float64)
            low[limited] = ctrl_range[limited, 0]
            high[limited] = ctrl_range[limited, 1]
    return low, high


def _sample_controls(
    env_num: int,
    ctrl_low: np.ndarray,
    ctrl_high: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    if ctrl_low.size == 0:
        return np.empty((env_num, 0), dtype=np.float64)
    controls = rng.uniform(ctrl_low, ctrl_high, size=(env_num, ctrl_low.shape[0]))
    return np.ascontiguousarray(controls, dtype=np.float64)


def _median_ms(samples: list[float]) -> float:
    return float(np.median(np.asarray(samples, dtype=np.float64)))


def _benchmark_one(
    task: str,
    env_num: int,
    nstep: int,
    warmup: int,
    iters: int,
    seed: int,
) -> BenchRecord:
    spec = locomotion_task_spec(task)
    cfg = spec.config_cls()
    model = _load_discardvisual_model(cfg.model_file)
    np_dtype = get_global_dtype()

    state0, _ = _keyframe0_state_and_ctrl(model)
    ctrl_low, ctrl_high = _control_limits(model)
    nthread = min(env_num, cpu_count() * 2)
    rng = np.random.default_rng(seed)

    total_iters = warmup + iters
    controls = [_sample_controls(env_num, ctrl_low, ctrl_high, rng) for _ in range(total_iters)]

    set_ctrl_samples: list[float] = []
    pool_step_samples: list[float] = []
    state_copy_samples: list[float] = []
    forward_samples: list[float] = []
    sensor_copy_samples: list[float] = []

    with BatchEnvPool(model, nbatch=env_num, nthread=nthread) as pool:
        physics_state = np.broadcast_to(state0.astype(np_dtype), (env_num, state0.shape[0])).copy()
        sensor_data = np.zeros((env_num, model.nsensordata), dtype=np_dtype)
        sensor_init = pool.forward(physics_state)
        sensor_data[:] = sensor_init.astype(np_dtype)

        for iteration_idx, ctrl in enumerate(controls):
            t0 = time.perf_counter()
            control_traj = np.broadcast_to(ctrl[:, None, :], (env_num, nstep, ctrl.shape[-1]))
            set_ctrl_ms = (time.perf_counter() - t0) * 1000.0

            t0 = time.perf_counter()
            state_np = pool.step(
                physics_state,
                nstep=nstep,
                control=control_traj,
                control_spec=int(mujoco.mjtState.mjSTATE_CTRL),
            )
            pool_step_ms = (time.perf_counter() - t0) * 1000.0

            t0 = time.perf_counter()
            physics_state[:] = state_np.astype(np_dtype)
            state_copy_ms = (time.perf_counter() - t0) * 1000.0

            t0 = time.perf_counter()
            sensor_np = pool.forward(physics_state)
            forward_ms = (time.perf_counter() - t0) * 1000.0

            t0 = time.perf_counter()
            sensor_data[:] = sensor_np.astype(np_dtype)
            sensor_copy_ms = (time.perf_counter() - t0) * 1000.0

            if iteration_idx >= warmup:
                set_ctrl_samples.append(set_ctrl_ms)
                pool_step_samples.append(pool_step_ms)
                state_copy_samples.append(state_copy_ms)
                forward_samples.append(forward_ms)
                sensor_copy_samples.append(sensor_copy_ms)

    set_ctrl_ms = _median_ms(set_ctrl_samples)
    pool_step_ms = _median_ms(pool_step_samples)
    state_copy_ms = _median_ms(state_copy_samples)
    forward_ms = _median_ms(forward_samples)
    sensor_copy_ms = _median_ms(sensor_copy_samples)
    physics_total_ms = pool_step_ms + state_copy_ms
    refresh_total_ms = forward_ms + sensor_copy_ms
    backend_total_ms = set_ctrl_ms + physics_total_ms + refresh_total_ms

    return BenchRecord(
        task=task,
        env_num=env_num,
        nstep=nstep,
        nthread=nthread,
        warmup=warmup,
        iters=iters,
        control_dim=model.nu,
        state_dim=state0.shape[0],
        sensor_dim=model.nsensordata,
        set_ctrl_ms=set_ctrl_ms,
        pool_step_ms=pool_step_ms,
        state_copy_ms=state_copy_ms,
        physics_total_ms=physics_total_ms,
        forward_ms=forward_ms,
        sensor_copy_ms=sensor_copy_ms,
        refresh_total_ms=refresh_total_ms,
        backend_total_ms=backend_total_ms,
        physics_per_substep_ms=physics_total_ms / nstep,
        refresh_per_substep_ms=refresh_total_ms / nstep,
        total_per_substep_ms=backend_total_ms / nstep,
    )


def _grouped_bar_positions(
    env_nums: Sequence[int],
    nsteps: Sequence[int],
    *,
    bar_width: float = 0.22,
    intra_gap: float = 0.06,
    group_gap: float = 0.48,
) -> tuple[dict[tuple[int, int], float], list[float], float]:
    positions: dict[tuple[int, int], float] = {}
    group_centers: list[float] = []
    cursor = 0.0
    stride = bar_width + intra_gap

    for env_num in env_nums:
        group_positions: list[float] = []
        for nstep in nsteps:
            positions[(env_num, nstep)] = cursor
            group_positions.append(cursor)
            cursor += stride
        group_centers.append(float(sum(group_positions) / len(group_positions)))
        cursor += group_gap

    return positions, group_centers, bar_width


def _decorate_grouped_env_axis(
    ax,
    env_nums: Sequence[int],
    nsteps: Sequence[int],
    group_centers: Sequence[float],
    positions: dict[tuple[int, int], float],
    bar_width: float,
) -> None:
    tick_positions = [positions[(env_num, nstep)] for env_num in env_nums for nstep in nsteps]
    tick_labels = [f"n={nstep}" for env_num in env_nums for nstep in nsteps]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=0, ha="center", fontsize=8)

    for idx, env_num in enumerate(env_nums):
        left = positions[(env_num, nsteps[0])] - bar_width * 0.7
        right = positions[(env_num, nsteps[-1])] + bar_width * 0.7
        ax.axvspan(left, right, color="#000000", alpha=0.025, zorder=0)
        ax.text(
            group_centers[idx],
            -0.16,
            str(env_num),
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=9,
            fontweight="bold",
        )
        if idx < len(env_nums) - 1:
            boundary = (right + positions[(env_nums[idx + 1], nsteps[0])] - bar_width * 0.7) / 2.0
            ax.axvline(boundary, color="#B0B0B0", linewidth=1.0, alpha=0.8)


def _plot_component_bars(
    records: Sequence[BenchRecord],
    output_path: Path,
    *,
    amortized_per_substep: bool,
) -> bool:
    if not records:
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tasks = [task for task in DEFAULT_TASK_IDS if any(record.task == task for record in records)]
    nsteps = sorted({record.nstep for record in records})
    env_nums = sorted({record.env_num for record in records})
    positions, group_centers, bar_width = _grouped_bar_positions(env_nums, nsteps)
    component_keys = [
        "set_ctrl_ms",
        "pool_step_ms",
        "state_copy_ms",
        "forward_ms",
        "sensor_copy_ms",
    ]

    fig, axes = plt.subplots(
        len(tasks), 1, figsize=(max(11.5, len(env_nums) * 1.9), 4.2 * len(tasks))
    )
    if len(tasks) == 1:
        axes = [axes]

    ylabel = (
        "amortized time per substep (ms)"
        if amortized_per_substep
        else "time per env.step call (ms)"
    )
    title_suffix = "amortized-per-substep" if amortized_per_substep else "total"

    for ax, task in zip(axes, tasks, strict=False):
        task_records = [record for record in records if record.task == task]
        bars = {(record.env_num, record.nstep): record for record in task_records}
        x_values = [positions[(env_num, nstep)] for env_num in env_nums for nstep in nsteps]
        cumulative = np.zeros(len(x_values), dtype=np.float64)
        for component_key in component_keys:
            values = []
            for env_num in env_nums:
                for nstep in nsteps:
                    record = bars.get((env_num, nstep))
                    if record is None:
                        values.append(0.0)
                        continue
                    value = getattr(record, component_key)
                    if amortized_per_substep:
                        value /= nstep
                    values.append(value)
            values_np = np.asarray(values, dtype=np.float64)
            ax.bar(
                x_values,
                values_np,
                width=bar_width,
                bottom=cumulative,
                label=component_key.removesuffix("_ms"),
                color=COMPONENT_COLORS[component_key],
                edgecolor="#444444",
                linewidth=0.5,
                alpha=0.95,
            )
            cumulative += values_np

        ax.set_title(f"{task} — {title_suffix}", fontsize=10)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
        _decorate_grouped_env_axis(ax, env_nums, nsteps, group_centers, positions, bar_width)

    component_handles = [
        Rectangle(
            (0, 0),
            1,
            1,
            facecolor=COMPONENT_COLORS[key],
            edgecolor="#444444",
            label=key.removesuffix("_ms"),
        )
        for key in component_keys
    ]
    axes[0].legend(
        handles=component_handles, ncol=min(3, len(component_handles)), fontsize=8, loc="upper left"
    )
    axes[-1].set_xlabel("num_envs")

    fig.suptitle(
        (
            "MuJoCo backend step detail — stacked component bars "
            "(amortized over nstep)\n"
            f"{get_device_info_line()}"
        )
        if amortized_per_substep
        else f"MuJoCo backend step detail — stacked component bars\n{get_device_info_line()}",
        fontsize=11,
    )
    fig.subplots_adjust(bottom=0.14, top=0.9, hspace=0.32)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path.resolve()}")
    return True


def _print_table(records: Sequence[BenchRecord]) -> None:
    if not records:
        print("No records.")
        return

    headers = (
        "task",
        "envs",
        "nstep",
        "backend_total",
        "physics_total",
        "refresh_total",
        "pool_step",
        "state_copy",
        "forward",
        "sensor_copy",
    )
    rows = []
    for record in records:
        rows.append(
            (
                record.task,
                str(record.env_num),
                str(record.nstep),
                f"{record.backend_total_ms:.3f}",
                f"{record.physics_total_ms:.3f}",
                f"{record.refresh_total_ms:.3f}",
                f"{record.pool_step_ms:.3f}",
                f"{record.state_copy_ms:.3f}",
                f"{record.forward_ms:.3f}",
                f"{record.sensor_copy_ms:.3f}",
            )
        )

    widths = [len(header) for header in headers]
    for row in rows:
        for idx, value in enumerate(row):
            widths[idx] = max(widths[idx], len(value))

    def _fmt(values: Sequence[str]) -> str:
        return " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(values))

    print()
    print(_fmt(headers))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(_fmt(row))


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark detailed MuJoCo backend step overhead.")
    parser.add_argument("--tasks", type=str, default=",".join(DEFAULT_TASK_IDS))
    parser.add_argument("--env-nums", type=str, default=",".join(str(v) for v in DEFAULT_ENV_NUMS))
    parser.add_argument("--nsteps", type=str, default=",".join(str(v) for v in DEFAULT_NSTEPS))
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--iters", type=int, default=DEFAULT_ITERS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--plot-dir", type=Path, default=None)
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args()

    tasks = _parse_csv_tasks(args.tasks)
    env_nums = _parse_csv_ints(args.env_nums)
    nsteps = _parse_csv_ints(args.nsteps)
    plot_dir = args.plot_dir if args.plot_dir is not None else args.out_json.parent

    print(f"tasks: {tasks}")
    print(f"env_nums: {env_nums}")
    print(f"nsteps: {nsteps}")
    print(f"warmup={args.warmup}, iters={args.iters}")

    records: list[BenchRecord] = []
    for task_idx, task in enumerate(tasks):
        for env_num in env_nums:
            for nstep in nsteps:
                seed = args.seed + task_idx * 100_000 + env_num * 17 + nstep * 997
                print(f"[run] task={task} envs={env_num} nstep={nstep}", flush=True)
                record = _benchmark_one(
                    task=task,
                    env_num=env_num,
                    nstep=nstep,
                    warmup=args.warmup,
                    iters=args.iters,
                    seed=seed,
                )
                records.append(record)
                print(
                    f"      total={record.backend_total_ms:.3f}ms "
                    f"physics={record.physics_total_ms:.3f}ms "
                    f"refresh={record.refresh_total_ms:.3f}ms "
                    f"per_substep={record.total_per_substep_ms:.3f}ms"
                )

    _print_table(records)

    plot_files: list[str] = []
    if not args.skip_plots:
        total_plot = plot_dir / "component_breakdown_total_ms.png"
        amortized_plot = plot_dir / "component_breakdown_amortized_per_substep_ms.png"
        if _plot_component_bars(records, total_plot, amortized_per_substep=False):
            plot_files.append(str(total_plot.resolve()))
        if _plot_component_bars(records, amortized_plot, amortized_per_substep=True):
            plot_files.append(str(amortized_plot.resolve()))

    save_json(
        args.out_json,
        [asdict(record) for record in records],
        {
            "device_info": get_device_info_dict(),
            "xml_preprocess": "create_discardvisual_xml",
            "tasks": tasks,
            "env_nums": env_nums,
            "nsteps": nsteps,
            "warmup": args.warmup,
            "iters": args.iters,
            "dtype": str(get_global_dtype()),
            "plot_files": plot_files,
        },
    )


if __name__ == "__main__":
    main()
