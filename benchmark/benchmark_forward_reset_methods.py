#!/usr/bin/env python3
"""Benchmark C++ batch forward vs Python forward methods."""

import argparse
import sys
from dataclasses import asdict, dataclass
from multiprocessing import cpu_count
from pathlib import Path

import mujoco
import numpy as np
from mujoco.batch_env import BatchEnvPool

try:
    from benchmark.core.device_info import get_device_info_dict, get_device_info_line
except ModuleNotFoundError:
    from core.device_info import get_device_info_dict, get_device_info_line

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

try:
    from benchmark.core.task_names import (
        canonical_locomotion_task_ids,
        locomotion_task_spec,
        normalize_locomotion_task_id,
    )
except ModuleNotFoundError:
    from core.task_names import (
        canonical_locomotion_task_ids,
        locomotion_task_spec,
        normalize_locomotion_task_id,
    )

CONSISTENCY_TASKS = canonical_locomotion_task_ids()
DEFAULT_RESET_TASK = "go2_joystick_flat"


@dataclass
class ConsistencyRecord:
    task: str
    batch_size: int
    max_abs_diff: float
    mean_abs_diff: float
    allclose: bool


@dataclass
class SpeedRecord:
    task: str
    method: str
    env_num: int
    elapsed_sec: float
    us_per_env: float


def load_model_for_task(task: str) -> mujoco.MjModel:
    task_key = normalize_locomotion_task_id(task)
    return mujoco.MjModel.from_xml_path(locomotion_task_spec(task_key).config_cls().model_file)


def _set_state_for_forward(model, data, state):
    mujoco.mj_setState(model, data, state, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    data.ctrl[:] = 0.0
    data.qfrc_applied[:] = 0.0
    data.xfrc_applied[:] = 0.0
    data.qacc_warmstart[:] = 0.0


def python_forward_for_loop(model, states):
    out = np.empty((states.shape[0], model.nsensordata), dtype=np.float64)
    data = mujoco.MjData(model)
    for i in range(states.shape[0]):
        _set_state_for_forward(model, data, states[i])
        mujoco.mj_forward(model, data)
        out[i] = data.sensordata
    return out


def python_forward_chunk_loop(model, states, chunk_size):
    out = np.empty((states.shape[0], model.nsensordata), dtype=np.float64)
    data = mujoco.MjData(model)
    for start in range(0, states.shape[0], chunk_size):
        end = min(start + chunk_size, states.shape[0])
        for i in range(start, end):
            _set_state_for_forward(model, data, states[i])
            mujoco.mj_forward(model, data)
            out[i] = data.sensordata
    return out


def batch_env_forward(pool: BatchEnvPool, states: np.ndarray) -> np.ndarray:
    sensordata = pool.forward(states)
    return np.asarray(sensordata, dtype=np.float64)


def collect_random_reset_states(task, batch_size, random_rounds, seed):
    model = load_model_for_task(task)
    rng = np.random.default_rng(seed)
    data = mujoco.MjData(model)
    nstate = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS)

    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    else:
        mujoco.mj_resetData(model, data)

    ctrl_low = np.full((model.nu,), -1.0)
    ctrl_high = np.full((model.nu,), 1.0)
    if model.nu > 0 and hasattr(model, "actuator_ctrllimited"):
        limited = np.asarray(model.actuator_ctrllimited, dtype=bool)
        if np.any(limited):
            ranges = np.asarray(model.actuator_ctrlrange)
            ctrl_low[limited] = ranges[limited, 0]
            ctrl_high[limited] = ranges[limited, 1]

    states = np.empty((batch_size, nstate))
    round_steps = max(2, int(random_rounds))
    for i in range(batch_size):
        if i % round_steps == 0:
            if model.nkey > 0:
                mujoco.mj_resetDataKeyframe(model, data, 0)
            else:
                mujoco.mj_resetData(model, data)

        nsteps = int(rng.integers(1, round_steps + 1))
        for _ in range(nsteps):
            if model.nu > 0:
                data.ctrl[:] = rng.uniform(ctrl_low, ctrl_high)
            mujoco.mj_step(model, data)

        state_buf = np.empty((nstate,))
        mujoco.mj_getState(model, data, state_buf, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        states[i] = state_buf

    return model, states


def run_consistency(tasks, batch_size, random_rounds, seed, atol, rtol, chunk_size):
    _ = chunk_size  # BatchEnvPool.forward does not expose chunk-size tuning.
    records = []
    for task in tasks:
        task_key = normalize_locomotion_task_id(task)
        model, states = collect_random_reset_states(
            task_key, batch_size, random_rounds, seed + abs(hash(task_key)) % 10000
        )
        nthread = min(batch_size, cpu_count())
        py_sensor = python_forward_for_loop(model, states)
        with BatchEnvPool(model, nbatch=states.shape[0], nthread=nthread) as pool:
            batch_sensor = batch_env_forward(pool, states)
        diff = np.abs(py_sensor - batch_sensor)
        max_abs = float(np.max(diff))
        mean_abs = float(np.mean(diff))
        ok = bool(np.allclose(py_sensor, batch_sensor, atol=atol, rtol=rtol))
        records.append(ConsistencyRecord(task_key, batch_size, max_abs, mean_abs, ok))
        print(
            f"[Consistency] {task_key}: allclose={ok}, max_abs={max_abs:.3e}, mean_abs={mean_abs:.3e}"
        )
    return records


def _bench_method(func, repeats):
    import time

    samples = [time.perf_counter() or func() or time.perf_counter() for _ in range(repeats)]
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        func()
        samples.append(time.perf_counter() - t0)
    return float(np.median(samples))


def run_reset_speed(task, env_nums, random_rounds, chunk_size, repeats, seed):
    _ = chunk_size  # BatchEnvPool.forward does not expose chunk-size tuning.
    task_key = normalize_locomotion_task_id(task)
    records = []
    for env_num in env_nums:
        model, states = collect_random_reset_states(
            task_key, env_num, random_rounds, seed + env_num
        )
        nthread = min(env_num, cpu_count())

        t_for = _bench_method(lambda: python_forward_for_loop(model, states), repeats)
        t_chunk = _bench_method(
            lambda: python_forward_chunk_loop(model, states, chunk_size), repeats
        )
        with BatchEnvPool(model, nbatch=states.shape[0], nthread=nthread) as pool:
            t_batch = _bench_method(lambda: batch_env_forward(pool, states), repeats)

        for method, elapsed in [
            ("for_loop", t_for),
            ("chunk_for_loop", t_chunk),
            ("batch_env_forward", t_batch),
        ]:
            records.append(
                SpeedRecord(task_key, method, env_num, elapsed, elapsed * 1e6 / max(env_num, 1))
            )

        print(
            f"[Speed] {task_key} env={env_num}: for={t_for * 1e3:.3f}ms, chunk={t_chunk * 1e3:.3f}ms, batch_env={t_batch * 1e3:.3f}ms"
        )
    return records


def plot_speed(records, out_png, reset_task):
    out_png.parent.mkdir(parents=True, exist_ok=True)
    env_nums = sorted({r.env_num for r in records})
    methods = ["for_loop", "chunk_for_loop", "batch_env_forward"]
    colors = {"for_loop": "#3B82F6", "chunk_for_loop": "#10B981", "batch_env_forward": "#F59E0B"}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharex=True)
    x = np.arange(len(env_nums))
    width = 0.24

    for i, method in enumerate(methods):
        total_ms = [
            next(r.elapsed_sec for r in records if r.method == method and r.env_num == n) * 1e3
            for n in env_nums
        ]
        throughput = [
            n
            / max(
                next(r.elapsed_sec for r in records if r.method == method and r.env_num == n), 1e-12
            )
            for n in env_nums
        ]
        offset = (i - 1) * width
        ax1.bar(x + offset, total_ms, width, label=method, color=colors[method], alpha=0.95)
        ax2.bar(x + offset, throughput, width, label=method, color=colors[method], alpha=0.95)

    for ax in (ax1, ax2):
        ax.set_xticks(x)
        ax.set_xticklabels([str(v) for v in env_nums])
        ax.set_xlabel("num_env")
        ax.grid(axis="y", alpha=0.3)

    ax1.set_yscale("log")
    ax1.set_ylabel("total time (ms)")
    ax1.set_title("Total Time")
    ax2.set_yscale("log")
    ax2.set_ylabel("throughput (env/s)")
    ax2.set_title("Throughput")
    fig.suptitle(
        f"Reset-forward method speed on {normalize_locomotion_task_id(reset_task)}\n{get_device_info_line()}"
    )
    ax2.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Benchmark C++ batch forward vs Python forward")
    parser.add_argument("--consistency-tasks", type=str, default=",".join(CONSISTENCY_TASKS))
    parser.add_argument("--reset-task", type=str, default=DEFAULT_RESET_TASK)
    parser.add_argument("--consistency-batch", type=int, default=1024)
    parser.add_argument("--env-nums", type=str, default="256,512,1024,2048,4096,8192,16384")
    parser.add_argument("--random-rounds", type=int, default=8)
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=64,
        help="Legacy no-op retained for CLI compatibility; BatchEnvPool.forward does not use chunk_size.",
    )
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--atol", type=float, default=1e-9)
    parser.add_argument("--rtol", type=float, default=1e-8)
    parser.add_argument(
        "--out-json", type=str, default="benchmark/outputs/reset_forward_batch/results.json"
    )
    parser.add_argument(
        "--out-png", type=str, default="benchmark/outputs/reset_forward_batch/speed_plot.png"
    )
    args = parser.parse_args()

    consistency_tasks = [
        normalize_locomotion_task_id(t) for t in args.consistency_tasks.split(",") if t.strip()
    ]
    reset_task = normalize_locomotion_task_id(args.reset_task)
    env_nums = [int(x.strip()) for x in args.env_nums.split(",") if x.strip()]

    consistency = run_consistency(
        consistency_tasks,
        args.consistency_batch,
        args.random_rounds,
        args.seed,
        args.atol,
        args.rtol,
        args.chunk_size,
    )
    speed = run_reset_speed(
        reset_task, env_nums, args.random_rounds, args.chunk_size, args.repeats, args.seed
    )

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    import json
    from datetime import datetime, timezone

    with out_json.open("w") as f:
        json.dump(
            {
                "meta": {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "device_info": get_device_info_dict(),
                    "consistency_tasks": consistency_tasks,
                    "reset_task": reset_task,
                    "env_nums": env_nums,
                    "chunk_size": args.chunk_size,
                },
                "consistency": [asdict(r) for r in consistency],
                "speed": [asdict(r) for r in speed],
            },
            f,
            indent=2,
        )

    plot_speed(speed, Path(args.out_png), reset_task=reset_task)
    print(f"Consistency: {all(r.allclose for r in consistency)}")
    print(f"Saved: {out_json}, {args.out_png}")


if __name__ == "__main__":
    main()
