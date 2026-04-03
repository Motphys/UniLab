#!/usr/bin/env python3
"""
Benchmark MuJoCo rollout throughput for two batching strategies:

1. UniLab backend style: one shared MjModel broadcast across the batch.
2. Per-env model style: one distinct MjModel per env in the batch.

This isolates rollout throughput only. Model construction is done before timing.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import resource
import subprocess
import sys
import time
from multiprocessing import cpu_count
from pathlib import Path
from typing import Sequence

import numpy as np

try:
    import mujoco
    import mujoco.rollout
except ImportError:
    mujoco = None


def _resolve_default_xml() -> str:
    return str(Path(__file__).with_name("xmls") / "humanoid" / "humanoid.xml")


def _resolve_xml_path(xml_arg: str) -> str:
    xml_path = Path(xml_arg)
    if xml_path.is_absolute():
        return str(xml_path)
    return str((Path.cwd() / xml_path).resolve())


def _default_nthread(batch_size: int) -> int:
    return min(batch_size, cpu_count() * 2)


def _rss_bytes() -> int:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return int(usage)
    return int(usage) * 1024


def _mb_str(value_bytes: int) -> str:
    return f"{value_bytes / (1024 * 1024):.1f}"


def _initial_state_and_ctrl(
    model: "mujoco.MjModel", batch_size: int
) -> tuple[np.ndarray, np.ndarray]:
    data = mujoco.MjData(model)
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    else:
        mujoco.mj_resetData(model, data)

    nstate = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    state0 = np.empty((nstate,), dtype=np.float64)
    mujoco.mj_getState(model, data, state0, mujoco.mjtState.mjSTATE_FULLPHYSICS)

    initial_state = np.empty((batch_size, nstate), dtype=np.float64)
    initial_state[:] = state0

    ctrl0 = np.zeros((model.nu,), dtype=np.float64)
    if model.nkey > 0 and model.nu > 0:
        ctrl0[:] = np.asarray(model.key_ctrl[0], dtype=np.float64)

    control = np.empty((batch_size, 1, model.nu), dtype=np.float64)
    control[:] = ctrl0.reshape((1, 1, model.nu))
    return initial_state, control


def _build_model_batch(
    xml_path: str, batch_size: int, distinct_models: bool
) -> tuple["mujoco.MjModel | Sequence[mujoco.MjModel]", "mujoco.MjModel"]:
    base_model = mujoco.MjModel.from_xml_path(xml_path)
    if not distinct_models:
        return base_model, base_model

    model_batch = [mujoco.MjModel.from_xml_path(xml_path) for _ in range(batch_size)]
    return model_batch, model_batch[0]


def run_rollout_benchmark(
    xml_path: str,
    batch_size: int,
    steps: int,
    nthread: int,
    warmup: int,
    distinct_models: bool,
) -> tuple[float, float]:
    if mujoco is None or not hasattr(mujoco, "rollout"):
        raise RuntimeError("MuJoCo rollout is unavailable in the current environment")

    model_batch, worker_model = _build_model_batch(
        xml_path=xml_path, batch_size=batch_size, distinct_models=distinct_models
    )
    worker_data = [mujoco.MjData(worker_model) for _ in range(nthread)]
    initial_state, control = _initial_state_and_ctrl(worker_model, batch_size)
    nstate = initial_state.shape[-1]
    nsensordata = worker_model.nsensordata

    state_buf = np.empty((batch_size, 1, nstate), dtype=np.float64)
    sensor_buf = np.empty((batch_size, 1, nsensordata), dtype=np.float64)

    with mujoco.rollout.Rollout(nthread=nthread) as runner:
        for _ in range(warmup):
            state_traj, _ = runner.rollout(
                model_batch,
                worker_data,
                initial_state,
                control,
                nstep=1,
                state=state_buf,
                sensordata=sensor_buf,
            )
            initial_state[:] = state_traj[:, -1, :]

        start = time.perf_counter()
        for _ in range(steps):
            state_traj, _ = runner.rollout(
                model_batch,
                worker_data,
                initial_state,
                control,
                nstep=1,
                state=state_buf,
                sensordata=sensor_buf,
            )
            initial_state[:] = state_traj[:, -1, :]
        elapsed = max(time.perf_counter() - start, 1e-9)

    sps = (batch_size * steps) / elapsed
    return elapsed, sps


def collect_mode_metrics(
    xml_path: str,
    batch_size: int,
    steps: int,
    nthread: int,
    warmup: int,
    distinct_models: bool,
) -> dict[str, int | float | str]:
    if mujoco is None or not hasattr(mujoco, "rollout"):
        raise RuntimeError("MuJoCo rollout is unavailable in the current environment")

    gc.collect()
    rss_before = _rss_bytes()

    build_start = time.perf_counter()
    model_batch, worker_model = _build_model_batch(
        xml_path=xml_path, batch_size=batch_size, distinct_models=distinct_models
    )
    build_elapsed = max(time.perf_counter() - build_start, 1e-9)
    rss_after_model_build = _rss_bytes()

    setup_start = time.perf_counter()
    worker_data = [mujoco.MjData(worker_model) for _ in range(nthread)]
    initial_state, control = _initial_state_and_ctrl(worker_model, batch_size)
    nstate = initial_state.shape[-1]
    nsensordata = worker_model.nsensordata
    state_buf = np.empty((batch_size, 1, nstate), dtype=np.float64)
    sensor_buf = np.empty((batch_size, 1, nsensordata), dtype=np.float64)
    setup_elapsed = max(time.perf_counter() - setup_start, 1e-9)
    rss_after_setup = _rss_bytes()
    peak_rss = max(rss_before, rss_after_model_build, rss_after_setup)

    with mujoco.rollout.Rollout(nthread=nthread) as runner:
        for _ in range(warmup):
            state_traj, _ = runner.rollout(
                model_batch,
                worker_data,
                initial_state,
                control,
                nstep=1,
                state=state_buf,
                sensordata=sensor_buf,
            )
            initial_state[:] = state_traj[:, -1, :]
        peak_rss = max(peak_rss, _rss_bytes())

        start = time.perf_counter()
        for _ in range(steps):
            state_traj, _ = runner.rollout(
                model_batch,
                worker_data,
                initial_state,
                control,
                nstep=1,
                state=state_buf,
                sensordata=sensor_buf,
            )
            initial_state[:] = state_traj[:, -1, :]
        elapsed = max(time.perf_counter() - start, 1e-9)
        peak_rss = max(peak_rss, _rss_bytes())

    sps = (batch_size * steps) / elapsed
    return {
        "mode": "distinct_batch_models" if distinct_models else "shared_single_model",
        "models": batch_size if distinct_models else 1,
        "batch": batch_size,
        "threads": nthread,
        "build_sec": build_elapsed,
        "setup_sec": setup_elapsed,
        "rollout_sec": elapsed,
        "sps": sps,
        "rss_before_bytes": rss_before,
        "rss_after_model_build_bytes": rss_after_model_build,
        "rss_after_setup_bytes": rss_after_setup,
        "peak_rss_bytes": peak_rss,
        "rss_model_delta_bytes": rss_after_model_build - rss_before,
        "rss_setup_delta_bytes": rss_after_setup - rss_before,
    }


def _run_isolated_mode(
    script_path: Path,
    xml_path: str,
    batch_size: int,
    steps: int,
    warmup: int,
    nthread: int,
    mode: str,
) -> dict[str, int | float | str]:
    cmd = [
        sys.executable,
        str(script_path),
        "--xml",
        xml_path,
        "--num-envs",
        str(batch_size),
        "--steps",
        str(steps),
        "--warmup",
        str(warmup),
        "--nthread",
        str(nthread),
        "--mode",
        mode,
        "--emit-json",
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark MuJoCo rollout: shared single model vs per-env distinct models"
    )
    parser.add_argument(
        "--xml",
        type=str,
        default=_resolve_default_xml(),
        help="Path to XML model.",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=1024,
        help="Number of environments (batch size).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="Timed rollout iterations.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Warmup rollout iterations.",
    )
    parser.add_argument(
        "--nthread",
        type=int,
        default=None,
        help="MuJoCo rollout worker threads. Defaults to UniLab backend policy: min(num_envs, cpu_count * 2).",
    )
    parser.add_argument(
        "--mode",
        choices=["both", "shared_single_model", "distinct_batch_models"],
        default="both",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--emit-json",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    xml_path = _resolve_xml_path(args.xml)
    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"Model file not found: {xml_path}")

    if mujoco is None or not hasattr(mujoco, "rollout"):
        raise RuntimeError("MuJoCo rollout is unavailable in the current environment")

    nthread = args.nthread if args.nthread is not None else _default_nthread(args.num_envs)

    if args.mode != "both":
        result = collect_mode_metrics(
            xml_path=xml_path,
            batch_size=args.num_envs,
            steps=args.steps,
            nthread=nthread,
            warmup=args.warmup,
            distinct_models=args.mode == "distinct_batch_models",
        )
        if args.emit_json:
            print(json.dumps(result))
            return

        print(json.dumps(result, indent=2))
        return

    script_path = Path(__file__).resolve()
    results = [
        _run_isolated_mode(
            script_path=script_path,
            xml_path=xml_path,
            batch_size=args.num_envs,
            steps=args.steps,
            warmup=args.warmup,
            nthread=nthread,
            mode=mode,
        )
        for mode in ("shared_single_model", "distinct_batch_models")
    ]

    print(f"Model file: {xml_path}")
    print(f"Number of environments: {args.num_envs}")
    print(f"Warmup iterations: {args.warmup}")
    print(f"Timed iterations: {args.steps}")
    print(f"Worker threads: {nthread}")
    print()
    print("UniLab MuJoCo backend parallelism:")
    print("- one shared MjModel for all envs")
    print("- nthread worker MjData objects, not num_envs worker objects")
    print("- batched full-physics states passed into mujoco.rollout.Rollout")
    print()
    print(
        f"{'Mode':<24} | {'Models':<8} | {'Build(s)':<8} | {'RSS+Model':<10} | {'RSS+Setup':<10} | {'PeakRSS':<8} | {'SPS':<12} | {'Time(s)':<8}"
    )
    print("-" * 116)

    for result in results:
        print(
            f"{result['mode']:<24} | "
            f"{result['models']:<8} | "
            f"{result['build_sec']:<8.4f} | "
            f"{_mb_str(int(result['rss_model_delta_bytes'])):<10} | "
            f"{_mb_str(int(result['rss_setup_delta_bytes'])):<10} | "
            f"{_mb_str(int(result['peak_rss_bytes'])):<8} | "
            f"{result['sps']:<12.1f} | "
            f"{result['rollout_sec']:<8.4f}"
        )

    shared = results[0]
    distinct = results[1]

    print()
    print(
        "Relative throughput: "
        f"shared/distinct = {float(shared['sps']) / float(distinct['sps']):.3f}x, "
        f"distinct/shared time = {float(distinct['rollout_sec']) / float(shared['rollout_sec']):.3f}x"
    )
    print(
        "Model build time ratio: "
        f"distinct/shared = {float(distinct['build_sec']) / float(shared['build_sec']):.3f}x"
    )
    print(
        "Setup memory delta ratio: "
        f"distinct/shared = "
        f"{float(distinct['rss_setup_delta_bytes']) / max(float(shared['rss_setup_delta_bytes']), 1.0):.3f}x"
    )
    print("RSS uses per-process peak resident memory (`ru_maxrss`) inside each isolated mode.")


if __name__ == "__main__":
    main()
