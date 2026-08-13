"""MuJoCo NpEnv multi-shard CPU throughput benchmark (issue #960).

Splits a fixed global number of env rows evenly across ``S = 1/2/4/...``
shards. Each shard runs in its own process, pins itself to a disjoint CPU set
(``os.sched_setaffinity``) *before* creating its NpEnv, and then steps a real
MuJoCo NpEnv (``g1_walk_flat`` / ``g1_motion_tracking``) with NumPy actions
prepared outside the timed window. The timed window covers only the full
NpEnv ``step/reset/postprocess`` hot path.

Total throughput is ``global env rows x measured steps / paired wall time``,
where paired wall time spans from the synchronized start signal until the last
shard reports done — so shard-wait losses are included in the scaling numbers.

No learner / actor / torch / CUDA / runner lifecycle is involved.

Run:
    uv run benchmark/env/benchmark_np_env_shard_throughput.py

    # subset + tuning:
    uv run benchmark/env/benchmark_np_env_shard_throughput.py \
        --tasks g1_walk_flat --num-envs 8192 --shards 1,2,4 \
        --warmup-steps 10 --measured-steps 100 --repeats 1 \
        --out-json tmp/np_env_shard.json
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import queue
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Sequence

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from benchmark.core.device_info import get_device_info_dict, get_device_info_line
from benchmark.core.output import print_table, save_json

DEFAULT_OUTPUT_JSON = (
    ROOT_DIR / "benchmark" / "outputs" / "np_env_shard_throughput" / "results.json"
)

BACKEND = "mujoco"
# Task keys of benchmark.env.benchmark_env_step.TASK_CONFIGS covered by this
# benchmark, mapped to their owner task id (imported lazily in shard workers).
TASK_KEYS = {"g1_walk_flat": "g1", "g1_mt": "g1_mt", "g1_motion_tracking": "g1_mt", "g1": "g1"}
TASK_IDS = {"g1": "g1_walk_flat", "g1_mt": "g1_motion_tracking"}

DEFAULT_TASKS = "g1_walk_flat,g1_motion_tracking"
DEFAULT_NUM_ENVS = "8192,32768"
DEFAULT_SHARDS = "1,2,4"
DEFAULT_WARMUP_STEPS = 10
DEFAULT_MEASURED_STEPS = 50
DEFAULT_REPEATS = 1
DEFAULT_SEED = 0
DEFAULT_READY_TIMEOUT_S = 1800.0
DEFAULT_DONE_TIMEOUT_S = 1800.0


# =====================================================================
# Pure helpers (unit-tested in tests/benchmark/)
# =====================================================================


def shard_env_rows(total_envs: int, num_shards: int) -> list[int]:
    """Split ``total_envs`` rows into ``num_shards`` contiguous partitions.

    Every row is assigned exactly once (remainder goes to the first shards),
    so ``sum(result) == total_envs`` and the row ranges do not overlap.
    """
    if total_envs < 1:
        raise ValueError(f"total_envs must be >= 1, got {total_envs}")
    if not 1 <= num_shards <= total_envs:
        raise ValueError(f"num_shards must be in [1, {total_envs}], got {num_shards}")
    base, remainder = divmod(total_envs, num_shards)
    return [base + (1 if i < remainder else 0) for i in range(num_shards)]


def shard_cpu_sets(available_cpus: Sequence[int], num_shards: int) -> list[frozenset[int]]:
    """Split available CPU ids into ``num_shards`` disjoint contiguous sets."""
    cpus = sorted(set(available_cpus))
    if num_shards < 1:
        raise ValueError(f"num_shards must be >= 1, got {num_shards}")
    if num_shards > len(cpus):
        raise ValueError(f"num_shards={num_shards} exceeds available CPUs={len(cpus)}")
    base, remainder = divmod(len(cpus), num_shards)
    sets: list[frozenset[int]] = []
    start = 0
    for i in range(num_shards):
        stop = start + base + (1 if i < remainder else 0)
        sets.append(frozenset(cpus[start:stop]))
        start = stop
    return sets


def prepare_actions(seed: int, num_steps: int, num_envs: int, action_dim: int) -> np.ndarray:
    """Pre-sample legal NumPy actions for ``num_steps`` steps (outside timing)."""
    rng = np.random.default_rng(seed)
    return rng.uniform(-1.0, 1.0, size=(num_steps, num_envs, action_dim)).astype(np.float32)


def run_measured_steps(step_fn: Any, actions: np.ndarray) -> float:
    """Run the timed hot path over pre-prepared actions; returns elapsed seconds."""
    t0 = time.perf_counter()
    for i in range(actions.shape[0]):
        step_fn(actions[i])
    return time.perf_counter() - t0


def total_throughput(total_envs: int, measured_steps: int, wall_time_s: float) -> float:
    """Global env Steps/s: ``global env rows x measured steps / paired wall time``."""
    if wall_time_s <= 0:
        return 0.0
    return float(total_envs) * float(measured_steps) / wall_time_s


def summarize_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-repeat total throughput of one case."""
    values = [float(r["total_env_steps_per_s"]) for r in runs]
    arr = np.asarray(values, dtype=np.float64)
    return {
        "num_runs": len(values),
        "mean_env_steps_per_s": float(arr.mean()),
        "min_env_steps_per_s": float(arr.min()),
        "max_env_steps_per_s": float(arr.max()),
        "scaling_vs_s1": None,
    }


def attach_scaling(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Fill ``summary.scaling_vs_s1`` against the S=1 case of the same task/envs."""
    baselines: dict[tuple[str, int], float] = {}
    for rec in records:
        if rec["status"] == "ok" and rec["num_shards"] == 1:
            key = (rec["task_id"], rec["total_envs"])
            baselines[key] = float(rec["summary"]["mean_env_steps_per_s"])
    for rec in records:
        if rec["status"] != "ok":
            continue
        baseline = baselines.get((rec["task_id"], rec["total_envs"]))
        if baseline:
            rec["summary"]["scaling_vs_s1"] = (
                float(rec["summary"]["mean_env_steps_per_s"]) / baseline
            )
    return records


def git_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(ROOT_DIR), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return out.strip()
    except Exception:
        return "unknown"


# =====================================================================
# Shard worker (one process per shard)
# =====================================================================


def _shard_worker(
    task_key: str,
    shard_index: int,
    shard_rows: int,
    cpu_ids: list[int],
    warmup_steps: int,
    measured_steps: int,
    seed: int,
    msg_queue: Any,
    start_event: Any,
) -> None:
    """Pin CPUs, build one NpEnv, then step pre-prepared actions on signal."""
    try:
        if not hasattr(os, "sched_setaffinity"):
            raise RuntimeError("os.sched_setaffinity is required (Linux only)")
        os.sched_setaffinity(0, set(cpu_ids))
        actual_cpu_ids = sorted(os.sched_getaffinity(0))

        from benchmark.env.benchmark_env_step import TASK_CONFIGS

        task_config = TASK_CONFIGS[task_key]
        cfg = task_config.build_cfg(BACKEND)
        task_config.finalize_cfg(cfg, BACKEND)
        cfg.validate()
        # Skip the adaptive chunk_size sweep: it re-probes on every env
        # materialization and dominates shard setup time. Use the native
        # default chunk_size instead (issue #960).
        cfg.adaptive_chunk_size = False
        env = task_config.env_cls_factory()(cfg, num_envs=shard_rows, backend_type=BACKEND)
        try:
            env.init_state()
            action_dim = env._backend.num_actuators  # type: ignore[reportAttributeAccessIssue]
            # Action preparation happens before the timed window by construction.
            actions = prepare_actions(
                seed + shard_index, warmup_steps + measured_steps, shard_rows, action_dim
            )
            for i in range(warmup_steps):
                env.step(actions[i])
            msg_queue.put(("ready", shard_index, {"cpu_ids": actual_cpu_ids}))
            start_event.wait()
            elapsed = run_measured_steps(env.step, actions[warmup_steps:])
        finally:
            env.close()
        msg_queue.put(
            (
                "done",
                shard_index,
                {
                    "rows": shard_rows,
                    "measured_time_s": elapsed,
                    "env_steps_per_s": total_throughput(shard_rows, measured_steps, elapsed),
                },
            )
        )
    except BaseException as exc:  # noqa: BLE001 - report any shard failure to the parent
        msg_queue.put(
            (
                "error",
                shard_index,
                {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()},
            )
        )


def _collect_shard_messages(
    msg_queue: Any, expected: int, timeout_s: float, phase: str
) -> tuple[dict[int, Any] | None, str | None]:
    """Collect one non-error message per shard; fail on shard error or timeout."""
    messages: dict[int, Any] = {}
    deadline = time.monotonic() + timeout_s
    while len(messages) < expected:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None, (
                f"timeout after {timeout_s:.0f}s waiting for {phase} "
                f"({len(messages)}/{expected} shards reported)"
            )
        try:
            kind, shard, payload = msg_queue.get(timeout=remaining)
        except queue.Empty:
            return None, f"timeout after {timeout_s:.0f}s waiting for {phase}"
        if kind == "error":
            return None, f"shard {shard} failed during {phase}: {payload['error']}"
        messages[shard] = payload
    return messages, None


def _failed_run(repeat: int, error: str) -> dict[str, Any]:
    return {
        "repeat": repeat,
        "status": "failed",
        "error": error,
        "wall_time_s": None,
        "total_env_steps_per_s": None,
        "shards": [],
    }


def _run_repeat(
    task_key: str,
    rows_per_shard: list[int],
    cpu_sets: list[frozenset[int]],
    warmup_steps: int,
    measured_steps: int,
    seed: int,
    repeat: int,
    ready_timeout_s: float,
    done_timeout_s: float,
) -> dict[str, Any]:
    """Run one paired repeat: spawn shards, sync start, measure paired wall time."""
    ctx = mp.get_context("fork")
    msg_queue = ctx.Queue()
    start_event = ctx.Event()
    num_shards = len(rows_per_shard)
    procs = [
        ctx.Process(
            target=_shard_worker,
            args=(
                task_key,
                i,
                rows_per_shard[i],
                sorted(cpu_sets[i]),
                warmup_steps,
                measured_steps,
                seed,
                msg_queue,
                start_event,
            ),
            daemon=True,
        )
        for i in range(num_shards)
    ]
    try:
        for proc in procs:
            proc.start()
        _, error = _collect_shard_messages(msg_queue, num_shards, ready_timeout_s, "readiness")
        if error is not None:
            return _failed_run(repeat, error)
        t0 = time.perf_counter()
        start_event.set()
        done, error = _collect_shard_messages(
            msg_queue, num_shards, done_timeout_s, "measured steps"
        )
        wall_time_s = time.perf_counter() - t0
        if error is not None:
            return _failed_run(repeat, error)
        assert done is not None
        return {
            "repeat": repeat,
            "status": "ok",
            "error": None,
            "wall_time_s": wall_time_s,
            "total_env_steps_per_s": total_throughput(
                sum(rows_per_shard), measured_steps, wall_time_s
            ),
            "shards": [{"shard": i, **done[i]} for i in range(num_shards)],
        }
    finally:
        # Reap every shard process, including on failure paths.
        for proc in procs:
            proc.join(timeout=5.0)
            if proc.is_alive():
                proc.terminate()
        for proc in procs:
            proc.join(timeout=5.0)


def run_case(
    task_key: str,
    total_envs: int,
    num_shards: int,
    *,
    warmup_steps: int,
    measured_steps: int,
    repeats: int,
    seed: int,
    available_cpus: Sequence[int] | None = None,
    ready_timeout_s: float = DEFAULT_READY_TIMEOUT_S,
    done_timeout_s: float = DEFAULT_DONE_TIMEOUT_S,
) -> dict[str, Any]:
    """Run all repeats of one (task, total_envs, num_shards) case."""
    rows_per_shard = shard_env_rows(total_envs, num_shards)
    cpus = list(available_cpus) if available_cpus is not None else sorted(os.sched_getaffinity(0))
    cpu_sets = shard_cpu_sets(cpus, num_shards)
    record: dict[str, Any] = {
        "task_id": TASK_IDS[task_key],
        "sim_backend": BACKEND,
        "total_envs": total_envs,
        "num_shards": num_shards,
        "shard_env_rows": rows_per_shard,
        "cpu_sets": [sorted(s) for s in cpu_sets],
        "warmup_steps": warmup_steps,
        "measured_steps": measured_steps,
        "repeats": repeats,
        "seed": seed,
        "status": "ok",
        "error": None,
        "runs": [],
        "summary": None,
    }
    for repeat in range(repeats):
        run = _run_repeat(
            task_key,
            rows_per_shard,
            cpu_sets,
            warmup_steps,
            measured_steps,
            seed,
            repeat,
            ready_timeout_s,
            done_timeout_s,
        )
        record["runs"].append(run)
        print(
            f"  [{record['task_id']} envs={total_envs} S={num_shards} rep={repeat}] "
            f"{run['status']}"
            + (
                f"  total={run['total_env_steps_per_s']:,.0f} env-steps/s"
                if run["status"] == "ok"
                else f"  error={run['error']}"
            ),
            flush=True,
        )
        if run["status"] != "ok":
            record["status"] = "failed"
            record["error"] = run["error"]
            break
    if record["status"] == "ok":
        record["summary"] = summarize_runs(record["runs"])
    return record


# =====================================================================
# CLI
# =====================================================================


def _parse_int_list(text: str, name: str) -> list[int]:
    values = [int(v) for v in text.split(",") if v.strip()]
    if not values:
        raise ValueError(f"{name} must not be empty")
    return values


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MuJoCo NpEnv multi-shard CPU throughput benchmark (issue #960)."
    )
    parser.add_argument("--tasks", default=DEFAULT_TASKS, help="comma-separated task ids")
    parser.add_argument("--num-envs", default=DEFAULT_NUM_ENVS, help="comma-separated totals")
    parser.add_argument("--shards", default=DEFAULT_SHARDS, help="comma-separated shard counts")
    parser.add_argument("--warmup-steps", type=int, default=DEFAULT_WARMUP_STEPS)
    parser.add_argument("--measured-steps", type=int, default=DEFAULT_MEASURED_STEPS)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--ready-timeout", type=float, default=DEFAULT_READY_TIMEOUT_S)
    parser.add_argument("--done-timeout", type=float, default=DEFAULT_DONE_TIMEOUT_S)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> list[dict[str, Any]]:
    args = parse_args(argv)
    task_keys = []
    for name in (t.strip() for t in args.tasks.split(",") if t.strip()):
        if name not in TASK_KEYS:
            raise ValueError(f"Unknown task {name!r}. Available: {sorted(TASK_KEYS)}")
        key = TASK_KEYS[name]
        if key not in task_keys:
            task_keys.append(key)
    totals = _parse_int_list(args.num_envs, "--num-envs")
    shard_counts = _parse_int_list(args.shards, "--shards")

    available_cpus = sorted(os.sched_getaffinity(0))
    print(f"Device: {get_device_info_line()}")
    print(f"Available CPUs: {len(available_cpus)} | commit: {git_commit()}")

    records: list[dict[str, Any]] = []
    for task_key in task_keys:
        for total_envs in totals:
            for num_shards in shard_counts:
                if num_shards > total_envs:
                    raise ValueError(f"shards={num_shards} exceeds total_envs={total_envs}")
                print(f"case: {TASK_IDS[task_key]} envs={total_envs} S={num_shards}", flush=True)
                records.append(
                    run_case(
                        task_key,
                        total_envs,
                        num_shards,
                        warmup_steps=args.warmup_steps,
                        measured_steps=args.measured_steps,
                        repeats=args.repeats,
                        seed=args.seed,
                        available_cpus=available_cpus,
                        ready_timeout_s=args.ready_timeout,
                        done_timeout_s=args.done_timeout,
                    )
                )
    attach_scaling(records)

    print()
    print_table(
        [
            {
                "task": r["task_id"],
                "envs": r["total_envs"],
                "S": r["num_shards"],
                "mean env-steps/s": (
                    f"{r['summary']['mean_env_steps_per_s']:,.0f}" if r["summary"] else "-"
                ),
                "scaling vs S=1": (
                    f"{r['summary']['scaling_vs_s1']:.2f}x"
                    if r["summary"] and r["summary"]["scaling_vs_s1"]
                    else "-"
                ),
                "status": r["status"],
            }
            for r in records
        ],
        ["task", "envs", "S", "mean env-steps/s", "scaling vs S=1", "status"],
    )

    save_json(
        args.out_json,
        records,
        {
            "benchmark": "np_env_shard_throughput",
            "issue": 960,
            "commit": git_commit(),
            "device": get_device_info_dict(),
            "available_cpus": available_cpus,
            "params": {
                "tasks": [TASK_IDS[k] for k in task_keys],
                "num_envs": totals,
                "shards": shard_counts,
                "warmup_steps": args.warmup_steps,
                "measured_steps": args.measured_steps,
                "repeats": args.repeats,
                "seed": args.seed,
                "backend": BACKEND,
            },
        },
    )
    return records


if __name__ == "__main__":
    main()
