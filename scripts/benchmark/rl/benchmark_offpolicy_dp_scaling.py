"""Off-policy multi-GPU data-parallel scaling benchmark (issue #968).

Runs real ``src/unilab/scripts/train_sac.py`` training via subprocess (same Hydra
overrides as the production CLI entry, never importing training internals) and
compares single-device N=1 (no ``training.devices``) against N-way data
parallel (``training.devices=[d0..dN-1]``, default ``[0,1]``). Every config
keeps the owner YAML production defaults (sac / g1_walk_flat / mujoco) except
``algo.max_iterations``, ``training.no_play=true`` and ``training.log_dir``
pointing into this benchmark's own work directory. Runs execute sequentially
to avoid resource contention.

Measurement conventions:

- Collector throughput (Steps/s): mean of the last 50% of canonical rank-0
  ``perf/steps_per_sec`` samples. In DP runs the runner already sums the
  per-rank collector rates before logging.
- Learner throughput (Samples/s): the same tail mean over
  ``perf/effective_samples_per_sec``. This counts effective learner samples
  (including configured sample multipliers), summed across ranks.
- Both fields report their own N-way / N=1 scaling ratio. The roadmap verdict
  remains attached to collector Steps/s: ``pass`` at >= 1.7
  (``SCALING_PASS_THRESHOLD``), otherwise ``below threshold``. The verdict is
  data only and does not affect the exit code.
- The tail uses ceil(n/2) points (with n samples, ``n - n//2``), skipping
  collector warm-up and replay prefill. Missing either throughput tag is a
  hard error, never silently skipped.
- Exit code is non-zero only when a run itself failed (subprocess error,
  non-completed summary, or missing artifacts).

Run:
    uv run scripts/benchmark/rl/benchmark_offpolicy_dp_scaling.py

    # tuning / passthrough overrides:
    uv run scripts/benchmark/rl/benchmark_offpolicy_dp_scaling.py \
        --iterations 300 --devices 0,1 \
        --extra-overrides algo.num_envs=2048 \
        --out-json scripts/benchmark/outputs/offpolicy_dp_scaling/results.json
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from scripts.benchmark.core.device_info import get_device_info_dict, get_device_info_line
from scripts.benchmark.core.output import print_table, save_json

DEFAULT_OUTPUT_JSON = (
    ROOT_DIR / "scripts" / "benchmark" / "outputs" / "offpolicy_dp_scaling" / "results.json"
)
DEFAULT_RUNS_ROOT = ROOT_DIR / "scripts" / "benchmark" / "outputs" / "offpolicy_dp_scaling" / "runs"

TRAIN_SCRIPT = ROOT_DIR / "src" / "unilab" / "scripts" / "train_sac.py"

# Route overrides equivalent to `uv run train --algo sac --task g1_walk_flat
# --sim mujoco` (see src/unilab/cli.py build_route for off-policy algos).
ROUTE_OVERRIDES = ("task=g1_walk_flat/mujoco",)

STEPS_PER_SEC_TAG = "perf/steps_per_sec"
SAMPLES_PER_SEC_TAG = "perf/effective_samples_per_sec"
REWARD_TAG = "reward/mean"
DP_SYNC_TIME_TAG = "train/dp_sync_time"

DEFAULT_ITERATIONS = 300
DEFAULT_DEVICES = "0,1"

# Roadmap #964 acceptance target for 2-way data-parallel aggregate throughput.
SCALING_PASS_THRESHOLD = 1.7

STEADY_STATE_TAIL_FRACTION = 0.5


class RunParseError(RuntimeError):
    """A finished run directory is missing required artifacts or is invalid."""


# =====================================================================
# Pure helpers (unit-tested in tests/benchmark/)
# =====================================================================


def steady_state_mean(
    values: Sequence[float], tail_fraction: float = STEADY_STATE_TAIL_FRACTION
) -> float:
    """Mean of the last ``tail_fraction`` of samples (ceil tail size, min 1)."""
    if not values:
        raise ValueError("steady_state_mean requires at least one sample")
    if not 0 < tail_fraction <= 1:
        raise ValueError(f"tail_fraction must be in (0, 1], got {tail_fraction}")
    n = len(values)
    tail = max(1, n - int(n * (1 - tail_fraction)))
    window = [float(v) for v in values[n - tail :]]
    return sum(window) / len(window)


def verdict_for_ratio(ratio: float | None, threshold: float = SCALING_PASS_THRESHOLD) -> str | None:
    """Map a scaling ratio to its verdict string; None stays None (baseline)."""
    if ratio is None:
        return None
    return "pass" if ratio >= threshold else "below threshold"


def find_event_files(rank_dir: Path) -> list[Path]:
    """All tfevents files directly under one rank's log directory."""
    rank_dir = Path(rank_dir)
    if not rank_dir.is_dir():
        raise RunParseError(f"rank log directory does not exist: {rank_dir}")
    return sorted(rank_dir.glob("events.out.tfevents.*"))


def read_scalar_series(rank_dir: Path, tag: str) -> list[float]:
    """Scalar values of ``tag`` from a rank's tfevents (in event order).

    An absent tag yields ``[]``; a rank directory without any tfevents file
    raises ``RunParseError`` so missing rank data is never silent.
    """
    event_files = find_event_files(rank_dir)
    if not event_files:
        raise RunParseError(f"no tfevents file found under {rank_dir}")
    if len(event_files) > 1:
        raise RunParseError(
            f"expected exactly one tfevents file under {rank_dir}, got {len(event_files)}"
        )
    from tensorboard.backend.event_processing import event_accumulator

    accumulator = event_accumulator.EventAccumulator(str(event_files[0]))
    accumulator.Reload()
    if tag not in accumulator.Tags()["scalars"]:
        return []
    return [float(event.value) for event in accumulator.Scalars(tag)]


def parse_run(run_dir: Path, world_size: int) -> dict[str, Any]:
    """Parse one finished run directory into a metrics record.

    Rank 0 must carry a ``run_summary.json`` with ``status == "completed"``
    and the canonical aggregate tfevents file. Non-owner ranks intentionally
    create no TensorBoard writer.
    """
    run_dir = Path(run_dir)
    if world_size < 1:
        raise ValueError(f"world_size must be >= 1, got {world_size}")

    summary_path = run_dir / "run_summary.json"
    if not summary_path.is_file():
        raise RunParseError(f"missing run summary: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    status = summary.get("status")
    if status != "completed":
        raise RunParseError(
            f"run {run_dir} did not complete: status={status!r} error={summary.get('error')!r}"
        )

    collector_series = read_scalar_series(run_dir, STEPS_PER_SEC_TAG)
    if not collector_series:
        raise RunParseError(f"run has no {STEPS_PER_SEC_TAG!r} samples under {run_dir}")
    learner_series = read_scalar_series(run_dir, SAMPLES_PER_SEC_TAG)
    if not learner_series:
        raise RunParseError(f"run has no {SAMPLES_PER_SEC_TAG!r} samples under {run_dir}")
    dp_sync_samples = read_scalar_series(run_dir, DP_SYNC_TIME_TAG)
    reward_series = read_scalar_series(run_dir, REWARD_TAG)
    return {
        "run_dir": str(run_dir),
        "world_size": world_size,
        "completed_iterations": summary.get("completed_iterations"),
        "total_env_steps": summary.get("total_env_steps"),
        "training_wall_time_sec": summary.get("training_wall_time_sec"),
        "num_collector_throughput_samples": len(collector_series),
        "num_learner_throughput_samples": len(learner_series),
        "steady_state_collector_steps_per_s": steady_state_mean(collector_series),
        "steady_state_learner_samples_per_s": steady_state_mean(learner_series),
        "final_mean_reward": reward_series[-1] if reward_series else None,
        "mean_dp_sync_time_sec": (
            sum(dp_sync_samples) / len(dp_sync_samples) if dp_sync_samples else None
        ),
    }


def build_train_command(
    run_dir: Path,
    *,
    iterations: int,
    devices: Sequence[int] | None = None,
    extra_overrides: Sequence[str] = (),
) -> list[str]:
    """Subprocess argv matching the production off-policy CLI overrides.

    ``devices=None`` is the N=1 baseline and intentionally carries no
    ``training.devices`` override at all.
    """
    command = [
        sys.executable,
        str(TRAIN_SCRIPT),
        *ROUTE_OVERRIDES,
        "training.no_play=true",
        f"algo.max_iterations={iterations}",
        f"training.log_dir={Path(run_dir)}",
    ]
    if devices is not None:
        command.append(f"training.devices=[{','.join(str(d) for d in devices)}]")
    command.extend(extra_overrides)
    return command


def attach_scaling(
    records: list[dict[str, Any]], threshold: float = SCALING_PASS_THRESHOLD
) -> list[dict[str, Any]]:
    """Attach separate collector and learner scaling ratios against N=1."""
    baseline = next(
        (
            r
            for r in records
            if r["config"] == "n1" and r["status"] == "ok" and r["metrics"] is not None
        ),
        None,
    )
    baseline_collector = (
        float(baseline["metrics"]["steady_state_collector_steps_per_s"]) if baseline else None
    )
    baseline_learner = (
        float(baseline["metrics"]["steady_state_learner_samples_per_s"]) if baseline else None
    )
    for record in records:
        record["collector_scaling_vs_n1"] = None
        record["learner_scaling_vs_n1"] = None
        record["verdict"] = None
        if record["config"] == "n1" or record["status"] != "ok" or record["metrics"] is None:
            continue
        if baseline_collector:
            ratio = (
                float(record["metrics"]["steady_state_collector_steps_per_s"]) / baseline_collector
            )
            record["collector_scaling_vs_n1"] = ratio
            record["verdict"] = verdict_for_ratio(ratio, threshold)
        if baseline_learner:
            record["learner_scaling_vs_n1"] = (
                float(record["metrics"]["steady_state_learner_samples_per_s"]) / baseline_learner
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


def _parse_int_list(text: str, name: str) -> list[int]:
    values = [int(v) for v in text.split(",") if v.strip()]
    if not values:
        raise ValueError(f"{name} must not be empty")
    return values


# =====================================================================
# Subprocess execution
# =====================================================================


def execute_run(command: list[str], run_dir: Path) -> None:
    """Run one training config to completion; raise on subprocess failure."""
    run_dir = Path(run_dir)
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    completed = subprocess.run(command, cwd=ROOT_DIR)  # noqa: S603 - fixed argv, no shell
    if completed.returncode != 0:
        raise RunParseError(
            f"training subprocess exited with code {completed.returncode}: {' '.join(command)}"
        )


def run_config(
    name: str,
    devices: Sequence[int] | None,
    *,
    iterations: int,
    extra_overrides: Sequence[str],
    runs_root: Path,
) -> dict[str, Any]:
    """Execute and parse one benchmark config; failures are recorded, not raised."""
    world_size = len(devices) if devices is not None else 1
    run_dir = runs_root / name
    record: dict[str, Any] = {
        "config": name,
        "world_size": world_size,
        "devices": list(devices) if devices is not None else None,
        "status": "ok",
        "error": None,
        "metrics": None,
        "collector_scaling_vs_n1": None,
        "learner_scaling_vs_n1": None,
        "verdict": None,
    }
    try:
        command = build_train_command(
            run_dir,
            iterations=iterations,
            devices=devices,
            extra_overrides=extra_overrides,
        )
        print(f"[{name}] launching: {' '.join(command)}", flush=True)
        execute_run(command, run_dir)
        record["metrics"] = parse_run(run_dir, world_size)
    except (RunParseError, ValueError) as exc:
        record["status"] = "failed"
        record["error"] = str(exc)
    print(
        f"[{name}] {record['status']}"
        + (
            f"  collector={record['metrics']['steady_state_collector_steps_per_s']:,.0f} Steps/s"
            f"  learner={record['metrics']['steady_state_learner_samples_per_s']:,.0f} Samples/s"
            if record["metrics"]
            else f"  error={record['error']}"
        ),
        flush=True,
    )
    return record


# =====================================================================
# CLI
# =====================================================================


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Off-policy multi-GPU data-parallel scaling benchmark (issue #968)."
    )
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument(
        "--devices",
        default=DEFAULT_DEVICES,
        help="comma-separated CUDA indices for the data-parallel config "
        "(length 1 skips the DP config)",
    )
    parser.add_argument(
        "--extra-overrides",
        nargs="*",
        default=(),
        help="extra Hydra overrides appended verbatim to every config "
        "(e.g. algo.num_envs=2048); owner YAML defaults are used otherwise",
    )
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument(
        "--keep-runs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="keep per-config training run directories under the output root",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    devices = _parse_int_list(args.devices, "--devices")

    print(f"Device: {get_device_info_line()}")
    print(f"commit: {git_commit()}")

    configs: list[tuple[str, list[int] | None]] = [("n1", None)]
    if len(devices) > 1:
        configs.append((f"n{len(devices)}", devices))
    else:
        print(
            f"--devices={args.devices!r} has length {len(devices)}; "
            "skipping the data-parallel config (need >= 2 devices)."
        )

    records = [
        run_config(
            name,
            config_devices,
            iterations=args.iterations,
            extra_overrides=tuple(args.extra_overrides),
            runs_root=DEFAULT_RUNS_ROOT,
        )
        for name, config_devices in configs
    ]
    attach_scaling(records)

    if not args.keep_runs:
        shutil.rmtree(DEFAULT_RUNS_ROOT, ignore_errors=True)

    print()
    print_table(
        [
            {
                "config": r["config"],
                "devices": ",".join(str(d) for d in r["devices"]) if r["devices"] else "-",
                "collector Steps/s": (
                    f"{r['metrics']['steady_state_collector_steps_per_s']:,.0f}"
                    if r["metrics"]
                    else "-"
                ),
                "learner Samples/s": (
                    f"{r['metrics']['steady_state_learner_samples_per_s']:,.0f}"
                    if r["metrics"]
                    else "-"
                ),
                "final reward": (
                    f"{r['metrics']['final_mean_reward']:.2f}"
                    if r["metrics"] and r["metrics"]["final_mean_reward"] is not None
                    else "-"
                ),
                "dp_sync mean (s)": (
                    f"{r['metrics']['mean_dp_sync_time_sec']:.4f}"
                    if r["metrics"] and r["metrics"]["mean_dp_sync_time_sec"] is not None
                    else "-"
                ),
                "collector scaling": (
                    f"{r['collector_scaling_vs_n1']:.2f}x"
                    if r["collector_scaling_vs_n1"] is not None
                    else "-"
                ),
                "learner scaling": (
                    f"{r['learner_scaling_vs_n1']:.2f}x"
                    if r["learner_scaling_vs_n1"] is not None
                    else "-"
                ),
                "verdict": r["verdict"] or "-",
                "status": r["status"],
            }
            for r in records
        ],
        [
            "config",
            "devices",
            "collector Steps/s",
            "learner Samples/s",
            "final reward",
            "dp_sync mean (s)",
            "collector scaling",
            "learner scaling",
            "verdict",
            "status",
        ],
    )

    save_json(
        args.out_json,
        records,
        {
            "benchmark": "offpolicy_dp_scaling",
            "issue": 968,
            "commit": git_commit(),
            "device": get_device_info_dict(),
            "params": {
                "route_overrides": list(ROUTE_OVERRIDES),
                "iterations": args.iterations,
                "devices": devices,
                "extra_overrides": list(args.extra_overrides),
                "scaling_pass_threshold": SCALING_PASS_THRESHOLD,
                "steady_state_tail_fraction": STEADY_STATE_TAIL_FRACTION,
                "throughput_tags": {
                    "collector_steps_per_sec": STEPS_PER_SEC_TAG,
                    "learner_samples_per_sec": SAMPLES_PER_SEC_TAG,
                },
            },
        },
    )
    failures = [r["config"] for r in records if r["status"] != "ok"]
    if failures:
        print(f"failed configs: {', '.join(failures)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
