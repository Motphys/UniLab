#!/usr/bin/env python3
"""Run a real FlashSAC training loop and report learner/collector timing.

Unlike the collector-only benchmark, this command launches the production
``train_flashsac.py`` entrypoint, constructs the selected physics backend, and
reads the timing scalars emitted by the real off-policy runner.  The child
process output is forwarded live so the user can see training progress.

Example (MuJoCo):

    uv run scripts/benchmark/rl/benchmark_flashsac_training.py \
        --backend mujoco --iterations 20 --num-envs 256

The same command works with ``--backend motrix`` when the Motrix extra is
installed.  Use ``--uni-rl-src`` to test a local unilab-rl checkout instead of
the installed package, for example the FlashSAC optimization branch.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import fmean
from typing import Iterable, Sequence

ROOT_DIR = Path(__file__).resolve().parents[3]
TRAIN_SCRIPT = ROOT_DIR / "src" / "unilab" / "scripts" / "train_flashsac.py"
DEFAULT_OUTPUT_ROOT = ROOT_DIR / "scripts" / "benchmark" / "outputs" / "flashsac_training"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.benchmark.core.device_info import get_device_info_dict


@dataclass(frozen=True)
class Scalar:
    step: int
    value: float


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise ValueError("cannot calculate a percentile of an empty sequence")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (position - lower) * (ordered[upper] - ordered[lower])


def summarize(values: Iterable[float]) -> dict[str, float | int]:
    samples = [float(value) for value in values]
    if not samples:
        raise ValueError("no timing samples were emitted")
    return {
        "count": len(samples),
        "mean_ms": fmean(samples),
        "median_ms": _percentile(samples, 0.50),
        "p90_ms": _percentile(samples, 0.90),
        "p95_ms": _percentile(samples, 0.95),
        "min_ms": min(samples),
        "max_ms": max(samples),
    }


def _event_file(run_dir: Path) -> Path:
    candidates = sorted(run_dir.rglob("events.out.tfevents.*"))
    if not candidates:
        raise RuntimeError(f"no TensorBoard event file found under {run_dir}")
    return candidates[-1]


def _read_scalars(run_dir: Path, tag: str) -> list[Scalar]:
    from tensorboard.backend.event_processing import event_accumulator

    accumulator = event_accumulator.EventAccumulator(str(_event_file(run_dir)))
    accumulator.Reload()
    if tag not in accumulator.Tags().get("scalars", []):
        return []
    return [Scalar(int(event.step), float(event.value)) for event in accumulator.Scalars(tag)]


def _by_step(samples: Sequence[Scalar]) -> dict[int, float]:
    return {sample.step: sample.value for sample in samples}


def parse_timing(run_dir: Path, *, skip_first: int = 0) -> dict[str, object]:
    """Read per-iteration learner, collector, wall and reward timings."""
    learner = _read_scalars(run_dir, "timing/learner_train_ms")
    collector = _read_scalars(run_dir, "perf/collector_cycle_ms")
    wall = _read_scalars(run_dir, "perf/iter_ms")
    reward = _read_scalars(run_dir, "reward/mean")
    if not learner:
        raise RuntimeError("training log has no timing/learner_train_ms samples")
    if not collector:
        # Older runners do not have the aggregate scalar, but emit the three
        # mutually exclusive collector phases.  Reconstruct the cycle exactly.
        phases = [
            _by_step(_read_scalars(run_dir, f"timing/collector_{name}"))
            for name in ("env_step_ms", "replay_ms", "bookkeeping_ms")
        ]
        steps = sorted(set().union(*(phase.keys() for phase in phases)))
        collector = [Scalar(step, sum(phase.get(step, 0.0) for phase in phases)) for step in steps]
    if not collector:
        raise RuntimeError("training log has no collector timing samples")

    learner_by_step = _by_step(learner)
    collector_by_step = _by_step(collector)
    wall_by_step = _by_step(wall)
    reward_by_step = _by_step(reward)
    steps = sorted(set(learner_by_step) & set(collector_by_step))
    rows_all = [
        {
            "step": step,
            "learner_train_ms": learner_by_step[step],
            "collector_cycle_ms": collector_by_step[step],
            "iter_ms": wall_by_step.get(step),
            "reward": reward_by_step.get(step),
        }
        for step in steps
    ]
    if skip_first < 0:
        raise ValueError("skip_first must be non-negative")
    rows = rows_all[skip_first:]
    if not rows:
        raise RuntimeError(
            f"learner and collector timing series have fewer than {skip_first + 1} common steps"
        )
    return {
        "num_samples": len(rows),
        "num_samples_all": len(rows_all),
        "skip_first": skip_first,
        "rows_all": rows_all,
        "learner_train_ms": summarize(row["learner_train_ms"] for row in rows),
        "collector_cycle_ms": summarize(row["collector_cycle_ms"] for row in rows),
        "iter_ms": summarize(row["iter_ms"] for row in rows if row["iter_ms"] is not None),
        "rows": rows,
    }


def _command(args: argparse.Namespace, run_dir: Path) -> list[str]:
    task = f"{args.task}/{args.backend}"
    command = [
        sys.executable,
        str(TRAIN_SCRIPT),
        f"task={task}",
        "training.no_play=true",
        f"training.sim_backend={args.backend}",
        f"training.log_dir={run_dir}",
        f"algo.max_iterations={args.iterations}",
        f"algo.num_envs={args.num_envs}",
        f"algo.batch_size={args.batch_size}",
        f"algo.replay_buffer_n={args.replay_buffer_n}",
        f"algo.learning_starts={args.learning_starts}",
        f"algo.updates_per_step={args.updates_per_step}",
        "algo.save_interval=1000000",
        f"algo.algo_params.use_compile={str(args.compile).lower()}",
        f"algo.algo_params.compile_full_objectives={str(args.compile).lower()}",
    ]
    command.extend(args.extra_override)
    return command


def _run(command: Sequence[str], *, env: dict[str, str]) -> None:
    print("$ " + " ".join(command), flush=True)
    process = subprocess.Popen(
        list(command),
        cwd=ROOT_DIR,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
    return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, list(command))


def _print_report(report: dict[str, object]) -> None:
    rows = report["rows"]
    assert isinstance(rows, list)
    print("\nPer-iteration timing (ms):")
    print(f"{'step':>8} {'learner':>12} {'collector':>12} {'iter':>12} {'reward':>12}")
    for row in rows:
        assert isinstance(row, dict)
        reward = row.get("reward")
        reward_text = f"{float(reward):12.4f}" if reward is not None else f"{'-':>12}"
        iter_time = row.get("iter_ms")
        iter_text = f"{float(iter_time):12.3f}" if iter_time is not None else f"{'-':>12}"
        print(
            f"{int(row['step']):8d} {float(row['learner_train_ms']):12.3f} "
            f"{float(row['collector_cycle_ms']):12.3f} {iter_text} {reward_text}"
        )
    print("\nSummary (ms):")
    print(f"{'phase':<18} {'mean':>10} {'median':>10} {'p90':>10} {'p95':>10} {'n':>6}")
    for name in ("learner_train_ms", "collector_cycle_ms", "iter_ms"):
        stats = report[name]
        assert isinstance(stats, dict)
        print(
            f"{name:<18} {float(stats['mean_ms']):10.3f} {float(stats['median_ms']):10.3f} "
            f"{float(stats['p90_ms']):10.3f} {float(stats['p95_ms']):10.3f} {int(stats['count']):6d}"
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("mujoco", "motrix"), default="mujoco")
    parser.add_argument("--task", default="g1_walk_flat")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--replay-buffer-n", type=int, default=32)
    parser.add_argument("--learning-starts", type=int, default=8)
    parser.add_argument("--updates-per-step", type=int, default=2)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--keep-run", action="store_true")
    parser.add_argument(
        "--skip-first",
        type=int,
        default=0,
        help="Exclude the first N common timing rows from summary statistics (compile warm-up).",
    )
    parser.add_argument("--uni-rl-src", type=Path, default=None)
    parser.add_argument("--extra-override", action="append", default=[])
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.run_dir is None:
        run_dir = DEFAULT_OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    else:
        run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    source_paths = [str(ROOT_DIR / "src")]
    if args.uni_rl_src is not None:
        source_paths.insert(0, str(args.uni_rl_src.resolve()))
    elif (ROOT_DIR.parent / "unilab_rl-flashsac-optimization" / "src").is_dir():
        source_paths.insert(0, str(ROOT_DIR.parent / "unilab_rl-flashsac-optimization" / "src"))
    env["PYTHONPATH"] = os.pathsep.join(source_paths + [env.get("PYTHONPATH", "")])
    command = _command(args, run_dir)
    _run(command, env=env)
    report = parse_timing(run_dir, skip_first=args.skip_first)
    payload = {
        "command": command,
        "run_dir": str(run_dir),
        "backend": args.backend,
        "task": args.task,
        "device": get_device_info_dict(),
        **report,
    }
    _print_report(report)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"\nSaved JSON: {args.output}")
    if not args.keep_run and args.output is None:
        # Keep the run by default when --output is requested; otherwise leave
        # it available for TensorBoard inspection because it contains the
        # actual physics-backed training artifacts.
        print(f"Training logs: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
