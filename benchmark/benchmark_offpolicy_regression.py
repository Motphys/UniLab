"""Micro regression benchmark for off-policy training loop latency.

This script runs short training jobs and parses collector/train times from stdout.
It is intended for before/after infra comparisons under the same machine setup.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
from pathlib import Path

TIMING_RE = re.compile(r"Collect\s+([0-9]+(?:\.[0-9]+)?)ms\s+Train\s+([0-9]+(?:\.[0-9]+)?)ms")


def _run_once(cmd: list[str], cwd: Path) -> tuple[float, float, str]:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        check=False,
        capture_output=True,
        text=True,
    )
    output = (proc.stdout or "") + "\n" + (proc.stderr or "")
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with code {proc.returncode}\n{output}")

    matches = TIMING_RE.findall(output)
    if not matches:
        raise RuntimeError(f"Could not parse timing from output\n{output}")

    collect_ms = [float(m[0]) for m in matches]
    train_ms = [float(m[1]) for m in matches]
    return statistics.mean(collect_ms), statistics.mean(train_ms), output


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark off-policy infra latency")
    parser.add_argument("--algo", type=str, default="sac", choices=["sac", "td3"])
    parser.add_argument("--task", type=str, default="Go1JoystickFlatTerrain")
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--script", type=str, default="scripts/train_offpolicy.py")
    parser.add_argument("--out", type=str, default="outputs/backends/offpolicy_regression.json")
    args = parser.parse_args()

    root_dir = Path(__file__).resolve().parents[1]
    cmd = [
        args.python,
        args.script,
        "--algo",
        args.algo,
        "--task",
        args.task,
        "--max_iterations",
        str(args.iterations),
        "--num_envs",
        str(args.num_envs),
        "--no_play",
        "--logger",
        "tensorboard",
    ]

    collect_samples: list[float] = []
    train_samples: list[float] = []

    for idx in range(args.repeats):
        collect_ms, train_ms, _ = _run_once(cmd, root_dir)
        collect_samples.append(collect_ms)
        train_samples.append(train_ms)
        print(f"run={idx+1}/{args.repeats} collect_ms={collect_ms:.2f} train_ms={train_ms:.2f}")

    result = {
        "algo": args.algo,
        "task": args.task,
        "iterations": args.iterations,
        "num_envs": args.num_envs,
        "repeats": args.repeats,
        "collect_ms_mean": statistics.mean(collect_samples),
        "collect_ms_stdev": statistics.pstdev(collect_samples),
        "train_ms_mean": statistics.mean(train_samples),
        "train_ms_stdev": statistics.pstdev(train_samples),
        "collect_samples": collect_samples,
        "train_samples": train_samples,
    }

    out_path = root_dir / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved benchmark result to {out_path}")


if __name__ == "__main__":
    main()
