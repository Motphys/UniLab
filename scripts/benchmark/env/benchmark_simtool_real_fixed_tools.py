"""Benchmark representative SimToolReal fixed-tool construction and rollouts."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from scripts.benchmark.core.mem_profile import current_memory_bytes, peak_rss_bytes
from unilab.envs import make_manager_based_rl_env
from unilab.tasks.manipulation.simtool_real import (
    build_representative_simtool_real_env_cfg,
    write_representative_simtool_real_sources,
)


def _rss_bytes() -> int:
    value = current_memory_bytes().get("rss_bytes")
    if not isinstance(value, int):
        raise RuntimeError("current RSS is unavailable")
    return value


def run_benchmark(
    *,
    backend: str,
    num_envs: int,
    num_variants: int,
    steps: int,
    warmup_steps: int,
    variant_root: Path,
) -> dict[str, Any]:
    rss_before_source = _rss_bytes()
    source_started = time.perf_counter()
    sources = write_representative_simtool_real_sources(
        variant_root,
        variant_count=num_variants,
    )
    source_materialization_seconds = time.perf_counter() - source_started
    cfg = build_representative_simtool_real_env_cfg(sources)

    rss_before_env = _rss_bytes()
    construction_started = time.perf_counter()
    env = make_manager_based_rl_env(cfg, num_envs=num_envs, backend_type=backend)
    env_construction_seconds = time.perf_counter() - construction_started

    reset_started = time.perf_counter()
    env.init_state()
    first_reset_seconds = time.perf_counter() - reset_started
    rss_after_construction = _rss_bytes()

    actions = np.zeros((num_envs, 1), dtype=np.float32)
    for _ in range(warmup_steps):
        env.step(actions)
    step_started = time.perf_counter()
    for _ in range(steps):
        state = env.step(actions)
    measured_steps_seconds = time.perf_counter() - step_started

    peak_rss = peak_rss_bytes()
    plan = cfg.scene.fixed_variant_plan
    assert plan is not None
    result: dict[str, Any] = {
        "backend": backend,
        "num_envs": num_envs,
        "num_variants": num_variants,
        "steps": steps,
        "warmup_steps": warmup_steps,
        "unique_assigned_variants": len(set(int(value) for value in plan.assignment)),
        "finite": bool(np.isfinite(state.obs["obs"]).all() and np.isfinite(state.reward).all()),
        "timings": {
            "source_materialization_seconds": source_materialization_seconds,
            "env_construction_seconds": env_construction_seconds,
            "first_reset_seconds": first_reset_seconds,
            "measured_steps_seconds": measured_steps_seconds,
            "throughput_env_steps_per_s": float(steps * num_envs / measured_steps_seconds),
        },
        "memory": {
            "rss_before_source_bytes": rss_before_source,
            "rss_before_env_bytes": rss_before_env,
            "rss_after_construction_bytes": rss_after_construction,
            "peak_rss_bytes": peak_rss,
            "construction_delta_bytes": rss_after_construction - rss_before_env,
        },
    }
    env.close()
    result["memory"]["rss_after_close_bytes"] = _rss_bytes()
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("mujoco", "mjwarp"), default="mujoco")
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--num-variants", type=int, default=64)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("scripts/benchmark/outputs/simtool_real_fixed_tools/result.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.num_envs <= 0:
        raise SystemExit("num-envs must be positive")
    if min(args.num_variants, args.steps, args.warmup_steps) < 0:
        raise SystemExit("num-variants, steps, and warmup-steps must be non-negative")
    if args.num_variants == 0 or args.steps == 0:
        raise SystemExit("num-variants and steps must be positive")
    with tempfile.TemporaryDirectory(prefix="simtool-real-fixed-tools-") as temporary:
        result = run_benchmark(
            backend=args.backend,
            num_envs=args.num_envs,
            num_variants=args.num_variants,
            steps=args.steps,
            warmup_steps=args.warmup_steps,
            variant_root=Path(temporary),
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"Saved: {args.output.resolve()}")


if __name__ == "__main__":
    main()
