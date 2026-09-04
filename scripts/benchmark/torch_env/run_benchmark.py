#!/usr/bin/env python3
"""NumPy vs Torch comparison for collector-timed numeric manager kernels.

Reproduces the numeric work represented by the `env_step_update_state_ms` and
`env_step_reset_done_ms` collector metrics for two SAC/MuJoCo tasks, at
identical scale and computation items:

- g1_walk_flat       (num_envs=2048, 29-dof, obs 98 / critic 101)
- g1_motion_tracking (num_envs=2048, 29-dof, 14 bodies, obs 160 / critic 289)

Variants:
    numpy               single-process NumPy (the production code path)
    torch-cpu           same kernels on torch CPU tensors
    torch-cuda          same kernels on torch CUDA tensors (state stays on GPU)
    torch-cuda-compile  torch.compile'd CUDA kernels (dynamic=False)
    torch-mps(-compile) only when MPS is available (not on Linux)

Usage:
    uv run scripts/benchmark/torch_env/run_benchmark.py
    uv run scripts/benchmark/torch_env/run_benchmark.py --iters 300 --warmup 50
    uv run scripts/benchmark/torch_env/run_benchmark.py --variants numpy,torch-cpu
    uv run scripts/benchmark/torch_env/run_benchmark.py --tasks walk
    uv run scripts/benchmark/torch_env/run_benchmark.py --num-envs 2048,8192,32768
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from statistics import mean, median, pstdev

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from scripts.benchmark.core.device_info import get_device_info_dict
from scripts.benchmark.torch_env import motion_tracking, walk_flat
from scripts.benchmark.torch_env.xp import (
    NumpyBackend,
    NumpyRng,
    RecordingRng,
    ReplayRng,
    TorchBackend,
    TorchRng,
)

DEFAULT_OUTPUT_JSON = ROOT_DIR / "scripts" / "benchmark" / "outputs" / "torch_env" / "results.json"
RESET_SIZES = (16, 256, 2048)
TASKS = ("walk", "tracking")


def build_workload(
    task: str,
    variant: str,
    seed: int = 0,
    vectorized_reset_rng: bool = False,
    num_envs: int = walk_flat.N_ENVS,
):
    """Build (workload, backend) for a task/variant pair."""
    compile_requested = variant.endswith("-compile")
    device = variant[: -len("-compile")] if compile_requested else variant
    if device == "numpy":
        backend = NumpyBackend()
        rng = NumpyRng()
    else:
        torch_device = device[len("torch-") :]
        backend = TorchBackend(torch_device)
        rng = TorchRng(torch_device)
    cls = {
        "walk": walk_flat.WalkFlatWorkload,
        "tracking": motion_tracking.MotionTrackingWorkload,
    }[task]
    if task == "tracking":
        workload = cls(
            backend, rng, seed=seed, vectorized_reset_rng=vectorized_reset_rng, num_envs=num_envs
        )
    else:
        workload = cls(backend, rng, seed=seed, num_envs=num_envs)
    if compile_requested:
        import torch

        # The xp shim helpers are shared across many call sites with distinct
        # static shapes; raise the specialization ceiling so later sites do not
        # silently fall back to eager.
        torch._dynamo.config.recompile_limit = 64
        torch._dynamo.config.accumulated_recompile_limit = 512
        workload.update_state = torch.compile(workload.update_state, dynamic=False)
        workload.reset_done = torch.compile(workload.reset_done, dynamic=False)
    return workload, backend


def available_variants() -> list[str]:
    variants = ["numpy", "torch-cpu"]
    try:
        import torch

        if torch.cuda.is_available():
            variants += ["torch-cuda", "torch-cuda-compile"]
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            variants += ["torch-mps", "torch-mps-compile"]
    except ImportError:
        pass
    return variants


def time_call(fn, sync, warmup: int, iters: int) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    sync()
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        sync()
        samples.append((time.perf_counter() - t0) * 1000.0)
    return {
        "mean_ms": mean(samples),
        "std_ms": pstdev(samples),
        "p50_ms": median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "iters": iters,
    }


def _to_numpy_tree(b, value):
    if isinstance(value, dict):
        return {k: _to_numpy_tree(b, v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_to_numpy_tree(b, v) for v in value]
    return b.to_numpy(value)


def _assert_close(name, ref, got, rtol=2e-4, atol=1e-5):
    if ref.dtype == bool:
        assert (ref == got).all(), f"{name}: bool mismatch"
        return
    if not np.allclose(ref, got, rtol=rtol, atol=atol):
        diff = np.abs(ref.astype(np.float64) - got.astype(np.float64))
        raise AssertionError(f"{name}: max abs diff {diff.max():.3e}")


def validate_task(task: str, variants: list[str], num_envs: int) -> dict[str, str]:
    """Replay identical RNG draws on every variant and compare outputs."""
    # Reference: numpy with vectorized reset RNG so the draw sequence matches
    # the (vectorized) torch implementations.
    workload, b = build_workload(task, "numpy", vectorized_reset_rng=True, num_envs=num_envs)
    recording = RecordingRng()
    workload.rng = recording
    rs = np.random.RandomState(1)
    env_ids = np.sort(rs.choice(num_envs, 64, replace=False)).astype(np.int64)

    ref_obs, ref_reward, ref_terminated = workload.update_state(should_log=True)
    ref_reset = workload.reset_done(b.index(env_ids))
    ref = {
        "obs": _to_numpy_tree(b, ref_obs),
        "reward": b.to_numpy(ref_reward),
        "terminated": b.to_numpy(ref_terminated),
        "reset": _to_numpy_tree(b, ref_reset),
    }

    results = {}
    for variant in variants:
        if variant == "numpy" or variant.endswith("-compile"):
            continue
        workload, b = build_workload(task, variant, vectorized_reset_rng=True, num_envs=num_envs)
        workload.rng = ReplayRng(b, recording.calls)
        obs, reward, terminated = workload.update_state(should_log=True)
        reset = workload.reset_done(b.index(env_ids))
        try:
            for key in ("obs", "critic"):
                _assert_close(f"obs[{key}]", ref["obs"][key], b.to_numpy(obs[key]))
            _assert_close("reward", ref["reward"], b.to_numpy(reward))
            _assert_close("terminated", ref["terminated"], b.to_numpy(terminated))
            _assert_close("reset.qpos", ref["reset"][0], b.to_numpy(reset[0]))
            _assert_close("reset.qvel", ref["reset"][1], b.to_numpy(reset[1]))
            for key in ("obs", "critic"):
                _assert_close(f"reset.obs[{key}]", ref["reset"][2][key], b.to_numpy(reset[2][key]))
            results[variant] = "ok"
        except AssertionError as exc:
            results[variant] = f"FAILED: {exc}"
    return results


def benchmark_task(
    task: str,
    variants: list[str],
    warmup: int,
    iters: int,
    reset_iters: int,
    num_envs_list: list[int],
):
    task_result = {"update_state": {}, "reset_done": {}, "validation": {}}
    for variant in variants:
        for num_envs in num_envs_list:
            workload, b = build_workload(task, variant, num_envs=num_envs)
            step = {"count": 0}

            def run_update_state():
                step["count"] += 1
                workload.update_state(step["count"] % 4 == 0)

            task_result["update_state"].setdefault(variant, {})[f"n={num_envs}"] = time_call(
                run_update_state, b.sync, warmup, iters
            )

        # reset_done is measured at the base (training) scale only.
        base_envs = num_envs_list[0]
        workload, b = build_workload(task, variant, num_envs=base_envs)
        rs = np.random.RandomState(2)
        for n_reset in RESET_SIZES:
            env_ids = b.index(np.sort(rs.choice(base_envs, n_reset, replace=False)))

            def run_reset():
                workload.reset_done(env_ids)

            task_result["reset_done"].setdefault(variant, {})[f"n={n_reset}"] = time_call(
                run_reset, b.sync, max(5, warmup // 2), reset_iters
            )
    return task_result


def fmt(stats):
    return f"{stats['mean_ms']:8.3f} ± {stats['std_ms']:6.3f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=str, default=",".join(TASKS))
    parser.add_argument("--variants", type=str, default=None)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--reset-iters", type=int, default=100)
    parser.add_argument(
        "--num-envs",
        type=str,
        default="2048",
        help="Comma-separated num_envs scales for update_state (reset_done uses the first).",
    )
    parser.add_argument("--out-json", type=str, default=str(DEFAULT_OUTPUT_JSON))
    parser.add_argument("--skip-validation", action="store_true")
    args = parser.parse_args()

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    num_envs_list = [int(v) for v in args.num_envs.split(",") if v.strip()]
    all_variants = available_variants()
    if args.variants:
        requested = [v.strip() for v in args.variants.split(",") if v.strip()]
        variants = [v for v in requested if v in all_variants]
        skipped = sorted(set(requested) - set(variants))
    else:
        variants = all_variants
        skipped = []
    for wanted in ("torch-mps", "torch-mps-compile"):
        if wanted not in all_variants and (not args.variants or wanted in args.variants):
            skipped.append(wanted)

    import torch

    print(f"tasks: {tasks}")
    print(f"variants: {variants} (skipped/unavailable: {skipped or 'none'})")
    print(f"torch {torch.__version__}, cpu threads: {torch.get_num_threads()}")

    report = {
        "device_info": get_device_info_dict(),
        "torch_version": torch.__version__,
        "torch_num_threads": torch.get_num_threads(),
        "variants": variants,
        "skipped_variants": skipped,
        "params": {
            "warmup": args.warmup,
            "iters": args.iters,
            "reset_iters": args.reset_iters,
            "reset_sizes": list(RESET_SIZES),
            "num_envs_list": num_envs_list,
        },
        "tasks": {},
    }

    for task in tasks:
        print(f"\n=== task: {task} ===")
        if not args.skip_validation:
            validation = validate_task(task, variants, num_envs_list[0])
        else:
            validation = {}
        result = benchmark_task(
            task, variants, args.warmup, args.iters, args.reset_iters, num_envs_list
        )
        result["validation"] = validation
        report["tasks"][task] = result
        for variant, status in validation.items():
            print(f"  validation[{variant}]: {status}")

        base = result["update_state"].get("numpy", {})
        print("\n  update_state (ms per call):")
        for variant, by_size in result["update_state"].items():
            for size_key, stats in by_size.items():
                base_stats = base.get(size_key)
                speedup = base_stats["mean_ms"] / stats["mean_ms"] if base_stats else float("nan")
                print(
                    f"    {variant:22s} {size_key:9s} {fmt(stats)}"
                    f"   speedup vs numpy: {speedup:6.2f}x"
                )
        print("  reset_done (ms per call):")
        for n_reset in RESET_SIZES:
            print(f"    n_reset={n_reset}:")
            for variant, by_size in result["reset_done"].items():
                stats = by_size[f"n={n_reset}"]
                base_stats = result["reset_done"].get("numpy", {}).get(f"n={n_reset}")
                speedup = base_stats["mean_ms"] / stats["mean_ms"] if base_stats else float("nan")
                print(f"      {variant:22s} {fmt(stats)}   speedup vs numpy: {speedup:6.2f}x")

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nresults written to {out_path}")

    failures = {
        task: {v: s for v, s in r["validation"].items() if s != "ok"}
        for task, r in report["tasks"].items()
    }
    failures = {t: f for t, f in failures.items() if f}
    if failures:
        print(f"VALIDATION FAILURES: {failures}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
