#!/usr/bin/env python3
"""Benchmark Numba RNG against NumPy RNG for UniLab noise-buffer workloads.

Issue #680 asks for an isolated measurement before changing env hot paths. This
script uses owner-config shapes from G1 motion tracking and G1 walk/joystick
training profiles, then compares:

* ``numpy_random_uniform_alloc``: current-style ``np.random.uniform(...).astype``;
* ``numpy_generator_random_out``: NumPy with preallocated output buffers;
* ``numba_random_prange``: ``np.random.random`` inside an ``njit(parallel=True)``
  kernel, also writing into preallocated buffers.

Run:
    uv run python -m benchmark.benchmark_numba_random_noise
    uv run python benchmark/benchmark_numba_random_noise.py --quick
    uv run python -m benchmark.benchmark_numba_random_noise --profiles sac_g1_motion_tracking_mujoco --threads 1 4 8
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

try:  # pragma: no cover - exercised when numba is installed
    from numba import get_num_threads, njit, prange, set_num_threads

    NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover - benchmark still reports NumPy baselines
    get_num_threads = njit = prange = set_num_threads = None  # type: ignore[assignment]
    NUMBA_AVAILABLE = False

from benchmark.core.device_info import get_device_info_dict, get_device_info_line

DEFAULT_THREADS = [1, 2, 4, 8, 16, 32, 64]
QUICK_THREADS = [1, 2]
DEFAULT_NUM_ENVS = [1024, 2048, 4096, 8192, 16384, 32768]
QUICK_NUM_ENVS = [1024, 2048]
NUM_ACTION = 29


@dataclass(frozen=True)
class NoiseField:
    name: str
    width: int
    scale: float


@dataclass(frozen=True)
class NoiseProfile:
    name: str
    owner_config: str
    default_num_envs: int
    fields: tuple[NoiseField, ...]
    note: str

    @property
    def total_width(self) -> int:
        return sum(field.width for field in self.fields)

    @property
    def values_per_step(self) -> int:
        return self.default_num_envs * self.total_width


@dataclass
class BenchCase:
    profile: str
    owner_config: str
    num_envs: int
    dtype: str
    path: str
    threads: int | None
    values: int
    mean_ms: float
    min_ms: float
    std_ms: float
    values_per_s: float
    speedup_vs_numpy_alloc: float
    compile_ms: float | None = None
    deterministic_same_seed: bool | None = None
    distribution_mean: float | None = None
    distribution_std: float | None = None
    distribution_min: float | None = None
    distribution_max: float | None = None


def make_profiles() -> dict[str, NoiseProfile]:
    """Profiles mirror current env noise calls and owner-config scales.

    G1Walk/G1Joystick still calls ``_obs_noise`` for gyro and gravity even when a
    scale is zero, so those zero-scale fields are intentionally kept to measure
    the current hot-path RNG cost.
    """

    return {
        "ppo_g1_motion_tracking_motrix": NoiseProfile(
            name="ppo_g1_motion_tracking_motrix",
            owner_config="conf/ppo/task/g1_motion_tracking/motrix.yaml",
            default_num_envs=1024,
            fields=(
                NoiseField("linvel", 3, 0.1),
                NoiseField("gyro", 3, 0.2),
                NoiseField("joint_pos", NUM_ACTION, 0.01),
                NoiseField("dof_vel", NUM_ACTION, 1.5),
            ),
            note="G1MotionTracking actor observation noise.",
        ),
        "sac_g1_motion_tracking_mujoco": NoiseProfile(
            name="sac_g1_motion_tracking_mujoco",
            owner_config="conf/offpolicy/task/sac/g1_motion_tracking/mujoco.yaml",
            default_num_envs=2048,
            fields=(
                NoiseField("linvel", 3, 0.1),
                NoiseField("gyro", 3, 0.2),
                NoiseField("joint_pos", NUM_ACTION, 0.01),
                NoiseField("dof_vel", NUM_ACTION, 1.5),
            ),
            note="G1MotionTrackingSAC actor observation noise.",
        ),
        "ppo_g1_walk_flat_motrix": NoiseProfile(
            name="ppo_g1_walk_flat_motrix",
            owner_config="conf/ppo/task/g1_walk_flat/motrix.yaml",
            default_num_envs=2048,
            fields=(
                NoiseField("gyro", 3, 0.2),
                NoiseField("gravity", 3, 0.0),
                NoiseField("joint_pos", NUM_ACTION, 0.01),
                NoiseField("dof_vel", NUM_ACTION, 1.5),
            ),
            note="G1WalkFlat joystick actor observation noise.",
        ),
        "sac_g1_walk_flat_mujoco": NoiseProfile(
            name="sac_g1_walk_flat_mujoco",
            owner_config="conf/offpolicy/task/sac/g1_walk_flat/mujoco.yaml",
            default_num_envs=2048,
            fields=(
                NoiseField("gyro", 3, 0.0),
                NoiseField("gravity", 3, 0.0),
                NoiseField("joint_pos", NUM_ACTION, 0.01),
                NoiseField("dof_vel", NUM_ACTION, 0.1),
            ),
            note="G1WalkFlat SAC actor observation noise.",
        ),
    }


def _dtype(name: str) -> np.dtype:
    if name == "float32":
        return np.dtype(np.float32)
    if name == "float64":
        return np.dtype(np.float64)
    raise ValueError(f"unsupported dtype: {name}")


def _allocate_buffers(profile: NoiseProfile, num_envs: int, dtype: np.dtype) -> list[np.ndarray]:
    return [np.empty((num_envs, field.width), dtype=dtype) for field in profile.fields]


def _scale_array(profile: NoiseProfile, noise_level: float) -> np.ndarray:
    return np.asarray([noise_level * field.scale for field in profile.fields], dtype=np.float64)


def _numpy_random_uniform_alloc(
    profile: NoiseProfile,
    num_envs: int,
    dtype: np.dtype,
    noise_level: float,
) -> list[np.ndarray]:
    out = []
    for field in profile.fields:
        noise = np.random.uniform(-1.0, 1.0, size=(num_envs, field.width)).astype(dtype)
        noise *= noise_level * field.scale
        out.append(noise)
    return out


def _numpy_generator_random_out(
    rng: np.random.Generator,
    buffers: list[np.ndarray],
    profile: NoiseProfile,
    noise_level: float,
) -> None:
    for buffer, field in zip(buffers, profile.fields):
        rng.random(out=buffer, dtype=buffer.dtype)
        buffer *= 2.0
        buffer -= 1.0
        buffer *= noise_level * field.scale


if NUMBA_AVAILABLE:

    @njit(cache=True)  # type: ignore[misc]
    def _numba_seed(seed: int) -> None:
        np.random.seed(seed)

    @njit(parallel=True, fastmath=True, cache=True, nogil=True)  # type: ignore[misc]
    def _numba_fill_four(
        buffer0: np.ndarray,
        buffer1: np.ndarray,
        buffer2: np.ndarray,
        buffer3: np.ndarray,
        scales: np.ndarray,
    ) -> None:
        n = buffer0.shape[0]
        for i in prange(n):
            for j in range(buffer0.shape[1]):
                buffer0[i, j] = (np.random.random() * 2.0 - 1.0) * scales[0]
            for j in range(buffer1.shape[1]):
                buffer1[i, j] = (np.random.random() * 2.0 - 1.0) * scales[1]
            for j in range(buffer2.shape[1]):
                buffer2[i, j] = (np.random.random() * 2.0 - 1.0) * scales[2]
            for j in range(buffer3.shape[1]):
                buffer3[i, j] = (np.random.random() * 2.0 - 1.0) * scales[3]


def _numba_random_prange(buffers: list[np.ndarray], scales: np.ndarray) -> None:
    if not NUMBA_AVAILABLE:
        raise RuntimeError("numba is not available")
    if len(buffers) != 4:
        raise ValueError("numba benchmark expects exactly four noise fields")
    _numba_fill_four(buffers[0], buffers[1], buffers[2], buffers[3], scales)


def _time_ms(fn: Any, *, iters: int, warmup: int) -> tuple[float, float, float]:
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000.0)
    return mean(samples), min(samples), stdev(samples) if len(samples) > 1 else 0.0


def _flatten(buffers: list[np.ndarray]) -> np.ndarray:
    return np.concatenate([buffer.ravel() for buffer in buffers])


def _distribution(buffers: list[np.ndarray]) -> dict[str, float]:
    flat = _flatten(buffers)
    if flat.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(np.mean(flat)),
        "std": float(np.std(flat)),
        "min": float(np.min(flat)),
        "max": float(np.max(flat)),
    }


def _numba_same_seed_is_deterministic(
    profile: NoiseProfile,
    num_envs: int,
    dtype: np.dtype,
    scales: np.ndarray,
    threads: int,
    seed: int,
) -> bool:
    if not NUMBA_AVAILABLE:
        return False
    previous_threads = get_num_threads()
    set_num_threads(threads)
    first = _allocate_buffers(profile, num_envs, dtype)
    second = _allocate_buffers(profile, num_envs, dtype)
    _numba_seed(seed)
    _numba_random_prange(first, scales)
    _numba_seed(seed)
    _numba_random_prange(second, scales)
    set_num_threads(previous_threads)
    return all(np.array_equal(a, b) for a, b in zip(first, second))


def bench_one(
    *,
    profile: NoiseProfile,
    num_envs: int,
    dtype_name: str = "float32",
    thread_counts: list[int] | None = None,
    iters: int = 20,
    warmup: int = 5,
    seed: int = 0,
    noise_level: float = 1.0,
) -> list[BenchCase]:
    dtype = _dtype(dtype_name)
    values = num_envs * profile.total_width
    records: list[BenchCase] = []

    np.random.seed(seed)
    numpy_buffers = _numpy_random_uniform_alloc(profile, num_envs, dtype, noise_level)
    numpy_dist = _distribution(numpy_buffers)

    def numpy_alloc_call() -> None:
        _numpy_random_uniform_alloc(profile, num_envs, dtype, noise_level)

    numpy_ms, numpy_min, numpy_std = _time_ms(
        numpy_alloc_call,
        iters=iters,
        warmup=warmup,
    )
    records.append(
        BenchCase(
            profile=profile.name,
            owner_config=profile.owner_config,
            num_envs=num_envs,
            dtype=dtype_name,
            path="numpy_random_uniform_alloc",
            threads=None,
            values=values,
            mean_ms=numpy_ms,
            min_ms=numpy_min,
            std_ms=numpy_std,
            values_per_s=values / (numpy_ms * 1.0e-3),
            speedup_vs_numpy_alloc=1.0,
            distribution_mean=numpy_dist["mean"],
            distribution_std=numpy_dist["std"],
            distribution_min=numpy_dist["min"],
            distribution_max=numpy_dist["max"],
        )
    )

    generator_buffers = _allocate_buffers(profile, num_envs, dtype)
    generator = np.random.default_rng(seed)
    _numpy_generator_random_out(generator, generator_buffers, profile, noise_level)
    generator_dist = _distribution(generator_buffers)

    def numpy_out_call() -> None:
        _numpy_generator_random_out(generator, generator_buffers, profile, noise_level)

    numpy_out_ms, numpy_out_min, numpy_out_std = _time_ms(
        numpy_out_call,
        iters=iters,
        warmup=warmup,
    )
    records.append(
        BenchCase(
            profile=profile.name,
            owner_config=profile.owner_config,
            num_envs=num_envs,
            dtype=dtype_name,
            path="numpy_generator_random_out",
            threads=None,
            values=values,
            mean_ms=numpy_out_ms,
            min_ms=numpy_out_min,
            std_ms=numpy_out_std,
            values_per_s=values / (numpy_out_ms * 1.0e-3),
            speedup_vs_numpy_alloc=numpy_ms / numpy_out_ms,
            distribution_mean=generator_dist["mean"],
            distribution_std=generator_dist["std"],
            distribution_min=generator_dist["min"],
            distribution_max=generator_dist["max"],
        )
    )

    if not NUMBA_AVAILABLE:
        return records

    assert thread_counts is not None
    max_threads = get_num_threads()
    scales = _scale_array(profile, noise_level)
    previous_threads = max_threads
    try:
        for threads in thread_counts:
            if threads > max_threads:
                continue
            set_num_threads(threads)
            numba_buffers = _allocate_buffers(profile, num_envs, dtype)
            _numba_seed(seed)
            t0 = time.perf_counter()
            _numba_random_prange(numba_buffers, scales)
            compile_ms = (time.perf_counter() - t0) * 1000.0
            numba_dist = _distribution(numba_buffers)

            def numba_call() -> None:
                _numba_random_prange(numba_buffers, scales)

            numba_ms, numba_min, numba_std = _time_ms(
                numba_call,
                iters=iters,
                warmup=warmup,
            )
            same_seed = _numba_same_seed_is_deterministic(
                profile=profile,
                num_envs=min(num_envs, 256),
                dtype=dtype,
                scales=scales,
                threads=threads,
                seed=seed,
            )
            records.append(
                BenchCase(
                    profile=profile.name,
                    owner_config=profile.owner_config,
                    num_envs=num_envs,
                    dtype=dtype_name,
                    path="numba_random_prange",
                    threads=threads,
                    values=values,
                    mean_ms=numba_ms,
                    min_ms=numba_min,
                    std_ms=numba_std,
                    values_per_s=values / (numba_ms * 1.0e-3),
                    speedup_vs_numpy_alloc=numpy_ms / numba_ms,
                    compile_ms=compile_ms,
                    deterministic_same_seed=same_seed,
                    distribution_mean=numba_dist["mean"],
                    distribution_std=numba_dist["std"],
                    distribution_min=numba_dist["min"],
                    distribution_max=numba_dist["max"],
                )
            )
    finally:
        set_num_threads(previous_threads)

    return records


def _case_to_dict(case: BenchCase) -> dict[str, Any]:
    return {
        "profile": case.profile,
        "owner_config": case.owner_config,
        "num_envs": case.num_envs,
        "dtype": case.dtype,
        "path": case.path,
        "threads": case.threads,
        "values": case.values,
        "mean_ms": case.mean_ms,
        "min_ms": case.min_ms,
        "std_ms": case.std_ms,
        "values_per_s": case.values_per_s,
        "mvalues_per_s": case.values_per_s / 1.0e6,
        "speedup_vs_numpy_alloc": case.speedup_vs_numpy_alloc,
        "compile_ms": case.compile_ms,
        "deterministic_same_seed": case.deterministic_same_seed,
        "distribution_mean": case.distribution_mean,
        "distribution_std": case.distribution_std,
        "distribution_min": case.distribution_min,
        "distribution_max": case.distribution_max,
    }


def _format_table(records: list[BenchCase]) -> str:
    headers = [
        "profile",
        "envs",
        "path",
        "thr",
        "mean ms",
        "min ms",
        "M vals/s",
        "speedup",
        "seed",
    ]
    rows = []
    for record in records:
        rows.append(
            [
                record.profile,
                str(record.num_envs),
                record.path,
                "-" if record.threads is None else str(record.threads),
                f"{record.mean_ms:.3f}",
                f"{record.min_ms:.3f}",
                f"{record.values_per_s / 1.0e6:.1f}",
                f"{record.speedup_vs_numpy_alloc:.2f}x",
                "-" if record.deterministic_same_seed is None else str(record.deterministic_same_seed),
            ]
        )
    widths = [len(header) for header in headers]
    for row in rows:
        for idx, value in enumerate(row):
            widths[idx] = max(widths[idx], len(value))

    def fmt(values: list[str]) -> str:
        return " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(values))

    out = [fmt(headers), "-+-".join("-" * width for width in widths)]
    out.extend(fmt(row) for row in rows)
    return "\n".join(out)


def _format_profile_notes(profiles: list[NoiseProfile]) -> str:
    lines = ["Profiles:"]
    for profile in profiles:
        fields = ", ".join(f"{field.name}[{field.width}]x{field.scale:g}" for field in profile.fields)
        lines.append(f"- {profile.name}: {profile.owner_config}; {fields}")
    return "\n".join(lines)


def _select_profiles(names: list[str]) -> list[NoiseProfile]:
    profiles = make_profiles()
    if names == ["all"]:
        return list(profiles.values())
    missing = [name for name in names if name not in profiles]
    if missing:
        raise ValueError(f"unknown profiles: {missing}; available: {sorted(profiles)}")
    return [profiles[name] for name in names]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profiles",
        nargs="+",
        default=["all"],
        help="Profile names to run, or 'all'.",
    )
    parser.add_argument(
        "--num-envs",
        nargs="+",
        type=int,
        default=None,
        help="Override env counts. Defaults to each profile's owner-config count plus benchmark sizes.",
    )
    parser.add_argument("--threads", nargs="+", type=int, default=None)
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--noise-level", type=float, default=1.0)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected_profiles = _select_profiles(args.profiles)
    if args.quick:
        selected_profiles = selected_profiles[:2]
        args.iters = min(args.iters, 3)
        args.warmup = min(args.warmup, 1)

    thread_counts = args.threads or (QUICK_THREADS if args.quick else DEFAULT_THREADS)
    num_envs_default = QUICK_NUM_ENVS if args.quick else DEFAULT_NUM_ENVS

    if NUMBA_AVAILABLE:
        max_threads = get_num_threads()
        skipped_threads = [thread for thread in thread_counts if thread > max_threads]
        thread_counts = [thread for thread in thread_counts if thread <= max_threads]
    else:
        max_threads = 0
        skipped_threads = thread_counts
        thread_counts = []

    print("=" * 88)
    print("Numba RNG vs NumPy RNG: UniLab observation-noise buffer benchmark")
    print("=" * 88)
    print(get_device_info_line())
    print(f"python={platform.python_version()} numpy={np.__version__} numba_available={NUMBA_AVAILABLE}")
    print(f"host numba threads={max_threads}")
    if skipped_threads:
        print(f"skipped numba threads: {skipped_threads}")
    print(_format_profile_notes(selected_profiles))
    print(
        "Seed note: NumPy and Numba RNG streams are not expected to be bit-identical; "
        "the benchmark only checks same-seed repeatability within each numba thread count."
    )

    all_records: list[BenchCase] = []
    for profile in selected_profiles:
        sizes = args.num_envs or sorted({profile.default_num_envs, *num_envs_default})
        for num_envs in sizes:
            print()
            print(f"profile={profile.name} num_envs={num_envs} values={num_envs * profile.total_width}")
            records = bench_one(
                profile=profile,
                num_envs=num_envs,
                dtype_name=args.dtype,
                thread_counts=thread_counts,
                iters=args.iters,
                warmup=args.warmup,
                seed=args.seed,
                noise_level=args.noise_level,
            )
            all_records.extend(records)
            print(_format_table(records))

    if args.output is not None:
        payload = {
            "meta": {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "device": get_device_info_dict(),
                "numpy_version": np.__version__,
                "numba_available": NUMBA_AVAILABLE,
                "dtype": args.dtype,
                "iters": args.iters,
                "warmup": args.warmup,
                "seed": args.seed,
                "noise_level": args.noise_level,
            },
            "results": [_case_to_dict(record) for record in all_records],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nSaved JSON: {args.output.resolve()}")


if __name__ == "__main__":
    main()
