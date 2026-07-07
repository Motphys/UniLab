"""Speed benchmark: numpy oracle vs. fused numba kernel for the
``g1_motion_tracking`` update_state (reward + termination).

Measures the Env-overhead slice issue #663/#665 identified as the single-threaded
bottleneck.  Reports, per ``num_envs`` and thread count:

  * numpy baseline ms, numba ms, speedup,
  * single-core numba ms (fusion-only gain, no parallelism),
  * throughput (env/s) for the overhead slice,
  * a first-call compile time (amortised once per process; ``cache=True`` reuses
    it across runs).

This is the update_state half only — per issue #665, reset_done and RNG are
separate bottlenecks not addressed here.

Run:
    python -m benchmark.numba_profile.bench
    PROBE_NUM_ENVS=32768 python -m benchmark.numba_profile.bench
"""

from __future__ import annotations

import os
import time

from numba import get_num_threads, set_num_threads

from . import numpy_reference as ref
from .numba_fused import FusedUpdateState
from .state import make_batch


def _time(fn, iters: int, warmup: int = 3) -> float:
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters * 1e3  # ms/call


def _bench_size(num_envs: int, thread_counts: list[int], iters: int) -> None:
    b = make_batch(num_envs, seed=0)
    max_threads = get_num_threads()

    # numpy baseline
    numpy_ms = _time(lambda: ref.update_state(b), iters)

    # numba: compile once (first call), then time
    driver1 = FusedUpdateState(num_threads=1)
    t0 = time.perf_counter()
    driver1(b)  # triggers JIT compile on first process run
    compile_ms = (time.perf_counter() - t0) * 1e3
    single_ms = _time(lambda: driver1(b), iters)

    print(f"\n── num_envs = {num_envs:>6}  (iters={iters}, host threads={max_threads}) ──")
    print(f"  numpy baseline           : {numpy_ms:8.3f} ms")
    print(
        f"  numba 1 thread (fusion)  : {single_ms:8.3f} ms   "
        f"({numpy_ms / single_ms:5.2f}x vs numpy)   [compile {compile_ms:.0f} ms once]"
    )

    best = (single_ms, 1)
    for nt in thread_counts:
        if nt > max_threads:
            continue
        driver = FusedUpdateState(num_threads=nt)
        driver(b)  # warm this thread setting
        ms = _time(lambda: driver(b), iters)
        env_per_s = num_envs / (ms * 1e-3)
        marker = ""
        if ms < best[0]:
            best = (ms, nt)
            marker = " *"
        print(
            f"  numba {nt:>3} threads         : {ms:8.3f} ms   "
            f"({numpy_ms / ms:5.2f}x)   {env_per_s / 1e6:5.2f}M env/s{marker}"
        )

    set_num_threads(max_threads)
    best_ms, best_nt = best
    print(
        f"  BEST: {numpy_ms / best_ms:.1f}x at {best_nt} threads "
        f"(fusion {numpy_ms / single_ms:.1f}x + parallel {single_ms / best_ms:.1f}x)"
    )


def main() -> None:
    default_envs = int(os.environ.get("PROBE_NUM_ENVS", "8192"))
    sizes = sorted({default_envs, 8192, 32768}) if default_envs in (8192, 32768) else [default_envs]
    thread_counts = [2, 4, 8, 16, 24, 32, 48, 64]
    iters = int(os.environ.get("PROBE_ITERS", "50"))

    print("=" * 70)
    print("SPEED — g1_motion_tracking update_state (reward + termination)")
    print("numpy oracle vs fused numba prange kernel")
    print("=" * 70)
    for n in sizes:
        _bench_size(n, thread_counts, iters)
    print("\nNote: update_state slice only; reset_done / RNG are separate (issue #665).")


if __name__ == "__main__":
    main()
