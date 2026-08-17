"""Benchmark execution utilities."""

import statistics
import time
from typing import Callable, List


def bench_callable(
    fn: Callable[[], None],
    sync_fn: Callable[[], None],
    warmup: int,
    repeat: int,
) -> List[float]:
    for _ in range(warmup):
        fn()
        sync_fn()
    samples = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        sync_fn()
        samples.append(time.perf_counter() - t0)
    return samples


def summarize(elapsed: List[float], metric: float, metric_name: str, **kwargs) -> dict:
    return {
        **kwargs,
        "elapsed_sec": elapsed,
        "mean_sec": statistics.mean(elapsed),
        "std_sec": statistics.pstdev(elapsed) if len(elapsed) > 1 else 0.0,
        "min_sec": min(elapsed),
        "max_sec": max(elapsed),
        "metric": metric,
        "metric_name": metric_name,
    }
