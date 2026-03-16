"""Benchmark record dataclasses."""

from dataclasses import dataclass
from typing import List


@dataclass
class BenchRecord:
    backend: str
    workload: str
    dtype: str
    size: int
    warmup: int
    repeat: int
    elapsed_sec: List[float]
    mean_sec: float
    std_sec: float
    min_sec: float
    max_sec: float
    metric: float
    metric_name: str
