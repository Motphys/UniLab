"""MLP benchmark utilities shared between ANE and MLP inference benchmarks."""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class MLPBenchRecord:
    backend: str
    env_num: int
    warmup: int
    repeat: int
    elapsed_sec: List[float]
    mean_sec: float
    std_sec: float
    min_sec: float
    max_sec: float
    envs_per_sec: float


def env_nums_pow2(pow_min: int, pow_max: int) -> List[int]:
    return [2**k for k in range(pow_min, pow_max + 1)]


def mlp_param_count(obs_dim: int, action_dim: int, hidden_dims: List[int]) -> int:
    """Total number of parameters (weights + biases) for obs->hidden->...->action MLP."""
    dims = [obs_dim] + hidden_dims + [action_dim]
    return sum((dims[i] + 1) * dims[i + 1] for i in range(len(dims) - 1))


def trimmed_mean(samples: List[float]) -> Tuple[float, float, List[float]]:
    """Drop one min and one max, return (mean of middle, std of middle, trimmed list)."""
    if len(samples) < 3:
        m = statistics.mean(samples)
        s = statistics.pstdev(samples) if len(samples) > 1 else 0.0
        return m, s, list(samples)
    sorted_s = sorted(samples)
    trimmed = sorted_s[1:-1]
    return statistics.mean(trimmed), statistics.pstdev(trimmed), trimmed


def print_mlp_table(records: List[MLPBenchRecord]) -> None:
    if not records:
        print("No benchmark records.")
        return
    headers = [
        "backend",
        "env_num",
        "mean_sec",
        "std_sec",
        "min_sec",
        "max_sec",
        "envs_per_sec",
    ]
    rows = []
    for r in records:
        rows.append(
            [
                r.backend,
                str(r.env_num),
                f"{r.mean_sec:.6f}",
                f"{r.std_sec:.6f}",
                f"{r.min_sec:.6f}",
                f"{r.max_sec:.6f}",
                f"{r.envs_per_sec:.1f}",
            ]
        )
    col_w = [len(h) for h in headers]
    for row in rows:
        for i, v in enumerate(row):
            col_w[i] = max(col_w[i], len(v))

    def fmt_row(vals: List[str]) -> str:
        return " | ".join(v.ljust(col_w[i]) for i, v in enumerate(vals))

    sep = "-+-".join("-" * w for w in col_w)
    print(fmt_row(headers))
    print(sep)
    for row in rows:
        print(fmt_row(row))
