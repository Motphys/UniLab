#!/usr/bin/env python3
"""Benchmark compute performance: NumPy/PyTorch/MLX."""

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List

try:
    from benchmark.core import (
        available_backends,
        bench_callable,
        mlx_dtype,
        normalize_dtypes,
        numpy_dtype,
        parse_dtypes,
        parse_sizes,
        print_table,
        torch_dtype,
    )
    from benchmark.core.device_info import get_device_info_dict
except ModuleNotFoundError:
    from core import (
        available_backends,
        bench_callable,
        mlx_dtype,
        normalize_dtypes,
        numpy_dtype,
        parse_dtypes,
        parse_sizes,
        print_table,
        torch_dtype,
    )
    from core.device_info import get_device_info_dict

try:
    import numpy as np
except Exception:
    np = None

try:
    import torch
except Exception:
    torch = None

try:
    import mlx.core as mx
except Exception:
    mx = None


@dataclass
class BenchRecord:
    backend: str
    dtype: str
    workload: str
    size: int
    warmup: int
    repeat: int
    elapsed_sec: List[float]
    mean_sec: float
    std_sec: float
    min_sec: float
    max_sec: float
    approx_gflops: float
    gflops_per_sec: float


def matmul_gflops(n: int) -> float:
    return 2.0 * (n**3) / 1e9


def elemwise_gflops(n: int, ops: int = 6) -> float:
    return ops * n * n / 1e9


def summarize(backend, dtype, workload, size, warmup, repeat, elapsed, gflops):
    import statistics

    mean = statistics.mean(elapsed)
    return BenchRecord(
        backend=backend,
        dtype=dtype,
        workload=workload,
        size=size,
        warmup=warmup,
        repeat=repeat,
        elapsed_sec=elapsed,
        mean_sec=mean,
        std_sec=statistics.pstdev(elapsed) if len(elapsed) > 1 else 0.0,
        min_sec=min(elapsed),
        max_sec=max(elapsed),
        approx_gflops=gflops,
        gflops_per_sec=gflops / mean if mean > 0 else 0,
    )


def run_numpy(size, warmup, repeat, dtype_name):
    if not np:
        return []
    dt = numpy_dtype(dtype_name)
    a = np.random.randn(size, size).astype(dt)
    b = np.random.randn(size, size).astype(dt)

    mm_t = bench_callable(lambda: a @ b, lambda: None, warmup, repeat)
    ew_t = bench_callable(
        lambda: np.tanh(a * 1.1 + b * 0.9) + np.sin(a - b), lambda: None, warmup, repeat
    )

    return [
        summarize("numpy", dtype_name, "matmul", size, warmup, repeat, mm_t, matmul_gflops(size)),
        summarize(
            "numpy", dtype_name, "elemwise", size, warmup, repeat, ew_t, elemwise_gflops(size)
        ),
    ]


def run_torch_cpu(size, warmup, repeat, dtype_name):
    if not torch:
        return []
    dt = torch_dtype(dtype_name)
    a = torch.randn(size, size, dtype=dt, device="cpu")
    b = torch.randn(size, size, dtype=dt, device="cpu")

    mm_t = bench_callable(lambda: a @ b, lambda: None, warmup, repeat)
    ew_t = bench_callable(
        lambda: torch.tanh(a * 1.1 + b * 0.9) + torch.sin(a - b), lambda: None, warmup, repeat
    )

    return [
        summarize(
            "torch_cpu", dtype_name, "matmul", size, warmup, repeat, mm_t, matmul_gflops(size)
        ),
        summarize(
            "torch_cpu", dtype_name, "elemwise", size, warmup, repeat, ew_t, elemwise_gflops(size)
        ),
    ]


def run_torch_mps(size, warmup, repeat, dtype_name):
    if not torch or not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
        return []
    dt = torch_dtype(dtype_name)
    a = torch.randn(size, size, dtype=dt, device="mps")
    b = torch.randn(size, size, dtype=dt, device="mps")

    def sync():
        return torch.mps.synchronize()

    mm_t = bench_callable(lambda: a @ b, sync, warmup, repeat)
    ew_t = bench_callable(
        lambda: torch.tanh(a * 1.1 + b * 0.9) + torch.sin(a - b), sync, warmup, repeat
    )

    return [
        summarize(
            "torch_mps", dtype_name, "matmul", size, warmup, repeat, mm_t, matmul_gflops(size)
        ),
        summarize(
            "torch_mps", dtype_name, "elemwise", size, warmup, repeat, ew_t, elemwise_gflops(size)
        ),
    ]


def run_mlx(size, warmup, repeat, dtype_name):
    if not mx:
        return []
    dt = mlx_dtype(dtype_name)
    a = mx.random.normal((size, size), dtype=dt)
    b = mx.random.normal((size, size), dtype=dt)

    def mm():
        out = a @ b
        mx.eval(out)

    def ew():
        out = mx.tanh(a * 1.1 + b * 0.9) + mx.sin(a - b)
        mx.eval(out)

    mm_t = bench_callable(mm, lambda: None, warmup, repeat)
    ew_t = bench_callable(ew, lambda: None, warmup, repeat)

    return [
        summarize("mlx", dtype_name, "matmul", size, warmup, repeat, mm_t, matmul_gflops(size)),
        summarize("mlx", dtype_name, "elemwise", size, warmup, repeat, ew_t, elemwise_gflops(size)),
    ]


def main():
    parser = argparse.ArgumentParser(description="Benchmark NumPy/Torch/MLX compute.")
    parser.add_argument("--sizes", type=str, default=",".join(str(2**k) for k in range(5, 12)))
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--dtypes", type=str, default="float16,float32")
    parser.add_argument("--out", type=str, default="benchmark/outputs/backends/results.json")
    args = parser.parse_args()

    sizes = parse_sizes(args.sizes)
    dtypes = normalize_dtypes(parse_dtypes(args.dtypes))
    backends = available_backends()

    print("Backends:", {k: v for k, v in backends.items() if v})

    all_records = []
    for n in sizes:
        print(f"\nsize={n}")
        for dt in dtypes:
            for fn in [run_numpy, run_torch_cpu, run_torch_mps, run_mlx]:
                try:
                    all_records.extend(fn(n, args.warmup, args.repeat, dt))
                except Exception as e:
                    print(f"  skip {fn.__name__}: {e}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    import json
    from datetime import datetime, timezone

    payload = {
        "meta": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "device_info": get_device_info_dict(),
            "sizes": sizes,
            "dtypes": dtypes,
        },
        "results": [asdict(r) for r in all_records],
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved: {out_path}")

    headers = ["backend", "workload", "dtype", "size", "mean_sec", "gflops/s"]
    rows = [
        [
            r.backend,
            r.workload,
            r.dtype,
            str(r.size),
            f"{r.mean_sec:.6f}",
            f"{r.gflops_per_sec:.1f}",
        ]
        for r in all_records
    ]
    print_table([dict(zip(headers, row)) for row in rows], headers)


if __name__ == "__main__":
    main()
