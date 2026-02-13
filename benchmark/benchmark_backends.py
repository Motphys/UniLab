#!/usr/bin/env python3
"""
Benchmark large-scale compute performance across:
- NumPy
- PyTorch (CPU)
- PyTorch (MPS, if available)
- MLX

Outputs structured JSON results and a concise console table.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Tuple

try:
    import numpy as np
except Exception:  # pragma: no cover - optional dependency at runtime
    np = None

try:
    import torch
except Exception:  # pragma: no cover - optional dependency at runtime
    torch = None

try:
    import mlx.core as mx
except Exception:  # pragma: no cover - optional dependency at runtime
    mx = None

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - optional dependency at runtime
    plt = None


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


def parse_sizes(s: str) -> List[int]:
    vals = []
    for p in s.split(","):
        p = p.strip()
        if not p:
            continue
        vals.append(int(p))
    if not vals:
        raise ValueError("sizes cannot be empty")
    return vals


def pow2_sizes(start_pow: int, end_pow: int) -> List[int]:
    if start_pow > end_pow:
        raise ValueError("pow2_start must be <= pow2_end")
    if start_pow < 0:
        raise ValueError("pow2_start must be >= 0")
    return [2**k for k in range(start_pow, end_pow + 1)]


def format_seconds(x: float) -> str:
    return f"{x:.6f}"


def parse_dtypes(s: str) -> List[str]:
    vals = []
    for p in s.split(","):
        p = p.strip()
        if p:
            vals.append(p)
    if not vals:
        raise ValueError("dtypes cannot be empty")
    return vals


def normalize_dtypes(dtypes: List[str]) -> List[str]:
    # Apple GPU/MPS and AMX-oriented testing: keep float16/float32 only.
    allowed = {"float16", "float32"}
    kept: List[str] = []
    for dt in dtypes:
        if dt in allowed and dt not in kept:
            kept.append(dt)
        elif dt not in allowed:
            print(f"  - skip dtype={dt}: disabled for this benchmark profile")
    if not kept:
        raise ValueError("No valid dtypes left. Use float16,float32.")
    return kept


def numpy_dtype(dtype_name: str):
    if np is None:
        raise RuntimeError("numpy unavailable")
    mapping = {
        "float16": np.float16,
        "float32": np.float32,
    }
    if dtype_name not in mapping:
        raise ValueError(f"unsupported numpy dtype: {dtype_name}")
    return mapping[dtype_name]


def torch_dtype(dtype_name: str):
    if torch is None:
        raise RuntimeError("torch unavailable")
    mapping = {
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if dtype_name not in mapping:
        raise ValueError(f"unsupported torch dtype: {dtype_name}")
    return mapping[dtype_name]


def mlx_dtype(dtype_name: str):
    if mx is None:
        raise RuntimeError("mlx unavailable")
    mapping = {
        "float16": mx.float16,
        "float32": mx.float32,
    }
    if dtype_name not in mapping:
        raise ValueError(f"unsupported mlx dtype: {dtype_name}")
    return mapping[dtype_name]


def matmul_gflops(n: int) -> float:
    # Matrix multiply: roughly 2 * n^3 FLOPs
    return 2.0 * (n ** 3) / 1e9


def elemwise_gflops(n: int, ops_per_elem: int = 6) -> float:
    # Approximate operation count for a synthetic elementwise chain.
    total_elem = n * n
    return ops_per_elem * total_elem / 1e9


def bench_callable(
    fn: Callable[[], None],
    sync_fn: Callable[[], None],
    warmup: int,
    repeat: int,
) -> List[float]:
    for _ in range(warmup):
        fn()
        sync_fn()

    samples: List[float] = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        sync_fn()
        t1 = time.perf_counter()
        samples.append(t1 - t0)
    return samples


def summarize(
    backend: str,
    dtype: str,
    workload: str,
    size: int,
    warmup: int,
    repeat: int,
    elapsed: List[float],
    approx_gflops: float,
) -> BenchRecord:
    mean_sec = statistics.mean(elapsed)
    std_sec = statistics.pstdev(elapsed) if len(elapsed) > 1 else 0.0
    min_sec = min(elapsed)
    max_sec = max(elapsed)
    gflops_per_sec = approx_gflops / mean_sec if mean_sec > 0 else math.inf

    return BenchRecord(
        backend=backend,
        dtype=dtype,
        workload=workload,
        size=size,
        warmup=warmup,
        repeat=repeat,
        elapsed_sec=elapsed,
        mean_sec=mean_sec,
        std_sec=std_sec,
        min_sec=min_sec,
        max_sec=max_sec,
        approx_gflops=approx_gflops,
        gflops_per_sec=gflops_per_sec,
    )


def run_numpy(size: int, warmup: int, repeat: int, dtype_name: str) -> List[BenchRecord]:
    if np is None:
        return []

    rng = np.random.default_rng(42)
    dtype = numpy_dtype(dtype_name)
    a = rng.standard_normal((size, size)).astype(dtype)
    b = rng.standard_normal((size, size)).astype(dtype)

    def mm():
        _ = a @ b

    def ew():
        _ = np.tanh(a * 1.1 + b * 0.9) + np.sin(a - b)

    sync = lambda: None
    mm_t = bench_callable(mm, sync, warmup, repeat)
    ew_t = bench_callable(ew, sync, warmup, repeat)

    return [
        summarize(
            "numpy",
            dtype_name,
            "matmul",
            size,
            warmup,
            repeat,
            mm_t,
            matmul_gflops(size),
        ),
        summarize(
            "numpy",
            dtype_name,
            "elemwise",
            size,
            warmup,
            repeat,
            ew_t,
            elemwise_gflops(size),
        ),
    ]


def run_torch_cpu(size: int, warmup: int, repeat: int, dtype_name: str) -> List[BenchRecord]:
    if torch is None:
        return []

    dtype = torch_dtype(dtype_name)
    a = torch.randn((size, size), dtype=dtype, device="cpu")
    b = torch.randn((size, size), dtype=dtype, device="cpu")

    def mm():
        _ = a @ b

    def ew():
        _ = torch.tanh(a * 1.1 + b * 0.9) + torch.sin(a - b)

    sync = lambda: None
    mm_t = bench_callable(mm, sync, warmup, repeat)
    ew_t = bench_callable(ew, sync, warmup, repeat)

    return [
        summarize(
            "torch_cpu",
            dtype_name,
            "matmul",
            size,
            warmup,
            repeat,
            mm_t,
            matmul_gflops(size),
        ),
        summarize(
            "torch_cpu",
            dtype_name,
            "elemwise",
            size,
            warmup,
            repeat,
            ew_t,
            elemwise_gflops(size),
        ),
    ]


def run_torch_mps(size: int, warmup: int, repeat: int, dtype_name: str) -> List[BenchRecord]:
    if torch is None:
        return []

    if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
        return []

    device = torch.device("mps")
    dtype = torch_dtype(dtype_name)
    a = torch.randn((size, size), dtype=dtype, device=device)
    b = torch.randn((size, size), dtype=dtype, device=device)

    def mm():
        _ = a @ b

    def ew():
        _ = torch.tanh(a * 1.1 + b * 0.9) + torch.sin(a - b)

    sync = lambda: torch.mps.synchronize()
    mm_t = bench_callable(mm, sync, warmup, repeat)
    ew_t = bench_callable(ew, sync, warmup, repeat)

    return [
        summarize(
            "torch_mps",
            dtype_name,
            "matmul",
            size,
            warmup,
            repeat,
            mm_t,
            matmul_gflops(size),
        ),
        summarize(
            "torch_mps",
            dtype_name,
            "elemwise",
            size,
            warmup,
            repeat,
            ew_t,
            elemwise_gflops(size),
        ),
    ]


def run_mlx(size: int, warmup: int, repeat: int, dtype_name: str) -> List[BenchRecord]:
    if mx is None:
        return []

    dtype = mlx_dtype(dtype_name)
    a = mx.random.normal((size, size), dtype=dtype)
    b = mx.random.normal((size, size), dtype=dtype)

    def mm():
        out = a @ b
        mx.eval(out)

    def ew():
        out = mx.tanh(a * 1.1 + b * 0.9) + mx.sin(a - b)
        mx.eval(out)

    sync = lambda: None
    mm_t = bench_callable(mm, sync, warmup, repeat)
    ew_t = bench_callable(ew, sync, warmup, repeat)

    return [
        summarize(
            "mlx",
            dtype_name,
            "matmul",
            size,
            warmup,
            repeat,
            mm_t,
            matmul_gflops(size),
        ),
        summarize(
            "mlx",
            dtype_name,
            "elemwise",
            size,
            warmup,
            repeat,
            ew_t,
            elemwise_gflops(size),
        ),
    ]


def available_backends() -> Dict[str, bool]:
    return {
        "numpy": np is not None,
        "torch_cpu": torch is not None,
        "torch_mps": bool(
            torch is not None
            and hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ),
        "mlx": mx is not None,
    }


def print_table(records: List[BenchRecord]) -> None:
    if not records:
        print("No benchmark records.")
        return

    headers = [
        "backend",
        "workload",
        "dtype",
        "size",
        "mean_sec",
        "std_sec",
        "min_sec",
        "max_sec",
        "gflops/s",
    ]
    rows: List[List[str]] = []
    for r in records:
        rows.append(
            [
                r.backend,
                r.workload,
                r.dtype,
                str(r.size),
                format_seconds(r.mean_sec),
                format_seconds(r.std_sec),
                format_seconds(r.min_sec),
                format_seconds(r.max_sec),
                f"{r.gflops_per_sec:.3f}",
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


def save_plots(
    records: List[BenchRecord],
    plot_dir: Path,
    file_prefix: str,
) -> List[str]:
    if plt is None or not records:
        return []

    plot_dir.mkdir(parents=True, exist_ok=True)
    saved: List[str] = []

    workloads = sorted({r.workload for r in records})
    dtypes = sorted({r.dtype for r in records})
    for workload in workloads:
        for dtype in dtypes:
            subset = [r for r in records if r.workload == workload and r.dtype == dtype]
            if not subset:
                continue

            backends = sorted({r.backend for r in subset})
            gfig, gax = plt.subplots(figsize=(8, 5))
            for backend in backends:
                b_records = sorted(
                    [r for r in subset if r.backend == backend],
                    key=lambda x: x.size,
                )
                x = [r.size for r in b_records]
                y = [r.gflops_per_sec for r in b_records]
                gax.plot(x, y, marker="o", label=backend)

            gax.set_title(f"GFLOPS/s vs size ({workload}, {dtype})")
            gax.set_xlabel("matrix size (N for NxN)")
            gax.set_ylabel("GFLOPS/s")
            gax.set_xscale("log", base=2)
            gax.grid(True, alpha=0.3)
            gax.legend()
            gfig.tight_layout()

            g_out = plot_dir / f"{file_prefix}_{workload}_{dtype}_gflops.png"
            gfig.savefig(g_out, dpi=160)
            plt.close(gfig)
            saved.append(str(g_out.resolve()))

    return saved


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark NumPy / Torch(CPU,MPS) / MLX large-scale compute."
    )
    parser.add_argument(
        "--sizes",
        type=str,
        default="",
        help="Comma-separated square matrix sizes, e.g. 1024,2048,4096. If empty, uses --pow2-start/--pow2-end.",
    )
    parser.add_argument(
        "--pow2-start",
        type=int,
        default=5,
        help="Start exponent for power-of-two sizes when --sizes is empty (default: 5 -> 32)",
    )
    parser.add_argument(
        "--pow2-end",
        type=int,
        default=14,
        help="End exponent for power-of-two sizes when --sizes is empty (default: 14 -> 16384)",
    )
    parser.add_argument("--warmup", type=int, default=2, help="Warmup iterations")
    parser.add_argument("--repeat", type=int, default=5, help="Measured iterations")
    parser.add_argument(
        "--dtypes",
        type=str,
        default="float16,float32",
        help="Comma-separated dtypes to test, e.g. float16,float32",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="benchmark/outputs/backends/benchmark_results.json",
        help="Output JSON file path (default under benchmark/outputs/backends/)",
    )
    parser.add_argument(
        "--plot-dir",
        type=str,
        default="",
        help="Directory to save plot images, defaults to the same directory as --out",
    )
    args = parser.parse_args()

    sizes = parse_sizes(args.sizes) if args.sizes.strip() else pow2_sizes(args.pow2_start, args.pow2_end)
    dtypes = normalize_dtypes(parse_dtypes(args.dtypes))
    backends = available_backends()

    print("Detected backends:")
    for k, v in backends.items():
        print(f"  - {k}: {'yes' if v else 'no'}")

    all_records: List[BenchRecord] = []
    skipped_cases: List[Dict[str, str]] = []
    for n in sizes:
        print(f"\nRunning size={n} ...")
        for dtype_name in dtypes:
            for backend_name, fn in (
                ("numpy", run_numpy),
                ("torch_cpu", run_torch_cpu),
                ("torch_mps", run_torch_mps),
                ("mlx", run_mlx),
            ):
                if n > 1024 and backend_name in {"numpy", "torch_cpu"}:
                    skipped_cases.append(
                        {
                            "backend": backend_name,
                            "dtype": dtype_name,
                            "size": str(n),
                            "reason": "Skipped by policy: backend disabled for size > 1024",
                        }
                    )
                    print(
                        f"  - skipped {backend_name}[{dtype_name}] size={n}: "
                        "disabled for size > 1024"
                    )
                    continue
                try:
                    all_records.extend(fn(n, args.warmup, args.repeat, dtype_name))
                except Exception as e:
                    skipped_cases.append(
                        {
                            "backend": backend_name,
                            "dtype": dtype_name,
                            "size": str(n),
                            "reason": str(e),
                        }
                    )
                    print(f"  - skipped {backend_name}[{dtype_name}] size={n}: {e}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plot_dir = (
        Path(args.plot_dir)
        if args.plot_dir
        else out_path.resolve().parent
    )
    plot_files = save_plots(all_records, plot_dir=plot_dir, file_prefix=out_path.stem)

    payload = {
        "meta": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "sizes": sizes,
            "dtypes": dtypes,
            "warmup": args.warmup,
            "repeat": args.repeat,
            "available_backends": backends,
            "matplotlib_available": plt is not None,
            "plot_files": plot_files,
            "skipped_cases": skipped_cases,
        },
        "results": [asdict(r) for r in all_records],
    }

    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved structured results to: {out_path.resolve()}")
    if plt is None:
        print("matplotlib not available; skipped plot generation.")
    elif plot_files:
        print("Saved plots:")
        for f in plot_files:
            print(f"  - {f}")
    print()
    print_table(all_records)


if __name__ == "__main__":
    main()
