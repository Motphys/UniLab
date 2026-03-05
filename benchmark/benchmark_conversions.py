#!/usr/bin/env python3
"""
Benchmark same-dtype transfer efficiency across backends:
- numpy
- torch (cpu)
- torch (mps)
- mlx
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

try:
    from benchmark.core import bench_callable, parse_sizes, parse_dtypes, normalize_dtypes, available_backends, numpy_dtype, torch_dtype, mlx_dtype, print_table
    from benchmark.core.device_info import get_device_info_dict, get_device_info_line
except ModuleNotFoundError:
    from core import bench_callable, parse_sizes, parse_dtypes, normalize_dtypes, available_backends, numpy_dtype, torch_dtype, mlx_dtype, print_table
    from core.device_info import get_device_info_dict, get_device_info_line

_IS_MACOS = platform.system() == "Darwin"

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

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker
except Exception:
    plt = None

@dataclass
class ConversionRecord:
    source_backend: str
    target_backend: str
    source_dtype: str
    target_dtype: str
    size: int
    warmup: int
    repeat: int
    bytes_in: int
    bytes_out: int
    elapsed_sec: List[float]
    mean_sec: float
    std_sec: float
    min_sec: float
    max_sec: float
    effective_gbps: float

def dtype_bytes(dtype_name: str) -> int:
    if dtype_name in ("float16",):
        return 2
    if dtype_name in ("float32", "int32"):
        return 4
    if dtype_name in ("float64", "int64"):
        return 8
    raise ValueError(f"Unsupported dtype: {dtype_name}")

def create_source(backend: str, size: int, dtype_name: str):
    shape = (size, size)
    if backend == "numpy":
        if np is None:
            raise RuntimeError("numpy unavailable")
        return np.random.standard_normal(shape).astype(numpy_dtype(dtype_name))

    if backend == "torch_cpu":
        if torch is None:
            raise RuntimeError("torch unavailable")
        return torch.randn(shape, dtype=torch_dtype(dtype_name), device="cpu")

    if backend == "torch_mps":
        if torch is None or not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
            raise RuntimeError("torch mps unavailable")
        return torch.randn(shape, dtype=torch_dtype(dtype_name), device="mps")

    if backend == "torch_cuda":
        if torch is None or not torch.cuda.is_available():
            raise RuntimeError("torch cuda unavailable")
        return torch.randn(shape, dtype=torch_dtype(dtype_name), device="cuda")

    if backend == "mlx":
        if mx is None:
            raise RuntimeError("mlx unavailable")
        arr = mx.random.normal(shape, dtype=mlx_dtype(dtype_name))
        mx.eval(arr)
        return arr

    raise ValueError(f"Unsupported backend: {backend}")

def to_numpy(value, source_backend: str):
    if np is None:
        raise RuntimeError("numpy unavailable")
    if source_backend == "numpy":
        return value
    if source_backend in ("torch_cpu", "torch_mps", "torch_cuda"):
        return value.detach().to("cpu").numpy()
    if source_backend == "mlx":
        return np.array(value)
    raise ValueError(f"Unsupported source backend: {source_backend}")

def from_numpy(arr, target_backend: str, target_dtype_name: str):
    if target_backend == "numpy":
        return arr.astype(numpy_dtype(target_dtype_name), copy=False)

    if target_backend == "torch_cpu":
        if torch is None:
            raise RuntimeError("torch unavailable")
        t = torch.from_numpy(arr)
        if t.dtype != torch_dtype(target_dtype_name):
            t = t.to(dtype=torch_dtype(target_dtype_name))
        return t

    if target_backend == "torch_mps":
        if torch is None:
            raise RuntimeError("torch unavailable")
        t = torch.from_numpy(arr)
        if t.dtype != torch_dtype(target_dtype_name):
            t = t.to(dtype=torch_dtype(target_dtype_name))
        return t.to(device="mps")

    if target_backend == "torch_cuda":
        if torch is None or not torch.cuda.is_available():
            raise RuntimeError("torch cuda unavailable")
        t = torch.from_numpy(arr)
        if t.dtype != torch_dtype(target_dtype_name):
            t = t.to(dtype=torch_dtype(target_dtype_name))
        return t.to(device="cuda")

    if target_backend == "mlx":
        if mx is None:
            raise RuntimeError("mlx unavailable")
        return mx.array(arr, dtype=mlx_dtype(target_dtype_name))

    raise ValueError(f"Unsupported target backend: {target_backend}")

def convert_value(value, source_backend: str, target_backend: str, target_dtype_name: str):
    # Only benchmark same-dtype transfers. Cross-dtype conversion is intentionally excluded.
    if source_backend == target_backend:
        return value

    # Fast paths first.
    if source_backend == "torch_cpu" and target_backend == "torch_mps":
        return value.to(device="mps")
    if source_backend == "torch_mps" and target_backend == "torch_cpu":
        return value.to(device="cpu")
    if source_backend == "torch_cpu" and target_backend == "torch_cuda":
        return value.to(device="cuda")
    if source_backend == "torch_cuda" and target_backend == "torch_cpu":
        return value.to(device="cpu")
    if source_backend == "torch_cuda" and target_backend == "torch_mps":
        return value.to(device="cpu").to(device="mps")
    if source_backend == "torch_mps" and target_backend == "torch_cuda":
        return value.to(device="cpu").to(device="cuda")

    # DLPack bridge: mlx -> torch
    if source_backend == "mlx" and target_backend == "torch_cpu":
        t = torch.from_dlpack(value)
        if t.dtype != torch_dtype(target_dtype_name):
            t = t.to(dtype=torch_dtype(target_dtype_name))
        return t
    if source_backend == "mlx" and target_backend == "torch_mps":
        t = torch.from_dlpack(value)
        if t.dtype != torch_dtype(target_dtype_name):
            t = t.to(dtype=torch_dtype(target_dtype_name))
        return t.to(device="mps")

    # DLPack bridge: torch -> mlx
    if source_backend == "torch_cpu" and target_backend == "mlx":
        return mx.array(torch.utils.dlpack.to_dlpack(value.detach()))
    if source_backend == "torch_mps" and target_backend == "mlx":
        # MLX currently cannot consume an MPS torch DLPack capsule directly.
        # Keep the path minimal: one device hop to CPU then DLPack import.
        cpu_value = value.detach().to(device="cpu")
        return mx.array(torch.utils.dlpack.to_dlpack(cpu_value))

    arr = to_numpy(value, source_backend)
    return from_numpy(arr, target_backend, target_dtype_name)

def sync_if_needed(source_backend: str, target_backend: str, out_value) -> None:
    if target_backend == "mlx":
        mx.eval(out_value)
    if torch is not None and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        if source_backend == "torch_mps" or target_backend == "torch_mps":
            torch.mps.synchronize()
    if torch is not None and torch.cuda.is_available():
        if source_backend == "torch_cuda" or target_backend == "torch_cuda":
            torch.cuda.synchronize()

def summarize(
    source_backend: str,
    target_backend: str,
    source_dtype_name: str,
    target_dtype_name: str,
    size: int,
    warmup: int,
    repeat: int,
    elapsed: List[float],
) -> ConversionRecord:
    mean_sec = statistics.mean(elapsed)
    std_sec = statistics.pstdev(elapsed) if len(elapsed) > 1 else 0.0
    min_sec = min(elapsed)
    max_sec = max(elapsed)
    bytes_in = size * size * dtype_bytes(source_dtype_name)
    bytes_out = size * size * dtype_bytes(target_dtype_name)
    effective_gbps = (bytes_in + bytes_out) / mean_sec / 1e9 if mean_sec > 0 else math.inf

    return ConversionRecord(
        source_backend=source_backend,
        target_backend=target_backend,
        source_dtype=source_dtype_name,
        target_dtype=target_dtype_name,
        size=size,
        warmup=warmup,
        repeat=repeat,
        bytes_in=bytes_in,
        bytes_out=bytes_out,
        elapsed_sec=elapsed,
        mean_sec=mean_sec,
        std_sec=std_sec,
        min_sec=min_sec,
        max_sec=max_sec,
        effective_gbps=effective_gbps,
    )

def _pair_key(r: ConversionRecord) -> str:
    return f"{r.source_backend}->{r.target_backend}"

def _dtype_key(r: ConversionRecord) -> str:
    return f"{r.source_dtype}->{r.target_dtype}"

def _positive_ylim(records: List[ConversionRecord], metric_name: str) -> Tuple[float, float]:
    vals = [float(getattr(r, metric_name)) for r in records if float(getattr(r, metric_name)) > 0]
    if not vals:
        return (1e-9, 1.0)
    lo = min(vals)
    hi = max(vals)
    return (max(lo * 0.8, 1e-12), hi * 1.25)

def load_records_from_json(json_path: Path) -> List[ConversionRecord]:
    payload: Dict[str, Any] = json.loads(json_path.read_text(encoding="utf-8"))
    raw_results = payload.get("results", [])
    records: List[ConversionRecord] = []
    for item in raw_results:
        records.append(
            ConversionRecord(
                source_backend=item["source_backend"],
                target_backend=item["target_backend"],
                source_dtype=item["source_dtype"],
                target_dtype=item["target_dtype"],
                size=int(item["size"]),
                warmup=int(item["warmup"]),
                repeat=int(item["repeat"]),
                bytes_in=int(item["bytes_in"]),
                bytes_out=int(item["bytes_out"]),
                elapsed_sec=list(item.get("elapsed_sec", [])),
                mean_sec=float(item["mean_sec"]),
                std_sec=float(item["std_sec"]),
                min_sec=float(item["min_sec"]),
                max_sec=float(item["max_sec"]),
                effective_gbps=float(item["effective_gbps"]),
            )
        )
    return records

def save_plots(records: List[ConversionRecord], plot_dir: Path, file_prefix: str) -> List[str]:
    if plt is None or not records:
        return []

    plot_dir.mkdir(parents=True, exist_ok=True)
    saved: List[str] = []
    if _IS_MACOS:
        backend_order = ["numpy", "mlx", "torch_cpu", "torch_mps"]
        backend_label = {
            "numpy": "numpy",
            "mlx": "mlx",
            "torch_cpu": "torch.cpu",
            "torch_mps": "torch.mps",
        }
    else:
        backend_order = ["numpy", "torch_cpu", "torch_cuda"]
        backend_label = {
            "numpy": "numpy",
            "torch_cpu": "torch.cpu",
            "torch_cuda": "torch.cuda",
        }
    dtype_order = ["float32", "float16"]
    dtype_style = {
        "float32": {"color": "#1f77b4", "marker": "o"},
        "float16": {"color": "#ff7f0e", "marker": "s"},
    }

    n_backends = len(backend_order)
    fig, axes = plt.subplots(
        n_backends,
        n_backends,
        figsize=(5.5 * n_backends, 4.5 * n_backends),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    y_lo, y_hi = _positive_ylim(records, "mean_sec")
    legend_handles = {}

    for row, dst in enumerate(backend_order):
        for col, src in enumerate(backend_order):
            ax = axes[row][col]
            has_any_curve = False
            for dtype_name in dtype_order:
                vals = sorted(
                    [
                        r
                        for r in records
                        if r.source_backend == src
                        and r.target_backend == dst
                        and r.source_dtype == dtype_name
                        and r.target_dtype == dtype_name
                    ],
                    key=lambda x: x.size,
                )
                if not vals:
                    continue

                x = [v.size for v in vals]
                y = [v.mean_sec for v in vals]
                style = dtype_style[dtype_name]
                line = ax.plot(
                    x,
                    y,
                    marker=style["marker"],
                    linewidth=1.2,
                    markersize=3.8,
                    color=style["color"],
                    label=dtype_name,
                )[0]
                if dtype_name not in legend_handles:
                    legend_handles[dtype_name] = line
                has_any_curve = True

            ax.set_yscale("log")
            ax.set_xscale("log", base=2)
            ax.set_ylim(y_lo, y_hi)
            ax.grid(True, alpha=0.25)

            if row == 0:
                ax.set_title(f"From: {backend_label[src]}", fontsize=10.5)
            if col == 0:
                ax.set_ylabel(f"To: {backend_label[dst]}\ntime (sec)", fontsize=9.5)
            if row == n_backends - 1:
                ax.set_xlabel("size (N for NxN)", fontsize=9.5)
            if not has_any_curve:
                ax.text(
                    0.5,
                    0.5,
                    "no data",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=8.5,
                    alpha=0.7,
                )

    fig.suptitle(
        f"Conversion time vs size (4x4 To/From storage)\n{get_device_info_line()}",
        fontsize=13,
        y=0.995,
    )
    if legend_handles:
        fig.legend(
            [legend_handles[d] for d in dtype_order if d in legend_handles],
            [d for d in dtype_order if d in legend_handles],
            loc="upper center",
            bbox_to_anchor=(0.5, 0.972),
            ncol=2,
            fontsize=10,
            frameon=False,
        )
    all_sizes_fig1 = sorted({r.size for r in records})
    for ax_row in axes:
        for ax in ax_row:
            ax.set_xticks(all_sizes_fig1)
            ax.xaxis.set_major_formatter(
                matplotlib.ticker.FuncFormatter(lambda v, _: str(int(v)))
            )
            ax.tick_params(axis="x", labelrotation=45, labelsize=7.5)
    fig.tight_layout(rect=[0.02, 0.03, 1, 0.95])
    outfile = plot_dir / f"{file_prefix}_conversion_time_4x4.png"
    fig.savefig(outfile, dpi=180, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    saved.append(str(outfile.resolve()))

    # Focused figure: only the main 3 backends per platform.
    # Layout: 2 rows (float16, float32) x 3 cols (one From backend per column).
    if _IS_MACOS:
        focused_backends = ["numpy", "mlx", "torch_mps"]
        to_colors = {
            "numpy": "#1f77b4",
            "mlx": "#ff7f0e",
            "torch_mps": "#2ca02c",
        }
    else:
        focused_backends = ["numpy", "torch_cpu", "torch_cuda"]
        to_colors = {
            "numpy": "#1f77b4",
            "torch_cpu": "#9467bd",
            "torch_cuda": "#2ca02c",
        }
    row_dtypes = ["float16", "float32"]

    fig2, axes2 = plt.subplots(
        2,
        3,
        figsize=(14.5, 8.0),
        sharex=True,
        sharey=True,
    )
    y_lo2, y_hi2 = _positive_ylim(
        [
            r
            for r in records
            if r.source_backend in {"numpy", "mlx", "torch_mps"}
            and r.target_backend in {"numpy", "mlx", "torch_mps"}
            and r.source_dtype == r.target_dtype
            and r.source_dtype in {"float16", "float32"}
            and r.source_backend != r.target_backend
        ],
        "mean_sec",
    )
    legend_handles2 = {}

    for row, dtype_name in enumerate(row_dtypes):
        for col, src in enumerate(focused_backends):
            ax = axes2[row][col]
            dst_candidates = [b for b in focused_backends if b != src]
            for dst in dst_candidates:
                vals = sorted(
                    [
                        r
                        for r in records
                        if r.source_backend == src
                        and r.target_backend == dst
                        and r.source_dtype == dtype_name
                        and r.target_dtype == dtype_name
                    ],
                    key=lambda x: x.size,
                )
                if not vals:
                    continue
                x = [v.size for v in vals]
                y = [v.mean_sec for v in vals]
                label = f"{backend_label[src]} -> {backend_label[dst]}"
                line = ax.plot(
                    x,
                    y,
                    marker="o",
                    linewidth=1.3,
                    markersize=3.8,
                    color=to_colors[dst],
                    label=label,
                )[0]
                if label not in legend_handles2:
                    legend_handles2[label] = line

            if row == 0:
                ax.set_title(f"From: {backend_label[src]}", fontsize=10.5)
            if col == 0:
                ax.set_ylabel(f"{dtype_name}\ntime (sec)", fontsize=9.5)
            if row == len(row_dtypes) - 1:
                ax.set_xlabel("size (N for NxN)", fontsize=9.5)
            ax.set_yscale("log")
            ax.set_xscale("log", base=2)
            ax.set_ylim(y_lo2, y_hi2)
            ax.grid(True, alpha=0.25)

            # Mark empty panels explicitly when a backend is unavailable.
            if not ax.lines:
                ax.text(
                    0.5,
                    0.5,
                    "no data",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=8.5,
                    alpha=0.7,
                )

    focused_label = "numpy/mlx/torch.mps" if _IS_MACOS else "numpy/torch.cpu/torch.cuda"
    fig2.suptitle(
        f"Conversion time vs size (cols=From, lines=To; {focused_label})\n{get_device_info_line()}",
        fontsize=12.5,
        y=0.995,
    )
    if legend_handles2:
        fig2.legend(
            list(legend_handles2.values()),
            list(legend_handles2.keys()),
            loc="upper center",
            bbox_to_anchor=(0.5, 0.96),
            ncol=3,
            fontsize=9.5,
            frameon=False,
        )
    all_sizes_fig2 = sorted({r.size for r in records})
    for ax_row in axes2:
        for ax in ax_row:
            ax.set_xticks(all_sizes_fig2)
            ax.xaxis.set_major_formatter(
                matplotlib.ticker.FuncFormatter(lambda v, _: str(int(v)))
            )
            ax.tick_params(axis="x", labelrotation=45, labelsize=7.5)
    fig2.tight_layout(rect=[0.02, 0.03, 1, 0.92])
    outfile2 = plot_dir / f"{file_prefix}_conversion_time_numpy_mlx_torchmps_2x3.png"
    fig2.savefig(outfile2, dpi=180, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig2)
    saved.append(str(outfile2.resolve()))

    return saved

def print_table(records: List[ConversionRecord]) -> None:
    if not records:
        print("No conversion records.")
        return

    headers = ["src", "dst", "dtype", "size", "mean_sec", "eff_GB/s"]
    rows: List[List[str]] = []
    for r in records:
        rows.append(
            [
                r.source_backend,
                r.target_backend,
                f"{r.source_dtype}->{r.target_dtype}",
                str(r.size),
                f"{r.mean_sec:.6f}",
                f"{r.effective_gbps:.3f}",
            ]
        )

    col_w = [len(h) for h in headers]
    for row in rows:
        for i, v in enumerate(row):
            col_w[i] = max(col_w[i], len(v))

    def fmt(vals: List[str]) -> str:
        return " | ".join(v.ljust(col_w[i]) for i, v in enumerate(vals))

    print(fmt(headers))
    print("-+-".join("-" * w for w in col_w))
    for row in rows:
        print(fmt(row))

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark conversion efficiency among numpy/torch(cpu,mps)/mlx."
    )
    parser.add_argument("--sizes", type=str, default=",".join(str(2**k) for k in range(5, 15)), help="Comma-separated sizes.")
    parser.add_argument("--dtypes", type=str, default="float16,float32", help="Dtypes to benchmark (same-dtype paths only).")
    parser.add_argument("--warmup", type=int, default=2, help="Warmup iterations.")
    parser.add_argument("--repeat", type=int, default=5, help="Measured iterations.")
    parser.add_argument(
        "--out",
        type=str,
        default="benchmark/outputs/conversions/benchmark_conversions.json",
        help="Output JSON path.",
    )
    parser.add_argument(
        "--plot-dir",
        type=str,
        default="",
        help="Directory for plots, defaults to the same directory as --out.",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Skip benchmarking and redraw plots from an existing JSON result file.",
    )
    parser.add_argument(
        "--plot-json",
        type=str,
        default="",
        help="Input JSON path for --plot-only. Defaults to --out if omitted.",
    )
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plot_dir = Path(args.plot_dir) if args.plot_dir else out_path.resolve().parent

    if args.plot_only:
        json_in = Path(args.plot_json) if args.plot_json else out_path
        if not json_in.exists():
            raise FileNotFoundError(f"plot-only JSON not found: {json_in}")
        records = load_records_from_json(json_in)
        plot_files = save_plots(records, plot_dir=plot_dir, file_prefix=out_path.stem)
        print(f"Loaded records from: {json_in.resolve()}")
        if plot_files:
            print("Saved plots:")
            for f in plot_files:
                print(f"  - {f}")
        return

    sizes = parse_sizes(args.sizes)
    dtypes = normalize_dtypes(parse_dtypes(args.dtypes))
    backends = available_backends()
    enabled_backends = [k for k, v in backends.items() if v]

    print("Detected backends:")
    for k, v in backends.items():
        print(f"  - {k}: {'yes' if v else 'no'}")

    records: List[ConversionRecord] = []
    skipped: List[Dict[str, str]] = []

    pairs: List[Tuple[str, str]] = []
    for src in enabled_backends:
        for dst in enabled_backends:
            if src == dst:
                continue
            pairs.append((src, dst))

    for size in sizes:
        print(f"\nRunning conversion benchmarks for size={size} ...")
        for src, dst in pairs:
            for dtype_name in dtypes:
                case_name = f"{src}({dtype_name})->{dst}({dtype_name})"
                try:
                    source = create_source(src, size, dtype_name)

                    def op() -> None:
                        out = convert_value(source, src, dst, dtype_name)
                        sync_if_needed(src, dst, out)

                    elapsed = bench_callable(op, lambda: None, args.warmup, args.repeat)
                    records.append(
                        summarize(
                            source_backend=src,
                            target_backend=dst,
                            source_dtype_name=dtype_name,
                            target_dtype_name=dtype_name,
                            size=size,
                            warmup=args.warmup,
                            repeat=args.repeat,
                            elapsed=elapsed,
                        )
                    )
                except Exception as e:
                    skipped.append(
                        {
                            "size": str(size),
                            "case": case_name,
                            "reason": str(e),
                        }
                    )
                    print(f"  - skipped {case_name}: {e}")

    plot_files = save_plots(records, plot_dir=plot_dir, file_prefix=out_path.stem)
    payload = {
        "meta": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "device_info": get_device_info_dict(),
            "sizes": sizes,
            "dtypes": dtypes,
            "warmup": args.warmup,
            "repeat": args.repeat,
            "available_backends": backends,
            "matplotlib_available": plt is not None,
            "plot_files": plot_files,
        },
        "results": [asdict(r) for r in records],
        "skipped": skipped,
    }

    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved structured results to: {out_path.resolve()}")
    if plt is None:
        print("matplotlib not available; skipped plot generation.")
    elif plot_files:
        print("Saved plots:")
        for f in plot_files:
            print(f"  - {f}")
    if skipped:
        print(f"Skipped cases: {len(skipped)}")
    print()
    print_table(records)

if __name__ == "__main__":
    main()
