#!/usr/bin/env python3
"""
Benchmark MLX compile acceleration (mx.compile vs plain MLX).

Tests the speedup gained from mx.compile across different kernel types:
  - element-wise activation (tanh chain)
  - fused linear + activation (single layer)
  - MLP forward pass (multi-layer)
  - custom composed math (softmax + layer_norm inspired)

Sweeps batch sizes 2^7..2^14.  For each kernel / batch-size, measures:
  - first-call latency (includes compile cache miss)
  - steady-state latency (plain and compiled)
  - speedup = plain_mean / compiled_mean

Outputs JSON results and a multi-panel PNG plot.

Usage:
    python benchmark/benchmark_mlx_compile.py
    python benchmark/benchmark_mlx_compile.py --warmup 10 --repeat 50
from __future__ import annotations

"""

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

try:
    from benchmark.core.device_info import get_device_info_dict, get_device_info_line
except ModuleNotFoundError:
    from core.device_info import get_device_info_dict, get_device_info_line

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import mlx.core as mx
    import mlx.nn as nn

    _HAS_MLX = True
except ImportError:
    mx = None  # type: ignore
    nn = None  # type: ignore
    _HAS_MLX = False

# ── default sweep ──────────────────────────────────────────────────────────────
_DEFAULT_BATCH_SIZES = [2**i for i in range(7, 15)]  # 128 .. 16384
_DEFAULT_HIDDEN = 256
_DEFAULT_OBS_DIM = 48
_DEFAULT_ACTION_DIM = 12
_DEFAULT_NUM_LAYERS = 3


# ── data classes ───────────────────────────────────────────────────────────────
@dataclass
class CompileRecord:
    kernel: str
    batch_size: int
    # first-call (includes compile)
    first_call_plain_sec: float
    first_call_compiled_sec: float
    # steady-state (after warmup)
    plain_mean_sec: float
    plain_std_sec: float
    compiled_mean_sec: float
    compiled_std_sec: float
    speedup: float  # plain_mean / compiled_mean


# ── timing helper ──────────────────────────────────────────────────────────────
def _bench(fn: Callable[[], None], warmup: int, repeat: int) -> List[float]:
    """Run *fn* warmup times, then time it repeat times. Returns list of elapsed [sec]."""
    for _ in range(warmup):
        fn()
    mx.eval(mx.array(0))  # ensure GPU work flushed before timing
    times: List[float] = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        mx.eval(mx.array(0))  # synchronize
        times.append(time.perf_counter() - t0)
    return times


def _first_call_time(fn: Callable[[], None]) -> float:
    """Measure a single cold-call latency (before any compilation caching)."""
    mx.eval(mx.array(0))
    t0 = time.perf_counter()
    fn()
    mx.eval(mx.array(0))
    return time.perf_counter() - t0


# ── kernel definitions for plain and compiled variants ────────────────────────
def _build_elementwise(batch_size: int, hidden: int):
    """Chained tanh + element-wise ops — a common activation chain."""
    x = mx.random.normal((batch_size, hidden), dtype=mx.float32)
    mx.eval(x)

    def plain():
        y = mx.tanh(x)
        y = mx.tanh(y + 0.1)
        y = mx.tanh(y * 1.5 - 0.5)
        mx.eval(y)

    compiled_fn = mx.compile(lambda v: mx.tanh(mx.tanh(v + 0.1) * 1.5 - 0.5))

    def compiled():
        y = compiled_fn(x)
        mx.eval(y)

    return plain, compiled


def _build_linear_activation(batch_size: int, obs_dim: int, hidden: int):
    """Single linear layer + tanh — fundamental op in policy networks."""
    layer = nn.Linear(obs_dim, hidden)
    # force weight init
    x = mx.zeros((1, obs_dim), dtype=mx.float32)
    mx.eval(layer(x))

    x = mx.random.normal((batch_size, obs_dim), dtype=mx.float32)
    mx.eval(x)

    def plain():
        y = mx.tanh(layer(x))
        mx.eval(y)

    compiled_fn = mx.compile(lambda v: mx.tanh(layer(v)))

    def compiled():
        y = compiled_fn(x)
        mx.eval(y)

    return plain, compiled


def _build_mlp(batch_size: int, obs_dim: int, action_dim: int, hidden: int, n_layers: int):
    """Multi-layer MLP — typical locomotion policy forward pass."""
    dims = [obs_dim] + [hidden] * n_layers + [action_dim]

    class _MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.linears = [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]

        def __call__(self, x):
            for i, lin in enumerate(self.linears):
                x = lin(x)
                if i < len(self.linears) - 1:
                    x = mx.tanh(x)
            return x

    model = _MLP()
    # force weight init
    _init_x = mx.zeros((1, obs_dim), dtype=mx.float32)
    mx.eval(model(_init_x))

    x = mx.random.normal((batch_size, obs_dim), dtype=mx.float32)
    mx.eval(x)

    def plain():
        y = model(x)
        mx.eval(y)

    compiled_fn = mx.compile(lambda v: model(v))

    def compiled():
        y = compiled_fn(x)
        mx.eval(y)

    return plain, compiled


def _build_softmax_norm(batch_size: int, hidden: int):
    """Softmax + layer-norm inspired: fused reduction-heavy kernel."""
    x = mx.random.normal((batch_size, hidden), dtype=mx.float32)
    mx.eval(x)

    def _fused(v):
        # row-wise softmax then mean/std normalisation
        sm = mx.softmax(v, axis=-1)
        mean = sm.mean(axis=-1, keepdims=True)
        std = mx.sqrt(((sm - mean) ** 2).mean(axis=-1, keepdims=True) + 1e-6)
        return (sm - mean) / std

    def plain():
        y = _fused(x)
        mx.eval(y)

    compiled_fn = mx.compile(_fused)

    def compiled():
        y = compiled_fn(x)
        mx.eval(y)

    return plain, compiled


# ── per-kernel benchmark runner ───────────────────────────────────────────────
_KERNEL_BUILDERS: Dict[str, Callable] = {
    "elementwise_tanh": _build_elementwise,
    "linear_activation": _build_linear_activation,
    "mlp_forward": _build_mlp,
    "softmax_norm": _build_softmax_norm,
}


def _build_kernel(
    kernel: str,
    batch_size: int,
    obs_dim: int,
    action_dim: int,
    hidden: int,
    n_layers: int,
) -> Tuple[Callable, Callable]:
    if kernel == "elementwise_tanh":
        return _build_elementwise(batch_size, hidden)
    elif kernel == "linear_activation":
        return _build_linear_activation(batch_size, obs_dim, hidden)
    elif kernel == "mlp_forward":
        return _build_mlp(batch_size, obs_dim, action_dim, hidden, n_layers)
    elif kernel == "softmax_norm":
        return _build_softmax_norm(batch_size, hidden)
    else:
        raise ValueError(f"Unknown kernel: {kernel}")


def run_compile_benchmark(
    kernel: str,
    batch_size: int,
    obs_dim: int,
    action_dim: int,
    hidden: int,
    n_layers: int,
    warmup: int,
    repeat: int,
) -> Optional[CompileRecord]:
    if not _HAS_MLX:
        return None
    if not hasattr(mx, "compile"):
        print("  [SKIP] this MLX version does not expose mx.compile")
        return None

    try:
        plain_fn, compiled_fn = _build_kernel(
            kernel, batch_size, obs_dim, action_dim, hidden, n_layers
        )
    except Exception as e:
        print(f"  [ERROR] build kernel {kernel}: {e}")
        return None

    # cold first-call measurements (captures compile latency for compiled path)
    fc_plain = _first_call_time(plain_fn)
    fc_compiled = _first_call_time(compiled_fn)

    # steady-state
    try:
        plain_times = _bench(plain_fn, warmup, repeat)
        compiled_times = _bench(compiled_fn, warmup, repeat)
    except Exception as e:
        print(f"  [ERROR] timing {kernel}: {e}")
        return None

    plain_mean = statistics.mean(plain_times)
    plain_std = statistics.pstdev(plain_times) if len(plain_times) > 1 else 0.0
    compiled_mean = statistics.mean(compiled_times)
    compiled_std = statistics.pstdev(compiled_times) if len(compiled_times) > 1 else 0.0
    speedup = plain_mean / (compiled_mean + 1e-12)

    return CompileRecord(
        kernel=kernel,
        batch_size=batch_size,
        first_call_plain_sec=fc_plain,
        first_call_compiled_sec=fc_compiled,
        plain_mean_sec=plain_mean,
        plain_std_sec=plain_std,
        compiled_mean_sec=compiled_mean,
        compiled_std_sec=compiled_std,
        speedup=speedup,
    )


# ── plotting ──────────────────────────────────────────────────────────────────
def plot_results(
    all_records: Dict[str, List[CompileRecord]],
    batch_sizes: List[int],
    plot_dir: Path,
) -> None:
    plot_dir.mkdir(parents=True, exist_ok=True)
    kernels = list(all_records.keys())
    n = len(kernels)
    if n == 0:
        return

    # ── panel 1: speedup per kernel across batch sizes ────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f"MLX mx.compile Speedup\n{get_device_info_line()}",
        fontsize=9,
    )

    ax_speedup = axes[0]
    ax_latency = axes[1]

    for kname, records in all_records.items():
        if not records:
            continue
        records = sorted(records, key=lambda r: r.batch_size)
        x = [r.batch_size for r in records]

        # speedup
        y_speedup = [r.speedup for r in records]
        ax_speedup.plot(x, y_speedup, marker="o", label=kname)

        # latency comparison: plain vs compiled (mean ms)
        y_plain = [r.plain_mean_sec * 1e3 for r in records]
        y_compiled = [r.compiled_mean_sec * 1e3 for r in records]
        color = ax_latency.plot(x, y_plain, marker="o", linestyle="--", label=f"{kname} plain")[
            0
        ].get_color()
        ax_latency.plot(
            x, y_compiled, marker="s", linestyle="-", color=color, label=f"{kname} compiled"
        )

    ax_speedup.axhline(1.0, color="grey", linestyle=":", linewidth=0.8, label="speedup=1×")
    ax_speedup.set_xscale("log", base=2)
    ax_speedup.set_xticks(batch_sizes)
    ax_speedup.set_xticklabels([str(b) for b in batch_sizes], rotation=30, ha="right")
    ax_speedup.set_xlabel("Batch Size")
    ax_speedup.set_ylabel("Speedup (plain / compiled)")
    ax_speedup.set_title("Steady-state Speedup")
    ax_speedup.grid(True, alpha=0.3)
    ax_speedup.legend(fontsize=7)

    ax_latency.set_xscale("log", base=2)
    ax_latency.set_xticks(batch_sizes)
    ax_latency.set_xticklabels([str(b) for b in batch_sizes], rotation=30, ha="right")
    ax_latency.set_xlabel("Batch Size")
    ax_latency.set_ylabel("Latency (ms)")
    ax_latency.set_title("Plain vs Compiled Latency")
    ax_latency.grid(True, alpha=0.3)
    ax_latency.legend(fontsize=6)

    fig.tight_layout()
    out = plot_dir / "mlx_compile_speedup.png"
    fig.savefig(out, dpi=150)
    print(f"Saved speedup plot → {out}")
    plt.close(fig)

    # ── panel 2: first-call overhead (compile latency) ────────────────────────
    fig2, ax2 = plt.subplots(figsize=(9, 4))
    ax2.set_title(
        f"MLX mx.compile First-Call Overhead\n{get_device_info_line()}",
        fontsize=9,
    )
    for kname, records in all_records.items():
        if not records:
            continue
        records = sorted(records, key=lambda r: r.batch_size)
        x = [r.batch_size for r in records]
        y = [(r.first_call_compiled_sec - r.first_call_plain_sec) * 1e3 for r in records]
        ax2.plot(x, y, marker="o", label=kname)

    ax2.axhline(0, color="grey", linestyle=":", linewidth=0.8)
    ax2.set_xscale("log", base=2)
    ax2.set_xticks(batch_sizes)
    ax2.set_xticklabels([str(b) for b in batch_sizes], rotation=30, ha="right")
    ax2.set_xlabel("Batch Size")
    ax2.set_ylabel("Extra overhead vs plain (ms)")
    ax2.set_title("First-Call Compile Overhead")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)
    fig2.tight_layout()
    out2 = plot_dir / "mlx_compile_overhead.png"
    fig2.savefig(out2, dpi=150)
    print(f"Saved overhead plot → {out2}")
    plt.close(fig2)


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Apple MLX mx.compile acceleration across kernel types."
    )
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default=",".join(str(b) for b in _DEFAULT_BATCH_SIZES),
        help="Comma-separated batch sizes to sweep (default: 128..16384).",
    )
    parser.add_argument(
        "--obs-dim", type=int, default=_DEFAULT_OBS_DIM, help="Input (observation) dimension."
    )
    parser.add_argument(
        "--action-dim", type=int, default=_DEFAULT_ACTION_DIM, help="Output (action) dimension."
    )
    parser.add_argument("--hidden", type=int, default=_DEFAULT_HIDDEN, help="Hidden layer width.")
    parser.add_argument(
        "--n-layers", type=int, default=_DEFAULT_NUM_LAYERS, help="Number of hidden layers in MLP."
    )
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations.")
    parser.add_argument("--repeat", type=int, default=30, help="Timed iterations.")
    parser.add_argument(
        "--kernels",
        type=str,
        default=",".join(_KERNEL_BUILDERS.keys()),
        help="Comma-separated list of kernels to benchmark.",
    )
    parser.add_argument("--out", type=str, default="benchmark/outputs/mlx_compile/results.json")
    parser.add_argument("--plot-dir", type=str, default="benchmark/outputs/mlx_compile")
    args = parser.parse_args()

    if not _HAS_MLX:
        print("[ERROR] mlx not available. Install it with:  pip install mlx")
        return

    batch_sizes = [int(b) for b in args.batch_sizes.split(",")]
    kernels = [k.strip() for k in args.kernels.split(",")]

    print(f"MLX version : {mx.__version__}")
    print(f"Device info : {get_device_info_line()}")
    print(f"Kernels     : {kernels}")
    print(f"Batch sizes : {batch_sizes}")
    print(f"Warmup/Repeat: {args.warmup}/{args.repeat}")
    print()

    all_records: Dict[str, List[CompileRecord]] = {k: [] for k in kernels}

    for kernel in kernels:
        print(f"=== Kernel: {kernel}")
        header = f"  {'Batch':<8} | {'Plain(ms)':<12} | {'Compiled(ms)':<14} | {'Speedup':<8} | {'Compile-overhead(ms)'}"
        print(header)
        print("  " + "-" * 72)

        for bs in batch_sizes:
            rec = run_compile_benchmark(
                kernel=kernel,
                batch_size=bs,
                obs_dim=args.obs_dim,
                action_dim=args.action_dim,
                hidden=args.hidden,
                n_layers=args.n_layers,
                warmup=args.warmup,
                repeat=args.repeat,
            )
            if rec is None:
                print(f"  {bs:<8} | SKIP")
                continue
            overhead_ms = (rec.first_call_compiled_sec - rec.first_call_plain_sec) * 1e3
            print(
                f"  {bs:<8} | "
                f"{rec.plain_mean_sec * 1e3:<12.3f} | "
                f"{rec.compiled_mean_sec * 1e3:<14.3f} | "
                f"{rec.speedup:<8.2f} | "
                f"{overhead_ms:+.1f} ms"
            )
            all_records[kernel].append(rec)
        print()

    # ── save JSON ──────────────────────────────────────────────────────────────
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "device_info": get_device_info_dict(),
            "mlx_version": mx.__version__,
            "warmup": args.warmup,
            "repeat": args.repeat,
            "obs_dim": args.obs_dim,
            "action_dim": args.action_dim,
            "hidden": args.hidden,
            "n_layers": args.n_layers,
        },
        "results": {k: [asdict(r) for r in v] for k, v in all_records.items()},
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Results saved → {out_path}")

    # ── plot ───────────────────────────────────────────────────────────────────
    if any(all_records.values()):
        plot_results(all_records, batch_sizes, Path(args.plot_dir))


if __name__ == "__main__":
    main()
