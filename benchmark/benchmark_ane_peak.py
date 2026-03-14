#!/usr/bin/env python3
"""
Stress-test ANE peak compute with a sustained heavy MLP workload (~5s by default).

Design goals:
- Dedicated ANE path only (Core ML `CPU_AND_NE` by default)
- High arithmetic intensity: large static batch + wide/deep MLP
- Sustained run: warmup + timed loop (duration-based, not fixed repeat)
- Report peak/sustained estimated TFLOPS from theoretical MLP FLOPs

Notes:
- Apple does not expose a Python API to confirm ANE runtime usage per inference.
  Use Xcode Instruments to verify ANE activity if required.
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import numpy as np

try:
    import torch
except Exception:
    torch = None

try:
    import coremltools as ct
except Exception:
    ct = None


@dataclass
class AnePeakResult:
    compute_units: str
    duration_sec: float
    warmup_sec: float
    batch_size: int
    obs_dim: int
    hidden_dim: int
    num_hidden_layers: int
    action_dim: int
    ops_per_infer: float
    total_infers: int
    total_samples: int
    elapsed_sec: float
    throughput_infer_per_sec: float
    throughput_samples_per_sec: float
    latency_mean_ms: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_min_ms: float
    latency_max_ms: float
    tflops_sustained: float
    tflops_peak_est: float


def _build_mlp_torch(
    obs_dim: int, hidden_dim: int, num_hidden_layers: int, action_dim: int, seed: int
) -> "torch.nn.Module":
    torch.manual_seed(seed)
    dims = [obs_dim] + [hidden_dim] * num_hidden_layers + [action_dim]
    layers = []
    for i in range(len(dims) - 1):
        layers.append(torch.nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(torch.nn.Tanh())
    return torch.nn.Sequential(*layers).float().eval()


def _estimate_mlp_flops(batch_size: int, dims: List[int]) -> float:
    # Count GEMM FLOPs only: for each Linear, 2 * B * in * out
    flops = 0.0
    for i in range(len(dims) - 1):
        flops += 2.0 * float(batch_size) * float(dims[i]) * float(dims[i + 1])
    return flops


def _convert_to_coreml_mlpackage(
    model: "torch.nn.Module",
    batch_size: int,
    obs_dim: int,
    output_dir: Path,
) -> Path:
    example = torch.randn(batch_size, obs_dim, dtype=torch.float32)
    traced = torch.jit.trace(model, example)
    mlpackage_path = output_dir / "ane_peak_model.mlpackage"
    mlmodel = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=[ct.TensorType(name="x", shape=(batch_size, obs_dim))],
    )
    mlmodel.save(str(mlpackage_path))
    return mlpackage_path


def _load_coreml_model(mlpackage_path: Path, compute_units: str) -> "ct.models.MLModel":
    cu = getattr(ct.ComputeUnit, compute_units)
    return ct.models.MLModel(str(mlpackage_path), compute_units=cu)


def _run_sustained(
    model: "ct.models.MLModel",
    x: np.ndarray,
    warmup_sec: float,
    duration_sec: float,
) -> List[float]:
    input_name = list(model.get_spec().description.input)[0].name

    def fwd() -> None:
        _ = model.predict({input_name: x})

    # Warmup by time to stabilize runtime caches before measuring.
    t_warm_end = time.perf_counter() + max(0.0, warmup_sec)
    while time.perf_counter() < t_warm_end:
        fwd()

    latencies: List[float] = []
    t_end = time.perf_counter() + max(0.0, duration_sec)
    while time.perf_counter() < t_end:
        t0 = time.perf_counter()
        fwd()
        latencies.append(time.perf_counter() - t0)
    return latencies


def _compute_result(
    latencies: List[float],
    compute_units: str,
    duration_sec: float,
    warmup_sec: float,
    batch_size: int,
    obs_dim: int,
    hidden_dim: int,
    num_hidden_layers: int,
    action_dim: int,
) -> AnePeakResult:
    if not latencies:
        raise RuntimeError("No timed inference samples collected.")

    dims = [obs_dim] + [hidden_dim] * num_hidden_layers + [action_dim]
    ops_per_infer = _estimate_mlp_flops(batch_size, dims)

    elapsed_sec = float(sum(latencies))
    total_infers = len(latencies)
    total_samples = total_infers * batch_size
    throughput_infer_per_sec = total_infers / elapsed_sec
    throughput_samples_per_sec = total_samples / elapsed_sec

    latency_ms = [t * 1000.0 for t in latencies]
    latency_mean_ms = statistics.mean(latency_ms)
    latency_p50_ms = float(np.percentile(latency_ms, 50))
    latency_p95_ms = float(np.percentile(latency_ms, 95))
    latency_min_ms = min(latency_ms)
    latency_max_ms = max(latency_ms)

    total_ops = ops_per_infer * float(total_infers)
    tflops_sustained = total_ops / elapsed_sec / 1e12
    tflops_peak_est = ops_per_infer / min(latencies) / 1e12

    return AnePeakResult(
        compute_units=compute_units,
        duration_sec=duration_sec,
        warmup_sec=warmup_sec,
        batch_size=batch_size,
        obs_dim=obs_dim,
        hidden_dim=hidden_dim,
        num_hidden_layers=num_hidden_layers,
        action_dim=action_dim,
        ops_per_infer=ops_per_infer,
        total_infers=total_infers,
        total_samples=total_samples,
        elapsed_sec=elapsed_sec,
        throughput_infer_per_sec=throughput_infer_per_sec,
        throughput_samples_per_sec=throughput_samples_per_sec,
        latency_mean_ms=latency_mean_ms,
        latency_p50_ms=latency_p50_ms,
        latency_p95_ms=latency_p95_ms,
        latency_min_ms=latency_min_ms,
        latency_max_ms=latency_max_ms,
        tflops_sustained=tflops_sustained,
        tflops_peak_est=tflops_peak_est,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dedicated ANE peak-compute stress benchmark (~5s sustained)"
    )
    parser.add_argument(
        "--duration-sec", type=float, default=5.0, help="Timed run duration in seconds"
    )
    parser.add_argument("--warmup-sec", type=float, default=1.0, help="Warmup duration in seconds")
    parser.add_argument("--batch-size", type=int, default=1024, help="Heavy static batch size")
    parser.add_argument("--obs-dim", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=2048)
    parser.add_argument("--num-hidden-layers", type=int, default=6)
    parser.add_argument("--action-dim", type=int, default=1024)
    parser.add_argument(
        "--compute-units",
        type=str,
        default="CPU_AND_NE",
        choices=["CPU_ONLY", "CPU_AND_GPU", "CPU_AND_NE", "ALL"],
        help="Core ML compute units. For ANE stress use CPU_AND_NE",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out",
        type=str,
        default="benchmark/outputs/ane_peak/ane_peak_result.json",
        help="Output JSON path",
    )
    args = parser.parse_args()

    if torch is None:
        raise RuntimeError("PyTorch is required (failed to import torch).")
    if ct is None:
        raise RuntimeError("coremltools is required (failed to import coremltools).")

    if (
        args.batch_size <= 0
        or args.obs_dim <= 0
        or args.hidden_dim <= 0
        or args.num_hidden_layers <= 0
        or args.action_dim <= 0
    ):
        raise ValueError("All dimension and layer arguments must be positive.")

    print("[ANE Peak] Building heavy MLP...", flush=True)
    model = _build_mlp_torch(
        obs_dim=args.obs_dim,
        hidden_dim=args.hidden_dim,
        num_hidden_layers=args.num_hidden_layers,
        action_dim=args.action_dim,
        seed=args.seed,
    )

    tmp_dir = Path(tempfile.mkdtemp(prefix="ane_peak_"))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        print("[ANE Peak] Converting to Core ML (mlprogram, static shape)...", flush=True)
        mlpackage_path = _convert_to_coreml_mlpackage(
            model=model,
            batch_size=args.batch_size,
            obs_dim=args.obs_dim,
            output_dir=tmp_dir,
        )

        print(f"[ANE Peak] Loading model with compute_units={args.compute_units}...", flush=True)
        cm_model = _load_coreml_model(mlpackage_path, args.compute_units)

        rng = np.random.default_rng(args.seed + 1)
        x = np.clip(
            rng.standard_normal((args.batch_size, args.obs_dim)).astype(np.float32),
            -3.0,
            3.0,
        )

        print(
            f"[ANE Peak] Sustained run: warmup={args.warmup_sec:.2f}s, timed={args.duration_sec:.2f}s, "
            f"batch={args.batch_size}, hidden={args.hidden_dim}x{args.num_hidden_layers}",
            flush=True,
        )
        print(
            "[ANE Peak] Note: Python cannot directly confirm ANE per-inference usage; use Instruments if needed.",
            flush=True,
        )

        latencies = _run_sustained(
            model=cm_model,
            x=x,
            warmup_sec=args.warmup_sec,
            duration_sec=args.duration_sec,
        )

        result = _compute_result(
            latencies=latencies,
            compute_units=args.compute_units,
            duration_sec=args.duration_sec,
            warmup_sec=args.warmup_sec,
            batch_size=args.batch_size,
            obs_dim=args.obs_dim,
            hidden_dim=args.hidden_dim,
            num_hidden_layers=args.num_hidden_layers,
            action_dim=args.action_dim,
        )

        payload = {
            "meta": {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "host": "macOS",
                "note": "TFLOPS are estimated from theoretical GEMM FLOPs of the MLP.",
            },
            "result": asdict(result),
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        print("\n[ANE Peak] Done")
        print(f"  infers: {result.total_infers}")
        print(f"  elapsed: {result.elapsed_sec:.3f} s")
        print(f"  infer/s: {result.throughput_infer_per_sec:.2f}")
        print(f"  samples/s: {result.throughput_samples_per_sec:.2f}")
        print(
            f"  latency mean/p50/p95: {result.latency_mean_ms:.3f} / {result.latency_p50_ms:.3f} / {result.latency_p95_ms:.3f} ms"
        )
        print(f"  TFLOPS sustained: {result.tflops_sustained:.3f}")
        print(f"  TFLOPS peak(est): {result.tflops_peak_est:.3f}")
        print(f"  saved: {out_path.resolve()}")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
