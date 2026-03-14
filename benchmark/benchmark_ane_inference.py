#!/usr/bin/env python3
"""
Benchmark MLP inference overhead for ANE (Apple Neural Engine).

env_num from 2^8 to 2^14. Model size aligned with locomotion policy (obs_dim=48,
hidden=[256,256,256], action_dim=12). ANE uses EnumeratedShapes and Core ML CPU_AND_NE.
Per config: 5 runs, drop min and max, report mean (and std) of the middle 3.
Outputs JSON and plots.
"""

from __future__ import annotations

try:
    from benchmark.core import (
        MLPBenchRecord,
        bench_callable,
        mlp_param_count,
        print_mlp_table,
        trimmed_mean,
    )
    from benchmark.core.device_info import get_device_info_dict, get_device_info_line
except ModuleNotFoundError:
    from core import MLPBenchRecord, bench_callable, mlp_param_count, print_mlp_table, trimmed_mean
    from core.device_info import get_device_info_dict, get_device_info_line


import argparse
import json
import shutil
import tempfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import coremltools as ct
import numpy as np
import torch

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None

# Model config aligned with locomotion tasks (Go1/Go2/G1 style policy)
DEFAULT_OBS_DIM = 48
DEFAULT_ACTION_DIM = 12
DEFAULT_HIDDEN_DIMS = [256, 256, 256]
DEFAULT_ENV_NUM_POW_MIN = 8
DEFAULT_ENV_NUM_POW_MAX = 15

_OBS_CLIP_LOW, _OBS_CLIP_HIGH = -3.0, 3.0
_MIN_MEAN_SEC = 1e-12


def _safe_envs_per_sec(env_num: int, mean_sec: float) -> float:
    if mean_sec is None or mean_sec <= 0 or env_num <= 0:
        return 0.0
    mean_sec = max(mean_sec, _MIN_MEAN_SEC)
    return env_num / mean_sec


# 测 5 次，去掉最小、最大各 1 个，对中间 3 个取平均
REPEAT_COUNT = 5


# ---------- Core ML ANE (Apple Neural Engine, no CPU fallback) ----------
def _build_and_convert_coreml_ane(
    obs_dim: int,
    action_dim: int,
    hidden_dims: List[int],
    env_nums: List[int],
    mlmodel_path: str,
    seed: int = 42,
    precision: ct.precision = ct.precision.FLOAT32,
) -> bool:
    """Convert to Core ML with EnumeratedShapes so model can run on ANE (no flexible batch)."""
    dims = [obs_dim] + hidden_dims + [action_dim]
    layers = []
    torch.manual_seed(seed)
    for i in range(len(dims) - 1):
        layers.append(torch.nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(torch.nn.Tanh())
    model = torch.nn.Sequential(*layers).float().eval()
    example_input = torch.randn(env_nums[0], obs_dim)
    try:
        traced = torch.jit.trace(model, example_input)
        shapes = [[n, obs_dim] for n in env_nums]
        default_shape = [env_nums[-1], obs_dim]
        input_shape = ct.EnumeratedShapes(shapes=shapes, default=default_shape)

        # For FP8, we need to use an optimization pass after conversion
        compute_precision = (
            ct.precision.FLOAT16 if precision == ct.precision.FLOAT16 else ct.precision.FLOAT32
        )

        mlmodel = ct.convert(
            traced,
            convert_to="mlprogram",
            inputs=[ct.TensorType(name="x", shape=input_shape)],
            compute_precision=compute_precision,
        )

        if precision == "FLOAT8":
            import coremltools.optimize.coreml as cto

            op_config = cto.OptimizationConfig(
                global_config=cto.OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8")
            )
            mlmodel = cto.linear_quantize_weights(mlmodel, config=op_config)

        mlmodel.save(mlmodel_path)
        return True
    except Exception as e:
        print(f"Error converting to Core ML: {e}")
        return False


def _log_ane_phase() -> None:
    print(
        "  [ANE] Core ML loaded with compute_units=CPU_AND_NE (no runtime ANE-usage check; see doc).",
        flush=True,
    )


def run_ane(
    env_num: int,
    obs_dim: int,
    model: "ct.models.MLModel",
    warmup: int,
    repeat: int,
) -> Optional[MLPBenchRecord]:
    if np is None or model is None:
        return None
    rng = np.random.default_rng(43)
    x = np.clip(
        rng.standard_normal((env_num, obs_dim)).astype(np.float32),
        _OBS_CLIP_LOW,
        _OBS_CLIP_HIGH,
    )
    input_name = list(model.get_spec().description.input)[0].name

    def fwd():
        _ = model.predict({input_name: x})

    elapsed = bench_callable(fwd, lambda: None, warmup, repeat)
    mean_sec, std_sec, _ = trimmed_mean(elapsed)
    return MLPBenchRecord(
        backend="ane",
        env_num=env_num,
        warmup=warmup,
        repeat=repeat,
        elapsed_sec=elapsed,
        mean_sec=mean_sec,
        std_sec=std_sec,
        min_sec=min(elapsed),
        max_sec=max(elapsed),
        envs_per_sec=_safe_envs_per_sec(env_num, mean_sec),
    )


def save_plots(
    records: List[MLPBenchRecord],
    plot_dir: Path,
    file_prefix: str,
) -> List[str]:
    if plt is None or not records:
        return []
    plot_dir.mkdir(parents=True, exist_ok=True)
    saved: List[str] = []

    backends = sorted({r.backend for r in records})
    env_nums = sorted({r.env_num for r in records})

    _min_time_ms = 1e-9
    _min_throughput = 1e-6

    # 1) Mean time (ms) vs env_num
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    for backend in backends:
        subset = sorted(
            [r for r in records if r.backend == backend],
            key=lambda x: x.env_num,
        )
        if not subset:
            continue
        x = [r.env_num for r in subset]
        y = [max(_min_time_ms, r.mean_sec * 1000.0) for r in subset]
        ax1.plot(x, y, marker="o", label=backend)
    ax1.set_title(f"MLP inference time (ms) vs env_num\n{get_device_info_line()}")
    ax1.set_xlabel("env_num")
    ax1.set_ylabel("Mean time (ms)")
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(env_nums)
    ax1.set_xticklabels([str(v) for v in env_nums])
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    fig1.tight_layout()
    out1 = plot_dir / f"{file_prefix}_time_ms.png"
    fig1.savefig(out1, dpi=150)
    plt.close(fig1)
    saved.append(str(out1.resolve()))

    # 2) Throughput (envs/s) vs env_num
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    for backend in backends:
        subset = sorted(
            [r for r in records if r.backend == backend],
            key=lambda x: x.env_num,
        )
        if not subset:
            continue
        x = [r.env_num for r in subset]
        y = [max(_min_throughput, r.envs_per_sec) for r in subset]
        ax2.plot(x, y, marker="o", label=backend)
    ax2.set_title(f"MLP inference throughput (envs/s) vs env_num\n{get_device_info_line()}")
    ax2.set_xlabel("env_num")
    ax2.set_ylabel("Envs per second")
    ax2.set_xscale("log", base=2)
    ax2.set_xticks(env_nums)
    ax2.set_xticklabels([str(v) for v in env_nums])
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    fig2.tight_layout()
    out2 = plot_dir / f"{file_prefix}_throughput.png"
    fig2.savefig(out2, dpi=150)
    plt.close(fig2)
    saved.append(str(out2.resolve()))

    return saved


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark MLP inference for ANE (Apple Neural Engine)."
    )
    parser.add_argument(
        "--sizes",
        type=str,
        default=",".join(
            str(2**k) for k in range(DEFAULT_ENV_NUM_POW_MIN, DEFAULT_ENV_NUM_POW_MAX + 1)
        ),
        help="Comma-separated env_num sizes (default: 256,512,1024,...)",
    )
    parser.add_argument("--obs-dim", type=int, default=DEFAULT_OBS_DIM)
    parser.add_argument("--action-dim", type=int, default=DEFAULT_ACTION_DIM)
    parser.add_argument(
        "--hidden-dims",
        type=str,
        default=",".join(map(str, DEFAULT_HIDDEN_DIMS)),
        help="Comma-separated hidden layer sizes",
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument(
        "--repeat",
        type=int,
        default=REPEAT_COUNT,
        help=f"Number of samples per config; we drop min/max and average the rest (default: {REPEAT_COUNT})",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="benchmark/outputs/mlp_inference/benchmark_ane_results.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--plot-dir",
        type=str,
        default="",
        help="Plot directory (default: same as --out parent)",
    )
    args = parser.parse_args()

    env_nums = [int(x.strip()) for x in args.sizes.split(",") if x.strip()]
    hidden_dims = [int(x.strip()) for x in args.hidden_dims.split(",") if x.strip()]
    if not hidden_dims:
        hidden_dims = list(DEFAULT_HIDDEN_DIMS)

    print(f"env_nums: {env_nums}", flush=True)
    print(
        f"model: obs_dim={args.obs_dim}, hidden_dims={hidden_dims}, action_dim={args.action_dim}",
        flush=True,
    )

    all_records: List[MLPBenchRecord] = []
    skipped: List[Dict[str, str]] = []

    precisions = {
        "fp32": ct.precision.FLOAT32,
        "fp16": ct.precision.FLOAT16,
        "int8": "FLOAT8",  # CoreML uses int8 for weight quantization
    }

    for prec_name, prec_val in precisions.items():
        ane_path = tempfile.mkdtemp(suffix=f"_{prec_name}.mlpackage")
        ane_model = None
        cpu_model = None

        print(f"\nBuilding and converting model for ANE ({prec_name})...", flush=True)
        if _build_and_convert_coreml_ane(
            args.obs_dim, args.action_dim, hidden_dims, env_nums, ane_path, precision=prec_val
        ):
            try:
                ane_model = ct.models.MLModel(ane_path, compute_units=ct.ComputeUnit.CPU_AND_NE)
                cpu_model = ct.models.MLModel(ane_path, compute_units=ct.ComputeUnit.CPU_ONLY)
            except Exception as e:
                print(f"Failed to load Core ML model ({prec_name}): {e}")
        else:
            print(f"Failed to convert model to Core ML ({prec_name}).")

        if cpu_model is not None:
            backend_name = f"cpu_only_{prec_name}"
            print(f"\nDetected backends ({backend_name} Baseline):", flush=True)
            print(f"  - {backend_name}: yes", flush=True)

            for env_num in env_nums:
                print(f"\nRunning env_num={env_num} ...", flush=True)
                print(f"  >> {backend_name} env_num={env_num} starting...", flush=True)
                try:
                    rec = run_ane(env_num, args.obs_dim, cpu_model, args.warmup, args.repeat)
                    if rec is not None:
                        rec.backend = backend_name
                        all_records.append(rec)
                        print(
                            f"  {backend_name}: mean={rec.mean_sec * 1000:.3f} ms, envs/s={rec.envs_per_sec:.1f}",
                            flush=True,
                        )
                    else:
                        skipped.append(
                            {
                                "backend": backend_name,
                                "env_num": str(env_num),
                                "reason": "unavailable",
                            }
                        )
                except Exception as e:
                    skipped.append(
                        {"backend": backend_name, "env_num": str(env_num), "reason": str(e)}
                    )
                    print(f"  {backend_name}: skip - {e}", flush=True)

        if ane_model is not None:
            backend_name = f"ane_{prec_name}"
            print(f"\nDetected backends ({backend_name}):", flush=True)
            print(f"  - {backend_name}: yes", flush=True)
            _log_ane_phase()

            for env_num in env_nums:
                print(f"\nRunning env_num={env_num} ...", flush=True)
                print(f"  >> {backend_name} env_num={env_num} starting...", flush=True)
                try:
                    rec = run_ane(env_num, args.obs_dim, ane_model, args.warmup, args.repeat)
                    if rec is not None:
                        rec.backend = backend_name
                        all_records.append(rec)
                        print(
                            f"  {backend_name}: mean={rec.mean_sec * 1000:.3f} ms, envs/s={rec.envs_per_sec:.1f}",
                            flush=True,
                        )
                    else:
                        skipped.append(
                            {
                                "backend": backend_name,
                                "env_num": str(env_num),
                                "reason": "unavailable",
                            }
                        )
                except Exception as e:
                    skipped.append(
                        {"backend": backend_name, "env_num": str(env_num), "reason": str(e)}
                    )
                    print(f"  {backend_name}: skip - {e}", flush=True)
        else:
            print(f"\nANE model ({prec_name}) could not be loaded. Skipping benchmark.")

        try:
            shutil.rmtree(ane_path, ignore_errors=True)
        except Exception:
            pass

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plot_dir = Path(args.plot_dir) if args.plot_dir else out_path.resolve().parent
    plot_files = save_plots(all_records, plot_dir=plot_dir, file_prefix=out_path.stem)

    n_params = mlp_param_count(args.obs_dim, args.action_dim, hidden_dims)
    payload = {
        "meta": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "device_info": get_device_info_dict(),
            "env_nums": env_nums,
            "obs_dim": args.obs_dim,
            "action_dim": args.action_dim,
            "hidden_dims": hidden_dims,
            "approx_params": n_params,
            "warmup": args.warmup,
            "repeat": args.repeat,
            "available_backends": {"ane": ane_model is not None},
            "skip_backends": [],
            "plot_files": plot_files,
            "skipped": skipped,
        },
        "results": [asdict(r) for r in all_records],
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved results to: {out_path.resolve()}")
    print()
    print_mlp_table(all_records)
    if plot_files:
        print("\n生成图片路径:")
        for f in plot_files:
            print(f"  {f}")


if __name__ == "__main__":
    main()
