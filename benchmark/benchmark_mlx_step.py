#!/usr/bin/env python3
"""
Benchmark parallel physics step speed using mujoco.mlx_step (MlxStepRunner).
Sweeps batch sizes 2^8..2^14 across go1/go2/g1 and plots Total FPS (万).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from multiprocessing import cpu_count
from pathlib import Path
from typing import Dict, List

import mujoco
from mujoco import mlx_step
import mlx.core as mx

try:
    from benchmark.device_info import get_device_info_dict, get_device_info_line
except ModuleNotFoundError:
    from device_info import get_device_info_dict, get_device_info_line

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_LOCO = os.path.join(os.path.dirname(__file__), "../unilab/envs/locomotion")
_DEFAULT_MODELS = {
    "go1": os.path.join(_LOCO, "go1/xml/scene_flat.xml"),
    "go2": os.path.join(_LOCO, "go2/xml/scene_flat.xml"),
    "g1":  os.path.join(_LOCO, "g1/xml/scene_flat.xml"),
}
_DEFAULT_BATCH_SIZES = [2**i for i in range(9, 15)]  # 256..16384


@dataclass
class MlxStepRecord:
    model_name: str
    batch_size: int
    nsubsteps: int
    nthread: int
    chunk_size: int
    steps: int
    elapsed_sec: float
    sps: float


def run_mlx_step(
    model_name: str,
    model: mujoco.MjModel,
    batch_size: int,
    steps: int,
    nsubsteps: int,
    chunk_size: int,
    warmup: int = 5,
) -> MlxStepRecord:
    nstate = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    nthread = min(batch_size, cpu_count())
    worker_data = [mujoco.MjData(model) for _ in range(nthread)]
    runner = mlx_step.MlxStepRunner(nthread=nthread)

    initial_state = mx.zeros((batch_size, nstate), dtype=mx.float32)
    ctrl = mx.zeros((batch_size, nsubsteps, model.nu), dtype=mx.float32)

    for _ in range(warmup):
        state_out, _ = runner.step(
            model=model, data=worker_data, initial_state=initial_state,
            control=ctrl, nstep=nsubsteps, chunk_size=chunk_size,
            out_dtype=mx.float32, return_last_only=True,
        )
    mx.eval(state_out)

    start = time.perf_counter()
    for _ in range(steps):
        state_out, _ = runner.step(
            model=model, data=worker_data, initial_state=initial_state,
            control=ctrl, nstep=nsubsteps, chunk_size=chunk_size,
            out_dtype=mx.float32, return_last_only=True,
        )
        initial_state = state_out
    mx.eval(initial_state)
    elapsed = max(time.perf_counter() - start, 1e-9)

    return MlxStepRecord(
        model_name=model_name,
        batch_size=batch_size,
        nsubsteps=nsubsteps,
        nthread=nthread,
        chunk_size=chunk_size,
        steps=steps,
        elapsed_sec=elapsed,
        sps=batch_size * steps / elapsed,
    )


def plot_results(all_records: Dict[str, List[MlxStepRecord]], batch_sizes: List[int], plot_dir: Path):
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_title(f"MLX Step Parallel Physics — Total FPS\n{get_device_info_line()}", fontsize=9)

    for model_name, records in all_records.items():
        records = sorted(records, key=lambda r: r.batch_size)
        x = [r.batch_size for r in records]
        y = [r.sps / 1e4 for r in records]
        ax.plot(x, y, marker="o", label=model_name)

    ax.set_xscale("log", base=2)
    ax.set_xticks(batch_sizes)
    ax.set_xticklabels([str(b) for b in batch_sizes], rotation=30, ha="right")
    ax.set_xlabel("Batch Size (Num Envs)")
    ax.set_ylabel("Total FPS (x1e4)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    out = plot_dir / "mlx_step_benchmark.png"
    fig.savefig(out, dpi=150)
    print(f"Saved plot to {out}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Benchmark mujoco mlx_step parallel physics")
    parser.add_argument("--batch-sizes", type=str,
                        default=",".join(str(b) for b in _DEFAULT_BATCH_SIZES))
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--nsubsteps", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--out", type=str, default="benchmark/outputs/mlx_step/results.json")
    parser.add_argument("--plot-dir", type=str, default="benchmark/outputs/mlx_step")
    args = parser.parse_args()

    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    all_records: Dict[str, List[MlxStepRecord]] = {}

    for model_name, xml_path in _DEFAULT_MODELS.items():
        xml_path = str(Path(xml_path).resolve())
        print(f"\n=== {model_name}: {xml_path}")
        try:
            model = mujoco.MjModel.from_xml_path(xml_path)
        except Exception as e:
            print(f"  Failed to load: {e}")
            continue
        print(f"  nq={model.nq}, nv={model.nv}, nu={model.nu}")
        print(f"  {'Batch':<8} | {'FPS(万)':<12} | {'Time(s)':<8}")
        print("  " + "-" * 36)

        records = []
        for bs in batch_sizes:
            try:
                r = run_mlx_step(model_name, model, bs, args.steps,
                                 args.nsubsteps, args.chunk_size)
                print(f"  {bs:<8} | {r.sps/1e4:<12.2f} | {r.elapsed_sec:<8.4f}")
                records.append(r)
            except Exception as e:
                print(f"  {bs:<8} | ERROR: {e}")
        all_records[model_name] = records

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(
            {
                "meta": {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "device_info": get_device_info_dict(),
                    "steps": args.steps,
                    "nsubsteps": args.nsubsteps,
                },
                "results": {k: [asdict(r) for r in v] for k, v in all_records.items()},
            },
            f, indent=2,
        )
    print(f"\nResults saved to {out_path}")

    if any(all_records.values()):
        plot_results(all_records, batch_sizes, Path(args.plot_dir))


if __name__ == "__main__":
    main()
