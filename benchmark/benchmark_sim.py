#!/usr/bin/env python3
"""
Benchmark MuJoCo simulation speed:
- MuJoCo CPU Naive (Single Thread Loop)
- MuJoCo Rollout Single-Core (nstep=1 loop)
- MuJoCo Rollout Multi-Core (nstep=1 loop)
- MJX (GPU/Metal via JAX)
- MotrixSim (batch physics backend)

Supports loading standard XML models or custom paths.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
try:
    from benchmark.device_info import get_device_info_dict, get_device_info_line
except ModuleNotFoundError:
    from device_info import get_device_info_dict, get_device_info_line

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

try:
    import mujoco
except ImportError:
    mujoco = None

try:
    import mujoco.rollout
except ImportError:
    pass # handled in checks

try:
    import jax
    from jax import numpy as jp
    from mujoco import mjx
except ImportError:
    jax = None
    mjx = None

try:
    import motrixsim as mtx
except ImportError:
    mtx = None


@dataclass
class SimRecord:
    backend: str
    model_name: str
    batch_size: int
    steps: int
    elapsed_sec: float
    sps: float # steps per second (total across batch)
    sps_per_env: float

def get_model(xml_path: str) -> "mujoco.MjModel":
    path = Path(xml_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {xml_path}")
    
    return mujoco.MjModel.from_xml_path(str(path))

def run_mujoco_cpu_naive(model, batch_size: int, steps: int, warmup: int = 5) -> SimRecord:
    """Naive loop: for each step, loop over batch. Single thread."""
    data = mujoco.MjData(model)
    
    # Warmup
    for _ in range(warmup):
        mujoco.mj_step(model, data)
        
    start = time.perf_counter()
    
    # Run loop
    # For large batch, we reduce steps to keep runtime reasonable for benchmark
    actual_batch = batch_size
    actual_steps = steps
    
    # Safety: limit total ops
    if batch_size * steps > 100000:
        actual_steps = max(1, 100000 // batch_size)
    
    # Run
    for _ in range(actual_steps):
        for _ in range(actual_batch):
            mujoco.mj_step(model, data)
            
    end = time.perf_counter()
    elapsed = end - start
    
    # Scale time back if we reduced steps
    if actual_steps < steps:
        if elapsed > 0:
            elapsed = elapsed * (steps / actual_steps)
        else:
            elapsed = 1e-6
    
    total_steps = batch_size * steps
    sps = total_steps / elapsed if elapsed > 0 else 0
    
    return SimRecord(
        backend="mujoco_cpu_naive",
        model_name="xml",
        batch_size=batch_size,
        steps=steps,
        elapsed_sec=elapsed,
        sps=sps,
        sps_per_env=sps / batch_size
    )

def run_mujoco_rollout_runner(
    model,
    batch_size: int,
    steps: int,
    nthread: int,
    backend_name: str,
    skip_checks: bool = False,
) -> SimRecord:
    """Run rollout using persistent Rollout runner (class)."""
    if not hasattr(mujoco, "rollout"):
        return SimRecord(backend_name, "error_no_class", batch_size, steps, 0, 0, 0)

    # We need one MjData per thread.
    worker_data = [mujoco.MjData(model) for _ in range(nthread)]
    
    nstate = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    nu = model.nu
    
    # Initial state (batch)
    initial_state = np.zeros((batch_size, nstate))
    # Control (batch, nstep=1, nu)
    ctrl = np.zeros((batch_size, 1, nu))
    
    model_batch = [model] * batch_size if skip_checks else None
    state_buf = np.empty((batch_size, 1, nstate), dtype=np.float64) if skip_checks else None
    sensor_buf = np.empty((batch_size, 1, model.nsensordata), dtype=np.float64) if skip_checks else None

    # Warmup
    try:
        if skip_checks:
            _ = mujoco.rollout.rollout(
                model_batch,
                worker_data,
                initial_state=initial_state,
                control=ctrl,
                nstep=1,
                skip_checks=True,
                persistent_pool=True,
                state=state_buf,
                sensordata=sensor_buf,
            )
        else:
            runner = mujoco.rollout.Rollout(nthread=nthread)
            runner.rollout(
                model,
                worker_data,
                initial_state=initial_state,
                control=ctrl,
                nstep=1,
            )
    except Exception as e:
        print(f"Rollout warmup failed ({backend_name}): {e}")
        return SimRecord(backend_name, "failed", batch_size, steps, 0, 0, 0)

    start = time.perf_counter()
    # Step-by-step loop
    for _ in range(steps):
        # In a real env, we would update ctrl here based on obs
        if skip_checks:
            state_traj, sensor_traj = mujoco.rollout.rollout(
                model_batch,
                worker_data,
                initial_state=initial_state,
                control=ctrl,
                nstep=1,
                skip_checks=True,
                persistent_pool=True,
                state=state_buf,
                sensordata=sensor_buf,
            )
        else:
            state_traj, sensor_traj = runner.rollout(
                model,
                worker_data,
                initial_state=initial_state,
                control=ctrl,
                nstep=1,
            )
        # In a real env, we would update initial_state from state_traj[:, -1, :]
        # Here we just loop to measure throughput
        # Updating initial_state might add overhead, let's include it for realism
        if state_traj is not None:
             initial_state[:] = state_traj[:, -1, :]

    end = time.perf_counter()
    
    elapsed = end - start
    if elapsed < 1e-6: elapsed = 1e-6
    total_steps = batch_size * steps
    sps = total_steps / elapsed if elapsed > 0 else 0
    
    return SimRecord(
        backend=backend_name,
        model_name="xml",
        batch_size=batch_size,
        steps=steps,
        elapsed_sec=elapsed,
        sps=sps,
        sps_per_env=sps / batch_size
    )


def parse_skip_checks_mode(mode: str) -> List[bool]:
    m = mode.strip().lower()
    if m == "both":
        return [False, True]
    if m == "true":
        return [True]
    if m == "false":
        return [False]
    raise ValueError(f"Invalid --rollout-multi-skip-checks mode: {mode}")

def run_mjx(model, batch_size: int, steps: int, warmup: int = 5, device=None) -> SimRecord:
    """Run MJX via JAX."""
    if mjx is None or jax is None:
        return SimRecord("mjx", "missing_dependency", batch_size, steps, 0, 0, 0)

    # Determine backend name
    backend_name = "mjx"
    if device:
        backend_name += f"_{device.platform}"
    else:
        # Check default device
        try:
            default_dev = jax.devices()[0]
            backend_name += f"_{default_dev.platform}"
        except:
            pass

    try:
        # IMPORTANT: mjx.put_model and make_data might default to default backend (Metal)
        # We must wrap them in jax.default_device context if a specific device is requested
        if device:
            mx_model = mjx.put_model(model, device=device)
            mx_data = mjx.make_data(model, device=device)
        else:
            mx_model = mjx.put_model(model)
            mx_data = mjx.make_data(model)
        
        # Batch data: replicate structure
        batch_mx_data = jax.tree_util.tree_map(lambda x: jp.stack([x] * batch_size), mx_data)
        
        # Ensure data is on the correct device
        if device:
            batch_mx_data = jax.device_put(batch_mx_data, device)
        
        def step_fn(m, d):
            return mjx.step(m, d)
            
        # vmap over data (arg 1)
        # Force the vmap to run on the specific device if provided
        # Though jax.jit usually handles placement, explicit jit is safer
        batch_step_fn = jax.vmap(step_fn, in_axes=(None, 0))
        
        # Compile scan loop
        def loop_fn(d, _):
            return batch_step_fn(mx_model, d), None
            
        # JIT compilation
        if device:
            # We use jax.jit with explicit device
            @jax.jit(device=device)
            def run_loop(d):
                d, _ = jax.lax.scan(loop_fn, d, None, length=steps)
                return d
            
            @jax.jit(device=device)
            def warmup_run(d):
                d, _ = jax.lax.scan(lambda dd, _: (batch_step_fn(mx_model, dd), None), d, None, length=10)
                return d
        else:
            @jax.jit
            def run_loop(d):
                d, _ = jax.lax.scan(loop_fn, d, None, length=steps)
                return d
                
            @jax.jit
            def warmup_run(d):
                d, _ = jax.lax.scan(lambda dd, _: (batch_step_fn(mx_model, dd), None), d, None, length=10)
                return d

        # Warmup
        # Ensure inputs are on device before calling jit function
        if device:
            batch_mx_data = jax.device_put(batch_mx_data, device)

        res = warmup_run(batch_mx_data)
        jax.block_until_ready(res.qpos) 
        
        # Run Benchmark
        start = time.perf_counter()
        final_data = run_loop(batch_mx_data)
        jax.block_until_ready(final_data.qpos)
        end = time.perf_counter()
        
        elapsed = end - start
        if elapsed < 1e-6: elapsed = 1e-6
        
        total_steps = batch_size * steps
        sps = total_steps / elapsed if elapsed > 0 else 0
        
        return SimRecord(
            backend=backend_name,
            model_name="xml",
            batch_size=batch_size,
            steps=steps,
            elapsed_sec=elapsed,
            sps=sps,
            sps_per_env=sps / batch_size
        )
    except Exception as e:
        # Re-raise to let caller handle or fallback
        raise e


def run_motrixsim(xml_path: str, batch_size: int, steps: int, warmup: int = 20) -> SimRecord:
    if mtx is None:
        return SimRecord("motrixsim", "missing_dependency", batch_size, steps, 0, 0, 0)

    try:
        model = mtx.load_model(xml_path)
        data = mtx.SceneData(model, batch=(batch_size,))

        for _ in range(warmup):
            model.step(data)

        start = time.perf_counter()
        for _ in range(steps):
            model.step(data)
        end = time.perf_counter()

        elapsed = max(end - start, 1e-6)
        total_steps = batch_size * steps
        sps = total_steps / elapsed
        return SimRecord(
            backend="motrixsim",
            model_name="xml",
            batch_size=batch_size,
            steps=steps,
            elapsed_sec=elapsed,
            sps=sps,
            sps_per_env=sps / batch_size if batch_size > 0 else 0.0,
        )
    except Exception as e:
        print(f"motrixsim error bs={batch_size}: {e}")
        return SimRecord("motrixsim_failed", "failed", batch_size, steps, 0, 0, 0)

def plot_results(records: List[SimRecord], plot_dir: Path):
    if not records:
        return
    
    plot_dir.mkdir(parents=True, exist_ok=True)
    backends = sorted({r.backend for r in records})
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for backend in backends:
        subset = sorted([r for r in records if r.backend == backend], key=lambda x: x.batch_size)
        if not subset:
            continue
        
        x = [r.batch_size for r in subset]
        y = [r.sps for r in subset]
        
        ax.plot(x, y, marker='o', label=backend)
        
    ax.set_title(
        "MuJoCo Simulation Speed (Steps Per Second) vs Batch Size\n"
        f"{get_device_info_line()}"
    )
    all_batch_sizes = sorted({r.batch_size for r in records})
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Total Steps Per Second (log scale)")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(all_batch_sizes)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: str(int(v))))
    ax.tick_params(axis="x", rotation=45)
    # Show denser log ticks (1/2/5 * 10^n) so the y-axis is easier to read.
    ax.yaxis.set_major_locator(mticker.LogLocator(base=10.0, subs=(1.0, 2.0, 5.0)))
    ax.yaxis.set_major_formatter(mticker.LogFormatterSciNotation(base=10, labelOnlyBase=False))
    ax.yaxis.set_minor_locator(
        mticker.LogLocator(base=10.0, subs=np.arange(1, 10) * 0.1)
    )
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax.grid(True, which="major", alpha=0.3)
    ax.grid(True, which="minor", alpha=0.15)
    ax.legend()
    
    out_file = plot_dir / "sim_benchmark_sps.png"
    fig.savefig(out_file, dpi=150)
    print(f"Saved plot to {out_file}")
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description="Benchmark MuJoCo Simulation Speed")
    parser.add_argument("--xml", type=str, default=os.path.join(os.path.dirname(__file__), "xmls/humanoid/humanoid.xml"), help="Path to XML model.")
    parser.add_argument("--batch-sizes", type=str, default="64,128,256,512,1024,2048,4096", help="Comma separated batch sizes")
    parser.add_argument("--steps", type=int, default=1000, help="Simulation steps per run")
    parser.add_argument(
        "--rollout-multi-skip-checks",
        type=str,
        default="both",
        choices=["both", "true", "false"],
        help="Run multi-thread rollout with skip_checks=false/true/both.",
    )
    parser.add_argument("--out", type=str, default="benchmark/outputs/sim/results.json", help="Output JSON path")
    parser.add_argument("--plot-dir", type=str, default="benchmark/outputs/sim", help="Plot output directory")
    parser.add_argument(
        "--motrixsim",
        type=str,
        default="auto",
        choices=["auto", "on", "off"],
        help="Enable MotrixSim backend benchmark: auto=run when dependency exists.",
    )
    
    args = parser.parse_args()
    
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    
    print(f"Loading model: {args.xml}")
    try:
        model = get_model(args.xml)
        print(f"Model loaded: nq={model.nq}, nv={model.nv}, nu={model.nu}")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    records = []
    
    print("\nStarting Benchmark...")
    print(f"{'Backend':<25} | {'Batch':<6} | {'SPS':<12} | {'Time(s)':<8}")
    print("-" * 60)
    
    # Check JAX devices
    if jax:
        print(f"JAX Devices: {jax.devices()}")
        try:
            # Force enable CPU fallback if not already
            jax.config.update("jax_platform_name", "cpu")
            _ = jax.devices("cpu")
            # Revert to default (which might be metal) but we ensured cpu is init
            jax.config.update("jax_platform_name", None) 
        except:
            pass
    
    skip_checks_flags = parse_skip_checks_mode(args.rollout_multi_skip_checks)
    run_motrixsim_flag = args.motrixsim == "on" or (args.motrixsim == "auto" and mtx is not None)

    for bs in batch_sizes:
        # 1. MuJoCo Naive Serial (Single-thread CPU loop)
        if bs <= 1024:
            try:
                r_cpu = run_mujoco_cpu_naive(model, bs, args.steps)
                print(f"{r_cpu.backend:<25} | {bs:<6} | {r_cpu.sps:<12.1f} | {r_cpu.elapsed_sec:<8.4f}")
                if r_cpu.elapsed_sec > 0 or r_cpu.sps > 0:
                    records.append(r_cpu)
            except Exception as e:
                print(f"mujoco_cpu error bs={bs}: {e}")
        else:
             print(f"{'mujoco_cpu_naive':<25} | {bs:<6} | {'SKIP':<12} | {'-'}")

        # 2. MuJoCo Rollout (Single-core)
        if bs <= 2048:
            try:
                r_roll1 = run_mujoco_rollout_runner(model, bs, args.steps, 1, "mujoco_rollout_single")
                print(f"{r_roll1.backend:<25} | {bs:<6} | {r_roll1.sps:<12.1f} | {r_roll1.elapsed_sec:<8.4f}")
                if r_roll1.elapsed_sec > 0 or r_roll1.sps > 0:
                    records.append(r_roll1)
            except Exception as e:
                print(f"mujoco_rollout_single error bs={bs}: {e}")
        else:
             print(f"{'mujoco_rollout_single':<25} | {bs:<6} | {'SKIP':<12} | {'-'}")

        # 3. MuJoCo Rollout (Multi-core), compare skip_checks settings
        for skip_checks in skip_checks_flags:
            # backend_name = f"mujoco_rollout_multi_skip_checks_{str(skip_checks).lower()}"
            backend_name = "mujoco_rollout_multi" if skip_checks else "mujoco_rollout_multi_sc"
            try:
                n_threads = min(bs, os.cpu_count())
                r_roll = run_mujoco_rollout_runner(
                    model,
                    bs,
                    args.steps,
                    n_threads,
                    backend_name,
                    skip_checks=skip_checks,
                )
                print(f"{r_roll.backend:<25} | {bs:<6} | {r_roll.sps:<12.1f} | {r_roll.elapsed_sec:<8.4f}")
                if r_roll.elapsed_sec > 0 or r_roll.sps > 0:
                    records.append(r_roll)
            except Exception as e:
                print(f"{backend_name} error bs={bs}: {e}")


        # 4. MJX (GPU)
        if mjx and jax and jax.devices()[0].platform == "cpu":
            print("JAX CPU device found, skipping MJX benchmark")
        else:
            try:
                r_mjx = run_mjx(model, bs, args.steps)
                print(f"{r_mjx.backend:<25} | {bs:<6} | {r_mjx.sps:<12.1f} | {r_mjx.elapsed_sec:<8.4f}")
                if r_mjx.elapsed_sec > 0 or r_mjx.sps > 0:
                    records.append(r_mjx)
            except Exception as e:
                # Fallback to CPU if Metal fails (common on current MJX + Metal combination)
                if "Unsupported device" in str(e) or "METAL" in str(e):
                    if bs <= 1024:
                        try:
                            # Explicitly get CPU device
                            cpu_devs = jax.devices("cpu")
                            if not cpu_devs:
                                print(f"mjx_cpu_fallback error bs={bs}: No CPU devices found")
                            else:
                                cpu_dev = cpu_devs[0]
                                # IMPORTANT: We must temporarily set default device to CPU
                                # because some internal mjx ops might not respect explicit device args
                                with jax.default_device(cpu_dev):
                                    r_mjx_cpu = run_mjx(model, bs, args.steps, device=cpu_dev)
                                
                                print(f"{r_mjx_cpu.backend:<25} | {bs:<6} | {r_mjx_cpu.sps:<12.1f} | {r_mjx_cpu.elapsed_sec:<8.4f}")
                                if r_mjx_cpu.elapsed_sec > 0:
                                    records.append(r_mjx_cpu)
                        except Exception as e2:
                            print(f"mjx_cpu_fallback error bs={bs}: {e2}")
                    else:
                        print(f"{'mjx':<25} | {bs:<6} | {'SKIP':<12} | {'-'}")
                else:
                    print(f"mjx_metal error bs={bs}: {e}")

        # 5. MotrixSim backend
        if run_motrixsim_flag:
            try:
                r_mtx = run_motrixsim(args.xml, bs, args.steps)
                print(f"{r_mtx.backend:<25} | {bs:<6} | {r_mtx.sps:<12.1f} | {r_mtx.elapsed_sec:<8.4f}")
                if r_mtx.elapsed_sec > 0 or r_mtx.sps > 0:
                    records.append(r_mtx)
            except Exception as e:
                print(f"motrixsim error bs={bs}: {e}")
        elif args.motrixsim == "on":
            print(f"{'motrixsim':<25} | {bs:<6} | {'MISSING':<12} | {'-'}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    data = {
        "meta": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "device_info": get_device_info_dict(),
            "xml": args.xml,
            "steps": args.steps
        },
        "results": [asdict(r) for r in records]
    }
    
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
        
    print(f"\nResults saved to {out_path}")
    plot_results(records, Path(args.plot_dir))

if __name__ == "__main__":
    main()
