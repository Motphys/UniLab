#!/usr/bin/env python3
"""
Benchmark comparison specifically between MuJoCo Multi-thread Rollout and MotrixSim.
"""
import argparse
import os
import time
from pathlib import Path
import numpy as np

try:
    import mujoco
    import mujoco.rollout
except ImportError:
    mujoco = None

try:
    import motrixsim as mtx
except ImportError:
    mtx = None

def run_mujoco_multi(xml_path: str, batch_size: int, steps: int, warmup: int = 5):
    """Run MuJoCo Rollout Multi-Core."""
    if mujoco is None or not hasattr(mujoco, "rollout"):
        return None, 0.0

    model = mujoco.MjModel.from_xml_path(xml_path)
    nthread = min(batch_size, os.cpu_count())
    worker_data = [mujoco.MjData(model) for _ in range(nthread)]
    
    nstate = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    nu = model.nu
    
    initial_state = np.zeros((batch_size, nstate))
    ctrl = np.zeros((batch_size, 1, nu))
    model_batch = [model] * batch_size
    state_buf = np.empty((batch_size, 1, nstate), dtype=np.float64)
    sensor_buf = np.empty((batch_size, 1, model.nsensordata), dtype=np.float64)

    # Warmup
    try:
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
    except Exception as e:
        print(f"Rollout warmup failed: {e}")
        return None, 0.0

    start = time.perf_counter()
    
    # Run Benchmark
    for _ in range(steps):
        state_traj, _ = mujoco.rollout.rollout(
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
        if state_traj is not None:
             initial_state[:] = state_traj[:, -1, :]

    end = time.perf_counter()
    
    elapsed = max(end - start, 1e-6)
    sps = (batch_size * steps) / elapsed
    return elapsed, sps

def run_motrixsim(xml_path: str, batch_size: int, steps: int, warmup: int = 20):
    """Run MotrixSim batch physics backend."""
    if mtx is None:
        return None, 0.0

    try:
        model = mtx.load_model(xml_path)
        data = mtx.SceneData(model, batch=(batch_size,))

        # Warmup
        for _ in range(warmup):
            model.step(data)

        start = time.perf_counter()
        
        # Run Benchmark
        for _ in range(steps):
            model.step(data)
            
        end = time.perf_counter()

        elapsed = max(end - start, 1e-6)
        sps = (batch_size * steps) / elapsed
        return elapsed, sps
    except Exception as e:
        print(f"MotrixSim error: {e}")
        return None, 0.0


def main():
    parser = argparse.ArgumentParser(description="Benchmark MuJoCo Multi-thread vs MotrixSim")
    parser.add_argument("--xml", type=str, default=os.path.join(os.path.dirname(__file__), "xmls/humanoid/humanoid.xml"), help="Path to XML model.")
    parser.add_argument("--num-envs", type=int, default=8192, help="Number of environments (batch size), default: 8192")
    parser.add_argument("--steps", type=int, default=20, help="Simulation steps per run, default: 100")

    args = parser.parse_args()
    
    if not os.path.exists(args.xml):
        print(f"Error: Model file not found: {args.xml}")
        return

    print(f"Loading model: {args.xml}")
    print(f"Number of environments: {args.num_envs}")
    print(f"Simulation steps: {args.steps}")
    print("\nStarting Benchmark...")
    print(f"{'Backend':<25} | {'Batch':<6} | {'SPS':<12} | {'Time(s)':<8}")
    print("-" * 60)
    
    # Run MuJoCo Multi-thread
    if mujoco is not None:
        try:
            elapsed, sps = run_mujoco_multi(args.xml, args.num_envs, args.steps)
            if elapsed is not None:
                print(f"{'mujoco_rollout_multi':<25} | {args.num_envs:<6} | {sps:<12.1f} | {elapsed:<8.4f}")
            else:
                print(f"{'mujoco_rollout_multi':<25} | {args.num_envs:<6} | {'FAILED':<12} | {'-':<8}")
        except Exception as e:
            print(f"MuJoCo error: {e}")
    else:
        print(f"{'mujoco_rollout_multi':<25} | {args.num_envs:<6} | {'MISSING':<12} | {'-':<8}")

    # Run MotrixSim
    if mtx is not None:
        try:
            elapsed, sps = run_motrixsim(args.xml, args.num_envs, args.steps)
            if elapsed is not None:
                print(f"{'motrixsim':<25} | {args.num_envs:<6} | {sps:<12.1f} | {elapsed:<8.4f}")
            else:
                print(f"{'motrixsim':<25} | {args.num_envs:<6} | {'FAILED':<12} | {'-':<8}")
        except Exception as e:
            print(f"MotrixSim error: {e}")
    else:
        print(f"{'motrixsim':<25} | {args.num_envs:<6} | {'MISSING':<12} | {'-':<8}")

if __name__ == "__main__":
    main()
