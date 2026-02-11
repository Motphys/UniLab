import argparse
import sys
import os
import datetime
from pathlib import Path
import torch
import numpy as np

# Add project root to sys.path
ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR))

from unilab.algo.boprl.runner import BOPRLRunner
from unilab.config import locomotion_params


def main():
    parser = argparse.ArgumentParser(description="Train BOPRL Baseline (Ray-based)")
    parser.add_argument("--task", type=str, default="Go2JoystickFlatTerrain", help="Task name")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of Ray rollout workers")

    # Auto-detect device if not specified or 'auto'
    default_device = "cpu"
    if torch.cuda.is_available():
        default_device = "cuda:0"
    elif torch.backends.mps.is_available():
        default_device = "mps"

    parser.add_argument("--device", type=str, default=default_device, help="Learner device (e.g. cuda:0, mps, cpu)")
    parser.add_argument("--steps_per_env", type=int, default=24, help="Steps per environment per iteration")
    parser.add_argument("--total_envs", type=int, default=1024, help="Total number of environments")
    parser.add_argument("--max_iterations", type=int, default=1500, help="Total iterations")
    parser.add_argument("--save_interval", type=int, default=50, help="Save checkpoint every N iterations")

    args = parser.parse_args()

    # Get config
    print(f"Loading config for {args.task}...")
    # This returns ml_collections.ConfigDict
    rl_cfg = locomotion_params.rsl_rl_config(args.task)
    # Convert to standard dict for serialization
    rl_cfg = rl_cfg.to_dict()

    # Force log std for stability (prevents negative std crashes on MPS)
    env_cfg_overrides = {}
    print(f"DEBUG: Config keys: {rl_cfg.keys()}")
    if "policy" in rl_cfg:
        print("DEBUG: Found 'policy' key. Setting noise_std_type='log'.")
        rl_cfg["policy"]["noise_std_type"] = "log"
    elif "actor" in rl_cfg:
        print("DEBUG: Found 'actor' key. Setting noise_std_type='log'.")
        rl_cfg["actor"]["noise_std_type"] = "log"
    else:
        print("WARNING: Could not find 'policy' or 'actor' in config to set noise_std_type.")

    num_envs_per_worker = max(1, args.total_envs // args.num_workers)

    # Setup logging directory
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = str(ROOT_DIR / "logs" / "boprl_train" / args.task / timestamp)

    print(
        f"Starting BOPRL Runner with {args.num_workers} workers, total_envs={args.total_envs} ({num_envs_per_worker}/worker) on learner_device={args.device}..."
    )
    print(f"Log dir: {log_dir}")

    runner = BOPRLRunner(
        env_name=args.task,
        env_cfg_overrides=env_cfg_overrides,
        rl_cfg=rl_cfg,
        device=args.device,
        num_workers=args.num_workers,
        steps_per_env=args.steps_per_env,
        num_envs_per_worker=num_envs_per_worker,
    )

    try:
        runner.learn(
            max_iterations=args.max_iterations,
            save_interval=args.save_interval,
            log_dir=log_dir,
        )
    except KeyboardInterrupt:
        print("Interrupted by user.")
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback

        traceback.print_exc()
    finally:
        print("Closing runner...")
        runner.close()


if __name__ == "__main__":
    main()
