"""Train APPO agent — native multiprocessing, no Ray."""

import argparse
import sys
import os
import datetime
from pathlib import Path
import torch
import pkgutil
import importlib

ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR))


def ensure_registries():
    try:
        import unilab.envs.locomotion
        package = unilab.envs.locomotion
        if hasattr(package, "__path__"):
            for _, name, ispkg in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
                try:
                    importlib.import_module(name)
                except Exception:
                    pass
    except ImportError:
        pass

    try:
        import unilab.envs.locomotion.walking
        package = unilab.envs.locomotion.walking
        if hasattr(package, "__path__"):
            for _, name, ispkg in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
                try:
                    importlib.import_module(name)
                except Exception:
                    pass
    except ImportError:
        pass


def main():
    parser = argparse.ArgumentParser(description="Train APPO (no Ray)")
    parser.add_argument("--task", type=str, default="Go2JoystickFlatTerrain")
    parser.add_argument("--max_iterations", type=int, default=1500)
    parser.add_argument("--save_interval", type=int, default=50)
    parser.add_argument("--total_envs", type=int, default=4096)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--collector_device", type=str, default=None)
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument("--steps_per_env", type=int, default=24)

    args = parser.parse_args()

    ensure_registries()

    if args.log_dir is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.log_dir = os.path.join("logs", f"appo_{args.task}_{timestamp}")

    rl_cfg = {
        "obs_groups": {
            "actor": {"policy": 48},
        },
        "actor": {
            "class_name": "rsl_rl.models.MLPModel",
            "num_actions": 12,
        },
    }

    from unilab.algos.torch.appo.runner import APPORunner

    runner = APPORunner(
        env_name=args.task,
        env_cfg_overrides={},
        rl_cfg=rl_cfg,
        device=args.device,
        collector_device=args.collector_device,
        num_envs=args.total_envs,
        steps_per_env=args.steps_per_env,
    )

    try:
        runner.learn(
            max_iterations=args.max_iterations,
            save_interval=args.save_interval,
            log_dir=args.log_dir,
        )
    finally:
        runner.close()


if __name__ == "__main__":
    main()
