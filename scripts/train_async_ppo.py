"""Train async PPO agent."""

import argparse
import datetime
import os
import sys
from pathlib import Path

import torch

ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR))


def ensure_registries():
    import importlib
    import pkgutil

    try:
        import unilab.envs.locomotion

        package = unilab.envs.locomotion
        if hasattr(package, "__path__"):
            for _, name, _ in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
                try:
                    importlib.import_module(name)
                except Exception:
                    pass
    except ImportError:
        pass


def main():
    parser = argparse.ArgumentParser(description="Train async PPO")
    parser.add_argument("--task", type=str, default="Go1JoystickFlatTerrain")
    parser.add_argument("--max_iterations", type=int, default=None)
    parser.add_argument("--save_interval", type=int, default=50)
    parser.add_argument("--total_envs", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--collector_device", type=str, default=None)
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument(
        "--logger",
        type=str,
        default="tensorboard",
        choices=["tensorboard", "wandb", "none", "no_print"],
    )

    args = parser.parse_args()
    ensure_registries()

    from unilab.config.locomotion_params import async_ppo_config

    rl_cfg = async_ppo_config(args.task)

    if args.total_envs:
        rl_cfg["num_envs"] = args.total_envs
    if args.max_iterations:
        rl_cfg["max_iterations"] = args.max_iterations

    if args.log_dir is None:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        args.log_dir = os.path.join(ROOT_DIR, "logs", "async_ppo", args.task, f"{timestamp}_mujoco")

    from unilab.algos.torch.async_ppo.runner import AsyncPPORunner

    runner = AsyncPPORunner(
        env_name=args.task,
        env_cfg_overrides={},
        rl_cfg=rl_cfg,
        device=args.device,
        collector_device=args.collector_device or "cpu",
        num_envs=rl_cfg["num_envs"],
    )

    try:
        runner.learn(
            max_iterations=rl_cfg["max_iterations"],
            save_interval=args.save_interval,
            log_dir=args.log_dir,
            logger_type=args.logger,
        )
    finally:
        runner.close()


if __name__ == "__main__":
    main()
