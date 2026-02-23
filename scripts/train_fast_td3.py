"""Train FastTD3 agent — native multiprocessing, no Ray."""

import argparse
import sys
import os
import datetime
from pathlib import Path
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


def main():
    parser = argparse.ArgumentParser(description="Train FastTD3 (no Ray)")
    parser.add_argument("--task", type=str, default="Go2JoystickFlatTerrain")
    parser.add_argument("--max_iterations", type=int, default=1500)
    parser.add_argument("--save_interval", type=int, default=50)
    parser.add_argument("--num_envs", type=int, default=4096)
    parser.add_argument("--obs_dim", type=int, default=48)
    parser.add_argument("--action_dim", type=int, default=12)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--collector_device", type=str, default=None)
    parser.add_argument("--log_dir", type=str, default=None)
    # Holosoma-aligned
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument("--updates_per_step", type=int, default=8)
    parser.add_argument("--policy_delay", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=0.97)
    parser.add_argument("--tau", type=float, default=0.125)
    parser.add_argument("--actor_lr", type=float, default=3e-4)
    parser.add_argument("--critic_lr", type=float, default=3e-4)
    parser.add_argument("--actor_hidden_dim", type=int, default=512)
    parser.add_argument("--critic_hidden_dim", type=int, default=768)
    parser.add_argument("--num_atoms", type=int, default=101)
    parser.add_argument("--warmup_steps", type=int, default=5000)
    parser.add_argument("--replay_buffer_n", type=int, default=1024)
    parser.add_argument("--steps_per_env", type=int, default=24)
    parser.add_argument("--exploration_noise", type=float, default=0.5)

    args = parser.parse_args()

    ensure_registries()

    if args.log_dir is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.log_dir = os.path.join("logs", f"fast_td3_{args.task}_{timestamp}")

    from unilab.algos.torch.fast_td3.runner import FastTD3Runner

    runner = FastTD3Runner(
        env_name=args.task,
        device=args.device,
        collector_device=args.collector_device,
        num_envs=args.num_envs,
        obs_dim=args.obs_dim,
        action_dim=args.action_dim,
        steps_per_env=args.steps_per_env,
        replay_buffer_n=args.replay_buffer_n,
        batch_size=args.batch_size,
        warmup_steps=args.warmup_steps,
        updates_per_step=args.updates_per_step,
        policy_delay=args.policy_delay,
        gamma=args.gamma,
        tau=args.tau,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        actor_hidden_dim=args.actor_hidden_dim,
        critic_hidden_dim=args.critic_hidden_dim,
        num_atoms=args.num_atoms,
        exploration_noise=args.exploration_noise,
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
