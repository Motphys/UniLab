"""Short end-to-end Go2 collector benchmark for the SuperDex roadmap.

The timed loop includes policy inference, action conversion, env.step (including
physics, observations, reward and termination/reset), and rollout bookkeeping.
It intentionally does not run PPO updates or claim training quality.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from unilab.base import registry
from unilab.base.config_adapter import BackendAdapter


def run(mode: str, *, num_envs: int, workers: int, warmup: int, steps: int, repeats: int) -> dict:
    root = Path(__file__).resolve().parents[2]
    values: list[float] = []
    rewards: list[float] = []
    for repeat in range(repeats):
        GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=str(root / "src/unilab/conf/ppo"), version_base="1.3"):
            cfg = compose("config", overrides=["task=go2_joystick_flat/superdex"])
        override = BackendAdapter(cfg, root_dir=root).build_task_env_cfg_override()
        env = registry.make(
            "Go2JoystickFlat", sim_backend="superdex", num_envs=num_envs,
            env_cfg_override={**override, "superdex_num_workers": workers, "seed": repeat + 1},
        )
        try:
            state = env.init_state()
            obs_dim = int(env.obs_groups_spec["obs"])
            policy = torch.nn.Sequential(torch.nn.Linear(obs_dim, 128), torch.nn.Tanh(),
                                         torch.nn.Linear(128, env.action_space.shape[0]),
                                         torch.nn.Tanh()).eval()
            if mode == "unbatched":
                env._backend._pre_step_control_fn = lambda _backend, controls: controls
            for _ in range(warmup):
                with torch.inference_mode():
                    action = policy(torch.from_numpy(state.obs["obs"]).float()).numpy()
                state = env.step(action)
            started = time.perf_counter()
            total_reward = 0.0
            total_done = 0
            for _ in range(steps):
                with torch.inference_mode():
                    action = policy(torch.from_numpy(state.obs["obs"]).float()).numpy()
                state = env.step(action)
                total_reward += float(np.asarray(state.reward).mean())
                total_done += int(np.asarray(state.terminated).sum())
            values.append(num_envs * steps / (time.perf_counter() - started))
            rewards.append(total_reward / steps)
        finally:
            env.close()
    return {"mode": mode, "metric": "collector-env-step/s", "num_envs": num_envs,
            "workers": workers, "warmup": warmup, "steps": steps, "repeats": values,
            "median": float(np.median(values)), "mean_reward_per_step": float(np.mean(rewards)),
            "done_count": total_done}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("batched", "unbatched"), required=True)
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    print("SUPERDEX_GO2_COLLECTOR_RESULT=" + json.dumps(run(**vars(args)), sort_keys=True))


if __name__ == "__main__":
    main()
