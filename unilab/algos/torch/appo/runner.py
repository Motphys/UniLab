"""APPO Runner — Asynchronous PPO with native multiprocessing (no Ray).

Pipeline:
  1. Collector subprocess collects on-policy rollouts → SharedOnPolicyStorage
  2. Learner reads rollouts, computes V-trace corrected updates
  3. Weights synced back to collector via SharedWeightSync
"""

import multiprocessing as mp
import os
import time
import statistics
import torch
from collections import defaultdict, deque

from unilab.algos.torch.common.async_runner import (
    AsyncRunner,
    SharedOnPolicyStorage,
    SharedWeightSync,
)
from unilab.algos.torch.appo.worker import appo_collector_fn
from unilab.algos.torch.appo.learner import APPOLearner
from unilab.utils.rsl_rl_compat import convert_config_v3_to_v4, is_rsl_rl_v4
from rsl_rl.utils import resolve_callable


class APPORunner(AsyncRunner):
    """APPO async runner using shared memory (no Ray dependency)."""

    def __init__(
        self,
        env_name: str,
        env_cfg_overrides: dict,
        rl_cfg: dict,
        device: str | None = None,
        collector_device: str | None = None,
        num_envs: int = 1024,
        steps_per_env: int = 24,
        num_workers: int = 1,  # kept for API compat, but only 1 collector used
    ):
        super().__init__(
            env_name=env_name,
            env_cfg_overrides=env_cfg_overrides,
            rl_cfg=rl_cfg,
            device=device,
            collector_device=collector_device,
            num_envs=num_envs,
        )

        self.steps_per_env = steps_per_env

        # Resolve dims
        self._resolve_dims()

    def _resolve_dims(self):
        cfg = dict(self.rl_cfg)
        if is_rsl_rl_v4():
            cfg = convert_config_v3_to_v4(cfg)

        obs_groups = cfg.get("obs_groups", {})
        actor_group = obs_groups.get("actor", obs_groups.get("policy", {}))
        if isinstance(actor_group, dict):
            self.obs_dim = sum(v for v in actor_group.values() if isinstance(v, int))
        else:
            self.obs_dim = actor_group

        actor_cfg = cfg.get("actor", {})
        self.action_dim = actor_cfg.get("output_dim", actor_cfg.get("num_actions", 12))

    def _build_learner(self):
        cfg = dict(self.rl_cfg)
        if is_rsl_rl_v4():
            cfg = convert_config_v3_to_v4(cfg)

        from tensordict import TensorDict
        obs_example = torch.zeros((self.num_envs, self.obs_dim), device=self.device)
        td_example = TensorDict({"policy": obs_example}, batch_size=self.num_envs)

        learner = APPOLearner(
            td_example=td_example,
            rl_cfg=cfg,
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            device=self.device,
            num_envs=self.num_envs,
            steps_per_env=self.steps_per_env,
        )
        return learner

    def _collector_fn(self, stop_event, **kwargs):
        appo_collector_fn(stop_event=stop_event, **kwargs)

    def learn(
        self,
        max_iterations: int = 1500,
        save_interval: int = 50,
        log_dir: str = "logs",
    ):
        os.makedirs(log_dir, exist_ok=True)

        learner = self._build_learner()

        # Create shared storage
        shared_storage = SharedOnPolicyStorage(
            num_envs=self.num_envs,
            num_steps=self.steps_per_env,
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            create=True,
        )
        self._shared_resources.append(shared_storage)

        # Create weight sync
        weight_sync = SharedWeightSync.from_state_dict(
            learner.actor.state_dict(), create=True
        )
        self._shared_resources.append(weight_sync)

        weight_param_shapes = {
            name: p.shape for name, p in learner.actor.state_dict().items()
        }

        # Start collector
        collector_kwargs = {
            "env_name": self.env_name,
            "env_cfg_overrides": self.env_cfg_overrides,
            "rl_cfg": self.rl_cfg,
            "num_envs": self.num_envs,
            "steps_per_env": self.steps_per_env,
            "shm_storage_name": shared_storage.name,
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "weight_sync_name": weight_sync.name,
            "weight_param_shapes": weight_param_shapes,
            "collector_device": self.collector_device,
        }
        self._start_collector(
            target_fn=appo_collector_fn,
            kwargs={"stop_event": self._stop_event, **collector_kwargs},
        )

        # Training loop
        print(f"APPO training started. Waiting for first rollout...")

        reward_history = deque(maxlen=100)
        start_time = time.time()

        for iteration in range(1, max_iterations + 1):
            # Wait for collector to provide data
            if not shared_storage.wait_for_data(timeout=60.0):
                print(f"Warning: Timeout waiting for data at iteration {iteration}")
                continue

            # Read data and update
            rollout_data = shared_storage.read_torch(self.device)

            # Learner update
            metrics = learner.update(rollout_data)

            # Sync weights
            weight_sync.write_weights(learner.actor.state_dict())

            # Track rewards
            rewards = rollout_data.get("rewards", None)
            if rewards is not None:
                mean_rew = rewards.mean().item()
                reward_history.append(mean_rew)

            # Logging
            if iteration % 10 == 0:
                elapsed = time.time() - start_time
                mean_reward = statistics.mean(reward_history) if reward_history else 0.0
                print(
                    f"[{iteration}/{max_iterations}] "
                    f"t={elapsed:.0f}s | "
                    f"rew={mean_reward:.4f} | "
                    f"p_loss={metrics.get('policy_loss', 0):.4f} | "
                    f"v_loss={metrics.get('value_loss', 0):.4f}"
                )

            # Save
            if save_interval > 0 and iteration % save_interval == 0:
                ckpt_path = os.path.join(log_dir, f"model_{iteration}.pt")
                torch.save({
                    "iteration": iteration,
                    "actor_state_dict": learner.actor.state_dict(),
                    "critic_state_dict": learner.critic.state_dict(),
                }, ckpt_path)
                print(f"Saved checkpoint: {ckpt_path}")

        # Final save
        ckpt_path = os.path.join(log_dir, f"model_{max_iterations}.pt")
        torch.save({
            "iteration": max_iterations,
            "actor_state_dict": learner.actor.state_dict(),
            "critic_state_dict": learner.critic.state_dict(),
        }, ckpt_path)
        print(f"Training complete. Final checkpoint: {ckpt_path}")
