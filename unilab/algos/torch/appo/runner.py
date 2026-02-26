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
from unilab.algos.torch.appo.learner import APPOLearner, APPOActorWrapper
from unilab.algos.torch.common.logger import TrainingLogger
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
        self.obs_dim, self.action_dim = self._detect_dims()

        # Update rl_cfg so internal RSL-RL networks get correct observation dimension
        if "obs_groups" not in self.rl_cfg:
            self.rl_cfg["obs_groups"] = {"actor": {"policy": self.obs_dim}}
        else:
            actor_group = self.rl_cfg["obs_groups"].get("actor", self.rl_cfg["obs_groups"].get("policy", {}))
            if isinstance(actor_group, dict) and "policy" in actor_group:
                actor_group["policy"] = self.obs_dim

    def _detect_dims(self):
        """Create a tiny env to read obs/action dims, then close it."""
        from unilab.envs import registry
        from unilab.algos.torch.common.worker import ensure_registries
        ensure_registries()

        env = registry.make(self.env_name, num_envs=1, sim_backend="mujoco")
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        env.close()

        return obs_dim, action_dim

    def _build_learner(self):
        cfg = dict(self.rl_cfg)
        if is_rsl_rl_v4():
            cfg = convert_config_v3_to_v4(cfg)

        from tensordict import TensorDict
        import torch
        obs_example = torch.zeros((self.num_envs, self.obs_dim), device=self.device)
        td_example = TensorDict({"policy": obs_example}, batch_size=self.num_envs)

        # Build actor 
        actor_cfg = cfg.get("policy", cfg.get("actor", {})).copy()
        actor_cls = resolve_callable(actor_cfg.pop("class_name"))
        actor_cfg.pop("num_actions", None)
        actor_core = actor_cls(td_example, cfg["obs_groups"], "actor", self.action_dim, **actor_cfg)
        actor = APPOActorWrapper(actor_core, self.action_dim)

        # Build critic
        critic_cfg = cfg.get("critic", cfg.get("policy", cfg.get("actor", {}))).copy()
        critic_cls = resolve_callable(critic_cfg.pop("class_name", "rsl_rl.models.MLPModel"))
        critic_cfg.pop("num_actions", None)
        critic = critic_cls(td_example, cfg["obs_groups"], "actor", 1, **critic_cfg)

        learner = APPOLearner(
            actor=actor,
            critic=critic,
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

        metrics_queue = mp.Queue(maxsize=100)

        # Start collector
        collector_kwargs = {
            "env_name": self.env_name,
            "env_cfg_overrides": self.env_cfg_overrides,
            "rl_cfg": self.rl_cfg,
            "num_envs": self.num_envs,
            "steps_per_env": self.steps_per_env,
            "shm_storage_name": shared_storage.name,
            "sync_primitives": (shared_storage._write_idx, shared_storage._read_idx, shared_storage._ready),
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "weight_sync_name": weight_sync.name,
            "weight_param_shapes": weight_param_shapes,
            "metrics_queue": metrics_queue,
            "collector_device": self.collector_device,
        }
        self._start_collector(
            target_fn=appo_collector_fn,
            kwargs={"stop_event": self._stop_event, **collector_kwargs},
        )

        logger = TrainingLogger(
            algo_name="APPO",
            max_iterations=max_iterations,
            num_envs=self.num_envs,
            env_name=self.env_name,
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
        )
        logger.start()
        logger.log_status("Waiting for first rollout...")

        reward_history = deque(maxlen=100)
        start_time = time.time()
        total_steps = 0
        last_metrics_msg = {}

        for iteration in range(1, max_iterations + 1):
            iter_start = time.time()
            # Wait for collector to provide data
            if not shared_storage.wait_for_data(timeout=60.0):
                logger.log_status(f"[yellow]Warning: Timeout waiting for data at iteration {iteration}[/]")
                continue

            # Read data and update
            rollout_data = shared_storage.read_torch(self.device)
            shared_storage.advance_read()
            collect_time = time.time() - iter_start

            train_start = time.time()
            batch_dict = {}
            for k, v in rollout_data.items():
                if k != "last_obs" and v.ndim >= 2:
                    batch_dict[k] = v.transpose(0, 1)
                else:
                    batch_dict[k] = v

            if "obs" in batch_dict:
                batch_dict["observations"] = batch_dict.pop("obs")
            if "log_probs" in batch_dict:
                batch_dict["actions_log_prob"] = batch_dict.pop("log_probs")

            learner.process_batch(batch_dict)
            metrics = learner.update(batch_dict)

            # Sync weights
            weight_sync.write_weights(learner.actor.state_dict())
            train_time = time.time() - train_start

            # Logging
            try:
                while not metrics_queue.empty():
                    last_metrics_msg = metrics_queue.get_nowait()
            except Exception:
                pass

            mean_reward = last_metrics_msg.get("mean_ep_reward", 0.0)
            mean_length = last_metrics_msg.get("mean_ep_length", 0.0)
            reward_components = last_metrics_msg.get("reward_components", {})
            metrics["episode_length"] = mean_length

            logger.log_step(
                iteration=iteration,
                metrics=metrics,
                reward=mean_reward,
                reward_components=reward_components,
                collect_time=collect_time,
                train_time=train_time,
            )

            # Save
            if save_interval > 0 and iteration % save_interval == 0:
                ckpt_path = os.path.join(log_dir, f"model_{iteration}.pt")
                torch.save(learner.get_state_dict(), ckpt_path)
                logger.log_save(ckpt_path)

        ckpt_path = os.path.join(log_dir, f"model_{max_iterations}.pt")
        torch.save(learner.get_state_dict(), ckpt_path)
        logger.log_save(ckpt_path)
        logger.finish()

    def _check_collector_alive(self) -> bool:
        if self._collector_process is not None and not self._collector_process.is_alive():
            return False
        return True
