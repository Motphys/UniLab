"""Async PPO runner."""

import os
import time

import torch
from rsl_rl.algorithms import PPO
from tensordict import TensorDict

from unilab.ipc import AsyncRunner, SharedWeightSync
from unilab.utils.hardware_monitor import HardwareMonitor
from unilab.utils.offpolicy_logger import OffPolicyLogger

from .buffer import OnPolicyReplayBuffer
from .learner import AsyncPPOLearner


class AsyncPPORunner(AsyncRunner):
    """Async PPO training orchestrator."""

    def __init__(self, env_name: str, env_cfg_overrides: dict, rl_cfg: dict, **kwargs):
        super().__init__(env_name, env_cfg_overrides, rl_cfg, **kwargs)
        if hasattr(self.rl_cfg, "to_dict"):
            self.rl_cfg = self.rl_cfg.to_dict()
        self._resolve_dims()

    def _get_default_device(self) -> str:
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _resolve_dims(self):
        from unilab.base import registry
        from unilab.utils.algo_utils import ensure_registries

        ensure_registries()
        env = registry.make(self.env_name, num_envs=1, sim_backend="mujoco")
        self.obs_dim = env.observation_space.shape[0]  # type: ignore[index]
        self.action_dim = env.action_space.shape[0]  # type: ignore[index]
        env.close()

    def _build_learner(self) -> AsyncPPOLearner:
        from unilab.base import registry
        from unilab.utils.rsl_rl_compat import convert_config_v3_to_v4, is_rsl_rl_v4

        cfg = dict(self.rl_cfg)
        if is_rsl_rl_v4():
            cfg = convert_config_v3_to_v4(cfg)

        # Create temporary env for PPO construction
        env = registry.make(self.env_name, num_envs=self.num_envs, sim_backend="mujoco")

        # Add num_actions attribute for PPO.construct_algorithm
        if not hasattr(env, 'num_actions'):
            env.num_actions = self.action_dim

        obs_example = torch.zeros((self.num_envs, self.obs_dim), device=self.device)
        td_example = TensorDict({"policy": obs_example}, batch_size=self.num_envs)

        ppo = PPO.construct_algorithm(env=env, obs=td_example, cfg=cfg, device=self.device)

        steps_per_env = cfg.get("num_steps_per_env", 24)
        buffer = OnPolicyReplayBuffer(
            capacity_rollouts=cfg.get("buffer_capacity_rollouts", 10),
            num_envs=self.num_envs,
            num_steps=steps_per_env,
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            device=self.device,
        )
        env.close()
        return AsyncPPOLearner(ppo, buffer)

    def _collector_fn(self, stop_event, **kwargs):
        from .worker import async_ppo_collector_fn

        return async_ppo_collector_fn(stop_event, **kwargs)

    def learn(self, max_iterations: int, save_interval: int = 50, log_dir: str = "logs"):
        os.makedirs(log_dir, exist_ok=True)
        learner = self._build_learner()
        buffer = learner.buffer
        ppo = learner.ppo

        # Share memory
        if hasattr(buffer, "obs"):
            for attr in ["obs", "actions", "rewards", "dones", "log_probs", "values", "last_obs"]:
                getattr(buffer, attr).share_memory_()
        else:
            buffer._storage.share_memory_()
        buffer.ptr.share_memory_()
        buffer.count.share_memory_()

        # Weight sync
        state_dict = {
            f"actor.{k}": v for k, v in ppo.actor.state_dict().items()
        } | {
            f"critic.{k}": v for k, v in ppo.critic.state_dict().items()
        }
        weight_sync = SharedWeightSync.from_state_dict(state_dict, create=True)
        weight_param_shapes = {name: p.shape for name, p in state_dict.items()}
        self._shared_resources.extend([buffer, weight_sync])

        # Start collector
        steps_per_env = self.rl_cfg.get("num_steps_per_env", 24)
        self._start_collector(
            self._collector_fn,
            {
                "stop_event": self._stop_event,
                "env_name": self.env_name,
                "rl_cfg": self.rl_cfg,
                "num_envs": self.num_envs,
                "steps_per_env": steps_per_env,
                "buffer": buffer,
                "weight_sync_name": weight_sync.name,
                "weight_param_shapes": weight_param_shapes,
                "metrics_queue": None,
                "collector_device": self.collector_device,
            },
        )

        # Logger
        logger = OffPolicyLogger(
            algo_name="AsyncPPO",
            max_iterations=max_iterations,
            num_envs=self.num_envs,
            env_name=self.env_name,
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            log_dir=log_dir,
            log_backend="tensorboard",
        )
        logger.start()
        logger.log_status("Waiting for first rollout...")

        # Hardware monitor
        hw_monitor = HardwareMonitor()

        # Training loop
        for iteration in range(1, max_iterations + 1):
            iter_start = time.time()

            # Wait for data
            timeout = 60.0
            wait_start = time.time()
            while not buffer.is_ready() and (time.time() - wait_start) < timeout:
                time.sleep(0.01)

            if not buffer.is_ready():
                logger.log_status(f"[yellow]Timeout waiting for data at iter {iteration}[/]")
                continue

            collect_time = time.time() - iter_start
            train_start = time.time()

            metrics = learner.update()
            state_dict = {
                f"actor.{k}": v for k, v in ppo.actor.state_dict().items()
            } | {
                f"critic.{k}": v for k, v in ppo.critic.state_dict().items()
            }
            weight_sync.write_weights(state_dict)

            train_time = time.time() - train_start

            # Hardware metrics
            if iteration % 10 == 0:
                hw_metrics = hw_monitor.get_metrics()
                metrics.update({f"hardware/{k}": v for k, v in hw_metrics.items()})

            logger.log_step(
                iteration=iteration,
                metrics=metrics,
                reward=0.0,
                reward_components={},
                collect_time=collect_time,
                train_time=train_time,
            )

            if save_interval > 0 and iteration % save_interval == 0:
                ckpt_path = os.path.join(log_dir, f"model_{iteration}.pt")
                torch.save(
                    {"actor": ppo.actor.state_dict(), "critic": ppo.critic.state_dict()}, ckpt_path
                )
                logger.log_save(ckpt_path)

        ckpt_path = os.path.join(log_dir, f"model_{max_iterations}.pt")
        torch.save({"actor": ppo.actor.state_dict(), "critic": ppo.critic.state_dict()}, ckpt_path)
        logger.log_save(ckpt_path)
        logger.finish()
