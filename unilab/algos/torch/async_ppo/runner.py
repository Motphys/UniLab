"""Async PPO runner."""

import torch
from rsl_rl.algorithms import PPO
from tensordict import TensorDict

from unilab.ipc import AsyncRunner, SharedWeightSync

from .buffer import OnPolicyReplayBuffer
from .learner import AsyncPPOLearner


class AsyncPPORunner(AsyncRunner):
    """Async PPO training orchestrator."""

    def _get_default_device(self) -> str:
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _build_learner(self) -> AsyncPPOLearner:
        from unilab.base import registry

        env = registry.make(self.env_name, num_envs=self.num_envs, sim_backend="mujoco")
        obs_dim = env.observation_space.shape[0]  # type: ignore[index]
        action_dim = env.action_space.shape[0]  # type: ignore[index]

        obs_example = torch.zeros((self.num_envs, obs_dim), device=self.device)
        td_example = TensorDict({"policy": obs_example}, batch_size=self.num_envs)

        ppo = PPO.construct_algorithm(
            env=env,
            obs=td_example,
            cfg=self.rl_cfg,
            device=self.device,
        )

        steps_per_env = self.rl_cfg.get("num_steps_per_env", 24)
        buffer = OnPolicyReplayBuffer(
            capacity_rollouts=10,
            num_envs=self.num_envs,
            num_steps=steps_per_env,
            obs_dim=obs_dim,
            action_dim=action_dim,
            device=self.device,
        )

        return AsyncPPOLearner(ppo, buffer)

    def _collector_fn(self, stop_event, **kwargs):
        from .worker import async_ppo_collector_fn

        return async_ppo_collector_fn(stop_event, **kwargs)

    def learn(self, max_iterations: int, save_interval: int = 50, log_dir: str = "logs"):
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
        state_dict = {**ppo.actor.state_dict(), **ppo.critic.state_dict()}
        weight_sync = SharedWeightSync.from_state_dict(state_dict, shm_name="async_ppo_weights")
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
                "weight_sync_name": weight_sync.shm_name,
                "weight_param_shapes": weight_sync.param_shapes,
                "metrics_queue": None,
                "collector_device": self.collector_device,
            },
        )

        # Training loop
        for it in range(max_iterations):
            if not buffer.is_ready():
                continue

            metrics = learner.update()

            # Sync weights
            state_dict = {**ppo.actor.state_dict(), **ppo.critic.state_dict()}
            weight_sync.write_weights(state_dict)

            if it % 10 == 0:
                print(f"Iter {it}: {metrics}")
