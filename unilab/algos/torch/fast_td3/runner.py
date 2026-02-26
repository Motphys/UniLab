"""FastTD3 Runner — async training with native multiprocessing (no Ray).

Pipeline identical to FastSAC but uses TD3 deterministic policy with
exploration noise and delayed policy updates.
"""

import multiprocessing as mp
import os
import time
import statistics
import torch
from collections import defaultdict, deque

from unilab.algos.torch.common.async_runner import (
    AsyncRunner,
    SharedReplayBuffer,
    SharedWeightSync,
)
from unilab.algos.torch.common.worker import off_policy_collector_fn
from unilab.algos.torch.common.logger import TrainingLogger
from unilab.algos.torch.fast_td3.learner import FastTD3Learner


class FastTD3Runner(AsyncRunner):
    """FastTD3 async runner using shared memory (no Ray dependency)."""

    def __init__(
        self,
        env_name: str,
        env_cfg_overrides: dict = None,
        device: str | None = None,
        collector_device: str | None = None,
        num_envs: int = 4096,
        steps_per_env: int = 24,
        replay_buffer_n: int = 1024,
        batch_size: int = 8192,
        warmup_steps: int = 5000,
        updates_per_step: int = 8,
        policy_delay: int = 4,
        exploration_noise: float = 0.5,
        gamma: float = 0.97,
        tau: float = 0.125,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        actor_hidden_dim: int = 512,
        critic_hidden_dim: int = 768,
        num_atoms: int = 101,
        use_layer_norm: bool = True,
        **kwargs,
    ):
        super().__init__(
            env_name=env_name,
            env_cfg_overrides=env_cfg_overrides or {},
            rl_cfg={},
            device=device,
            collector_device=collector_device,
            num_envs=num_envs,
        )

        self.steps_per_env = steps_per_env
        self.replay_buffer_n = replay_buffer_n
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.updates_per_step = updates_per_step
        self.policy_delay = policy_delay
        self.exploration_noise = exploration_noise
        self.use_layer_norm = use_layer_norm
        self.gamma = gamma
        self.tau = tau
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.actor_hidden_dim = actor_hidden_dim
        self.critic_hidden_dim = critic_hidden_dim
        self.num_atoms = num_atoms

        # Auto-detect dims
        self.obs_dim, self.action_dim = self._detect_dims()

    def _detect_dims(self):
        from unilab.envs import registry
        from unilab.algos.torch.common.worker import ensure_registries
        ensure_registries()
        env = registry.make(self.env_name, num_envs=1, sim_backend="mujoco")
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        env.close()

        return obs_dim, action_dim

    def _build_learner(self) -> FastTD3Learner:
        return FastTD3Learner(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            device=self.device,
            gamma=self.gamma,
            tau=self.tau,
            actor_lr=self.actor_lr,
            critic_lr=self.critic_lr,
            actor_hidden_dim=self.actor_hidden_dim,
            critic_hidden_dim=self.critic_hidden_dim,
            num_atoms=self.num_atoms,
            use_layer_norm=self.use_layer_norm,
            exploration_noise=self.exploration_noise,
        )

    def _collector_fn(self, stop_event, **kwargs):
        off_policy_collector_fn(stop_event=stop_event, **kwargs)

    def learn(
        self,
        max_iterations: int = 1500,
        save_interval: int = 50,
        log_dir: str = "logs",
    ):
        os.makedirs(log_dir, exist_ok=True)

        learner = self._build_learner()

        buffer_capacity = self.replay_buffer_n * self.num_envs
        shared_buffer = SharedReplayBuffer(
            capacity=buffer_capacity,
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            create=True,
        )
        self._shared_resources.append(shared_buffer)

        weight_sync = SharedWeightSync.from_state_dict(
            learner.actor.state_dict(), create=True
        )
        self._shared_resources.append(weight_sync)

        metrics_queue = mp.Queue(maxsize=100)
        weight_param_shapes = {
            name: p.shape for name, p in learner.actor.state_dict().items()
        }

        collector_kwargs = {
            "env_name": self.env_name,
            "env_cfg_overrides": self.env_cfg_overrides,
            "num_envs": self.num_envs,
            "shm_buffer_name": shared_buffer.name,
            "buffer_capacity": buffer_capacity,
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "weight_sync_name": weight_sync.name,
            "weight_param_shapes": weight_param_shapes,
            "algo_type": "td3",
            "actor_hidden_dim": self.actor_hidden_dim,
            "use_layer_norm": self.use_layer_norm,
            "collector_device": self.collector_device,
            "exploration_noise": self.exploration_noise,
            "warmup_steps": self.warmup_steps,
            "metrics_queue": metrics_queue,
        }
        self._start_collector(
            target_fn=off_policy_collector_fn,
            kwargs={"stop_event": self._stop_event, **collector_kwargs},
        )

        logger = TrainingLogger(
            algo_name="FastTD3",
            max_iterations=max_iterations,
            num_envs=self.num_envs,
            env_name=self.env_name,
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
        )
        logger.start()

        reward_history = deque(maxlen=100)
        latest_reward_components = {}
        last_buf_log = 0

        for iteration in range(1, max_iterations + 1):
            iter_start = time.time()
            while shared_buffer.size < self.batch_size:
                if not self._check_collector_alive():
                    self._drain_metrics(metrics_queue, reward_history, latest_reward_components, logger)
                    logger.log_status("[red]ERROR: Collector process died. Exiting.[/]")
                    logger.finish()
                    return
                # Progress during buffer fill
                cur_size = shared_buffer.size
                if cur_size - last_buf_log >= self.num_envs * 10:
                    last_buf_log = cur_size
                    logger.log_buffer_fill(cur_size, self.batch_size)
                time.sleep(0.1)
                self._drain_metrics(metrics_queue, reward_history, latest_reward_components, logger)

            self._drain_metrics(metrics_queue, reward_history, latest_reward_components, logger)
            collect_time = time.time() - iter_start

            train_start = time.time()
            iter_metrics = defaultdict(list)
            for update_idx in range(self.updates_per_step):
                batch = shared_buffer.sample_torch(self.batch_size, self.device)

                critic_metrics = learner.update_critic(batch)
                for k, v in critic_metrics.items():
                    iter_metrics[k].append(v)

                if update_idx % self.policy_delay == 1:
                    actor_metrics = learner.update_actor(batch)
                    for k, v in actor_metrics.items():
                        iter_metrics[k].append(v)
                    learner.soft_update_targets()

            learner.update_count += 1
            weight_sync.write_weights(learner.actor.state_dict())
            train_time = time.time() - train_start

            avg_metrics = {k: statistics.mean(v) for k, v in iter_metrics.items() if v}
            mean_reward = statistics.mean(reward_history) if reward_history else 0.0

            logger.log_step(
                iteration=iteration,
                metrics=avg_metrics,
                reward=mean_reward,
                reward_components=latest_reward_components,
                collect_time=collect_time,
                train_time=train_time,
            )

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

    @staticmethod
    def _drain_metrics(queue, reward_history, reward_components, logger: TrainingLogger):
        while not queue.empty():
            try:
                m = queue.get_nowait()
                if "error" in m:
                    logger.log_status(f"[red]Collector ERROR: {m['error']}[/]")
                
                updated_rew = False
                if "mean_ep_reward" in m:
                    reward_history.append(m["mean_ep_reward"])
                    updated_rew = True
                
                if "reward_components" in m:
                    reward_components.clear()
                    reward_components.update(m["reward_components"])

                if "mean_ep_length" in m:
                    logger.update_ep_length(m["mean_ep_length"])

                if "total_steps" in m and "buffer_size" in m:
                    logger.log_collector(m["total_steps"], m["buffer_size"], m.get("mean_ep_reward", 0.0) if updated_rew else 0.0)

            except Exception:
                break
