"""FastSAC Runner — async training with native multiprocessing (no Ray).

Pipeline:
  1. Collector process continuously collects transitions → SharedReplayBuffer
  2. Learner process samples from buffer and trains on device (MPS/GPU)
  3. Periodically sync actor weights to collector
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
from unilab.algos.torch.fast_sac.learner import FastSACLearner
from unilab.utils.rsl_rl_compat import convert_config_v3_to_v4, is_rsl_rl_v4


class FastSACRunner(AsyncRunner):
    """FastSAC async runner using shared memory (no Ray dependency)."""

    def __init__(
        self,
        env_name: str,
        env_cfg_overrides: dict,
        rl_cfg: dict,
        device: str | None = None,
        collector_device: str | None = None,
        num_envs: int = 4096,
        steps_per_env: int = 24,
        replay_buffer_n: int = 1024,
        batch_size: int = 8192,
        warmup_steps: int = 5000,
        updates_per_step: int = 8,
        policy_frequency: int = 4,
        # Holosoma-aligned defaults
        gamma: float = 0.97,
        tau: float = 0.125,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        alpha_lr: float = 3e-4,
        alpha_init: float = 0.001,
        target_entropy_ratio: float = 0.0,
        actor_hidden_dim: int = 512,
        critic_hidden_dim: int = 768,
        num_atoms: int = 101,
        exploration_noise: float = 0.1,
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
        self.replay_buffer_n = replay_buffer_n
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.updates_per_step = updates_per_step
        self.policy_frequency = policy_frequency
        self.exploration_noise = exploration_noise

        # Learner hyperparameters
        self.gamma = gamma
        self.tau = tau
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.alpha_lr = alpha_lr
        self.alpha_init = alpha_init
        self.target_entropy_ratio = target_entropy_ratio
        self.actor_hidden_dim = actor_hidden_dim
        self.critic_hidden_dim = critic_hidden_dim
        self.num_atoms = num_atoms

        # Determine obs/action dims from config
        self._resolve_dims()

    def _resolve_dims(self):
        """Resolve obs_dim and action_dim from the RL config."""
        cfg = dict(self.rl_cfg)
        if is_rsl_rl_v4():
            cfg = convert_config_v3_to_v4(cfg)

        # Extract dims from config's obs_groups
        obs_groups = cfg.get("obs_groups", {})
        actor_group = obs_groups.get("actor", obs_groups.get("policy", {}))

        if isinstance(actor_group, dict):
            self.obs_dim = sum(v for v in actor_group.values() if isinstance(v, int))
        else:
            self.obs_dim = actor_group

        # Action dim from actor config
        actor_cfg = cfg.get("actor", {})
        self.action_dim = actor_cfg.get("output_dim", actor_cfg.get("num_actions", 12))

    def _build_learner(self) -> FastSACLearner:
        return FastSACLearner(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            device=self.device,
            gamma=self.gamma,
            tau=self.tau,
            actor_lr=self.actor_lr,
            critic_lr=self.critic_lr,
            alpha_lr=self.alpha_lr,
            alpha_init=self.alpha_init,
            target_entropy_ratio=self.target_entropy_ratio,
            actor_hidden_dim=self.actor_hidden_dim,
            critic_hidden_dim=self.critic_hidden_dim,
            num_atoms=self.num_atoms,
        )

    def _collector_fn(self, stop_event, **kwargs):
        """Collector subprocess entry point."""
        off_policy_collector_fn(stop_event=stop_event, **kwargs)

    def learn(
        self,
        max_iterations: int = 1500,
        save_interval: int = 50,
        log_dir: str = "logs",
    ):
        """Main training loop."""
        os.makedirs(log_dir, exist_ok=True)

        # Build learner
        learner = self._build_learner()

        # Create shared replay buffer
        buffer_capacity = self.replay_buffer_n * self.num_envs
        shared_buffer = SharedReplayBuffer(
            capacity=buffer_capacity,
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            create=True,
        )
        self._shared_resources.append(shared_buffer)

        # Create shared weight sync
        weight_sync = SharedWeightSync.from_state_dict(
            learner.actor.state_dict(), create=True
        )
        self._shared_resources.append(weight_sync)

        # Metrics queue
        metrics_queue = mp.Queue(maxsize=100)

        # Resolve weight param shapes for collector
        weight_param_shapes = {
            name: p.shape for name, p in learner.actor.state_dict().items()
        }

        # Start collector process
        collector_kwargs = {
            "env_name": self.env_name,
            "env_cfg_overrides": self.env_cfg_overrides,
            "rl_cfg": self.rl_cfg,
            "num_envs": self.num_envs,
            "steps_per_env": self.steps_per_env,
            "shm_buffer_name": shared_buffer.name,
            "buffer_capacity": buffer_capacity,
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "weight_sync_name": weight_sync.name,
            "weight_param_shapes": weight_param_shapes,
            "collector_device": self.collector_device,
            "exploration_noise": self.exploration_noise,
            "warmup_steps": self.warmup_steps,
            "metrics_queue": metrics_queue,
            "algo_type": "sac",
        }
        self._start_collector(
            target_fn=off_policy_collector_fn,
            kwargs={"stop_event": self._stop_event, **collector_kwargs},
        )

        # --- Training loop ---
        print(f"FastSAC training: waiting for buffer to fill (warmup={self.warmup_steps})...")

        reward_history = deque(maxlen=100)
        log_data = defaultdict(list)
        start_time = time.time()

        for iteration in range(1, max_iterations + 1):
            # Wait for enough data
            while shared_buffer.size < self.batch_size:
                time.sleep(0.1)
                # Check collector metrics
                while not metrics_queue.empty():
                    try:
                        m = metrics_queue.get_nowait()
                        if "mean_ep_reward" in m:
                            reward_history.append(m["mean_ep_reward"])
                    except Exception:
                        break

            # Process any pending metrics
            while not metrics_queue.empty():
                try:
                    m = metrics_queue.get_nowait()
                    if "mean_ep_reward" in m:
                        reward_history.append(m["mean_ep_reward"])
                except Exception:
                    break

            # Training updates
            iter_metrics = defaultdict(list)
            for update_idx in range(self.updates_per_step):
                batch = shared_buffer.sample_torch(self.batch_size, self.device)

                # Critic update
                critic_metrics = learner.update_critic(batch)
                for k, v in critic_metrics.items():
                    iter_metrics[k].append(v)

                # Actor update (delayed)
                if self.updates_per_step > 1:
                    if update_idx % self.policy_frequency == 1:
                        actor_metrics = learner.update_actor(batch)
                        for k, v in actor_metrics.items():
                            iter_metrics[k].append(v)
                elif iteration % self.policy_frequency == 0:
                    actor_metrics = learner.update_actor(batch)
                    for k, v in actor_metrics.items():
                        iter_metrics[k].append(v)

                # Soft update target
                learner.soft_update_target()

            learner.update_count += 1

            # Sync weights to collector
            weight_sync.write_weights(learner.actor.state_dict())

            # Logging
            if iteration % 10 == 0:
                elapsed = time.time() - start_time
                avg_metrics = {k: statistics.mean(v) for k, v in iter_metrics.items() if v}
                mean_reward = statistics.mean(reward_history) if reward_history else 0.0

                print(
                    f"[{iteration}/{max_iterations}] "
                    f"t={elapsed:.0f}s | "
                    f"buf={shared_buffer.size} | "
                    f"rew={mean_reward:.3f} | "
                    f"q_loss={avg_metrics.get('qf_loss', 0):.3f} | "
                    f"a_loss={avg_metrics.get('actor_loss', 0):.3f} | "
                    f"alpha={avg_metrics.get('alpha', 0):.4f}"
                )

            # Save checkpoint
            if save_interval > 0 and iteration % save_interval == 0:
                ckpt_path = os.path.join(log_dir, f"model_{iteration}.pt")
                torch.save(learner.get_state_dict(), ckpt_path)
                print(f"Saved checkpoint: {ckpt_path}")

        # Final save
        ckpt_path = os.path.join(log_dir, f"model_{max_iterations}.pt")
        torch.save(learner.get_state_dict(), ckpt_path)
        print(f"Training complete. Final checkpoint: {ckpt_path}")
