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

from unilab.ipc import SharedReplayBuffer, SharedWeightSync
from unilab.algos.torch.common.async_runner import AsyncRunner
from unilab.algos.torch.common.worker import off_policy_collector_fn
from unilab.algos.torch.common.logger import TrainingLogger
from unilab.algos.torch.fast_sac.learner import FastSACLearner
from unilab.ipc.async_runner import _SPAWN_CTX


class FastSACRunner(AsyncRunner):
    """FastSAC async runner using shared memory (no Ray dependency).

    obs_dim and action_dim are auto-detected from the environment if not provided.
    """

    def __init__(
        self,
        env_name: str,
        device: str | None = None,
        collector_device: str | None = None,
        num_envs: int = 4096,
        replay_buffer_n: int = 1024,
        batch_size: int = 8192,
        warmup_steps: int = 0,
        updates_per_step: int = 8,
        policy_frequency: int = 4,
        # Collection/training synchronization
        sync_collection: bool = False,
        env_steps_per_sync: int = 1,
        # Holosoma-aligned defaults
        gamma: float = 0.97,
        tau: float = 0.125,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        alpha_lr: float = 3e-4,
        alpha_init: float = 0.001,
        target_entropy_ratio: float = 1.0,
        actor_hidden_dim: int = 512,
        critic_hidden_dim: int = 768,
        num_atoms: int = 101,
        use_layer_norm: bool = True,
        max_grad_norm: float = 0.0,
    ):
        super().__init__(
            env_name=env_name,
            env_cfg_overrides={},
            rl_cfg={},
            device=device,
            collector_device=collector_device,
            num_envs=num_envs,
        )

        self.replay_buffer_n = replay_buffer_n
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.updates_per_step = updates_per_step
        self.policy_frequency = policy_frequency
        self.use_layer_norm = use_layer_norm
        self.sync_collection = sync_collection
        self.env_steps_per_sync = env_steps_per_sync

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
        self.max_grad_norm = max_grad_norm

        # Auto-detect obs/action dim from env
        self.obs_dim, self.action_dim = self._detect_dims()

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
            use_layer_norm=self.use_layer_norm,
            max_grad_norm=self.max_grad_norm,
        )

    def _get_default_device(self) -> str:
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _collector_fn(self, stop_event, **kwargs):
        off_policy_collector_fn(stop_event=stop_event, **kwargs)

    def learn(
        self,
        max_iterations: int = 1500,
        save_interval: int = 50,
        log_dir: str = "logs",
        logger_type: str = "tensorboard",
    ):
        """Main training loop."""
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

        # Coordinator for synchronized collection/training
        collection_ready_queue = None
        trainer_done_queue = None
        if self.sync_collection:
            collection_ready_queue = _SPAWN_CTX.Queue(maxsize=1)
            trainer_done_queue = _SPAWN_CTX.Queue(maxsize=1)
            trainer_done_queue.put(1)  # Initial signal
            print(f"[Runner] Collection sync enabled: env_steps_per_sync={self.env_steps_per_sync}")

        metrics_queue = _SPAWN_CTX.Queue(maxsize=100)

        weight_param_shapes = {
            name: p.shape for name, p in learner.actor.state_dict().items()
        }

        collector_kwargs = {
            "env_name": self.env_name,
            "num_envs": self.num_envs,
            "shm_buffer_name": shared_buffer.name,
            "buffer_capacity": buffer_capacity,
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "weight_sync_name": weight_sync.name,
            "weight_sync_lock": weight_sync._lock,
            "weight_param_shapes": weight_param_shapes,
            "algo_type": "sac",
            "actor_hidden_dim": self.actor_hidden_dim,
            "use_layer_norm": self.use_layer_norm,
            "collector_device": self.collector_device,
            "warmup_steps": self.warmup_steps,
            "metrics_queue": metrics_queue,
            "buffer_lock": shared_buffer._lock,
            "sync_collection": self.sync_collection,
            "collection_ready_queue": collection_ready_queue,
            "trainer_done_queue": trainer_done_queue,
            "env_steps_per_sync": self.env_steps_per_sync,
        }
        self._start_collector(
            target_fn=off_policy_collector_fn,
            kwargs={"stop_event": self._stop_event, **collector_kwargs},
        )

        # Check if collector started
        import time
        time.sleep(0.5)
        if self._collector_process is not None:
            print(f"[Runner] Collector process alive: {self._collector_process.is_alive()}")
            if not self._collector_process.is_alive():
                print(f"[Runner] Collector process exit code: {self._collector_process.exitcode}")
        else:
            print("[Runner] Collector process is None!")

        logger = TrainingLogger(
            algo_name="FastSAC",
            max_iterations=max_iterations,
            num_envs=self.num_envs,
            env_name=self.env_name,
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            log_dir=log_dir,
            log_backend=logger_type,
        )
        logger.set_collection_sync(self.sync_collection, self.env_steps_per_sync)
        logger.start()

        reward_history = deque(maxlen=100)
        latest_reward_components = {}
        last_buf_log = 0
        write_read_ema = 0.0  # EMA of write/consume ratio

        for iteration in range(1, max_iterations + 1):
            iter_start = time.time()

            # Synchronized collection: wait for the collector to gather the next chunk.
            if self.sync_collection and collection_ready_queue is not None:
                collection_ready_queue.get()
            else:
                # Async mode: wait for enough data, checking collector health
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
            ptr_before = shared_buffer.ptr
            # Sample once for all updates, then slice on GPU — avoids N×CPU→GPU transfers
            large_batch = shared_buffer.sample_torch(self.batch_size * self.updates_per_step, self.device)
            for update_idx in range(self.updates_per_step):
                s = update_idx * self.batch_size
                e = s + self.batch_size
                batch = {k: v[s:e] for k, v in large_batch.items()}
                critic_metrics = learner.update_critic(batch)
                for k, v in critic_metrics.items():
                    iter_metrics[k].append(v)

                if update_idx % self.policy_frequency == 1:
                    actor_metrics = learner.update_actor(batch)
                    for k, v in actor_metrics.items():
                        iter_metrics[k].append(v)

                learner.soft_update_target()

            learner.update_count += 1
            weight_sync.write_weights(learner.actor.state_dict())
            train_time = time.time() - train_start

            # Let collection resume after the learner finishes this phase.
            if self.sync_collection and trainer_done_queue is not None:
                trainer_done_queue.put(1)

            write_delta = shared_buffer.ptr - ptr_before
            consume = self.batch_size * self.updates_per_step
            write_read_ema = 0.9 * write_read_ema + 0.1 * (write_delta / max(consume, 1))
            logger.update_buffer_utilization(write_read_ema)

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
        """Check if the collector subprocess is still running."""
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

                if "collector_timing_ms" in m:
                    logger.update_collector_timing(m["collector_timing_ms"])

                if "timeout_rate" in m or "terminated_rate" in m:
                    logger.update_done_rates(
                        timeout_rate=float(m.get("timeout_rate", 0.0)),
                        terminated_rate=float(m.get("terminated_rate", 0.0)),
                    )

                if "total_steps" in m and "buffer_size" in m:
                    logger.log_collector(m["total_steps"], m["buffer_size"], m.get("mean_ep_reward", 0.0) if updated_rew else 0.0)

            except Exception:
                break
