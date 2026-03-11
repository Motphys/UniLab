"""Unified runner for off-policy RL algorithms (SAC, TD3)."""

import os
import time
import torch
from collections import deque

from unilab.ipc import SharedReplayBuffer, SharedWeightSync, SharedObsNormStats
from unilab.ipc.async_runner import AsyncRunner
from unilab.algos.torch.offpolicy.worker import off_policy_collector_fn
from unilab.utils.offpolicy_logger import OffPolicyLogger
from unilab.ipc.async_runner import _SPAWN_CTX


class OffPolicyRunner(AsyncRunner):
    """Unified runner for SAC and TD3."""

    def __init__(
        self,
        learner,
        env_name: str,
        algo_type: str,  # "sac" or "td3"
        num_envs: int = 4096,
        replay_buffer_n: int = 1024,
        batch_size: int = 8192,
        warmup_steps: int = 0,
        updates_per_step: int = 8,
        policy_frequency: int = 4,
        sync_collection: bool = True,
        env_steps_per_sync: int = 1,
        device: str = None,
        collector_device: str = None,
        actor_hidden_dim: int = 512,
        use_layer_norm: bool = True,
        obs_normalization: bool = False,
        sim_backend: str = "mujoco",
        use_gpu_buffer: bool = True,
    ):
        super().__init__(
            env_name=env_name,
            env_cfg_overrides={},
            rl_cfg={},
            device=device,
            collector_device=collector_device or "cpu",
            num_envs=num_envs,
            sim_backend=sim_backend,
        )

        self.learner = learner
        self.algo_type = algo_type
        self.replay_buffer_n = replay_buffer_n
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.updates_per_step = updates_per_step
        self.policy_frequency = policy_frequency
        self.sync_collection = sync_collection
        self.env_steps_per_sync = env_steps_per_sync
        self.actor_hidden_dim = actor_hidden_dim
        self.use_layer_norm = use_layer_norm
        self.obs_normalization = obs_normalization
        self.use_gpu_buffer = use_gpu_buffer and device != "cpu"

        self.obs_dim, self.action_dim = self._detect_dims()

    def _detect_dims(self):
        from unilab.envs import registry
        from unilab.utils.algo_utils import ensure_registries
        ensure_registries()
        env = registry.make(self.env_name, num_envs=1, sim_backend="mujoco")
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        env.close()
        return obs_dim, action_dim

    def _get_default_device(self) -> str:
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _build_learner(self):
        return self.learner

    def _collector_fn(self, stop_event, **kwargs):
        off_policy_collector_fn(stop_event=stop_event, **kwargs)

    def learn(self, max_iterations: int = 1500, save_interval: int = 50,
              log_dir: str = "logs", logger_type: str = "tensorboard"):
        """Unified training loop for off-policy algorithms."""
        os.makedirs(log_dir, exist_ok=True)

        # Setup shared buffer
        buffer_capacity = self.replay_buffer_n * self.num_envs
        shared_buffer = SharedReplayBuffer(
            capacity=buffer_capacity,
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            create=True,
        )
        self._shared_resources.append(shared_buffer)

        # Setup GPU buffer if enabled
        gpu_buffer = None
        if self.use_gpu_buffer:
            from unilab.ipc import GPUReplayBuffer
            gpu_buffer = GPUReplayBuffer(
                capacity=buffer_capacity,
                obs_dim=self.obs_dim,
                action_dim=self.action_dim,
                device=self.device,
            )
            print(f"[Runner] GPU buffer enabled on {self.device}")

        # Setup weight sync
        weight_sync = SharedWeightSync.from_state_dict(
            self.learner.actor.state_dict(), create=True
        )
        self._shared_resources.append(weight_sync)

        # Setup sync queues
        collection_ready_queue = None
        trainer_done_queue = None
        if self.sync_collection:
            collection_ready_queue = _SPAWN_CTX.Queue(maxsize=1)
            trainer_done_queue = _SPAWN_CTX.Queue(maxsize=1)
            trainer_done_queue.put(1)
            print(f"[Runner] Collection sync enabled: env_steps_per_sync={self.env_steps_per_sync}")

        metrics_queue = _SPAWN_CTX.Queue(maxsize=100)

        # Setup obs normalization
        shared_obs_normalizer_stats = None
        if self.obs_normalization:
            shared_obs_normalizer_stats = SharedObsNormStats(_SPAWN_CTX)

        # Start collector
        weight_param_shapes = {k: v.shape for k, v in self.learner.actor.state_dict().items()}
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
            "algo_type": self.algo_type,
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
            "obs_normalization": self.obs_normalization,
            "shared_obs_normalizer_stats": shared_obs_normalizer_stats,
        }
        self._start_collector(
            target_fn=off_policy_collector_fn,
            kwargs={"stop_event": self._stop_event, **collector_kwargs},
        )

        time.sleep(0.5)
        if self._collector_process:
            print(f"[Runner] Collector process alive: {self._collector_process.is_alive()}")

        # Setup logger
        logger = OffPolicyLogger(
            algo_name=f"Fast{self.algo_type.upper()}",
            max_iterations=max_iterations,
            num_envs=self.num_envs,
            env_name=self.env_name,
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            log_dir=log_dir,
            log_backend=logger_type,
        )
        logger.set_collection_sync(self.sync_collection, self.env_steps_per_sync)
        if hasattr(self.learner, 'use_symmetry') and self.learner.use_symmetry:
            logger.log_status("Symmetry augmentation: enabled")
        logger.start()

        reward_history = deque(maxlen=100)
        latest_reward_components = {}
        last_buf_log = 0
        write_read_ema = 0.0

        # Training loop
        for iteration in range(1, max_iterations + 1):
            iter_start = time.time()

            # Wait for data
            if self.sync_collection and collection_ready_queue:
                import queue
                while True:
                    try:
                        collection_ready_queue.get(timeout=1.0)
                        break
                    except queue.Empty:
                        if not self._check_collector_alive():
                            self._drain_metrics(metrics_queue, reward_history, latest_reward_components, logger)
                            logger.log_status("[red]ERROR: Collector died[/]")
                            logger.finish()
                            return
            else:
                while shared_buffer.size < self.batch_size:
                    if not self._check_collector_alive():
                        self._drain_metrics(metrics_queue, reward_history, latest_reward_components, logger)
                        logger.log_status("[red]ERROR: Collector died[/]")
                        logger.finish()
                        return
                    cur_size = shared_buffer.size
                    if cur_size - last_buf_log >= self.num_envs * 10:
                        last_buf_log = cur_size
                        logger.log_buffer_fill(cur_size, self.batch_size)
                    time.sleep(0.1)
                    self._drain_metrics(metrics_queue, reward_history, latest_reward_components, logger)

            self._drain_metrics(metrics_queue, reward_history, latest_reward_components, logger)
            collect_time = time.time() - iter_start

            train_start = time.time()
            from collections import defaultdict
            iter_metrics = defaultdict(list)
            ptr_before = shared_buffer.ptr

            # Local variable for faster access in hot loop
            learner = self.learner

            # Sample from GPU buffer or host buffer
            if gpu_buffer is not None:
                # Sync new data before sampling
                gpu_buffer.sync_new_data(shared_buffer)

                if gpu_buffer.size >= self.batch_size * self.updates_per_step:
                    large_batch = gpu_buffer.sample(self.batch_size * self.updates_per_step)
                else:
                    large_batch = shared_buffer.sample_torch(self.batch_size * self.updates_per_step, self.device)
            else:
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

            if self.obs_normalization and getattr(self.learner, "obs_normalizer", None) is not None:
                shared_obs_normalizer_stats.put((self.learner.obs_normalizer.mean.cpu().numpy(), self.learner.obs_normalizer.std.cpu().numpy()))

            self.learner.update_count += 1
            weight_sync.write_weights(self.learner.actor.state_dict())
            train_time = time.time() - train_start

            if self.sync_collection and trainer_done_queue:
                trainer_done_queue.put(1)

            write_delta = shared_buffer.ptr - ptr_before
            consume = self.batch_size * self.updates_per_step
            write_read_ema = 0.9 * write_read_ema + 0.1 * (write_delta / max(consume, 1))
            logger.update_buffer_utilization(write_read_ema)

            import statistics
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
                torch.save(self.learner.get_state_dict(), ckpt_path)
                logger.log_save(ckpt_path)

        ckpt_path = os.path.join(log_dir, f"model_{max_iterations}.pt")
        torch.save(self.learner.get_state_dict(), ckpt_path)
        logger.log_save(ckpt_path)
        logger.finish()

    def _check_collector_alive(self) -> bool:
        if self._collector_process is not None and not self._collector_process.is_alive():
            return False
        return True

    @staticmethod
    def _drain_metrics(queue, reward_history, reward_components, logger):
        while not queue.empty():
            try:
                m = queue.get_nowait()
                if "error" in m:
                    logger.log_status(f"[red]Collector ERROR: {m['error']}[/]")
                    raise RuntimeError(f"Collector process failed: {m['error']}")

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
