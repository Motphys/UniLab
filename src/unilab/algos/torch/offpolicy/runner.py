"""Unified runner for off-policy RL algorithms (SAC, TD3)."""

import os
import statistics
import sys
import time
from collections import deque

import torch

from unilab.algos.torch.offpolicy.worker import off_policy_collector_fn
from unilab.ipc import SharedObsNormStats, SharedWeightSync
from unilab.ipc.async_runner import _SPAWN_CTX, AsyncRunner
from unilab.ipc.replay_buffer import ReplayBuffer
from unilab.utils.device_utils import get_default_device, get_env_dims
from unilab.utils.offpolicy_logger import OffPolicyLogger


class OffPolicyRunner(AsyncRunner):
    """Unified runner for SAC and TD3."""

    def __init__(
        self,
        learner,
        env_name: str,
        algo_type: str,  # "sac", "td3", or "flashsac"
        num_envs: int = 4096,
        replay_buffer_n: int = 1024,
        batch_size: int = 8192,
        warmup_steps: int = 0,
        updates_per_step: int = 8,
        policy_frequency: int = 4,
        sync_collection: bool = True,
        env_steps_per_sync: int = 1,
        device: str | None = None,
        actor_hidden_dim: int = 512,
        use_layer_norm: bool = True,
        obs_normalization: bool = False,
        sim_backend: str = "mujoco",
        env_cfg_override: dict | None = None,
        actor_kwargs: dict | None = None,
    ):
        super().__init__(
            env_name=env_name,
            env_cfg_overrides={},
            rl_cfg={},
            device=device,
            collector_device="cpu",
            num_envs=num_envs,
            sim_backend=sim_backend,
        )

        self.learner = learner
        self.env_cfg_override = env_cfg_override
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
        self.actor_kwargs = actor_kwargs or {}

        self.obs_dim, self.action_dim, self.privileged_dim = get_env_dims(
            self.env_name, sim_backend, env_cfg_override
        )

    def _get_default_device(self) -> str:
        return get_default_device()

    def _build_learner(self):
        return self.learner

    def _collector_fn(self, stop_event, **kwargs):
        off_policy_collector_fn(stop_event=stop_event, **kwargs)

    @staticmethod
    def _read_recent_replay_field(
        replay_buffer, field_name: str, start_ptr: int, count: int
    ) -> torch.Tensor:
        idx = start_ptr % replay_buffer.capacity

        if hasattr(replay_buffer, field_name):
            source = getattr(replay_buffer, field_name)
        else:
            packed_key = {
                "rewards": "_rew_col",
                "dones": "_done_col",
                "truncated": "_trunc_col",
            }[field_name]
            source = replay_buffer._storage[:, getattr(replay_buffer, packed_key)]

        if idx + count <= replay_buffer.capacity:
            return source[idx : idx + count].clone()

        split = replay_buffer.capacity - idx
        return torch.cat([source[idx:], source[: count - split]], dim=0).clone()

    def _update_reward_stats_from_replay(self, replay_buffer, start_ptr: int, end_ptr: int) -> int:
        if not hasattr(self.learner, "update_reward_stats"):
            return end_ptr
        if getattr(self.learner, "reward_normalizer", None) is None:
            return end_ptr

        count = end_ptr - start_ptr
        if count <= 0:
            return end_ptr
        if count > replay_buffer.capacity:
            count = replay_buffer.capacity
            start_ptr = end_ptr - count
        if count % self.num_envs != 0:
            count -= count % self.num_envs
            start_ptr = end_ptr - count
        if count <= 0:
            return end_ptr

        rewards = self._read_recent_replay_field(replay_buffer, "rewards", start_ptr, count)
        dones = self._read_recent_replay_field(replay_buffer, "dones", start_ptr, count)
        truncated = self._read_recent_replay_field(replay_buffer, "truncated", start_ptr, count)
        num_steps = count // self.num_envs
        self.learner.update_reward_stats(
            rewards.view(num_steps, self.num_envs),
            dones.view(num_steps, self.num_envs),
            truncated.view(num_steps, self.num_envs),
        )
        return end_ptr

    def learn(
        self,
        max_iterations: int = 1500,
        save_interval: int = 50,
        log_dir: str = "logs",
        logger_type: str = "tensorboard",
    ) -> None:
        """Unified training loop for off-policy algorithms."""
        os.makedirs(log_dir, exist_ok=True)
        train_start_wall = time.time()
        best_mean_reward = float("-inf")
        last_mean_reward = 0.0
        ckpt_path: str | None = None
        iteration = 0

        # Setup replay buffer
        buffer_capacity = self.replay_buffer_n * self.num_envs
        replay_buffer = ReplayBuffer(
            capacity=buffer_capacity,
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            device=self.device,
            privileged_dim=self.privileged_dim,
        )
        self._shared_resources.append(replay_buffer)

        # Setup weight sync
        weight_sync = SharedWeightSync.from_state_dict(self.learner.actor.state_dict(), create=True)
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
            "replay_buffer": replay_buffer,
            "weight_sync_name": weight_sync.name,
            "weight_sync_lock": weight_sync._lock,
            "weight_param_shapes": weight_param_shapes,
            "algo_type": self.algo_type,
            "actor_hidden_dim": self.actor_hidden_dim,
            "use_layer_norm": self.use_layer_norm,
            "warmup_steps": self.warmup_steps,
            "metrics_queue": metrics_queue,
            "sync_collection": self.sync_collection,
            "collection_ready_queue": collection_ready_queue,
            "trainer_done_queue": trainer_done_queue,
            "env_steps_per_sync": self.env_steps_per_sync,
            "obs_normalization": self.obs_normalization,
            "shared_obs_normalizer_stats": shared_obs_normalizer_stats,
            "env_cfg_override": self.env_cfg_override,
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "actor_kwargs": self.actor_kwargs,
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
            algo_name=(
                "FlashSAC" if self.algo_type == "flashsac" else f"Fast{self.algo_type.upper()}"
            ),
            max_iterations=max_iterations,
            num_envs=self.num_envs,
            env_name=self.env_name,
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            log_dir=log_dir,
            log_backend=logger_type,
        )
        logger.set_collection_sync(self.sync_collection, self.env_steps_per_sync)
        if hasattr(self.learner, "use_symmetry") and self.learner.use_symmetry:
            logger.log_status("Symmetry augmentation: enabled")
        logger.start()

        reward_history: deque = deque(maxlen=100)
        latest_reward_components: dict[str, float] = {}
        last_buf_log = 0
        write_read_ema = 0.0
        reward_stats_ptr = 0

        # Training loop
        for iteration in range(1, max_iterations + 1):
            iter_start = time.time()

            # Wait for data
            wait_start = time.time()
            if self.sync_collection and collection_ready_queue:
                import queue

                while True:
                    try:
                        collection_ready_queue.get(timeout=1.0)
                        break
                    except queue.Empty:
                        if not self._check_collector_alive():
                            self._drain_metrics(
                                metrics_queue, reward_history, latest_reward_components, logger
                            )
                            logger.log_status("[red]ERROR: Collector died[/]")
                            logger.finish()
                            summary = {
                                "status": "collector_died",
                                "completed_iterations": iteration,
                                "total_env_steps": int(logger._total_steps),
                                "final_mean_reward": None,
                                "best_mean_reward": None,
                                "mean_episode_length": float(logger._mean_ep_length),
                                "last_checkpoint": ckpt_path,
                                "training_wall_time_sec": time.time() - train_start_wall,
                            }
                            self.last_run_summary = summary
                            return
            else:
                while int(replay_buffer.size[0]) < self.batch_size:
                    if not self._check_collector_alive():
                        self._drain_metrics(
                            metrics_queue, reward_history, latest_reward_components, logger
                        )
                        logger.log_status("[red]ERROR: Collector died[/]")
                        logger.finish()
                        summary = {
                            "status": "collector_died",
                            "completed_iterations": iteration,
                            "total_env_steps": int(logger._total_steps),
                            "final_mean_reward": None,
                            "best_mean_reward": None,
                            "mean_episode_length": float(logger._mean_ep_length),
                            "last_checkpoint": ckpt_path,
                            "training_wall_time_sec": time.time() - train_start_wall,
                        }
                        self.last_run_summary = summary
                        return
                    cur_size = int(replay_buffer.size[0])
                    if cur_size - last_buf_log >= self.num_envs * 10:
                        last_buf_log = cur_size
                        logger.log_buffer_fill(cur_size, self.batch_size)
                    time.sleep(0.1)
                    self._drain_metrics(
                        metrics_queue, reward_history, latest_reward_components, logger
                    )

            wait_time = time.time() - wait_start
            self._drain_metrics(metrics_queue, reward_history, latest_reward_components, logger)
            collect_time = time.time() - iter_start
            reward_stats_ptr = self._update_reward_stats_from_replay(
                replay_buffer,
                reward_stats_ptr,
                int(replay_buffer.ptr[0]),
            )

            train_start = time.time()
            from collections import defaultdict

            iter_metrics = defaultdict(list)
            ptr_before = int(replay_buffer.ptr[0])

            # Local variable for faster access in hot loop
            learner = self.learner

            # Sample from torch buffer (zero-copy on CUDA/MPS)
            large_batch = replay_buffer.sample(self.batch_size * self.updates_per_step)

            for update_idx in range(self.updates_per_step):
                s = update_idx * self.batch_size
                e = s + self.batch_size
                batch = {k: v[s:e] for k, v in large_batch.items()}

                critic_metrics = learner.update_critic(batch)
                for k, v in critic_metrics.items():
                    iter_metrics[k].append(v)

                actor_updated = False
                if update_idx % self.policy_frequency == 0:
                    actor_metrics = learner.update_actor(batch)
                    for k, v in actor_metrics.items():
                        iter_metrics[k].append(v)
                    actor_updated = True

                if self.algo_type == "td3":
                    if actor_updated:
                        learner.soft_update_target()
                else:
                    learner.soft_update_target()

            if self.obs_normalization and getattr(self.learner, "obs_normalizer", None) is not None:
                assert shared_obs_normalizer_stats is not None
                shared_obs_normalizer_stats.put(
                    (
                        self.learner.obs_normalizer.mean.cpu().numpy(),
                        self.learner.obs_normalizer.std.cpu().numpy(),
                    )
                )

            self.learner.update_count += 1
            weight_sync.write_weights(self.learner.actor.state_dict())
            train_time = time.time() - train_start

            if self.sync_collection and trainer_done_queue:
                trainer_done_queue.put(1)

            write_delta = int(replay_buffer.ptr[0]) - ptr_before
            consume = self.batch_size * self.updates_per_step
            write_read_ema = 0.9 * write_read_ema + 0.1 * (write_delta / max(consume, 1))
            logger.update_buffer_utilization(write_read_ema)

            avg_metrics = {k: statistics.mean(v) for k, v in iter_metrics.items() if v}
            mean_reward = statistics.mean(reward_history) if reward_history else 0.0
            last_mean_reward = float(mean_reward)
            best_mean_reward = max(best_mean_reward, last_mean_reward)

            logger.log_step(
                iteration=iteration,
                metrics=avg_metrics,
                reward=mean_reward,
                reward_components=latest_reward_components,
                collect_time=collect_time,
                train_time=train_time,
                wait_time=wait_time,
            )

            if save_interval > 0 and iteration % save_interval == 0:
                ckpt_path = os.path.join(log_dir, f"model_{iteration}.pt")
                torch.save(self.learner.get_state_dict(), ckpt_path)
                logger.log_save(ckpt_path)

        ckpt_path = os.path.join(log_dir, f"model_{max_iterations}.pt")
        torch.save(self.learner.get_state_dict(), ckpt_path)
        logger.log_save(ckpt_path)
        logger.finish()
        summary = {
            "status": "completed",
            "completed_iterations": iteration,
            "total_env_steps": int(logger._total_steps),
            "final_mean_reward": last_mean_reward if reward_history else None,
            "best_mean_reward": best_mean_reward if reward_history else None,
            "mean_episode_length": float(logger._mean_ep_length),
            "last_checkpoint": ckpt_path,
            "training_wall_time_sec": time.time() - train_start_wall,
        }
        self.last_run_summary = summary

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
                    logger.log_collector(
                        m["total_steps"],
                        m["buffer_size"],
                        m.get("mean_ep_reward", 0.0) if updated_rew else 0.0,
                    )

            except Exception as e:
                print(f"[OffPolicyRunner] metrics drain error: {e}", file=sys.stderr)
                break
