"""Off-policy runner using CPU-pinned double-buffer replay pipeline (B path)."""

from __future__ import annotations

import os
import statistics
import sys
import time
from collections import defaultdict, deque
from contextlib import nullcontext
from pathlib import Path

import torch

from unilab.algos.torch.offpolicy.runner import (
    OffPolicyRunner,
    compute_train_start_threshold,
    replay_buffer_ready_for_learning,
)
from unilab.algos.torch.offpolicy.worker import off_policy_collector_fn
from unilab.ipc import SharedObsNormStats, SharedWeightSync
from unilab.ipc.async_runner import _SPAWN_CTX
from unilab.ipc.replay_buffer import ReplayBuffer
from unilab.ipc.replay_pipelines.cpu_pinned_double_buffer import (
    CPUPinnedDoubleBufferReplayPipeline,
)
from unilab.logging import OffPolicyLogger, TraceRecorder
from unilab.training.seed import derive_worker_seed


class DoubleBufferOffPolicyRunner(OffPolicyRunner):
    """OffPolicyRunner variant that uses CPUPinnedDoubleBufferReplayPipeline.

    The only behavioural difference from the parent class is in learn():
    - ReplayBuffer is created with defer_gpu=True (no GPU cache allocation).
    - Sampling goes through CPUPinnedDoubleBufferReplayPipeline instead of
      ReplayBuffer.sample().
    """

    LEARNER_LOG_INTERVAL = 10

    def __init__(
        self,
        *,
        replay_prefetch_mode: str = "one_tick",
        verbose_metrics: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if replay_prefetch_mode != "one_tick":
            raise ValueError(
                "DoubleBufferOffPolicyRunner only supports replay_prefetch_mode='one_tick'"
            )
        self.replay_prefetch_mode = replay_prefetch_mode
        self.verbose_metrics = bool(verbose_metrics)
        self.replay_pack_layout = "packed"
        self.replay_pack_executor = "collector_thread"
        self.replay_h2d_submitter = "auto"

    def learn(
        self,
        max_iterations: int = 1500,
        save_interval: int = 50,
        log_dir: str = "logs",
        logger_type: str = "tensorboard",
    ) -> None:
        os.makedirs(log_dir, exist_ok=True)
        trace_output_path = None
        trace_recorder: TraceRecorder | None = None
        if self.trace_enabled:
            trace_root = Path(self.trace_output_dir or log_dir)
            trace_output_path = trace_root / "perfetto_offpolicy_timeline.json"
            trace_recorder = TraceRecorder("offpolicy_learner")
        train_start_wall = time.time()
        best_mean_reward = float("-inf")
        last_mean_reward = 0.0
        ckpt_path: str | None = None
        iteration = 0

        # --- replay buffer (no GPU cache) ---
        buffer_capacity = self.replay_buffer_n * self.num_envs
        replay_buffer = ReplayBuffer(
            capacity=buffer_capacity,
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            device=self.device,
            defer_gpu=True,
            critic_dim=self.critic_obs_dim,
            packed_cpu_storage=self.replay_pack_layout == "packed",
        )
        self._shared_resources.append(replay_buffer)
        replay_buffer.trace_recorder = trace_recorder
        replay_buffer.trace_thread_time = self.trace_thread_time
        replay_buffer.trace_cuda_events = self.trace_cuda_events

        # --- replay pipeline (double buffer) ---
        sample_count = self.batch_size * self.updates_per_step
        collector_pack_request_queue = _SPAWN_CTX.Queue(maxsize=1)
        collector_pack_ready_queue = _SPAWN_CTX.Queue(maxsize=1)
        packed_width = int(replay_buffer._storage.shape[1])
        collector_pack_shared_slots = [
            torch.empty((sample_count, packed_width), dtype=torch.float32).share_memory_()
            for _ in range(2)
        ]
        _verbose_output_dir: str | None = None
        if self.verbose_metrics:
            _vroot = Path(self.trace_output_dir) if self.trace_output_dir else Path(log_dir)
            _verbose_output_dir = str(_vroot)
        replay_pipeline = CPUPinnedDoubleBufferReplayPipeline(
            replay_buffer,
            device=self.device,
            sample_count=sample_count,
            base_seed=int(self.seed or 0),
            trace_recorder=trace_recorder,
            trace_cuda_events=self.trace_cuda_events,
            verbose=self.verbose_metrics,
            verbose_output_dir=_verbose_output_dir,
            collector_pack_request_queue=collector_pack_request_queue,
            collector_pack_ready_queue=collector_pack_ready_queue,
            collector_pack_shared_slots=collector_pack_shared_slots,
        )
        self.replay_h2d_submitter = getattr(
            replay_pipeline,
            "h2d_submitter",
            self.replay_h2d_submitter,
        )

        # --- weight sync ---
        weight_sync = SharedWeightSync.from_state_dict(self.learner.actor.state_dict(), create=True)
        self._shared_resources.append(weight_sync)
        weight_sync.trace_recorder = trace_recorder
        weight_sync.trace_thread_time = self.trace_thread_time

        # --- sync queues ---
        collection_ready_queue = None
        trainer_done_queue = None
        if self.sync_collection:
            collection_ready_queue = _SPAWN_CTX.Queue(maxsize=1)
            trainer_done_queue = _SPAWN_CTX.Queue(maxsize=1)
            trainer_done_queue.put(1)
            print(
                f"[DoubleBufferRunner] Collection sync enabled: "
                f"env_steps_per_sync={self.env_steps_per_sync}"
            )

        metrics_queue = _SPAWN_CTX.Queue(maxsize=100)

        # --- obs normalization ---
        shared_obs_normalizer_stats = None
        if self.obs_normalization:
            shared_obs_normalizer_stats = SharedObsNormStats(_SPAWN_CTX)

        # --- start collector ---
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
            "learning_starts": self.learning_starts,
            "metrics_queue": metrics_queue,
            "sync_collection": self.sync_collection,
            "collection_ready_queue": collection_ready_queue,
            "trainer_done_queue": trainer_done_queue,
            "env_steps_per_sync": self.env_steps_per_sync,
            "obs_normalization": self.obs_normalization,
            "shared_obs_normalizer_stats": shared_obs_normalizer_stats,
            "sim_backend": self.sim_backend,
            "env_cfg_override": self.env_cfg_override,
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "actor_kwargs": self.actor_kwargs,
            "seed": derive_worker_seed(self.seed, worker_index=0),
            "trace_enabled": self.trace_enabled,
            "trace_thread_time": self.trace_thread_time,
            "collector_pack_request_queue": collector_pack_request_queue,
            "collector_pack_ready_queue": collector_pack_ready_queue,
            "collector_pack_shared_slots": collector_pack_shared_slots,
        }
        self._start_collector(
            target_fn=off_policy_collector_fn,
            kwargs={"stop_event": self._stop_event, **collector_kwargs},
        )

        time.sleep(0.5)
        if self._collector_process:
            print(
                f"[DoubleBufferRunner] Collector process alive: "
                f"{self._collector_process.is_alive()}"
            )

        # --- logger ---
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
        if hasattr(self.learner, "use_symmetry") and self.learner.use_symmetry:
            logger.log_status("Symmetry augmentation: enabled")
        logger.log_status("Replay pipeline: cpu_pinned_double_buffer")
        logger.log_status(f"Replay prefetch mode: {self.replay_prefetch_mode}")
        logger.log_status(f"Replay pack layout: {self.replay_pack_layout}")
        logger.log_status(f"Replay pack executor: {self.replay_pack_executor}")
        logger.log_status(f"Replay H2D submitter: {self.replay_h2d_submitter}")
        logger.log_status(
            f"Replay learner lightweight: fixed (log_interval={self.LEARNER_LOG_INTERVAL})"
        )
        if self.verbose_metrics:
            logger.log_status("Verbose metrics: enabled (field-level pack CSV)")
        logger.start()

        reward_history: deque = deque(maxlen=100)
        latest_reward_components: dict[str, float] = {}
        last_buf_log = 0
        write_read_ema = 0.0
        reward_stats_ptr = 0
        train_start_threshold = self.train_start_threshold
        prepared_tick: int | None = None

        training_e2e_start_ns = time.perf_counter_ns() if trace_recorder else 0

        # ---- training loop ----
        for iteration in range(1, max_iterations + 1):
            # -- wait for data --
            wait_start = time.time()
            wait_start_ns = time.perf_counter_ns()
            if self.sync_collection and collection_ready_queue:
                import queue

                while True:
                    try:
                        collection_ready_queue.get(timeout=1.0)
                    except queue.Empty:
                        if not self._check_collector_alive():
                            self._drain_metrics(
                                metrics_queue,
                                reward_history,
                                latest_reward_components,
                                logger,
                                trace_recorder,
                            )
                            logger.log_status("[red]ERROR: Collector died[/]")
                            logger.finish()
                            self.last_run_summary = self._make_summary(
                                "collector_died",
                                iteration,
                                logger,
                                None,
                                None,
                                ckpt_path,
                                train_start_wall,
                                None,
                            )
                            replay_pipeline.close()
                            return
                        continue

                    self._drain_metrics(
                        metrics_queue,
                        reward_history,
                        latest_reward_components,
                        logger,
                        trace_recorder,
                    )
                    cur_size = int(replay_buffer.size[0])
                    if replay_buffer_ready_for_learning(
                        cur_size,
                        batch_size=self.batch_size,
                        learning_starts=self.learning_starts,
                        num_envs=self.num_envs,
                    ):
                        if prepared_tick != iteration:
                            replay_pipeline.start_prepare(iteration, sample_count)
                            prepared_tick = iteration
                        break
                    if cur_size - last_buf_log >= self.num_envs * 10:
                        last_buf_log = cur_size
                        logger.log_buffer_fill(cur_size, train_start_threshold)
                    if trainer_done_queue:
                        trainer_done_queue.put(1)
            else:
                while not replay_buffer_ready_for_learning(
                    int(replay_buffer.size[0]),
                    batch_size=self.batch_size,
                    learning_starts=self.learning_starts,
                    num_envs=self.num_envs,
                ):
                    if not self._check_collector_alive():
                        self._drain_metrics(
                            metrics_queue,
                            reward_history,
                            latest_reward_components,
                            logger,
                        )
                        logger.log_status("[red]ERROR: Collector died[/]")
                        logger.finish()
                        self.last_run_summary = self._make_summary(
                            "collector_died",
                            iteration,
                            logger,
                            None,
                            None,
                            ckpt_path,
                            train_start_wall,
                            None,
                        )
                        replay_pipeline.close()
                        return
                    cur_size = int(replay_buffer.size[0])
                    if cur_size - last_buf_log >= self.num_envs * 10:
                        last_buf_log = cur_size
                        logger.log_buffer_fill(cur_size, train_start_threshold)
                    time.sleep(0.1)
                    self._drain_metrics(
                        metrics_queue,
                        reward_history,
                        latest_reward_components,
                        logger,
                        trace_recorder,
                    )

            wait_time = time.time() - wait_start
            if trace_recorder:
                trace_recorder.add_slice(
                    "learner/wait_for_data",
                    category="learner",
                    start_ns=wait_start_ns,
                    end_ns=time.perf_counter_ns(),
                    args={"iteration": iteration},
                )
            self._drain_metrics(
                metrics_queue,
                reward_history,
                latest_reward_components,
                logger,
                trace_recorder,
            )
            _reward_stats_ns = time.perf_counter_ns()
            reward_stats_ptr = self._update_reward_stats_from_replay(
                replay_buffer,
                reward_stats_ptr,
                int(replay_buffer.ptr[0]),
            )
            if trace_recorder:
                trace_recorder.add_slice(
                    "learner/update_reward_stats",
                    category="learner",
                    start_ns=_reward_stats_ns,
                    end_ns=time.perf_counter_ns(),
                )

            # -- train --
            iter_metrics = defaultdict(list)
            ptr_before = int(replay_buffer.ptr[0])
            collector_released_for_next = False
            learner = self.learner

            with nullcontext():
                _sample_ns = time.perf_counter_ns()
                batch_ready = replay_pipeline.batch_ready(iteration, sample_count)
                _wait_batch_ns = time.perf_counter_ns()
                if not batch_ready:
                    batch_ready = replay_pipeline.wait_until_ready(iteration, sample_count)
                if trace_recorder:
                    trace_recorder.add_slice(
                        "learner/wait_for_replay_batch",
                        category="learner",
                        start_ns=_wait_batch_ns,
                        end_ns=time.perf_counter_ns(),
                        args={"iteration": iteration, "batch_ready": batch_ready},
                    )
                large_batch = replay_pipeline.sample_large_batch(
                    tick_id=iteration,
                    sample_count=sample_count,
                )
                learner_incremental_h2d_time = float(
                    getattr(replay_pipeline, "last_incremental_h2d_time_s", 0.0)
                )
                if iteration < max_iterations:
                    min_snapshot_ptr = int(replay_buffer.ptr[0]) + (
                        self.num_envs * self.env_steps_per_sync
                    )
                    replay_pipeline.start_prepare(
                        iteration + 1,
                        sample_count,
                        min_snapshot_ptr=min_snapshot_ptr,
                    )
                    if self.sync_collection and trainer_done_queue:
                        trainer_done_queue.put(1)
                        collector_released_for_next = True
                    prepared_tick = iteration + 1
                if trace_recorder:
                    trace_recorder.add_slice(
                        "learner/replay_sample",
                        category="learner",
                        start_ns=_sample_ns,
                        end_ns=time.perf_counter_ns(),
                        args={
                            "total_batch": sample_count,
                            "pipeline": "cpu_pinned_double_buffer",
                            "batch_ready": batch_ready,
                            "prefetch_mode": self.replay_prefetch_mode,
                            "replay_pack_layout": self.replay_pack_layout,
                            "replay_pack_executor": self.replay_pack_executor,
                            "replay_h2d_submitter": self.replay_h2d_submitter,
                            "prepared_tick": prepared_tick,
                            "explicit_compute_stream": False,
                        },
                    )

                train_start = time.time()

                for update_idx in range(self.updates_per_step):
                    s = update_idx * self.batch_size
                    e = s + self.batch_size
                    batch = {k: v[s:e] for k, v in large_batch.items()}

                    _critic_ns = time.perf_counter_ns()
                    critic_metrics = learner.update_critic(batch)
                    if trace_recorder:
                        trace_recorder.add_slice(
                            "learner/update_critic",
                            category="learner",
                            start_ns=_critic_ns,
                            end_ns=time.perf_counter_ns(),
                            args={"update_idx": update_idx},
                        )
                    for k, v in critic_metrics.items():
                        iter_metrics[k].append(v)

                    actor_updated = False
                    if update_idx % self.policy_frequency == 0:
                        _actor_ns = time.perf_counter_ns()
                        actor_metrics = learner.update_actor(batch)
                        if trace_recorder:
                            trace_recorder.add_slice(
                                "learner/update_actor",
                                category="learner",
                                start_ns=_actor_ns,
                                end_ns=time.perf_counter_ns(),
                                args={"update_idx": update_idx},
                            )
                        for k, v in actor_metrics.items():
                            iter_metrics[k].append(v)
                        actor_updated = True

                    _target_ns = time.perf_counter_ns()
                    if self.algo_type == "td3":
                        if actor_updated:
                            learner.soft_update_target()
                    else:
                        learner.soft_update_target()
                    if trace_recorder:
                        trace_recorder.add_slice(
                            "learner/soft_update_target",
                            category="learner",
                            start_ns=_target_ns,
                            end_ns=time.perf_counter_ns(),
                            args={"update_idx": update_idx},
                        )

                replay_pipeline.after_tick()

            if self.obs_normalization and getattr(self.learner, "obs_normalizer", None) is not None:
                assert shared_obs_normalizer_stats is not None
                shared_obs_normalizer_stats.put(
                    (
                        self.learner.obs_normalizer.mean.cpu().numpy(),
                        self.learner.obs_normalizer.std.cpu().numpy(),
                    )
                )

            train_time = time.time() - train_start
            self.learner.update_count += 1
            _ws_ns = time.perf_counter_ns()
            weight_sync_start = time.perf_counter()
            weight_sync.write_weights(self.learner.actor.state_dict())
            weight_sync_time = time.perf_counter() - weight_sync_start
            if trace_recorder:
                trace_recorder.add_slice(
                    "learner/weight_sync_write",
                    category="learner",
                    start_ns=_ws_ns,
                    end_ns=time.perf_counter_ns(),
                    args={"mode": "sync"},
                )
                trace_recorder.add_counter(
                    "replay_size",
                    int(replay_buffer.size[0]),
                    category="replay",
                )

            if self.sync_collection and trainer_done_queue and not collector_released_for_next:
                trainer_done_queue.put(1)

            write_delta = int(replay_buffer.ptr[0]) - ptr_before
            consume = self.batch_size * self.updates_per_step
            write_read_ema = 0.9 * write_read_ema + 0.1 * (write_delta / max(consume, 1))
            logger.update_buffer_utilization(write_read_ema)

            avg_metrics = {k: statistics.mean(v) for k, v in iter_metrics.items() if v}
            mean_reward = statistics.mean(reward_history) if reward_history else 0.0
            last_mean_reward = float(mean_reward)
            best_mean_reward = max(best_mean_reward, last_mean_reward)

            if (
                iteration == 1
                or iteration == max_iterations
                or iteration % self.LEARNER_LOG_INTERVAL == 0
            ):
                logger.log_step(
                    iteration=iteration,
                    metrics=avg_metrics,
                    reward=mean_reward,
                    reward_components=latest_reward_components,
                    train_time=train_time,
                    wait_time=wait_time,
                    learner_incremental_h2d_time=learner_incremental_h2d_time,
                    weight_sync_time=weight_sync_time,
                    extra_info={
                        "throughput_steps": self.num_envs * self.env_steps_per_sync,
                    },
                )

            if save_interval > 0 and iteration % save_interval == 0:
                ckpt_path = os.path.join(log_dir, f"model_{iteration}.pt")
                torch.save(self.learner.get_state_dict(), ckpt_path)
                logger.log_save(ckpt_path)

        if trace_recorder:
            trace_recorder.add_slice(
                "learner/training_e2e",
                category="learner",
                start_ns=training_e2e_start_ns,
                end_ns=time.perf_counter_ns(),
                args={
                    "iterations": iteration,
                    "pipeline": "cpu_pinned_double_buffer",
                    "replay_h2d_submitter": self.replay_h2d_submitter,
                    "learner_log_interval": self.LEARNER_LOG_INTERVAL,
                },
            )

        # -- finalize --
        replay_pipeline.close()
        ckpt_path = os.path.join(log_dir, f"model_{max_iterations}.pt")
        torch.save(self.learner.get_state_dict(), ckpt_path)
        logger.log_save(ckpt_path)
        logger.finish()
        if trace_recorder and trace_output_path:
            trace_recorder.write_json(trace_output_path)
            print(f"[DoubleBufferRunner] Perfetto trace written to {trace_output_path}")
        self.last_run_summary = self._make_summary(
            "completed",
            iteration,
            logger,
            last_mean_reward if reward_history else None,
            best_mean_reward if reward_history else None,
            ckpt_path,
            train_start_wall,
            str(trace_output_path) if trace_output_path else None,
        )

    @staticmethod
    def _make_summary(
        status,
        iteration,
        logger,
        final_reward,
        best_reward,
        ckpt_path,
        train_start_wall,
        trace_path,
    ) -> dict:
        return {
            "status": status,
            "completed_iterations": iteration,
            "total_env_steps": int(logger._total_steps),
            "final_mean_reward": final_reward,
            "best_mean_reward": best_reward,
            "mean_episode_length": float(logger._mean_ep_length),
            "last_checkpoint": ckpt_path,
            "trace_path": trace_path,
            "training_wall_time_sec": time.time() - train_start_wall,
        }
