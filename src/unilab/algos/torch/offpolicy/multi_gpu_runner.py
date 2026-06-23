"""Multi-GPU off-policy runner using per-rank updates with configurable sync.

Architecture:
  Main process   → creates ReplayBuffer (host-only), WeightSync, queues
                 → spawns Collector subprocess (CPU, env simulation)
                 → spawns N Learner workers via mp.spawn (one per GPU)
  Learner rank i → samples packed CPU replay rows to its rank device through
                   a rank-local H2D pipeline, then either averages gradients
                   per update or averages parameters at local-SGD sync boundaries.
  Collector      → talks only to rank 0 via collection_ready_queue / trainer_done_queue
"""

from __future__ import annotations

import os
import queue
import socket
import sys
import time
from collections import defaultdict, deque
from datetime import timedelta
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist
import torch.multiprocessing as tmp  # torch.multiprocessing for spawn

from unilab.algos.torch.offpolicy.runner import (
    OffPolicyRunner,
    build_offpolicy_sample_info,
    build_reward_comparison_metrics,
    compute_train_start_threshold,
    replay_buffer_ready_for_learning,
)
from unilab.algos.torch.offpolicy.worker import off_policy_collector_fn
from unilab.ipc import SharedWeightSync
from unilab.ipc.async_runner import _SPAWN_CTX
from unilab.ipc.replay_buffer import ReplayBuffer
from unilab.ipc.replay_pipelines.multi_gpu_cpu_pinned import MultiGPUCPUPinnedReplayPipeline
from unilab.logging import OffPolicyLogger
from unilab.training.seed import apply_training_seed, derive_worker_seed

MULTIGPU_REPLAY_READY_POLL_SEC = 0.001
MULTIGPU_SYNC_MODES = {"sync_sgd", "local_sgd"}


class _CollectorDiedError(RuntimeError):
    """Raised when the collector dies while multi-GPU learners are running."""


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return int(s.getsockname()[1])


def normalize_multi_gpu_sync_mode(mode: str) -> str:
    """Return a validated multi-GPU learner synchronization mode."""
    normalized = str(mode).strip().lower()
    if normalized not in MULTIGPU_SYNC_MODES:
        supported = ", ".join(sorted(MULTIGPU_SYNC_MODES))
        raise ValueError(f"training.multi_gpu_sync_mode must be one of: {supported}; got {mode!r}")
    return normalized


def normalize_multi_gpu_sync_interval(interval: int) -> int:
    """Return a validated positive local-SGD synchronization interval."""
    normalized = int(interval)
    if normalized < 1:
        raise ValueError(f"training.multi_gpu_sync_interval must be >= 1; got {interval!r}")
    return normalized


def _drain_metrics(
    metrics_queue: Any,
    reward_history: deque,
    reward_components: dict,
    logger: Optional[OffPolicyLogger],
) -> None:
    while not metrics_queue.empty():
        try:
            m = metrics_queue.get_nowait()
            if "error" in m:
                if logger:
                    logger.log_status(f"[red]Collector ERROR: {m['error']}[/]")
                return

            if "mean_ep_reward" in m:
                reward_history.append(m["mean_ep_reward"])
            if "reward_components" in m:
                reward_components.clear()
                reward_components.update(m["reward_components"])
            if "mean_ep_length" in m and logger:
                logger.update_ep_length(m["mean_ep_length"])
            if "collector_timing_ms" in m and logger:
                logger.update_collector_timing(m["collector_timing_ms"])
            if ("timeout_rate" in m or "terminated_rate" in m) and logger:
                logger.update_done_rates(
                    timeout_rate=float(m.get("timeout_rate", 0.0)),
                    terminated_rate=float(m.get("terminated_rate", 0.0)),
                )
            if "total_steps" in m and "buffer_size" in m and logger:
                logger.log_collector(
                    m["total_steps"],
                    m["buffer_size"],
                    m.get("mean_ep_reward", 0.0),
                )
        except Exception as e:
            print(f"[MultiGPU] metrics drain error: {e}", file=sys.stderr)
            break


def _put_trainer_done_or_stop(trainer_done_queue: Any, stop_event: Any) -> bool:
    if trainer_done_queue is None:
        return True
    while not stop_event.is_set():
        try:
            trainer_done_queue.put(1, timeout=0.5)
            return True
        except queue.Full:
            continue
    return False


def _learner_worker(
    rank: int,
    world_size: int,
    learner_cls: Any,
    learner_kwargs: Dict[str, Any],
    runner_kwargs: Dict[str, Any],
    replay_buffer: ReplayBuffer,
    weight_sync_name: str,
    weight_sync_lock: Any,
    weight_param_shapes: Dict[str, Any],
    stop_event: Any,
    collection_ready_queue: Any,
    trainer_done_queue: Any,
    metrics_queue: Any,
    collector_pack_request_queue: Any,
    collector_pack_ready_queue: Any,
    collector_pack_shared_slots: Any,
    master_port: int,
) -> None:
    """Worker function executed on each GPU (called via torch.multiprocessing.spawn)."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(master_port)
    device = f"cuda:{rank}"
    torch.cuda.set_device(rank)
    backend = str(runner_kwargs.get("distributed_backend", "nccl"))
    dist.init_process_group(
        backend, rank=rank, world_size=world_size, timeout=timedelta(seconds=120)
    )

    logger: Optional[OffPolicyLogger] = None
    weight_sync: SharedWeightSync | None = None
    replay_pipeline: MultiGPUCPUPinnedReplayPipeline | None = None
    try:
        apply_training_seed(
            derive_worker_seed(runner_kwargs.get("seed"), worker_index=rank + 1000),
            torch_runtime=True,
            cuda=True,
        )
        # 1. Bind this worker's process-local replay samples to its rank device.
        replay_buffer.device = device

        # 2. Create learner on this device
        learner_kwargs = dict(learner_kwargs)
        learner_kwargs["distributed_sync_mode"] = normalize_multi_gpu_sync_mode(
            str(runner_kwargs.get("multi_gpu_sync_mode", "local_sgd"))
        )
        learner = learner_cls(device=device, world_size=world_size, **learner_kwargs)

        # 3. Broadcast rank-0 params so all workers start identically.
        sync_initial_parameters = getattr(learner, "sync_initial_parameters", None)
        if not callable(sync_initial_parameters):
            raise ValueError(
                "Multi-GPU off-policy learner must implement sync_initial_parameters(src=0)"
            )
        sync_initial_parameters(src=0)

        # 4. Reconnect to the shared weight-sync buffer
        weight_sync = SharedWeightSync(
            weight_param_shapes, create=False, shm_name=weight_sync_name, lock=weight_sync_lock
        )

        # 5. Unpack runner config
        max_iterations: int = runner_kwargs["max_iterations"]
        save_interval: int = runner_kwargs["save_interval"]
        log_dir: str = runner_kwargs["log_dir"]
        batch_size: int = runner_kwargs["batch_size"]
        updates_per_step: int = runner_kwargs["updates_per_step"]
        policy_frequency: int = runner_kwargs["policy_frequency"]
        sync_collection: bool = runner_kwargs["sync_collection"]
        env_steps_per_sync: int = runner_kwargs.get("env_steps_per_sync", 1)
        env_name: str = runner_kwargs["env_name"]
        num_envs: int = runner_kwargs["num_envs"]
        obs_dim: int = runner_kwargs["obs_dim"]
        action_dim: int = runner_kwargs["action_dim"]
        logger_type: str = runner_kwargs.get("logger_type", "tensorboard")
        sync_mode = normalize_multi_gpu_sync_mode(
            str(runner_kwargs.get("multi_gpu_sync_mode", "local_sgd"))
        )
        sync_interval = normalize_multi_gpu_sync_interval(
            int(runner_kwargs.get("multi_gpu_sync_interval", 1))
        )
        learning_starts = max(int(runner_kwargs.get("learning_starts", 0)), 0)
        train_start_threshold = compute_train_start_threshold(batch_size, learning_starts, num_envs)
        sample_count = batch_size * updates_per_step

        replay_pipeline = MultiGPUCPUPinnedReplayPipeline(
            replay_buffer,
            rank=rank,
            world_size=world_size,
            device=device,
            sample_count=sample_count,
            base_seed=int(runner_kwargs.get("seed") or 0),
            collector_pack_request_queue=collector_pack_request_queue[rank],
            collector_pack_ready_queue=collector_pack_ready_queue[rank],
            collector_pack_shared_slots=collector_pack_shared_slots[rank],
        )

        # 6. Logger (rank 0 only)
        if rank == 0:
            os.makedirs(log_dir, exist_ok=True)
            logger = OffPolicyLogger(
                algo_name=f"Fast{str(runner_kwargs.get('algo_type', 'offpolicy')).upper()}_x{world_size}GPU",
                max_iterations=max_iterations,
                num_envs=num_envs,
                env_name=env_name,
                obs_dim=obs_dim,
                action_dim=action_dim,
                log_dir=log_dir,
                log_backend=logger_type,
            )
            logger.set_collection_sync(sync_collection, env_steps_per_sync)
            logger.log_status("Replay pipeline: multi_gpu_cpu_pinned")
            logger.log_status(
                "Batch semantics: "
                f"algo.batch_size={batch_size} per learner rank; "
                f"global_batch={batch_size * world_size}"
            )
            logger.log_status(
                "Multi-GPU learner sync: "
                f"{sync_mode} (interval={sync_interval} iteration"
                f"{'s' if sync_interval != 1 else ''})"
            )
            if sync_mode == "local_sgd":
                logger.log_status(
                    "Local-SGD optimizer state: rank-local; parameters averaged at sync boundary"
                )
            logger.start()

        reward_history: deque = deque(maxlen=100)
        latest_reward_components: dict = {}
        write_read_ema = 0.0
        last_buf_log = 0
        prepared_tick: int | None = None

        # 7. Training loop
        for it in range(1, max_iterations + 1):
            iteration_start = time.perf_counter()
            collector_released_for_next = False
            # --- Wait for data (rank 0 only, then barrier syncs everyone) ---
            wait_start = time.perf_counter()
            if rank == 0:
                if sync_collection and collection_ready_queue is not None:
                    while True:
                        try:
                            collection_ready_queue.get(timeout=1.0)
                        except queue.Empty:
                            if stop_event.is_set():
                                return
                            continue
                        if stop_event.is_set():
                            return
                        cur_size = int(replay_buffer.size[0])
                        if replay_buffer_ready_for_learning(
                            cur_size,
                            batch_size=batch_size,
                            learning_starts=learning_starts,
                            num_envs=num_envs,
                        ):
                            break
                        if logger and cur_size - last_buf_log >= num_envs * 10:
                            last_buf_log = cur_size
                            logger.log_buffer_fill(cur_size, train_start_threshold)
                        if trainer_done_queue is not None:
                            if not _put_trainer_done_or_stop(trainer_done_queue, stop_event):
                                return
                else:
                    while not replay_buffer_ready_for_learning(
                        int(replay_buffer.size[0]),
                        batch_size=batch_size,
                        learning_starts=learning_starts,
                        num_envs=num_envs,
                    ):
                        if stop_event.is_set():
                            return
                        cur_size = int(replay_buffer.size[0])
                        if logger and cur_size - last_buf_log >= num_envs * 10:
                            last_buf_log = cur_size
                            logger.log_buffer_fill(cur_size, train_start_threshold)
                        time.sleep(MULTIGPU_REPLAY_READY_POLL_SEC)
                _drain_metrics(metrics_queue, reward_history, latest_reward_components, logger)

            dist.barrier()
            wait_time = time.perf_counter() - wait_start if rank == 0 else 0.0

            # --- Training: each rank independently samples a different mini-batch ---
            iter_metrics: dict = defaultdict(list)
            ptr_before = int(replay_buffer.ptr[0]) if rank == 0 else 0

            if prepared_tick != it:
                replay_pipeline.start_prepare(it, sample_count)
                prepared_tick = it
            if not replay_pipeline.batch_ready(it, sample_count):
                while not replay_pipeline.batch_ready(it, sample_count):
                    if stop_event.is_set():
                        return
                    time.sleep(MULTIGPU_REPLAY_READY_POLL_SEC)
            large_batch = replay_pipeline.sample_large_batch(it, sample_count)
            learner_incremental_h2d_time = (
                float(getattr(replay_pipeline, "last_incremental_h2d_time_s", 0.0))
                if rank == 0
                else 0.0
            )

            if it < max_iterations:
                min_snapshot_ptr = int(replay_buffer.ptr[0]) + (num_envs * env_steps_per_sync)
                replay_pipeline.start_prepare(
                    it + 1,
                    sample_count,
                    min_snapshot_ptr=min_snapshot_ptr,
                )
                prepared_tick = it + 1
                if rank == 0 and sync_collection and trainer_done_queue is not None:
                    if not _put_trainer_done_or_stop(trainer_done_queue, stop_event):
                        return
                    collector_released_for_next = True

            train_start = time.perf_counter()

            for update_idx in range(updates_per_step):
                s = update_idx * batch_size
                e = s + batch_size
                batch = {k: v[s:e] for k, v in large_batch.items()}

                critic_metrics = learner.update_critic(batch)
                for k, v in critic_metrics.items():
                    iter_metrics[k].append(v)

                if update_idx % policy_frequency == 0:
                    actor_metrics = learner.update_actor(batch)
                    for k, v in actor_metrics.items():
                        iter_metrics[k].append(v)

                learner.soft_update_target()

            replay_pipeline.after_tick()

            should_save_checkpoint = save_interval > 0 and it % save_interval == 0
            should_param_sync = sync_mode == "local_sgd" and (
                it % sync_interval == 0 or it == max_iterations or should_save_checkpoint
            )
            param_sync_time = 0.0
            did_param_sync = False
            if should_param_sync:
                param_sync_start = time.perf_counter()
                average_parameters = getattr(learner, "average_distributed_parameters", None)
                if not callable(average_parameters):
                    raise ValueError(
                        "Multi-GPU local_sgd requires learner.average_distributed_parameters()"
                    )
                average_parameters()
                param_sync_time = time.perf_counter() - param_sync_start
                did_param_sync = True

            # train_time intentionally includes local-SGD parameter sync and the
            # final rank barrier. The separate param-sync timing is a sub-breakdown.
            dist.barrier()
            train_time = time.perf_counter() - train_start if rank == 0 else 0.0

            # --- Post-iteration work: rank 0 only ---
            if rank == 0:
                learner.update_count += 1
                weight_sync_time = 0.0
                if sync_mode != "local_sgd" or did_param_sync:
                    weight_sync_start = time.perf_counter()
                    weight_sync.write_weights(learner.actor.state_dict())
                    weight_sync_time = time.perf_counter() - weight_sync_start

                if (
                    sync_collection
                    and trainer_done_queue is not None
                    and not collector_released_for_next
                ):
                    if not _put_trainer_done_or_stop(trainer_done_queue, stop_event):
                        return
                iteration_time = time.perf_counter() - iteration_start

                write_delta = int(replay_buffer.ptr[0]) - ptr_before
                consume = batch_size * updates_per_step * world_size
                write_read_ema = 0.9 * write_read_ema + 0.1 * (write_delta / max(consume, 1))

                import statistics as _stats

                avg_metrics = {k: _stats.mean(v) for k, v in iter_metrics.items() if v}
                mean_reward = _stats.mean(reward_history) if reward_history else 0.0

                if logger:
                    logger.update_buffer_utilization(write_read_ema)
                    logger.log_step(
                        iteration=it,
                        metrics=avg_metrics,
                        reward=mean_reward,
                        reward_metrics=build_reward_comparison_metrics(reward_history, mean_reward),
                        reward_components=latest_reward_components,
                        train_time=train_time,
                        wait_time=wait_time,
                        learner_incremental_h2d_time=learner_incremental_h2d_time,
                        weight_sync_time=weight_sync_time,
                        learner_param_sync_time=param_sync_time,
                        iteration_time=iteration_time,
                        extra_info={
                            "throughput_steps": num_envs * env_steps_per_sync,
                            "world_size": world_size,
                            "multi_gpu_sync_mode": sync_mode,
                            "multi_gpu_sync_interval": sync_interval,
                            **build_offpolicy_sample_info(
                                replay_batch_size_per_rank=batch_size,
                                updates_per_step=updates_per_step,
                                learner=learner,
                                world_size=world_size,
                            ),
                        },
                    )

                if should_save_checkpoint:
                    ckpt_path = os.path.join(log_dir, f"model_{it}.pt")
                    torch.save(learner.get_state_dict(), ckpt_path)
                    if logger:
                        logger.log_save(ckpt_path)

        # Final checkpoint (rank 0)
        if rank == 0:
            ckpt_path = os.path.join(log_dir, f"model_{max_iterations}.pt")
            torch.save(learner.get_state_dict(), ckpt_path)
            if logger:
                logger.log_save(ckpt_path)
                logger.finish()

        if replay_pipeline is not None:
            replay_pipeline.close()
            replay_pipeline = None
        weight_sync.close()
        weight_sync = None

    finally:
        if logger is not None:
            logger.close()
        if replay_pipeline is not None:
            replay_pipeline.close()
        if weight_sync is not None:
            weight_sync.close()
        dist.destroy_process_group()


class MultiGPUOffPolicyRunner(OffPolicyRunner):
    """Multi-GPU off-policy runner.

    Keeps a single Collector on CPU and spawns *num_gpus* Learner workers via
    ``torch.multiprocessing.spawn``. Each worker processes an independent
    mini-batch from the same shared ReplayBuffer through a rank-local H2D
    pipeline. SAC defaults to local-SGD: ranks apply local updates and average
    parameters at runner-controlled synchronization boundaries. Strict per-update
    gradient averaging remains available through ``training.multi_gpu_sync_mode=sync_sgd``.

    Falls back transparently to single-GPU when ``num_gpus <= 1``.
    """

    @staticmethod
    def validate_capabilities(
        *,
        algo_type: str,
        learner_kwargs: Dict[str, Any],
        num_gpus: int,
    ) -> None:
        if num_gpus <= 1:
            return
        if algo_type == "sac" and bool(learner_kwargs.get("use_symmetry", False)):
            raise ValueError(
                "Off-policy symmetry augmentation does not support training.num_gpus > 1; "
                "set training.num_gpus=1 or algo.use_symmetry=false"
            )

    def __init__(
        self,
        learner: Any,
        env_name: str,
        algo_type: str,
        learner_cls: Any,
        learner_kwargs: Dict[str, Any],
        num_gpus: int = 1,
        distributed_backend: str = "nccl",
        multi_gpu_sync_mode: str = "local_sgd",
        multi_gpu_sync_interval: int = 1,
        **kwargs: Any,
    ) -> None:
        self.validate_capabilities(
            algo_type=algo_type,
            learner_kwargs=learner_kwargs,
            num_gpus=num_gpus,
        )
        super().__init__(learner=learner, env_name=env_name, algo_type=algo_type, **kwargs)
        self.num_gpus = num_gpus
        self.world_size = num_gpus
        self._learner_cls = learner_cls
        self._learner_kwargs = learner_kwargs
        self.distributed_backend = distributed_backend
        self.multi_gpu_sync_mode = normalize_multi_gpu_sync_mode(multi_gpu_sync_mode)
        self.multi_gpu_sync_interval = normalize_multi_gpu_sync_interval(
            int(multi_gpu_sync_interval)
        )

    def _join_learner_context_with_collector_monitor(self, process_context: Any) -> None:
        """Join spawned learners while preserving collector liveness diagnostics."""
        while True:
            if not self._check_collector_alive():
                self._stop_event.set()
                self._terminate_learner_context(process_context, grace_period=2.0)
                raise _CollectorDiedError(
                    "Collector process died during multi-GPU off-policy training"
                )
            if process_context.join(timeout=0.5, grace_period=2.0):
                return

    @staticmethod
    def _terminate_learner_context(process_context: Any, *, grace_period: float) -> None:
        deadline = time.monotonic() + grace_period
        while time.monotonic() < deadline:
            try:
                if process_context.join(timeout=0.1, grace_period=grace_period):
                    return
            except Exception:
                return
        for process in getattr(process_context, "processes", []):
            if process.is_alive():
                process.terminate()
        for process in getattr(process_context, "processes", []):
            process.join(timeout=grace_period)

    def learn(
        self,
        max_iterations: int = 1500,
        save_interval: int = 50,
        log_dir: str = "logs",
        logger_type: str = "tensorboard",
    ) -> None:
        if self.num_gpus <= 1:
            super().learn(
                max_iterations=max_iterations,
                save_interval=save_interval,
                log_dir=log_dir,
                logger_type=logger_type,
            )
            return
        if not self.sync_collection:
            raise ValueError("Multi-GPU off-policy replay requires synchronized collection")
        self._learn_multi_gpu(
            max_iterations=max_iterations,
            save_interval=save_interval,
            log_dir=log_dir,
            logger_type=logger_type,
        )

    def _learn_multi_gpu(
        self,
        max_iterations: int,
        save_interval: int,
        log_dir: str,
        logger_type: str,
    ) -> None:
        os.makedirs(log_dir, exist_ok=True)

        # --- Shared objects (main process owns, workers share via IPC) ---
        buffer_capacity = self.replay_buffer_n * self.num_envs
        replay_buffer = ReplayBuffer(
            capacity=buffer_capacity,
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            device=self.device,
            defer_gpu=True,
            critic_dim=self.critic_obs_dim,
            packed_cpu_storage=True,
        )
        self._shared_resources.append(replay_buffer)

        weight_sync = SharedWeightSync.from_state_dict(self.learner.actor.state_dict(), create=True)
        self._shared_resources.append(weight_sync)

        collection_ready_queue = None
        trainer_done_queue = None
        if self.sync_collection:
            collection_ready_queue = _SPAWN_CTX.Queue(maxsize=1)
            trainer_done_queue = _SPAWN_CTX.Queue(maxsize=1)
            trainer_done_queue.put(1)
            print(
                f"[MultiGPURunner] Collection sync enabled: "
                f"env_steps_per_sync={self.env_steps_per_sync}"
            )

        metrics_queue = _SPAWN_CTX.Queue(maxsize=100)
        collector_pack_request_queues = [_SPAWN_CTX.Queue(maxsize=2) for _ in range(self.num_gpus)]
        collector_pack_ready_queues = [_SPAWN_CTX.Queue(maxsize=2) for _ in range(self.num_gpus)]
        sample_count = self.batch_size * self.updates_per_step
        packed_width = int(replay_buffer._storage.shape[1])
        collector_pack_shared_slots = [
            [
                torch.empty((sample_count, packed_width), dtype=torch.float32).share_memory_()
                for _ in range(2)
            ]
            for _ in range(self.num_gpus)
        ]

        # --- Start Collector (CPU, single process, unchanged) ---
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
            "obs_normalization": False,
            "shared_obs_normalizer_stats": None,
            "sim_backend": self.sim_backend,
            "env_cfg_override": self.env_cfg_override,
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "actor_kwargs": self.actor_kwargs,
            "seed": derive_worker_seed(self.seed, worker_index=0),
            "collector_pack_request_queue": collector_pack_request_queues,
            "collector_pack_ready_queue": collector_pack_ready_queues,
            "collector_pack_shared_slots": collector_pack_shared_slots,
        }
        self._start_collector(
            target_fn=off_policy_collector_fn,
            kwargs={"stop_event": self._stop_event, **collector_kwargs},
        )
        time.sleep(0.5)
        if self._collector_process:
            print(f"[MultiGPURunner] Collector process alive: {self._collector_process.is_alive()}")

        master_port = _find_free_port()
        print(
            f"[MultiGPURunner] Spawning {self.num_gpus} Learner workers (NCCL port {master_port})"
        )

        runner_kwargs: Dict[str, Any] = {
            "max_iterations": max_iterations,
            "save_interval": save_interval,
            "log_dir": log_dir,
            "batch_size": self.batch_size,
            "learning_starts": self.learning_starts,
            "updates_per_step": self.updates_per_step,
            "policy_frequency": self.policy_frequency,
            "sync_collection": self.sync_collection,
            "env_steps_per_sync": self.env_steps_per_sync,
            "env_name": self.env_name,
            "num_envs": self.num_envs,
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "logger_type": logger_type,
            "seed": self.seed,
            "distributed_backend": self.distributed_backend,
            "multi_gpu_sync_mode": self.multi_gpu_sync_mode,
            "multi_gpu_sync_interval": self.multi_gpu_sync_interval,
            "algo_type": self.algo_type,
        }

        try:
            process_context = tmp.spawn(  # pyright: ignore[reportPrivateImportUsage]
                _learner_worker,
                args=(
                    self.num_gpus,
                    self._learner_cls,
                    self._learner_kwargs,
                    runner_kwargs,
                    replay_buffer,
                    weight_sync.name,
                    weight_sync._lock,
                    weight_param_shapes,
                    self._stop_event,
                    collection_ready_queue,
                    trainer_done_queue,
                    metrics_queue,
                    collector_pack_request_queues,
                    collector_pack_ready_queues,
                    collector_pack_shared_slots,
                    master_port,
                ),
                nprocs=self.num_gpus,
                join=False,
            )
            self._join_learner_context_with_collector_monitor(process_context)
        finally:
            self._stop_event.set()
