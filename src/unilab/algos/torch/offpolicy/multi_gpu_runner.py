"""Multi-GPU off-policy runner using NCCL all-reduce for FastSAC.

Architecture:
  Main process   → creates ReplayBuffer (host-only), WeightSync, queues
                 → spawns Collector subprocess (CPU, env simulation)
                 → spawns N Learner workers via mp.spawn (one per GPU)
  Learner rank i → creates its own GPU cache, communicates via NCCL all_reduce
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
from typing import Any, Dict, Optional, cast

import torch
import torch.distributed as dist
import torch.multiprocessing as tmp  # torch.multiprocessing for spawn

from unilab.algos.torch.fast_sac.learner import FastSACLearner
from unilab.algos.torch.offpolicy.runner import (
    OffPolicyRunner,
    compute_train_start_threshold,
    replay_buffer_ready_for_learning,
)
from unilab.algos.torch.offpolicy.worker import off_policy_collector_fn
from unilab.ipc import SharedWeightSync
from unilab.ipc.async_runner import _SPAWN_CTX
from unilab.ipc.replay_buffer import ReplayBuffer
from unilab.training.logging.offpolicy import OffPolicyLogger


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return int(s.getsockname()[1])


def _broadcast_initial_params(learner: FastSACLearner, rank: int) -> None:
    """Broadcast rank-0 initial parameters to all workers for consistent starting point."""
    for model in (
        cast(torch.nn.Module, learner.actor),
        cast(torch.nn.Module, learner.qnet),
    ):
        for p in model.parameters():
            dist.broadcast(p.data, src=0)
    dist.broadcast(learner.log_alpha.data, src=0)


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


def _learner_worker(
    rank: int,
    world_size: int,
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
    master_port: int,
) -> None:
    """Worker function executed on each GPU (called via torch.multiprocessing.spawn)."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(master_port)
    device = f"cuda:{rank}"
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl", rank=rank, world_size=world_size, timeout=timedelta(seconds=120)
    )

    try:
        # 1. Initialise per-process GPU cache (host tensors already shared)
        replay_buffer.init_local_gpu_cache(device)

        # 2. Create learner on this device
        learner = FastSACLearner(device=device, world_size=world_size, **learner_kwargs)

        # 3. Broadcast rank-0 params so all workers start identically
        _broadcast_initial_params(learner, rank)

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
        learning_starts = max(int(runner_kwargs.get("learning_starts", 0)), 0)
        train_start_threshold = compute_train_start_threshold(batch_size, learning_starts, num_envs)

        # 6. Logger (rank 0 only)
        logger: Optional[OffPolicyLogger] = None
        if rank == 0:
            os.makedirs(log_dir, exist_ok=True)
            logger = OffPolicyLogger(
                algo_name=f"FastSAC_x{world_size}GPU",
                max_iterations=max_iterations,
                num_envs=num_envs,
                env_name=env_name,
                obs_dim=obs_dim,
                action_dim=action_dim,
                log_dir=log_dir,
                log_backend=logger_type,
            )
            logger.set_collection_sync(sync_collection, env_steps_per_sync)
            logger.start()

        reward_history: deque = deque(maxlen=100)
        latest_reward_components: dict = {}
        write_read_ema = 0.0
        last_buf_log = 0

        # 7. Training loop
        for it in range(1, max_iterations + 1):
            iter_start = time.time()

            # --- Wait for data (rank 0 only, then barrier syncs everyone) ---
            wait_start = time.time()
            if rank == 0:
                if sync_collection and collection_ready_queue is not None:
                    while True:
                        try:
                            collection_ready_queue.get(timeout=1.0)
                        except queue.Empty:
                            if stop_event.is_set():
                                return
                            continue
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
                            trainer_done_queue.put(1)
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
                        time.sleep(0.1)
                _drain_metrics(metrics_queue, reward_history, latest_reward_components, logger)

            dist.barrier()
            wait_time = time.time() - wait_start if rank == 0 else 0.0
            collect_time = time.time() - iter_start if rank == 0 else 0.0

            # --- Training: each rank independently samples a different mini-batch ---
            train_start = time.time()
            iter_metrics: dict = defaultdict(list)
            ptr_before = int(replay_buffer.ptr[0]) if rank == 0 else 0

            large_batch = replay_buffer.sample(batch_size * updates_per_step)

            for update_idx in range(updates_per_step):
                s = update_idx * batch_size
                e = s + batch_size
                batch = {k: v[s:e] for k, v in large_batch.items()}

                critic_metrics = learner.update_critic(batch)
                for k, v in critic_metrics.items():
                    iter_metrics[k].append(v)

                if update_idx % policy_frequency == 1:
                    actor_metrics = learner.update_actor(batch)
                    for k, v in actor_metrics.items():
                        iter_metrics[k].append(v)

                learner.soft_update_target()

            # Barrier: all ranks must finish this iteration before rank 0 proceeds
            dist.barrier()
            train_time = time.time() - train_start if rank == 0 else 0.0

            # --- Post-iteration work: rank 0 only ---
            if rank == 0:
                learner.update_count += 1
                weight_sync.write_weights(learner.actor.state_dict())

                if sync_collection and trainer_done_queue is not None:
                    trainer_done_queue.put(1)

                write_delta = int(replay_buffer.ptr[0]) - ptr_before
                consume = batch_size * updates_per_step
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
                        reward_components=latest_reward_components,
                        collect_time=collect_time,
                        train_time=train_time,
                        wait_time=wait_time,
                    )

                if save_interval > 0 and it % save_interval == 0:
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

        weight_sync.close()

    finally:
        dist.destroy_process_group()


class MultiGPUOffPolicyRunner(OffPolicyRunner):
    """Multi-GPU off-policy runner.

    Keeps a single Collector on CPU and spawns *num_gpus* Learner workers via
    ``torch.multiprocessing.spawn``.  Each worker processes independent
    mini-batches from the same shared ReplayBuffer; gradients are averaged
    with NCCL all_reduce — equivalent to training on a *num_gpus× larger*
    effective batch size per wall-clock second.

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
        learner_kwargs: Dict[str, Any],
        num_gpus: int = 1,
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
        self._learner_kwargs = learner_kwargs

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
            defer_gpu=True,  # No GPU tensors in main process; workers call init_local_gpu_cache
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
        }

        try:
            tmp.spawn(  # pyright: ignore[reportPrivateImportUsage]
                _learner_worker,
                args=(
                    self.num_gpus,
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
                    master_port,
                ),
                nprocs=self.num_gpus,
                join=True,
            )
        finally:
            self._stop_event.set()
