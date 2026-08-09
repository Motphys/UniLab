"""Shared contracts for single-device off-policy runners."""

from __future__ import annotations

import sys
from collections import deque
from typing import Any

import torch

from unilab.algos.torch.common.device import get_env_dims
from unilab.algos.torch.offpolicy.worker import off_policy_collector_fn
from unilab.ipc.async_runner import AsyncRunner
from unilab.logging import OffPolicyLogger
from unilab.training.seed import apply_training_seed
from unilab.utils.device import get_default_device, resolve_torch_device_alias
from unilab.utils.nan_guard import NanGuardCfg


def compute_train_start_threshold(batch_size: int, learning_starts: int, num_envs: int) -> int:
    """Return the minimum replay size required before learner updates may start."""
    return max(int(batch_size), max(int(learning_starts), 0) * max(int(num_envs), 1), 0)


def replay_buffer_ready_for_learning(
    replay_buffer_size: int,
    *,
    batch_size: int,
    learning_starts: int,
    num_envs: int,
) -> bool:
    """Whether the replay buffer has enough samples for the first learner step."""
    return int(replay_buffer_size) >= compute_train_start_threshold(
        batch_size,
        learning_starts,
        num_envs,
    )


def get_learner_batch_multiplier(learner: Any) -> int:
    """Return the effective learner batch multiplier for one replay row."""
    if not bool(getattr(learner, "use_symmetry", False)):
        return 1
    symmetry = getattr(learner, "symmetry", None)
    multiplier = int(getattr(symmetry, "batch_multiplier", 1) or 1)
    return max(multiplier, 1)


def build_offpolicy_sample_info(
    *,
    replay_batch_size_per_rank: int,
    updates_per_step: int,
    learner: Any,
) -> dict[str, int]:
    """Describe replay rows and effective learner samples for logging."""
    updates_per_step = max(int(updates_per_step), 0)
    replay_batch_size_per_rank = max(int(replay_batch_size_per_rank), 0)
    batch_multiplier = get_learner_batch_multiplier(learner)
    batch_size_per_rank = replay_batch_size_per_rank * batch_multiplier
    return {
        "batch_size_per_rank": batch_size_per_rank,
        "effective_batch_size": batch_size_per_rank,
        "replay_samples_per_iter": replay_batch_size_per_rank * updates_per_step,
        "learner_samples_per_iter": batch_size_per_rank * updates_per_step,
    }


def build_reward_comparison_metrics(
    reward_history: deque,
    smoothed_reward: float,
) -> dict[str, float]:
    """Return the latest collector-side 100-episode mean for reward comparison."""
    del smoothed_reward
    if not reward_history:
        return {}
    return {"mean_ep100": float(reward_history[-1])}


def update_reward_stats_from_replay(
    learner: Any,
    replay_buffer: Any,
    *,
    start_ptr: int,
    end_ptr: int,
    num_envs: int,
    replay_source: Any | None = None,
) -> int:
    """Update reward statistics from rows committed to the device replay owner."""
    if not hasattr(learner, "update_reward_stats"):
        return end_ptr
    if getattr(learner, "reward_normalizer", None) is None:
        return end_ptr
    if replay_source is None:
        raise RuntimeError("Reward statistics require the device-authoritative replay source")

    end_ptr, committed_fields = replay_source.read_committed_fields(
        ("rewards", "dones"),
        start_ptr=start_ptr,
    )
    count = end_ptr - start_ptr
    if count <= 0:
        return end_ptr
    if count > replay_buffer.capacity:
        count = replay_buffer.capacity
        start_ptr = end_ptr - count
    if count % num_envs != 0:
        count -= count % num_envs
        start_ptr = end_ptr - count
    if count <= 0:
        return end_ptr

    rewards = committed_fields["rewards"][-count:]
    dones = committed_fields["dones"][-count:]
    num_steps = count // num_envs
    learner.update_reward_stats(
        rewards.view(num_steps, num_envs),
        dones.view(num_steps, num_envs),
    )
    return end_ptr


class OffPolicyRunner(AsyncRunner):
    """Shared lifecycle and metrics helpers for the device replay runner."""

    def __init__(
        self,
        learner,
        env_name: str,
        algo_type: str,
        num_envs: int = 4096,
        replay_buffer_n: int = 1024,
        batch_size: int = 8192,
        learning_starts: int = 0,
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
        seed: int | None = None,
        trace_enabled: bool = False,
        trace_output_dir: str | None = None,
        trace_thread_time: bool = False,
        trace_cuda_events: bool = True,
        nan_guard_cfg: NanGuardCfg | None = None,
        collector_infer_device: str | None = "cpu",
        torch_thread_runtime: dict[str, Any] | None = None,
    ):
        self.collector_infer_device_raw = str(collector_infer_device or "cpu")
        self.collector_infer_device = resolve_torch_device_alias(
            self.collector_infer_device_raw,
            default="cpu",
        )
        super().__init__(
            env_name=env_name,
            env_cfg_overrides={},
            rl_cfg={},
            device=device,
            collector_device=self.collector_infer_device,
            num_envs=num_envs,
            sim_backend=sim_backend,
        )

        self.learner = learner
        self.env_cfg_override = env_cfg_override
        self.algo_type = algo_type
        self.replay_buffer_n = replay_buffer_n
        self.batch_size = batch_size
        self.learning_starts = max(int(learning_starts), 0)
        self.train_start_threshold = compute_train_start_threshold(
            batch_size,
            self.learning_starts,
            num_envs,
        )
        self.updates_per_step = updates_per_step
        self.policy_frequency = policy_frequency
        self.sync_collection = sync_collection
        self.env_steps_per_sync = env_steps_per_sync
        self.actor_hidden_dim = actor_hidden_dim
        self.use_layer_norm = use_layer_norm
        self.obs_normalization = obs_normalization
        self.actor_kwargs = actor_kwargs or {}
        self.seed = seed
        self._active_logger: OffPolicyLogger | None = None
        self.trace_enabled = trace_enabled
        self.trace_output_dir = trace_output_dir
        self.trace_thread_time = trace_thread_time
        self.trace_cuda_events = trace_cuda_events
        self.nan_guard_cfg = nan_guard_cfg
        self.torch_thread_runtime = torch_thread_runtime

        apply_training_seed(self.seed, torch_runtime=True, cuda=True)
        self.obs_dim, self.action_dim, self.critic_obs_dim = get_env_dims(
            self.env_name,
            sim_backend,
            env_cfg_override,
        )

    def _get_default_device(self) -> str:
        return get_default_device()

    def _build_learner(self):
        return self.learner

    def _collector_fn(self, stop_event, **kwargs):
        off_policy_collector_fn(stop_event=stop_event, **kwargs)

    @staticmethod
    def _sync_logger_replay_counters(logger, replay_buffer) -> None:
        logger.log_collector(
            int(replay_buffer.ptr[0]),
            int(replay_buffer.size[0]),
        )

    def _update_reward_stats_from_replay(
        self,
        replay_buffer,
        start_ptr: int,
        end_ptr: int,
        replay_source: Any | None = None,
    ) -> int:
        return update_reward_stats_from_replay(
            self.learner,
            replay_buffer,
            start_ptr=start_ptr,
            end_ptr=end_ptr,
            num_envs=self.num_envs,
            replay_source=replay_source,
        )

    def close(self) -> None:
        active_logger = getattr(self, "_active_logger", None)
        if active_logger is not None:
            active_logger.close()
            self._active_logger = None
        super().close()

    @staticmethod
    def _drain_metrics(queue, reward_history, reward_components, logger, trace_recorder=None):
        while True:
            try:
                metrics = queue.get_nowait()
            except Exception:
                break
            if "error" in metrics:
                logger.log_status(f"[red]Collector ERROR: {metrics['error']}[/]")
                raise RuntimeError(f"Collector process failed: {metrics['error']}")

            try:
                updated_reward = False
                if "mean_ep_reward" in metrics:
                    reward_history.append(metrics["mean_ep_reward"])
                    updated_reward = True
                if "reward_components" in metrics:
                    reward_components.clear()
                    reward_components.update(metrics["reward_components"])
                if "mean_ep_length" in metrics:
                    logger.update_ep_length(metrics["mean_ep_length"])
                if "collector_timing_ms" in metrics:
                    logger.update_collector_timing(metrics["collector_timing_ms"])
                active_steps_per_sec = metrics.get("collector_active_steps_per_sec")
                if active_steps_per_sec is not None:
                    logger.update_collector_active_steps_per_sec(float(active_steps_per_sec))
                if "timeout_rate" in metrics or "terminated_rate" in metrics:
                    logger.update_done_rates(
                        timeout_rate=float(metrics.get("timeout_rate", 0.0)),
                        terminated_rate=float(metrics.get("terminated_rate", 0.0)),
                    )
                if "total_steps" in metrics and "buffer_size" in metrics:
                    logger.log_collector(
                        metrics["total_steps"],
                        metrics["buffer_size"],
                        metrics.get("mean_ep_reward", 0.0) if updated_reward else 0.0,
                    )
                if trace_recorder and "trace_events" in metrics:
                    trace_recorder.extend(metrics["trace_events"])
            except Exception as exc:
                print(f"[OffPolicyRunner] metrics drain error: {exc}", file=sys.stderr)
                break
