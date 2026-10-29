"""Shared contracts for single-device off-policy runners."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Callable

from unilab.base.contracts.env import EnvFactory
from unilab.base.contracts.probe import probe_env_contract
from unilab.rl.logging import OffPolicyLogger
from unilab.rl.logging.metrics_drain import RewardComponentWindow, drain_collector_metrics
from unilab.rl.runtime.async_runner import AsyncRunner
from unilab.rl.runtime.device import get_default_device
from unilab.rl.runtime.offpolicy.actor_adapter import import_actor_adapter_modules
from unilab.rl.runtime.seed import apply_worker_seed


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


def build_offpolicy_sample_info(
    *,
    replay_batch_size_per_rank: int,
    updates_per_step: int,
) -> dict[str, int]:
    """Describe replay rows and effective learner samples for logging."""
    updates_per_step = max(int(updates_per_step), 0)
    batch_size_per_rank = max(int(replay_batch_size_per_rank), 0)
    return {
        "batch_size_per_rank": batch_size_per_rank,
        "effective_batch_size": batch_size_per_rank,
        "learner_replay_rows_per_iter": batch_size_per_rank * updates_per_step,
    }


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
        env_factory: EnvFactory,
        num_envs: int = 4096,
        replay_buffer_n: int = 1024,
        batch_size: int = 8192,
        learning_starts: int = 0,
        updates_per_step: int = 8,
        policy_frequency: int = 4,
        env_steps_per_sync: int = 1,
        device: str | None = None,
        obs_normalization: bool = False,
        sim_backend: str = "mujoco",
        env_cfg_override: dict | None = None,
        seed: int | None = None,
        trace_enabled: bool = False,
        trace_output_dir: str | None = None,
        trace_thread_time: bool = False,
        trace_cuda_events: bool = True,
        nan_guard_cfg: Any | None = None,
        nan_guard_factory: Callable[[Any, int, bool], Any] | None = None,
        torch_thread_runtime: dict[str, Any] | None = None,
        actor_adapter_modules: Iterable[str] | None = None,
        log_interval: int = 1,
    ):
        if int(env_steps_per_sync) < 1:
            raise ValueError("Off-policy env_steps_per_sync must be >= 1")
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
        self.env_factory = env_factory
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
        self.env_steps_per_sync = int(env_steps_per_sync)
        self.obs_normalization = obs_normalization
        self.seed = seed
        self._active_logger: OffPolicyLogger | None = None
        self.trace_enabled = trace_enabled
        self.trace_output_dir = trace_output_dir
        self.trace_thread_time = trace_thread_time
        self.trace_cuda_events = trace_cuda_events
        self.nan_guard_cfg = nan_guard_cfg
        self.nan_guard_factory = nan_guard_factory
        self.torch_thread_runtime = torch_thread_runtime
        self.log_interval = max(1, int(log_interval))
        # Dotted modules whose import registers custom off-policy actor
        # adapters. Imported here (learner process) and forwarded to the
        # spawn collector, which re-imports them at startup.
        self.actor_adapter_modules = tuple(str(m) for m in (actor_adapter_modules or ()))
        import_actor_adapter_modules(self.actor_adapter_modules)

        apply_worker_seed(self.seed, torch_runtime=True, cuda=True)
        probe = probe_env_contract(
            self.env_factory,
            self.env_cfg_override,
        )
        self.obs_dim = probe.obs_dim
        self.action_dim = probe.action_dim
        self.critic_obs_dim = probe.critic_obs_dim
        # Producer-declared receipt semantics: synchronous producers that
        # never emit receipts relax the CUDA ingress fail-closed gate; the
        # undeclared default keeps requiring a receipt.
        self.produces_completion_receipts = probe.produces_completion_receipts

    def _get_default_device(self) -> str:
        return get_default_device()

    def _build_learner(self):
        return self.learner

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
        try:
            if active_logger is not None:
                active_logger.close()
        finally:
            self._active_logger = None
            # Terminal restoration must not be able to prevent collector,
            # shared-memory, queue, and pipe cleanup.
            super().close()

    @staticmethod
    def _drain_metrics(
        queue,
        reward_history,
        reward_components: RewardComponentWindow,
        logger,
        trace_recorder=None,
    ):
        drain_collector_metrics(
            queue,
            reward_history,
            reward_components,
            logger,
            trace_recorder,
            runner_label="OffPolicyRunner",
            raise_on_collector_error=True,
            require_buffer_size=True,
        )
