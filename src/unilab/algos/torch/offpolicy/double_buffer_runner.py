"""Off-policy runner using device-authoritative bounded-ingress replay."""

from __future__ import annotations

import os
import queue as queue_module
import statistics
import sys
import time
from collections import defaultdict, deque
from contextlib import nullcontext
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict, cast

import torch

if TYPE_CHECKING:
    from unilab.ipc.dp_sync import DpParameterSync

from unilab.algos.torch.offpolicy.runner import (
    OffPolicyRunner,
    build_offpolicy_sample_info,
    build_reward_comparison_metrics,
    compute_train_start_threshold,
    replay_buffer_ready_for_learning,
)
from unilab.algos.torch.offpolicy.thread_budget import (
    format_torch_thread_runtime,
    torch_thread_env,
)
from unilab.algos.torch.offpolicy.worker import off_policy_collector_fn, sample_offpolicy_actions
from unilab.ipc.async_runner import _SPAWN_CTX
from unilab.ipc.inference_slot import SharedInferenceSlot
from unilab.ipc.replay_buffer import DEFAULT_REPLAY_INGRESS_DEPTH, ReplayBuffer
from unilab.ipc.replay_pipelines.gpu_resident import (
    GPUResidentReplayPipeline,
    require_offpolicy_replay_device,
)
from unilab.logging import OffPolicyLogger, TraceRecorder
from unilab.training.seed import derive_worker_seed

# Terminal/W&B display names for the off-policy algo types. Keep these
# user-facing (no internal "Fast*" implementation prefixes).
_ALGO_DISPLAY_NAMES = {
    "sac": "SAC",
    "td3": "TD3",
    "flashsac": "FlashSAC",
}

_DP_METRIC_PREFIX = "metric::"
_DP_REWARD_METRIC_PREFIX = "reward_metric::"
_DP_REWARD_COMPONENT_PREFIX = "reward_component::"
_DP_COLLECTOR_TIMING_PREFIX = "collector_timing::"


class _LogStepPayload(TypedDict):
    metrics: dict[str, float]
    reward: float | None
    reward_metrics: dict[str, float]
    reward_components: dict[str, float]
    train_time: float
    collector_wait_time: float
    replay_batch_wait_time: float
    learner_replay_sample_time: float
    sync_coordination_time: float
    replay_ingress_h2d_submit_time: float
    inference_h2d_time: float
    inference_forward_time: float
    inference_d2h_time: float
    inference_time: float
    iteration_time: float
    extra_info: dict[str, int | float | None]


class _LocalLoggerStatistics(TypedDict):
    total_steps: int
    buffer_size: int
    buffer_target: int
    collector_active_steps_per_sec: float | None
    mean_ep_length: float
    timeout_rate: float
    buffer_utilization: float
    collector_timing: dict[str, float]
    staging_pool_len: int
    staging_pool_max: int


def algo_display_name(algo_type: str) -> str:
    return _ALGO_DISPLAY_NAMES.get(algo_type, algo_type.upper())


class _CollectorDiedError(RuntimeError):
    """Raised when an inference response cannot be delivered to the collector.

    Caught by learn() to trigger the standard cleanup via _fail_collector_died
    using the currently-bound logger/replay_buffer/replay_pipeline/iteration/
    ckpt_path/train_start_wall context. Module-private (_-prefixed)."""


class _LearnerInferenceScheduler:
    """Track fixed inference ticks and the learner update boundary."""

    def __init__(self, *, env_steps_per_sync: int, initial_policy_version: int = 0) -> None:
        if int(env_steps_per_sync) < 1:
            raise ValueError("Off-policy env_steps_per_sync must be >= 1")
        self.env_steps_per_sync = int(env_steps_per_sync)
        self.next_tick = 0
        self.policy_version = int(initial_policy_version)
        self._ticks_since_update = 0
        self._pending_tick: int | None = None

    @property
    def update_ready(self) -> bool:
        return self._ticks_since_update >= self.env_steps_per_sync

    @property
    def pending_tick(self) -> int | None:
        return self._pending_tick

    def record_inference(self, tick_id: int) -> None:
        if self._pending_tick is not None:
            raise RuntimeError("Previous learner inference tick has not been released")
        if int(tick_id) != self.next_tick:
            raise RuntimeError(
                f"Collector inference tick mismatch: expected {self.next_tick}, got {tick_id}"
            )
        self._pending_tick = int(tick_id)
        self.next_tick += 1
        self._ticks_since_update += 1

    def release_pending(self) -> int:
        if self._pending_tick is None:
            raise RuntimeError("Learner inference lost the pending collector tick")
        tick_id = self._pending_tick
        self._pending_tick = None
        return tick_id

    def finish_update(self) -> None:
        if not self.update_ready:
            raise RuntimeError(
                "Learner update started before the configured inference tick boundary"
            )
        if self._pending_tick is not None:
            raise RuntimeError("Learner update completed before releasing the collector tick")
        self._ticks_since_update = 0
        self.policy_version += 1


class DoubleBufferOffPolicyRunner(OffPolicyRunner):
    """Single-device off-policy runner with a double-buffered device batch."""

    REPLAY_BATCH_READY_POLL_SEC = 0.001
    INFERENCE_REQUEST_TIMEOUT_SEC = 30.0

    def __init__(
        self,
        *,
        replay_prefetch_mode: str = "one_tick",
        collector_cpu_ids: list[int] | None = None,
        dp_sync: DpParameterSync | None = None,
        **kwargs,
    ):
        kwargs["device"] = require_offpolicy_replay_device(kwargs.get("device"))
        super().__init__(**kwargs)
        if replay_prefetch_mode != "one_tick":
            raise ValueError(
                "DoubleBufferOffPolicyRunner only supports replay_prefetch_mode='one_tick'"
            )
        self.replay_prefetch_mode = replay_prefetch_mode
        # Per-rank CPU block owned by this rank's collector (multi-GPU DP);
        # merged into the collector-only env override at collector startup.
        self.collector_cpu_ids = list(collector_cpu_ids) if collector_cpu_ids is not None else None
        # Multi-GPU synchronous data parallelism (None = the bit-identical
        # single-rank path): startup model broadcast, then gradient averaging
        # before every actor/critic/temperature optimizer step.
        self.dp_sync = dp_sync
        self._local_logger_statistics: _LocalLoggerStatistics | None = None
        self._attach_dp_gradient_sync()
        self.replay_pack_layout = "packed"
        self.replay_pack_executor = "collector_thread"
        self.replay_h2d_submitter = "auto"
        self.replay_transfer_backend: dict[str, object] = {}
        self.runtime_manifest = {
            "inference_owner": "learner",
            "collector_actor": False,
            "collector_accelerator_context": False,
            "collector_torch_inference": False,
            "learner_actor_reused": True,
            "logger_owner_rank": 0,
            "logger_cross_rank_aggregation": self.dp_sync is not None,
        }
        if self.dp_sync is not None:
            captures_gradient_sync = bool(
                getattr(self.learner, "dp_cuda_graph_gradient_sync", False)
            )
            self.runtime_manifest["dp_sync"] = {
                "world_size": self.dp_sync.world_size,
                "backend": self.dp_sync.backend,
                "mode": "gradient_mean_per_optimizer_step",
                "cuda_graph_optimizer_capture": (
                    "enabled_after_collective_warmup" if captures_gradient_sync else "not_requested"
                ),
                "cuda_graph_collective_warmup": captures_gradient_sync,
            }

    def _attach_dp_gradient_sync(self) -> None:
        if self.dp_sync is None:
            return
        setter = getattr(self.learner, "set_gradient_sync", None)
        if not callable(setter):
            raise TypeError(
                f"{type(self.learner).__name__} must implement set_gradient_sync() "
                "for multi-GPU data parallelism"
            )
        setter(
            self.dp_sync.allreduce_gradients,
            graph_replay_recorder=self.dp_sync.record_cuda_graph_gradient_replay,
        )

    def _dp_initial_sync_tensors(self) -> dict[str, torch.Tensor]:
        """Live model-state references broadcast once before collection."""
        initial_tensors = getattr(self.learner, "dp_initial_sync_tensors", None)
        if not callable(initial_tensors):
            raise TypeError(
                f"{type(self.learner).__name__} must implement dp_initial_sync_tensors() "
                "for multi-GPU data parallelism"
            )
        return cast(dict[str, torch.Tensor], initial_tensors())

    def _dp_init_broadcast(self) -> None:
        """Align initial parameters from rank 0 before the collector starts.

        Ranks train with per-rank seeds, so without this broadcast each
        rank's actor would serve different inference from the first tick.
        """
        if self.dp_sync is None:
            return
        self.dp_sync.start()
        self.dp_sync.broadcast_from_rank0(self._dp_initial_sync_tensors())
        if bool(getattr(self.learner, "dp_cuda_graph_gradient_sync", False)):
            self.dp_sync.prepare_cuda_graph_collectives()

    def _collect_dp_sync_metrics(self, iter_metrics: defaultdict[str, list]) -> None:
        """Move per-optimizer collective timing into this iteration's metrics."""
        if self.dp_sync is None:
            return
        sync_time, sync_calls = self.dp_sync.take_gradient_sync_metrics()
        if sync_calls > 0:
            iter_metrics["dp_sync_time"].append(sync_time)
            iter_metrics["dp_gradient_sync_calls"].append(float(sync_calls))

    def _aggregate_log_statistics(
        self,
        logger: OffPolicyLogger,
        *,
        metrics: dict[str, float],
        reward: float | None,
        reward_metrics: dict[str, float],
        reward_components: dict[str, float],
        train_time: float,
        collector_wait_time: float,
        replay_batch_wait_time: float,
        learner_replay_sample_time: float,
        sync_coordination_time: float,
        replay_ingress_h2d_submit_time: float,
        inference_h2d_time: float,
        inference_forward_time: float,
        inference_d2h_time: float,
        inference_time: float,
        iteration_time: float,
        extra_info: dict[str, int | float | None],
    ) -> _LogStepPayload:
        """Return one log-step payload, reduced across ranks when DP is active.

        Model/reward/timing scalars are means. Concurrent work rates and
        capacity/sample counters are totals, so the rank-0 logger reports
        aggregate collector Steps/s and aggregate learner Samples/s. Optional
        collector fields use the sparse presence-mask contract implemented by
        ``DpParameterSync.allreduce_statistics``.
        """
        payload: _LogStepPayload = {
            "metrics": metrics,
            "reward": reward,
            "reward_metrics": reward_metrics,
            "reward_components": reward_components,
            "train_time": train_time,
            "collector_wait_time": collector_wait_time,
            "replay_batch_wait_time": replay_batch_wait_time,
            "learner_replay_sample_time": learner_replay_sample_time,
            "sync_coordination_time": sync_coordination_time,
            "replay_ingress_h2d_submit_time": replay_ingress_h2d_submit_time,
            "inference_h2d_time": inference_h2d_time,
            "inference_forward_time": inference_forward_time,
            "inference_d2h_time": inference_d2h_time,
            "inference_time": inference_time,
            "iteration_time": iteration_time,
            "extra_info": extra_info,
        }
        if self.dp_sync is None:
            return payload

        # ``logger`` also receives collector telemetry asynchronously. Keep a
        # rank-local snapshot before replacing its presentation state with the
        # aggregate; the next iteration restores this snapshot so a field that
        # is reported only every few collector cycles is never summed twice.
        self._local_logger_statistics = {
            "total_steps": logger._total_steps,
            "buffer_size": logger._buffer_size,
            "buffer_target": logger._buffer_target,
            "collector_active_steps_per_sec": logger._collector_active_steps_per_sec,
            "mean_ep_length": logger._mean_ep_length,
            "timeout_rate": logger._timeout_rate,
            "buffer_utilization": logger._buffer_utilization,
            "collector_timing": dict(logger._collector_timing),
            "staging_pool_len": logger._staging_pool_len,
            "staging_pool_max": logger._staging_pool_max,
        }

        mean: dict[str, float] = {
            **{f"{_DP_METRIC_PREFIX}{key}": float(value) for key, value in metrics.items()},
            **{
                f"{_DP_REWARD_METRIC_PREFIX}{key}": float(value)
                for key, value in reward_metrics.items()
            },
            **{
                f"{_DP_REWARD_COMPONENT_PREFIX}{key}": float(value)
                for key, value in reward_components.items()
            },
            "timing::train_time": train_time,
            "timing::collector_wait_time": collector_wait_time,
            "timing::replay_batch_wait_time": replay_batch_wait_time,
            "timing::learner_replay_sample_time": learner_replay_sample_time,
            "timing::sync_coordination_time": sync_coordination_time,
            "timing::replay_ingress_h2d_submit_time": replay_ingress_h2d_submit_time,
            "timing::inference_h2d_time": inference_h2d_time,
            "timing::inference_forward_time": inference_forward_time,
            "timing::inference_d2h_time": inference_d2h_time,
            "timing::inference_time": inference_time,
            "timing::iteration_time": iteration_time,
            "logger::timeout_rate": float(logger._timeout_rate),
            "logger::buffer_utilization": float(logger._buffer_utilization),
            "extra::batch_size_per_rank": float(extra_info.get("batch_size_per_rank", 0) or 0),
        }
        if reward is not None:
            mean["reward::mean"] = float(reward)
        if logger._mean_ep_length > 0:
            mean["logger::mean_ep_length"] = float(logger._mean_ep_length)
        mean.update(
            {
                f"{_DP_COLLECTOR_TIMING_PREFIX}{key}": float(value)
                for key, value in logger._collector_timing.items()
            }
        )

        throughput_steps = int(extra_info.get("throughput_steps", 0) or 0)
        learner_samples = int(extra_info.get("learner_samples_per_iter", 0) or 0)
        total: dict[str, float] = {
            "logger::total_steps": float(logger._total_steps),
            "logger::buffer_size": float(logger._buffer_size),
            "logger::buffer_target": float(logger._buffer_target),
            "logger::staging_pool_len": float(logger._staging_pool_len),
            "logger::staging_pool_max": float(logger._staging_pool_max),
            "extra::throughput_steps": float(throughput_steps),
            "extra::effective_batch_size": float(extra_info.get("effective_batch_size", 0) or 0),
            "extra::replay_samples_per_iter": float(
                extra_info.get("replay_samples_per_iter", 0) or 0
            ),
            "extra::learner_samples_per_iter": float(learner_samples),
        }
        if iteration_time > 0:
            total["rate::steps_per_sec"] = throughput_steps / iteration_time
            total["rate::learner_samples_per_sec"] = learner_samples / iteration_time
        collector_active_steps_per_sec = extra_info.get("collector_active_steps_per_sec")
        if collector_active_steps_per_sec is not None:
            total["rate::collector_active_steps_per_sec"] = float(collector_active_steps_per_sec)

        aggregated = self.dp_sync.allreduce_statistics(mean=mean, total=total)

        logger._total_steps = int(round(aggregated["logger::total_steps"]))
        logger._buffer_size = int(round(aggregated["logger::buffer_size"]))
        logger._buffer_target = int(round(aggregated["logger::buffer_target"]))
        logger._staging_pool_len = int(round(aggregated["logger::staging_pool_len"]))
        logger._staging_pool_max = int(round(aggregated["logger::staging_pool_max"]))
        logger._timeout_rate = aggregated["logger::timeout_rate"]
        logger._buffer_utilization = aggregated["logger::buffer_utilization"]
        if "logger::mean_ep_length" in aggregated:
            logger._mean_ep_length = aggregated["logger::mean_ep_length"]
        logger._collector_timing = {
            key.removeprefix(_DP_COLLECTOR_TIMING_PREFIX): value
            for key, value in aggregated.items()
            if key.startswith(_DP_COLLECTOR_TIMING_PREFIX)
        }

        aggregated_extra_info: dict[str, int | float | None] = {
            "throughput_steps": int(round(aggregated["extra::throughput_steps"])),
            "steps_per_sec": aggregated.get("rate::steps_per_sec"),
            "learner_samples_per_sec": aggregated.get("rate::learner_samples_per_sec"),
            "collector_active_steps_per_sec": aggregated.get(
                "rate::collector_active_steps_per_sec"
            ),
            "batch_size_per_rank": int(round(aggregated["extra::batch_size_per_rank"])),
            "effective_batch_size": int(round(aggregated["extra::effective_batch_size"])),
            "replay_samples_per_iter": int(round(aggregated["extra::replay_samples_per_iter"])),
            "learner_samples_per_iter": int(round(aggregated["extra::learner_samples_per_iter"])),
        }
        return {
            "metrics": {
                key.removeprefix(_DP_METRIC_PREFIX): value
                for key, value in aggregated.items()
                if key.startswith(_DP_METRIC_PREFIX)
            },
            "reward": aggregated.get("reward::mean"),
            "reward_metrics": {
                key.removeprefix(_DP_REWARD_METRIC_PREFIX): value
                for key, value in aggregated.items()
                if key.startswith(_DP_REWARD_METRIC_PREFIX)
            },
            "reward_components": {
                key.removeprefix(_DP_REWARD_COMPONENT_PREFIX): value
                for key, value in aggregated.items()
                if key.startswith(_DP_REWARD_COMPONENT_PREFIX)
            },
            "train_time": aggregated["timing::train_time"],
            "collector_wait_time": aggregated["timing::collector_wait_time"],
            "replay_batch_wait_time": aggregated["timing::replay_batch_wait_time"],
            "learner_replay_sample_time": aggregated["timing::learner_replay_sample_time"],
            "sync_coordination_time": aggregated["timing::sync_coordination_time"],
            "replay_ingress_h2d_submit_time": aggregated["timing::replay_ingress_h2d_submit_time"],
            "inference_h2d_time": aggregated["timing::inference_h2d_time"],
            "inference_forward_time": aggregated["timing::inference_forward_time"],
            "inference_d2h_time": aggregated["timing::inference_d2h_time"],
            "inference_time": aggregated["timing::inference_time"],
            "iteration_time": aggregated["timing::iteration_time"],
            "extra_info": aggregated_extra_info,
        }

    def _restore_local_logger_statistics(self, logger: OffPolicyLogger) -> None:
        """Restore per-rank collector state after the previous aggregate log step."""
        state = self._local_logger_statistics
        if state is None:
            return
        logger._total_steps = int(state["total_steps"])
        logger._buffer_size = int(state["buffer_size"])
        logger._buffer_target = int(state["buffer_target"])
        active_rate = state["collector_active_steps_per_sec"]
        logger._collector_active_steps_per_sec = None if active_rate is None else float(active_rate)
        logger._mean_ep_length = float(state["mean_ep_length"])
        logger._timeout_rate = float(state["timeout_rate"])
        logger._buffer_utilization = float(state["buffer_utilization"])
        logger._collector_timing = dict(state["collector_timing"])
        logger._staging_pool_len = int(state["staging_pool_len"])
        logger._staging_pool_max = int(state["staging_pool_max"])
        self._local_logger_statistics = None

    def _logger_backend(self, requested: str) -> str:
        """Only rank 0 owns terminal and external logging backends."""
        if not self._is_primary_rank():
            return "no_print"
        return requested

    def _is_primary_rank(self) -> bool:
        return self.dp_sync is None or self.dp_sync.rank == 0

    def _save_checkpoint(
        self,
        *,
        log_dir: str,
        iteration: int,
        logger: OffPolicyLogger,
    ) -> str | None:
        """Persist the single canonical checkpoint from rank 0 only."""
        if not self._is_primary_rank():
            return None
        ckpt_path = os.path.join(log_dir, f"model_{iteration}.pt")
        torch.save(self.learner.get_state_dict(), ckpt_path)
        logger.log_save(ckpt_path)
        return ckpt_path

    def close(self) -> None:
        try:
            # Rank 0 owns the live terminal, and every rank owns a collector and
            # shared IPC resources. Release those before NCCL teardown, which
            # may wait on a peer during Ctrl+C shutdown.
            super().close()
        finally:
            if self.dp_sync is not None:
                try:
                    if bool(getattr(self.learner, "dp_cuda_graph_gradient_sync", False)):
                        release_cuda_graphs = getattr(self.learner, "release_cuda_graphs", None)
                        if not callable(release_cuda_graphs):
                            raise TypeError(
                                f"{type(self.learner).__name__} must implement "
                                "release_cuda_graphs() before an NCCL process group with "
                                "captured collectives is destroyed"
                            )
                        release_cuda_graphs()
                finally:
                    self.dp_sync.close()

    def _collector_env_cfg_override(self) -> dict | None:
        """Env override copy for the collector process, with per-rank CPU ids.

        MuJoCo sizes its BatchEnvPool worker count from ``len(cpu_ids)``, so
        the affinity list must only reach the collector's copy — never the
        learner-side probe envs, which keep the base override untouched.
        """
        if self.collector_cpu_ids is None:
            return self.env_cfg_override
        return {**(self.env_cfg_override or {}), "cpu_ids": list(self.collector_cpu_ids)}

    def _wait_for_inference_request(
        self,
        queue,
        *,
        expected_tick: int,
        replay_pipeline,
        metrics_queue,
        reward_history,
        latest_reward_components,
        logger,
        trace_recorder,
        replay_buffer,
        ckpt_path: str | None,
        train_start_wall: float,
    ) -> int:
        deadline = time.monotonic() + self.INFERENCE_REQUEST_TIMEOUT_SEC
        while True:
            try:
                tick_id = int(queue.get(timeout=0.1))
            except queue_module.Empty:
                replay_pipeline.progress()
                self._drain_metrics(
                    metrics_queue,
                    reward_history,
                    latest_reward_components,
                    logger,
                    trace_recorder,
                    log_collector_reward=self.dp_sync is None,
                )
                if not self._check_collector_alive():
                    self._fail_collector_died(
                        logger,
                        replay_buffer,
                        replay_pipeline,
                        expected_tick,
                        ckpt_path,
                        train_start_wall,
                    )
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"Timed out waiting for collector inference tick {expected_tick}"
                    )
                continue
            if tick_id != int(expected_tick):
                raise RuntimeError(
                    f"Collector inference tick mismatch: expected {expected_tick}, got {tick_id}"
                )
            return tick_id

    def _serve_learner_inference(
        self,
        inference_slot: SharedInferenceSlot,
        *,
        tick_id: int,
        policy_version: int,
        obs_device: torch.Tensor,
        dones_device: torch.Tensor,
        trace_recorder: TraceRecorder | None,
    ) -> dict[str, float]:
        device = torch.device(self.device)
        h2d_start_ns = time.perf_counter_ns()
        inference_slot.copy_observation_to(
            tick_id=tick_id,
            observations=obs_device,
            dones=dones_device,
            non_blocking=False,
        )
        h2d_end_ns = time.perf_counter_ns()

        actor_obs = obs_device[:, : self.obs_dim]
        actor_context = obs_device[:, self.obs_dim :] if self.algo_type == "hora_sac" else None
        if self.obs_normalization:
            actor_obs = self.learner.obs_normalizer(actor_obs, update=False)
        forward_start_ns = time.perf_counter_ns()
        with torch.no_grad():
            actions_device = sample_offpolicy_actions(
                actor=self.learner.actor,
                algo_type=self.algo_type,
                obs_torch=actor_obs,
                prev_dones_torch=dones_device,
                priv_info_torch=actor_context,
            )
        if device.type == "cuda":
            torch.cuda.current_stream(device).synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()
        forward_end_ns = time.perf_counter_ns()

        d2h_start_ns = time.perf_counter_ns()
        inference_slot.publish_action(
            tick_id=tick_id,
            policy_version=policy_version,
            actions=actions_device,
            non_blocking=False,
        )
        d2h_end_ns = time.perf_counter_ns()
        timings = {
            "inference_h2d_time": (h2d_end_ns - h2d_start_ns) / 1e9,
            "inference_forward_time": (forward_end_ns - forward_start_ns) / 1e9,
            "inference_d2h_time": (d2h_end_ns - d2h_start_ns) / 1e9,
            "inference_time": (d2h_end_ns - h2d_start_ns) / 1e9,
        }
        if trace_recorder:
            trace_args = {"tick_id": tick_id, "policy_version": policy_version}
            trace_recorder.add_slice(
                "learner/inference_h2d",
                category="learner_inference",
                start_ns=h2d_start_ns,
                end_ns=h2d_end_ns,
                args=trace_args,
            )
            trace_recorder.add_slice(
                "learner/inference_forward",
                category="learner_inference",
                start_ns=forward_start_ns,
                end_ns=forward_end_ns,
                args=trace_args,
            )
            trace_recorder.add_slice(
                "learner/inference_d2h",
                category="learner_inference",
                start_ns=d2h_start_ns,
                end_ns=d2h_end_ns,
                args=trace_args,
            )
            trace_recorder.add_slice(
                "learner/inference",
                category="learner_inference",
                start_ns=h2d_start_ns,
                end_ns=d2h_end_ns,
                args=trace_args,
            )
            trace_recorder.add_counter(
                "policy_version",
                policy_version,
                category="learner_inference",
            )
        return timings

    def _fail_collector_died(
        self,
        logger,
        replay_buffer,
        replay_pipeline,
        iteration: int,
        ckpt_path: str | None,
        train_start_wall: float,
    ) -> None:
        logger.log_status("[red]ERROR: Collector died[/]")
        self._sync_logger_replay_counters(logger, replay_buffer)
        logger.close()
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
        raise RuntimeError("Collector process died during off-policy training")

    def _publish_inference_response(
        self,
        queue,
        *,
        value: int = 1,
        timeout: float = 5.0,
        label: str = "inference_response",
    ) -> None:
        """Publish an inference response tick with timeout and liveness checks.

        Raises _CollectorDiedError if collector is dead or queue stays full
        beyond timeout. Caller (learn) must catch and dispatch to
        _fail_collector_died for full cleanup. This avoids an unbounded blocking
        put when the collector dies before consuming the previous response.
        """
        deadline = time.monotonic() + timeout
        while True:
            try:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise _CollectorDiedError(f"{label} (queue full timeout)")
                queue.put(int(value), timeout=min(0.5, remaining))
                return
            except queue_module.Full:
                if not self._check_collector_alive():
                    raise _CollectorDiedError(f"{label} (collector dead)")
                # else loop & retry until deadline

    def _release_inference_tick(
        self,
        queue,
        *,
        inference_scheduler: _LearnerInferenceScheduler,
        replay_buffer,
        trace_recorder: TraceRecorder | None,
    ) -> int | None:
        """Release the action immediately and freeze the next replay boundary."""
        tick_id = inference_scheduler.release_pending()
        next_prepare_min_snapshot_ptr = None
        if inference_scheduler.update_ready:
            next_prepare_min_snapshot_ptr = replay_buffer.published_ptr + (
                self.num_envs * self.env_steps_per_sync
            )
        release_start_ns = time.perf_counter_ns()
        self._publish_inference_response(
            queue,
            value=tick_id,
            label="inference_response",
        )
        if trace_recorder:
            trace_recorder.add_slice(
                "learner/inference_response",
                category="learner_inference",
                start_ns=release_start_ns,
                end_ns=time.perf_counter_ns(),
                args={
                    "tick_id": tick_id,
                    "policy_version": inference_scheduler.policy_version,
                    "next_prepare_min_snapshot_ptr": next_prepare_min_snapshot_ptr,
                },
            )
        return next_prepare_min_snapshot_ptr

    def _wait_for_replay_batch_ready(
        self,
        replay_pipeline,
        tick_id: int,
        sample_count: int,
        metrics_queue,
        reward_history,
        latest_reward_components,
        logger,
        trace_recorder,
        replay_buffer,
        ckpt_path: str | None,
        train_start_wall: float,
    ) -> bool:
        if not replay_pipeline.batch_ready(tick_id, sample_count):
            replay_pipeline.start_prepare(tick_id, sample_count)
        while not replay_pipeline.batch_ready(tick_id, sample_count):
            self._drain_metrics(
                metrics_queue,
                reward_history,
                latest_reward_components,
                logger,
                trace_recorder,
                log_collector_reward=self.dp_sync is None,
            )
            if not self._check_collector_alive():
                self._fail_collector_died(
                    logger,
                    replay_buffer,
                    replay_pipeline,
                    tick_id,
                    ckpt_path,
                    train_start_wall,
                )
            time.sleep(self.REPLAY_BATCH_READY_POLL_SEC)
        return True

    def learn(
        self,
        max_iterations: int = 1500,
        save_interval: int = 50,
        log_dir: str = "logs",
        logger_type: str = "tensorboard",
    ) -> None:
        if self._is_primary_rank():
            os.makedirs(log_dir, exist_ok=True)
        trace_output_path = None
        trace_recorder: TraceRecorder | None = None
        if self.trace_enabled and self._is_primary_rank():
            trace_root = Path(self.trace_output_dir or log_dir)
            trace_output_path = trace_root / "perfetto_offpolicy_timeline.json"
            trace_recorder = TraceRecorder("offpolicy_learner")
        train_start_wall = time.time()
        best_mean_reward = float("-inf")
        last_mean_reward = 0.0
        ckpt_path: str | None = None
        iteration = 0

        # --- memory budget check ---
        from unilab.ipc.memory_budget import (
            estimate_offpolicy_bytes,
            raise_if_shared_memory_over_budget,
            warn_if_over_budget,
        )

        graph_packed_staging_supported = self.algo_type == "sac" or (
            self.algo_type == "flashsac"
            and bool(getattr(self.learner, "supports_cuda_graph_packed_staging", False))
        )
        use_critic_graph_packed_source = (
            graph_packed_staging_supported
            and bool(getattr(self.learner, "use_cuda_graph_critic_packed_staging", False))
            and self.critic_obs_dim > 0
        )
        use_sac_graph_pack_layout = use_critic_graph_packed_source and bool(
            getattr(self.learner, "use_cuda_graph_actor_packed_staging", False)
        )
        use_critic_graph_packed_source = (
            use_critic_graph_packed_source and not use_sac_graph_pack_layout
        )
        mem_est = estimate_offpolicy_bytes(
            num_envs=self.num_envs,
            replay_buffer_n=self.replay_buffer_n,
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            critic_dim=self.critic_obs_dim,
            ingress_depth=DEFAULT_REPLAY_INGRESS_DEPTH,
        )
        warn_if_over_budget(mem_est, label=f"Off-policy ({self.algo_type})")
        raise_if_shared_memory_over_budget(mem_est, label=f"Off-policy ({self.algo_type})")

        # --- bounded collector ingress (the complete ring lives on device) ---
        buffer_capacity = self.replay_buffer_n * self.num_envs
        replay_buffer = ReplayBuffer(
            capacity=buffer_capacity,
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            device=self.device,
            critic_dim=self.critic_obs_dim,
            ingress_slot_rows=self.num_envs,
            ingress_depth=DEFAULT_REPLAY_INGRESS_DEPTH,
        )
        self._shared_resources.append(replay_buffer)
        replay_buffer.trace_recorder = trace_recorder
        replay_buffer.trace_thread_time = self.trace_thread_time
        replay_buffer.trace_cuda_events = self.trace_cuda_events

        # --- authoritative device ring and hot/cold learner batches ---
        sample_count = self.batch_size * self.updates_per_step
        replay_pipeline = GPUResidentReplayPipeline(
            replay_buffer,
            device=self.device,
            sample_count=sample_count,
            base_seed=int(self.seed or 0),
            trace_recorder=trace_recorder,
            trace_cuda_events=self.trace_cuda_events,
            pack_layout="sac_graph" if use_sac_graph_pack_layout else "packed",
            use_critic_graph_packed_source=use_critic_graph_packed_source,
        )
        self._shared_resources.insert(0, replay_pipeline)
        self.replay_h2d_submitter = getattr(
            replay_pipeline,
            "h2d_submitter",
            self.replay_h2d_submitter,
        )
        self.replay_transfer_backend = getattr(
            replay_pipeline,
            "transfer_manifest",
            {},
        )
        self.runtime_manifest.update(
            {
                "replay_h2d_submitter": self.replay_h2d_submitter,
                "replay_device_submission_thread": self.replay_transfer_backend.get(
                    "device_submission_thread"
                ),
            }
        )

        actor_context_dim = int(getattr(self.learner, "priv_info_dim", 0))
        inference_input_dim = self.obs_dim + actor_context_dim
        inference_slot = SharedInferenceSlot(
            self.num_envs,
            inference_input_dim,
            self.action_dim,
        )
        self._shared_resources.append(inference_slot)
        inference_obs_device = torch.empty(
            (self.num_envs, inference_input_dim),
            dtype=torch.float32,
            device=self.device,
        )
        inference_dones_device = torch.empty(
            self.num_envs,
            dtype=torch.float32,
            device=self.device,
        )
        self.runtime_manifest["inference_slot_bytes"] = inference_slot.nbytes

        # --- logger ---
        logger = OffPolicyLogger(
            algo_name=algo_display_name(self.algo_type),
            max_iterations=max_iterations,
            num_envs=self.num_envs * (self.dp_sync.world_size if self.dp_sync is not None else 1),
            env_name=self.env_name,
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            num_gpus=(self.dp_sync.world_size if self.dp_sync is not None else 1),
            log_dir=log_dir,
            log_backend=self._logger_backend(logger_type),
        )
        logger.update_runtime_manifest(self.runtime_manifest)
        if hasattr(self.learner, "use_symmetry") and self.learner.use_symmetry:
            logger.log_status("Symmetry augmentation: enabled")
        logger.log_status(format_torch_thread_runtime(self.torch_thread_runtime))
        logger.log_status("Replay storage: device-authoritative bounded ingress")
        logger.log_status(f"Replay prefetch mode: {self.replay_prefetch_mode}")
        logger.log_status(f"Replay pack layout: {self.replay_pack_layout}")
        logger.log_status(f"Replay pack executor: {self.replay_pack_executor}")
        logger.log_status(f"Replay H2D submitter: {self.replay_h2d_submitter}")
        if self.replay_transfer_backend:
            logger.log_status(
                "Replay transfer backend: "
                f"{self.replay_transfer_backend.get('backend')} "
                f"({self.replay_transfer_backend.get('device_family')})"
            )
        logger.log_status(f"Inference owner: learner.actor ({self.device})")
        logger.log_status("Collector model/accelerator ownership: none")
        logger.log_status("Replay learner lightweight: fixed (log_interval=1)")
        self._active_logger = logger
        logger.start()
        try:
            # --- inference coordination queues ---
            inference_request_queue = _SPAWN_CTX.Queue(maxsize=1)
            inference_response_queue = _SPAWN_CTX.Queue(maxsize=1)

            metrics_queue = _SPAWN_CTX.Queue(maxsize=100)

            # --- DP init broadcast must land before the collector's first ---
            # --- inference request reaches learner.actor ---
            self._dp_init_broadcast()

            # --- start collector ---
            collector_kwargs = {
                "env_name": self.env_name,
                "num_envs": self.num_envs,
                "replay_buffer": replay_buffer,
                "algo_type": self.algo_type,
                "metrics_queue": metrics_queue,
                "inference_request_queue": inference_request_queue,
                "inference_response_queue": inference_response_queue,
                "sim_backend": self.sim_backend,
                "env_cfg_override": self._collector_env_cfg_override(),
                "inference_slot": inference_slot,
                "seed": derive_worker_seed(self.seed, worker_index=0),
                "trace_enabled": self.trace_enabled,
                "trace_thread_time": self.trace_thread_time,
                "nan_guard_cfg": self.nan_guard_cfg,
                "torch_thread_runtime": self.torch_thread_runtime,
            }
            with torch_thread_env(self.torch_thread_runtime, role="collector"):
                self._start_collector(
                    target_fn=off_policy_collector_fn,
                    kwargs={"stop_event": self._stop_event, **collector_kwargs},
                )

            time.sleep(0.5)

            reward_history: deque = deque(maxlen=100)
            latest_reward_components: dict[str, float] = {}
            has_logged_reward = False
            last_buf_log = 0
            write_read_ema = 0.0
            reward_stats_ptr = 0
            train_start_threshold = self.train_start_threshold
            prepared_tick: int | None = None
            inference_scheduler = _LearnerInferenceScheduler(
                env_steps_per_sync=self.env_steps_per_sync,
                initial_policy_version=int(getattr(self.learner, "update_count", 0)),
            )

            if trace_recorder:
                manifest_ns = time.perf_counter_ns()
                trace_recorder.add_slice(
                    "learner/runtime_manifest",
                    category="learner",
                    start_ns=manifest_ns,
                    end_ns=manifest_ns,
                    args=dict(self.runtime_manifest),
                )

            training_e2e_start_ns = 0

            # ---- training loop ----
            for iteration in range(1, max_iterations + 1):
                self._restore_local_logger_statistics(logger)
                iteration_start = time.perf_counter()
                # -- wait for data --
                wait_start = time.perf_counter()
                wait_start_ns = time.perf_counter_ns()
                sync_coordination_time = 0.0
                collector_wait_overhead = 0.0
                inference_h2d_time = 0.0
                inference_forward_time = 0.0
                inference_d2h_time = 0.0
                inference_time = 0.0
                next_prepare_min_snapshot_ptr: int | None = None
                while True:
                    request_tick = self._wait_for_inference_request(
                        inference_request_queue,
                        expected_tick=inference_scheduler.next_tick,
                        replay_pipeline=replay_pipeline,
                        metrics_queue=metrics_queue,
                        reward_history=reward_history,
                        latest_reward_components=latest_reward_components,
                        logger=logger,
                        trace_recorder=trace_recorder,
                        replay_buffer=replay_buffer,
                        ckpt_path=ckpt_path,
                        train_start_wall=train_start_wall,
                    )
                    inference_timings = self._serve_learner_inference(
                        inference_slot,
                        tick_id=request_tick,
                        policy_version=inference_scheduler.policy_version,
                        obs_device=inference_obs_device,
                        dones_device=inference_dones_device,
                        trace_recorder=trace_recorder,
                    )
                    inference_h2d_time += inference_timings["inference_h2d_time"]
                    inference_forward_time += inference_timings["inference_forward_time"]
                    inference_d2h_time += inference_timings["inference_d2h_time"]
                    inference_time += inference_timings["inference_time"]
                    collector_wait_overhead += inference_timings["inference_time"]
                    inference_scheduler.record_inference(request_tick)
                    _coord_t = time.perf_counter()
                    frozen_prepare_ptr = self._release_inference_tick(
                        inference_response_queue,
                        inference_scheduler=inference_scheduler,
                        replay_buffer=replay_buffer,
                        trace_recorder=trace_recorder,
                    )
                    _coord_d = time.perf_counter() - _coord_t
                    sync_coordination_time += _coord_d
                    collector_wait_overhead += _coord_d
                    if frozen_prepare_ptr is not None:
                        next_prepare_min_snapshot_ptr = frozen_prepare_ptr

                    self._drain_metrics(
                        metrics_queue,
                        reward_history,
                        latest_reward_components,
                        logger,
                        trace_recorder,
                        log_collector_reward=self.dp_sync is None,
                    )
                    replay_pipeline.progress(wait=True)
                    cur_size = int(replay_buffer.size[0])
                    replay_ready = replay_buffer_ready_for_learning(
                        cur_size,
                        batch_size=self.batch_size,
                        learning_starts=self.learning_starts,
                        num_envs=self.num_envs,
                    )
                    if replay_ready and inference_scheduler.update_ready:
                        if prepared_tick != iteration:
                            replay_pipeline.start_prepare(iteration, sample_count)
                            prepared_tick = iteration
                        break
                    if cur_size - last_buf_log >= self.num_envs * 10:
                        last_buf_log = cur_size
                        _fill_t = time.perf_counter()
                        logger.log_buffer_fill(cur_size, train_start_threshold)
                        collector_wait_overhead += time.perf_counter() - _fill_t

                collector_wait_time = time.perf_counter() - wait_start - collector_wait_overhead
                if trace_recorder:
                    trace_recorder.add_slice(
                        "learner/wait_for_data",
                        category="learner",
                        start_ns=wait_start_ns,
                        end_ns=time.perf_counter_ns(),
                        args={"iteration": iteration},
                    )
                if iteration == 1:
                    train_start_wall = logger.start_training_timer()
                    if trace_recorder:
                        training_e2e_start_ns = time.perf_counter_ns()
                self._drain_metrics(
                    metrics_queue,
                    reward_history,
                    latest_reward_components,
                    logger,
                    trace_recorder,
                    log_collector_reward=self.dp_sync is None,
                )
                _reward_stats_ns = time.perf_counter_ns()
                reward_stats_ptr = self._update_reward_stats_from_replay(
                    replay_buffer,
                    reward_stats_ptr,
                    int(replay_buffer.ptr[0]),
                    replay_source=replay_pipeline,
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
                learner = self.learner

                with nullcontext():
                    _sample_ns = time.perf_counter_ns()
                    _replay_batch_wait_start = time.perf_counter()
                    batch_ready = replay_pipeline.batch_ready(iteration, sample_count)
                    _wait_batch_ns = time.perf_counter_ns()
                    if not batch_ready:
                        batch_ready = self._wait_for_replay_batch_ready(
                            replay_pipeline,
                            iteration,
                            sample_count,
                            metrics_queue,
                            reward_history,
                            latest_reward_components,
                            logger,
                            trace_recorder,
                            replay_buffer,
                            ckpt_path,
                            train_start_wall,
                        )
                    replay_batch_wait_time = time.perf_counter() - _replay_batch_wait_start
                    if trace_recorder:
                        trace_recorder.add_slice(
                            "learner/wait_for_replay_batch",
                            category="learner",
                            start_ns=_wait_batch_ns,
                            end_ns=time.perf_counter_ns(),
                            args={"iteration": iteration, "batch_ready": batch_ready},
                        )
                    replay_sample_start = time.perf_counter()
                    large_batch = replay_pipeline.sample_large_batch(
                        tick_id=iteration,
                        sample_count=sample_count,
                    )
                    learner_replay_sample_time = time.perf_counter() - replay_sample_start
                    replay_ingress_h2d_submit_time = float(
                        getattr(replay_pipeline, "last_incremental_h2d_time_s", 0.0)
                    )
                    if iteration < max_iterations:
                        if next_prepare_min_snapshot_ptr is None:
                            raise RuntimeError(
                                "Off-policy replay prefetch lost the inference update boundary"
                            )
                        replay_pipeline.start_prepare(
                            iteration + 1,
                            sample_count,
                            min_snapshot_ptr=next_prepare_min_snapshot_ptr,
                        )
                        prepared_tick = iteration + 1
                    if trace_recorder:
                        trace_recorder.add_slice(
                            "learner/replay_sample",
                            category="learner",
                            start_ns=_sample_ns,
                            end_ns=time.perf_counter_ns(),
                            args={
                                "total_batch": sample_count,
                                "pipeline": "gpu_resident",
                                "batch_ready": batch_ready,
                                "prefetch_mode": self.replay_prefetch_mode,
                                "replay_pack_layout": self.replay_pack_layout,
                                "replay_pack_executor": self.replay_pack_executor,
                                "replay_h2d_submitter": self.replay_h2d_submitter,
                                "replay_transfer_backend": self.replay_transfer_backend,
                                "prepared_tick": prepared_tick,
                                "explicit_compute_stream": False,
                            },
                        )

                    train_start = time.perf_counter()
                    train_phase_start_ns = time.perf_counter_ns()

                    for update_idx in range(self.updates_per_step):
                        s = update_idx * self.batch_size
                        e = s + self.batch_size
                        batch = {k: v[s:e] for k, v in large_batch.items()}
                        read_critic_graph_metrics = update_idx == self.updates_per_step - 1

                        _critic_ns = time.perf_counter_ns()
                        if getattr(learner, "use_cuda_graph_critic", False) and hasattr(
                            learner, "update_critic_cuda_graph"
                        ):
                            critic_metrics = learner.update_critic_cuda_graph(
                                batch,
                                read_metrics=read_critic_graph_metrics,
                            )
                        else:
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

                        if update_idx % self.policy_frequency == 0:
                            next_actor_update = update_idx + self.policy_frequency
                            read_actor_graph_metrics = next_actor_update >= self.updates_per_step
                            _actor_ns = time.perf_counter_ns()
                            if getattr(learner, "use_cuda_graph_actor", False) and hasattr(
                                learner, "update_actor_cuda_graph"
                            ):
                                actor_metrics = learner.update_actor_cuda_graph(
                                    batch,
                                    read_metrics=read_actor_graph_metrics,
                                )
                            else:
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

                        _target_ns = time.perf_counter_ns()
                        learner.soft_update_target()
                        replay_pipeline.progress()
                        if trace_recorder:
                            trace_recorder.add_slice(
                                "learner/soft_update_target",
                                category="learner",
                                start_ns=_target_ns,
                                end_ns=time.perf_counter_ns(),
                                args={"update_idx": update_idx},
                            )

                    replay_pipeline.after_tick()
                    device = torch.device(self.device)
                    if device.type == "cuda":
                        torch.cuda.current_stream(device).synchronize()
                    elif device.type == "mps":
                        torch.mps.synchronize()
                    if trace_recorder:
                        trace_recorder.add_slice(
                            "learner/update_phase",
                            category="learner_update",
                            start_ns=train_phase_start_ns,
                            end_ns=time.perf_counter_ns(),
                            args={
                                "iteration": iteration,
                                "updates_per_step": self.updates_per_step,
                                "policy_version_before": inference_scheduler.policy_version,
                            },
                        )

                train_time = time.perf_counter() - train_start
                self.learner.update_count += 1
                inference_scheduler.finish_update()
                self._collect_dp_sync_metrics(iter_metrics)
                if trace_recorder:
                    trace_recorder.add_counter(
                        "replay_size",
                        int(replay_buffer.size[0]),
                        category="replay",
                    )

                iteration_time = time.perf_counter() - iteration_start

                write_delta = int(replay_buffer.ptr[0]) - ptr_before
                consume = self.batch_size * self.updates_per_step
                write_read_ema = 0.9 * write_read_ema + 0.1 * (write_delta / max(consume, 1))
                logger.update_buffer_utilization(write_read_ema)

                avg_metrics = {k: statistics.mean(v) for k, v in iter_metrics.items() if v}
                mean_reward = statistics.mean(reward_history) if reward_history else None

                self._sync_logger_replay_counters(logger, replay_buffer)
                log_payload = self._aggregate_log_statistics(
                    logger,
                    metrics=avg_metrics,
                    reward=mean_reward,
                    reward_metrics=build_reward_comparison_metrics(
                        reward_history,
                        mean_reward or 0.0,
                    ),
                    reward_components=latest_reward_components,
                    train_time=train_time,
                    collector_wait_time=collector_wait_time,
                    replay_batch_wait_time=replay_batch_wait_time,
                    learner_replay_sample_time=learner_replay_sample_time,
                    sync_coordination_time=sync_coordination_time,
                    replay_ingress_h2d_submit_time=replay_ingress_h2d_submit_time,
                    inference_h2d_time=inference_h2d_time,
                    inference_forward_time=inference_forward_time,
                    inference_d2h_time=inference_d2h_time,
                    inference_time=inference_time,
                    iteration_time=iteration_time,
                    extra_info={
                        "throughput_steps": self.num_envs * self.env_steps_per_sync,
                        "collector_active_steps_per_sec": (logger._collector_active_steps_per_sec),
                        **build_offpolicy_sample_info(
                            replay_batch_size_per_rank=self.batch_size,
                            updates_per_step=self.updates_per_step,
                            learner=self.learner,
                        ),
                    },
                )
                logged_reward = log_payload["reward"]
                if logged_reward is not None:
                    last_mean_reward = float(logged_reward)
                    best_mean_reward = max(best_mean_reward, last_mean_reward)
                    has_logged_reward = True
                logger.log_step(
                    iteration=iteration,
                    **log_payload,
                )

                if save_interval > 0 and iteration % save_interval == 0:
                    saved_path = self._save_checkpoint(
                        log_dir=log_dir,
                        iteration=iteration,
                        logger=logger,
                    )
                    if saved_path is not None:
                        ckpt_path = saved_path

            if trace_recorder:
                trace_recorder.add_slice(
                    "learner/training_e2e",
                    category="learner",
                    start_ns=training_e2e_start_ns,
                    end_ns=time.perf_counter_ns(),
                    args={
                        "iterations": iteration,
                        "pipeline": "gpu_resident",
                        "replay_h2d_submitter": self.replay_h2d_submitter,
                        "replay_transfer_backend": self.replay_transfer_backend,
                        "learner_log_interval": 1,
                    },
                )

            # -- finalize --
            replay_pipeline.close()
            final_ckpt_path = os.path.join(log_dir, f"model_{max_iterations}.pt")
            if ckpt_path != final_ckpt_path:
                saved_path = self._save_checkpoint(
                    log_dir=log_dir,
                    iteration=max_iterations,
                    logger=logger,
                )
                if saved_path is not None:
                    ckpt_path = saved_path
            if self.dp_sync is None:
                self._sync_logger_replay_counters(logger, replay_buffer)
            logger.finish()
            if trace_recorder and trace_output_path:
                trace_recorder.write_json(trace_output_path)
                print(f"[DoubleBufferRunner] Perfetto trace written to {trace_output_path}")
            self.last_run_summary = self._make_summary(
                "completed",
                iteration,
                logger,
                last_mean_reward if has_logged_reward else None,
                best_mean_reward if has_logged_reward else None,
                ckpt_path,
                train_start_wall,
                str(trace_output_path) if trace_output_path else None,
            )
            self._active_logger = None
        except _CollectorDiedError:
            self._fail_collector_died(
                logger,
                replay_buffer,
                replay_pipeline,
                iteration,
                ckpt_path,
                train_start_wall,
            )
            raise

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
            "runtime_manifest": dict(getattr(logger, "_runtime_manifest", {})),
            "final_env_steps_per_sec": logger._get_iter_steps_per_sec(),
            "final_learner_samples_per_sec": logger._get_effective_samples_per_sec(),
            "final_cycle_wall_ms": logger._get_iter_wall_time() * 1000.0,
            "final_inference_ms": getattr(logger, "_inference_time", 0.0) * 1000.0,
            "final_inference_h2d_ms": getattr(logger, "_inference_h2d_time", 0.0) * 1000.0,
            "final_inference_forward_ms": getattr(logger, "_inference_forward_time", 0.0) * 1000.0,
            "final_inference_d2h_ms": getattr(logger, "_inference_d2h_time", 0.0) * 1000.0,
        }
