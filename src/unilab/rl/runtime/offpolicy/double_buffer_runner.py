"""Off-policy runner using device-authoritative bounded-ingress replay."""

from __future__ import annotations

import os
import queue as queue_module
import statistics
import time
from collections import defaultdict, deque
from collections.abc import Callable, Mapping
from contextlib import nullcontext
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict, cast

import torch

if TYPE_CHECKING:
    from unilab.rl.runtime.distributed.sync import DpParameterSync

from unilab.rl.logging import OffPolicyLogger, TraceRecorder
from unilab.rl.logging.metric_schema import METRIC_SCHEMA_VERSION, metric_spec, normalize_metric_map
from unilab.rl.logging.metrics_drain import RewardComponentWindow
from unilab.rl.logging.runtime_manifest_schema import (
    RUNTIME_MANIFEST_SCHEMA_VERSION,
    validate_runtime_manifest,
)
from unilab.rl.runtime.async_runner import _SPAWN_CTX
from unilab.rl.runtime.device import (
    DeviceTimer,
    build_device_timer,
    device_runtime_capabilities,
    sync_device,
)
from unilab.rl.runtime.ipc.inference_ring import (
    SharedInferenceRing,
    estimate_inference_ring_bytes,
)
from unilab.rl.runtime.ipc.replay_buffer import ReplayBuffer
from unilab.rl.runtime.ipc.replay_pipelines.device import (
    DeviceReplayPipeline,
    require_offpolicy_replay_device,
)
from unilab.rl.runtime.offpolicy.actor_adapter import get_offpolicy_actor_adapter
from unilab.rl.runtime.offpolicy.coordination import LearnerCoordinationState
from unilab.rl.runtime.offpolicy.runner import (
    OffPolicyRunner,
    build_offpolicy_sample_info,
    replay_buffer_ready_for_learning,
)
from unilab.rl.runtime.offpolicy.shutdown_diagnostics import ShutdownDiagnosticsRecorder
from unilab.rl.runtime.offpolicy.thread_budget import (
    format_torch_thread_runtime,
    torch_thread_env,
)
from unilab.rl.runtime.offpolicy.warmup import (
    OffPolicyWarmupContext,
    capture_rng_state,
    restore_rng_state,
)
from unilab.rl.runtime.offpolicy.worker import (
    COLLECTOR_READY_TICK,
    offpolicy_collector_fn,
    sample_offpolicy_actions,
)
from unilab.rl.runtime.placement import (
    InferencePlacement,
    InferenceTransport,
    cpu_ring_staging_policy,
)
from unilab.rl.runtime.seed import derive_worker_seed
from unilab.rl.runtime.settings import (
    DEFAULT_COLLECTOR_METRICS_INTERVAL,
    DEFAULT_INFERENCE_SLOT_CAPACITY,
    DEFAULT_REPLAY_INGRESS_DEPTH,
    RuntimeSettings,
)
from unilab.rl.runtime.tf32 import format_tf32_matmul

# Terminal/W&B display names for the off-policy algo types. Keep these
# user-facing (no internal "Fast*" implementation prefixes).
_ALGO_DISPLAY_NAMES = {
    "sac": "SAC",
    "flashsac": "FlashSAC",
}


def validate_no_inference_request_timeout(training: Mapping[str, Any]) -> None:
    if "inference_request_timeout_sec" in training:
        raise ValueError(
            "training.inference_request_timeout_sec was removed; off-policy "
            "coordination is liveness/phase-aware. Remove it from owner YAML."
        )


_DP_METRIC_PREFIX = "metric::"
_DP_REWARD_COMPONENT_PREFIX = "reward_component::"
_DP_COLLECTOR_TIMING_PREFIX = "collector_timing::"


class _AggregatedLogStatistics(TypedDict):
    metrics: dict[str, float]
    checkpoint_return_mean_reports10: float | None
    return_mean_ep100: float | None
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
    mean_ep_length: float
    timeout_rate: float | None
    buffer_utilization: float
    collector_timing: dict[str, float]


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

    def __init__(
        self,
        *,
        replay_prefetch_mode: str = "one_tick",
        collector_cpu_ids: list[int] | None = None,
        dp_sync: DpParameterSync | None = None,
        learner_prepare_hook: Callable[[Any, OffPolicyWarmupContext], None] | None = None,
        backend_device_resolver: Callable[[str, str | None], str | None] | None = None,
        backend_device_configurer: Callable[[str, str | None], str | None] | None = None,
        replay_pipeline_factory: Callable[..., DeviceReplayPipeline] | None = None,
        target_frequency: int = 1,
        policy_before_critic: bool = False,
        inference_placement: InferencePlacement | None = None,
        runtime_settings: RuntimeSettings | None = None,
        inference_slot_capacity: int | None = None,
        inference_epoch: int = 0,
        collector_metrics_interval: int | None = None,
        tf32_matmul_policy: Any | None = None,
        **kwargs,
    ):
        kwargs["device"] = require_offpolicy_replay_device(kwargs.get("device"))
        # Backend-owned process-device resolution is injected by the owner
        # layer (e.g. ``unilab.base.process_device.resolve_backend_process_device``);
        # no resolver means this collector needs no backend device binding.
        collector_backend_device = (
            backend_device_resolver(
                str(kwargs.get("sim_backend", "mujoco")),
                kwargs["device"],
            )
            if backend_device_resolver is not None
            else None
        )
        if runtime_settings is None:
            effective_inference_capacity = (
                DEFAULT_INFERENCE_SLOT_CAPACITY
                if inference_slot_capacity is None
                else inference_slot_capacity
            )
            effective_metrics_interval = (
                DEFAULT_COLLECTOR_METRICS_INTERVAL
                if collector_metrics_interval is None
                else collector_metrics_interval
            )
            runtime_settings = RuntimeSettings(
                inference_slot_capacity=effective_inference_capacity,
                collector_metrics_interval=effective_metrics_interval,
                replay_ingress_depth=DEFAULT_REPLAY_INGRESS_DEPTH,
                replay_ingress_slot_rows=kwargs.get("num_envs", 4096),
                batch_size=kwargs.get("batch_size", 8192),
                updates_per_step=kwargs.get("updates_per_step", 8),
                num_envs=kwargs.get("num_envs", 4096),
            )
        else:
            explicit_values = {
                "inference_slot_capacity": (
                    inference_slot_capacity,
                    runtime_settings.inference_slot_capacity,
                ),
                "collector_metrics_interval": (
                    collector_metrics_interval,
                    runtime_settings.collector_metrics_interval,
                ),
                "batch_size": (
                    kwargs.get("batch_size"),
                    runtime_settings.batch_size,
                ),
                "updates_per_step": (
                    kwargs.get("updates_per_step"),
                    runtime_settings.updates_per_step,
                ),
                "num_envs": (kwargs.get("num_envs"), runtime_settings.num_envs),
            }
            for name, (value, _) in explicit_values.items():
                if value is not None and type(value) is not int:
                    raise TypeError(f"{name} must be a positive integer, got {value!r}")
            mismatches = {
                name: (value, expected)
                for name, (value, expected) in explicit_values.items()
                if value is not None and value != expected
            }
            if mismatches:
                details = ", ".join(
                    f"{name}={value!r} (settings={expected!r})"
                    for name, (value, expected) in mismatches.items()
                )
                raise ValueError(
                    f"Direct runner arguments conflict with runtime_settings: {details}"
                )
        super().__init__(**kwargs)
        if replay_prefetch_mode != "one_tick":
            raise ValueError(
                "DoubleBufferOffPolicyRunner only supports replay_prefetch_mode='one_tick'"
            )
        self.replay_prefetch_mode = replay_prefetch_mode
        # Kept only for source-compatible construction. It is
        # intentionally not retained as a latency SLA anywhere in the runtime.
        self.learner_prepare_hook = learner_prepare_hook
        self.target_frequency = max(int(target_frequency), 1)
        self.policy_before_critic = bool(policy_before_critic)
        # Per-rank CPU block owned by this rank's collector (multi-GPU DP);
        # merged into the collector-only env override at collector startup.
        self.collector_cpu_ids = list(collector_cpu_ids) if collector_cpu_ids is not None else None
        self.collector_backend_device = collector_backend_device
        if inference_placement is None:
            inference_placement = InferencePlacement(
                mode=InferenceTransport.HOST,
                env_device="cpu",
                ring_device="cpu",
                learner_device=str(kwargs["device"]),
                collector_tensor_native=False,
                staging_policy=cpu_ring_staging_policy(str(kwargs["device"])),
            )
        if not isinstance(inference_placement, InferencePlacement):
            raise TypeError(
                f"inference_placement must be an InferencePlacement, got {inference_placement!r}"
            )
        self._validate_inference_placement(
            inference_placement, learner_device=str(kwargs["device"])
        )
        self.inference_placement = inference_placement
        self.collector_tensor_native = inference_placement.collector_tensor_native
        self.runtime_settings = runtime_settings
        self.inference_slot_capacity = runtime_settings.inference_slot_capacity
        if isinstance(inference_epoch, bool) or not isinstance(inference_epoch, int):
            raise TypeError(f"inference_epoch must be an integer, got {inference_epoch!r}")
        if inference_epoch < 0:
            raise ValueError("inference_epoch must be non-negative")
        self.inference_epoch = int(inference_epoch)
        self.collector_metrics_interval = runtime_settings.collector_metrics_interval
        self.replay_ingress_depth = runtime_settings.replay_ingress_depth
        self.replay_ingress_slot_rows = runtime_settings.replay_ingress_slot_rows
        # Backend-owned process-device configurer forwarded to the collector
        # subprocess (e.g. mjwarp); None for backends that need no binding.
        self.backend_device_configurer = backend_device_configurer
        self.replay_pipeline_factory = replay_pipeline_factory
        self._collector_ready = False
        self._learner_coordination = LearnerCoordinationState()
        # Multi-GPU synchronous data parallelism (None = the bit-identical
        # single-rank path): startup model broadcast, then gradient averaging
        # before every actor/critic/temperature optimizer step.
        self.dp_sync = dp_sync
        self._local_logger_statistics: _LocalLoggerStatistics | None = None
        self._attach_dp_gradient_sync()
        self.replay_pack_executor = "collector_thread"
        self.replay_h2d_submitter = "auto"
        self.replay_transfer_backend: dict[str, object] = {}
        self.last_run_summary: dict[str, object] | None = None
        self._shutdown_recorder = ShutdownDiagnosticsRecorder(inference_epoch=self.inference_epoch)
        self._active_inference_ring: Any = None
        self._active_replay_buffer: ReplayBuffer | None = None
        self.tf32_matmul_policy = tf32_matmul_policy
        self.runtime_manifest = {
            "schema_version": RUNTIME_MANIFEST_SCHEMA_VERSION,
            "inference_owner": "learner",
            "collector_actor": False,
            "collector_accelerator_context": self.collector_backend_device is not None,
            "collector_backend_device": self.collector_backend_device,
            "collector_torch_inference": False,
            "collector_tensor_native": self.collector_tensor_native,
            "inference_transport": self.inference_placement.manifest(),
            "inference_ring_device": self.inference_placement.ring_device,
            "env_public_device": self.inference_placement.env_device,
            "learner_device": self.inference_placement.learner_device,
            "inference_staging_policy": self.inference_placement.staging_policy,
            "inference_ring_capacity": self.inference_slot_capacity,
            "runtime_limits": self.runtime_settings.manifest(),
            "inference_flight": {
                "queue_depth": 0,
                "publication_lag": 0,
                "max_in_flight": 0,
                "max_publication_lag": 0,
            },
            "inference_publication_ordering": "contiguous_ticks",
            "inference_epoch": self.inference_epoch,
            "learner_actor_reused": True,
            "logger_owner_rank": 0,
            "logger_cross_rank_aggregation": self.dp_sync is not None,
            "learner_performance_tier": self._learner_performance_tier_manifest(),
            "learner_amp_policy": getattr(self.learner, "amp_policy", None),
            "learner_tf32_matmul": (
                tf32_matmul_policy.manifest() if tf32_matmul_policy is not None else None
            ),
            "device_runtime_capabilities": device_runtime_capabilities(self.device).manifest(),
            "cuda_graph": self._cuda_graph_runtime_manifest(),
        }
        if self.dp_sync is not None:
            self.runtime_manifest["dp_sync"] = {
                "world_size": self.dp_sync.world_size,
                "backend": self.dp_sync.backend,
                "mode": "gradient_mean_per_optimizer_step",
            }

    def _validate_inference_placement(
        self, placement: InferencePlacement, *, learner_device: str
    ) -> None:
        """Fail closed before collector spawn on an incoherent topology."""
        expected_learner = str(learner_device)
        placement_learner = placement.learner_device
        expected_device = torch.device(expected_learner)
        placement_device = torch.device(placement_learner)
        same_device = placement_device == expected_device or (
            expected_device.type == "cuda"
            and expected_device.index is None
            and placement_device.type == "cuda"
            and placement_device.index == torch.cuda.current_device()
        )
        if not same_device:
            raise ValueError(
                "inference_placement learner device must match the runner device: "
                f"{placement_learner!r} != {expected_learner!r}"
            )
        if placement.mode is InferenceTransport.CUDA:
            ring = torch.device(placement.ring_device)
            env = torch.device(placement.env_device)
            learner = placement_device
            if learner.type != "cuda":
                raise ValueError(
                    "CUDA inference transport requires a CUDA runner device; "
                    f"got {expected_learner!r}"
                )
            if ring != learner or env != learner:
                raise ValueError(
                    "CUDA inference transport requires the env, inference ring, and "
                    f"learner to share one rank-local CUDA device; got env={placement.env_device!r}, "
                    f"ring={placement.ring_device!r}, learner={expected_learner!r}"
                )
            if not placement.collector_tensor_native:
                raise ValueError("CUDA inference transport requires tensor-native collection")
            return

        if torch.device(placement.ring_device).type != "cpu":
            raise ValueError(
                "Host inference transport requires a CPU inference ring; "
                f"got {placement.ring_device!r}"
            )
        if placement.collector_tensor_native:
            raise ValueError(
                "Host inference transport requires non-tensor-native collector transitions"
            )
        if torch.device(placement.env_device).type != "cpu":
            raise ValueError(
                "Host inference transport requires CPU env public tensors; "
                f"got {placement.env_device!r}"
            )

    def _attach_dp_gradient_sync(self) -> None:
        if self.dp_sync is None:
            return
        setter = getattr(self.learner, "set_gradient_sync", None)
        if not callable(setter):
            raise TypeError(
                f"{type(self.learner).__name__} must implement set_gradient_sync() "
                "for multi-GPU data parallelism"
            )
        setter(self.dp_sync.allreduce_gradients)
        graph_hooks = getattr(self.learner, "set_gradient_graph_hooks", None)
        if callable(graph_hooks):
            graph_hooks(
                begin_capture=partial(
                    self.dp_sync.begin_gradient_graph_capture,
                    enable_timing=self.trace_enabled and self.trace_cuda_events,
                ),
                end_capture=self.dp_sync.end_gradient_graph_capture,
                before_replay=self.dp_sync.record_gradient_graph_replay,
            )

    def _cuda_graph_runtime_manifest(self) -> dict[str, object]:
        """Describe the effective learner graph backend."""
        compile_enabled = bool(getattr(self.learner, "use_compile", False))
        backend = str(
            getattr(
                self.learner,
                "compile_backend",
                "inductor" if compile_enabled else "eager",
            )
        )
        manifest: dict[str, object] = {
            "backend": backend,
            "critic": compile_enabled,
            "actor": compile_enabled,
            "device_finite_optimizer_gating": not bool(
                getattr(self.learner, "_host_finite_checks", True)
            ),
        }
        if bool(getattr(self.learner, "use_update_cycle", False)):
            manifest["scope"] = "update_cycle"
            manifest["orchestration"] = "cuda_graph"
        elif compile_enabled:
            manifest["scope"] = "loss_tensors"
            manifest["orchestration"] = (
                "torch_compile" if backend == "torch_compile" else "inductor_cuda_graph_trees"
            )
        return manifest

    def _learner_performance_tier_manifest(self) -> dict[str, object] | None:
        tier = getattr(self.learner, "_performance_tier", None)
        return tier.manifest() if tier is not None else None

    def _dp_initial_sync(self) -> dict[str, torch.Tensor]:
        """Live model-state references broadcast once before collection."""
        initial_sync = getattr(self.learner, "dp_initial_sync", None)
        if not callable(initial_sync):
            raise TypeError(
                f"{type(self.learner).__name__} must implement dp_initial_sync() "
                "for multi-GPU data parallelism"
            )
        return cast(dict[str, torch.Tensor], initial_sync())

    def _dp_init_broadcast(self) -> None:
        """Align initial parameters from rank 0 before the collector starts.

        Ranks train with per-rank seeds, so without this broadcast each
        rank's actor would serve different inference from the first tick.
        """
        if self.dp_sync is None:
            return
        self.dp_sync.start()
        self.dp_sync.broadcast_from_rank0(self._dp_initial_sync())

    def _warm_representative_actor(self, context: OffPolicyWarmupContext) -> None:
        """Run one actor-shaped cold path and restore exploration RNG state."""
        adapter = get_offpolicy_actor_adapter(self.algo_type)
        actor_obs = context.inference_observations[:, : self.obs_dim]
        actor_context = None
        if adapter is not None and adapter.actor_context_from_obs is not None:
            actor_context = adapter.actor_context_from_obs(
                context.inference_observations,
                self.obs_dim,
            )
        if self.obs_normalization:
            actor_obs = self.learner.obs_normalizer(actor_obs, update=False)
        action_fn = (adapter.warmup_actions if adapter is not None else None) or (
            adapter.sample_actions if adapter is not None else None
        )
        if action_fn is None and not callable(getattr(self.learner.actor, "explore", None)):
            # Minimal custom learners may compose inference dynamically. Their
            # owner hook remains responsible for warmup; this is the documented
            # no-op compatibility default.
            return
        rng_state = capture_rng_state(self.device)
        try:
            with torch.no_grad():
                if action_fn is None:
                    actions = sample_offpolicy_actions(
                        actor=self.learner.actor,
                        algo_type=self.algo_type,
                        obs=actor_obs,
                        prev_dones=context.inference_dones,
                        priv_info=actor_context,
                    )
                else:
                    actions = action_fn(
                        self.learner.actor,
                        actor_obs,
                        context.inference_dones,
                        actor_context,
                    )
            if torch.device(self.device).type == "cuda":
                sync_device(self.device)
            if actions.device.type == "cuda":
                sync_device(actions.device)
        finally:
            restore_rng_state(rng_state)

    def _prepare_learner(
        self,
        *,
        inference_observations: torch.Tensor,
        inference_dones: torch.Tensor,
        replay_pipeline,
    ) -> None:
        """Prepare every learner-owned cold path before collection can start."""
        inference_observations.zero_()
        inference_dones.zero_()
        context = OffPolicyWarmupContext(
            inference_observations=inference_observations,
            inference_dones=inference_dones,
            batch_size=self.batch_size,
            updates_per_step=self.updates_per_step,
            policy_frequency=self.policy_frequency,
            target_frequency=self.target_frequency,
            policy_before_critic=self.policy_before_critic,
        )
        self._warm_representative_actor(context)
        pipeline_warmup = getattr(replay_pipeline, "warmup", None)
        replay_batch = None
        if callable(pipeline_warmup):
            warmup_result = pipeline_warmup()
            if isinstance(warmup_result, dict):
                replay_batch = cast(dict[str, torch.Tensor], warmup_result)
                context = replace(context, replay_batch=replay_batch)
        prepare = getattr(self.learner, "prepare_for_collection", None)
        if callable(prepare):
            prepare(context)
        if self.learner_prepare_hook is not None:
            self.learner_prepare_hook(self.learner, context)

    def _persistent_inference_scratch_bytes(self) -> int:
        """Read algorithm-owned, inference-only persistent scratch categories."""
        startup_hook = getattr(self.learner, "inference_startup_memory_categories", None)
        if startup_hook is None:
            # Unknown custom learners remain covered by the conservative
            # workspace reserve; no exact per-category claim is made for them.
            return 0
        if not callable(startup_hook):
            raise TypeError(
                f"{type(self.learner).__name__}.inference_startup_memory_categories "
                "must be callable"
            )
        categories = startup_hook(self.num_envs)
        if not isinstance(categories, dict) or set(categories) != {
            "persistent_exploration_scratch"
        }:
            raise TypeError(
                "inference_startup_memory_categories() must return exactly "
                "{'persistent_exploration_scratch': bytes}"
            )
        value = categories["persistent_exploration_scratch"]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("persistent exploration scratch bytes must be a non-negative int")
        return int(value)

    def _prepare_inference_timing_events(self) -> None:
        """Allocate the validated inference forward timer."""
        try:
            self._inference_forward_timer = build_device_timer(self.device)
        except BaseException:
            if hasattr(self, "_inference_forward_timer"):
                delattr(self, "_inference_forward_timer")
            raise

    def _shutdown_collector(self) -> None:
        """Release the lock-step collector without waiting on a tick deadline."""
        # Set the stop event before publishing STOPPED: a collector that
        # observes STOPPED must already see the stop request, so a normal
        # training completion is released gracefully instead of raising
        # "Learner stopped before inference tick".
        self._stop_event.set()
        self._learner_coordination.mark_stopped()
        process = self._collector_process
        if process is not None and process.is_alive():
            process.join(timeout=5.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5.0)

    def _drain_collector_metrics_after_shutdown(
        self,
        metrics_queue,
        reward_history: deque,
        latest_reward_components: RewardComponentWindow,
        logger,
    ) -> None:
        """Consume the collector's final metric publication after quiesce.

        ``_shutdown_collector`` joins the process before this call. The worker
        publishes its final partial collector-metric window from its ``finally``
        block, so a bounded queue still preserves every message. Draining here
        is therefore safe even when the process exits between the learner's
        normal polling points.
        """
        self._drain_metrics(
            metrics_queue,
            reward_history,
            latest_reward_components,
            logger,
        )

    def _collect_dp_sync_metrics(self, iter_metrics: defaultdict[str, list]) -> None:
        """Move per-optimizer collective timing into this iteration's metrics."""
        if self.dp_sync is None:
            return
        sync_time, sync_calls = self.dp_sync.take_gradient_sync_metrics()
        if sync_calls > 0:
            graph_mode = bool(getattr(self.learner, "use_update_cycle", False))
            if not graph_mode or (self.trace_enabled and self.trace_cuda_events):
                iter_metrics["Perf/dp_gradient_sync_ms_per_rank"].append(sync_time * 1000.0)
            iter_metrics["Perf/dp_gradient_sync_calls_per_rank"].append(float(sync_calls))

    @staticmethod
    def _replay_ingress_metrics(replay_pipeline) -> dict[str, float]:
        """Snapshot host-only bounded ingress gauges/counters for one log step."""
        diagnostics_method = getattr(replay_pipeline, "ingress_diagnostics", None)
        if not callable(diagnostics_method):
            return {}
        diagnostics = cast(Mapping[str, int | float], diagnostics_method())
        return {
            "Train/replay_ingress_depth": float(diagnostics["ingress_depth"]),
            "Train/replay_ingress_occupancy": float(diagnostics["occupancy"]),
            "Train/replay_ingress_high_water": float(diagnostics["high_water_occupancy"]),
            "Train/replay_ingress_backpressure_wait_ms": float(diagnostics["backpressure_wait_s"])
            * 1000.0,
            "Train/replay_ingress_dropped_batches": float(diagnostics["dropped_batches"]),
        }

    def _update_replay_ingress_manifest(
        self,
        logger: OffPolicyLogger,
        replay_pipeline,
    ) -> None:
        diagnostics_method = getattr(replay_pipeline, "ingress_diagnostics", None)
        if not callable(diagnostics_method):
            return
        diagnostics = diagnostics_method()
        self.runtime_manifest["replay_ingress"] = diagnostics
        logger.update_runtime_manifest({"replay_ingress": diagnostics})

    def _record_final_replay_ingress_diagnostics(
        self,
        logger: OffPolicyLogger,
        replay_buffer,
    ) -> None:
        diagnostics_method = getattr(replay_buffer, "ingress_diagnostics", None)
        if not callable(diagnostics_method):
            return
        diagnostics = diagnostics_method()
        self.runtime_manifest["replay_ingress"] = diagnostics
        logger_runtime_manifest = getattr(logger, "_runtime_manifest", None)
        if isinstance(logger_runtime_manifest, dict):
            logger_runtime_manifest["replay_ingress"] = diagnostics
        if isinstance(self.last_run_summary, dict):
            summary_manifest = self.last_run_summary.get("runtime_manifest")
            if isinstance(summary_manifest, dict):
                summary_manifest["replay_ingress"] = diagnostics

    def _minimal_failed_summary(self, status: str) -> dict[str, object]:
        """Build a schema-stamped summary when normal summary assembly fails."""

        return {
            "status": status,
            "metric_schema_version": METRIC_SCHEMA_VERSION,
            "runtime_manifest": dict(self.runtime_manifest),
        }

    def _record_shutdown_diagnostics(self, logger: OffPolicyLogger) -> dict[str, object]:
        try:
            diagnostics = self._shutdown_recorder.snapshot(
                learner_coordination=self._learner_coordination,
                inference_ring=self._active_inference_ring,
                replay_buffer=getattr(self, "_active_replay_buffer", None),
                collector_process=getattr(self, "_collector_process", None),
            )
        except BaseException as exc:
            self._shutdown_recorder.record_cleanup_error(exc)
            return {}
        self.runtime_manifest["shutdown"] = diagnostics
        logger_runtime_manifest = getattr(logger, "_runtime_manifest", None)
        if isinstance(logger_runtime_manifest, dict):
            logger_runtime_manifest["shutdown"] = diagnostics
        if isinstance(self.last_run_summary, dict):
            summary_manifest = self.last_run_summary.get("runtime_manifest")
            if isinstance(summary_manifest, dict):
                summary_manifest["shutdown"] = diagnostics
        return diagnostics

    def _aggregate_log_statistics(
        self,
        logger: OffPolicyLogger,
        *,
        metrics: dict[str, float],
        checkpoint_return_mean_reports10: float | None,
        return_mean_ep100: float | None,
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
    ) -> _AggregatedLogStatistics:
        """Return one log-step payload, reduced across ranks when DP is active.

        Model/reward/timing scalars are means. Concurrent work rates and
        capacity/sample counters are totals, so the rank-0 logger reports
        aggregate collector env steps/s and aggregate learner replay rows/s. Optional
        collector fields use the sparse presence-mask contract implemented by
        ``DpParameterSync.allreduce_statistics``.
        """
        metrics = normalize_metric_map(metrics)
        payload: _AggregatedLogStatistics = {
            "metrics": metrics,
            "checkpoint_return_mean_reports10": checkpoint_return_mean_reports10,
            "return_mean_ep100": return_mean_ep100,
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

        mean_metric_fields: dict[str, float] = {}
        total_metric_fields: dict[str, float] = {}
        for key, value in metrics.items():
            spec = metric_spec(key)
            if spec is not None and spec.distributed_aggregation == "cross-rank sum":
                total_metric_fields[f"{_DP_METRIC_PREFIX}{key}"] = float(value)
            else:
                mean_metric_fields[f"{_DP_METRIC_PREFIX}{key}"] = float(value)

        # ``logger`` also receives collector telemetry asynchronously. Keep a
        # rank-local snapshot before replacing its presentation state with the
        # aggregate; the next iteration restores this snapshot so a field that
        # is reported only every few collector cycles is never summed twice.
        self._local_logger_statistics = {
            "total_steps": logger._total_steps,
            "buffer_size": logger._buffer_size,
            "buffer_target": logger._buffer_target,
            "mean_ep_length": logger._mean_ep_length,
            "timeout_rate": logger._timeout_rate,
            "buffer_utilization": logger._buffer_utilization,
            "collector_timing": dict(logger._collector_timing),
        }

        mean: dict[str, float] = {
            **mean_metric_fields,
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
            "logger::buffer_utilization": float(logger._buffer_utilization),
            "extra::batch_size_per_rank": float(extra_info.get("batch_size_per_rank", 0) or 0),
        }
        if logger._timeout_rate is not None:
            mean["logger::timeout_rate"] = float(logger._timeout_rate)
        if checkpoint_return_mean_reports10 is not None:
            mean["checkpoint::return_reports10"] = float(checkpoint_return_mean_reports10)
        if return_mean_ep100 is not None:
            mean["return::ep100"] = float(return_mean_ep100)
        if logger._mean_ep_length > 0:
            mean["logger::mean_ep_length"] = float(logger._mean_ep_length)
        mean.update(
            {
                f"{_DP_COLLECTOR_TIMING_PREFIX}{key}": float(value)
                for key, value in logger._collector_timing.items()
            }
        )

        throughput_steps = int(extra_info.get("throughput_steps", 0) or 0)
        learner_replay_rows = int(extra_info.get("learner_replay_rows_per_iter", 0) or 0)
        total: dict[str, float] = {
            **total_metric_fields,
            "logger::total_steps": float(logger._total_steps),
            "logger::buffer_size": float(logger._buffer_size),
            "logger::buffer_target": float(logger._buffer_target),
            "extra::throughput_steps": float(throughput_steps),
            "extra::effective_batch_size": float(extra_info.get("effective_batch_size", 0) or 0),
            "extra::learner_replay_rows_per_iter": float(learner_replay_rows),
        }
        if iteration_time > 0:
            total["rate::env_steps_per_sec"] = throughput_steps / iteration_time
            total["rate::learner_replay_rows_per_sec"] = learner_replay_rows / iteration_time
        aggregated = self.dp_sync.allreduce_statistics(mean=mean, total=total)

        logger._total_steps = int(round(aggregated["logger::total_steps"]))
        logger._buffer_size = int(round(aggregated["logger::buffer_size"]))
        logger._buffer_target = int(round(aggregated["logger::buffer_target"]))
        logger._timeout_rate = aggregated.get("logger::timeout_rate")
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
            "env_steps_per_sec": aggregated.get("rate::env_steps_per_sec"),
            "learner_replay_rows_per_sec": aggregated.get("rate::learner_replay_rows_per_sec"),
            "batch_size_per_rank": int(round(aggregated["extra::batch_size_per_rank"])),
            "effective_batch_size": int(round(aggregated["extra::effective_batch_size"])),
            "learner_replay_rows_per_iter": int(
                round(aggregated["extra::learner_replay_rows_per_iter"])
            ),
        }
        return {
            "metrics": {
                key.removeprefix(_DP_METRIC_PREFIX): value
                for key, value in aggregated.items()
                if key.startswith(_DP_METRIC_PREFIX)
            },
            "checkpoint_return_mean_reports10": aggregated.get("checkpoint::return_reports10"),
            "return_mean_ep100": aggregated.get("return::ep100"),
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
        logger._mean_ep_length = float(state["mean_ep_length"])
        logger._timeout_rate = (
            float(state["timeout_rate"]) if state["timeout_rate"] is not None else None
        )
        logger._buffer_utilization = float(state["buffer_utilization"])
        logger._collector_timing = dict(state["collector_timing"])
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

    def _restore_resume_checkpoint(self, resume_checkpoint: str) -> int:
        """Restore learner state and return the first training iteration.

        Checkpoints are produced by :meth:`_save_checkpoint` and may contain
        optimizer state, so this path is intentionally not restricted to
        tensor-only loading. Malformed progress metadata fails closed rather
        than silently restarting training from old model weights.
        """
        state = torch.load(resume_checkpoint, map_location=self.device, weights_only=False)
        if not isinstance(state, dict):
            raise TypeError("resume checkpoint must contain a learner state dictionary")
        try:
            update_count = int(state["update_count"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("resume checkpoint is missing a valid integer update_count") from exc
        if update_count < 0:
            raise ValueError(
                f"resume checkpoint update_count must be non-negative, got {update_count}"
            )

        self.learner.load_state_dict(state)
        print(
            f"Resumed learner state from {resume_checkpoint} "
            f"(update_count={update_count}, next_iteration={update_count + 1})"
        )
        return update_count + 1

    def close(self) -> None:
        try:
            # Rank 0 owns the live terminal, and every rank owns a collector and
            # shared IPC resources. Release those before NCCL teardown, which
            # may wait on a peer during Ctrl+C shutdown.
            super().close()
        finally:
            if self.dp_sync is not None:
                # Captured NCCL nodes retain the communicator. Destroy the
                # learner graph before destroying its process group, otherwise
                # ncclCommDestroy can wait forever for the graph reference.
                try:
                    self.learner.set_gradient_sync(None)
                finally:
                    self.dp_sync.close()

    def _collector_env_cfg_override(self) -> dict | None:
        """Env override copy for the collector process, with per-rank CPU ids.

        MuJoCo sizes its BatchEnvPool worker count from ``len(cpu_ids)``, so
        the affinity list must only reach the collector's copy — never the
        learner-side probe envs, which keep the base override untouched.
        """
        override = dict(self.env_cfg_override or {})
        if self.collector_cpu_ids is not None:
            override["cpu_ids"] = list(self.collector_cpu_ids)
        removed_tensor_runtime_fields = {"tensor_runtime", "tensor_runtime_device"} & set(override)
        if removed_tensor_runtime_fields:
            raise ValueError(
                "Collector env override must not set removed tensor_runtime fields; "
                "inference transport is an explicit worker argument and collector "
                "tensor placement is owned by training.collector_tensor_device"
            )
        # Inference transport is process-local metadata, not an EnvCfg field.
        # The resolved placement reaches the worker only through the explicit
        # collector argument below; EnvCfg remains owned by UniLab.
        override.pop("inference_transport", None)
        return override or None

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
        self._learner_coordination.mark_waiting()
        while True:
            try:
                message = int(queue.get(timeout=0.1))
            except queue_module.Empty:
                self._learner_coordination.mark_progress()
                replay_pipeline.progress()
                self._drain_metrics(
                    metrics_queue,
                    reward_history,
                    latest_reward_components,
                    logger,
                    trace_recorder,
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
                continue
            self._learner_coordination.mark_busy()
            if message == COLLECTOR_READY_TICK:
                if self._collector_ready:
                    raise RuntimeError("Collector sent duplicate ready signal")
                self._collector_ready = True
                # The ready signal marks the end of cold-path startup on both
                # sides: learner compile/capture ran before the training loop
                # and collector env construction precedes this publish. Open
                # the live terminal display only now so startup output cannot
                # interleave with Live redraws.
                logger.start()
                continue
            if not self._collector_ready:
                raise RuntimeError(
                    f"Collector sent inference tick before its ready signal: got {message}"
                )
            if message != int(expected_tick):
                raise RuntimeError(
                    f"Collector inference tick mismatch: expected {expected_tick}, got {message}"
                )
            return message

    def _serve_learner_inference(
        self,
        inference_slot: SharedInferenceRing,
        *,
        tick_id: int,
        policy_version: int,
        obs_device: torch.Tensor,
        dones_device: torch.Tensor,
        actor_obs_device: torch.Tensor,
        actor_dones_device: torch.Tensor,
        actions_host: torch.Tensor | None,
        trace_recorder: TraceRecorder | None,
    ) -> dict[str, float]:
        h2d_start_ns = time.perf_counter_ns()
        inference_slot.copy_observation_to(
            tick_id=tick_id,
            observations=obs_device,
            dones=dones_device,
            non_blocking=False,
            epoch=self.inference_epoch,
        )
        h2d_end_ns = time.perf_counter_ns()

        # Host transport owns exactly one persistent ring->actor H2D boundary.
        if self.inference_placement.mode is InferenceTransport.HOST:
            actor_obs_device.copy_(obs_device, non_blocking=False)
            actor_dones_device.copy_(dones_device, non_blocking=False)
        else:
            if actor_obs_device is not obs_device or actor_dones_device is not dones_device:
                raise RuntimeError("CUDA inference transport must reuse ring-device scratch")
        actor_obs = actor_obs_device[:, : self.obs_dim]
        actor_adapter = get_offpolicy_actor_adapter(self.algo_type)
        actor_context = None
        if actor_adapter is not None and actor_adapter.actor_context_from_obs is not None:
            actor_context = actor_adapter.actor_context_from_obs(actor_obs_device, self.obs_dim)
        if self.obs_normalization:
            actor_obs = self.learner.obs_normalizer(actor_obs, update=False)
        forward_timer: DeviceTimer | None = getattr(self, "_inference_forward_timer", None)
        if forward_timer is None:
            raise RuntimeError("Inference forward timer was not prepared at startup")
        forward_timer.start()
        forward_start_ns = time.perf_counter_ns()
        with torch.no_grad():
            actions_device = sample_offpolicy_actions(
                actor=self.learner.actor,
                algo_type=self.algo_type,
                obs=actor_obs,
                prev_dones=actor_dones_device,
                priv_info=actor_context,
            )
        forward_end_ns = time.perf_counter_ns()
        forward_timer.stop()

        d2h_start_ns = time.perf_counter_ns()
        # Host transport owns exactly one actor->host action D2H boundary into a
        # persistent tensor; CUDA actions remain on the ring device.
        actions_for_ring = actions_device
        if self.inference_placement.mode is InferenceTransport.HOST:
            if actions_host is None:
                raise RuntimeError("Host inference transport lost its action staging tensor")
            actions_host.copy_(actions_device, non_blocking=False)
            actions_for_ring = actions_host
        inference_slot.publish_action(
            tick_id=tick_id,
            policy_version=policy_version,
            actions=actions_for_ring,
            non_blocking=False,
            epoch=self.inference_epoch,
        )
        d2h_end_ns = time.perf_counter_ns()
        # CUDA event elapsed-time reads synchronize after publication; MPS/CPU
        # wall timers synchronize only if the platform requires it.
        inference_forward_time = forward_timer.elapsed_s()
        timings = {
            "inference_h2d_time": (h2d_end_ns - h2d_start_ns) / 1e9,
            "inference_forward_time": inference_forward_time,
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
        failure = RuntimeError("Collector process died during off-policy training")
        self._shutdown_recorder.record_collector_failure(failure)
        try:
            logger.log_status("[red]ERROR: Collector died[/]")
            self._record_shutdown_diagnostics(logger)
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
        except BaseException as cleanup_exc:
            self._shutdown_recorder.record_cleanup_error(cleanup_exc)
            if not isinstance(self.last_run_summary, dict):
                self.last_run_summary = self._minimal_failed_summary("collector_died")
        try:
            replay_pipeline.close()
        except BaseException as cleanup_exc:
            self._shutdown_recorder.record_cleanup_error(cleanup_exc)
        raise failure

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
        del timeout
        while True:
            try:
                queue.put(int(value), timeout=0.5)
                return
            except queue_module.Full:
                if not self._check_collector_alive():
                    raise _CollectorDiedError(f"{label} (collector dead)")
                # A healthy collector owns the response slot. Its speed is not a
                # learner-side SLA; actual process death is checked each retry.

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
        resume_checkpoint: str | None = None,
    ) -> None:
        resume_start_iteration = 1
        if resume_checkpoint is not None:
            resume_start_iteration = self._restore_resume_checkpoint(resume_checkpoint)
        self.resume_start_iteration = resume_start_iteration
        if resume_start_iteration > max_iterations + 1:
            raise ValueError(
                "resume checkpoint is newer than max_iterations: "
                f"next_iteration={resume_start_iteration}, max_iterations={max_iterations}"
            )
        self._shutdown_recorder.reset(inference_epoch=self.inference_epoch)
        self.runtime_manifest.pop("shutdown", None)
        self.last_run_summary = None
        self._active_logger = None
        self._active_replay_buffer = None
        self._active_inference_ring = None
        try:
            self._learn_impl(
                max_iterations=max_iterations,
                save_interval=save_interval,
                log_dir=log_dir,
                logger_type=logger_type,
            )
        except BaseException as exc:
            if "shutdown" not in self.runtime_manifest:
                self._shutdown_recorder.record_failure(exc)
                self._record_startup_shutdown_diagnostics()
            if not isinstance(self.last_run_summary, dict):
                self.last_run_summary = self._minimal_failed_summary("failed")
            raise

    def _record_startup_shutdown_diagnostics(self) -> None:
        try:
            self.runtime_manifest["shutdown"] = self._shutdown_recorder.snapshot(
                learner_coordination=self._learner_coordination,
                inference_ring=self._active_inference_ring,
                replay_buffer=getattr(self, "_active_replay_buffer", None),
                collector_process=getattr(self, "_collector_process", None),
            )
        except BaseException as diagnostics_exc:
            self._shutdown_recorder.record_cleanup_error(diagnostics_exc)

    def _synchronize_after_update_phase(
        self, device: torch.device, *, phase_ordered_metrics: bool
    ) -> None:
        """Apply the validated platform policy after a learner update phase."""
        if device.type == "cuda":
            sync_device(device, stream=True)
        elif device.type == "mps" and not phase_ordered_metrics:
            # torch.mps preserves same-queue ordering, and phase-ordered metric
            # reads already synchronize the host. A global sync would only
            # drain unrelated queued work and recreate a GPU idle gap.
            sync_device(device)

    def _learn_impl(
        self,
        max_iterations: int = 1500,
        save_interval: int = 50,
        log_dir: str = "logs",
        logger_type: str = "tensorboard",
    ) -> None:
        self._collector_ready = False
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
        start_iteration = self.resume_start_iteration

        self._shutdown_recorder.set_phase(owner="learner", phase="startup/memory_budget")
        # --- memory budget check ---
        from unilab.rl.runtime.ipc.memory_budget import (
            estimate_cuda_inference_ipc_bytes,
            estimate_cuda_tensor_runtime_bytes,
            estimate_host_offpolicy_bytes,
            raise_if_cuda_memory_over_budget,
            raise_if_shared_memory_over_budget,
            warn_if_over_budget,
        )

        gpu_centric_collector = self.collector_tensor_native
        inference_ring_device = self.inference_placement.ring_device
        actor_context_dim = int(getattr(self.learner, "priv_info_dim", 0))
        inference_input_dim = self.obs_dim + actor_context_dim
        inference_ring_bytes = estimate_inference_ring_bytes(
            self.num_envs,
            inference_input_dim,
            self.action_dim,
            capacity=self.inference_slot_capacity,
        )
        mem_est = estimate_host_offpolicy_bytes(
            num_envs=self.num_envs,
            replay_buffer_n=self.replay_buffer_n,
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            critic_dim=self.critic_obs_dim,
            ingress_depth=self.replay_ingress_depth,
            ingress_slot_rows=self.replay_ingress_slot_rows,
            inference_ring_bytes=0 if gpu_centric_collector else inference_ring_bytes,
            ingress_on_device=gpu_centric_collector,
        )
        warn_if_over_budget(mem_est, label=f"Off-policy ({self.algo_type})")
        raise_if_shared_memory_over_budget(mem_est, label=f"Off-policy ({self.algo_type})")

        self._shutdown_recorder.set_phase(owner="learner", phase="startup/cuda_preflight")
        # This guard runs before replay ingress, the inference ring, IPC events,
        # or learner inference scratch is materialized. It intentionally uses a
        # conservative reserve rather than claiming exact allocator accounting.
        learner_device = torch.device(self.device)
        if learner_device.type == "cuda":
            cuda_free_bytes, _cuda_total_bytes = torch.cuda.mem_get_info(learner_device)
            cuda_inference_budget = estimate_cuda_inference_ipc_bytes(
                self.num_envs,
                inference_input_dim,
                self.action_dim,
                capacity=self.inference_slot_capacity,
                ring_on_device=gpu_centric_collector,
                persistent_exploration_scratch=self._persistent_inference_scratch_bytes(),
            )
            cuda_tensor_budget = estimate_cuda_tensor_runtime_bytes(
                num_envs=self.num_envs,
                replay_buffer_n=self.replay_buffer_n,
                obs_dim=self.obs_dim,
                action_dim=self.action_dim,
                critic_dim=self.critic_obs_dim,
                inference_obs_dim=inference_input_dim,
                sample_count=self.runtime_settings.learner_sample_count,
                inference_slot_capacity=self.inference_slot_capacity,
                replay_ingress_depth=self.replay_ingress_depth,
                replay_ingress_slot_rows=self.replay_ingress_slot_rows,
                collector_tensor_native=gpu_centric_collector,
                persistent_exploration_scratch=self._persistent_inference_scratch_bytes(),
            )
            raise_if_cuda_memory_over_budget(
                cuda_tensor_budget,
                label=f"Off-policy ({self.algo_type})",
                available_bytes=cuda_free_bytes,
                user_knob=(
                    "training.inference_slot_capacity, training.replay_ingress_depth, "
                    "training.replay_ingress_slot_rows, algo.batch_size, "
                    "algo.updates_per_step, algo.replay_buffer_n, or algo.num_envs"
                ),
                budget_kind="CUDA tensor runtime (inference + replay)",
            )
            self.runtime_manifest["inference_memory_budget"] = {
                **cuda_inference_budget,
                "available_bytes": cuda_free_bytes,
                "threshold": 0.8,
                "allowed_bytes": int(cuda_free_bytes * 0.8),
            }
            self.runtime_manifest["tensor_memory_budget"] = {
                **cuda_tensor_budget,
                "available_bytes": cuda_free_bytes,
                "threshold": 0.8,
                "allowed_bytes": int(cuda_free_bytes * 0.8),
            }
            self._prepare_inference_timing_events()
            self.runtime_manifest["inference_timing"] = {
                "timer_kind": "cuda_events",
                "has_timing_events": True,
                "persistent_timing_event_count": 2,
                "classification": "capability",
            }

        self._shutdown_recorder.set_phase(owner="learner", phase="startup/replay_resources")
        # --- bounded collector ingress (the complete ring lives on device) ---
        buffer_capacity = self.replay_buffer_n * self.num_envs
        replay_buffer = ReplayBuffer(
            capacity=buffer_capacity,
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            device=self.device,
            critic_dim=self.critic_obs_dim,
            ingress_slot_rows=self.replay_ingress_slot_rows,
            ingress_depth=self.replay_ingress_depth,
            ingress_device=self.device if gpu_centric_collector else "cpu",
            # Host transport still owns a CUDA replay device and CPU ingress.
            require_completion_receipt=self.produces_completion_receipts,
        )
        self._active_replay_buffer = replay_buffer
        self._shared_resources.append(replay_buffer)
        replay_buffer.trace_recorder = trace_recorder
        replay_buffer.trace_thread_time = self.trace_thread_time
        replay_buffer.trace_cuda_events = self.trace_cuda_events

        # --- authoritative device ring and hot/cold learner batches ---
        sample_count = self.runtime_settings.learner_sample_count
        replay_pipeline_factory = self.replay_pipeline_factory or DeviceReplayPipeline
        replay_pipeline = replay_pipeline_factory(
            replay_buffer,
            device=self.device,
            sample_count=sample_count,
            base_seed=int(self.seed or 0),
            trace_recorder=trace_recorder,
            trace_cuda_events=self.trace_cuda_events,
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
        ingress_diagnostics_method = getattr(
            replay_pipeline,
            "ingress_diagnostics",
            None,
        )
        self.runtime_manifest.update(
            {
                "replay_h2d_submitter": self.replay_h2d_submitter,
                "replay_device_submission_thread": self.replay_transfer_backend.get(
                    "device_submission_thread"
                ),
                "replay_ingress": (
                    ingress_diagnostics_method() if callable(ingress_diagnostics_method) else {}
                ),
            }
        )

        self._shutdown_recorder.set_phase(owner="learner", phase="startup/inference_resources")
        inference_slot = SharedInferenceRing(
            self.num_envs,
            inference_input_dim,
            self.action_dim,
            device=inference_ring_device,
            capacity=self.inference_slot_capacity,
            epoch=self.inference_epoch,
        )
        self._active_inference_ring = inference_slot
        self._shared_resources.append(inference_slot)
        inference_obs_device = torch.empty(
            (self.num_envs, inference_input_dim),
            dtype=torch.float32,
            device=inference_ring_device,
        )
        inference_dones_device = torch.empty(
            self.num_envs,
            dtype=torch.float32,
            device=inference_ring_device,
        )
        # Host transport owns persistent learner-side actor staging. Copies into
        # these tensors are its sole ring->actor H2D boundary; CUDA transport
        # deliberately reuses the ring-device scratch with no host detour.
        if self.inference_placement.mode is InferenceTransport.HOST:
            inference_obs_actor = torch.empty(
                inference_obs_device.shape,
                dtype=inference_obs_device.dtype,
                device=self.device,
            )
            inference_dones_actor = torch.empty(
                inference_dones_device.shape,
                dtype=inference_dones_device.dtype,
                device=self.device,
            )
            inference_actions_host = torch.empty(
                (self.num_envs, self.action_dim), dtype=torch.float32, device="cpu"
            )
            self.runtime_manifest.update(
                {
                    "inference_learner_scratch_device": self.inference_placement.learner_device,
                    "inference_action_host_staging": True,
                }
            )
        else:
            inference_obs_actor = inference_obs_device
            inference_dones_actor = inference_dones_device
            inference_actions_host = None
            self.runtime_manifest.update(
                {
                    "inference_learner_scratch_device": self.inference_placement.learner_device,
                    "inference_action_host_staging": False,
                }
            )
        self.runtime_manifest.update(
            {
                "inference_slot_bytes": inference_slot.nbytes,
                "inference_publication_sync": self.inference_placement.transport_spec.publication_sync,
                "collector_metrics_interval": self.collector_metrics_interval,
            }
        )

        self._shutdown_recorder.set_phase(owner="learner", phase="startup/logger")
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
            log_interval=self.log_interval,
        )
        logger.update_runtime_manifest(self.runtime_manifest)
        logger.log_status(format_torch_thread_runtime(self.torch_thread_runtime))
        logger.log_status(format_tf32_matmul(self.tf32_matmul_policy))
        logger.log_status("Replay storage: device-authoritative bounded ingress")
        logger.log_status(f"Replay prefetch mode: {self.replay_prefetch_mode}")
        logger.log_status(f"Replay pack executor: {self.replay_pack_executor}")
        logger.log_status(f"Replay H2D submitter: {self.replay_h2d_submitter}")
        graph_manifest = cast(dict[str, object], self.runtime_manifest["cuda_graph"])
        logger.log_status(
            "CUDA Graph: "
            f"backend={graph_manifest['backend']}, "
            f"critic={graph_manifest['critic']}, "
            f"actor={graph_manifest['actor']}, "
            f"device_finite_gate={graph_manifest['device_finite_optimizer_gating']}"
        )
        if self.replay_transfer_backend:
            logger.log_status(
                "Replay transfer backend: "
                f"{self.replay_transfer_backend.get('backend')} "
                f"({self.replay_transfer_backend.get('device_family')})"
            )
        logger.log_status(f"Inference owner: learner.actor ({self.device})")
        if self.collector_backend_device is not None:
            logger.log_status(f"Collector backend device: {self.collector_backend_device}")
        logger.log_status("Collector actor/inference ownership: none")
        logger.log_status(
            f"Replay learner lightweight: batched event write (log_interval={self.log_interval})"
        )
        self._active_logger = logger
        try:
            self._shutdown_recorder.set_phase(owner="learner", phase="startup/dp_init")
            # --- inference coordination queues ---
            inference_request_queue = _SPAWN_CTX.Queue(maxsize=1)
            inference_response_queue = _SPAWN_CTX.Queue(maxsize=1)

            metrics_queue = _SPAWN_CTX.Queue(maxsize=100)

            # --- DP init broadcast must land before the collector's first ---
            # --- inference request reaches learner.actor ---
            self._dp_init_broadcast()
            self._shutdown_recorder.set_phase(owner="learner", phase="startup/learner_prepare")

            # Algorithm-owned compilation, graph capture, actor warmup, and
            # custom preparation hooks complete before a collector can request
            # tick 0. DP initialization remains first so warmup sees broadcast
            # parameters and can capture rank-aligned graphs.
            self._prepare_learner(
                inference_observations=inference_obs_actor,
                inference_dones=inference_dones_actor,
                replay_pipeline=replay_pipeline,
            )
            graph_manifest = self._cuda_graph_runtime_manifest()
            self.runtime_manifest["cuda_graph"] = graph_manifest
            logger.update_runtime_manifest(dict(self.runtime_manifest))
            self._shutdown_recorder.set_phase(owner="learner", phase="startup/collector_start")

            # --- start collector ---
            collector_kwargs = {
                "env_factory": self.env_factory,
                "num_envs": self.num_envs,
                "replay_buffer": replay_buffer,
                "algo_type": self.algo_type,
                "actor_adapter_modules": list(self.actor_adapter_modules),
                "metrics_queue": metrics_queue,
                "inference_request_queue": inference_request_queue,
                "inference_response_queue": inference_response_queue,
                "sim_backend": self.sim_backend,
                "backend_device": self.collector_backend_device,
                "env_cfg_override": self._collector_env_cfg_override(),
                "inference_transport": self.inference_placement.mode.value,
                "inference_slot": inference_slot,
                "inference_epoch": self.inference_epoch,
                "collector_metrics_interval": self.collector_metrics_interval,
                "seed": derive_worker_seed(self.seed, worker_index=0),
                "trace_enabled": self.trace_enabled,
                "trace_thread_time": self.trace_thread_time,
                "nan_guard_cfg": self.nan_guard_cfg,
                "nan_guard_factory": self.nan_guard_factory,
                "torch_thread_runtime": self.torch_thread_runtime,
                "backend_device_configurer": self.backend_device_configurer,
                "learner_coordination": self._learner_coordination,
                "learner_pid": os.getpid(),
            }
            # The collector may finish env construction quickly while the
            # learner is still in its startup sleep. It must observe a healthy
            # waiting learner, never the pre-start STOPPED phase.
            self._learner_coordination.mark_waiting()
            with torch_thread_env(self.torch_thread_runtime, role="collector"):
                self._start_collector(
                    target_fn=offpolicy_collector_fn,
                    kwargs={"stop_event": self._stop_event, **collector_kwargs},
                )
            self._shutdown_recorder.set_phase(owner="learner", phase="training/startup_complete")

            time.sleep(0.5)

            # Recent collector reports; each entry is already the collector's
            # rolling 100-episode mean, so a short window keeps the logged
            # reward timely without losing smoothing.
            reward_history: deque = deque(maxlen=10)
            latest_reward_components = RewardComponentWindow()
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
            for iteration in range(start_iteration, max_iterations + 1):
                self._shutdown_recorder.set_phase(
                    owner="learner",
                    phase="training/wait_for_inference_request",
                    iteration=iteration,
                )
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
                    self._shutdown_recorder.set_phase(
                        owner="learner",
                        phase="training/learner_inference",
                        iteration=iteration,
                        coordination_tick=request_tick,
                    )
                    inference_timings = self._serve_learner_inference(
                        inference_slot,
                        tick_id=request_tick,
                        policy_version=inference_scheduler.policy_version,
                        obs_device=inference_obs_device,
                        dones_device=inference_dones_device,
                        actor_obs_device=inference_obs_actor,
                        actor_dones_device=inference_dones_actor,
                        actions_host=inference_actions_host,
                        trace_recorder=trace_recorder,
                    )
                    self._shutdown_recorder.set_phase(
                        owner="learner",
                        phase="training/inference_response",
                        iteration=iteration,
                        coordination_tick=request_tick,
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

                    self._shutdown_recorder.set_phase(
                        owner="learner",
                        phase="training/replay_prepare",
                        iteration=iteration,
                        coordination_tick=request_tick,
                    )

                    self._drain_metrics(
                        metrics_queue,
                        reward_history,
                        latest_reward_components,
                        logger,
                        trace_recorder,
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
                    self._shutdown_recorder.set_phase(
                        owner="learner",
                        phase="training/learner_update",
                        iteration=iteration,
                        coordination_tick=request_tick,
                    )
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
                                "pipeline": "device",
                                "batch_ready": batch_ready,
                                "prefetch_mode": self.replay_prefetch_mode,
                                "replay_pack_executor": self.replay_pack_executor,
                                "replay_h2d_submitter": self.replay_h2d_submitter,
                                "replay_transfer_backend": self.replay_transfer_backend,
                                "prepared_tick": prepared_tick,
                                "explicit_compute_stream": False,
                            },
                        )

                    train_start = time.perf_counter()
                    train_phase_start_ns = time.perf_counter_ns()
                    phase_ordered_metrics = any(
                        callable(
                            getattr(
                                learner,
                                metric_reader,
                                None,
                            )
                        )
                        for metric_reader in (
                            "read_deferred_update_metrics",
                            "read_deferred_cycle_metrics",
                        )
                    )

                    update_cycle = cast(
                        Callable[..., object] | None,
                        getattr(learner, "update_cycle", None),
                    )
                    if update_cycle is not None and bool(
                        getattr(learner, "use_update_cycle", False)
                    ):
                        update_cycle(
                            large_batch,
                            updates_per_step=self.updates_per_step,
                            policy_frequency=self.policy_frequency,
                            target_frequency=self.target_frequency,
                            policy_before_critic=self.policy_before_critic,
                            read_metrics=False,
                        )
                        # Replay ingress used to be polled after each update.  It is
                        # kept at the same count after the submitted whole-cycle
                        # graph so polling remains outside the compiled region.
                        for _ in range(self.updates_per_step):
                            replay_pipeline.progress()
                        deferred_cycle_metrics = getattr(
                            learner,
                            "read_deferred_cycle_metrics",
                            None,
                        )
                        if callable(deferred_cycle_metrics):
                            for key, value in cast(
                                dict[str, float],
                                deferred_cycle_metrics(),
                            ).items():
                                iter_metrics[key].append(value)
                    else:

                        def run_actor_update(
                            batch: dict[str, torch.Tensor],
                            update_idx: int,
                        ) -> None:
                            next_actor_update = update_idx + self.policy_frequency
                            defer_actor_metrics = bool(
                                getattr(learner, "supports_deferred_update_metrics", False)
                            )
                            read_deferred_actor_metrics = (
                                next_actor_update >= self.updates_per_step
                                and not defer_actor_metrics
                            )
                            _actor_ns = time.perf_counter_ns()
                            if getattr(learner, "supports_deferred_update_metrics", False):
                                actor_metrics = learner.update_actor(
                                    batch,
                                    read_metrics=read_deferred_actor_metrics,
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

                        deferred_update_metrics = getattr(
                            learner,
                            "read_deferred_update_metrics",
                            None,
                        )
                        begin_deferred_update_metrics = getattr(
                            learner,
                            "begin_deferred_update_metrics",
                            None,
                        )
                        if callable(begin_deferred_update_metrics):
                            begin_deferred_update_metrics()
                        for update_idx in range(self.updates_per_step):
                            s = update_idx * self.batch_size
                            e = s + self.batch_size
                            batch = {k: v[s:e] for k, v in large_batch.items()}
                            read_deferred_critic_metrics = (
                                update_idx == self.updates_per_step - 1
                                and not callable(deferred_update_metrics)
                            )
                            do_actor_update = update_idx % self.policy_frequency == 0

                            if self.policy_before_critic and do_actor_update:
                                run_actor_update(batch, update_idx)

                            _critic_ns = time.perf_counter_ns()
                            if getattr(learner, "supports_deferred_update_metrics", False):
                                critic_metrics = learner.update_critic(
                                    batch,
                                    read_metrics=read_deferred_critic_metrics,
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

                            if not self.policy_before_critic and do_actor_update:
                                run_actor_update(batch, update_idx)

                            _target_ns = time.perf_counter_ns()
                            if update_idx % self.target_frequency == 0:
                                learner.soft_update_target()
                            replay_pipeline.progress()
                            if trace_recorder:
                                trace_recorder.add_slice(
                                    "learner/soft_update_target",
                                    category="learner",
                                    start_ns=_target_ns,
                                    end_ns=time.perf_counter_ns(),
                                    args={
                                        "update_idx": update_idx,
                                    },
                                )

                        if callable(deferred_update_metrics):
                            for key, value in cast(
                                dict[str, float],
                                deferred_update_metrics(),
                            ).items():
                                iter_metrics[key].append(value)
                        else:
                            deferred_actor_metrics = getattr(
                                learner,
                                "read_deferred_actor_metrics",
                                None,
                            )
                            if callable(deferred_actor_metrics):
                                for key, value in cast(
                                    dict[str, float],
                                    deferred_actor_metrics(),
                                ).items():
                                    iter_metrics[key].append(value)

                    replay_pipeline.after_tick()
                    self._synchronize_after_update_phase(
                        torch.device(self.device),
                        phase_ordered_metrics=phase_ordered_metrics,
                    )
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
                avg_metrics.update(self._replay_ingress_metrics(replay_pipeline))
                self._update_replay_ingress_manifest(logger, replay_pipeline)
                mean_return_reports10 = statistics.mean(reward_history) if reward_history else None

                self._sync_logger_replay_counters(logger, replay_buffer)
                log_payload = self._aggregate_log_statistics(
                    logger,
                    metrics=avg_metrics,
                    checkpoint_return_mean_reports10=mean_return_reports10,
                    return_mean_ep100=(float(reward_history[-1]) if reward_history else None),
                    reward_components=latest_reward_components.take(),
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
                        **build_offpolicy_sample_info(
                            replay_batch_size_per_rank=self.batch_size,
                            updates_per_step=self.updates_per_step,
                        ),
                    },
                )
                logged_return_reports10 = log_payload["checkpoint_return_mean_reports10"]
                if logged_return_reports10 is not None:
                    last_mean_reward = float(logged_return_reports10)
                    best_mean_reward = max(best_mean_reward, last_mean_reward)
                    has_logged_reward = True
                logger.log_step(
                    iteration=iteration,
                    metrics=log_payload["metrics"],
                    return_mean_ep100=log_payload["return_mean_ep100"],
                    reward_components=log_payload["reward_components"],
                    train_time=log_payload["train_time"],
                    collector_wait_time=log_payload["collector_wait_time"],
                    replay_batch_wait_time=log_payload["replay_batch_wait_time"],
                    learner_replay_sample_time=log_payload["learner_replay_sample_time"],
                    sync_coordination_time=log_payload["sync_coordination_time"],
                    replay_ingress_h2d_submit_time=log_payload["replay_ingress_h2d_submit_time"],
                    inference_h2d_time=log_payload["inference_h2d_time"],
                    inference_forward_time=log_payload["inference_forward_time"],
                    inference_d2h_time=log_payload["inference_d2h_time"],
                    inference_time=log_payload["inference_time"],
                    iteration_time=log_payload["iteration_time"],
                    extra_info=log_payload["extra_info"],
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
                        "pipeline": "device",
                        "replay_h2d_submitter": self.replay_h2d_submitter,
                        "replay_transfer_backend": self.replay_transfer_backend,
                        "learner_log_interval": self.log_interval,
                    },
                )

            # -- finalize --
            self._shutdown_recorder.set_phase(
                owner="learner", phase="finalize/collector_quiesce", iteration=iteration
            )
            # Stop the collector before closing the replay pipeline. The
            # collector may already be inside its final vectorized transition
            # when the learner reaches max_iterations. Ingress publication is
            # chunk-granular: shutdown may retain a valid prefix of that vector,
            # but quiescing first prevents an unpublished suffix from racing
            # pipeline.close(), which drains and releases every published chunk.
            self._shutdown_collector()
            self._shutdown_recorder.set_phase(
                owner="learner",
                phase="finalize/collector_metrics_drain",
                iteration=iteration,
            )
            self._drain_collector_metrics_after_shutdown(
                metrics_queue,
                reward_history,
                latest_reward_components,
                logger,
            )
            # The final partial episode window is not represented by the last
            # periodic scalar. Recompute both summary values from the complete
            # collector history after shutdown so best/final can differ when an
            # earlier episode window was stronger.
            if reward_history:
                last_mean_reward = float(reward_history[-1])
                best_mean_reward = max(best_mean_reward, last_mean_reward)
                has_logged_reward = True
            self._shutdown_recorder.set_phase(
                owner="learner", phase="finalize/replay_pipeline_close", iteration=iteration
            )
            replay_pipeline.close()
            self._shutdown_recorder.set_phase(
                owner="learner", phase="finalize/replay_ingress_snapshot", iteration=iteration
            )
            try:
                self._record_final_replay_ingress_diagnostics(logger, replay_buffer)
            except BaseException as diagnostics_exc:
                self._shutdown_recorder.record_cleanup_error(diagnostics_exc)
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
            self._shutdown_recorder.set_phase(
                owner="learner", phase="finalize/logger_finish", iteration=iteration
            )
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
            self._shutdown_recorder.record_normal_completion()
            self._record_shutdown_diagnostics(logger)
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
        except BaseException as exc:
            self._shutdown_recorder.record_failure(exc)
            self._record_shutdown_diagnostics(logger)
            try:
                self.last_run_summary = self._make_summary(
                    "failed",
                    iteration,
                    logger,
                    None,
                    None,
                    ckpt_path,
                    train_start_wall,
                    None,
                )
            except BaseException as summary_exc:
                self._shutdown_recorder.record_cleanup_error(summary_exc)
                self.last_run_summary = self._minimal_failed_summary("failed")
            raise
        finally:
            # Learner stop/death must release the collector even when the active
            # exception is unrelated to collector liveness.
            try:
                self._shutdown_collector()
            except BaseException as cleanup_exc:
                self._shutdown_recorder.record_cleanup_error(cleanup_exc)
            try:
                self._record_final_replay_ingress_diagnostics(logger, replay_buffer)
            except BaseException as cleanup_exc:
                self._shutdown_recorder.record_cleanup_error(cleanup_exc)
            self._record_shutdown_diagnostics(logger)

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
        summary = {
            "status": status,
            "completed_iterations": iteration,
            "total_env_steps": int(logger._total_steps),
            "final_mean_reward": final_reward,
            "best_mean_reward": best_reward,
            "mean_episode_length": float(logger._mean_ep_length),
            "last_checkpoint": ckpt_path,
            "trace_path": trace_path,
            "training_wall_time_sec": time.time() - train_start_wall,
            "metric_schema_version": METRIC_SCHEMA_VERSION,
            "runtime_manifest": dict(getattr(logger, "_runtime_manifest", {})),
            "final_env_steps_per_sec": logger._get_iter_env_steps_per_sec(),
            "tail_env_steps_per_sec": logger._get_tail_env_steps_per_sec(),
            "tail_iteration_count": len(getattr(logger, "_tail_iteration_times", ())),
            "tail_window_iterations": getattr(logger, "tail_window_iterations", 0),
            "final_learner_replay_rows_per_sec": (logger._get_learner_replay_rows_per_sec()),
            "final_cycle_wall_ms": logger._get_iter_wall_time() * 1000.0,
            "final_inference_ms": getattr(logger, "_inference_time", 0.0) * 1000.0,
            "final_inference_h2d_ms": getattr(logger, "_inference_h2d_time", 0.0) * 1000.0,
            "final_inference_forward_ms": getattr(logger, "_inference_forward_time", 0.0) * 1000.0,
            "final_inference_d2h_ms": getattr(logger, "_inference_d2h_time", 0.0) * 1000.0,
        }
        validate_runtime_manifest(
            summary["runtime_manifest"],
            completed=status == "completed",
        )
        return summary
