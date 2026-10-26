"""Environment and replay collector for learner-owned off-policy inference."""

import queue
import sys
import time
from collections.abc import Callable
from typing import Any, cast

import torch

from unilab.rl.contracts.env import (
    EnvFactory,
    get_algo_capabilities,
    get_backend_runtime_diagnostics,
)
from unilab.rl.contracts.observations import split_obs_dict
from unilab.rl.runtime.collector_timing import extract_env_step_breakdown_timing_ms
from unilab.rl.runtime.offpolicy.actor_adapter import (
    get_offpolicy_actor_adapter,
    import_actor_adapter_modules,
)
from unilab.rl.runtime.offpolicy.coordination import (
    LearnerCoordinationState,
    LearnerPhase,
    learner_pid_is_alive,
)
from unilab.rl.runtime.offpolicy.metrics import CollectorMetrics, MetricFlush
from unilab.rl.runtime.offpolicy.thread_budget import apply_torch_thread_runtime
from unilab.rl.runtime.placement import InferenceTransport
from unilab.rl.runtime.seed import apply_worker_seed

# Exclusive phases for one collector loop iteration (one vectorized env.step).
# Every key is recorded once per iteration so the reported averages share one
# denominator and can be summed without double counting.
# - replay_write_ms: pack transitions and write them into the bounded ingress
COLLECTOR_TIMING_KEYS = (
    "inference_request_ms",
    "learner_action_wait_ms",
    "env_step_ms",
    "transition_extract_ms",
    "replay_write_ms",
    "metrics_publish_ms",
)
COLLECTOR_READY_TICK = -1
INFERENCE_SCHEDULING_POLICY = "sequential_transition_dependency"
INFERENCE_DEPENDENCY_GRAPH = {
    "observation_to_action": "observation[t] -> learner action[t]",
    "action_to_transition": "action[t] -> env.step(action[t])",
    "transition_to_next_observation": "env.step(action[t]) -> observation[t+1]",
    "replay_write_boundary": "transition[t] -> replay ingress[t] (order-preserving)",
    "reset_boundary": "terminal/reset rows are committed before observation[t+1]",
    "legal_concurrent_work": "replay/metrics work may overlap only after its transition is published",
}


def _inference_flight_metrics(inference_slot) -> tuple[int, int, int, int]:
    diagnostics = inference_slot.diagnostics
    published_tick = int(diagnostics["published_tick"])
    observation_tick = int(diagnostics["observation_tick"])
    action_tick = int(diagnostics["action_tick"])
    consumed_tick = int(diagnostics["consumed_tick"])
    queue_depth = max(published_tick - observation_tick, 0)
    action_backlog = max(action_tick - consumed_tick, 0)
    in_flight = max(published_tick - consumed_tick, 0)
    publication_lag = in_flight
    return queue_depth, action_backlog, in_flight, publication_lag


def timeout_bootstrap_mask(
    terminated: torch.Tensor,
    truncated: torch.Tensor,
) -> torch.Tensor:
    """Timeout-bootstrap eligibility mask for replay writes.

    Only rows that truncated without terminating may bootstrap from the
    terminal observation: a failure that coincides with the time limit is a
    real terminal transition and must never bootstrap. The returned float mask
    is what the off-policy replay ``truncated`` column stores (see
    ``ReplayBuffer.add``); the raw ``truncated`` flags remain the episode
    statistics source (``timeout_rate`` counts raw truncations, not
    bootstrap-eligible timeouts).
    """
    return (truncated & ~terminated).to(dtype=torch.float32)


def sample_offpolicy_actions(
    actor,
    algo_type: str,
    obs: torch.Tensor,
    prev_dones: torch.Tensor,
    priv_info: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sample actions using the algorithm's exploration policy."""
    if algo_type in ("sac", "flashsac"):
        return cast(
            torch.Tensor,
            actor.explore(obs, dones=prev_dones, deterministic=False),
        )
    adapter = get_offpolicy_actor_adapter(algo_type)
    if adapter is not None:
        if adapter.sample_actions is None:
            raise ValueError(
                f"OffPolicyActorAdapter for algo_type={algo_type!r} does not provide "
                "sample_actions."
            )
        return adapter.sample_actions(actor, obs, prev_dones, priv_info)
    raise ValueError(
        f"Unsupported off-policy algo_type for learner action sampling: {algo_type}. "
        "Custom actor types must register an OffPolicyActorAdapter via "
        "register_offpolicy_actor_adapter() or list their registration module in "
        "actor_adapter_modules."
    )


def _record_timing_ms(timing_accum_ms, timing_counts, key: str, value: float) -> None:
    timing_accum_ms[key] += float(value)
    timing_counts[key] += 1


def _record_phase_ms(cycle_timing_ms: dict[str, float], key: str, start_ns: int) -> int:
    end_ns = time.perf_counter_ns()
    cycle_timing_ms[key] += (end_ns - start_ns) / 1e6
    return end_ns


def _publish_coordination_tick(
    coordination_queue,
    tick_id: int,
    stop_event,
    *,
    timeout: float = 30.0,
    action: str,
) -> bool:
    deadline = time.monotonic() + timeout
    while stop_event is None or not stop_event.is_set():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out publishing off-policy {action}")
        try:
            coordination_queue.put(int(tick_id), timeout=0.1)
            return True
        except queue.Full:
            continue
    return False


def _publish_inference_tick(
    coordination_queue,
    tick_id: int,
    stop_event,
    *,
    timeout: float = 30.0,
) -> bool:
    return _publish_coordination_tick(
        coordination_queue,
        tick_id,
        stop_event,
        timeout=timeout,
        action=f"inference tick {tick_id}",
    )


def _publish_collector_ready(coordination_queue, stop_event) -> bool:
    """Signal that collector-owned cold-path initialization has completed."""
    return _publish_coordination_tick(
        coordination_queue,
        COLLECTOR_READY_TICK,
        stop_event,
        action="collector ready signal",
    )


def _wait_for_inference_tick(
    coordination_queue,
    tick_id: int,
    stop_event,
    *,
    learner_coordination: LearnerCoordinationState | None = None,
    learner_pid: int | None = None,
    timeout: float = 30.0,
) -> bool:
    """Wait for a learner action using phase and process liveness.

    There is no wall-clock deadline while the learner reports ``BUSY``: compile,
    graph capture, replay, and update durations are machine-dependent healthy
    work. The compatibility timeout is retained only for calls that do not
    provide learner state. With state present, the only timeout is a static
    ``WAITING_FOR_COLLECTOR`` loop, which indicates request-queue progress
    stopped rather than slow learning.
    """
    last_phase = LearnerPhase.STOPPED
    last_progress = -1
    last_progress_change = time.monotonic()
    while not stop_event.is_set():
        try:
            received_tick = int(coordination_queue.get(timeout=0.1))
        except queue.Empty:
            if stop_event.is_set():
                return False
            if learner_coordination is None:
                if time.monotonic() - last_progress_change >= timeout:
                    raise TimeoutError(f"Timed out waiting for off-policy inference tick {tick_id}")
                continue
            phase, progress = learner_coordination.snapshot()
            if not learner_pid_is_alive(learner_pid):
                raise RuntimeError(f"Learner process died before inference tick {tick_id}")
            if phase is LearnerPhase.STOPPED:
                if stop_event.is_set():
                    # Normal shutdown: the learner finished (or is tearing down
                    # after an error it reports itself) and requested stop
                    # before publishing STOPPED. Release without an error.
                    return False
                raise RuntimeError(f"Learner stopped before inference tick {tick_id}")
            if phase is not last_phase or progress != last_progress:
                last_phase = phase
                last_progress = progress
                last_progress_change = time.monotonic()
            elif (
                phase is LearnerPhase.WAITING_FOR_COLLECTOR
                and time.monotonic() - last_progress_change >= timeout
            ):
                raise TimeoutError(
                    "Off-policy learner request coordination stalled while waiting "
                    f"for inference tick {tick_id} (learner process is alive)"
                )
            continue
        if received_tick != int(tick_id):
            raise RuntimeError(
                f"Off-policy inference tick mismatch: expected {tick_id}, got {received_tick}"
            )
        return True
    return False


def _merge_reward_component_means(
    left: dict[str, float] | None,
    right: dict[str, float] | None,
    *,
    left_updates: int = 1,
    right_updates: int = 1,
) -> dict[str, float] | None:
    if left is None and right is None:
        return None
    if left is None:
        return dict(right or {})
    if right is None:
        return dict(left)
    total = left_updates + right_updates
    return {
        key: (left.get(key, 0.0) * left_updates + right.get(key, 0.0) * right_updates)
        / max(total, 1)
        for key in left.keys() | right.keys()
    }


def offpolicy_collector_fn(
    stop_event,
    env_factory: EnvFactory,
    num_envs: int,
    replay_buffer,
    inference_slot,
    inference_request_queue,
    inference_response_queue,
    algo_type: str = "sac",
    actor_adapter_modules: list[str] | tuple[str, ...] | None = None,
    metrics_queue=None,
    inference_transport: str = "host",
    inference_epoch: int = 0,
    collector_metrics_interval: int = 1,
    sim_backend: str = "mujoco",
    backend_device: str | None = None,
    env_cfg_override: dict | None = None,
    seed: int | None = None,
    trace_enabled: bool = False,
    trace_thread_time: bool = False,
    nan_guard_cfg=None,
    nan_guard_factory=None,
    torch_thread_runtime=None,
    backend_device_configurer: Callable[[str, str | None], str | None] | None = None,
    learner_coordination: LearnerCoordinationState | None = None,
    learner_pid: int | None = None,
):
    """Entry point for the off-policy collector subprocess.

    Error handling is provided by ``_collector_entry_wrapper`` in
    ``async_runner.py``.
    """
    _run_collector(
        stop_event=stop_event,
        env_factory=env_factory,
        num_envs=num_envs,
        replay_buffer=replay_buffer,
        inference_slot=inference_slot,
        inference_request_queue=inference_request_queue,
        inference_response_queue=inference_response_queue,
        algo_type=algo_type,
        actor_adapter_modules=actor_adapter_modules,
        metrics_queue=metrics_queue,
        inference_transport=inference_transport,
        inference_epoch=inference_epoch,
        collector_metrics_interval=collector_metrics_interval,
        sim_backend=sim_backend,
        backend_device=backend_device,
        env_cfg_override=env_cfg_override,
        seed=seed,
        trace_enabled=trace_enabled,
        trace_thread_time=trace_thread_time,
        nan_guard_cfg=nan_guard_cfg,
        nan_guard_factory=nan_guard_factory,
        torch_thread_runtime=torch_thread_runtime,
        backend_device_configurer=backend_device_configurer,
        learner_coordination=learner_coordination,
        learner_pid=learner_pid,
    )


def _run_collector(
    stop_event,
    env_factory,
    num_envs,
    replay_buffer,
    inference_slot,
    inference_request_queue,
    inference_response_queue,
    algo_type,
    actor_adapter_modules,
    metrics_queue,
    inference_transport,
    sim_backend,
    backend_device,
    env_cfg_override,
    seed,
    trace_enabled,
    trace_thread_time,
    inference_epoch=0,
    collector_metrics_interval=1,
    nan_guard_cfg=None,
    nan_guard_factory=None,
    torch_thread_runtime=None,
    backend_device_configurer=None,
    learner_coordination=None,
    learner_pid=None,
):
    # Spawn subprocesses do not inherit the parent's adapter registrations;
    # import the configured modules so registration side effects run here too.
    import_actor_adapter_modules(actor_adapter_modules)
    apply_torch_thread_runtime(torch_thread_runtime, role="collector", torch_module=torch)
    # Backend-owned process-device resolution/binding is injected by the owner
    # layer (e.g. ``unilab.base.process_device.configure_backend_process_device``,
    # which is top-level and pickleable across this spawn boundary); no
    # configurer means this collector needs no backend device binding.
    configured_backend_device = (
        backend_device_configurer(sim_backend, backend_device)
        if backend_device_configurer is not None
        else None
    )
    # The collector consumes only Python `random` and NumPy RNG (host-side env
    # physics/DR); inference is learner-owned, so no torch CPU/CUDA RNG is
    # consumed on this path. Tensor-native collectors only copy env-provided
    # CUDA obs into the ring (a deterministic copy), and mjwarp device
    # randomness is Warp-owned via the backend device binding above — so
    # skipping torch/CUDA seeding here remains correct.
    apply_worker_seed(seed, torch_runtime=False, cuda=False)

    trace_recorder = None
    if trace_enabled:
        from unilab.rl.logging.trace_event import TraceRecorder

        trace_recorder = TraceRecorder("offpolicy_collector")

    # Initialize environment through the injected factory (see
    # ``unilab.rl.contracts.env``); this layer never touches an env registry.
    env = env_factory(num_envs, env_cfg_override)
    requested_transport = inference_transport
    if requested_transport not in {item.value for item in InferenceTransport}:
        raise ValueError(
            "Collector inference transport must be one of "
            f"{{'host', 'cuda'}}, got {requested_transport!r}"
        )
    if "inference_transport" in (env_cfg_override or {}):
        raise ValueError("Collector env override must not set inference_transport")
    tensor_runtime_requested = requested_transport == "cuda"
    if tensor_runtime_requested:
        if inference_slot is None or inference_slot.device.type != "cuda":
            ring_device = getattr(inference_slot, "device", None)
            raise ValueError(
                f"CUDA collector inference requires a CUDA inference ring; got ring={ring_device}"
            )
    if "tensor_runtime" in (env_cfg_override or {}) or "tensor_runtime_device" in (
        env_cfg_override or {}
    ):
        raise ValueError(
            "Collector env override must not set removed tensor_runtime fields; "
            "inference transport is an explicit worker argument"
        )
    if nan_guard_cfg is not None and nan_guard_cfg.enabled:
        if nan_guard_factory is None:
            raise ValueError("collector diagnostics require an injected nan_guard_factory")
        env.set_nan_guard(
            nan_guard_factory(
                nan_guard_cfg,
                num_envs=env.num_envs,
                supports_state_playback=env.play_capabilities.supports_physics_state_playback,
            )
        )
    if env.state is None:
        env.init_state()

    replay_buffer.trace_recorder = trace_recorder
    replay_buffer.trace_thread_time = trace_thread_time
    replay_buffer.attach_stop_event(stop_event)
    from collections import defaultdict, deque

    total_steps = 0
    # Bounded rolling window of the most recent completed episodes; an
    # unbounded list here grows for the entire run.
    ep_rewards: deque[float] = deque(maxlen=100)
    ep_lengths: deque[int] = deque(maxlen=100)
    state = env.state
    assert state is not None
    # Fail closed on the observation contract: the collector runtime is
    # tensor-only. NumPy environment observations are never converted or
    # carried through the collector; NumPy remains valid only at explicit
    # non-environment boundaries (host diagnostics, serialization, tests).
    actor_observation = state.obs.get("obs")
    critic_observation = state.obs.get("critic", actor_observation)
    if not isinstance(actor_observation, torch.Tensor) or not isinstance(
        critic_observation, torch.Tensor
    ):
        raise TypeError(
            "Off-policy collector requires Torch tensor observations for the "
            "actor and critic observation groups; NumPy environment "
            "observations are not a supported collector runtime "
            f"(obs={type(actor_observation).__name__}, "
            f"critic={type(critic_observation).__name__})"
        )
    observed_env_device = actor_observation.device
    inference_ring_device = (
        inference_slot.device if inference_slot is not None else torch.device("cpu")
    )
    if tensor_runtime_requested:
        if observed_env_device != inference_ring_device:
            raise ValueError(
                "CUDA inference transport requires env observations on the ring device: "
                f"env={observed_env_device}, ring={inference_ring_device}"
            )
    elif requested_transport == "host" and observed_env_device.type == "cuda":
        raise ValueError(
            "Host inference transport received CUDA tensor observations from the "
            f"environment; env={observed_env_device}, ring={inference_ring_device}"
        )
    adapter = get_offpolicy_actor_adapter(algo_type)
    if adapter is not None and adapter.resolve_priv_info is not None:
        raise ValueError(
            "The tensor-only off-policy collector does not support privileged "
            "actor adapters (resolve_priv_info)"
        )
    reward_component_names = get_algo_capabilities(env).reward_component_names or ()
    collector_metrics: CollectorMetrics | None = CollectorMetrics(
        num_envs,
        interval=collector_metrics_interval,
        device=actor_observation.device,
        reward_component_names=reward_component_names,
    )

    ep_reward_components = defaultdict(list)
    timing_accum_ms: defaultdict[str, float] = defaultdict(float)
    timing_counts: defaultdict[str, int] = defaultdict(int)
    done_count_window = 0
    timeout_count_window = 0
    inference_queue_depth = 0
    inference_action_backlog = 0
    inference_in_flight = 0
    max_action_backlog_since_metric = 0
    max_in_flight_since_metric = 0
    max_publication_lag_since_metric = 0
    final_metrics_flushed = False
    pending_metric_flush: MetricFlush | None = None
    latest_curriculum_metrics: dict[str, float] = {}

    def enqueue_collector_metrics(
        *,
        final: bool = False,
        reward_history: deque[float] | None = None,
        length_history: deque[int] | None = None,
        reward_component_means: dict[str, float] | None = None,
    ) -> bool:
        """Publish one collector metric snapshot without losing window state."""

        nonlocal done_count_window, timeout_count_window, max_action_backlog_since_metric
        nonlocal max_in_flight_since_metric, max_publication_lag_since_metric
        assert metrics_queue is not None
        assert replay_buffer is not None
        import statistics

        msg: dict[str, Any] = {
            "total_steps": total_steps,
            "buffer_size": int(replay_buffer.size[0]),
        }
        if latest_curriculum_metrics:
            msg["curriculum_metrics"] = dict(latest_curriculum_metrics)
        if final:
            msg["metric_flush"] = "final"
        rewards_for_report = reward_history if reward_history is not None else ep_rewards
        lengths_for_report = length_history if length_history is not None else ep_lengths
        if rewards_for_report:
            msg["return_mean_ep100"] = statistics.mean(rewards_for_report)
            msg["mean_episode_length"] = (
                statistics.mean(lengths_for_report) if lengths_for_report else 0.0
            )
        components_mean = {
            key: statistics.mean(values) for key, values in ep_reward_components.items() if values
        }
        if reward_component_means:
            components_mean.update(reward_component_means)
        if components_mean:
            msg["reward_components"] = components_mean
            if reward_component_means:
                msg["reward_component_means"] = True
        if timing_counts:
            msg["collector_timing_ms"] = {
                key: total / timing_counts[key]
                for key, total in timing_accum_ms.items()
                if timing_counts[key] > 0
            }
        inference_diagnostics = {
            "queue_depth": int(inference_queue_depth),
            "action_backlog": int(inference_action_backlog),
            "max_action_backlog": int(max_action_backlog_since_metric),
            "in_flight": int(inference_in_flight),
            "max_in_flight": int(max_in_flight_since_metric),
            "wait_time_ms": float(cycle_timing_ms["learner_action_wait_ms"]),
            "publication_lag": int(inference_in_flight),
            "max_publication_lag": int(max_publication_lag_since_metric),
        }
        msg["collector_inference"] = inference_diagnostics
        # Keep the latest bounded-flight snapshot alongside the canonical scalar
        # stream so run summaries do not need to infer it from configured ring
        # capacity.
        msg["runtime_manifest"] = {"inference_flight": dict(inference_diagnostics)}
        timeout_rate = timeout_count_window / done_count_window if done_count_window > 0 else None
        if timeout_rate is not None:
            msg["timeout_rate"] = timeout_rate
        if trace_recorder:
            msg["trace_events"] = trace_recorder.drain_events()

        try:
            if final:
                # Shutdown can wait briefly for a healthy learner to drain a
                # full bounded queue; normal collection remains non-blocking.
                metrics_queue.put(msg, timeout=5.0)
            else:
                metrics_queue.put_nowait(msg)
        except Exception as exc:
            label = "final metrics enqueue" if final else "metrics enqueue"
            print(f"[OffPolicyWorker] {label} error: {exc}", file=sys.stderr)
            return False

        # Mutable source windows are consumed only after the queue accepts the
        # message, so a full queue cannot silently discard a completed window.
        ep_reward_components.clear()
        if timeout_rate is not None:
            done_count_window = 0
            timeout_count_window = 0
        max_action_backlog_since_metric = 0
        max_in_flight_since_metric = 0
        max_publication_lag_since_metric = 0
        if "collector_timing_ms" in msg:
            timing_accum_ms.clear()
            timing_counts.clear()
        return True

    def flush_final_collector_metrics() -> None:
        """Flush a partial device metric window once during worker shutdown."""

        nonlocal final_metrics_flushed
        nonlocal done_count_window, timeout_count_window, ep_rewards, ep_lengths
        if final_metrics_flushed or collector_metrics is None or metrics_queue is None:
            return
        final_metrics_flushed = True
        try:
            pending_flush = pending_metric_flush
            metric_flush = collector_metrics.final_flush()
            if pending_flush is not None:
                completed_flush = MetricFlush(
                    pending_flush.rewards + metric_flush.rewards,
                    pending_flush.lengths + metric_flush.lengths,
                    pending_flush.done_count + metric_flush.done_count,
                    pending_flush.timeout_count + metric_flush.timeout_count,
                    reward_component_means=_merge_reward_component_means(
                        pending_flush.reward_component_means,
                        metric_flush.reward_component_means,
                        left_updates=pending_flush.reward_component_updates,
                        right_updates=metric_flush.reward_component_updates,
                    ),
                    reward_component_updates=(
                        pending_flush.reward_component_updates
                        + metric_flush.reward_component_updates
                    ),
                )
            else:
                completed_flush = metric_flush
            if pending_flush is None and not (
                completed_flush.rewards
                or completed_flush.lengths
                or completed_flush.done_count
                or completed_flush.timeout_count
            ):
                return
            publication_rewards = deque(ep_rewards, maxlen=ep_rewards.maxlen)
            publication_lengths = deque(ep_lengths, maxlen=ep_lengths.maxlen)
            publication_rewards.extend(completed_flush.rewards)
            publication_lengths.extend(completed_flush.lengths)
            done_count_window += metric_flush.done_count
            timeout_count_window += metric_flush.timeout_count
            published = enqueue_collector_metrics(
                final=True,
                reward_history=publication_rewards,
                length_history=publication_lengths,
                reward_component_means=completed_flush.reward_component_means,
            )
            if published:
                ep_rewards = publication_rewards
                ep_lengths = publication_lengths
        except Exception as exc:
            print(f"[OffPolicyWorker] final collector metric flush error: {exc}", file=sys.stderr)

    obs_t: torch.Tensor | None = None
    critic_t: torch.Tensor | None = None
    next_obs_t: torch.Tensor | None = None
    next_critic_t: torch.Tensor | None = None
    rewards_t: torch.Tensor | None = None
    truncated_t: torch.Tensor | None = None
    combined_dones_t: torch.Tensor | None = None
    timeout_bootstrap_t: torch.Tensor | None = None
    done_mask_t: torch.Tensor | None = None
    terminal_obs_t: torch.Tensor | None = None
    terminal_critic_t: torch.Tensor | None = None
    obs_t, critic_t = split_obs_dict(state.obs)
    assert obs_t is not None
    assert critic_t is not None
    prev_dones_t = torch.zeros(num_envs, dtype=torch.float32, device=actor_observation.device)
    import time as _time

    backend_diagnostics = get_backend_runtime_diagnostics(env)
    if backend_diagnostics is None:
        backend_diagnostics = {
            "available": False,
            "unavailable_reason": "env does not expose backend runtime diagnostics",
        }
    runtime_manifest = {
        "inference_owner": "learner",
        "actor_owned": False,
        "weight_sync_attached": False,
        "torch_inference": False,
        "collector_accelerator_context": configured_backend_device is not None,
        "collector_backend_device": configured_backend_device,
        "cuda_context_initialized": bool(torch.cuda.is_initialized()),
        "inference_slot_device": getattr(inference_slot, "device", torch.device("cpu")),
        "inference_queue_capacity": int(getattr(inference_slot, "capacity", 0)),
        "inference_scheduling_policy": INFERENCE_SCHEDULING_POLICY,
        "inference_legal_max_in_flight": 1,
        "inference_dependency_graph": dict(INFERENCE_DEPENDENCY_GRAPH),
        "backend_diagnostics": dict(backend_diagnostics),
    }
    if trace_recorder:
        manifest_ns = _time.perf_counter_ns()
        trace_recorder.add_slice(
            "collector/runtime_manifest",
            category="collector",
            start_ns=manifest_ns,
            end_ns=manifest_ns,
            args=runtime_manifest,
        )
    if metrics_queue is not None:
        manifest_message: dict[str, Any] = {"runtime_manifest": runtime_manifest}
        if trace_recorder:
            manifest_message["trace_events"] = trace_recorder.drain_events()
        try:
            metrics_queue.put_nowait(manifest_message)
        except queue.Full:
            pass

    if not _publish_collector_ready(inference_request_queue, stop_event):
        return

    inference_tick = 0
    replay_add = replay_buffer.add_batch
    # Collection loop
    try:
        while not stop_event.is_set():
            cycle_timing_ms: dict[str, float] = dict.fromkeys(COLLECTOR_TIMING_KEYS, 0.0)
            phase_start_ns = _time.perf_counter_ns()
            assert obs_t is not None
            actor_input = obs_t
            request_ns = _time.perf_counter_ns()
            observation_published = False
            while not stop_event.is_set():
                observation_published = inference_slot.try_publish_observation(
                    tick_id=inference_tick,
                    observations=actor_input,
                    dones=prev_dones_t,
                    epoch=inference_epoch,
                    timeout_sec=0.1,
                )
                if observation_published:
                    break
            if not observation_published:
                break
            (
                inference_queue_depth,
                inference_action_backlog,
                inference_in_flight,
                publication_lag,
            ) = _inference_flight_metrics(inference_slot)
            max_action_backlog_since_metric = max(
                max_action_backlog_since_metric,
                inference_action_backlog,
            )
            max_in_flight_since_metric = max(max_in_flight_since_metric, inference_in_flight)
            max_publication_lag_since_metric = max(
                max_publication_lag_since_metric, publication_lag
            )
            if trace_recorder:
                trace_recorder.add_counter(
                    "inference/in_flight",
                    inference_in_flight,
                    category="collector_inference",
                )
            if not _publish_inference_tick(
                inference_request_queue,
                inference_tick,
                stop_event,
            ):
                break
            if trace_recorder:
                trace_recorder.add_slice(
                    "collector/inference_request",
                    category="collector",
                    start_ns=request_ns,
                    end_ns=_time.perf_counter_ns(),
                    args={"tick_id": inference_tick},
                )
            phase_start_ns = _record_phase_ms(
                cycle_timing_ms, "inference_request_ms", phase_start_ns
            )
            wait_ns = _time.perf_counter_ns()
            if not _wait_for_inference_tick(
                inference_response_queue,
                inference_tick,
                stop_event,
                learner_coordination=learner_coordination,
                learner_pid=learner_pid,
            ):
                break
            actions, policy_version = inference_slot.consume_action(
                tick_id=inference_tick,
                epoch=inference_epoch,
            )
            (
                inference_queue_depth,
                inference_action_backlog,
                inference_in_flight,
                publication_lag,
            ) = _inference_flight_metrics(inference_slot)
            if trace_recorder:
                trace_recorder.add_slice(
                    "collector/wait_for_learner_action",
                    category="collector",
                    start_ns=wait_ns,
                    end_ns=_time.perf_counter_ns(),
                    args={
                        "tick_id": inference_tick,
                        "policy_version": policy_version,
                    },
                )
            phase_start_ns = _record_phase_ms(
                cycle_timing_ms, "learner_action_wait_ms", phase_start_ns
            )
            inference_tick += 1

            # Step environment. The inference ring returns a Torch action
            # tensor on every platform; fail closed if that ever regresses.
            _env_ns = _time.perf_counter_ns()
            if not isinstance(actions, torch.Tensor):
                raise TypeError(
                    f"inference ring actions must be a Torch tensor, got {type(actions).__name__}"
                )
            state = env.step(actions)
            if trace_recorder:
                trace_recorder.add_slice(
                    "collector/env_step",
                    category="collector",
                    start_ns=_env_ns,
                    end_ns=_time.perf_counter_ns(),
                    args={"num_envs": num_envs},
                )
            phase_start_ns = _record_phase_ms(cycle_timing_ms, "env_step_ms", phase_start_ns)
            cycle_timing_ms.update(extract_env_step_breakdown_timing_ms(state.info))

            # Extract transition tensors
            next_obs_t, next_critic_t = split_obs_dict(state.obs)
            rewards_t = state.reward.to(dtype=torch.float32).ravel()
            truncated_t = state.truncated.to(dtype=torch.float32).ravel()
            combined_dones_t = (state.terminated | state.truncated).to(dtype=torch.float32)
            timeout_bootstrap_t = timeout_bootstrap_mask(state.terminated, state.truncated).ravel()
            assert next_obs_t is not None
            assert next_critic_t is not None
            assert rewards_t is not None
            assert truncated_t is not None
            assert combined_dones_t is not None
            assert timeout_bootstrap_t is not None
            prev_dones_t = combined_dones_t
            done_mask_t = combined_dones_t > 0.5
            assert collector_metrics is not None
            reward_components_t = state.info.get("reward_components")
            # Metrics keep the raw truncation flags: `timeout_rate` counts raw
            # episode truncations, which is a different statistic from the
            # bootstrap-eligible timeout mask written to replay below.
            collector_metrics.update(
                rewards_t,
                done_mask_t,
                truncated_t > 0.5,
                reward_components=reward_components_t,
            )
            terminal_final = state.final_observation
            if terminal_final is not None:
                terminal_obs_t = terminal_final.get("obs")
                terminal_critic_t = terminal_final.get("critic")
            else:
                terminal_obs_t = None
                terminal_critic_t = None
            assert done_mask_t is not None
            # ReplayBuffer `dones` follows the UniLab env lifecycle contract:
            # done = terminated | truncated. The replay `truncated` column
            # stores the timeout-bootstrap eligibility mask
            # (truncated & ~terminated): learners keep bootstrap enabled only
            # for timeout-only rows, so a failure that coincides with the time
            # limit is treated as a true terminal and never bootstraps. Done
            # rows still carry the env's real terminal observations.
            phase_start_ns = _record_phase_ms(
                cycle_timing_ms, "transition_extract_ms", phase_start_ns
            )
            _rb_ns = _time.perf_counter_ns()
            assert obs_t is not None
            assert critic_t is not None
            published = replay_add(
                obs_t,
                actions.to(dtype=torch.float32),
                rewards_t,
                next_obs_t,
                combined_dones_t,
                timeout_bootstrap_t,
                terminal_mask=done_mask_t,
                terminal_next_obs=terminal_obs_t,
                critic=critic_t,
                next_critic=next_critic_t,
                terminal_next_critic=terminal_critic_t,
                completion=state.info.get("completion"),
            )
            if not published and not stop_event.is_set():
                raise RuntimeError("replay ingress closed before collector shutdown")
            if trace_recorder:
                trace_recorder.add_slice(
                    "collector/replay_add",
                    category="collector",
                    start_ns=_rb_ns,
                    end_ns=_time.perf_counter_ns(),
                )
            phase_start_ns = _record_phase_ms(cycle_timing_ms, "replay_write_ms", phase_start_ns)

            # Track episode rewards - vectorized
            obs_t = next_obs_t
            critic_t = next_critic_t
            total_steps += num_envs

            # Extract reward components and latest curriculum state from env info.
            log_info = state.info.get("log", {})
            if log_info:
                for k, v in log_info.items():
                    if k.startswith("reward/"):
                        ep_reward_components[k.removeprefix("reward/")].append(v)
                    elif k.startswith("Curriculum/") and isinstance(v, (int, float)):
                        latest_curriculum_metrics[k] = float(v)

            phase_start_ns = _record_phase_ms(cycle_timing_ms, "metrics_publish_ms", phase_start_ns)

            # Send metrics every collector cycle so learner-side reward and
            # throughput displays track the current policy without extra lag.
            if metrics_queue is not None and collector_metrics.ready:
                metric_flush = collector_metrics.flush()
                pending_flush = pending_metric_flush
                completed_flush = metric_flush
                if pending_flush is not None:
                    completed_flush = MetricFlush(
                        pending_flush.rewards + metric_flush.rewards,
                        pending_flush.lengths + metric_flush.lengths,
                        pending_flush.done_count + metric_flush.done_count,
                        pending_flush.timeout_count + metric_flush.timeout_count,
                        reward_component_means=_merge_reward_component_means(
                            pending_flush.reward_component_means,
                            metric_flush.reward_component_means,
                            left_updates=pending_flush.reward_component_updates,
                            right_updates=metric_flush.reward_component_updates,
                        ),
                        reward_component_updates=(
                            pending_flush.reward_component_updates
                            + metric_flush.reward_component_updates
                        ),
                    )
                publication_rewards = deque(ep_rewards, maxlen=ep_rewards.maxlen)
                publication_lengths = deque(ep_lengths, maxlen=ep_lengths.maxlen)
                publication_rewards.extend(completed_flush.rewards)
                publication_lengths.extend(completed_flush.lengths)
                done_count_window += metric_flush.done_count
                timeout_count_window += metric_flush.timeout_count
                published = enqueue_collector_metrics(
                    reward_history=publication_rewards,
                    length_history=publication_lengths,
                    reward_component_means=completed_flush.reward_component_means,
                )
                if published:
                    ep_rewards = publication_rewards
                    ep_lengths = publication_lengths
                    pending_metric_flush = None
                else:
                    pending_metric_flush = completed_flush
            for key, value in cycle_timing_ms.items():
                _record_timing_ms(timing_accum_ms, timing_counts, key, value)

    finally:
        flush_final_collector_metrics()
        cleanup = getattr(env, "cleanup", None)
        if callable(cleanup):
            cleanup()
        else:
            env.close()
        inference_slot.close()
        replay_buffer.release_ipc()
        # Drop the final tensor/state references before the spawn process
        # exits. This prevents CUDA IPC handles from outliving explicit env
        # teardown until interpreter finalization.
        state = None
        obs_t = None
        critic_t = None
        next_obs_t = None
        next_critic_t = None
        terminal_obs_t = None
        terminal_critic_t = None
        collector_metrics = None
        inference_slot = None
        replay_buffer = None
        env = None
        import gc

        gc.collect()
        if metrics_queue is not None and trace_recorder:
            try:
                metrics_queue.put_nowait({"trace_events": trace_recorder.drain_events()})
            except Exception:
                pass
