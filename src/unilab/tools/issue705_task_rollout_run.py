"""Validation of persisted production evidence for Issue #705 task rollout.

The validator is deliberately independent of the subprocess runner.  It only
consumes the public JSON receipts written by ``train_rsl_rl.py`` and therefore
can be reused by the integration test and a later Phase 7 evidence gate.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from unilab.tools.issue705_task_rollout import TaskRolloutEntry

_MISSING = object()
_ZERO_RUNTIME_COUNTERS = (
    "host_to_device_transfers",
    "device_to_host_transfers",
    "host_to_device_bytes",
    "device_to_host_bytes",
    "global_synchronizations",
    "backend_allocations",
    "dynamic_getter_calls",
    "selector_resolutions",
    "asset_metadata_reads",
    "registry_lookups",
)


@dataclass(frozen=True)
class TaskRolloutRunValidationReport:
    """Fail-closed result for one persisted seed run."""

    seed: int
    errors: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.errors


def _mapping(value: object, label: str, errors: list[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        errors.append(f"{label}: expected mapping")
        return {}
    return value


def _required(mapping: Mapping[str, Any], key: str, label: str, errors: list[str]) -> object:
    value = mapping.get(key, _MISSING)
    if value is _MISSING:
        errors.append(f"{label}.{key}: missing")
        return _MISSING
    return value


def _equal(
    mapping: Mapping[str, Any], key: str, expected: object, label: str, errors: list[str]
) -> None:
    actual = _required(mapping, key, label, errors)
    if actual is not _MISSING and actual != expected:
        errors.append(f"{label}.{key}: expected {expected!r}, got {actual!r}")


def _finite_number(
    mapping: Mapping[str, Any], key: str, label: str, errors: list[str], *, positive: bool = False
) -> None:
    value = _required(mapping, key, label, errors)
    if value is _MISSING:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        errors.append(f"{label}.{key}: expected numeric value")
        return
    number = float(value)
    if not math.isfinite(number) or (positive and number <= 0.0):
        errors.append(f"{label}.{key}: expected finite positive value")


def _non_negative_integer(
    mapping: Mapping[str, Any], key: str, label: str, errors: list[str]
) -> int | None:
    value = _required(mapping, key, label, errors)
    if value is _MISSING:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        errors.append(f"{label}.{key}: expected non-negative integer")
        return None
    return value


def _graph_identity(graph: Mapping[str, Any], label: str, errors: list[str]) -> tuple[object, ...]:
    active = _required(graph, "active_keys", label, errors)
    if not isinstance(active, list) or len(active) != 1:
        errors.append(f"{label}.active_keys: expected exactly one active graph key")
        return ()
    key = _mapping(active[0], f"{label}.active_keys[0]", errors)
    for field in (
        "backend_type",
        "plan_fingerprint",
        "num_envs",
        "state_dtype",
        "control_dtype",
        "physics_substeps",
        "storage_generation",
        "storage_fingerprint",
        "contract_version",
    ):
        _required(key, field, f"{label}.active_keys[0]", errors)
    return tuple(key.get(field) for field in (
        "backend_type",
        "plan_fingerprint",
        "num_envs",
        "state_dtype",
        "control_dtype",
        "physics_substeps",
        "storage_generation",
        "storage_fingerprint",
        "contract_version",
    ))


def _validate_graph(
    graph: Mapping[str, Any],
    *,
    label: str,
    entry: TaskRolloutEntry,
    expected_launches: int,
    errors: list[str],
) -> tuple[object, ...]:
    _equal(graph, "backend_type", entry.backend, label, errors)
    _equal(graph, "execution_mode", "cuda_graph", label, errors)
    identity = _graph_identity(graph, label, errors)
    if identity:
        if identity[0] != entry.backend:
            errors.append(f"{label}.active_keys[0].backend_type: backend mismatch")
        if identity[1] != entry.rollout_compiled_signature.backend_plan_fingerprint:
            errors.append(f"{label}.active_keys[0].plan_fingerprint: rollout signature mismatch")
        if identity[2] != entry.num_envs:
            errors.append(f"{label}.active_keys[0].num_envs: rollout budget mismatch")
        if identity[3:] and (identity[3], identity[4]) != ("float32", "float32"):
            errors.append(f"{label}.active_keys[0]: state/control dtype must be float32")
        if identity[5] is not None and (not isinstance(identity[5], int) or identity[5] <= 0):
            errors.append(f"{label}.active_keys[0].physics_substeps: must be positive")

    capture = _non_negative_integer(graph, "capture_count", label, errors)
    launches = _non_negative_integer(graph, "launch_count", label, errors)
    _non_negative_integer(graph, "recapture_count", label, errors)
    _non_negative_integer(graph, "stale_rejection_count", label, errors)
    _non_negative_integer(graph, "eager_fallback_count", label, errors)
    _non_negative_integer(graph, "storage_verification_count", label, errors)
    _equal(graph, "recapture_count", 0, label, errors)
    _equal(graph, "stale_rejection_count", 0, label, errors)
    _equal(graph, "eager_fallback_count", 0, label, errors)
    _equal(graph, "instrumentation_complete", True, label, errors)
    if capture is not None and capture <= 0:
        errors.append(f"{label}.capture_count: graph was never captured")
    if launches is not None and launches != expected_launches:
        errors.append(
            f"{label}.launch_count: expected lifecycle launch total {expected_launches}, got {launches}"
        )
    return identity


def _validate_lifecycle(
    lifecycle: Mapping[str, Any], *, label: str, errors: list[str]
) -> tuple[int | None, int | None, int | None]:
    barriers = _non_negative_integer(lifecycle, "runtime_barriers", label, errors)
    step = _non_negative_integer(lifecycle, "step_graph_launches", label, errors)
    reset = _non_negative_integer(lifecycle, "reset_graph_launches", label, errors)
    forward = _non_negative_integer(lifecycle, "forward_graph_launches", label, errors)
    refresh = _non_negative_integer(lifecycle, "state_refreshes", label, errors)
    _equal(lifecycle, "instrumentation_complete", True, label, errors)
    if barriers is not None and step is not None and reset is not None and barriers != step + reset:
        errors.append(f"{label}.runtime_barriers: does not equal step + reset graph launches")
    if step is not None and reset is not None and forward is not None and refresh is not None:
        if refresh < step + forward:
            errors.append(f"{label}.state_refreshes: does not cover state-producing graphs")
    return step, reset, forward


def _validate_zero_runtime_counters(
    traffic: Mapping[str, Any], *, label: str, expected_steps: int, errors: list[str]
) -> None:
    _equal(traffic, "policy_steps", expected_steps, label, errors)
    _equal(traffic, "step_barriers", expected_steps, label, errors)
    reset_barriers = _non_negative_integer(traffic, "reset_barriers", label, errors)
    if reset_barriers is not None and reset_barriers <= 0:
        errors.append(f"{label}.reset_barriers: expected reset activity")
    _equal(traffic, "state_materializations", expected_steps + (reset_barriers or 0), label, errors)
    for key in _ZERO_RUNTIME_COUNTERS:
        _equal(traffic, key, 0, label, errors)
    _equal(traffic, "instrumentation_complete", True, label, errors)


def validate_task_rollout_run(
    entry: TaskRolloutEntry,
    *,
    seed: int,
    run_dir: Path,
    run_config: Mapping[str, Any],
    run_summary: Mapping[str, Any],
    stdout: str,
) -> TaskRolloutRunValidationReport:
    """Validate one public training receipt against a frozen rollout entry."""

    errors: list[str] = []
    label = f"{'/'.join(entry.key)}/seed={seed}"
    if seed not in entry.seeds:
        errors.append(f"{label}: seed is not frozen in rollout plan")

    run = _mapping(run_config.get("run"), f"{label}.run_config.run", errors)
    config = _mapping(run_config.get("config"), f"{label}.run_config.config", errors)
    training = _mapping(config.get("training"), f"{label}.config.training", errors)
    algo = _mapping(config.get("algo"), f"{label}.config.algo", errors)
    env = _mapping(config.get("env"), f"{label}.config.env", errors)

    _equal(run, "algo", "ppo", label, errors)
    _equal(run, "task", entry.env_name, label, errors)
    _equal(run, "sim_backend", entry.backend, label, errors)
    device = _required(run, "device", label, errors)
    if not isinstance(device, str) or not device.startswith("cuda"):
        errors.append(f"{label}.run.device: expected CUDA device")
    _equal(run, "configured_seed", seed, label, errors)
    _equal(run, "effective_seed", seed, label, errors)

    training_checks: tuple[tuple[str, object], ...] = (
        ("task_name", entry.env_name),
        ("sim_backend", entry.backend),
        ("execution_profile", entry.execution_profile),
        ("no_play", True),
        ("play_render_mode", "none"),
    )
    for key, expected in training_checks:
        _equal(training, key, expected, f"{label}.config.training", errors)
    algo_checks: tuple[tuple[str, object], ...] = (
        ("seed", seed),
        ("num_envs", entry.num_envs),
        ("num_steps_per_env", entry.num_steps_per_env),
        ("max_iterations", entry.max_iterations),
        ("runtime_impl", entry.runtime_impl),
        ("runtime_resolver", entry.runtime_resolver),
        ("capture_performance_diagnostics", True),
    )
    for key, expected in algo_checks:
        _equal(algo, key, expected, f"{label}.config.algo", errors)

    domain_rand = _mapping(env.get("domain_rand"), f"{label}.config.env.domain_rand", errors)
    for key in entry.disabled_domain_rand:
        _equal(domain_rand, key, False, f"{label}.config.env.domain_rand", errors)
    noise = _mapping(env.get("noise_config"), f"{label}.config.env.noise_config", errors)
    _equal(noise, "level", 0, f"{label}.config.env.noise_config", errors)

    contract_snapshot = _mapping(
        run_config.get("contract_snapshot"), f"{label}.run_config.contract_snapshot", errors
    )
    policy = _mapping(
        contract_snapshot.get("manager.policy_abi"),
        f"{label}.contract_snapshot.manager.policy_abi",
        errors,
    )
    rollout_signature = entry.rollout_compiled_signature
    for key, expected in (
        ("task_key", rollout_signature.task_key),
        ("executor_key", rollout_signature.executor_key),
        ("plan_fingerprint", rollout_signature.task_plan_fingerprint),
        ("policy_abi_fingerprint", rollout_signature.policy_abi_fingerprint),
        ("execution_profile", entry.execution_profile),
    ):
        _equal(policy, key, expected, f"{label}.contract_snapshot.manager.policy_abi", errors)

    _equal(run_summary, "status", "completed", label, errors)
    _equal(run_summary, "algo", "ppo", label, errors)
    _equal(run_summary, "task", entry.env_name, label, errors)
    _equal(run_summary, "sim_backend", entry.backend, label, errors)
    _equal(run_summary, "configured_seed", seed, label, errors)
    _equal(run_summary, "effective_seed", seed, label, errors)
    _equal(run_summary, "completed_iterations", entry.max_iterations, label, errors)
    _equal(
        run_summary,
        "total_env_steps",
        entry.num_envs * entry.num_steps_per_env * entry.max_iterations,
        label,
        errors,
    )
    for key in ("training_wall_time_sec", "wall_time_sec"):
        _finite_number(run_summary, key, label, errors, positive=True)
    for key in ("peak_process_rss_bytes", "peak_gpu_memory_allocated_bytes", "peak_gpu_memory_reserved_bytes"):
        if run_summary.get(key) is not None:
            _finite_number(run_summary, key, label, errors, positive=True)

    checkpoint = run_dir / "model_0.pt"
    if not checkpoint.is_file() or checkpoint.stat().st_size <= 0:
        errors.append(f"{label}: model_0.pt checkpoint is missing or empty")
    recorded_checkpoint = run_summary.get("last_checkpoint")
    if not isinstance(recorded_checkpoint, str) or Path(recorded_checkpoint).resolve() != checkpoint.resolve():
        errors.append(f"{label}.last_checkpoint: does not identify model_0.pt")

    performance = _mapping(
        run_summary.get("runtime_performance_diagnostics"),
        f"{label}.runtime_performance_diagnostics",
        errors,
    )
    before = _mapping(
        run_summary.get("runtime_performance_diagnostics_before_training"),
        f"{label}.runtime_performance_diagnostics_before_training",
        errors,
    )
    for diagnostics, diagnostics_label in ((performance, "performance"), (before, "before")):
        _equal(diagnostics, "backend_type", entry.backend, f"{label}.{diagnostics_label}", errors)
        _equal(diagnostics, "model_targets", list(entry.expected_model_targets), f"{label}.{diagnostics_label}", errors)
        _equal(diagnostics, "recompute_kind", "none", f"{label}.{diagnostics_label}", errors)
        _equal(diagnostics, "direct_fields", [], f"{label}.{diagnostics_label}", errors)
        _equal(diagnostics, "derived_fields", [], f"{label}.{diagnostics_label}", errors)
        _equal(diagnostics, "recompute_capture_count", 0, f"{label}.{diagnostics_label}", errors)
        _equal(diagnostics, "recompute_launch_count", 0, f"{label}.{diagnostics_label}", errors)
        _equal(diagnostics, "materialization", None, f"{label}.{diagnostics_label}", errors)
        _equal(diagnostics, "instrumentation_complete", True, f"{label}.{diagnostics_label}", errors)

    lifecycle = _mapping(performance.get("lifecycle"), f"{label}.performance.lifecycle", errors)
    step_launches, reset_launches, forward_launches = _validate_lifecycle(
        lifecycle, label=f"{label}.performance.lifecycle", errors=errors
    )
    expected_step_launches = entry.num_steps_per_env * entry.max_iterations
    if step_launches is not None and step_launches != expected_step_launches:
        errors.append(f"{label}.performance.lifecycle.step_graph_launches: rollout step count mismatch")
    if reset_launches is not None and reset_launches <= 0:
        errors.append(f"{label}.performance.lifecycle.reset_graph_launches: no reset activity")
    if forward_launches is not None and forward_launches <= 0:
        errors.append(f"{label}.performance.lifecycle.forward_graph_launches: no forward activity")
    before_lifecycle = _mapping(before.get("lifecycle"), f"{label}.before.lifecycle", errors)
    before_step, _, _ = _validate_lifecycle(
        before_lifecycle, label=f"{label}.before.lifecycle", errors=errors
    )
    if before_step is not None and before_step != 0:
        errors.append(f"{label}.before.lifecycle.step_graph_launches: training ran before boundary")

    graph = _mapping(performance.get("graph"), f"{label}.performance.graph", errors)
    graph_identity = _validate_graph(
        graph,
        label=f"{label}.performance.graph",
        entry=entry,
        expected_launches=(step_launches or 0) + (reset_launches or 0) + (forward_launches or 0),
        errors=errors,
    )
    before_graph = _mapping(before.get("graph"), f"{label}.before.graph", errors)
    before_identity = _validate_graph(
        before_graph,
        label=f"{label}.before.graph",
        entry=entry,
        expected_launches=2,
        errors=errors,
    )
    if graph_identity and before_identity and graph_identity != before_identity:
        errors.append(f"{label}: graph identity changed between pre-training and final receipt")

    traffic = _mapping(
        run_summary.get("runtime_traffic_diagnostics"),
        f"{label}.runtime_traffic_diagnostics",
        errors,
    )
    expected_steps = entry.num_steps_per_env * entry.max_iterations
    _validate_zero_runtime_counters(
        traffic, label=f"{label}.runtime_traffic_diagnostics", expected_steps=expected_steps, errors=errors
    )
    stability = _mapping(
        run_summary.get("runtime_stability_diagnostics"),
        f"{label}.runtime_stability_diagnostics",
        errors,
    )
    buffers = stability.get("buffers")
    state_buffers = stability.get("state_buffers")
    if not isinstance(buffers, list) or not buffers:
        errors.append(f"{label}.runtime_stability_diagnostics.buffers: expected non-empty list")
    if not isinstance(state_buffers, list) or not state_buffers:
        errors.append(f"{label}.runtime_stability_diagnostics.state_buffers: expected non-empty list")
    _equal(stability, "warm_numeric_allocations", 0, label, errors)
    _equal(stability, "address_churn", 0, label, errors)
    observations = _non_negative_integer(stability, "observations", label, errors)
    if observations is not None and observations <= 0:
        errors.append(f"{label}.runtime_stability_diagnostics.observations: no warm observations")
    _equal(stability, "instrumentation_complete", True, label, errors)
    stability_traffic = _mapping(stability.get("traffic"), f"{label}.stability.traffic", errors)
    if stability_traffic != traffic:
        errors.append(f"{label}: stability traffic receipt differs from runtime traffic receipt")
    stability_graph = _mapping(stability.get("graph"), f"{label}.stability.graph", errors)
    stability_identity = _validate_graph(
        stability_graph,
        label=f"{label}.stability.graph",
        entry=entry,
        expected_launches=(step_launches or 0) + (reset_launches or 0) + (forward_launches or 0),
        errors=errors,
    )
    if graph_identity and stability_identity and graph_identity != stability_identity:
        errors.append(f"{label}: graph identity changed in stability receipt")

    _equal(run_summary, "runtime_event_traffic_diagnostics", {}, label, errors)
    wrapper = _mapping(run_summary.get("wrapper_traffic_diagnostics"), f"{label}.wrapper", errors)
    _equal(wrapper, "action_publications", expected_steps, label, errors)
    _equal(wrapper, "observation_snapshots", expected_steps + 1, label, errors)
    _equal(wrapper, "finite_metric_materializations", 1, label, errors)
    _finite_number(wrapper, "finite_metric_device_to_host_bytes", label, errors, positive=True)
    logging = _mapping(run_summary.get("logging_traffic_diagnostics"), f"{label}.logging", errors)
    _equal(logging, "rollout_steps", expected_steps, label, errors)
    _equal(logging, "metric_materializations", 1, label, errors)
    _finite_number(logging, "metric_device_to_host_bytes", label, errors, positive=True)
    memory = run_summary.get("iteration_memory_diagnostics")
    if not isinstance(memory, list) or len(memory) != entry.max_iterations:
        errors.append(f"{label}.iteration_memory_diagnostics: expected one sample per iteration")

    for marker in ("Using device: cuda", "Learning iteration 0/1", "Collection time:", "Learning time:"):
        if marker not in stdout:
            errors.append(f"{label}.stdout: missing {marker!r}")

    return TaskRolloutRunValidationReport(seed=seed, errors=tuple(errors))


__all__ = ["TaskRolloutRunValidationReport", "validate_task_rollout_run"]
