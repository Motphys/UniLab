"""Frozen acceptance thresholds and candidate gates for managed MuJoCo/MJWarp rollout."""

from __future__ import annotations

import json
import math
import re
import statistics
import subprocess
import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast

from omegaconf import OmegaConf

from unilab.tools.g1_baseline_provenance import (
    G1BaselinePlan,
    load_g1_baseline_artifact,
    load_g1_baseline_plan,
    sha256_file,
)

SCHEMA_VERSION = 1
ISSUE = 705
THRESHOLD_SET_ID = "g1-manager-mjwarp-v1"
THRESHOLD_MANIFEST_PATH = Path("tests/acceptance/manager_mjwarp/g1_threshold_manifest.yaml")
FREEZE_RECEIPT_PATH = Path("tests/acceptance/manager_mjwarp/g1_threshold_freeze_receipt.yaml")
AMENDMENT_SCHEMA_VERSION = 1
AMENDMENT_ISSUE = 807
AMENDMENT_ID = "g1-phase5-ppo-rss-ratio-v1"
AMENDMENT_MANIFEST_PATH = Path(
    "tests/acceptance/manager_mjwarp/g1_phase5_ppo_threshold_amendment.yaml"
)
AMENDMENT_FREEZE_RECEIPT_PATH = Path(
    "tests/acceptance/manager_mjwarp/g1_phase5_ppo_threshold_amendment_freeze_receipt.yaml"
)
AMENDMENT_ADR_PATH = Path("docs/sphinx/source/adr/ADR-0006-phase5-ppo-rss-threshold-amendment.md")
BASELINE_PLAN_PATH = Path("tests/acceptance/manager_mjwarp/g1_mujoco_baseline_plan.yaml")
BASELINE_ARTIFACT_PATH = Path(
    "tests/acceptance/manager_mjwarp/artifacts/g1_mujoco_phase0_baseline.json"
)

_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_GIT_BLOB_RE = re.compile(r"^[0-9a-f]{40,64}$")
_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


class ThresholdValidationError(ValueError):
    def __init__(self, source: Path, errors: Iterable[str]) -> None:
        self.source = source
        self.errors = tuple(errors)
        detail = "\n".join(f"- {error}" for error in self.errors)
        super().__init__(
            f"invalid managed MuJoCo/MJWarp rollout threshold data {source}:\n{detail}"
        )


@dataclass(frozen=True)
class ThresholdManifest:
    source_path: Path
    data: Mapping[str, Any]

    @property
    def gates(self) -> Mapping[str, Any]:
        return cast(Mapping[str, Any], self.data["gates"])

    @property
    def baseline_reference(self) -> Mapping[str, Any]:
        return cast(Mapping[str, Any], self.data["baseline_reference"])


@dataclass(frozen=True)
class FreezeReceipt:
    source_path: Path
    data: Mapping[str, Any]
    git_history_verified: bool = False

    @property
    def freeze_commit(self) -> str:
        return str(self.data["freeze_commit"])


@dataclass(frozen=True)
class ThresholdAmendment:
    source_path: Path
    data: Mapping[str, Any]

    @property
    def amendment_id(self) -> str:
        return str(self.data["amendment_id"])

    @property
    def host_memory_ratio_max(self) -> float:
        change = cast(Mapping[str, Any], self.data["change"])
        return float(change["amended_value"])


@dataclass(frozen=True)
class AmendmentFreezeReceipt:
    source_path: Path
    data: Mapping[str, Any]
    git_history_verified: bool = False

    @property
    def freeze_commit(self) -> str:
        return str(self.data["freeze_commit"])


@dataclass(frozen=True)
class CandidateProvenance:
    threshold_set_id: str
    threshold_manifest_sha256: str
    threshold_freeze_commit: str
    candidate_commit: str
    source_dirty: bool


@dataclass(frozen=True)
class CandidateEnvMetrics:
    process_repeats: int
    step_p50_median_ms: float
    step_p95_median_ms: float
    throughput_median_env_steps_per_sec: float
    throughput_population_cv: float
    host_memory_median_bytes: int
    host_memory_metric: str


@dataclass(frozen=True)
class CandidateTrainingMetrics:
    seeds: tuple[int, ...]
    failed_seeds: tuple[int, ...]
    nan_seeds: tuple[int, ...]
    fps_p50_median: float
    reward_auc_median: float
    final_reward_p50_median: float
    episode_length_p50_median: float
    peak_rss_median_bytes: int
    peak_gpu_reserved_median_bytes: int


@dataclass(frozen=True)
class CandidateDrMetrics:
    process_repeats: int
    actual_rows: int
    disabled_total_p50_median_ms: float
    disabled_total_p95_median_ms: float
    enabled_total_p50_median_ms: float
    enabled_total_p95_median_ms: float
    enabled_extra_resident_bytes: int
    resident_memory_metric: str


@dataclass(frozen=True)
class CandidateDeviceMetrics:
    gpu_capacity_bytes: int
    peak_gpu_reserved_bytes: int
    h2d_per_policy_step: float
    d2h_per_policy_step: float
    host_global_sync_per_policy_step: float
    metrics_materializations: int
    profiler_reconciled: bool
    profiler_trace_refs: tuple[str, ...]
    profiler_trace_sha256s: tuple[str, ...]


@dataclass(frozen=True)
class CandidateCompatibilityMetrics:
    identity_exact: bool
    policy_abi_exact: bool
    lifecycle_exact: bool
    unsupported_fail_closed: bool
    fallback_used: bool
    mandatory_cases_skipped: int
    manager_obs_max_abs: float
    manager_obs_max_rel: float
    manager_reward_max_abs: float
    manager_reward_max_rel: float
    physics_one_step_max_abs: float
    physics_one_step_max_rel: float
    physics_trajectory_qpos_max_abs: float
    physics_trajectory_qpos_max_rel: float
    physics_trajectory_qvel_max_abs: float
    physics_trajectory_qvel_max_rel: float


@dataclass(frozen=True)
class CandidateRawEvidence:
    planned_case_ids: tuple[str, ...]
    observed_case_ids: tuple[str, ...]
    included_case_ids: tuple[str, ...]
    failed_case_ids: tuple[str, ...]
    filtered_case_ids: tuple[str, ...]
    raw_artifact_sha256: str
    aggregate_recomputed_from_raw: bool


@dataclass(frozen=True)
class CandidateGateInput:
    profile: str
    provenance: CandidateProvenance
    raw_evidence: CandidateRawEvidence
    environment: Mapping[int, CandidateEnvMetrics]
    training: CandidateTrainingMetrics
    dr: Mapping[float, CandidateDrMetrics]
    compatibility: CandidateCompatibilityMetrics
    device: CandidateDeviceMetrics | None = None


_ROOT_KEYS = (
    "schema_version",
    "issue",
    "threshold_set_id",
    "state",
    "baseline",
    "measurement",
    "gates",
    "baseline_reference",
    "governance",
)
_BASELINE_KEYS = (
    "baseline_id",
    "plan_path",
    "plan_sha256",
    "artifact_path",
    "artifact_sha256",
    "source_commit",
    "task",
    "backend",
    "owner_yaml",
    "hardware",
)
_BASELINE_HARDWARE_KEYS = (
    "cpu_model",
    "gpu_name",
    "gpu_uuid",
    "gpu_memory_mib",
    "driver_version",
)
_MEASUREMENT_KEYS = (
    "dtype",
    "batch_sizes",
    "env_process_repeats",
    "ppo_seeds",
    "ppo_iterations",
    "ppo_warmup_iterations",
    "dr_num_envs",
    "dr_modes",
    "reset_densities",
    "dr_process_repeats",
    "prohibit_filtering",
)
_GATE_KEYS = (
    "performance",
    "training",
    "dr",
    "memory",
    "transfer",
    "compatibility",
)
_PERFORMANCE_KEYS = (
    "env_metric",
    "process_aggregation",
    "p50_latency_ratio_max",
    "p95_latency_ratio_max",
    "throughput_ratio_min",
    "max_population_cv_by_batch",
    "require_all_batches",
)
_TRAINING_KEYS = (
    "fps_p50_median_ratio_min",
    "reward_auc_median_drop_max",
    "final_reward_p50_median_drop_max",
    "episode_length_median_ratio_min",
    "max_failed_seeds",
    "max_nan_seeds",
    "require_all_seeds",
    "performance_and_behavior_must_both_pass",
)
_DR_KEYS = (
    "enabled_to_disabled_p50_ratio_max",
    "enabled_to_disabled_p95_ratio_max",
    "enabled_extra_resident_bytes_max",
    "require_all_modes_and_densities",
    "actual_rows_must_match",
)
_MEMORY_KEYS = (
    "host_preferred_metric_ratio_max",
    "device_peak_reserved_capacity_ratio_max",
    "device_peak_reserved_growth_bytes_max",
)
_TRANSFER_KEYS = (
    "h2d_per_policy_step_max",
    "d2h_per_policy_step_max",
    "host_global_sync_per_policy_step_max",
    "metrics_materialization_must_be_separate",
    "profiler_reconciliation_required",
)
_COMPATIBILITY_KEYS = (
    "exact_match_fields",
    "manager_obs_atol",
    "manager_obs_rtol",
    "manager_reward_atol",
    "manager_reward_rtol",
    "physics_one_step_atol",
    "physics_one_step_rtol",
    "physics_trajectory_steps",
    "physics_trajectory_qpos_atol",
    "physics_trajectory_qpos_rtol",
    "physics_trajectory_qvel_atol",
    "physics_trajectory_qvel_rtol",
    "lifecycle_exact",
    "unsupported_fail_closed",
    "fallback_forbidden",
    "mandatory_skip_xfail_forbidden",
)
_GOVERNANCE_KEYS = (
    "candidate_artifact_required_fields",
    "candidate_commit_must_descend_from_freeze",
    "candidate_and_threshold_same_commit_forbidden",
    "candidate_result_and_threshold_change_same_pr_forbidden",
    "threshold_change_requires",
    "failure_semantics",
)
_AMENDMENT_ROOT_KEYS = (
    "schema_version",
    "issue",
    "amendment_id",
    "state",
    "base_threshold",
    "scope",
    "change",
    "governance",
)
_AMENDMENT_BASE_KEYS = (
    "threshold_set_id",
    "manifest_path",
    "manifest_sha256",
    "freeze_commit",
)
_AMENDMENT_SCOPE_KEYS = (
    "phase",
    "artifact_kind",
    "profile",
    "benchmark_path",
    "metric",
    "threshold_path",
    "lanes",
    "references",
    "aggregation",
    "comparison",
)
_AMENDMENT_CHANGE_KEYS = (
    "previous_value",
    "amended_value",
    "owner_decision_date",
    "rationale",
)
_AMENDMENT_GOVERNANCE_KEYS = (
    "adr_path",
    "child_issue_url",
    "base_manifest_immutable",
    "all_unlisted_thresholds_inherit_base",
    "candidate_commit_must_descend_from_amendment_freeze",
    "candidate_and_amendment_same_commit_forbidden",
    "amendment_and_candidate_same_pr_forbidden",
    "threshold_only_pr",
    "no_protocol_change",
)
_AMENDMENT_RECEIPT_KEYS = (
    "schema_version",
    "issue",
    "amendment_id",
    "manifest_path",
    "manifest_sha256",
    "manifest_git_blob",
    "freeze_commit",
    "base_threshold_set_id",
    "base_manifest_sha256",
    "base_freeze_commit",
    "issue_url",
    "adr_path",
    "change_policy",
    "creation_verification",
    "shallow_checkout_policy",
    "final_merge_method",
)


def _load_yaml_mapping(path: Path) -> Mapping[str, Any]:
    try:
        raw = OmegaConf.to_container(OmegaConf.load(path), resolve=False)
    except Exception as exc:  # noqa: BLE001 - normalize parser errors
        raise ThresholdValidationError(
            path, [f"cannot load YAML: {type(exc).__name__}: {exc}"]
        ) from exc
    if not isinstance(raw, dict):
        raise ThresholdValidationError(path, ["root: expected mapping"])
    return cast(Mapping[str, Any], raw)


def _mapping(
    value: Any,
    path: str,
    expected_keys: Sequence[str],
    errors: list[str],
) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        errors.append(f"{path}: expected mapping")
        return {}
    expected = set(expected_keys)
    actual = set(value)
    for key in sorted(expected - actual):
        errors.append(f"{path}: missing key `{key}`")
    for key in sorted(actual - expected, key=str):
        errors.append(f"{path}: unknown key `{key}`")
    return cast(Mapping[str, Any], value)


def _string(value: Any, path: str, errors: list[str]) -> str:
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{path}: expected non-empty string")
        return ""
    if "${" in value:
        errors.append(f"{path}: interpolation is not allowed")
    return value


def _number(value: Any, path: str, errors: list[str], *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        errors.append(f"{path}: expected number")
        return 0.0
    result = float(value)
    if not math.isfinite(result):
        errors.append(f"{path}: must be finite")
    if result < minimum:
        errors.append(f"{path}: must be >= {minimum}")
    return result


def _integer(value: Any, path: str, errors: list[str], *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        errors.append(f"{path}: expected integer")
        return 0
    if value < minimum:
        errors.append(f"{path}: must be >= {minimum}")
    return int(value)


def _boolean(value: Any, path: str, errors: list[str]) -> bool:
    if not isinstance(value, bool):
        errors.append(f"{path}: expected boolean")
        return False
    return value


def _list(value: Any, path: str, errors: list[str]) -> list[Any]:
    if not isinstance(value, list) or not value:
        errors.append(f"{path}: expected non-empty list")
        return []
    if len({json.dumps(item, sort_keys=True) for item in value}) != len(value):
        errors.append(f"{path}: duplicate values are not allowed")
    return value


def _validate_schema(raw: Mapping[str, Any], source: Path) -> list[str]:
    errors: list[str] = []
    root = _mapping(raw, "manifest", _ROOT_KEYS, errors)
    baseline = _mapping(root.get("baseline"), "baseline", _BASELINE_KEYS, errors)
    _mapping(baseline.get("hardware"), "baseline.hardware", _BASELINE_HARDWARE_KEYS, errors)
    measurement = _mapping(root.get("measurement"), "measurement", _MEASUREMENT_KEYS, errors)
    gates = _mapping(root.get("gates"), "gates", _GATE_KEYS, errors)
    performance = _mapping(gates.get("performance"), "gates.performance", _PERFORMANCE_KEYS, errors)
    training = _mapping(gates.get("training"), "gates.training", _TRAINING_KEYS, errors)
    dr = _mapping(gates.get("dr"), "gates.dr", _DR_KEYS, errors)
    memory = _mapping(gates.get("memory"), "gates.memory", _MEMORY_KEYS, errors)
    transfer = _mapping(gates.get("transfer"), "gates.transfer", _TRANSFER_KEYS, errors)
    compatibility = _mapping(
        gates.get("compatibility"),
        "gates.compatibility",
        _COMPATIBILITY_KEYS,
        errors,
    )
    governance = _mapping(root.get("governance"), "governance", _GOVERNANCE_KEYS, errors)

    _integer(root.get("schema_version"), "schema_version", errors, minimum=1)
    _integer(root.get("issue"), "issue", errors, minimum=1)
    _string(root.get("threshold_set_id"), "threshold_set_id", errors)
    _string(root.get("state"), "state", errors)
    for key in _BASELINE_KEYS[:-1]:
        _string(baseline.get(key), f"baseline.{key}", errors)
    hardware = cast(Mapping[str, Any], baseline.get("hardware", {}))
    for key in _BASELINE_HARDWARE_KEYS:
        if key == "gpu_memory_mib":
            _integer(hardware.get(key), f"baseline.hardware.{key}", errors, minimum=1)
        else:
            _string(hardware.get(key), f"baseline.hardware.{key}", errors)

    _string(measurement.get("dtype"), "measurement.dtype", errors)
    for key in ("batch_sizes", "ppo_seeds"):
        for index, item in enumerate(_list(measurement.get(key), f"measurement.{key}", errors)):
            _integer(item, f"measurement.{key}[{index}]", errors, minimum=0)
    for key in ("dr_modes",):
        for index, item in enumerate(_list(measurement.get(key), f"measurement.{key}", errors)):
            _string(item, f"measurement.{key}[{index}]", errors)
    for index, item in enumerate(
        _list(measurement.get("reset_densities"), "measurement.reset_densities", errors)
    ):
        _number(item, f"measurement.reset_densities[{index}]", errors)
    for key in (
        "env_process_repeats",
        "ppo_iterations",
        "ppo_warmup_iterations",
        "dr_num_envs",
        "dr_process_repeats",
    ):
        _integer(measurement.get(key), f"measurement.{key}", errors, minimum=1)
    _boolean(measurement.get("prohibit_filtering"), "measurement.prohibit_filtering", errors)

    _string(performance.get("env_metric"), "gates.performance.env_metric", errors)
    _string(
        performance.get("process_aggregation"),
        "gates.performance.process_aggregation",
        errors,
    )
    for key in (
        "p50_latency_ratio_max",
        "p95_latency_ratio_max",
        "throughput_ratio_min",
    ):
        _number(performance.get(key), f"gates.performance.{key}", errors)
    cv = performance.get("max_population_cv_by_batch")
    if not isinstance(cv, dict):
        errors.append("gates.performance.max_population_cv_by_batch: expected mapping")
    else:
        for key, value in cv.items():
            _string(key, "gates.performance.max_population_cv_by_batch.<key>", errors)
            _number(
                value,
                f"gates.performance.max_population_cv_by_batch.{key}",
                errors,
            )
    _boolean(
        performance.get("require_all_batches"),
        "gates.performance.require_all_batches",
        errors,
    )

    for key in _TRAINING_KEYS[:-2]:
        if key in {"max_failed_seeds", "max_nan_seeds"}:
            _integer(training.get(key), f"gates.training.{key}", errors)
        else:
            _number(training.get(key), f"gates.training.{key}", errors)
    for key in _TRAINING_KEYS[-2:]:
        _boolean(training.get(key), f"gates.training.{key}", errors)
    for key in _DR_KEYS[:3]:
        if key.endswith("bytes_max"):
            _integer(dr.get(key), f"gates.dr.{key}", errors)
        else:
            _number(dr.get(key), f"gates.dr.{key}", errors)
    for key in _DR_KEYS[3:]:
        _boolean(dr.get(key), f"gates.dr.{key}", errors)
    for key in _MEMORY_KEYS:
        if key.endswith("bytes_max"):
            _integer(memory.get(key), f"gates.memory.{key}", errors)
        else:
            _number(memory.get(key), f"gates.memory.{key}", errors)
    for key in _TRANSFER_KEYS[:3]:
        _number(transfer.get(key), f"gates.transfer.{key}", errors)
    for key in _TRANSFER_KEYS[3:]:
        _boolean(transfer.get(key), f"gates.transfer.{key}", errors)
    for key in _COMPATIBILITY_KEYS[1:12]:
        if key == "physics_trajectory_steps":
            _integer(compatibility.get(key), f"gates.compatibility.{key}", errors, minimum=1)
        else:
            _number(compatibility.get(key), f"gates.compatibility.{key}", errors)
    for key in _COMPATIBILITY_KEYS[12:]:
        _boolean(compatibility.get(key), f"gates.compatibility.{key}", errors)
    for index, item in enumerate(
        _list(
            compatibility.get("exact_match_fields"),
            "gates.compatibility.exact_match_fields",
            errors,
        )
    ):
        _string(item, f"gates.compatibility.exact_match_fields[{index}]", errors)

    _mapping(root.get("baseline_reference"), "baseline_reference", ("env", "dr", "ppo"), errors)
    for key in (
        "candidate_artifact_required_fields",
        "threshold_change_requires",
    ):
        for index, item in enumerate(_list(governance.get(key), f"governance.{key}", errors)):
            _string(item, f"governance.{key}[{index}]", errors)
    for key in (
        "candidate_commit_must_descend_from_freeze",
        "candidate_and_threshold_same_commit_forbidden",
        "candidate_result_and_threshold_change_same_pr_forbidden",
    ):
        _boolean(governance.get(key), f"governance.{key}", errors)
    _string(governance.get("failure_semantics"), "governance.failure_semantics", errors)
    return errors


def _policy_errors(raw: Mapping[str, Any]) -> list[str]:
    expected: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "issue": ISSUE,
        "threshold_set_id": THRESHOLD_SET_ID,
        "state": "frozen",
        "measurement.dtype": "float32",
        "measurement.batch_sizes": [128, 1024, 4096],
        "measurement.env_process_repeats": 5,
        "measurement.ppo_seeds": [0, 1, 2, 3, 4],
        "measurement.ppo_iterations": 100,
        "measurement.ppo_warmup_iterations": 10,
        "measurement.dr_num_envs": 1024,
        "measurement.dr_modes": ["disabled", "default_kp_kd"],
        "measurement.reset_densities": [0.01, 0.1, 1.0],
        "measurement.dr_process_repeats": 5,
        "measurement.prohibit_filtering": True,
        "gates.performance.env_metric": "env_step_total_ms",
        "gates.performance.process_aggregation": "median_of_process_summaries",
        "gates.performance.p50_latency_ratio_max": 1.05,
        "gates.performance.p95_latency_ratio_max": 1.05,
        "gates.performance.throughput_ratio_min": 1.0 / 1.05,
        "gates.performance.max_population_cv_by_batch": {
            "128": 0.15,
            "1024": 0.03,
            "4096": 0.03,
        },
        "gates.performance.require_all_batches": True,
        "gates.training.fps_p50_median_ratio_min": 1.0 / 1.05,
        "gates.training.reward_auc_median_drop_max": 5.0,
        "gates.training.final_reward_p50_median_drop_max": 0.1,
        "gates.training.episode_length_median_ratio_min": 0.9,
        "gates.training.max_failed_seeds": 0,
        "gates.training.max_nan_seeds": 0,
        "gates.training.require_all_seeds": True,
        "gates.training.performance_and_behavior_must_both_pass": True,
        "gates.dr.enabled_to_disabled_p50_ratio_max": 1.25,
        "gates.dr.enabled_to_disabled_p95_ratio_max": 1.25,
        "gates.dr.enabled_extra_resident_bytes_max": 512 * 1024 * 1024,
        "gates.dr.require_all_modes_and_densities": True,
        "gates.dr.actual_rows_must_match": True,
        "gates.memory.host_preferred_metric_ratio_max": 1.25,
        "gates.memory.device_peak_reserved_capacity_ratio_max": 0.8,
        "gates.memory.device_peak_reserved_growth_bytes_max": 8 * 1024**3,
        "gates.transfer.h2d_per_policy_step_max": 0.0,
        "gates.transfer.d2h_per_policy_step_max": 0.0,
        "gates.transfer.host_global_sync_per_policy_step_max": 0.0,
        "gates.transfer.metrics_materialization_must_be_separate": True,
        "gates.transfer.profiler_reconciliation_required": True,
        "gates.compatibility.physics_trajectory_steps": 32,
        "gates.compatibility.exact_match_fields": [
            "task",
            "backend_identity",
            "owner_yaml_sha256",
            "policy_abi",
            "dtype",
            "plan_fingerprint",
            "batch_size",
            "seed",
            "reset_schedule",
        ],
        "gates.compatibility.manager_obs_atol": 1e-6,
        "gates.compatibility.manager_obs_rtol": 1e-6,
        "gates.compatibility.manager_reward_atol": 1e-6,
        "gates.compatibility.manager_reward_rtol": 1e-6,
        "gates.compatibility.physics_one_step_atol": 1e-5,
        "gates.compatibility.physics_one_step_rtol": 1e-4,
        "gates.compatibility.physics_trajectory_qpos_atol": 2e-3,
        "gates.compatibility.physics_trajectory_qpos_rtol": 5e-3,
        "gates.compatibility.physics_trajectory_qvel_atol": 1e-2,
        "gates.compatibility.physics_trajectory_qvel_rtol": 1e-2,
        "gates.compatibility.lifecycle_exact": True,
        "gates.compatibility.unsupported_fail_closed": True,
        "gates.compatibility.fallback_forbidden": True,
        "gates.compatibility.mandatory_skip_xfail_forbidden": True,
        "governance.candidate_commit_must_descend_from_freeze": True,
        "governance.candidate_and_threshold_same_commit_forbidden": True,
        "governance.candidate_result_and_threshold_change_same_pr_forbidden": True,
        "governance.candidate_artifact_required_fields": [
            "threshold_set_id",
            "threshold_manifest_sha256",
            "threshold_freeze_commit",
            "candidate_commit",
            "source_dirty",
            "profile",
            "raw_samples",
        ],
        "governance.threshold_change_requires": [
            "independent_adr",
            "child_issue",
            "threshold_only_pr",
            "fresh_baseline_if_invalidated",
            "before_candidate_measurement",
        ],
    }
    errors: list[str] = []
    for path, wanted in expected.items():
        actual = _get_path(raw, path)
        if isinstance(wanted, float) and isinstance(actual, (int, float)):
            if math.isclose(float(actual), wanted, rel_tol=0.0, abs_tol=1e-12):
                continue
        elif actual == wanted:
            continue
        errors.append(f"{path}: frozen value is {wanted!r}, got {actual!r}")
    return errors


def _get_path(data: Mapping[str, Any], path: str) -> Any:
    value: Any = data
    for part in path.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def _population_cv(values: Sequence[float]) -> float:
    mean = statistics.fmean(values)
    if mean == 0.0:
        raise ValueError("population CV requires a non-zero mean")
    return statistics.pstdev(values) / abs(mean)


def derive_baseline_reference(artifact: Mapping[str, Any], plan: G1BaselinePlan) -> dict[str, Any]:
    env_cases = cast(Mapping[str, Mapping[str, Any]], artifact["aggregates"]["env"]["cases"])
    env: dict[str, Any] = {}
    for batch in plan.env_lane.batch_sizes:
        cases = [value for key, value in env_cases.items() if key.startswith(f"env-b{batch}-")]
        throughputs = [float(case["throughput_env_steps_per_sec"]) for case in cases]
        env[str(batch)] = {
            "process_count": len(cases),
            "throughput_median_env_steps_per_sec": statistics.median(throughputs),
            "throughput_population_cv": _population_cv(throughputs),
            "step_p50_median_ms": statistics.median(
                float(case["timing_stats_ms"]["env_step_total_ms"]["p50"]) for case in cases
            ),
            "step_p95_median_ms": statistics.median(
                float(case["timing_stats_ms"]["env_step_total_ms"]["p95"]) for case in cases
            ),
            "host_uss_delta_median_bytes": int(
                statistics.median(int(case["memory"]["total_uss_delta_bytes"]) for case in cases)
            ),
        }

    dr_cases = cast(Mapping[str, Mapping[str, Any]], artifact["aggregates"]["dr"]["cases"])
    dr: dict[str, Any] = {}
    for density in plan.dr_lane.reset_densities:
        density_key = f"{density:.4f}".replace(".", "p")
        by_mode: dict[str, Any] = {}
        for mode in plan.dr_lane.modes:
            cases = [
                value
                for key, value in dr_cases.items()
                if key.startswith(f"dr-{mode}-d{density_key}-")
            ]
            by_mode[mode] = {
                "process_count": len(cases),
                "actual_rows": int(
                    statistics.median(float(case["row_count"]["p50"]) for case in cases)
                ),
                "total_p50_median_ms": statistics.median(
                    float(case["timing_stats_ms"]["dr_reset_total_ms"]["p50"]) for case in cases
                ),
                "total_p95_median_ms": statistics.median(
                    float(case["timing_stats_ms"]["dr_reset_total_ms"]["p95"]) for case in cases
                ),
                "host_uss_delta_median_bytes": int(
                    statistics.median(
                        int(case["memory"]["total_uss_delta_bytes"]) for case in cases
                    )
                ),
            }
        dr[str(density)] = by_mode

    ppo_cases = cast(Mapping[str, Mapping[str, Any]], artifact["aggregates"]["ppo"]["cases"])
    cases = list(ppo_cases.values())
    ppo = {
        "case_count": len(cases),
        "seeds": list(plan.ppo_lane.seeds),
        "fps_p50_median": statistics.median(
            float(case["scalar_stats"]["Perf/total_fps"]["p50"]) for case in cases
        ),
        "reward_auc_median": statistics.median(float(case["reward_auc"]) for case in cases),
        "final_reward_p50_median": statistics.median(
            float(case["scalar_stats"]["Train/mean_reward"]["p50"]) for case in cases
        ),
        "episode_length_p50_median": statistics.median(
            float(case["scalar_stats"]["Train/mean_episode_length"]["p50"]) for case in cases
        ),
        "peak_rss_median_bytes": int(
            statistics.median(int(case["peak_rss_bytes"]) for case in cases)
        ),
        "peak_gpu_reserved_median_bytes": int(
            statistics.median(int(case["peak_gpu_memory_reserved_bytes"]) for case in cases)
        ),
    }
    return {"env": env, "dr": dr, "ppo": ppo}


def _nested_mismatch_errors(actual: Any, expected: Any, path: str) -> list[str]:
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return [f"{path}: expected mapping"]
        errors: list[str] = []
        if set(actual) != set(expected):
            errors.append(f"{path}: keys differ; expected {sorted(expected)}, got {sorted(actual)}")
        for key in sorted(set(actual) & set(expected)):
            errors.extend(_nested_mismatch_errors(actual[key], expected[key], f"{path}.{key}"))
        return errors
    if isinstance(expected, float) and isinstance(actual, (int, float)):
        if math.isclose(float(actual), expected, rel_tol=1e-12, abs_tol=1e-12):
            return []
    elif actual == expected:
        return []
    return [f"{path}: baseline reference mismatch; expected {expected!r}, got {actual!r}"]


def load_threshold_manifest(path: Path, *, repo_root: Path) -> ThresholdManifest:
    raw = _load_yaml_mapping(path)
    errors = _validate_schema(raw, path)
    errors.extend(_policy_errors(raw))
    baseline = cast(Mapping[str, Any], raw.get("baseline", {}))
    expected_binding = {
        "baseline_id": "g1-mujoco-phase0-v1",
        "plan_path": BASELINE_PLAN_PATH.as_posix(),
        "artifact_path": BASELINE_ARTIFACT_PATH.as_posix(),
        "source_commit": "aa0a8e723e73e18d8b1b850eef7adfb442ef1bbb",
        "task": "g1_walk_flat",
        "backend": "mujoco",
        "owner_yaml": "conf/ppo/task/g1_walk_flat/mujoco.yaml",
    }
    for key, expected in expected_binding.items():
        if baseline.get(key) != expected:
            errors.append(f"baseline.{key}: expected {expected!r}, got {baseline.get(key)!r}")

    plan_path = repo_root / BASELINE_PLAN_PATH
    artifact_path = repo_root / BASELINE_ARTIFACT_PATH
    try:
        plan = replace(
            load_g1_baseline_plan(plan_path),
            source_path=BASELINE_PLAN_PATH,
        )
        artifact = load_g1_baseline_artifact(artifact_path, plan, repo_root=repo_root)
        plan_sha = sha256_file(plan_path)
        artifact_sha = sha256_file(artifact_path)
        if baseline.get("plan_sha256") != plan_sha:
            errors.append(
                f"baseline.plan_sha256: expected current {plan_sha}, got {baseline.get('plan_sha256')!r}"
            )
        if baseline.get("artifact_sha256") != artifact_sha:
            errors.append(
                "baseline.artifact_sha256: expected current "
                f"{artifact_sha}, got {baseline.get('artifact_sha256')!r}"
            )
        expected_hardware = {
            "cpu_model": plan.hardware.cpu_model,
            "gpu_name": plan.hardware.gpu_name,
            "gpu_uuid": plan.hardware.gpu_uuid,
            "gpu_memory_mib": plan.hardware.gpu_memory_mib,
            "driver_version": plan.hardware.driver_version,
        }
        if baseline.get("hardware") != expected_hardware:
            warnings.warn(
                "baseline.hardware provenance differs from the frozen baseline plan "
                f"(advisory): expected={expected_hardware!r}, "
                f"recorded={baseline.get('hardware')!r}",
                UserWarning,
                stacklevel=2,
            )
        reference = derive_baseline_reference(artifact, plan)
        errors.extend(
            _nested_mismatch_errors(raw.get("baseline_reference"), reference, "baseline_reference")
        )
    except Exception as exc:  # noqa: BLE001 - aggregate linked evidence errors
        errors.append(f"baseline binding: {type(exc).__name__}: {exc}")
    if errors:
        raise ThresholdValidationError(path, errors)
    return ThresholdManifest(source_path=path, data=raw)


def _validate_amendment_schema(raw: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    root = _mapping(raw, "amendment", _AMENDMENT_ROOT_KEYS, errors)
    base = _mapping(
        root.get("base_threshold"),
        "base_threshold",
        _AMENDMENT_BASE_KEYS,
        errors,
    )
    scope = _mapping(root.get("scope"), "scope", _AMENDMENT_SCOPE_KEYS, errors)
    references = _mapping(
        scope.get("references"),
        "scope.references",
        ("throughput", "behavior"),
        errors,
    )
    change = _mapping(root.get("change"), "change", _AMENDMENT_CHANGE_KEYS, errors)
    governance = _mapping(
        root.get("governance"),
        "governance",
        _AMENDMENT_GOVERNANCE_KEYS,
        errors,
    )

    _integer(root.get("schema_version"), "schema_version", errors, minimum=1)
    _integer(root.get("issue"), "issue", errors, minimum=1)
    for key in ("amendment_id", "state"):
        _string(root.get(key), key, errors)
    for key in _AMENDMENT_BASE_KEYS:
        _string(base.get(key), f"base_threshold.{key}", errors)
    _integer(scope.get("phase"), "scope.phase", errors, minimum=1)
    for key in (
        "artifact_kind",
        "profile",
        "benchmark_path",
        "metric",
        "threshold_path",
        "aggregation",
        "comparison",
    ):
        _string(scope.get(key), f"scope.{key}", errors)
    for index, lane in enumerate(_list(scope.get("lanes"), "scope.lanes", errors)):
        _string(lane, f"scope.lanes[{index}]", errors)
    for key in ("throughput", "behavior"):
        _string(references.get(key), f"scope.references.{key}", errors)
    for key in ("previous_value", "amended_value"):
        _number(change.get(key), f"change.{key}", errors)
    for key in ("owner_decision_date", "rationale"):
        _string(change.get(key), f"change.{key}", errors)
    for key in ("adr_path", "child_issue_url"):
        _string(governance.get(key), f"governance.{key}", errors)
    for key in _AMENDMENT_GOVERNANCE_KEYS[2:]:
        _boolean(governance.get(key), f"governance.{key}", errors)
    return errors


def _amendment_policy_errors(raw: Mapping[str, Any]) -> list[str]:
    expected: dict[str, Any] = {
        "schema_version": AMENDMENT_SCHEMA_VERSION,
        "issue": AMENDMENT_ISSUE,
        "amendment_id": AMENDMENT_ID,
        "state": "frozen",
        "base_threshold.threshold_set_id": THRESHOLD_SET_ID,
        "base_threshold.manifest_path": THRESHOLD_MANIFEST_PATH.as_posix(),
        "scope.phase": 5,
        "scope.artifact_kind": "manager_mjwarp-mjwarp-device-ppo-benchmark-v1",
        "scope.profile": "device_resident",
        "scope.benchmark_path": "benchmark/rl/benchmark_mjwarp_ppo.py",
        "scope.metric": "process_tree_peak_rss_median_bytes_ratio",
        "scope.threshold_path": "gates.memory.host_preferred_metric_ratio_max",
        "scope.lanes": ["throughput", "behavior"],
        "scope.references": {
            "throughput": "paired_mujoco_host_same_batch",
            "behavior": "phase0_mujoco_ppo",
        },
        "scope.aggregation": "median_of_independent_process_peaks",
        "scope.comparison": "candidate_over_reference_lte",
        "change.previous_value": 1.25,
        "change.amended_value": 1.26,
        "change.owner_decision_date": "2026-07-30",
        "governance.adr_path": AMENDMENT_ADR_PATH.as_posix(),
        "governance.child_issue_url": "https://github.com/unilabsim/UniLab/issues/807",
        "governance.base_manifest_immutable": True,
        "governance.all_unlisted_thresholds_inherit_base": True,
        "governance.candidate_commit_must_descend_from_amendment_freeze": True,
        "governance.candidate_and_amendment_same_commit_forbidden": True,
        "governance.amendment_and_candidate_same_pr_forbidden": True,
        "governance.threshold_only_pr": True,
        "governance.no_protocol_change": True,
    }
    errors: list[str] = []
    for path, wanted in expected.items():
        actual = _get_path(raw, path)
        if isinstance(wanted, float) and isinstance(actual, (int, float)):
            if math.isclose(float(actual), wanted, rel_tol=0.0, abs_tol=1e-12):
                continue
        elif actual == wanted:
            continue
        errors.append(f"{path}: frozen amendment value is {wanted!r}, got {actual!r}")
    return errors


def load_threshold_amendment(
    path: Path,
    *,
    base_manifest: ThresholdManifest,
    base_receipt: FreezeReceipt,
    repo_root: Path,
) -> ThresholdAmendment:
    """Load the narrowly scoped Phase-5 PPO RSS threshold amendment."""

    raw = _load_yaml_mapping(path)
    errors = _validate_amendment_schema(raw)
    errors.extend(_amendment_policy_errors(raw))
    base = cast(Mapping[str, Any], raw.get("base_threshold", {}))
    expected_base = {
        "threshold_set_id": base_manifest.data["threshold_set_id"],
        "manifest_path": THRESHOLD_MANIFEST_PATH.as_posix(),
        "manifest_sha256": sha256_file(base_manifest.source_path),
        "freeze_commit": base_receipt.freeze_commit,
    }
    for key, expected in expected_base.items():
        if base.get(key) != expected:
            errors.append(
                f"base_threshold.{key}: expected frozen base {expected!r}, got {base.get(key)!r}"
            )
    if base_receipt.data.get("manifest_sha256") != expected_base["manifest_sha256"]:
        errors.append("base_threshold: base receipt does not bind the current base manifest")
    base_memory = cast(Mapping[str, Any], base_manifest.gates["memory"])
    previous = cast(Mapping[str, Any], raw.get("change", {})).get("previous_value")
    if previous != base_memory.get("host_preferred_metric_ratio_max"):
        errors.append("change.previous_value: does not match the immutable base threshold")
    if not (repo_root / AMENDMENT_ADR_PATH).is_file():
        errors.append(f"governance.adr_path: file does not exist: {AMENDMENT_ADR_PATH}")
    if errors:
        raise ThresholdValidationError(path, errors)
    return ThresholdAmendment(source_path=path, data=raw)


def load_amendment_freeze_receipt(
    path: Path,
    *,
    amendment: ThresholdAmendment,
    base_receipt: FreezeReceipt,
    repo_root: Path,
    verify_git: bool = True,
) -> AmendmentFreezeReceipt:
    """Validate the amendment's own commit/hash/blob receipt."""

    raw = _load_yaml_mapping(path)
    errors: list[str] = []
    receipt = _mapping(raw, "receipt", _AMENDMENT_RECEIPT_KEYS, errors)
    for key in _AMENDMENT_RECEIPT_KEYS:
        if key in {"schema_version", "issue"}:
            _integer(receipt.get(key), key, errors, minimum=1)
        else:
            _string(receipt.get(key), key, errors)
    expected_values = {
        "schema_version": AMENDMENT_SCHEMA_VERSION,
        "issue": AMENDMENT_ISSUE,
        "amendment_id": AMENDMENT_ID,
        "manifest_path": AMENDMENT_MANIFEST_PATH.as_posix(),
        "base_threshold_set_id": THRESHOLD_SET_ID,
        "base_manifest_sha256": base_receipt.data["manifest_sha256"],
        "base_freeze_commit": base_receipt.freeze_commit,
        "issue_url": "https://github.com/unilabsim/UniLab/issues/807",
        "adr_path": AMENDMENT_ADR_PATH.as_posix(),
        "change_policy": "phase5_ppo_only_base_thresholds_immutable",
        "creation_verification": "full_git_history",
        "shallow_checkout_policy": "current_hash_and_receipt",
        "final_merge_method": "merge_commit",
    }
    for key, expected in expected_values.items():
        if receipt.get(key) != expected:
            errors.append(f"{key}: expected {expected!r}, got {receipt.get(key)!r}")
    manifest_sha = sha256_file(amendment.source_path)
    if receipt.get("manifest_sha256") != manifest_sha:
        errors.append(
            f"manifest_sha256: expected current {manifest_sha}, got {receipt.get('manifest_sha256')!r}"
        )
    for key in ("manifest_sha256", "base_manifest_sha256"):
        if not _SHA256_RE.fullmatch(str(receipt.get(key, ""))):
            errors.append(f"{key}: expected sha256:<64 lowercase hex>")
    freeze_commit = str(receipt.get("freeze_commit", ""))
    if not _COMMIT_RE.fullmatch(freeze_commit):
        errors.append("freeze_commit: expected full lowercase commit SHA")
    manifest_blob = str(receipt.get("manifest_git_blob", ""))
    if not _GIT_BLOB_RE.fullmatch(manifest_blob):
        errors.append("manifest_git_blob: expected full Git object ID")
    git_history_verified = False
    if verify_git and _COMMIT_RE.fullmatch(freeze_commit):
        git_errors, git_history_verified = _git_receipt_errors(
            repo_root=repo_root,
            freeze_commit=freeze_commit,
            manifest_path=AMENDMENT_MANIFEST_PATH,
            manifest_bytes=amendment.source_path.read_bytes(),
            expected_blob=manifest_blob,
        )
        errors.extend(git_errors)
        if git_history_verified and base_receipt.git_history_verified:
            if freeze_commit == base_receipt.freeze_commit:
                errors.append("freeze_commit: amendment must be after the base threshold freeze")
            else:
                try:
                    base_ancestor = _git(
                        repo_root,
                        [
                            "merge-base",
                            "--is-ancestor",
                            base_receipt.freeze_commit,
                            freeze_commit,
                        ],
                        check=False,
                    )
                    if base_ancestor.returncode != 0:
                        errors.append(
                            "freeze_commit: amendment must descend from the base threshold freeze"
                        )
                except OSError as exc:
                    errors.append(f"freeze_commit: cannot verify base ancestry: {exc}")
    if errors:
        raise ThresholdValidationError(path, errors)
    return AmendmentFreezeReceipt(
        source_path=path,
        data=raw,
        git_history_verified=git_history_verified,
    )


_RECEIPT_KEYS = (
    "schema_version",
    "issue",
    "threshold_set_id",
    "manifest_path",
    "manifest_sha256",
    "manifest_git_blob",
    "freeze_commit",
    "baseline_artifact_sha256",
    "issue_url",
    "change_policy",
    "creation_verification",
    "shallow_checkout_policy",
    "final_merge_method",
)


def load_freeze_receipt(
    path: Path,
    *,
    manifest: ThresholdManifest,
    repo_root: Path,
    verify_git: bool = True,
) -> FreezeReceipt:
    raw = _load_yaml_mapping(path)
    errors: list[str] = []
    receipt = _mapping(raw, "receipt", _RECEIPT_KEYS, errors)
    for key in _RECEIPT_KEYS:
        if key in {"schema_version", "issue"}:
            _integer(receipt.get(key), key, errors, minimum=1)
        else:
            _string(receipt.get(key), key, errors)
    if receipt.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version: expected {SCHEMA_VERSION}")
    if receipt.get("issue") != ISSUE:
        errors.append(f"issue: expected {ISSUE}")
    if receipt.get("threshold_set_id") != THRESHOLD_SET_ID:
        errors.append(f"threshold_set_id: expected {THRESHOLD_SET_ID!r}")
    if receipt.get("manifest_path") != THRESHOLD_MANIFEST_PATH.as_posix():
        errors.append("manifest_path: must point to the frozen threshold manifest")
    expected_history_policy = {
        "creation_verification": "full_git_history",
        "shallow_checkout_policy": "current_hash_and_receipt",
        "final_merge_method": "merge_commit",
    }
    for key, expected in expected_history_policy.items():
        if receipt.get(key) != expected:
            errors.append(f"{key}: expected {expected!r}")
    manifest_sha = sha256_file(manifest.source_path)
    if receipt.get("manifest_sha256") != manifest_sha:
        errors.append(
            f"manifest_sha256: expected current {manifest_sha}, got {receipt.get('manifest_sha256')!r}"
        )
    baseline_sha = str(manifest.data["baseline"]["artifact_sha256"])
    if receipt.get("baseline_artifact_sha256") != baseline_sha:
        errors.append("baseline_artifact_sha256: does not match threshold manifest")
    freeze_commit = str(receipt.get("freeze_commit", ""))
    if not _COMMIT_RE.fullmatch(freeze_commit):
        errors.append("freeze_commit: expected full lowercase commit SHA")
    manifest_blob = str(receipt.get("manifest_git_blob", ""))
    if not _GIT_BLOB_RE.fullmatch(manifest_blob):
        errors.append("manifest_git_blob: expected full Git object ID")
    for key in ("manifest_sha256", "baseline_artifact_sha256"):
        if not _SHA256_RE.fullmatch(str(receipt.get(key, ""))):
            errors.append(f"{key}: expected sha256:<64 lowercase hex>")
    git_history_verified = False
    if verify_git and _COMMIT_RE.fullmatch(freeze_commit):
        git_errors, git_history_verified = _git_receipt_errors(
            repo_root=repo_root,
            freeze_commit=freeze_commit,
            manifest_path=THRESHOLD_MANIFEST_PATH,
            manifest_bytes=manifest.source_path.read_bytes(),
            expected_blob=manifest_blob,
        )
        errors.extend(git_errors)
    if errors:
        raise ThresholdValidationError(path, errors)
    return FreezeReceipt(
        source_path=path,
        data=raw,
        git_history_verified=git_history_verified,
    )


def _git(
    repo_root: Path, args: Sequence[str], *, check: bool = True
) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=check,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _git_receipt_errors(
    *,
    repo_root: Path,
    freeze_commit: str,
    manifest_path: Path,
    manifest_bytes: bytes,
    expected_blob: str,
) -> tuple[list[str], bool]:
    errors: list[str] = []
    object_spec = f"{freeze_commit}:{manifest_path.as_posix()}"
    try:
        commit_exists = _git(
            repo_root,
            ["cat-file", "-e", f"{freeze_commit}^{{commit}}"],
            check=False,
        )
    except OSError as exc:
        return [f"freeze_commit: cannot invoke Git: {exc}"], False
    if commit_exists.returncode != 0:
        try:
            shallow = _git(
                repo_root,
                ["rev-parse", "--is-shallow-repository"],
                check=False,
            )
        except OSError as exc:
            return [f"freeze_commit: cannot determine repository depth: {exc}"], False
        if shallow.returncode == 0 and shallow.stdout.decode().strip() == "true":
            return [], False
        return ["freeze_commit: cannot verify Git object in a full-history checkout"], False
    try:
        actual_blob = _git(repo_root, ["rev-parse", object_spec]).stdout.decode().strip()
        if actual_blob != expected_blob:
            errors.append(
                f"manifest_git_blob: commit contains {actual_blob}, receipt declares {expected_blob}"
            )
        committed_bytes = _git(repo_root, ["show", object_spec]).stdout
        if committed_bytes != manifest_bytes:
            errors.append("freeze_commit: manifest bytes differ from the current frozen file")
        ancestor = _git(
            repo_root,
            ["merge-base", "--is-ancestor", freeze_commit, "HEAD"],
            check=False,
        )
        if ancestor.returncode != 0:
            errors.append("freeze_commit: is not an ancestor of HEAD")
    except (OSError, subprocess.CalledProcessError) as exc:
        errors.append(f"freeze_commit: cannot verify Git object: {exc}")
    return errors, not errors


def candidate_provenance_errors(
    provenance: CandidateProvenance,
    *,
    manifest: ThresholdManifest,
    receipt: FreezeReceipt,
    is_ancestor: Callable[[str, str], bool],
) -> list[str]:
    errors: list[str] = []
    if provenance.threshold_set_id != THRESHOLD_SET_ID:
        errors.append("provenance.threshold_set_id: does not match frozen thresholds")
    expected_manifest_sha = str(receipt.data["manifest_sha256"])
    if provenance.threshold_manifest_sha256 != expected_manifest_sha:
        errors.append("provenance.threshold_manifest_sha256: does not match freeze receipt")
    if provenance.threshold_freeze_commit != receipt.freeze_commit:
        errors.append("provenance.threshold_freeze_commit: does not match freeze receipt")
    if provenance.source_dirty:
        errors.append("provenance.source_dirty: candidate measurement must use a clean tree")
    if not _COMMIT_RE.fullmatch(provenance.candidate_commit):
        errors.append("provenance.candidate_commit: expected full lowercase commit SHA")
        return errors
    if provenance.candidate_commit == receipt.freeze_commit:
        errors.append("provenance.candidate_commit: threshold and candidate cannot share a commit")
    elif not is_ancestor(receipt.freeze_commit, provenance.candidate_commit):
        errors.append("provenance.candidate_commit: must descend from threshold freeze commit")
    if manifest.data["threshold_set_id"] != provenance.threshold_set_id:
        errors.append("provenance: manifest threshold identity mismatch")
    return errors


def candidate_gate_errors(
    candidate: CandidateGateInput,
    *,
    manifest: ThresholdManifest,
    receipt: FreezeReceipt,
    is_ancestor: Callable[[str, str], bool],
) -> list[str]:
    errors = candidate_provenance_errors(
        candidate.provenance,
        manifest=manifest,
        receipt=receipt,
        is_ancestor=is_ancestor,
    )
    if candidate.profile not in {"host_fused", "device_resident"}:
        errors.append(f"profile: unsupported execution profile {candidate.profile!r}")
        return errors
    errors.extend(_raw_evidence_errors(candidate.raw_evidence))
    gates = manifest.gates
    performance = cast(Mapping[str, Any], gates["performance"])
    reference_env = cast(Mapping[str, Mapping[str, Any]], manifest.baseline_reference["env"])
    required_batches = {128, 1024, 4096}
    if set(candidate.environment) != required_batches:
        errors.append(
            f"environment: expected batches {sorted(required_batches)}, got {sorted(candidate.environment)}"
        )
    for batch in sorted(required_batches & set(candidate.environment)):
        env_metrics = candidate.environment[batch]
        reference = reference_env[str(batch)]
        prefix = f"environment[{batch}]"
        _append_min_error(errors, f"{prefix}.process_repeats", env_metrics.process_repeats, 5)
        for name in (
            "step_p50_median_ms",
            "step_p95_median_ms",
            "throughput_median_env_steps_per_sec",
            "throughput_population_cv",
            "host_memory_median_bytes",
        ):
            _append_nonnegative_error(errors, f"{prefix}.{name}", getattr(env_metrics, name))
        if env_metrics.host_memory_metric != "uss":
            errors.append(f"{prefix}.host_memory_metric: expected frozen preferred metric 'uss'")
        _append_max_error(
            errors,
            f"{prefix}.p50_latency_ratio",
            _ratio(
                env_metrics.step_p50_median_ms,
                float(reference["step_p50_median_ms"]),
            ),
            float(performance["p50_latency_ratio_max"]),
        )
        _append_max_error(
            errors,
            f"{prefix}.p95_latency_ratio",
            _ratio(
                env_metrics.step_p95_median_ms,
                float(reference["step_p95_median_ms"]),
            ),
            float(performance["p95_latency_ratio_max"]),
        )
        _append_min_error(
            errors,
            f"{prefix}.throughput_ratio",
            _ratio(
                env_metrics.throughput_median_env_steps_per_sec,
                float(reference["throughput_median_env_steps_per_sec"]),
            ),
            float(performance["throughput_ratio_min"]),
        )
        cv_max = float(performance["max_population_cv_by_batch"][str(batch)])
        _append_max_error(
            errors,
            f"{prefix}.throughput_population_cv",
            env_metrics.throughput_population_cv,
            cv_max,
        )
        host_ratio = _ratio(
            env_metrics.host_memory_median_bytes,
            float(reference["host_uss_delta_median_bytes"]),
        )
        _append_max_error(
            errors,
            f"{prefix}.host_memory_ratio",
            host_ratio,
            float(gates["memory"]["host_preferred_metric_ratio_max"]),
        )

    training = candidate.training
    training_gate = cast(Mapping[str, Any], gates["training"])
    ppo_reference = cast(Mapping[str, Any], manifest.baseline_reference["ppo"])
    if training.seeds != (0, 1, 2, 3, 4):
        errors.append(f"training.seeds: expected (0, 1, 2, 3, 4), got {training.seeds!r}")
    if training.failed_seeds:
        errors.append(f"training.failed_seeds: all seeds must pass, got {training.failed_seeds!r}")
    if training.nan_seeds:
        errors.append(f"training.nan_seeds: all metrics must be finite, got {training.nan_seeds!r}")
    for name in (
        "fps_p50_median",
        "episode_length_p50_median",
        "peak_rss_median_bytes",
        "peak_gpu_reserved_median_bytes",
    ):
        _append_nonnegative_error(errors, f"training.{name}", getattr(training, name))
    _append_min_error(
        errors,
        "training.fps_p50_median_ratio",
        _ratio(training.fps_p50_median, float(ppo_reference["fps_p50_median"])),
        float(training_gate["fps_p50_median_ratio_min"]),
    )
    _append_min_error(
        errors,
        "training.reward_auc_median",
        training.reward_auc_median,
        float(ppo_reference["reward_auc_median"])
        - float(training_gate["reward_auc_median_drop_max"]),
    )
    _append_min_error(
        errors,
        "training.final_reward_p50_median",
        training.final_reward_p50_median,
        float(ppo_reference["final_reward_p50_median"])
        - float(training_gate["final_reward_p50_median_drop_max"]),
    )
    _append_min_error(
        errors,
        "training.episode_length_p50_median_ratio",
        _ratio(
            training.episode_length_p50_median,
            float(ppo_reference["episode_length_p50_median"]),
        ),
        float(training_gate["episode_length_median_ratio_min"]),
    )
    _append_max_error(
        errors,
        "training.peak_rss_ratio",
        _ratio(
            training.peak_rss_median_bytes,
            float(ppo_reference["peak_rss_median_bytes"]),
        ),
        float(gates["memory"]["host_preferred_metric_ratio_max"]),
    )

    required_densities = {0.01, 0.1, 1.0}
    if set(candidate.dr) != required_densities:
        errors.append(
            f"dr: expected densities {sorted(required_densities)}, got {sorted(candidate.dr)}"
        )
    dr_gate = cast(Mapping[str, Any], gates["dr"])
    for density in sorted(required_densities & set(candidate.dr)):
        dr_metrics = candidate.dr[density]
        prefix = f"dr[{density}]"
        _append_min_error(errors, f"{prefix}.process_repeats", dr_metrics.process_repeats, 5)
        for name in (
            "disabled_total_p50_median_ms",
            "disabled_total_p95_median_ms",
            "enabled_total_p50_median_ms",
            "enabled_total_p95_median_ms",
            "enabled_extra_resident_bytes",
        ):
            _append_nonnegative_error(errors, f"{prefix}.{name}", getattr(dr_metrics, name))
        if dr_metrics.resident_memory_metric != "uss":
            errors.append(
                f"{prefix}.resident_memory_metric: expected frozen preferred metric 'uss'"
            )
        expected_rows = max(1, int(1024 * density))
        if dr_metrics.actual_rows != expected_rows:
            errors.append(
                f"{prefix}.actual_rows: expected {expected_rows}, got {dr_metrics.actual_rows}"
            )
        _append_max_error(
            errors,
            f"{prefix}.enabled_to_disabled_p50_ratio",
            _ratio(
                dr_metrics.enabled_total_p50_median_ms,
                dr_metrics.disabled_total_p50_median_ms,
            ),
            float(dr_gate["enabled_to_disabled_p50_ratio_max"]),
        )
        _append_max_error(
            errors,
            f"{prefix}.enabled_to_disabled_p95_ratio",
            _ratio(
                dr_metrics.enabled_total_p95_median_ms,
                dr_metrics.disabled_total_p95_median_ms,
            ),
            float(dr_gate["enabled_to_disabled_p95_ratio_max"]),
        )
        _append_max_error(
            errors,
            f"{prefix}.enabled_extra_resident_bytes",
            dr_metrics.enabled_extra_resident_bytes,
            int(dr_gate["enabled_extra_resident_bytes_max"]),
        )

    errors.extend(_compatibility_errors(candidate.compatibility, manifest))
    if candidate.profile == "device_resident":
        if candidate.device is None:
            errors.append("device: required for device_resident profile")
        else:
            errors.extend(_device_errors(candidate.device, training, manifest))
    elif candidate.device is not None:
        errors.append("device: host_fused profile must not report a device gate payload")
    return errors


def _compatibility_errors(
    metrics: CandidateCompatibilityMetrics, manifest: ThresholdManifest
) -> list[str]:
    errors: list[str] = []
    for field in (
        "identity_exact",
        "policy_abi_exact",
        "lifecycle_exact",
        "unsupported_fail_closed",
    ):
        if not getattr(metrics, field):
            errors.append(f"compatibility.{field}: expected true")
    if metrics.fallback_used:
        errors.append("compatibility.fallback_used: fallback is forbidden")
    if metrics.mandatory_cases_skipped:
        errors.append("compatibility.mandatory_cases_skipped: mandatory skip/xfail is forbidden")
    gate = cast(Mapping[str, Any], manifest.gates["compatibility"])
    comparisons = (
        ("manager_obs_max_abs", metrics.manager_obs_max_abs, "manager_obs_atol"),
        ("manager_obs_max_rel", metrics.manager_obs_max_rel, "manager_obs_rtol"),
        ("manager_reward_max_abs", metrics.manager_reward_max_abs, "manager_reward_atol"),
        ("manager_reward_max_rel", metrics.manager_reward_max_rel, "manager_reward_rtol"),
        ("physics_one_step_max_abs", metrics.physics_one_step_max_abs, "physics_one_step_atol"),
        ("physics_one_step_max_rel", metrics.physics_one_step_max_rel, "physics_one_step_rtol"),
        (
            "physics_trajectory_qpos_max_abs",
            metrics.physics_trajectory_qpos_max_abs,
            "physics_trajectory_qpos_atol",
        ),
        (
            "physics_trajectory_qpos_max_rel",
            metrics.physics_trajectory_qpos_max_rel,
            "physics_trajectory_qpos_rtol",
        ),
        (
            "physics_trajectory_qvel_max_abs",
            metrics.physics_trajectory_qvel_max_abs,
            "physics_trajectory_qvel_atol",
        ),
        (
            "physics_trajectory_qvel_max_rel",
            metrics.physics_trajectory_qvel_max_rel,
            "physics_trajectory_qvel_rtol",
        ),
    )
    for name, value, threshold_key in comparisons:
        _append_nonnegative_error(errors, f"compatibility.{name}", value)
        _append_max_error(errors, f"compatibility.{name}", value, float(gate[threshold_key]))
    return errors


def _raw_evidence_errors(evidence: CandidateRawEvidence) -> list[str]:
    errors: list[str] = []
    expected_case_count = 15 + 30 + 5
    for name in ("planned_case_ids", "observed_case_ids", "included_case_ids"):
        case_ids = getattr(evidence, name)
        if len(case_ids) != expected_case_count:
            errors.append(
                f"raw_evidence.{name}: expected {expected_case_count} cases, got {len(case_ids)}"
            )
        if len(set(case_ids)) != len(case_ids):
            errors.append(f"raw_evidence.{name}: duplicate case IDs are forbidden")
        if any(not case_id.strip() for case_id in case_ids):
            errors.append(f"raw_evidence.{name}: case IDs must be non-empty")
    planned = set(evidence.planned_case_ids)
    observed = set(evidence.observed_case_ids)
    included = set(evidence.included_case_ids)
    if observed != planned:
        errors.append("raw_evidence.observed_case_ids: must exactly match the planned matrix")
    if included != observed:
        errors.append("raw_evidence.included_case_ids: every observed case must be aggregated")
    if evidence.failed_case_ids:
        errors.append(
            f"raw_evidence.failed_case_ids: failed cases are FAIL, got {evidence.failed_case_ids!r}"
        )
    if evidence.filtered_case_ids:
        errors.append(
            "raw_evidence.filtered_case_ids: post-hoc filtering is forbidden, got "
            f"{evidence.filtered_case_ids!r}"
        )
    if not _SHA256_RE.fullmatch(evidence.raw_artifact_sha256):
        errors.append("raw_evidence.raw_artifact_sha256: expected sha256:<64 lowercase hex>")
    if not evidence.aggregate_recomputed_from_raw:
        errors.append("raw_evidence.aggregate_recomputed_from_raw: must be true")
    return errors


def _device_errors(
    metrics: CandidateDeviceMetrics,
    training: CandidateTrainingMetrics,
    manifest: ThresholdManifest,
) -> list[str]:
    errors: list[str] = []
    memory = cast(Mapping[str, Any], manifest.gates["memory"])
    transfer = cast(Mapping[str, Any], manifest.gates["transfer"])
    ppo = cast(Mapping[str, Any], manifest.baseline_reference["ppo"])
    if metrics.peak_gpu_reserved_bytes != training.peak_gpu_reserved_median_bytes:
        errors.append("device.peak_gpu_reserved_bytes: does not reconcile with PPO summary")
    for name in (
        "gpu_capacity_bytes",
        "peak_gpu_reserved_bytes",
        "h2d_per_policy_step",
        "d2h_per_policy_step",
        "host_global_sync_per_policy_step",
        "metrics_materializations",
    ):
        _append_nonnegative_error(errors, f"device.{name}", getattr(metrics, name))
    _append_max_error(
        errors,
        "device.peak_reserved_capacity_ratio",
        _ratio(metrics.peak_gpu_reserved_bytes, metrics.gpu_capacity_bytes),
        float(memory["device_peak_reserved_capacity_ratio_max"]),
    )
    _append_max_error(
        errors,
        "device.peak_reserved_growth_bytes",
        metrics.peak_gpu_reserved_bytes - int(ppo["peak_gpu_reserved_median_bytes"]),
        int(memory["device_peak_reserved_growth_bytes_max"]),
    )
    for name in (
        "h2d_per_policy_step",
        "d2h_per_policy_step",
        "host_global_sync_per_policy_step",
    ):
        _append_max_error(
            errors, f"device.{name}", getattr(metrics, name), float(transfer[f"{name}_max"])
        )
    if not metrics.profiler_reconciled:
        errors.append("device.profiler_reconciled: counter/trace reconciliation is required")
    if not metrics.profiler_trace_refs:
        errors.append("device.profiler_trace_refs: at least one raw profiler trace is required")
    elif len(set(metrics.profiler_trace_refs)) != len(metrics.profiler_trace_refs):
        errors.append("device.profiler_trace_refs: duplicate trace references are forbidden")
    elif any(not reference.strip() for reference in metrics.profiler_trace_refs):
        errors.append("device.profiler_trace_refs: references must be non-empty")
    if len(metrics.profiler_trace_sha256s) != len(metrics.profiler_trace_refs):
        errors.append("device.profiler_trace_sha256s: must map one hash to every trace")
    elif any(not _SHA256_RE.fullmatch(value) for value in metrics.profiler_trace_sha256s):
        errors.append("device.profiler_trace_sha256s: every trace requires a SHA-256 hash")
    return errors


def _append_min_error(
    errors: list[str], path: str, value: float | int, minimum: float | int
) -> None:
    finite = math.isfinite(float(value))
    at_boundary = finite and math.isclose(
        float(value), float(minimum), rel_tol=1e-12, abs_tol=1e-12
    )
    if not finite or (value < minimum and not at_boundary):
        errors.append(f"{path}: {value!r} is below frozen minimum {minimum!r}")


def _append_nonnegative_error(errors: list[str], path: str, value: float | int) -> None:
    if not math.isfinite(float(value)) or value < 0:
        errors.append(f"{path}: expected a finite non-negative value, got {value!r}")


def _ratio(numerator: float | int, denominator: float | int) -> float:
    if not math.isfinite(float(denominator)) or denominator <= 0:
        return math.inf
    return float(numerator) / float(denominator)


def _append_max_error(
    errors: list[str], path: str, value: float | int, maximum: float | int
) -> None:
    finite = math.isfinite(float(value))
    at_boundary = finite and math.isclose(
        float(value), float(maximum), rel_tol=1e-12, abs_tol=1e-12
    )
    if not finite or (value > maximum and not at_boundary):
        errors.append(f"{path}: {value!r} exceeds frozen maximum {maximum!r}")


def git_is_ancestor(repo_root: Path, ancestor: str, descendant: str) -> bool:
    result = _git(
        repo_root,
        ["merge-base", "--is-ancestor", ancestor, descendant],
        check=False,
    )
    return result.returncode == 0
