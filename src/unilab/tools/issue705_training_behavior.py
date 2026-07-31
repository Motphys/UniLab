"""Frozen paired-seed training behavior evidence for Issue #705 Phase 7."""

from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

import yaml

from unilab.tools.g1_baseline_provenance import (
    canonical_sha256,
    load_g1_baseline_plan,
    sha256_file,
    source_tree_sha256_at_commit,
)
from unilab.tools.issue705_task_rollout import (
    ROLLOUT_PLAN_PATH,
    load_task_rollout_plan,
)
from unilab.tools.issue705_thresholds import (
    load_freeze_receipt,
    load_threshold_manifest,
)

PLAN_PATH = Path("tests/acceptance/issue_705/training_behavior_plan.yaml")
FREEZE_RECEIPT_PATH = Path("tests/acceptance/issue_705/training_behavior_freeze_receipt.yaml")
ARTIFACT_PATH = Path("tests/acceptance/issue_705/artifacts/phase_7_training_behavior.json")

SCHEMA_VERSION = 1
ISSUE = 837
PARENT_ISSUE = 705
CLAIM_ID = "P7-TRAINING-BEHAVIOR"
BENCHMARK_ID = "issue705-training-behavior-v1"
ARTIFACT_KIND = "issue705-training-behavior-evidence-v1"

_COMMIT_RE = re.compile(r"[0-9a-f]{40}")
_GIT_OBJECT_RE = re.compile(r"[0-9a-f]{40,64}")
_SHA256_RE = re.compile(r"sha256:[0-9a-f]{64}")

_ROOT_KEYS = (
    "schema_version",
    "issue",
    "parent_issue",
    "claim_id",
    "benchmark_id",
    "state",
    "source_contract",
    "dependencies",
    "hardware",
    "measurement",
    "compiled_signature",
    "gates",
    "artifact_contract",
    "governance",
)
_SOURCE_KEYS = (
    "integration_branch",
    "phase_manifest",
    "baseline_plan",
    "threshold_manifest",
    "threshold_receipt",
    "baseline_artifact",
    "owner_yaml",
    "freeze_receipt",
    "candidate_artifact",
    "candidate_must_be_clean",
    "candidate_must_descend_from_freeze",
    "candidate_must_differ_from_freeze",
    "source_inputs",
)
_MEASUREMENT_KEYS = (
    "task_slug",
    "env_name",
    "baseline_backend",
    "candidate_backend",
    "execution_profile",
    "dtype",
    "seeds",
    "num_envs",
    "num_steps_per_env",
    "max_iterations",
    "warmup_iterations",
    "final_window_iterations",
    "save_interval",
    "memory_poll_interval_sec",
    "process_isolation",
    "process_retries",
    "case_filtering_forbidden",
    "required_scalar_tags",
    "hydra_overrides",
    "success_metric",
)
_SIGNATURE_KEYS = (
    "task_key",
    "executor_key",
    "task_plan_fingerprint",
    "policy_abi_fingerprint",
    "backend_plan_fingerprint",
)
_ZERO_TRAFFIC_COUNTERS = (
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


class TrainingBehaviorContractError(RuntimeError):
    """Raised when a plan, receipt, or artifact violates the frozen contract."""

    def __init__(self, path: Path, errors: Sequence[str]):
        self.path = path
        self.errors = tuple(errors)
        super().__init__(f"{path}: " + "; ".join(self.errors))


@dataclass(frozen=True)
class TrainingBehaviorPlan:
    """Strict immutable view of the Phase 7C measurement contract."""

    source_path: Path
    data: Mapping[str, Any]

    @property
    def source_contract(self) -> Mapping[str, Any]:
        return cast(Mapping[str, Any], self.data["source_contract"])

    @property
    def measurement(self) -> Mapping[str, Any]:
        return cast(Mapping[str, Any], self.data["measurement"])

    @property
    def hardware(self) -> Mapping[str, Any]:
        return cast(Mapping[str, Any], self.data["hardware"])

    @property
    def signature(self) -> Mapping[str, Any]:
        return cast(Mapping[str, Any], self.data["compiled_signature"])

    @property
    def gates(self) -> Mapping[str, Any]:
        return cast(Mapping[str, Any], self.data["gates"])

    @property
    def seeds(self) -> tuple[int, ...]:
        return tuple(cast(Sequence[int], self.measurement["seeds"]))

    @property
    def source_inputs(self) -> tuple[str, ...]:
        return tuple(cast(Sequence[str], self.source_contract["source_inputs"]))


@dataclass(frozen=True)
class TrainingBehaviorFreezeReceipt:
    source_path: Path
    data: Mapping[str, Any]
    git_history_verified: bool

    @property
    def freeze_commit(self) -> str:
        return cast(str, self.data["freeze_commit"])


@dataclass(frozen=True)
class TrainingBehaviorValidationReport:
    errors: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.errors


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise TrainingBehaviorContractError(
            path, [f"cannot load YAML: {type(exc).__name__}: {exc}"]
        ) from exc
    if not isinstance(raw, dict):
        raise TrainingBehaviorContractError(path, ["document root must be a mapping"])
    return cast(dict[str, Any], raw)


def _mapping(
    value: object,
    label: str,
    errors: list[str],
    expected_keys: Sequence[str] | None = None,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        errors.append(f"{label}: expected mapping")
        return {}
    if expected_keys is not None:
        expected = set(expected_keys)
        actual = set(value)
        for key in sorted(expected - actual):
            errors.append(f"{label}: missing key `{key}`")
        for key in sorted(actual - expected):
            errors.append(f"{label}: unknown key `{key}`")
    return cast(Mapping[str, Any], value)


def _string(value: object, label: str, errors: list[str]) -> str:
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{label}: expected non-empty string")
        return ""
    return value


def _integer(value: object, label: str, errors: list[str], *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        errors.append(f"{label}: expected integer >= {minimum}")
        return minimum
    return value


def _number(value: object, label: str, errors: list[str], *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        errors.append(f"{label}: expected numeric value")
        return minimum
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        errors.append(f"{label}: expected finite value >= {minimum}")
        return minimum
    return result


def _string_list(value: object, label: str, errors: list[str]) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        errors.append(f"{label}: expected non-empty list")
        return ()
    result = tuple(_string(item, f"{label}[{index}]", errors) for index, item in enumerate(value))
    if len(result) != len(set(result)):
        errors.append(f"{label}: duplicate values are forbidden")
    return result


def _integer_list(value: object, label: str, errors: list[str]) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        errors.append(f"{label}: expected non-empty list")
        return ()
    result = tuple(_integer(item, f"{label}[{index}]", errors) for index, item in enumerate(value))
    if len(result) != len(set(result)):
        errors.append(f"{label}: duplicate values are forbidden")
    return result


def _expect(value: object, expected: object, label: str, errors: list[str]) -> None:
    if value != expected:
        errors.append(f"{label}: expected {expected!r}, got {value!r}")


def _sha256(value: object, label: str, errors: list[str]) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        errors.append(f"{label}: expected sha256:<64 lowercase hex>")
        return ""
    return value


def _path_value(root: Mapping[str, Any], dotted: str) -> object:
    value: object = root
    for part in dotted.split("."):
        if not isinstance(value, Mapping) or part not in value:
            raise KeyError(dotted)
        value = value[part]
    return value


def _plan_schema_errors(raw: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    root = _mapping(raw, "plan", errors, _ROOT_KEYS)
    source = _mapping(root.get("source_contract"), "source_contract", errors, _SOURCE_KEYS)
    dependencies = _mapping(
        root.get("dependencies"), "dependencies", errors, ("lockfile", "packages")
    )
    packages = _mapping(
        dependencies.get("packages"),
        "dependencies.packages",
        errors,
        ("mujoco-warp", "warp-lang", "torch"),
    )
    hardware = _mapping(
        root.get("hardware"),
        "hardware",
        errors,
        (
            "cpu_model",
            "affinity_cpus",
            "gpu_name",
            "gpu_uuid",
            "gpu_memory_mib",
            "driver_version",
            "environment_variables",
        ),
    )
    environment = _mapping(
        hardware.get("environment_variables"),
        "hardware.environment_variables",
        errors,
        ("MKL_NUM_THREADS", "NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"),
    )
    measurement = _mapping(root.get("measurement"), "measurement", errors, _MEASUREMENT_KEYS)
    success = _mapping(
        measurement.get("success_metric"),
        "measurement.success_metric",
        errors,
        ("disposition", "scalar_tag", "reason"),
    )
    signature = _mapping(
        root.get("compiled_signature"), "compiled_signature", errors, _SIGNATURE_KEYS
    )
    gates = _mapping(
        root.get("gates"),
        "gates",
        errors,
        (
            "threshold_paths",
            "apply_to_each_seed",
            "apply_to_aggregate",
            "final_window_uses_frozen_reward_and_length_thresholds",
            "maximum_candidate_fps_population_cv",
            "omitted_runs",
            "duplicate_runs",
            "failed_runs",
            "retried_runs",
            "nan_runs",
        ),
    )
    threshold_paths = _mapping(
        gates.get("threshold_paths"),
        "gates.threshold_paths",
        errors,
        (
            "fps_ratio_min",
            "reward_auc_drop_max",
            "reward_drop_max",
            "episode_length_ratio_min",
        ),
    )
    artifact = _mapping(
        root.get("artifact_contract"),
        "artifact_contract",
        errors,
        (
            "raw_scalars_required",
            "raw_memory_samples_required",
            "process_receipts_required",
            "case_order_must_match_seeds",
            "baseline_hash_required",
            "plan_hash_required",
            "freeze_receipt_required",
            "source_tree_hash_required",
            "owner_yaml_hash_required",
            "lockfile_hash_required",
            "hardware_match_required",
            "recorded_pairs_must_equal_recomputed_pairs",
            "recorded_aggregates_must_equal_recomputed_aggregates",
            "recorded_gate_must_equal_recomputed_gate",
        ),
    )
    governance = _mapping(
        root.get("governance"),
        "governance",
        errors,
        (
            "issue_url",
            "thresholds_frozen_before_candidate_capture",
            "threshold_change_requires_new_issue_and_freeze_receipt",
            "phase_promotion_is_out_of_scope",
            "failure_semantics",
        ),
    )

    _expect(root.get("schema_version"), SCHEMA_VERSION, "schema_version", errors)
    _expect(root.get("issue"), ISSUE, "issue", errors)
    _expect(root.get("parent_issue"), PARENT_ISSUE, "parent_issue", errors)
    _expect(root.get("claim_id"), CLAIM_ID, "claim_id", errors)
    _expect(root.get("benchmark_id"), BENCHMARK_ID, "benchmark_id", errors)
    _expect(root.get("state"), "frozen", "state", errors)

    expected_paths = {
        "integration_branch": "feat/issue-705-manager-mjwarp",
        "phase_manifest": "tests/acceptance/issue_705/manifests/phase_7.yaml",
        "baseline_plan": "tests/acceptance/issue_705/g1_mujoco_baseline_plan.yaml",
        "threshold_manifest": "tests/acceptance/issue_705/g1_threshold_manifest.yaml",
        "threshold_receipt": "tests/acceptance/issue_705/g1_threshold_freeze_receipt.yaml",
        "baseline_artifact": (
            "tests/acceptance/issue_705/artifacts/g1_mujoco_phase0_baseline.json"
        ),
        "owner_yaml": "conf/ppo/task/g1_walk_flat/mjwarp.yaml",
        "freeze_receipt": FREEZE_RECEIPT_PATH.as_posix(),
        "candidate_artifact": ARTIFACT_PATH.as_posix(),
    }
    for key, expected in expected_paths.items():
        _expect(source.get(key), expected, f"source_contract.{key}", errors)
    for key in (
        "candidate_must_be_clean",
        "candidate_must_descend_from_freeze",
        "candidate_must_differ_from_freeze",
    ):
        _expect(source.get(key), True, f"source_contract.{key}", errors)
    source_inputs = _string_list(
        source.get("source_inputs"), "source_contract.source_inputs", errors
    )
    for required in (
        "benchmark/rl/evaluate_issue705_training_behavior.py",
        "src/unilab/tools/issue705_training_behavior.py",
        "tests/benchmark/test_issue705_training_behavior.py",
        "tests/tools/test_issue705_training_behavior.py",
        "uv.lock",
    ):
        if required not in source_inputs:
            errors.append(f"source_contract.source_inputs: missing required path {required!r}")

    _expect(dependencies.get("lockfile"), "uv.lock", "dependencies.lockfile", errors)
    for key in packages:
        _string(packages.get(key), f"dependencies.packages.{key}", errors)
    _string(hardware.get("cpu_model"), "hardware.cpu_model", errors)
    affinity = _integer_list(hardware.get("affinity_cpus"), "hardware.affinity_cpus", errors)
    if affinity != tuple(range(16)):
        errors.append("hardware.affinity_cpus: expected frozen CPUs 0..15")
    for key in ("gpu_name", "gpu_uuid", "driver_version"):
        _string(hardware.get(key), f"hardware.{key}", errors)
    _integer(hardware.get("gpu_memory_mib"), "hardware.gpu_memory_mib", errors, minimum=1)
    for key, value in environment.items():
        _string(value, f"hardware.environment_variables.{key}", errors)

    expected_measurement = {
        "task_slug": "g1_walk_flat",
        "env_name": "G1WalkFlat",
        "baseline_backend": "mujoco",
        "candidate_backend": "mjwarp",
        "execution_profile": "device_resident",
        "dtype": "float32",
        "seeds": [0, 1, 2, 3, 4],
        "num_envs": 1024,
        "num_steps_per_env": 24,
        "max_iterations": 100,
        "warmup_iterations": 10,
        "final_window_iterations": 10,
        "save_interval": 1000,
        "memory_poll_interval_sec": 0.25,
        "process_isolation": True,
        "process_retries": 0,
        "case_filtering_forbidden": True,
    }
    for key, expected in expected_measurement.items():
        _expect(measurement.get(key), expected, f"measurement.{key}", errors)
    required_tags = _string_list(
        measurement.get("required_scalar_tags"), "measurement.required_scalar_tags", errors
    )
    _expect(
        required_tags,
        (
            "Perf/total_fps",
            "Perf/collection_time",
            "Perf/learning_time",
            "Train/mean_reward",
            "Train/mean_episode_length",
        ),
        "measurement.required_scalar_tags",
        errors,
    )
    _string_list(measurement.get("hydra_overrides"), "measurement.hydra_overrides", errors)
    _expect(success.get("disposition"), "not_applicable", "success_metric.disposition", errors)
    _expect(success.get("scalar_tag"), None, "success_metric.scalar_tag", errors)
    _string(success.get("reason"), "success_metric.reason", errors)
    for key in _SIGNATURE_KEYS:
        _string(signature.get(key), f"compiled_signature.{key}", errors)

    expected_threshold_paths = {
        "fps_ratio_min": "gates.training.fps_p50_median_ratio_min",
        "reward_auc_drop_max": "gates.training.reward_auc_median_drop_max",
        "reward_drop_max": "gates.training.final_reward_p50_median_drop_max",
        "episode_length_ratio_min": "gates.training.episode_length_median_ratio_min",
    }
    for key, expected in expected_threshold_paths.items():
        _expect(threshold_paths.get(key), expected, f"gates.threshold_paths.{key}", errors)
    for key in ("apply_to_each_seed", "apply_to_aggregate"):
        _expect(gates.get(key), True, f"gates.{key}", errors)
    _expect(
        gates.get("final_window_uses_frozen_reward_and_length_thresholds"),
        True,
        "gates.final_window_uses_frozen_reward_and_length_thresholds",
        errors,
    )
    _expect(
        gates.get("maximum_candidate_fps_population_cv"),
        0.10,
        "gates.maximum_candidate_fps_population_cv",
        errors,
    )
    for key in ("omitted_runs", "duplicate_runs", "failed_runs", "retried_runs", "nan_runs"):
        _expect(gates.get(key), 0, f"gates.{key}", errors)
    for key, value in artifact.items():
        _expect(value, True, f"artifact_contract.{key}", errors)
    _expect(
        governance.get("issue_url"),
        "https://github.com/unilabsim/UniLab/issues/837",
        "governance.issue_url",
        errors,
    )
    for key in (
        "thresholds_frozen_before_candidate_capture",
        "threshold_change_requires_new_issue_and_freeze_receipt",
        "phase_promotion_is_out_of_scope",
    ):
        _expect(governance.get(key), True, f"governance.{key}", errors)
    _string(governance.get("failure_semantics"), "governance.failure_semantics", errors)
    return errors


def _linked_plan_errors(plan: TrainingBehaviorPlan, repo_root: Path) -> list[str]:
    errors: list[str] = []
    source = plan.source_contract
    measurement = plan.measurement
    baseline_plan_path = repo_root / cast(str, source["baseline_plan"])
    threshold_path = repo_root / cast(str, source["threshold_manifest"])
    threshold_receipt_path = repo_root / cast(str, source["threshold_receipt"])
    try:
        baseline_plan = load_g1_baseline_plan(baseline_plan_path)
        frozen = baseline_plan.ppo_lane
        expected = {
            "seeds": list(frozen.seeds),
            "num_envs": frozen.num_envs,
            "num_steps_per_env": frozen.num_steps_per_env,
            "max_iterations": frozen.max_iterations,
            "warmup_iterations": frozen.warmup_iterations,
            "save_interval": frozen.save_interval,
            "memory_poll_interval_sec": frozen.memory_poll_interval_sec,
            "required_scalar_tags": list(frozen.required_scalar_tags),
        }
        for key, expected_value in expected.items():
            if measurement.get(key) != expected_value:
                errors.append(f"measurement.{key}: differs from Phase 0 baseline plan")
        expected_hardware = {
            "cpu_model": baseline_plan.hardware.cpu_model,
            "affinity_cpus": list(baseline_plan.hardware.affinity_cpus),
            "gpu_name": baseline_plan.hardware.gpu_name,
            "gpu_uuid": baseline_plan.hardware.gpu_uuid,
            "gpu_memory_mib": baseline_plan.hardware.gpu_memory_mib,
            "driver_version": baseline_plan.hardware.driver_version,
            "environment_variables": dict(baseline_plan.environment.env_vars),
        }
        if plan.hardware != expected_hardware:
            errors.append("hardware: differs from Phase 0 baseline plan")
    except Exception as exc:  # noqa: BLE001 - linked frozen evidence must fail closed
        errors.append(f"baseline plan binding failed: {type(exc).__name__}: {exc}")

    try:
        threshold = load_threshold_manifest(threshold_path, repo_root=repo_root)
        load_freeze_receipt(
            threshold_receipt_path,
            manifest=threshold,
            repo_root=repo_root,
        )
        baseline = cast(Mapping[str, Any], threshold.data["baseline"])
        if baseline.get("plan_path") != source.get("baseline_plan"):
            errors.append("source_contract.baseline_plan: differs from threshold binding")
        if baseline.get("artifact_path") != source.get("baseline_artifact"):
            errors.append("source_contract.baseline_artifact: differs from threshold binding")
        for name, dotted in cast(Mapping[str, str], plan.gates["threshold_paths"]).items():
            try:
                value = _path_value(threshold.data, dotted)
            except KeyError:
                errors.append(f"gates.threshold_paths.{name}: missing frozen threshold {dotted!r}")
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                errors.append(f"gates.threshold_paths.{name}: frozen value is not numeric")
    except Exception as exc:  # noqa: BLE001 - linked frozen evidence must fail closed
        errors.append(f"threshold binding failed: {type(exc).__name__}: {exc}")

    try:
        phase_raw = _load_yaml(repo_root / cast(str, source["phase_manifest"]))
        claims = phase_raw.get("claims")
        if not isinstance(claims, list):
            raise ValueError("phase claims must be a list")
        claim = next(
            item
            for item in claims
            if isinstance(item, Mapping) and item.get("claim_id") == CLAIM_ID
        )
        environment = _mapping(claim.get("environment"), "phase claim environment", errors)
        acceptance = _mapping(claim.get("acceptance"), "phase claim acceptance", errors)
        if environment.get("seeds") != list(plan.seeds):
            errors.append("Phase 7 behavior seeds differ from frozen plan")
        if environment.get("batch_sizes") != [measurement["num_envs"]]:
            errors.append("Phase 7 behavior batch differs from Phase 0 plan")
        if environment.get("plan_fingerprint") != BENCHMARK_ID:
            errors.append("Phase 7 behavior fingerprint differs from benchmark plan")
        if acceptance.get("max_dispersion") != plan.gates["maximum_candidate_fps_population_cv"]:
            errors.append("Phase 7 max_dispersion differs from benchmark plan")
    except Exception as exc:  # noqa: BLE001 - missing claim mapping must fail closed
        errors.append(f"Phase 7 manifest binding failed: {type(exc).__name__}: {exc}")

    try:
        rollout = load_task_rollout_plan(repo_root / ROLLOUT_PLAN_PATH)
        if len(rollout.entries) != 1:
            errors.append("compiled signature requires exactly one promoted rollout entry")
        else:
            support = rollout.entries[0].rollout_compiled_signature
            expected_signature = {
                "task_key": support.task_key,
                "executor_key": support.executor_key,
                "task_plan_fingerprint": support.task_plan_fingerprint,
                "policy_abi_fingerprint": support.policy_abi_fingerprint,
                "backend_plan_fingerprint": support.backend_plan_fingerprint,
            }
            if plan.signature != expected_signature:
                errors.append("compiled_signature: differs from benchmark rollout signature")
    except Exception as exc:  # noqa: BLE001 - linked support identity must fail closed
        errors.append(f"compiled signature binding failed: {type(exc).__name__}: {exc}")
    return errors


def load_training_behavior_plan(
    path: Path, *, repo_root: Path | None = None
) -> TrainingBehaviorPlan:
    raw = _load_yaml(path)
    errors = _plan_schema_errors(raw)
    plan = TrainingBehaviorPlan(source_path=path, data=raw)
    if repo_root is not None:
        errors.extend(_linked_plan_errors(plan, repo_root.resolve()))
    if errors:
        raise TrainingBehaviorContractError(path, errors)
    return plan


_RECEIPT_KEYS = (
    "schema_version",
    "issue",
    "parent_issue",
    "benchmark_id",
    "plan_path",
    "plan_sha256",
    "plan_git_blob",
    "freeze_commit",
    "issue_url",
    "change_policy",
    "creation_verification",
    "shallow_checkout_policy",
    "final_merge_method",
)


def _git(
    repo_root: Path, args: Sequence[str], *, check: bool = True
) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(["git", *args], cwd=repo_root, check=check, capture_output=True)


def _git_file_bytes(repo_root: Path, commit: str, path: str) -> bytes:
    return _git(repo_root, ["show", f"{commit}:{path}"]).stdout


def load_training_behavior_freeze_receipt(
    path: Path,
    *,
    plan: TrainingBehaviorPlan,
    repo_root: Path,
    verify_git: bool = True,
) -> TrainingBehaviorFreezeReceipt:
    raw = _load_yaml(path)
    errors: list[str] = []
    receipt = _mapping(raw, "receipt", errors, _RECEIPT_KEYS)
    expected = {
        "schema_version": SCHEMA_VERSION,
        "issue": ISSUE,
        "parent_issue": PARENT_ISSUE,
        "benchmark_id": BENCHMARK_ID,
        "plan_path": PLAN_PATH.as_posix(),
        "plan_sha256": sha256_file(plan.source_path),
        "issue_url": "https://github.com/unilabsim/UniLab/issues/837",
        "creation_verification": "full_git_history",
        "shallow_checkout_policy": "current_hash_and_receipt",
        "final_merge_method": "merge_commit",
    }
    for key, expected_value in expected.items():
        _expect(receipt.get(key), expected_value, key, errors)
    _string(receipt.get("change_policy"), "change_policy", errors)
    plan_sha = _sha256(receipt.get("plan_sha256"), "plan_sha256", errors)
    freeze_commit = _string(receipt.get("freeze_commit"), "freeze_commit", errors)
    plan_blob = _string(receipt.get("plan_git_blob"), "plan_git_blob", errors)
    if _COMMIT_RE.fullmatch(freeze_commit) is None:
        errors.append("freeze_commit: expected full lowercase commit SHA")
    if _GIT_OBJECT_RE.fullmatch(plan_blob) is None:
        errors.append("plan_git_blob: expected full Git object ID")
    git_history_verified = False
    repository = repo_root.resolve()
    if verify_git and _COMMIT_RE.fullmatch(freeze_commit):
        try:
            exists = _git(
                repository,
                ["cat-file", "-e", f"{freeze_commit}^{{commit}}"],
                check=False,
            )
            if exists.returncode != 0:
                shallow = _git(repository, ["rev-parse", "--is-shallow-repository"], check=False)
                if shallow.returncode != 0 or shallow.stdout.decode().strip() != "true":
                    errors.append("freeze_commit: cannot verify Git object in full history")
            else:
                git_history_verified = True
                frozen_bytes = _git_file_bytes(repository, freeze_commit, PLAN_PATH.as_posix())
                frozen_sha = f"sha256:{hashlib.sha256(frozen_bytes).hexdigest()}"
                if frozen_sha != plan_sha:
                    errors.append("freeze commit contains different plan bytes")
                actual_blob = (
                    _git(
                        repository,
                        ["rev-parse", f"{freeze_commit}:{PLAN_PATH.as_posix()}"],
                    )
                    .stdout.decode()
                    .strip()
                )
                if actual_blob != plan_blob:
                    errors.append("plan_git_blob: differs from freeze commit")
        except (OSError, subprocess.CalledProcessError) as exc:
            errors.append(f"freeze receipt Git verification failed: {type(exc).__name__}: {exc}")
    if errors:
        raise TrainingBehaviorContractError(path, errors)
    return TrainingBehaviorFreezeReceipt(
        source_path=path,
        data=raw,
        git_history_verified=git_history_verified,
    )


def load_frozen_training_inputs(
    plan: TrainingBehaviorPlan, *, repo_root: Path
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    source = plan.source_contract
    threshold_path = repo_root / cast(str, source["threshold_manifest"])
    threshold = load_threshold_manifest(threshold_path, repo_root=repo_root)
    load_freeze_receipt(
        repo_root / cast(str, source["threshold_receipt"]),
        manifest=threshold,
        repo_root=repo_root,
    )
    baseline_path = repo_root / cast(str, source["baseline_artifact"])
    if sha256_file(baseline_path) != threshold.data["baseline"]["artifact_sha256"]:
        raise TrainingBehaviorContractError(
            baseline_path, ["baseline artifact SHA differs from threshold manifest"]
        )
    try:
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TrainingBehaviorContractError(
            baseline_path, [f"cannot load baseline artifact: {type(exc).__name__}: {exc}"]
        ) from exc
    if not isinstance(baseline, Mapping):
        raise TrainingBehaviorContractError(baseline_path, ["baseline root must be a mapping"])
    return threshold.data, cast(Mapping[str, Any], baseline)


def _median(values: Sequence[float]) -> float:
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("median requires finite values")
    return float(statistics.median(values))


def _population_cv(values: Sequence[float]) -> float:
    if len(values) < 2 or any(not math.isfinite(value) for value in values):
        raise ValueError("population CV requires at least two finite values")
    mean = statistics.fmean(values)
    if mean <= 0.0:
        raise ValueError("population CV requires a positive mean")
    return float(statistics.pstdev(values) / mean)


def _curve_values(
    raw: Mapping[str, Any],
    plan: TrainingBehaviorPlan,
    *,
    label: str,
) -> dict[str, list[tuple[int, float]]]:
    errors: list[str] = []
    scalars = _mapping(raw.get("scalars"), f"{label}.scalars", errors)
    required = tuple(cast(Sequence[str], plan.measurement["required_scalar_tags"]))
    if set(scalars) != set(required):
        errors.append(f"{label}.scalars: tags differ from frozen protocol")
    if any("success" in str(tag).lower() for tag in scalars):
        errors.append(f"{label}.scalars: success metric is not registered for G1WalkFlat")
    iterations = cast(int, plan.measurement["max_iterations"])
    result: dict[str, list[tuple[int, float]]] = {}
    for tag in required:
        points = scalars.get(tag)
        if not isinstance(points, list):
            errors.append(f"{label}.{tag}: expected scalar point list")
            continue
        parsed: list[tuple[int, float]] = []
        for index, point_value in enumerate(points):
            point = _mapping(point_value, f"{label}.{tag}[{index}]", errors)
            step = _integer(point.get("step"), f"{label}.{tag}[{index}].step", errors)
            value = point.get("value")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                errors.append(f"{label}.{tag}[{index}].value: expected numeric value")
                continue
            number = float(value)
            if not math.isfinite(number):
                errors.append(f"{label}.{tag}[{index}].value: expected finite value")
                continue
            parsed.append((step, number))
        if tuple(step for step, _ in parsed) != tuple(range(iterations)):
            errors.append(f"{label}.{tag}: must contain every iteration exactly once")
        result[tag] = parsed
    if errors:
        raise TrainingBehaviorContractError(Path(label), errors)
    return result


def summarize_training_behavior_raw(
    raw: Mapping[str, Any], plan: TrainingBehaviorPlan, *, label: str
) -> dict[str, Any]:
    curves = _curve_values(raw, plan, label=label)
    warmup = cast(int, plan.measurement["warmup_iterations"])
    iterations = cast(int, plan.measurement["max_iterations"])
    window = cast(int, plan.measurement["final_window_iterations"])

    def selected(tag: str, start: int) -> list[tuple[int, float]]:
        values = [(step, value) for step, value in curves[tag] if step >= start]
        if not values:
            raise TrainingBehaviorContractError(Path(label), [f"{tag}: empty selected window"])
        return values

    reward = selected("Train/mean_reward", warmup)
    auc = 0.0
    for (left_step, left), (right_step, right) in zip(reward, reward[1:], strict=False):
        auc += (right_step - left_step) * (left + right) * 0.5
    memory_samples = raw.get("memory_samples")
    if not isinstance(memory_samples, list) or not memory_samples:
        raise TrainingBehaviorContractError(
            Path(label), ["memory_samples: expected non-empty raw sample list"]
        )
    rss_values: list[int] = []
    for index, sample_value in enumerate(memory_samples):
        errors: list[str] = []
        sample = _mapping(sample_value, f"{label}.memory_samples[{index}]", errors)
        rss_values.append(
            _integer(
                sample.get("rss_bytes"),
                f"{label}.memory_samples[{index}].rss_bytes",
                errors,
            )
        )
        if errors:
            raise TrainingBehaviorContractError(Path(label), errors)
    run_summary = raw.get("run_summary")
    if not isinstance(run_summary, Mapping):
        raise TrainingBehaviorContractError(Path(label), ["run_summary: expected mapping"])
    summary_errors: list[str] = []
    peak_gpu_memory_allocated_bytes = _integer(
        run_summary.get("peak_gpu_memory_allocated_bytes"),
        f"{label}.run_summary.peak_gpu_memory_allocated_bytes",
        summary_errors,
    )
    peak_gpu_memory_reserved_bytes = _integer(
        run_summary.get("peak_gpu_memory_reserved_bytes"),
        f"{label}.run_summary.peak_gpu_memory_reserved_bytes",
        summary_errors,
    )
    if summary_errors:
        raise TrainingBehaviorContractError(Path(label), summary_errors)
    return {
        "fps_p50": _median([value for _, value in selected("Perf/total_fps", warmup)]),
        "reward_auc": float(auc),
        "reward_p50": _median([value for _, value in reward]),
        "episode_length_p50": _median(
            [value for _, value in selected("Train/mean_episode_length", warmup)]
        ),
        "final_window_reward_p50": _median(
            [value for _, value in selected("Train/mean_reward", iterations - window)]
        ),
        "final_window_episode_length_p50": _median(
            [value for _, value in selected("Train/mean_episode_length", iterations - window)]
        ),
        "peak_rss_bytes": max(rss_values),
        "peak_gpu_memory_allocated_bytes": peak_gpu_memory_allocated_bytes,
        "peak_gpu_memory_reserved_bytes": peak_gpu_memory_reserved_bytes,
    }


def _baseline_cases(
    baseline: Mapping[str, Any], plan: TrainingBehaviorPlan
) -> dict[int, Mapping[str, Any]]:
    cases = baseline.get("cases")
    if not isinstance(cases, list):
        raise TrainingBehaviorContractError(Path("baseline"), ["cases must be a list"])
    selected: dict[int, Mapping[str, Any]] = {}
    for value in cases:
        if not isinstance(value, Mapping) or value.get("lane") != "ppo":
            continue
        seed = value.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int):
            continue
        if seed in selected:
            raise TrainingBehaviorContractError(Path("baseline"), [f"duplicate PPO seed {seed}"])
        selected[seed] = cast(Mapping[str, Any], value)
    if tuple(sorted(selected)) != plan.seeds:
        raise TrainingBehaviorContractError(
            Path("baseline"), ["PPO seed set differs from frozen paired-seed plan"]
        )
    for seed, case in selected.items():
        if case.get("batch_size") != plan.measurement["num_envs"]:
            raise TrainingBehaviorContractError(
                Path("baseline"), [f"seed {seed}: batch differs from frozen plan"]
            )
    return selected


def _threshold_values(threshold: Mapping[str, Any], plan: TrainingBehaviorPlan) -> dict[str, float]:
    result: dict[str, float] = {}
    for name, dotted in cast(Mapping[str, str], plan.gates["threshold_paths"]).items():
        try:
            value = _path_value(threshold, dotted)
        except KeyError as exc:
            raise TrainingBehaviorContractError(
                Path("threshold"), [f"missing frozen threshold {dotted!r}"]
            ) from exc
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TrainingBehaviorContractError(
                Path("threshold"), [f"{dotted}: expected numeric threshold"]
            )
        result[name] = float(value)
    return result


def _pair_metrics(baseline: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, float]:
    return {
        "fps_ratio": float(candidate["fps_p50"]) / float(baseline["fps_p50"]),
        "reward_auc_drop": float(baseline["reward_auc"]) - float(candidate["reward_auc"]),
        "reward_drop": float(baseline["reward_p50"]) - float(candidate["reward_p50"]),
        "episode_length_ratio": float(candidate["episode_length_p50"])
        / float(baseline["episode_length_p50"]),
        "final_window_reward_drop": float(baseline["final_window_reward_p50"])
        - float(candidate["final_window_reward_p50"]),
        "final_window_episode_length_ratio": float(candidate["final_window_episode_length_p50"])
        / float(baseline["final_window_episode_length_p50"]),
    }


def _metric_gate_errors(metrics: Mapping[str, float], limits: Mapping[str, float]) -> list[str]:
    errors: list[str] = []
    checks = (
        ("fps_ratio", ">=", limits["fps_ratio_min"]),
        ("reward_auc_drop", "<=", limits["reward_auc_drop_max"]),
        ("reward_drop", "<=", limits["reward_drop_max"]),
        ("episode_length_ratio", ">=", limits["episode_length_ratio_min"]),
        ("final_window_reward_drop", "<=", limits["reward_drop_max"]),
        (
            "final_window_episode_length_ratio",
            ">=",
            limits["episode_length_ratio_min"],
        ),
    )
    for name, operator, limit in checks:
        value = metrics[name]
        passed = value >= limit if operator == ">=" else value <= limit
        if not passed:
            errors.append(f"{name}: expected {operator} {limit}, got {value}")
    return errors


def _aggregate_summaries(summaries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "process_count": len(summaries),
        "fps_p50_median": _median([float(item["fps_p50"]) for item in summaries]),
        "fps_p50_population_cv": _population_cv([float(item["fps_p50"]) for item in summaries]),
        "reward_auc_median": _median([float(item["reward_auc"]) for item in summaries]),
        "reward_p50_median": _median([float(item["reward_p50"]) for item in summaries]),
        "episode_length_p50_median": _median(
            [float(item["episode_length_p50"]) for item in summaries]
        ),
        "final_window_reward_p50_median": _median(
            [float(item["final_window_reward_p50"]) for item in summaries]
        ),
        "final_window_episode_length_p50_median": _median(
            [float(item["final_window_episode_length_p50"]) for item in summaries]
        ),
        "peak_rss_max_bytes": max(int(item["peak_rss_bytes"]) for item in summaries),
        "peak_gpu_reserved_max_bytes": max(
            int(item["peak_gpu_memory_reserved_bytes"]) for item in summaries
        ),
    }


def _aggregate_pair_metrics(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, float]:
    def normalized(summary: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "fps_p50": summary["fps_p50_median"],
            "reward_auc": summary["reward_auc_median"],
            "reward_p50": summary["reward_p50_median"],
            "episode_length_p50": summary["episode_length_p50_median"],
            "final_window_reward_p50": summary["final_window_reward_p50_median"],
            "final_window_episode_length_p50": summary["final_window_episode_length_p50_median"],
        }

    return _pair_metrics(normalized(baseline), normalized(candidate))


def evaluate_training_behavior_cases(
    *,
    plan: TrainingBehaviorPlan,
    threshold: Mapping[str, Any],
    baseline: Mapping[str, Any],
    candidate_cases: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any], list[str]]:
    errors: list[str] = []
    expected_ids = [f"behavior-mjwarp_device-seed{seed}" for seed in plan.seeds]
    observed_ids = [case.get("case_id") for case in candidate_cases]
    if observed_ids != expected_ids:
        errors.append("candidate case order must exactly match frozen seeds")
    seeds = [case.get("seed") for case in candidate_cases]
    if len(seeds) != len(set(seeds)):
        errors.append("candidate contains duplicate seeds")
    if tuple(seed for seed in seeds if isinstance(seed, int)) != plan.seeds:
        errors.append("candidate seed set differs from frozen plan")
    if len(candidate_cases) != len(plan.seeds):
        errors.append("candidate omitted or added a training run")

    baseline_cases = _baseline_cases(baseline, plan)
    limits = _threshold_values(threshold, plan)
    candidate_by_seed = {
        cast(int, case["seed"]): case
        for case in candidate_cases
        if isinstance(case.get("seed"), int) and not isinstance(case.get("seed"), bool)
    }
    pairs: list[dict[str, Any]] = []
    baseline_summaries: list[Mapping[str, Any]] = []
    candidate_summaries: list[Mapping[str, Any]] = []
    for seed in plan.seeds:
        baseline_case = baseline_cases[seed]
        candidate_case = candidate_by_seed.get(seed)
        if candidate_case is None:
            errors.append(f"seed {seed}: candidate case missing")
            continue
        baseline_raw = baseline_case.get("raw")
        candidate_raw = candidate_case.get("raw")
        if not isinstance(baseline_raw, Mapping):
            raise TrainingBehaviorContractError(
                Path("baseline"), [f"seed {seed}: raw payload missing"]
            )
        if not isinstance(candidate_raw, Mapping):
            errors.append(f"seed {seed}: candidate raw payload missing")
            continue
        try:
            baseline_summary = summarize_training_behavior_raw(
                baseline_raw, plan, label=f"baseline/seed={seed}"
            )
            candidate_summary = summarize_training_behavior_raw(
                candidate_raw, plan, label=f"candidate/seed={seed}"
            )
        except TrainingBehaviorContractError as exc:
            errors.extend(exc.errors)
            continue
        if candidate_case.get("summary") != candidate_summary:
            errors.append(f"seed {seed}: candidate summary is not recomputed from raw curves")
        metrics = _pair_metrics(baseline_summary, candidate_summary)
        pair_errors = _metric_gate_errors(metrics, limits)
        pairs.append(
            {
                "seed": seed,
                "baseline": baseline_summary,
                "candidate": candidate_summary,
                "metrics": metrics,
                "gate": {"passed": not pair_errors, "errors": pair_errors},
            }
        )
        baseline_summaries.append(baseline_summary)
        candidate_summaries.append(candidate_summary)
        errors.extend(f"seed {seed}: {message}" for message in pair_errors)

    if len(candidate_summaries) != len(plan.seeds):
        aggregates: dict[str, Any] = {}
        return pairs, aggregates, errors
    baseline_aggregate = _aggregate_summaries(baseline_summaries)
    candidate_aggregate = _aggregate_summaries(candidate_summaries)
    aggregate_metrics = _aggregate_pair_metrics(baseline_aggregate, candidate_aggregate)
    aggregate_errors = _metric_gate_errors(aggregate_metrics, limits)
    cv_limit = float(plan.gates["maximum_candidate_fps_population_cv"])
    if candidate_aggregate["fps_p50_population_cv"] > cv_limit:
        aggregate_errors.append(
            "fps_p50_population_cv: expected <= "
            f"{cv_limit}, got {candidate_aggregate['fps_p50_population_cv']}"
        )
    aggregates = {
        "baseline": baseline_aggregate,
        "candidate": candidate_aggregate,
        "metrics": aggregate_metrics,
        "gate": {"passed": not aggregate_errors, "errors": aggregate_errors},
    }
    errors.extend(f"aggregate: {message}" for message in aggregate_errors)
    return pairs, aggregates, errors


def _validate_process_receipt(
    process: Mapping[str, Any],
    *,
    label: str,
    plan: TrainingBehaviorPlan,
    expected_command_prefix: Sequence[str],
    errors: list[str],
) -> None:
    if process.get("return_code") != 0:
        errors.append(f"{label}.return_code: process did not succeed")
    command = process.get("command")
    if not isinstance(command, list) or command[: len(expected_command_prefix)] != list(
        expected_command_prefix
    ):
        errors.append(f"{label}.command: unexpected process route")
    if process.get("affinity_cpus") != plan.hardware["affinity_cpus"]:
        errors.append(f"{label}.affinity_cpus: differs from frozen host")
    if process.get("env_vars") != plan.hardware["environment_variables"]:
        errors.append(f"{label}.env_vars: differs from frozen thread environment")
    if not isinstance(process.get("run_id"), str) or not process["run_id"].strip():
        errors.append(f"{label}.run_id: missing")
    value_errors: list[str] = []
    _number(process.get("duration_sec"), f"{label}.duration_sec", value_errors)
    _sha256(process.get("stdout_sha256"), f"{label}.stdout_sha256", value_errors)
    _sha256(process.get("stderr_sha256"), f"{label}.stderr_sha256", value_errors)
    errors.extend(value_errors)


def _validate_run_diagnostics(
    run_summary: Mapping[str, Any], plan: TrainingBehaviorPlan, *, label: str, errors: list[str]
) -> None:
    performance = _mapping(
        run_summary.get("runtime_performance_diagnostics"),
        f"{label}.runtime_performance_diagnostics",
        errors,
    )
    if performance.get("backend_type") != plan.measurement["candidate_backend"]:
        errors.append(f"{label}.runtime_performance_diagnostics.backend_type: mismatch")
    if performance.get("instrumentation_complete") is not True:
        errors.append(f"{label}.runtime_performance_diagnostics: incomplete")
    graph = _mapping(performance.get("graph"), f"{label}.graph", errors)
    keys = graph.get("active_keys")
    if not isinstance(keys, list) or len(keys) != 1 or not isinstance(keys[0], Mapping):
        errors.append(f"{label}.graph.active_keys: expected one graph identity")
    else:
        key = cast(Mapping[str, Any], keys[0])
        if key.get("plan_fingerprint") != plan.signature["backend_plan_fingerprint"]:
            errors.append(f"{label}.graph.plan_fingerprint: compiled signature mismatch")
        if key.get("num_envs") != plan.measurement["num_envs"]:
            errors.append(f"{label}.graph.num_envs: behavior budget mismatch")
    traffic = _mapping(run_summary.get("runtime_traffic_diagnostics"), f"{label}.traffic", errors)
    expected_steps = cast(int, plan.measurement["num_steps_per_env"]) * cast(
        int, plan.measurement["max_iterations"]
    )
    if traffic.get("policy_steps") != expected_steps:
        errors.append(f"{label}.traffic.policy_steps: behavior budget mismatch")
    for key in _ZERO_TRAFFIC_COUNTERS:
        if traffic.get(key) != 0:
            errors.append(f"{label}.traffic.{key}: expected zero")
    if traffic.get("instrumentation_complete") is not True:
        errors.append(f"{label}.traffic: incomplete")
    stability = _mapping(
        run_summary.get("runtime_stability_diagnostics"), f"{label}.stability", errors
    )
    for key in ("warm_numeric_allocations", "address_churn"):
        if stability.get(key) != 0:
            errors.append(f"{label}.stability.{key}: expected zero")


def _validate_candidate_case(
    case: Mapping[str, Any],
    *,
    index: int,
    seed: int,
    plan: TrainingBehaviorPlan,
    errors: list[str],
) -> None:
    label = f"cases[{index}]"
    expected_case_id = f"behavior-mjwarp_device-seed{seed}"
    expected_fields = {
        "case_id": expected_case_id,
        "seed": seed,
        "sequence_index": index,
        "process_retries": 0,
        "batch_size": plan.measurement["num_envs"],
        "num_steps_per_env": plan.measurement["num_steps_per_env"],
        "iterations": plan.measurement["max_iterations"],
        "mode": "mjwarp_device",
    }
    for key, expected in expected_fields.items():
        if case.get(key) != expected:
            errors.append(f"{label}.{key}: expected {expected!r}, got {case.get(key)!r}")
    worker = _mapping(case.get("worker_process"), f"{label}.worker_process", errors)
    _validate_process_receipt(
        worker,
        label=f"{label}.worker_process",
        plan=plan,
        expected_command_prefix=(
            "uv",
            "run",
            "benchmark/rl/evaluate_issue705_training_behavior.py",
            "--worker",
        ),
        errors=errors,
    )
    worker_command = worker.get("command")
    if isinstance(worker_command, list):
        if not {"--seed", str(seed), "--worker-out"}.issubset(set(worker_command)):
            errors.append(f"{label}.worker_process.command: seed/output binding missing")
    process = _mapping(case.get("process"), f"{label}.process", errors)
    _validate_process_receipt(
        process,
        label=f"{label}.process",
        plan=plan,
        expected_command_prefix=("uv", "run", "scripts/train_rsl_rl.py"),
        errors=errors,
    )
    command = process.get("command")
    if isinstance(command, list):
        required = {
            "task=g1_walk_flat/mjwarp",
            f"algo.seed={seed}",
            f"algo.num_envs={plan.measurement['num_envs']}",
            f"algo.num_steps_per_env={plan.measurement['num_steps_per_env']}",
            f"algo.max_iterations={plan.measurement['max_iterations']}",
            f"algo.save_interval={plan.measurement['save_interval']}",
            "algo.capture_performance_diagnostics=true",
            "training.no_play=true",
            "training.logger=tensorboard",
            *cast(Sequence[str], plan.measurement["hydra_overrides"]),
        }
        if not required.issubset(set(command)):
            errors.append(f"{label}.process.command: differs from frozen public owner protocol")
        log_roots = [item for item in command if str(item).startswith("training.log_root=")]
        if len(log_roots) != 1:
            errors.append(f"{label}.process.command: expected one isolated log root")
    raw = _mapping(case.get("raw"), f"{label}.raw", errors)
    run_config = _mapping(raw.get("run_config"), f"{label}.run_config", errors)
    if raw.get("run_config_sha256") != canonical_sha256(run_config):
        errors.append(f"{label}.run_config_sha256: does not match raw config")
    config = _mapping(run_config.get("config"), f"{label}.run_config.config", errors)
    training = _mapping(config.get("training"), f"{label}.config.training", errors)
    algo = _mapping(config.get("algo"), f"{label}.config.algo", errors)
    expected_training = {
        "task_name": plan.measurement["env_name"],
        "sim_backend": plan.measurement["candidate_backend"],
        "execution_profile": plan.measurement["execution_profile"],
        "no_play": True,
        "logger": "tensorboard",
    }
    for key, expected in expected_training.items():
        if training.get(key) != expected:
            errors.append(f"{label}.config.training.{key}: mismatch")
    expected_algo = {
        "seed": seed,
        "num_envs": plan.measurement["num_envs"],
        "num_steps_per_env": plan.measurement["num_steps_per_env"],
        "max_iterations": plan.measurement["max_iterations"],
        "capture_performance_diagnostics": True,
    }
    for key, expected in expected_algo.items():
        if algo.get(key) != expected:
            errors.append(f"{label}.config.algo.{key}: mismatch")
    snapshot = _mapping(run_config.get("contract_snapshot"), f"{label}.contract_snapshot", errors)
    policy = _mapping(snapshot.get("manager.policy_abi"), f"{label}.manager.policy_abi", errors)
    signature_checks = {
        "task_key": plan.signature["task_key"],
        "executor_key": plan.signature["executor_key"],
        "plan_fingerprint": plan.signature["task_plan_fingerprint"],
        "policy_abi_fingerprint": plan.signature["policy_abi_fingerprint"],
        "execution_profile": plan.measurement["execution_profile"],
    }
    for key, expected in signature_checks.items():
        if policy.get(key) != expected:
            errors.append(f"{label}.manager.policy_abi.{key}: compiled signature mismatch")
    run_summary = _mapping(raw.get("run_summary"), f"{label}.run_summary", errors)
    expected_summary = {
        "status": "completed",
        "algo": "ppo",
        "task": plan.measurement["env_name"],
        "sim_backend": plan.measurement["candidate_backend"],
        "configured_seed": seed,
        "effective_seed": seed,
        "completed_iterations": plan.measurement["max_iterations"],
        "total_env_steps": (
            cast(int, plan.measurement["num_envs"])
            * cast(int, plan.measurement["num_steps_per_env"])
            * cast(int, plan.measurement["max_iterations"])
        ),
    }
    for key, expected in expected_summary.items():
        if run_summary.get(key) != expected:
            errors.append(f"{label}.run_summary.{key}: mismatch")
    _validate_run_diagnostics(run_summary, plan, label=label, errors=errors)


def _source_validation_errors(
    source: Mapping[str, Any],
    *,
    plan: TrainingBehaviorPlan,
    receipt: TrainingBehaviorFreezeReceipt,
    repo_root: Path,
) -> list[str]:
    errors: list[str] = []
    commit = source.get("commit")
    if not isinstance(commit, str) or _COMMIT_RE.fullmatch(commit) is None:
        return ["source.commit: expected full lowercase commit SHA"]
    if source.get("dirty") is not False:
        errors.append("source.dirty: candidate must be clean")
    if commit == receipt.freeze_commit:
        errors.append("source.commit: candidate must differ from freeze commit")
    for command, message in (
        (
            ["merge-base", "--is-ancestor", receipt.freeze_commit, commit],
            "source.commit: candidate does not descend from freeze commit",
        ),
        (
            ["merge-base", "--is-ancestor", commit, "HEAD"],
            "source.commit: candidate is not available in current history",
        ),
    ):
        try:
            result = _git(repo_root, command, check=False)
            if result.returncode != 0:
                errors.append(message)
        except OSError as exc:
            errors.append(f"source.commit: cannot invoke Git: {exc}")
    try:
        expected = {
            "tree_sha256": source_tree_sha256_at_commit(repo_root, plan.source_inputs, commit),
            "uv_lock_sha256": (
                f"sha256:{hashlib.sha256(_git_file_bytes(repo_root, commit, 'uv.lock')).hexdigest()}"
            ),
            "owner_yaml_sha256": (
                "sha256:"
                + hashlib.sha256(
                    _git_file_bytes(
                        repo_root,
                        commit,
                        cast(str, plan.source_contract["owner_yaml"]),
                    )
                ).hexdigest()
            ),
        }
        for key, expected_value in expected.items():
            if source.get(key) != expected_value:
                errors.append(f"source.{key}: does not match candidate commit")
    except (OSError, subprocess.CalledProcessError) as exc:
        errors.append(f"source verification failed: {type(exc).__name__}: {exc}")
    return errors


def _core_artifact_errors(
    artifact: Mapping[str, Any],
    *,
    plan: TrainingBehaviorPlan,
    receipt: TrainingBehaviorFreezeReceipt,
    threshold: Mapping[str, Any],
    baseline: Mapping[str, Any],
    repo_root: Path | None,
    compare_recorded_sections: bool,
) -> tuple[list[str], list[dict[str, Any]], dict[str, Any]]:
    errors: list[str] = []
    expected_root_keys = (
        "schema_version",
        "issue",
        "parent_issue",
        "claim_id",
        "kind",
        "generated_at",
        "contract",
        "source",
        "hardware",
        "execution",
        "success_metric",
        "cases",
        "pairs",
        "aggregates",
        "gate",
    )
    _mapping(artifact, "artifact", errors, expected_root_keys)
    identity = {
        "schema_version": SCHEMA_VERSION,
        "issue": ISSUE,
        "parent_issue": PARENT_ISSUE,
        "claim_id": CLAIM_ID,
        "kind": ARTIFACT_KIND,
    }
    for key, expected in identity.items():
        if artifact.get(key) != expected:
            errors.append(f"artifact.{key}: expected {expected!r}")
    if not isinstance(artifact.get("generated_at"), str):
        errors.append("artifact.generated_at: expected timestamp string")
    contract = _mapping(artifact.get("contract"), "artifact.contract", errors)
    expected_contract = {
        "plan_path": PLAN_PATH.as_posix(),
        "plan_sha256": sha256_file(plan.source_path),
        "freeze_receipt_path": FREEZE_RECEIPT_PATH.as_posix(),
        "freeze_receipt_sha256": sha256_file(receipt.source_path),
        "freeze_commit": receipt.freeze_commit,
        "threshold_manifest_path": cast(str, plan.source_contract["threshold_manifest"]),
        "threshold_manifest_sha256": sha256_file(
            (repo_root or Path.cwd()) / cast(str, plan.source_contract["threshold_manifest"])
        ),
        "baseline_artifact_path": cast(str, plan.source_contract["baseline_artifact"]),
        "baseline_artifact_sha256": threshold["baseline"]["artifact_sha256"],
    }
    for key, expected in expected_contract.items():
        if contract.get(key) != expected:
            errors.append(f"artifact.contract.{key}: mismatch")
    source = _mapping(artifact.get("source"), "artifact.source", errors)
    for key in ("tree_sha256", "uv_lock_sha256", "owner_yaml_sha256"):
        _sha256(source.get(key), f"artifact.source.{key}", errors)
    if repo_root is not None:
        errors.extend(
            _source_validation_errors(
                source,
                plan=plan,
                receipt=receipt,
                repo_root=repo_root.resolve(),
            )
        )
    hardware = _mapping(artifact.get("hardware"), "artifact.hardware", errors)
    for key in (
        "cpu_model",
        "affinity_cpus",
        "gpu_name",
        "gpu_uuid",
        "gpu_memory_mib",
        "driver_version",
    ):
        if hardware.get(key) != plan.hardware[key]:
            errors.append(f"artifact.hardware.{key}: differs from frozen host")
    execution = _mapping(artifact.get("execution"), "artifact.execution", errors)
    expected_order = [f"behavior-mjwarp_device-seed{seed}" for seed in plan.seeds]
    if execution.get("case_order") != expected_order:
        errors.append("artifact.execution.case_order: differs from frozen seed order")
    if execution.get("process_isolation") is not True:
        errors.append("artifact.execution.process_isolation: expected true")
    if execution.get("process_retries") != 0:
        errors.append("artifact.execution.process_retries: expected zero")
    if execution.get("environment_variables") != plan.hardware["environment_variables"]:
        errors.append("artifact.execution.environment_variables: mismatch")
    for key in ("preflight_before", "preflight_after"):
        preflight = _mapping(execution.get(key), f"artifact.execution.{key}", errors)
        if preflight.get("gpu_compute_processes") != []:
            errors.append(f"artifact.execution.{key}: foreign GPU compute process recorded")
    if artifact.get("success_metric") != plan.measurement["success_metric"]:
        errors.append("artifact.success_metric: must remain explicitly not applicable")
    cases_value = artifact.get("cases")
    cases: list[Mapping[str, Any]] = []
    if not isinstance(cases_value, list):
        errors.append("artifact.cases: expected list")
    else:
        cases = [cast(Mapping[str, Any], item) for item in cases_value if isinstance(item, Mapping)]
        if len(cases) != len(cases_value):
            errors.append("artifact.cases: every item must be a mapping")
    run_ids: list[str] = []
    for index, seed in enumerate(plan.seeds):
        if index >= len(cases):
            break
        case = cases[index]
        _validate_candidate_case(case, index=index, seed=seed, plan=plan, errors=errors)
        for process_key in ("worker_process", "process"):
            process = case.get(process_key)
            if isinstance(process, Mapping) and isinstance(process.get("run_id"), str):
                run_ids.append(cast(str, process["run_id"]))
    if len(run_ids) != len(set(run_ids)):
        errors.append("artifact.cases: process receipts reuse a run_id")
    try:
        pairs, aggregates, evaluation_errors = evaluate_training_behavior_cases(
            plan=plan,
            threshold=threshold,
            baseline=baseline,
            candidate_cases=cases,
        )
        errors.extend(evaluation_errors)
    except TrainingBehaviorContractError as exc:
        errors.extend(exc.errors)
        pairs, aggregates = [], {}
    if compare_recorded_sections:
        if artifact.get("pairs") != pairs:
            errors.append("artifact.pairs: differs from independent raw recomputation")
        if artifact.get("aggregates") != aggregates:
            errors.append("artifact.aggregates: differs from independent raw recomputation")
    return errors, pairs, aggregates


def build_training_behavior_sections(
    artifact: Mapping[str, Any],
    *,
    plan: TrainingBehaviorPlan,
    receipt: TrainingBehaviorFreezeReceipt,
    threshold: Mapping[str, Any],
    baseline: Mapping[str, Any],
    repo_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    errors, pairs, aggregates = _core_artifact_errors(
        artifact,
        plan=plan,
        receipt=receipt,
        threshold=threshold,
        baseline=baseline,
        repo_root=repo_root,
        compare_recorded_sections=False,
    )
    return pairs, aggregates, {"passed": not errors, "errors": errors}


def validate_training_behavior_artifact(
    artifact: object,
    *,
    plan: TrainingBehaviorPlan,
    receipt: TrainingBehaviorFreezeReceipt,
    threshold: Mapping[str, Any],
    baseline: Mapping[str, Any],
    repo_root: Path | None,
) -> TrainingBehaviorValidationReport:
    if not isinstance(artifact, Mapping):
        return TrainingBehaviorValidationReport(("artifact root must be a mapping",))
    errors, _, _ = _core_artifact_errors(
        cast(Mapping[str, Any], artifact),
        plan=plan,
        receipt=receipt,
        threshold=threshold,
        baseline=baseline,
        repo_root=repo_root,
        compare_recorded_sections=True,
    )
    gate = artifact.get("gate")
    expected_gate = {"passed": not errors, "errors": errors}
    if gate != expected_gate:
        errors.append("artifact.gate: differs from independent core validation")
    return TrainingBehaviorValidationReport(tuple(errors))


def load_training_behavior_artifact(
    path: Path, *, repo_root: Path
) -> tuple[Mapping[str, Any], TrainingBehaviorValidationReport]:
    plan = load_training_behavior_plan(repo_root / PLAN_PATH, repo_root=repo_root)
    receipt = load_training_behavior_freeze_receipt(
        repo_root / FREEZE_RECEIPT_PATH,
        plan=plan,
        repo_root=repo_root,
    )
    threshold, baseline = load_frozen_training_inputs(plan, repo_root=repo_root)
    try:
        artifact = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TrainingBehaviorContractError(
            path, [f"cannot load artifact: {type(exc).__name__}: {exc}"]
        ) from exc
    report = validate_training_behavior_artifact(
        artifact,
        plan=plan,
        receipt=receipt,
        threshold=threshold,
        baseline=baseline,
        repo_root=repo_root,
    )
    return cast(Mapping[str, Any], artifact), report
