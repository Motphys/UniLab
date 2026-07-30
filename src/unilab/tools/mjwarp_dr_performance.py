"""Frozen contract loader for the Issue #829 mjwarp DR benchmark."""

from __future__ import annotations

import re
import subprocess
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from omegaconf import OmegaConf

from unilab.tools.g1_baseline_provenance import sha256_file

SCHEMA_VERSION = 1
ISSUE = 829
PARENT_ISSUE = 705
BENCHMARK_ID = "mjwarp-dr-performance-v1"
PLAN_PATH = Path("tests/acceptance/issue_705/mjwarp_dr_performance_plan.yaml")
FREEZE_RECEIPT_PATH = Path("tests/acceptance/issue_705/mjwarp_dr_performance_freeze_receipt.yaml")
PLAN_SHA256 = "sha256:094a49b35be6a7860d1c67716721700886912b765964b153dc06fbd1f1866950"
PLAN_GIT_BLOB = "b2f966ebfe329408c03de2f668e48d3fd9ae983e"
FREEZE_COMMIT = "9b1dc068f99802586f6c042f282be409292afae3"

_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_GIT_OBJECT_RE = re.compile(r"^[0-9a-f]{40,64}$")


class MjwarpDrPerformanceContractError(ValueError):
    """Raised when a plan, receipt, or candidate breaks the frozen contract."""

    def __init__(self, source: Path, errors: Iterable[str]) -> None:
        self.source = source
        self.errors = tuple(errors)
        detail = "\n".join(f"- {error}" for error in self.errors)
        super().__init__(f"invalid mjwarp DR performance contract {source}:\n{detail}")


@dataclass(frozen=True)
class DrPerformanceProfile:
    profile_id: str
    tier: str
    model_targets: tuple[str, ...]
    direct_fields: tuple[str, ...]
    derived_fields: tuple[str, ...]
    strongest_recompute: str
    g1_owner_overrides: Mapping[str, bool] | None
    lanes: tuple[str, ...]


@dataclass(frozen=True)
class MjwarpDrPerformancePlan:
    source_path: Path
    data: Mapping[str, Any]
    profiles: tuple[DrPerformanceProfile, ...]

    @property
    def plan_sha256(self) -> str:
        return sha256_file(self.source_path)

    @property
    def reset_worker_count(self) -> int:
        measurement = cast(Mapping[str, Any], self.data["measurement"])
        reset = cast(Mapping[str, Any], measurement["reset"])
        return (
            len(cast(list[Any], reset["batch_sizes"]))
            * len(cast(list[Any], reset["densities"]))
            * len(cast(list[Any], reset["profiles"]))
            * int(measurement["process_repeats"])
        )

    @property
    def env_worker_count(self) -> int:
        measurement = cast(Mapping[str, Any], self.data["measurement"])
        lane = cast(Mapping[str, Any], measurement["env"])
        return (
            len(cast(list[Any], lane["batch_sizes"]))
            * len(cast(list[Any], lane["profiles"]))
            * int(measurement["process_repeats"])
        )

    @property
    def train_worker_count(self) -> int:
        measurement = cast(Mapping[str, Any], self.data["measurement"])
        lane = cast(Mapping[str, Any], measurement["train"])
        return len(cast(list[Any], lane["profiles"])) * int(measurement["process_repeats"])


@dataclass(frozen=True)
class MjwarpDrPerformanceFreezeReceipt:
    source_path: Path
    data: Mapping[str, Any]
    git_history_verified: bool

    @property
    def freeze_commit(self) -> str:
        return str(self.data["freeze_commit"])


def _load_mapping(path: Path) -> Mapping[str, Any]:
    try:
        raw = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    except Exception as exc:
        raise MjwarpDrPerformanceContractError(path, (f"cannot parse YAML: {exc}",)) from exc
    if not isinstance(raw, Mapping):
        raise MjwarpDrPerformanceContractError(path, ("document must be a mapping",))
    return cast(Mapping[str, Any], raw)


def _mapping(value: object, path: str, errors: list[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        errors.append(f"{path}: expected a mapping")
        return {}
    return cast(Mapping[str, Any], value)


def _exact_keys(value: Mapping[str, Any], expected: set[str], path: str, errors: list[str]) -> None:
    missing = sorted(expected.difference(value))
    unknown = sorted(set(value).difference(expected))
    if missing:
        errors.append(f"{path}: missing keys {missing!r}")
    if unknown:
        errors.append(f"{path}: unknown keys {unknown!r}")


def _string_tuple(value: object, path: str, errors: list[str]) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        errors.append(f"{path}: expected a list of non-empty strings")
        return ()
    return tuple(cast(list[str], value))


def _profiles(raw: object, errors: list[str]) -> tuple[DrPerformanceProfile, ...]:
    if not isinstance(raw, list):
        errors.append("profiles: expected a list")
        return ()
    profiles: list[DrPerformanceProfile] = []
    keys = {
        "profile_id",
        "tier",
        "model_targets",
        "direct_fields",
        "derived_fields",
        "strongest_recompute",
        "g1_owner_overrides",
        "lanes",
    }
    for index, item in enumerate(raw):
        profile = _mapping(item, f"profiles[{index}]", errors)
        _exact_keys(profile, keys, f"profiles[{index}]", errors)
        overrides_raw = profile.get("g1_owner_overrides")
        overrides: Mapping[str, bool] | None
        if overrides_raw is None:
            overrides = None
        elif isinstance(overrides_raw, Mapping) and all(
            isinstance(key, str) and isinstance(value, bool) for key, value in overrides_raw.items()
        ):
            overrides = cast(Mapping[str, bool], overrides_raw)
        else:
            errors.append(f"profiles[{index}].g1_owner_overrides: expected bool mapping or null")
            overrides = {}
        profile_id = profile.get("profile_id")
        tier = profile.get("tier")
        strongest = profile.get("strongest_recompute")
        for value, label in (
            (profile_id, "profile_id"),
            (tier, "tier"),
            (strongest, "strongest_recompute"),
        ):
            if not isinstance(value, str) or not value:
                errors.append(f"profiles[{index}].{label}: expected a non-empty string")
        profiles.append(
            DrPerformanceProfile(
                profile_id=str(profile_id),
                tier=str(tier),
                model_targets=_string_tuple(
                    profile.get("model_targets"), f"profiles[{index}].model_targets", errors
                ),
                direct_fields=_string_tuple(
                    profile.get("direct_fields"), f"profiles[{index}].direct_fields", errors
                ),
                derived_fields=_string_tuple(
                    profile.get("derived_fields"), f"profiles[{index}].derived_fields", errors
                ),
                strongest_recompute=str(strongest),
                g1_owner_overrides=overrides,
                lanes=_string_tuple(profile.get("lanes"), f"profiles[{index}].lanes", errors),
            )
        )
    expected = (
        ("disabled", "none", "none", ("reset", "env", "train")),
        ("tier_b_pd", "B", "none", ("reset", "env", "train")),
        ("tier_c_armature", "C", "set_const_0", ("reset", "env", "train")),
        ("tier_c_mixed", "C", "set_const", ("reset",)),
    )
    actual = tuple(
        (profile.profile_id, profile.tier, profile.strongest_recompute, profile.lanes)
        for profile in profiles
    )
    if actual != expected:
        errors.append("profiles: IDs, tiers, strongest recompute, and lanes differ from v1")
    return tuple(profiles)


def _validate_matrix(data: Mapping[str, Any], errors: list[str]) -> None:
    measurement = _mapping(data.get("measurement"), "measurement", errors)
    reset = _mapping(measurement.get("reset"), "measurement.reset", errors)
    env = _mapping(measurement.get("env"), "measurement.env", errors)
    train = _mapping(measurement.get("train"), "measurement.train", errors)
    expected_values: tuple[tuple[object, object, str], ...] = (
        (measurement.get("process_repeats"), 5, "measurement.process_repeats"),
        (measurement.get("seeds"), [0, 1, 2, 3, 4], "measurement.seeds"),
        (reset.get("batch_sizes"), [128, 1024, 4096], "measurement.reset.batch_sizes"),
        (reset.get("densities"), [0.0, 0.01, 0.1, 1.0], "measurement.reset.densities"),
        (
            reset.get("profiles"),
            ["disabled", "tier_b_pd", "tier_c_armature", "tier_c_mixed"],
            "measurement.reset.profiles",
        ),
        (env.get("batch_sizes"), [128, 1024, 4096], "measurement.env.batch_sizes"),
        (
            env.get("profiles"),
            ["disabled", "tier_b_pd", "tier_c_armature"],
            "measurement.env.profiles",
        ),
        (train.get("batch_size"), 1024, "measurement.train.batch_size"),
        (
            train.get("profiles"),
            ["disabled", "tier_b_pd", "tier_c_armature"],
            "measurement.train.profiles",
        ),
    )
    for actual, expected, path in expected_values:
        if actual != expected:
            errors.append(f"{path}: expected frozen value {expected!r}")
    timing = _mapping(measurement.get("timing"), "measurement.timing", errors)
    if timing.get("source") != "cuda_events":
        errors.append("measurement.timing.source: expected 'cuda_events'")
    if timing.get("synchronize_per_phase_forbidden") is not True:
        errors.append("measurement.timing.synchronize_per_phase_forbidden: expected true")

    eligibility = _mapping(data.get("tier_d_eligibility"), "tier_d_eligibility", errors)
    if eligibility.get("production_capability_ids") != [] or eligibility.get("expected_count") != 0:
        errors.append("tier_d_eligibility: v1 requires zero production capabilities")
    if eligibility.get("synthetic_cases_forbidden") is not True:
        errors.append("tier_d_eligibility.synthetic_cases_forbidden: expected true")

    gates = _mapping(data.get("gates"), "gates", errors)
    storage = _mapping(gates.get("storage"), "gates.storage", errors)
    memory = _mapping(gates.get("steady_state_memory"), "gates.steady_state_memory", errors)
    if storage.get("absolute_rss_ceiling") is not None:
        errors.append("gates.storage.absolute_rss_ceiling: v1 must remain null")
    frozen_memory = {
        "host_rss_positive_slope_bytes_per_window_max": 4 * 1024**2,
        "host_rss_last_minus_first_median_bytes_max": 16 * 1024**2,
        "cuda_allocated_positive_growth_bytes_max": 4 * 1024**2,
        "cuda_reserved_positive_growth_bytes_max": 64 * 1024**2,
    }
    for key, expected in frozen_memory.items():
        if memory.get(key) != expected:
            errors.append(f"gates.steady_state_memory.{key}: expected {expected}")


def load_mjwarp_dr_performance_plan(path: Path) -> MjwarpDrPerformancePlan:
    """Load the exact pre-registered v1 plan as typed immutable metadata."""

    raw = _load_mapping(path)
    errors: list[str] = []
    _exact_keys(
        raw,
        {
            "schema_version",
            "issue",
            "parent_issue",
            "benchmark_id",
            "state",
            "source_contract",
            "dependencies",
            "hardware",
            "profiles",
            "tier_d_eligibility",
            "measurement",
            "gates",
            "artifact_contract",
            "governance",
        },
        "plan",
        errors,
    )
    expected_identity = {
        "schema_version": SCHEMA_VERSION,
        "issue": ISSUE,
        "parent_issue": PARENT_ISSUE,
        "benchmark_id": BENCHMARK_ID,
        "state": "frozen",
    }
    for key, expected in expected_identity.items():
        if raw.get(key) != expected:
            errors.append(f"{key}: expected {expected!r}")
    actual_sha = sha256_file(path)
    if actual_sha != PLAN_SHA256:
        errors.append(f"plan SHA256 differs from frozen v1: {actual_sha}")
    profiles = _profiles(raw.get("profiles"), errors)
    _validate_matrix(raw, errors)
    if errors:
        raise MjwarpDrPerformanceContractError(path, errors)
    return MjwarpDrPerformancePlan(source_path=path, data=raw, profiles=profiles)


def _git(repo_root: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=check,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def load_mjwarp_dr_performance_freeze_receipt(
    path: Path,
    *,
    plan: MjwarpDrPerformancePlan,
    repo_root: Path,
    verify_git: bool = True,
) -> MjwarpDrPerformanceFreezeReceipt:
    """Verify the receipt, frozen Git object, and ancestry before capture."""

    raw = _load_mapping(path)
    errors: list[str] = []
    keys = {
        "schema_version",
        "issue",
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
    }
    _exact_keys(raw, keys, "receipt", errors)
    expected = {
        "schema_version": SCHEMA_VERSION,
        "issue": ISSUE,
        "benchmark_id": BENCHMARK_ID,
        "plan_path": PLAN_PATH.as_posix(),
        "plan_sha256": PLAN_SHA256,
        "plan_git_blob": PLAN_GIT_BLOB,
        "freeze_commit": FREEZE_COMMIT,
        "issue_url": "https://github.com/unilabsim/UniLab/issues/829",
        "creation_verification": "full_git_history",
        "shallow_checkout_policy": "current_hash_and_receipt",
        "final_merge_method": "merge_commit",
    }
    for key, value in expected.items():
        if raw.get(key) != value:
            errors.append(f"{key}: expected {value!r}")
    if raw.get("plan_sha256") != plan.plan_sha256:
        errors.append("plan_sha256: does not match current plan bytes")
    freeze_commit = str(raw.get("freeze_commit", ""))
    blob = str(raw.get("plan_git_blob", ""))
    if not _COMMIT_RE.fullmatch(freeze_commit):
        errors.append("freeze_commit: expected a full lowercase commit SHA")
    if not _GIT_OBJECT_RE.fullmatch(blob):
        errors.append("plan_git_blob: expected a full lowercase Git object ID")

    history_verified = False
    if verify_git and _COMMIT_RE.fullmatch(freeze_commit):
        try:
            exists = _git(repo_root, "cat-file", "-e", f"{freeze_commit}^{{commit}}", check=False)
            if exists.returncode != 0:
                shallow = _git(repo_root, "rev-parse", "--is-shallow-repository", check=False)
                if shallow.returncode != 0 or shallow.stdout.decode().strip() != "true":
                    errors.append("freeze_commit: unavailable in a full-history checkout")
            else:
                object_spec = f"{freeze_commit}:{PLAN_PATH.as_posix()}"
                actual_blob = _git(repo_root, "rev-parse", object_spec).stdout.decode().strip()
                if actual_blob != blob:
                    errors.append(
                        f"plan_git_blob: freeze contains {actual_blob}, receipt declares {blob}"
                    )
                committed = _git(repo_root, "show", object_spec).stdout
                if committed != plan.source_path.read_bytes():
                    errors.append("freeze_commit: plan bytes differ from the current frozen file")
                ancestor = _git(
                    repo_root,
                    "merge-base",
                    "--is-ancestor",
                    freeze_commit,
                    "HEAD",
                    check=False,
                )
                if ancestor.returncode != 0:
                    errors.append("freeze_commit: is not an ancestor of HEAD")
                history_verified = not errors
        except (OSError, subprocess.CalledProcessError) as exc:
            errors.append(f"freeze_commit: cannot verify Git history: {exc}")
    if errors:
        raise MjwarpDrPerformanceContractError(path, errors)
    return MjwarpDrPerformanceFreezeReceipt(
        source_path=path,
        data=raw,
        git_history_verified=history_verified,
    )


__all__ = [
    "BENCHMARK_ID",
    "FREEZE_COMMIT",
    "FREEZE_RECEIPT_PATH",
    "ISSUE",
    "MjwarpDrPerformanceContractError",
    "MjwarpDrPerformanceFreezeReceipt",
    "MjwarpDrPerformancePlan",
    "PLAN_PATH",
    "PLAN_SHA256",
    "load_mjwarp_dr_performance_freeze_receipt",
    "load_mjwarp_dr_performance_plan",
]
