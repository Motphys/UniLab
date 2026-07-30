"""Frozen contract and evidence validator for the Issue #829 DR benchmark."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import math
import re
import statistics
import subprocess
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence, cast

import numpy as np
from omegaconf import OmegaConf
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version

from unilab.tools.g1_baseline_provenance import (
    canonical_sha256,
    numeric_stats,
    sha256_file,
    source_tree_sha256_at_commit,
)

SCHEMA_VERSION = 1
ISSUE = 829
PARENT_ISSUE = 705
BENCHMARK_ID = "mjwarp-dr-performance-v1"
ARTIFACT_KIND = "issue829-mjwarp-dr-performance-v1"
PLAN_PATH = Path("tests/acceptance/issue_705/mjwarp_dr_performance_plan.yaml")
FREEZE_RECEIPT_PATH = Path("tests/acceptance/issue_705/mjwarp_dr_performance_freeze_receipt.yaml")
DEFAULT_ARTIFACT_PATH = Path(
    "tests/acceptance/issue_705/artifacts/phase_6_mjwarp_dr_performance.json"
)
PLAN_SHA256 = "sha256:094a49b35be6a7860d1c67716721700886912b765964b153dc06fbd1f1866950"
PLAN_GIT_BLOB = "b2f966ebfe329408c03de2f668e48d3fd9ae983e"
FREEZE_COMMIT = "9b1dc068f99802586f6c042f282be409292afae3"
SOURCE_INPUTS = (
    "benchmark/issue705/process_evidence.py",
    "benchmark/mjwarp/benchmark_dr_profiles.py",
    "scripts/train_rsl_rl.py",
    "src/unilab/base/backend",
    "src/unilab/dr",
    "src/unilab/envs/locomotion/g1",
    "src/unilab/manager",
    "src/unilab/tools/g1_baseline_provenance.py",
    "src/unilab/tools/mjwarp_dr_performance.py",
    "src/unilab/training/rsl_rl_device.py",
    "conf/ppo/config.yaml",
    "conf/ppo/task/g1_walk_flat/mjwarp.yaml",
    PLAN_PATH.as_posix(),
    FREEZE_RECEIPT_PATH.as_posix(),
    "tests/benchmark/test_mjwarp_dr_benchmark.py",
    "uv.lock",
)

RESET_PHASES = (
    "mutation_sample",
    "mutation_commit",
    "recompute_constants",
    "reset_forward",
    "reset_barrier",
)
TRAIN_SCALAR_TAGS = (
    "Perf/total_fps",
    "Perf/collection_time",
    "Perf/learning_time",
)
_PROFILE_EVENT_TERMS = {
    "disabled": (),
    "tier_b_pd": ("g1_randomize_kd", "g1_randomize_kp"),
    "tier_c_armature": ("g1_randomize_dof_armature",),
    "tier_c_mixed": (
        "g1_randomize_body_gravity_compensation",
        "g1_randomize_dof_armature",
    ),
}
_MODEL_FIELD_ITEMSIZE_BYTES = {
    "actuator_acc0": 4,
    "actuator_biasprm": 4,
    "actuator_gainprm": 4,
    "body_gravcomp": 4,
    "body_invweight0": 8,
    "body_subtreemass": 4,
    "dof_armature": 4,
    "dof_invweight0": 4,
    "tendon_invweight0": 4,
    "tendon_length0": 4,
}

_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_GIT_OBJECT_RE = re.compile(r"^[0-9a-f]{40,64}$")
_GPU_PSTATE_RE = re.compile(r"^P[0-9]+$")


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
class DrPerformanceCaseSpec:
    """One process-isolated case in the canonical 300-worker matrix."""

    ordinal: int
    case_id: str
    lane: str
    profile_id: str
    tier: str
    batch_size: int
    reset_density: float | None
    repeat_index: int
    seed: int


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

    @property
    def worker_count(self) -> int:
        return self.reset_worker_count + self.env_worker_count + self.train_worker_count

    def profile(self, profile_id: str) -> DrPerformanceProfile:
        for profile in self.profiles:
            if profile.profile_id == profile_id:
                return profile
        raise KeyError(f"unknown mjwarp DR profile {profile_id!r}")


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


def _density_id(value: float) -> str:
    return f"{value:.4f}".replace(".", "p")


def expected_mjwarp_dr_performance_cases(
    plan: MjwarpDrPerformancePlan,
) -> tuple[DrPerformanceCaseSpec, ...]:
    """Expand the frozen matrix in its evidence-grade process order."""

    measurement = cast(Mapping[str, Any], plan.data["measurement"])
    repeats = int(measurement["process_repeats"])
    seeds = tuple(int(value) for value in cast(list[Any], measurement["seeds"]))
    if len(seeds) != repeats:
        raise MjwarpDrPerformanceContractError(
            plan.source_path,
            ("measurement.seeds must map one-to-one to process_repeats",),
        )
    specs: list[DrPerformanceCaseSpec] = []

    def append(
        *,
        lane: str,
        profile_id: str,
        batch_size: int,
        density: float | None,
        repeat_index: int,
    ) -> None:
        profile = plan.profile(profile_id)
        density_token = "" if density is None else f"-d{_density_id(density)}"
        case_id = f"{lane}-b{batch_size}{density_token}-{profile_id}-r{repeat_index}"
        specs.append(
            DrPerformanceCaseSpec(
                ordinal=len(specs),
                case_id=case_id,
                lane=lane,
                profile_id=profile_id,
                tier=profile.tier,
                batch_size=batch_size,
                reset_density=density,
                repeat_index=repeat_index,
                seed=seeds[repeat_index],
            )
        )

    reset = cast(Mapping[str, Any], measurement["reset"])
    for batch_size in cast(list[int], reset["batch_sizes"]):
        for density in cast(list[float], reset["densities"]):
            for repeat_index in range(repeats):
                for profile_id in cast(list[str], reset["profiles"]):
                    append(
                        lane="reset",
                        profile_id=profile_id,
                        batch_size=int(batch_size),
                        density=float(density),
                        repeat_index=repeat_index,
                    )

    env = cast(Mapping[str, Any], measurement["env"])
    for batch_size in cast(list[int], env["batch_sizes"]):
        for repeat_index in range(repeats):
            for profile_id in cast(list[str], env["profiles"]):
                append(
                    lane="env",
                    profile_id=profile_id,
                    batch_size=int(batch_size),
                    density=None,
                    repeat_index=repeat_index,
                )

    train = cast(Mapping[str, Any], measurement["train"])
    for repeat_index in range(repeats):
        for profile_id in cast(list[str], train["profiles"]):
            append(
                lane="train",
                profile_id=profile_id,
                batch_size=int(train["batch_size"]),
                density=None,
                repeat_index=repeat_index,
            )
    if len(specs) != plan.worker_count or len({spec.case_id for spec in specs}) != len(specs):
        raise MjwarpDrPerformanceContractError(
            plan.source_path,
            ("expanded case matrix is incomplete or contains duplicate IDs",),
        )
    return tuple(specs)


def _artifact_mapping(value: object, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path}: expected a mapping")
    return cast(Mapping[str, Any], value)


def _artifact_list(value: object, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{path}: expected a list")
    return cast(list[Any], value)


def _artifact_integer(value: object, path: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{path}: expected an integer >= {minimum}")
    return int(value)


def _artifact_number(value: object, path: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path}: expected a finite number")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise ValueError(f"{path}: expected a finite number >= {minimum}")
    return result


def _artifact_string(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path}: expected a non-empty string")
    return value


def _artifact_sha(value: object, path: str) -> str:
    result = _artifact_string(value, path)
    payload = result.removeprefix("sha256:")
    if (
        not result.startswith("sha256:")
        or len(payload) != 64
        or any(character not in "0123456789abcdef" for character in payload)
    ):
        raise ValueError(f"{path}: expected sha256:<64 lowercase hex>")
    return result


def dependency_version_satisfies(version: str, constraint: str) -> bool:
    """Return whether an installed version satisfies one frozen dependency constraint."""

    normalized = constraint.strip()
    if not normalized:
        raise ValueError("dependency constraint must be non-empty")
    if normalized[0] not in "<>=!~":
        normalized = f"=={normalized}"
    try:
        return Version(version) in SpecifierSet(normalized)
    except (InvalidSpecifier, InvalidVersion) as exc:
        raise ValueError(
            f"invalid dependency version/constraint pair {version!r} / {constraint!r}: {exc}"
        ) from exc


def _exact_artifact_keys(
    value: Mapping[str, Any], expected: set[str], path: str, errors: list[str]
) -> None:
    missing = sorted(expected.difference(value))
    unknown = sorted(set(value).difference(expected))
    if missing:
        errors.append(f"{path}: missing keys {missing!r}")
    if unknown:
        errors.append(f"{path}: unknown keys {unknown!r}")


def _finite_vector(value: object, path: str, *, count: int | None = None) -> list[float]:
    raw = _artifact_list(value, path)
    result = [_artifact_number(item, f"{path}[{index}]") for index, item in enumerate(raw)]
    if count is not None and len(result) != count:
        raise ValueError(f"{path}: expected {count} samples, got {len(result)}")
    if not result:
        raise ValueError(f"{path}: expected at least one sample")
    return result


def _counter_delta(before: Mapping[str, Any], after: Mapping[str, Any], key: str, path: str) -> int:
    start = _artifact_integer(before.get(key), f"{path}.before.{key}")
    end = _artifact_integer(after.get(key), f"{path}.after.{key}")
    if end < start:
        raise ValueError(f"{path}.{key}: cumulative counter regressed")
    return end - start


def _population_cv(values: Sequence[float], path: str) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size < 2 or not np.all(np.isfinite(array)):
        raise ValueError(f"{path}: population CV requires at least two finite values")
    mean = float(array.mean())
    if mean <= 0.0:
        raise ValueError(f"{path}: population CV requires a positive mean")
    return float(array.std(ddof=0) / mean)


def summarize_memory_windows(
    value: object,
    *,
    path: str,
    expected_windows: int = 4,
    expected_samples: int | None = None,
) -> dict[str, Any]:
    """Recompute steady-state memory trends from every raw window sample."""

    windows = _artifact_list(value, path)
    if len(windows) != expected_windows:
        raise ValueError(f"{path}: expected {expected_windows} windows, got {len(windows)}")
    rss_medians: list[float] = []
    allocated_medians: list[float] = []
    reserved_medians: list[float] = []
    for index, item in enumerate(windows):
        window_path = f"{path}[{index}]"
        window = _artifact_mapping(item, window_path)
        expected_keys = {
            "window_index",
            "rss_samples_bytes",
            "cuda_allocated_samples_bytes",
            "cuda_reserved_samples_bytes",
        }
        if set(window) != expected_keys:
            raise ValueError(f"{window_path}: memory window keys differ from v1")
        if window.get("window_index") != index:
            raise ValueError(f"{window_path}.window_index: windows must be dense and ordered")
        rss = _finite_vector(
            window.get("rss_samples_bytes"),
            f"{window_path}.rss_samples_bytes",
            count=expected_samples,
        )
        allocated = _finite_vector(
            window.get("cuda_allocated_samples_bytes"),
            f"{window_path}.cuda_allocated_samples_bytes",
            count=expected_samples,
        )
        reserved = _finite_vector(
            window.get("cuda_reserved_samples_bytes"),
            f"{window_path}.cuda_reserved_samples_bytes",
            count=expected_samples,
        )
        rss_medians.append(float(statistics.median(rss)))
        allocated_medians.append(float(statistics.median(allocated)))
        reserved_medians.append(float(statistics.median(reserved)))

    x = np.arange(len(rss_medians), dtype=np.float64)
    centered = x - float(x.mean())
    slope = float(
        np.dot(centered, np.asarray(rss_medians) - float(np.mean(rss_medians)))
        / np.dot(centered, centered)
    )
    return {
        "window_count": len(windows),
        "rss_window_medians_bytes": rss_medians,
        "cuda_allocated_window_medians_bytes": allocated_medians,
        "cuda_reserved_window_medians_bytes": reserved_medians,
        "host_rss_positive_slope_bytes_per_window": max(0.0, slope),
        "host_rss_last_minus_first_median_bytes": max(0.0, rss_medians[-1] - rss_medians[0]),
        "cuda_allocated_positive_growth_bytes": max(
            0.0, max(allocated_medians) - allocated_medians[0]
        ),
        "cuda_reserved_positive_growth_bytes": max(
            0.0, max(reserved_medians) - reserved_medians[0]
        ),
    }


def _decode_reset_masks(
    value: object,
    *,
    spec: DrPerformanceCaseSpec,
    sample_count: int,
) -> np.ndarray:
    path = f"{spec.case_id}.raw.reset_masks"
    payload = _artifact_mapping(value, path)
    if set(payload) != {"encoding", "shape", "data"}:
        raise ValueError(f"{path}: reset mask payload keys differ from v1")
    if payload.get("encoding") != "numpy-packbits-base64-little-v1":
        raise ValueError(f"{path}.encoding: unsupported reset mask encoding")
    if payload.get("shape") != [sample_count, spec.batch_size]:
        raise ValueError(f"{path}.shape: differs from the measured reset matrix")
    encoded = _artifact_string(payload.get("data"), f"{path}.data")
    try:
        packed = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError(f"{path}.data: invalid base64 payload") from exc
    packed_width = (spec.batch_size + 7) // 8
    if len(packed) != sample_count * packed_width:
        raise ValueError(f"{path}.data: packed byte count differs from declared shape")
    packed_array = np.frombuffer(packed, dtype=np.uint8).reshape(sample_count, packed_width)
    return np.unpackbits(packed_array, axis=1, bitorder="little")[:, : spec.batch_size].astype(
        np.bool_, copy=False
    )


def _profiler_counts(value: object, path: str) -> dict[str, int]:
    profiler = _artifact_mapping(value, path)
    if set(profiler) != {
        "scope_name",
        "coverage_lanes",
        "events",
        "runtime_delta",
        "event_traffic_delta",
    }:
        raise ValueError(f"{path}: profiler payload keys differ from v1")
    _artifact_string(profiler.get("scope_name"), f"{path}.scope_name")
    lanes = _artifact_list(profiler.get("coverage_lanes"), f"{path}.coverage_lanes")
    if any(item not in {"reset", "env", "train"} for item in lanes) or len(set(lanes)) != len(
        lanes
    ):
        raise ValueError(f"{path}.coverage_lanes: invalid or duplicate lanes")
    events = _artifact_list(profiler.get("events"), f"{path}.events")
    counts = {
        "host_to_device_transfers": 0,
        "device_to_host_transfers": 0,
        "global_synchronizations": 0,
        "numeric_allocations": 0,
    }
    for index, item in enumerate(events):
        event = _artifact_mapping(item, f"{path}.events[{index}]")
        if set(event) != {"name", "category", "timestamp_us", "duration_us", "args"}:
            raise ValueError(f"{path}.events[{index}]: event inventory keys differ from v1")
        name = _artifact_string(event.get("name"), f"{path}.events[{index}].name")
        _artifact_string(event.get("category"), f"{path}.events[{index}].category")
        _artifact_number(event.get("timestamp_us"), f"{path}.events[{index}].timestamp_us")
        _artifact_number(event.get("duration_us"), f"{path}.events[{index}].duration_us")
        args = event.get("args")
        if not isinstance(args, (Mapping, list, str, int, float, bool)) and args is not None:
            raise ValueError(f"{path}.events[{index}].args: value is not JSON-compatible")
        haystack = f"{name} {json.dumps(args, sort_keys=True)}".lower()
        if any(token in haystack for token in ("htod", "host to device", "host -> device")):
            counts["host_to_device_transfers"] += 1
        if any(token in haystack for token in ("dtoh", "device to host", "device -> host")):
            counts["device_to_host_transfers"] += 1
        if "cudadevicesynchronize" in haystack:
            counts["global_synchronizations"] += 1
        if any(
            token in haystack
            for token in (
                "aten::empty",
                "aten::empty_like",
                "aten::new_empty",
                "cudamalloc",
                "cumemalloc",
            )
        ):
            counts["numeric_allocations"] += 1

    runtime_delta = _artifact_mapping(profiler.get("runtime_delta"), f"{path}.runtime_delta")
    for key in (
        "host_to_device_transfers",
        "device_to_host_transfers",
        "global_synchronizations",
        "backend_allocations",
    ):
        _artifact_integer(runtime_delta.get(key), f"{path}.runtime_delta.{key}")
    event_delta = _artifact_mapping(
        profiler.get("event_traffic_delta"), f"{path}.event_traffic_delta"
    )
    for term, counters_raw in event_delta.items():
        _artifact_string(term, f"{path}.event_traffic_delta key")
        counters = _artifact_mapping(counters_raw, f"{path}.event_traffic_delta.{term}")
        for key in (
            "host_to_device_transfers",
            "device_to_host_transfers",
            "global_synchronizations",
            "sample_allocations",
        ):
            _artifact_integer(counters.get(key), f"{path}.event_traffic_delta.{term}.{key}")
    return counts


def _scalar_values(
    scalars: Mapping[str, Any], tag: str, *, iterations: int, path: str
) -> list[float]:
    points = _artifact_list(scalars.get(tag), f"{path}.{tag}")
    if len(points) != iterations:
        raise ValueError(f"{path}.{tag}: expected {iterations} points")
    values: list[float] = []
    for index, item in enumerate(points):
        point = _artifact_mapping(item, f"{path}.{tag}[{index}]")
        if set(point) != {"step", "wall_time", "value"}:
            raise ValueError(f"{path}.{tag}[{index}]: scalar point keys differ from v1")
        if point.get("step") != index:
            raise ValueError(f"{path}.{tag}: steps must be dense and ordered")
        _artifact_number(point.get("wall_time"), f"{path}.{tag}[{index}].wall_time")
        values.append(
            _artifact_number(
                point.get("value"),
                f"{path}.{tag}[{index}].value",
                minimum=-float("inf"),
            )
        )
    return values


def summarize_mjwarp_dr_performance_case(
    raw: Mapping[str, Any],
    *,
    spec: DrPerformanceCaseSpec,
    plan: MjwarpDrPerformancePlan,
) -> dict[str, Any]:
    """Build one case summary strictly from raw samples."""

    measurement = cast(Mapping[str, Any], plan.data["measurement"])
    if spec.lane == "reset":
        lane = cast(Mapping[str, Any], measurement["reset"])
        measured = int(lane["measured_barriers"])
        expected_keys = {
            "resolved_config",
            "resolved_config_sha256",
            "phase_samples_ms",
            "reset_masks",
            "memory_windows",
            "diagnostics",
            "timing_lifecycle",
            "profiler",
        }
        if set(raw) != expected_keys:
            raise ValueError(f"{spec.case_id}.raw: reset raw keys differ from v1")
        phases = _artifact_mapping(raw.get("phase_samples_ms"), f"{spec.case_id}.phases")
        if tuple(phases) != RESET_PHASES:
            raise ValueError(f"{spec.case_id}: reset phase order differs from v1")
        phase_stats = {
            phase: numeric_stats(
                _finite_vector(
                    phases[phase],
                    f"{spec.case_id}.raw.phase_samples_ms.{phase}",
                    count=measured,
                )
            )
            for phase in RESET_PHASES
        }
        masks = _decode_reset_masks(raw.get("reset_masks"), spec=spec, sample_count=measured)
        assert spec.reset_density is not None
        expected_rows = int(round(spec.batch_size * spec.reset_density))
        row_counts = np.count_nonzero(masks, axis=1)
        if not np.all(row_counts == expected_rows):
            raise ValueError(
                f"{spec.case_id}: actual reset masks differ from the frozen density schedule"
            )
        memory = summarize_memory_windows(
            raw.get("memory_windows"),
            path=f"{spec.case_id}.raw.memory_windows",
            expected_windows=int(lane["memory_windows"]),
            expected_samples=int(lane["samples_per_memory_window"]),
        )
        profiler = raw.get("profiler")
        return {
            "phase_metrics": phase_stats,
            "reset_masks": {
                "sample_count": measured,
                "rows_per_barrier": expected_rows,
                "total_reset_rows": int(row_counts.sum()),
            },
            "memory": memory,
            "profiler_counts": None
            if profiler is None
            else _profiler_counts(profiler, f"{spec.case_id}.raw.profiler"),
        }

    if spec.lane == "env":
        lane = cast(Mapping[str, Any], measurement["env"])
        measured = int(lane["measured_steps"])
        expected_keys = {
            "resolved_config",
            "resolved_config_sha256",
            "env_step_samples_ms",
            "memory_windows",
            "diagnostics",
            "timing_lifecycle",
            "profiler",
        }
        if set(raw) != expected_keys:
            raise ValueError(f"{spec.case_id}.raw: env raw keys differ from v1")
        samples = _finite_vector(
            raw.get("env_step_samples_ms"),
            f"{spec.case_id}.raw.env_step_samples_ms",
            count=measured,
        )
        stats = numeric_stats(samples)
        memory = summarize_memory_windows(
            raw.get("memory_windows"),
            path=f"{spec.case_id}.raw.memory_windows",
            expected_windows=int(lane["memory_windows"]),
            expected_samples=int(lane["samples_per_memory_window"]),
        )
        profiler = raw.get("profiler")
        return {
            "phase_metrics": {"env_step": stats},
            "throughput_env_steps_per_sec": spec.batch_size * 1000.0 / float(stats["mean"]),
            "memory": memory,
            "profiler_counts": None
            if profiler is None
            else _profiler_counts(profiler, f"{spec.case_id}.raw.profiler"),
        }

    if spec.lane != "train":
        raise ValueError(f"{spec.case_id}: unknown lane {spec.lane!r}")
    lane = cast(Mapping[str, Any], measurement["train"])
    iterations = int(lane["iterations"])
    warmup = int(lane["warmup_iterations"])
    expected_keys = {
        "scalars",
        "memory_windows",
        "run_config",
        "run_config_sha256",
        "run_summary",
    }
    if set(raw) != expected_keys:
        raise ValueError(f"{spec.case_id}.raw: train raw keys differ from v1")
    scalars = _artifact_mapping(raw.get("scalars"), f"{spec.case_id}.raw.scalars")
    if set(scalars) != set(TRAIN_SCALAR_TAGS):
        raise ValueError(f"{spec.case_id}.raw.scalars: tags differ from v1")
    values = {
        tag: _scalar_values(
            scalars,
            tag,
            iterations=iterations,
            path=f"{spec.case_id}.raw.scalars",
        )
        for tag in TRAIN_SCALAR_TAGS
    }
    fps = values["Perf/total_fps"][warmup:]
    collection = [value * 1000.0 for value in values["Perf/collection_time"][warmup:]]
    learning = [value * 1000.0 for value in values["Perf/learning_time"][warmup:]]
    iteration = [left + right for left, right in zip(collection, learning, strict=True)]
    if not fps or not iteration:
        raise ValueError(f"{spec.case_id}: training has no post-warmup samples")
    memory = summarize_memory_windows(
        raw.get("memory_windows"),
        path=f"{spec.case_id}.raw.memory_windows",
        expected_windows=4,
        expected_samples=(iterations - warmup) // 4,
    )
    return {
        "phase_metrics": {
            "ppo_iteration": numeric_stats(iteration),
            "collection": numeric_stats(collection),
            "learning": numeric_stats(learning),
        },
        "throughput_env_steps_per_sec": numeric_stats(fps),
        "memory": memory,
        "completed_iterations": iterations,
    }


def _nested_mapping(value: Mapping[str, Any], *keys: str) -> Mapping[str, Any]:
    current: Mapping[str, Any] = value
    path: list[str] = []
    for key in keys:
        path.append(key)
        current = _artifact_mapping(current.get(key), ".".join(path))
    return current


def _validate_resolved_config(
    raw: Mapping[str, Any], spec: DrPerformanceCaseSpec, profile: DrPerformanceProfile
) -> None:
    if spec.lane == "train":
        config_receipt = _artifact_mapping(raw.get("run_config"), f"{spec.case_id}.raw.run_config")
        if raw.get("run_config_sha256") != canonical_sha256(config_receipt):
            raise ValueError(f"{spec.case_id}: run_config_sha256 differs from raw run_config")
        config = _artifact_mapping(config_receipt.get("config"), f"{spec.case_id}.config")
    else:
        config = _artifact_mapping(
            raw.get("resolved_config"), f"{spec.case_id}.raw.resolved_config"
        )
        if raw.get("resolved_config_sha256") != canonical_sha256(config):
            raise ValueError(f"{spec.case_id}: resolved_config_sha256 differs from resolved config")
    training = _nested_mapping(config, "training")
    algo = _nested_mapping(config, "algo")
    domain_rand = _nested_mapping(config, "env", "domain_rand")
    if training.get("sim_backend") != "mjwarp" or training.get("execution_profile") != (
        "device_resident"
    ):
        raise ValueError(f"{spec.case_id}: config does not select the mjwarp device owner")
    if algo.get("num_envs") != spec.batch_size or algo.get("seed") != spec.seed:
        raise ValueError(f"{spec.case_id}: config batch size or seed differs from case identity")
    expected_flags = {
        "randomize_kp": profile.profile_id == "tier_b_pd",
        "randomize_kd": profile.profile_id == "tier_b_pd",
        "randomize_dof_armature": profile.profile_id in {"tier_c_armature", "tier_c_mixed"},
        "randomize_body_gravity_compensation": profile.profile_id == "tier_c_mixed",
    }
    for key, expected in expected_flags.items():
        if domain_rand.get(key) is not expected:
            raise ValueError(f"{spec.case_id}: env.domain_rand.{key} differs from profile")


def _performance_graph(value: Mapping[str, Any], path: str) -> Mapping[str, Any]:
    graph = _artifact_mapping(value.get("graph"), f"{path}.graph")
    if graph.get("backend_type") != "mjwarp" or graph.get("execution_mode") != "cuda_graph":
        raise ValueError(f"{path}.graph: expected complete mjwarp CUDA graph diagnostics")
    if graph.get("instrumentation_complete") is not True:
        raise ValueError(f"{path}.graph: instrumentation is incomplete")
    return graph


def _validate_materialization(
    performance: Mapping[str, Any],
    *,
    profile: DrPerformanceProfile,
    batch_size: int,
    path: str,
) -> None:
    if performance.get("backend_type") != "mjwarp":
        raise ValueError(f"{path}.backend_type: expected mjwarp")
    if performance.get("model_targets") != list(profile.model_targets):
        raise ValueError(f"{path}.model_targets: differs from frozen profile")
    if performance.get("direct_fields") != list(profile.direct_fields):
        raise ValueError(f"{path}.direct_fields: differs from frozen profile")
    if performance.get("derived_fields") != list(profile.derived_fields):
        raise ValueError(f"{path}.derived_fields: differs from frozen profile")
    if performance.get("recompute_kind") != profile.strongest_recompute:
        raise ValueError(f"{path}.recompute_kind: differs from frozen profile")
    if performance.get("instrumentation_complete") is not True:
        raise ValueError(f"{path}: mutation instrumentation is incomplete")
    lifecycle = _artifact_mapping(performance.get("lifecycle"), f"{path}.lifecycle")
    if lifecycle.get("instrumentation_complete") is not True:
        raise ValueError(f"{path}.lifecycle: instrumentation is incomplete")
    graph = _performance_graph(performance, path)
    materialization_raw = performance.get("materialization")
    if not profile.model_targets:
        if materialization_raw is not None:
            raise ValueError(f"{path}.materialization: disabled profile must be state-only")
        return
    materialization = _artifact_mapping(materialization_raw, f"{path}.materialization")
    if materialization.get("num_worlds") != batch_size:
        raise ValueError(f"{path}.materialization.num_worlds: differs from case batch")
    fields = _artifact_list(materialization.get("fields"), f"{path}.materialization.fields")
    expected_names = list((*profile.direct_fields, *profile.derived_fields))
    expected_names.sort()
    if [item.get("field_name") if isinstance(item, Mapping) else None for item in fields] != (
        expected_names
    ):
        raise ValueError(f"{path}.materialization.fields: names differ from frozen profile")
    field_sum = 0
    direct_baseline = 0
    for index, item in enumerate(fields):
        field = _artifact_mapping(item, f"{path}.materialization.fields[{index}]")
        name = _artifact_string(field.get("field_name"), f"{path}.field[{index}].name")
        expected_role = "direct" if name in profile.direct_fields else "derived"
        if field.get("role") != expected_role:
            raise ValueError(f"{path}.field.{name}.role: differs from recompute receipt")
        shape = _artifact_list(field.get("materialized_shape"), f"{path}.field.{name}.shape")
        if (
            not shape
            or shape[0] != batch_size
            or any(isinstance(dim, bool) or not isinstance(dim, int) or dim < 0 for dim in shape)
        ):
            raise ValueError(f"{path}.field.{name}.shape: invalid materialized shape")
        model_bytes = _artifact_integer(field.get("model_bytes"), f"{path}.field.{name}.bytes")
        try:
            itemsize = _MODEL_FIELD_ITEMSIZE_BYTES[name]
        except KeyError as exc:
            raise ValueError(f"{path}.field.{name}: unknown storage item size") from exc
        replaced = field.get("replaced")
        if not isinstance(replaced, bool):
            raise ValueError(f"{path}.field.{name}.replaced: expected bool")
        if replaced:
            if model_bytes != math.prod(cast(list[int], shape)) * itemsize:
                raise ValueError(f"{path}.field.{name}.bytes: not exact typed storage")
            field_sum += model_bytes
            address = _artifact_integer(
                field.get("materialized_address"), f"{path}.field.{name}.address"
            )
            if model_bytes > 0 and address <= 0:
                raise ValueError(f"{path}.field.{name}.address: replaced storage has no address")
            if model_bytes == 0 and address != 0:
                raise ValueError(
                    f"{path}.field.{name}.address: empty storage must use zero address"
                )
        compiled_shape = _artifact_list(
            field.get("compiled_default_shape"), f"{path}.field.{name}.compiled_shape"
        )
        if not compiled_shape or any(
            isinstance(dim, bool) or not isinstance(dim, int) or dim < 0 for dim in compiled_shape
        ):
            raise ValueError(f"{path}.field.{name}.compiled_shape: invalid")
        if expected_role == "direct":
            direct_baseline += math.prod(cast(list[int], compiled_shape)) * 4
    if materialization.get("expanded_model_bytes") != field_sum:
        raise ValueError(f"{path}.materialization: field byte sum is not exact")
    if materialization.get("baseline_bytes") != direct_baseline:
        raise ValueError(f"{path}.materialization: baseline byte sum is not exact")
    if materialization.get("storage_generation") != graph.get(
        "storage_generation"
    ) or materialization.get("storage_fingerprint") != graph.get("storage_fingerprint"):
        raise ValueError(f"{path}.materialization: storage identity differs from graph")


_RUNTIME_TRAFFIC_ZERO_KEYS = (
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
_RUNTIME_TRAFFIC_SEMANTIC_KEYS = (
    "policy_steps",
    "step_barriers",
    "reset_barriers",
    "state_materializations",
)
_EVENT_TRAFFIC_KEYS = (
    "host_to_device_transfers",
    "device_to_host_transfers",
    "global_synchronizations",
    "sample_allocations",
)


def _validate_runtime_traffic(value: Mapping[str, Any], path: str) -> None:
    if value.get("instrumentation_complete") is not True:
        raise ValueError(f"{path}: runtime traffic instrumentation is incomplete")
    for key in (*_RUNTIME_TRAFFIC_ZERO_KEYS, *_RUNTIME_TRAFFIC_SEMANTIC_KEYS):
        _artifact_integer(value.get(key), f"{path}.{key}")


def _validate_event_traffic(value: object, *, profile_id: str, path: str) -> None:
    events = _artifact_mapping(value, path)
    if tuple(events) != _PROFILE_EVENT_TERMS[profile_id]:
        raise ValueError(f"{path}: Event terms differ from the selected profile")
    for term, counters_raw in events.items():
        counters = _artifact_mapping(counters_raw, f"{path}.{term}")
        if set(counters) != set(_EVENT_TRAFFIC_KEYS):
            raise ValueError(f"{path}.{term}: Event traffic keys differ from v1")
        for key in _EVENT_TRAFFIC_KEYS:
            _artifact_integer(counters.get(key), f"{path}.{term}.{key}")


def _validate_stability_pair(
    before: Mapping[str, Any], after: Mapping[str, Any], path: str
) -> None:
    for label, value in (("before", before), ("after", after)):
        if value.get("instrumentation_complete") is not True:
            raise ValueError(f"{path}.{label}: stability instrumentation is incomplete")
        for key in (
            "warm_numeric_allocations",
            "address_churn",
            "observations",
            "output_epoch",
            "control_epoch",
            "reset_epoch",
        ):
            _artifact_integer(value.get(key), f"{path}.{label}.{key}")
        for key in ("buffers", "state_buffers", "state_epochs"):
            _artifact_list(value.get(key), f"{path}.{label}.{key}")
        traffic = _artifact_mapping(value.get("traffic"), f"{path}.{label}.traffic")
        _validate_runtime_traffic(traffic, f"{path}.{label}.traffic")
        graph = _artifact_mapping(value.get("graph"), f"{path}.{label}.graph")
        _performance_graph({"graph": graph}, f"{path}.{label}")


def _validate_profiler(
    raw: Mapping[str, Any], *, spec: DrPerformanceCaseSpec, summary: Mapping[str, Any]
) -> None:
    expected = (
        spec.repeat_index == 0
        and spec.batch_size == 1024
        and ((spec.lane == "reset" and spec.reset_density == 0.1) or spec.lane == "env")
    )
    profiler = raw.get("profiler") if spec.lane in {"reset", "env"} else None
    if expected != (profiler is not None):
        raise ValueError(f"{spec.case_id}: representative profiler presence differs from v1")
    if profiler is None:
        return
    profiler_map = _artifact_mapping(profiler, f"{spec.case_id}.raw.profiler")
    expected_lanes = ["reset"] if spec.lane == "reset" else ["env", "train"]
    if profiler_map.get("coverage_lanes") != expected_lanes:
        raise ValueError(f"{spec.case_id}.raw.profiler.coverage_lanes: differs from v1")
    counts = _artifact_mapping(summary.get("profiler_counts"), f"{spec.case_id}.profiler_counts")
    for key, value in counts.items():
        _artifact_integer(value, f"{spec.case_id}.profiler_counts.{key}")
    runtime_delta = _artifact_mapping(
        profiler_map.get("runtime_delta"), f"{spec.case_id}.raw.profiler.runtime_delta"
    )
    for key in (
        "host_to_device_transfers",
        "device_to_host_transfers",
        "global_synchronizations",
        "backend_allocations",
    ):
        _artifact_integer(runtime_delta.get(key), f"{spec.case_id}.profiler.{key}")
    _validate_event_traffic(
        profiler_map.get("event_traffic_delta"),
        profile_id=spec.profile_id,
        path=f"{spec.case_id}.raw.profiler.event_traffic_delta",
    )


def _validate_direct_diagnostics(
    value: object,
    *,
    spec: DrPerformanceCaseSpec,
    profile: DrPerformanceProfile,
    measured: int,
) -> None:
    path = f"{spec.case_id}.raw.diagnostics"
    diagnostics = _artifact_mapping(value, path)
    expected_keys = {
        "performance_before",
        "performance_after",
        "stability_before",
        "stability_after",
        "runtime_traffic_before",
        "runtime_traffic_after",
        "event_traffic_before",
        "event_traffic_after",
        "wrapper_traffic_before",
        "wrapper_traffic_after",
    }
    if set(diagnostics) != expected_keys:
        raise ValueError(f"{path}: diagnostic keys differ from v1")
    performance_before = _artifact_mapping(
        diagnostics.get("performance_before"), f"{path}.performance_before"
    )
    performance_after = _artifact_mapping(
        diagnostics.get("performance_after"), f"{path}.performance_after"
    )
    _validate_materialization(
        performance_before,
        profile=profile,
        batch_size=spec.batch_size,
        path=f"{path}.performance_before",
    )
    _validate_materialization(
        performance_after,
        profile=profile,
        batch_size=spec.batch_size,
        path=f"{path}.performance_after",
    )
    for key in (
        "backend_instance_id",
        "mutation_plan_fingerprint",
        "model_targets",
        "direct_fields",
        "derived_fields",
        "recompute_kind",
    ):
        if performance_before.get(key) != performance_after.get(key):
            raise ValueError(f"{path}.performance.{key}: changed in warmed window")
    graph_before = _performance_graph(performance_before, f"{path}.performance_before")
    graph_after = _performance_graph(performance_after, f"{path}.performance_after")
    for key in (
        "capture_count",
        "launch_count",
        "recapture_count",
        "stale_rejection_count",
        "eager_fallback_count",
    ):
        _counter_delta(graph_before, graph_after, key, f"{path}.graph")

    lifecycle_before = _artifact_mapping(
        performance_before.get("lifecycle"), f"{path}.lifecycle_before"
    )
    lifecycle_after = _artifact_mapping(
        performance_after.get("lifecycle"), f"{path}.lifecycle_after"
    )
    for key in (
        "runtime_barriers",
        "step_graph_launches",
        "reset_graph_launches",
        "forward_graph_launches",
        "state_refreshes",
    ):
        _counter_delta(lifecycle_before, lifecycle_after, key, f"{path}.lifecycle")
    for key in ("recompute_launch_count", "recompute_capture_count"):
        _counter_delta(performance_before, performance_after, key, f"{path}.recompute")

    runtime_before = _artifact_mapping(
        diagnostics.get("runtime_traffic_before"), f"{path}.runtime_traffic_before"
    )
    runtime_after = _artifact_mapping(
        diagnostics.get("runtime_traffic_after"), f"{path}.runtime_traffic_after"
    )
    _validate_runtime_traffic(runtime_before, f"{path}.runtime_traffic_before")
    _validate_runtime_traffic(runtime_after, f"{path}.runtime_traffic_after")
    for key in (*_RUNTIME_TRAFFIC_ZERO_KEYS, *_RUNTIME_TRAFFIC_SEMANTIC_KEYS):
        _counter_delta(runtime_before, runtime_after, key, f"{path}.runtime_traffic")
    _validate_event_traffic(
        diagnostics.get("event_traffic_before"),
        profile_id=spec.profile_id,
        path=f"{path}.event_traffic_before",
    )
    _validate_event_traffic(
        diagnostics.get("event_traffic_after"),
        profile_id=spec.profile_id,
        path=f"{path}.event_traffic_after",
    )

    wrapper_before = _artifact_mapping(
        diagnostics.get("wrapper_traffic_before"), f"{path}.wrapper_traffic_before"
    )
    wrapper_after = _artifact_mapping(
        diagnostics.get("wrapper_traffic_after"), f"{path}.wrapper_traffic_after"
    )
    for key in (
        "action_publications",
        "action_device_to_device_bytes",
        "observation_snapshots",
        "observation_device_to_device_bytes",
        "finite_metric_materializations",
        "finite_metric_device_to_host_bytes",
    ):
        _counter_delta(wrapper_before, wrapper_after, key, f"{path}.wrapper_traffic")
    stability_before = _artifact_mapping(
        diagnostics.get("stability_before"), f"{path}.stability_before"
    )
    stability_after = _artifact_mapping(
        diagnostics.get("stability_after"), f"{path}.stability_after"
    )
    _validate_stability_pair(stability_before, stability_after, f"{path}.stability")


def _validate_timing_lifecycle(
    raw: Mapping[str, Any], *, spec: DrPerformanceCaseSpec, plan: MjwarpDrPerformancePlan
) -> None:
    measurement = cast(Mapping[str, Any], plan.data["measurement"])
    timing = _artifact_mapping(raw.get("timing_lifecycle"), f"{spec.case_id}.raw.timing_lifecycle")
    if spec.lane == "reset":
        measured = int(cast(Mapping[str, Any], measurement["reset"])["measured_barriers"])
        required = {
            "backend_type",
            "backend_instance_id",
            "placement",
            "capacity",
            "samples",
            "events_preallocated",
            "priming_synchronizations",
            "materialization_synchronizations",
        }
        if set(timing) != required:
            raise ValueError(f"{spec.case_id}: reset timing lifecycle keys differ from v1")
        if timing.get("backend_type") != "mjwarp" or timing.get("capacity") != measured:
            raise ValueError(f"{spec.case_id}: reset timing owner/capacity differs")
        if timing.get("events_preallocated") != measured * len(RESET_PHASES) * 2:
            raise ValueError(f"{spec.case_id}: reset timing events were not fully preallocated")
        if timing.get("priming_synchronizations") != 1:
            raise ValueError(f"{spec.case_id}: reset timing must be primed once before measurement")
        if timing.get("materialization_synchronizations") != 1:
            raise ValueError(
                f"{spec.case_id}: reset timing must materialize once after measurement"
            )
        samples = _artifact_list(timing.get("samples"), f"{spec.case_id}.timing.samples")
        if len(samples) != measured:
            raise ValueError(f"{spec.case_id}: reset timing trace is incomplete")
        reconstructed: dict[str, list[float]] = {phase: [] for phase in RESET_PHASES}
        for index, sample_raw in enumerate(samples):
            sample = _artifact_mapping(sample_raw, f"{spec.case_id}.timing.samples[{index}]")
            if sample.get("sample_index") != index:
                raise ValueError(f"{spec.case_id}: reset timing samples are not ordered")
            intervals = _artifact_list(
                sample.get("intervals"), f"{spec.case_id}.timing.samples[{index}].intervals"
            )
            if [
                item.get("phase") if isinstance(item, Mapping) else None for item in intervals
            ] != list(RESET_PHASES):
                raise ValueError(f"{spec.case_id}: reset timing phases are not canonical")
            previous_end = 0.0
            for phase_index, interval_raw in enumerate(intervals):
                interval = _artifact_mapping(
                    interval_raw,
                    f"{spec.case_id}.timing.samples[{index}].intervals[{phase_index}]",
                )
                start = _artifact_number(interval.get("start_ms"), "timing.start_ms")
                end = _artifact_number(interval.get("end_ms"), "timing.end_ms")
                if end < start:
                    raise ValueError(f"{spec.case_id}: reset timing interval is negative")
                phase = RESET_PHASES[phase_index]
                if phase == "reset_barrier":
                    if start != 0.0:
                        raise ValueError(f"{spec.case_id}: reset barrier must start at zero")
                elif start < previous_end:
                    raise ValueError(f"{spec.case_id}: reset phase timestamps overlap out of order")
                previous_end = end if phase != "reset_barrier" else previous_end
                reconstructed[phase].append(end - start)
        raw_phases = _artifact_mapping(
            raw.get("phase_samples_ms"), f"{spec.case_id}.raw.phase_samples_ms"
        )
        if reconstructed != raw_phases:
            raise ValueError(f"{spec.case_id}: phase samples differ from raw timing trace")
        return

    if spec.lane != "env":
        raise ValueError(f"{spec.case_id}: timing lifecycle is only valid for direct lanes")
    measured = int(cast(Mapping[str, Any], measurement["env"])["measured_steps"])
    expected = {
        "capacity": measured,
        "events_preallocated": measured * 2,
        "priming_synchronizations": 1,
        "materialization_synchronizations": 1,
    }
    if timing != expected:
        raise ValueError(f"{spec.case_id}: env CUDA event lifecycle differs from v1")


def _validate_train_diagnostics(
    raw: Mapping[str, Any],
    *,
    spec: DrPerformanceCaseSpec,
    profile: DrPerformanceProfile,
    plan: MjwarpDrPerformancePlan,
) -> None:
    path = f"{spec.case_id}.raw.run_summary"
    summary = _artifact_mapping(raw.get("run_summary"), path)
    measurement = cast(Mapping[str, Any], plan.data["measurement"])
    lane = cast(Mapping[str, Any], measurement["train"])
    iterations = int(lane["iterations"])
    if summary.get("status") != "completed" or summary.get("completed_iterations") != iterations:
        raise ValueError(f"{spec.case_id}: production trainer did not complete 12 iterations")
    for key in (
        "training_wall_time_sec",
        "peak_process_rss_bytes",
        "peak_gpu_memory_allocated_bytes",
        "peak_gpu_memory_reserved_bytes",
    ):
        _artifact_number(summary.get(key), f"{path}.{key}")

    performance = _artifact_mapping(
        summary.get("runtime_performance_diagnostics"),
        f"{path}.runtime_performance_diagnostics",
    )
    performance_before = _artifact_mapping(
        summary.get("runtime_performance_diagnostics_before_training"),
        f"{path}.runtime_performance_diagnostics_before_training",
    )
    _validate_materialization(
        performance_before,
        profile=profile,
        batch_size=spec.batch_size,
        path=f"{path}.runtime_performance_diagnostics_before_training",
    )
    _validate_materialization(
        performance,
        profile=profile,
        batch_size=spec.batch_size,
        path=f"{path}.runtime_performance_diagnostics",
    )
    for key in (
        "backend_instance_id",
        "mutation_plan_fingerprint",
        "model_targets",
        "direct_fields",
        "derived_fields",
        "recompute_kind",
        "materialization",
    ):
        if performance_before.get(key) != performance.get(key):
            raise ValueError(f"{path}.runtime_performance.{key}: changed during training")
    lifecycle = _artifact_mapping(performance.get("lifecycle"), f"{path}.lifecycle")
    for key in (
        "runtime_barriers",
        "step_graph_launches",
        "reset_graph_launches",
        "forward_graph_launches",
        "state_refreshes",
    ):
        _artifact_integer(lifecycle.get(key), f"{path}.lifecycle.{key}")
    for key in ("recompute_launch_count", "recompute_capture_count"):
        _artifact_integer(performance.get(key), f"{path}.{key}")
    graph = _performance_graph(performance, f"{path}.runtime_performance_diagnostics")
    for key in (
        "capture_count",
        "launch_count",
        "recapture_count",
        "stale_rejection_count",
        "eager_fallback_count",
    ):
        _artifact_integer(graph.get(key), f"{path}.graph.{key}")

    traffic = _artifact_mapping(
        summary.get("runtime_traffic_diagnostics"), f"{path}.runtime_traffic_diagnostics"
    )
    _validate_runtime_traffic(traffic, f"{path}.runtime_traffic_diagnostics")
    _validate_event_traffic(
        summary.get("runtime_event_traffic_diagnostics"),
        profile_id=spec.profile_id,
        path=f"{path}.runtime_event_traffic_diagnostics",
    )
    wrapper = _artifact_mapping(
        summary.get("wrapper_traffic_diagnostics"), f"{path}.wrapper_traffic_diagnostics"
    )
    for key in (
        "action_publications",
        "action_device_to_device_bytes",
        "observation_snapshots",
        "observation_device_to_device_bytes",
        "finite_metric_materializations",
        "finite_metric_device_to_host_bytes",
    ):
        _artifact_integer(wrapper.get(key), f"{path}.wrapper_traffic_diagnostics.{key}")
    stability = _artifact_mapping(
        summary.get("runtime_stability_diagnostics"),
        f"{path}.runtime_stability_diagnostics",
    )
    _validate_stability_pair(stability, stability, f"{path}.runtime_stability_diagnostics")
    logging = _artifact_mapping(
        summary.get("logging_traffic_diagnostics"), f"{path}.logging_traffic_diagnostics"
    )
    for key in ("rollout_steps", "metric_materializations", "metric_device_to_host_bytes"):
        _artifact_integer(logging.get(key), f"{path}.logging_traffic_diagnostics.{key}")
    host_memory = _artifact_mapping(
        summary.get("runner_host_memory_diagnostics"),
        f"{path}.runner_host_memory_diagnostics",
    )
    _artifact_integer(
        host_memory.get("gc_collected_objects"), f"{path}.runner_host_memory.gc_collected_objects"
    )
    for key in ("allocator_trim_attempted", "allocator_trimmed"):
        if not isinstance(host_memory.get(key), bool):
            raise ValueError(f"{path}.runner_host_memory.{key}: expected bool")
    iteration_memory = _artifact_list(
        summary.get("iteration_memory_diagnostics"), f"{path}.iteration_memory_diagnostics"
    )
    if len(iteration_memory) != iterations:
        raise ValueError(f"{path}.iteration_memory_diagnostics: expected every iteration")
    for index, sample_raw in enumerate(iteration_memory):
        sample = _artifact_mapping(sample_raw, f"{path}.iteration_memory[{index}]")
        if sample.get("iteration") != index:
            raise ValueError(f"{path}.iteration_memory: samples are not ordered")
        for key in ("rss_bytes", "cuda_allocated_bytes", "cuda_reserved_bytes"):
            _artifact_integer(sample.get(key), f"{path}.iteration_memory[{index}].{key}")


def _validate_process_receipt(
    value: object,
    *,
    spec: DrPerformanceCaseSpec,
    plan: MjwarpDrPerformancePlan,
) -> str:
    path = f"{spec.case_id}.process"
    process = _artifact_mapping(value, path)
    expected_keys = {
        "run_id",
        "pid",
        "started_at",
        "duration_sec",
        "return_code",
        "command",
        "affinity_cpus",
        "env_vars",
        "stdout_sha256",
        "stderr_sha256",
    }
    if set(process) != expected_keys:
        raise ValueError(f"{path}: process receipt keys differ from v1")
    run_id = _artifact_string(process.get("run_id"), f"{path}.run_id")
    _artifact_integer(process.get("pid"), f"{path}.pid", minimum=1)
    _artifact_string(process.get("started_at"), f"{path}.started_at")
    _artifact_number(process.get("duration_sec"), f"{path}.duration_sec")
    if process.get("return_code") != 0:
        raise ValueError(f"{path}: worker process did not complete successfully")
    command = _artifact_list(process.get("command"), f"{path}.command")
    if any(not isinstance(item, str) or not item for item in command):
        raise ValueError(f"{path}.command: every argument must be a non-empty string")
    if spec.lane in {"reset", "env"}:
        if command[:3] != ["uv", "run", "benchmark/mjwarp/benchmark_dr_profiles.py"]:
            raise ValueError(
                f"{path}.command: direct worker did not use the public benchmark route"
            )
        required = {"--worker", "--case-id", spec.case_id, "--worker-output"}
        if not required.issubset(set(cast(list[str], command))):
            raise ValueError(f"{path}.command: direct worker identity is incomplete")
    else:
        if command[:3] != ["uv", "run", "scripts/train_rsl_rl.py"]:
            raise ValueError(f"{path}.command: train worker did not use the public trainer")
        required = {
            "task=g1_walk_flat/mjwarp",
            f"algo.seed={spec.seed}",
            f"algo.num_envs={spec.batch_size}",
            "algo.num_steps_per_env=8",
            "algo.max_iterations=12",
            "algo.capture_performance_diagnostics=true",
            "training.no_play=true",
            "training.logger=tensorboard",
        }
        if not required.issubset(set(cast(list[str], command))):
            raise ValueError(f"{path}.command: train worker differs from frozen owner protocol")
    hardware = cast(Mapping[str, Any], plan.data["hardware"])
    if process.get("affinity_cpus") != hardware.get("affinity_cpus"):
        raise ValueError(f"{path}.affinity_cpus: differs from frozen hardware binding")
    if process.get("env_vars") != hardware.get("environment_variables"):
        raise ValueError(f"{path}.env_vars: differs from frozen thread environment")
    _artifact_sha(process.get("stdout_sha256"), f"{path}.stdout_sha256")
    _artifact_sha(process.get("stderr_sha256"), f"{path}.stderr_sha256")
    return run_id


def _validate_case_and_recompute(
    value: object,
    *,
    spec: DrPerformanceCaseSpec,
    plan: MjwarpDrPerformancePlan,
) -> tuple[dict[str, Any], str]:
    case = _artifact_mapping(value, spec.case_id)
    expected_keys = {
        "case_id",
        "ordinal",
        "lane",
        "profile_id",
        "tier",
        "batch_size",
        "reset_density",
        "repeat_index",
        "seed",
        "process",
        "raw",
        "summary",
    }
    if set(case) != expected_keys:
        raise ValueError(f"{spec.case_id}: case keys differ from v1")
    expected_identity = {
        "case_id": spec.case_id,
        "ordinal": spec.ordinal,
        "lane": spec.lane,
        "profile_id": spec.profile_id,
        "tier": spec.tier,
        "batch_size": spec.batch_size,
        "reset_density": spec.reset_density,
        "repeat_index": spec.repeat_index,
        "seed": spec.seed,
    }
    for key, expected in expected_identity.items():
        if case.get(key) != expected:
            raise ValueError(f"{spec.case_id}.{key}: expected {expected!r}")
    run_id = _validate_process_receipt(case.get("process"), spec=spec, plan=plan)
    raw = _artifact_mapping(case.get("raw"), f"{spec.case_id}.raw")
    profile = plan.profile(spec.profile_id)
    _validate_resolved_config(raw, spec, profile)
    computed = summarize_mjwarp_dr_performance_case(raw, spec=spec, plan=plan)
    if case.get("summary") != computed:
        raise ValueError(f"{spec.case_id}.summary: differs from independently recomputed raw data")
    measurement = cast(Mapping[str, Any], plan.data["measurement"])
    if spec.lane in {"reset", "env"}:
        lane = cast(Mapping[str, Any], measurement[spec.lane])
        measured_key = "measured_barriers" if spec.lane == "reset" else "measured_steps"
        _validate_direct_diagnostics(
            raw.get("diagnostics"),
            spec=spec,
            profile=profile,
            measured=int(lane[measured_key]),
        )
        _validate_timing_lifecycle(raw, spec=spec, plan=plan)
        _validate_profiler(raw, spec=spec, summary=computed)
    else:
        _validate_train_diagnostics(raw, spec=spec, profile=profile, plan=plan)
        run_summary = _artifact_mapping(raw.get("run_summary"), f"{spec.case_id}.run_summary")
        iteration_memory = _artifact_list(
            run_summary.get("iteration_memory_diagnostics"),
            f"{spec.case_id}.iteration_memory_diagnostics",
        )
        warmup = int(cast(Mapping[str, Any], measurement["train"])["warmup_iterations"])
        post_warmup = iteration_memory[warmup:]
        windows = _artifact_list(raw.get("memory_windows"), f"{spec.case_id}.memory_windows")
        for window_key, iteration_key, label in (
            ("rss_samples_bytes", "rss_bytes", "RSS"),
            ("cuda_allocated_samples_bytes", "cuda_allocated_bytes", "CUDA allocated"),
            ("cuda_reserved_samples_bytes", "cuda_reserved_bytes", "CUDA reserved"),
        ):
            flattened = [
                sample
                for window in windows
                for sample in _artifact_list(
                    _artifact_mapping(window, "memory window").get(window_key),
                    f"{label} samples",
                )
            ]
            if flattened != [sample.get(iteration_key) for sample in post_warmup]:
                raise ValueError(f"{spec.case_id}: {label} windows differ from trainer raw data")
    return computed, run_id


def build_mjwarp_dr_performance_aggregates(
    cases: Sequence[Mapping[str, Any]],
    *,
    plan: MjwarpDrPerformancePlan,
) -> dict[str, Any]:
    """Aggregate only independently recomputed case summaries."""

    specs = expected_mjwarp_dr_performance_cases(plan)
    if len(cases) != len(specs):
        raise ValueError("cannot aggregate an incomplete mjwarp DR case matrix")
    by_id = {str(case["case_id"]): case for case in cases}

    def selected(**identity: object) -> list[Mapping[str, Any]]:
        result = [
            by_id[spec.case_id]
            for spec in specs
            if all(getattr(spec, key) == value for key, value in identity.items())
        ]
        if len(result) != 5:
            raise ValueError(f"aggregate group {identity!r} does not contain five processes")
        return result

    measurement = cast(Mapping[str, Any], plan.data["measurement"])
    reset_plan = cast(Mapping[str, Any], measurement["reset"])
    reset: list[dict[str, Any]] = []
    for batch_size in cast(list[int], reset_plan["batch_sizes"]):
        for density in cast(list[float], reset_plan["densities"]):
            for profile_id in cast(list[str], reset_plan["profiles"]):
                group = selected(
                    lane="reset",
                    batch_size=batch_size,
                    reset_density=float(density),
                    profile_id=profile_id,
                )
                phase_metrics: dict[str, Any] = {}
                for phase in RESET_PHASES:
                    process_p50 = [
                        float(
                            _artifact_mapping(
                                _artifact_mapping(case["summary"], "summary")["phase_metrics"],
                                "phase_metrics",
                            )[phase]["p50"]
                        )
                        for case in group
                    ]
                    process_p95 = [
                        float(
                            _artifact_mapping(
                                _artifact_mapping(case["summary"], "summary")["phase_metrics"],
                                "phase_metrics",
                            )[phase]["p95"]
                        )
                        for case in group
                    ]
                    phase_metrics[phase] = {
                        "process_p50": numeric_stats(process_p50),
                        "process_p95": numeric_stats(process_p95),
                        "p50_population_cv": _population_cv(process_p50, f"reset.{phase}.p50"),
                        "p95_population_cv": _population_cv(process_p95, f"reset.{phase}.p95"),
                    }
                reset.append(
                    {
                        "batch_size": batch_size,
                        "reset_density": float(density),
                        "profile_id": profile_id,
                        "process_repeats": len(group),
                        "phase_metrics": phase_metrics,
                        "primary_metric_population_cv": phase_metrics["reset_barrier"][
                            "p95_population_cv"
                        ],
                    }
                )

    env_plan = cast(Mapping[str, Any], measurement["env"])
    env: list[dict[str, Any]] = []
    for batch_size in cast(list[int], env_plan["batch_sizes"]):
        for profile_id in cast(list[str], env_plan["profiles"]):
            group = selected(lane="env", batch_size=batch_size, profile_id=profile_id)
            p50 = [
                float(
                    _artifact_mapping(
                        _artifact_mapping(case["summary"], "summary")["phase_metrics"],
                        "phase_metrics",
                    )["env_step"]["p50"]
                )
                for case in group
            ]
            p95 = [
                float(
                    _artifact_mapping(
                        _artifact_mapping(case["summary"], "summary")["phase_metrics"],
                        "phase_metrics",
                    )["env_step"]["p95"]
                )
                for case in group
            ]
            throughput = [
                float(_artifact_mapping(case["summary"], "summary")["throughput_env_steps_per_sec"])
                for case in group
            ]
            env.append(
                {
                    "batch_size": batch_size,
                    "profile_id": profile_id,
                    "process_repeats": len(group),
                    "step_process_p50": numeric_stats(p50),
                    "step_process_p95": numeric_stats(p95),
                    "throughput_process": numeric_stats(throughput),
                    "step_p95_population_cv": _population_cv(p95, "env.step.p95"),
                    "throughput_population_cv": _population_cv(throughput, "env.throughput"),
                    "primary_metric_population_cv": max(
                        _population_cv(p95, "env.step.p95"),
                        _population_cv(throughput, "env.throughput"),
                    ),
                }
            )

    train_plan = cast(Mapping[str, Any], measurement["train"])
    train: list[dict[str, Any]] = []
    for profile_id in cast(list[str], train_plan["profiles"]):
        group = selected(lane="train", profile_id=profile_id)
        p95 = [
            float(
                _artifact_mapping(
                    _artifact_mapping(case["summary"], "summary")["phase_metrics"],
                    "phase_metrics",
                )["ppo_iteration"]["p95"]
            )
            for case in group
        ]
        throughput = [
            float(
                _artifact_mapping(
                    _artifact_mapping(case["summary"], "summary")["throughput_env_steps_per_sec"],
                    "throughput",
                )["p50"]
            )
            for case in group
        ]
        train.append(
            {
                "batch_size": int(train_plan["batch_size"]),
                "profile_id": profile_id,
                "process_repeats": len(group),
                "iteration_process_p95": numeric_stats(p95),
                "throughput_process_p50": numeric_stats(throughput),
                "iteration_p95_population_cv": _population_cv(p95, "train.iteration.p95"),
                "throughput_population_cv": _population_cv(throughput, "train.throughput"),
                "primary_metric_population_cv": max(
                    _population_cv(p95, "train.iteration.p95"),
                    _population_cv(throughput, "train.throughput"),
                ),
            }
        )
    return {"reset": reset, "env": env, "train": train}


def _aggregate_index(values: Sequence[Mapping[str, Any]], **identity: object) -> Mapping[str, Any]:
    matches = [
        value
        for value in values
        if all(value.get(key) == expected for key, expected in identity.items())
    ]
    if len(matches) != 1:
        raise ValueError(f"aggregate identity {identity!r} has {len(matches)} matches")
    return matches[0]


def _case_performance(case: Mapping[str, Any]) -> Mapping[str, Any]:
    raw = _artifact_mapping(case.get("raw"), "case.raw")
    if case.get("lane") == "train":
        summary = _artifact_mapping(raw.get("run_summary"), "case.raw.run_summary")
        return _artifact_mapping(
            summary.get("runtime_performance_diagnostics"),
            "case.raw.run_summary.runtime_performance_diagnostics",
        )
    diagnostics = _artifact_mapping(raw.get("diagnostics"), "case.raw.diagnostics")
    return _artifact_mapping(
        diagnostics.get("performance_before"), "case.raw.diagnostics.performance_before"
    )


def _storage_gate_errors(cases: Sequence[Mapping[str, Any]]) -> list[str]:
    errors: list[str] = []
    signatures: dict[tuple[str, int], set[tuple[Any, ...]]] = {}
    for case in cases:
        profile_id = str(case["profile_id"])
        if profile_id == "disabled":
            continue
        batch_size = int(case["batch_size"])
        materialization = _artifact_mapping(
            _case_performance(case).get("materialization"), "performance.materialization"
        )
        fields = _artifact_list(materialization.get("fields"), "materialization.fields")
        field_bytes = tuple(
            (str(_artifact_mapping(item, "field")["field_name"]), int(item["model_bytes"]))
            for item in fields
            if isinstance(item, Mapping) and bool(item.get("replaced"))
        )
        signature = (
            int(materialization["expanded_model_bytes"]),
            int(materialization["baseline_bytes"]),
            field_bytes,
        )
        signatures.setdefault((profile_id, batch_size), set()).add(signature)
    canonical: dict[tuple[str, int], tuple[Any, ...]] = {}
    for key, values in signatures.items():
        if len(values) != 1:
            errors.append(f"storage {key}: receipts differ within the same profile/batch")
        else:
            canonical[key] = next(iter(values))
    for profile_id in ("tier_b_pd", "tier_c_armature", "tier_c_mixed"):
        base = canonical.get((profile_id, 128))
        if base is None:
            errors.append(f"storage {profile_id}: missing batch 128 receipt")
            continue
        for batch_size in (1024, 4096):
            candidate = canonical.get((profile_id, batch_size))
            if candidate is None:
                errors.append(f"storage {profile_id}: missing batch {batch_size} receipt")
                continue
            if candidate[1] != base[1]:
                errors.append(f"storage {profile_id}: baseline bytes changed with batch size")
            if int(candidate[0]) * 128 != int(base[0]) * batch_size:
                errors.append(f"storage {profile_id}: expanded bytes are not exactly batch-linear")
            base_fields = dict(cast(tuple[tuple[str, int], ...], base[2]))
            candidate_fields = dict(cast(tuple[tuple[str, int], ...], candidate[2]))
            if set(base_fields) != set(candidate_fields) or any(
                candidate_fields[name] * 128 != base_fields[name] * batch_size
                for name in base_fields
            ):
                errors.append(f"storage {profile_id}: field bytes are not exactly batch-linear")
    return errors


def _event_traffic_gate_errors(value: Mapping[str, Any], path: str) -> list[str]:
    errors: list[str] = []
    for term, counters_raw in value.items():
        counters = _artifact_mapping(counters_raw, f"{path}.{term}")
        for key in _EVENT_TRAFFIC_KEYS:
            if int(counters[key]) != 0:
                errors.append(f"{path}.{term}.{key}: warmed Event path is non-zero")
    return errors


def _stability_gate_errors(
    before: Mapping[str, Any], after: Mapping[str, Any], path: str
) -> list[str]:
    errors: list[str] = []
    for label, value in (("before", before), ("after", after)):
        for key in ("warm_numeric_allocations", "address_churn"):
            if int(value[key]) != 0:
                errors.append(f"{path}.{label}.{key}: expected zero")
    for key in ("buffers", "state_buffers"):
        if before[key] != after[key]:
            errors.append(f"{path}.{key}: warmed buffer addresses changed")
    before_graph = _artifact_mapping(before["graph"], f"{path}.before.graph")
    after_graph = _artifact_mapping(after["graph"], f"{path}.after.graph")
    for key in ("storage_buffers", "storage_generation", "storage_fingerprint", "active_keys"):
        if before_graph[key] != after_graph[key]:
            errors.append(f"{path}.graph.{key}: graph/storage identity changed")
    return errors


def _direct_diagnostic_gate_errors(
    case: Mapping[str, Any], *, plan: MjwarpDrPerformancePlan
) -> list[str]:
    case_id = str(case["case_id"])
    profile = plan.profile(str(case["profile_id"]))
    raw = _artifact_mapping(case["raw"], f"{case_id}.raw")
    diagnostics = _artifact_mapping(raw["diagnostics"], f"{case_id}.raw.diagnostics")
    measurement = cast(Mapping[str, Any], plan.data["measurement"])
    lane = str(case["lane"])
    lane_plan = cast(Mapping[str, Any], measurement[lane])
    measured = int(lane_plan["measured_barriers" if lane == "reset" else "measured_steps"])
    path = f"{case_id}.raw.diagnostics"
    errors: list[str] = []

    performance_before = _artifact_mapping(
        diagnostics["performance_before"], f"{path}.performance_before"
    )
    performance_after = _artifact_mapping(
        diagnostics["performance_after"], f"{path}.performance_after"
    )
    if performance_before["materialization"] != performance_after["materialization"]:
        errors.append(f"{path}.performance.materialization: changed in warmed window")
    graph_before = _artifact_mapping(performance_before["graph"], f"{path}.graph.before")
    graph_after = _artifact_mapping(performance_after["graph"], f"{path}.graph.after")
    for key in (
        "capture_count",
        "recapture_count",
        "stale_rejection_count",
        "eager_fallback_count",
    ):
        if int(graph_after[key]) - int(graph_before[key]) != 0:
            errors.append(f"{path}.graph.{key}: warmed graph state changed")
    if int(graph_after["launch_count"]) - int(graph_before["launch_count"]) != 3 * measured:
        errors.append(f"{path}.graph.launch_count: expected three launches per step")

    lifecycle_before = _artifact_mapping(
        performance_before["lifecycle"], f"{path}.lifecycle.before"
    )
    lifecycle_after = _artifact_mapping(performance_after["lifecycle"], f"{path}.lifecycle.after")
    expected_lifecycle = {
        "runtime_barriers": 2 * measured,
        "step_graph_launches": measured,
        "reset_graph_launches": measured,
        "forward_graph_launches": measured,
        "state_refreshes": 2 * measured,
    }
    for key, expected in expected_lifecycle.items():
        if int(lifecycle_after[key]) - int(lifecycle_before[key]) != expected:
            errors.append(f"{path}.lifecycle.{key}: differs from one fused step/reset barrier")
    expected_recompute = measured if profile.tier == "C" else 0
    if (
        int(performance_after["recompute_launch_count"])
        - int(performance_before["recompute_launch_count"])
        != expected_recompute
    ):
        errors.append(f"{path}.recompute: strongest recompute count differs from profile")
    if (
        int(performance_after["recompute_capture_count"])
        - int(performance_before["recompute_capture_count"])
        != 0
    ):
        errors.append(f"{path}.recompute: graph recaptured in warmed window")

    runtime_before = _artifact_mapping(
        diagnostics["runtime_traffic_before"], f"{path}.runtime_traffic_before"
    )
    runtime_after = _artifact_mapping(
        diagnostics["runtime_traffic_after"], f"{path}.runtime_traffic_after"
    )
    for key in _RUNTIME_TRAFFIC_ZERO_KEYS:
        if int(runtime_after[key]) - int(runtime_before[key]) != 0:
            errors.append(f"{path}.runtime_traffic.{key}: warmed runtime boundary is non-zero")
    expected_runtime = {
        "policy_steps": measured,
        "step_barriers": measured,
        "reset_barriers": measured,
        "state_materializations": 2 * measured,
    }
    for key, expected in expected_runtime.items():
        if int(runtime_after[key]) - int(runtime_before[key]) != expected:
            errors.append(f"{path}.runtime_traffic.{key}: semantic count differs")
    for label in ("event_traffic_before", "event_traffic_after"):
        events = _artifact_mapping(diagnostics[label], f"{path}.{label}")
        errors.extend(_event_traffic_gate_errors(events, f"{path}.{label}"))

    wrapper_before = _artifact_mapping(
        diagnostics["wrapper_traffic_before"], f"{path}.wrapper_traffic_before"
    )
    wrapper_after = _artifact_mapping(
        diagnostics["wrapper_traffic_after"], f"{path}.wrapper_traffic_after"
    )
    for key, expected in (("action_publications", measured), ("observation_snapshots", measured)):
        if int(wrapper_after[key]) - int(wrapper_before[key]) != expected:
            errors.append(f"{path}.wrapper_traffic.{key}: differs from measured steps")
    if (
        int(wrapper_after["finite_metric_materializations"])
        - int(wrapper_before["finite_metric_materializations"])
        != 0
    ):
        errors.append(f"{path}.wrapper_traffic: unexpected rollout boundary")

    stability_before = _artifact_mapping(
        diagnostics["stability_before"], f"{path}.stability_before"
    )
    stability_after = _artifact_mapping(diagnostics["stability_after"], f"{path}.stability_after")
    errors.extend(_stability_gate_errors(stability_before, stability_after, f"{path}.stability"))

    profiler = raw.get("profiler")
    if profiler is not None:
        summary = _artifact_mapping(case["summary"], f"{case_id}.summary")
        counts = _artifact_mapping(summary["profiler_counts"], f"{case_id}.profiler_counts")
        for key, value in counts.items():
            if int(value) != 0:
                errors.append(f"{case_id}: profiler found warmed {key}")
        profiler_map = _artifact_mapping(profiler, f"{case_id}.raw.profiler")
        runtime_delta = _artifact_mapping(
            profiler_map["runtime_delta"], f"{case_id}.raw.profiler.runtime_delta"
        )
        for key, value in runtime_delta.items():
            if int(value) != 0:
                errors.append(f"{case_id}: profiler/runtime {key} is non-zero")
        event_delta = _artifact_mapping(
            profiler_map["event_traffic_delta"], f"{case_id}.raw.profiler.event_traffic_delta"
        )
        errors.extend(
            _event_traffic_gate_errors(event_delta, f"{case_id}.raw.profiler.event_traffic_delta")
        )
    return errors


def _train_diagnostic_gate_errors(
    case: Mapping[str, Any], *, plan: MjwarpDrPerformancePlan
) -> list[str]:
    case_id = str(case["case_id"])
    profile = plan.profile(str(case["profile_id"]))
    raw = _artifact_mapping(case["raw"], f"{case_id}.raw")
    summary = _artifact_mapping(raw["run_summary"], f"{case_id}.raw.run_summary")
    measurement = cast(Mapping[str, Any], plan.data["measurement"])
    lane = cast(Mapping[str, Any], measurement["train"])
    iterations = int(lane["iterations"])
    steps = iterations * int(lane["num_steps_per_env"])
    path = f"{case_id}.raw.run_summary"
    errors: list[str] = []

    performance_before = _artifact_mapping(
        summary["runtime_performance_diagnostics_before_training"],
        f"{path}.runtime_performance_diagnostics_before_training",
    )
    performance = _artifact_mapping(
        summary["runtime_performance_diagnostics"], f"{path}.runtime_performance_diagnostics"
    )
    lifecycle_before = _artifact_mapping(
        performance_before["lifecycle"], f"{path}.lifecycle_before_training"
    )
    lifecycle = _artifact_mapping(performance["lifecycle"], f"{path}.lifecycle")
    expected_lifecycle = {
        "runtime_barriers": 2 * steps,
        "step_graph_launches": steps,
        "reset_graph_launches": steps,
        "forward_graph_launches": steps,
        "state_refreshes": 2 * steps,
    }
    for key, expected in expected_lifecycle.items():
        if int(lifecycle[key]) - int(lifecycle_before[key]) != expected:
            errors.append(f"{path}.lifecycle.{key}: differs from production iteration count")
    expected_recompute = steps if profile.tier == "C" else 0
    if (
        int(performance["recompute_launch_count"])
        - int(performance_before["recompute_launch_count"])
        != expected_recompute
    ):
        errors.append(f"{path}.recompute_launch_count: differs from profile")
    if (
        int(performance["recompute_capture_count"])
        - int(performance_before["recompute_capture_count"])
        != 0
    ):
        errors.append(f"{path}.recompute_capture_count: warmed graph recaptured")
    graph_before = _artifact_mapping(performance_before["graph"], f"{path}.graph_before_training")
    graph = _artifact_mapping(performance["graph"], f"{path}.graph")
    if int(graph["launch_count"]) - int(graph_before["launch_count"]) != 3 * steps:
        errors.append(f"{path}.graph.launch_count: differs from production iteration count")
    for key in (
        "capture_count",
        "recapture_count",
        "stale_rejection_count",
        "eager_fallback_count",
    ):
        if int(graph[key]) - int(graph_before[key]) != 0:
            errors.append(f"{path}.graph.{key}: warmed graph state changed")

    traffic = _artifact_mapping(summary["runtime_traffic_diagnostics"], f"{path}.runtime_traffic")
    for key in _RUNTIME_TRAFFIC_ZERO_KEYS:
        if int(traffic[key]) != 0:
            errors.append(f"{path}.runtime_traffic.{key}: expected zero")
    expected_runtime = {
        "policy_steps": steps,
        "step_barriers": steps,
        "reset_barriers": 1 + steps,
        "state_materializations": 1 + 2 * steps,
    }
    for key, expected in expected_runtime.items():
        if int(traffic[key]) != expected:
            errors.append(f"{path}.runtime_traffic.{key}: semantic count differs")
    events = _artifact_mapping(
        summary["runtime_event_traffic_diagnostics"], f"{path}.runtime_event_traffic"
    )
    errors.extend(_event_traffic_gate_errors(events, f"{path}.runtime_event_traffic"))

    wrapper = _artifact_mapping(summary["wrapper_traffic_diagnostics"], f"{path}.wrapper")
    expected_wrapper = {
        "action_publications": steps,
        "observation_snapshots": 1 + steps,
        "finite_metric_materializations": iterations,
    }
    for key, expected in expected_wrapper.items():
        if int(wrapper[key]) != expected:
            errors.append(f"{path}.wrapper.{key}: rollout count differs")
    logging = _artifact_mapping(summary["logging_traffic_diagnostics"], f"{path}.logging")
    if (
        int(logging["rollout_steps"]) != steps
        or int(logging["metric_materializations"]) != iterations
    ):
        errors.append(f"{path}.logging: rollout/materialization count differs")
    stability = _artifact_mapping(summary["runtime_stability_diagnostics"], f"{path}.stability")
    for key in ("warm_numeric_allocations", "address_churn"):
        if int(stability[key]) != 0:
            errors.append(f"{path}.stability.{key}: expected zero")
    return errors


def recompute_mjwarp_dr_performance_gate(
    cases: Sequence[Mapping[str, Any]],
    aggregates: Mapping[str, Any],
    *,
    plan: MjwarpDrPerformancePlan,
) -> dict[str, Any]:
    """Evaluate every frozen threshold from raw-derived summaries."""

    errors: list[str] = []
    gates = cast(Mapping[str, Any], plan.data["gates"])
    completeness = cast(Mapping[str, Any], gates["completeness"])
    cv_max = float(completeness["maximum_primary_metric_population_cv"])
    for lane in ("reset", "env", "train"):
        for aggregate in cast(list[Mapping[str, Any]], aggregates[lane]):
            cv = float(aggregate["primary_metric_population_cv"])
            if cv > cv_max:
                errors.append(
                    f"{lane} {aggregate.get('profile_id')} primary population CV "
                    f"{cv:.6f} exceeds {cv_max:.6f}"
                )

    memory_gate = cast(Mapping[str, Any], gates["steady_state_memory"])
    for case in cases:
        case_id = str(case["case_id"])
        memory = _artifact_mapping(
            _artifact_mapping(case["summary"], f"{case_id}.summary").get("memory"),
            f"{case_id}.summary.memory",
        )
        checks = (
            (
                "host_rss_positive_slope_bytes_per_window",
                "host_rss_positive_slope_bytes_per_window_max",
            ),
            (
                "host_rss_last_minus_first_median_bytes",
                "host_rss_last_minus_first_median_bytes_max",
            ),
            (
                "cuda_allocated_positive_growth_bytes",
                "cuda_allocated_positive_growth_bytes_max",
            ),
            (
                "cuda_reserved_positive_growth_bytes",
                "cuda_reserved_positive_growth_bytes_max",
            ),
        )
        for metric, threshold in checks:
            if float(memory[metric]) > float(memory_gate[threshold]):
                errors.append(
                    f"{case_id} {metric}={float(memory[metric]):.3f} exceeds "
                    f"{float(memory_gate[threshold]):.3f}"
                )
        if case.get("lane") in {"reset", "env"}:
            errors.extend(_direct_diagnostic_gate_errors(case, plan=plan))
        elif case.get("lane") == "train":
            errors.extend(_train_diagnostic_gate_errors(case, plan=plan))

    reset_aggregates = cast(list[Mapping[str, Any]], aggregates["reset"])
    reset_gate = cast(Mapping[str, Any], gates["reset_latency"])
    reset_plan = cast(Mapping[str, Any], cast(Mapping[str, Any], plan.data["measurement"])["reset"])
    for batch_size in cast(list[int], reset_plan["batch_sizes"]):
        for density in cast(list[float], reset_plan["densities"]):
            disabled = _aggregate_index(
                reset_aggregates,
                batch_size=batch_size,
                reset_density=float(density),
                profile_id="disabled",
            )
            disabled_p95 = float(
                _artifact_mapping(
                    _artifact_mapping(disabled["phase_metrics"], "phase_metrics")["reset_barrier"],
                    "reset_barrier",
                )["process_p95"]["p50"]
            )
            for profile_id in cast(list[str], reset_plan["profiles"]):
                aggregate = _aggregate_index(
                    reset_aggregates,
                    batch_size=batch_size,
                    reset_density=float(density),
                    profile_id=profile_id,
                )
                p95 = float(
                    _artifact_mapping(
                        _artifact_mapping(aggregate["phase_metrics"], "phase_metrics")[
                            "reset_barrier"
                        ],
                        "reset_barrier",
                    )["process_p95"]["p50"]
                )
                if p95 > float(reset_gate["all_profile_barrier_p95_ms_max"]):
                    errors.append(f"reset b{batch_size} d{density} {profile_id} p95 exceeds 20 ms")
                limit = None
                if profile_id == "tier_b_pd":
                    limit = disabled_p95 * float(
                        reset_gate["tier_b_to_disabled_p95_ratio_max"]
                    ) + float(reset_gate["tier_b_additive_p95_ms"])
                elif profile_id == "tier_c_armature":
                    limit = disabled_p95 + float(reset_gate["tier_c_armature_additive_p95_ms"])
                elif profile_id == "tier_c_mixed":
                    limit = disabled_p95 + float(reset_gate["tier_c_mixed_additive_p95_ms"])
                if limit is not None and p95 > limit:
                    errors.append(
                        f"reset b{batch_size} d{density} {profile_id} p95 "
                        f"{p95:.6f} exceeds paired limit {limit:.6f}"
                    )

    for lane_name, p95_key, throughput_key in (
        ("env", "step_process_p95", "throughput_process"),
        ("train", "iteration_process_p95", "throughput_process_p50"),
    ):
        lane_gate = cast(Mapping[str, Any], gates[lane_name])
        lane_aggregates = cast(list[Mapping[str, Any]], aggregates[lane_name])
        identities = (
            [("batch_size", batch) for batch in (128, 1024, 4096)]
            if lane_name == "env"
            else [("batch_size", 1024)]
        )
        for key, batch_size in identities:
            disabled = _aggregate_index(
                lane_aggregates, **{key: batch_size, "profile_id": "disabled"}
            )
            disabled_throughput = float(
                _artifact_mapping(disabled[throughput_key], throughput_key)["p50"]
            )
            disabled_p95 = float(_artifact_mapping(disabled[p95_key], p95_key)["p50"])
            for profile_id, tier in (("tier_b_pd", "tier_b"), ("tier_c_armature", "tier_c")):
                profile = _aggregate_index(
                    lane_aggregates, **{key: batch_size, "profile_id": profile_id}
                )
                throughput = float(
                    _artifact_mapping(profile[throughput_key], throughput_key)["p50"]
                )
                p95 = float(_artifact_mapping(profile[p95_key], p95_key)["p50"])
                throughput_ratio = throughput / disabled_throughput
                p95_ratio = p95 / disabled_p95
                if throughput_ratio < float(lane_gate[f"{tier}_throughput_ratio_min"]):
                    errors.append(
                        f"{lane_name} b{batch_size} {profile_id} throughput ratio "
                        f"{throughput_ratio:.6f} is below the frozen gate"
                    )
                p95_gate_key = (
                    f"{tier}_step_p95_ratio_max"
                    if lane_name == "env"
                    else f"{tier}_iteration_p95_ratio_max"
                )
                if p95_ratio > float(lane_gate[p95_gate_key]):
                    errors.append(
                        f"{lane_name} b{batch_size} {profile_id} p95 ratio "
                        f"{p95_ratio:.6f} exceeds the frozen gate"
                    )

    errors.extend(_storage_gate_errors(cases))
    return {"passed": not errors, "errors": errors}


def _sha256_bytes(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def _git_file_sha256(repo_root: Path, commit: str, path: str) -> str:
    completed = _git(repo_root, "show", f"{commit}:{path}")
    return _sha256_bytes(completed.stdout)


def _validate_artifact_binding(
    artifact: Mapping[str, Any],
    *,
    plan: MjwarpDrPerformancePlan,
    receipt: MjwarpDrPerformanceFreezeReceipt,
    repo_root: Path | None,
    errors: list[str],
) -> None:
    contract = _mapping(artifact.get("contract"), "artifact.contract", errors)
    _exact_artifact_keys(
        contract,
        {
            "plan_path",
            "plan_sha256",
            "freeze_receipt_path",
            "freeze_receipt_sha256",
            "freeze_commit",
        },
        "artifact.contract",
        errors,
    )
    expected_contract = {
        "plan_path": PLAN_PATH.as_posix(),
        "plan_sha256": plan.plan_sha256,
        "freeze_receipt_path": FREEZE_RECEIPT_PATH.as_posix(),
        "freeze_receipt_sha256": sha256_file(receipt.source_path),
        "freeze_commit": receipt.freeze_commit,
    }
    for key, expected in expected_contract.items():
        if contract.get(key) != expected:
            errors.append(f"artifact.contract.{key}: expected {expected!r}")

    hardware = _mapping(artifact.get("hardware"), "artifact.hardware", errors)
    frozen_hardware = cast(Mapping[str, Any], plan.data["hardware"])
    _exact_artifact_keys(hardware, set(frozen_hardware), "artifact.hardware", errors)
    for key, expected in frozen_hardware.items():
        if hardware.get(key) != expected:
            errors.append(f"artifact.hardware.{key}: differs from frozen hardware")

    dependencies = _mapping(artifact.get("dependencies"), "artifact.dependencies", errors)
    _exact_artifact_keys(dependencies, {"lockfile", "packages"}, "artifact.dependencies", errors)
    frozen_dependencies = cast(Mapping[str, Any], plan.data["dependencies"])
    if dependencies.get("lockfile") != frozen_dependencies.get("lockfile"):
        errors.append("artifact.dependencies.lockfile: differs from frozen plan")
    packages = _mapping(dependencies.get("packages"), "artifact.dependencies.packages", errors)
    expected_packages = cast(Mapping[str, Any], frozen_dependencies["packages"])
    if set(packages) != set(expected_packages):
        errors.append("artifact.dependencies.packages: package set differs from frozen plan")
    for name, constraint in expected_packages.items():
        package = _mapping(packages.get(name), f"artifact.dependencies.packages.{name}", errors)
        _exact_artifact_keys(
            package,
            {"constraint", "version"},
            f"artifact.dependencies.packages.{name}",
            errors,
        )
        if package.get("constraint") != constraint:
            errors.append(f"artifact.dependencies.packages.{name}.constraint: differs from plan")
        try:
            version = _artifact_string(
                package.get("version"), f"artifact.dependencies.packages.{name}.version"
            )
            if not dependency_version_satisfies(version, str(constraint)):
                errors.append(
                    f"artifact.dependencies.packages.{name}.version: {version!r} does not "
                    f"satisfy frozen constraint {constraint!r}"
                )
        except ValueError as exc:
            errors.append(str(exc))

    tier_d = artifact.get("tier_d_eligibility")
    if tier_d != plan.data["tier_d_eligibility"]:
        errors.append("artifact.tier_d_eligibility: must record the frozen zero-capability audit")

    source = _mapping(artifact.get("source"), "artifact.source", errors)
    _exact_artifact_keys(
        source,
        {
            "commit",
            "git_status",
            "source_inputs",
            "source_tree_sha256",
            "owner_yaml_sha256",
            "lockfile_sha256",
        },
        "artifact.source",
        errors,
    )
    commit = str(source.get("commit", ""))
    if not _COMMIT_RE.fullmatch(commit):
        errors.append("artifact.source.commit: expected a full lowercase commit SHA")
    if source.get("git_status") != "":
        errors.append("artifact.source.git_status: candidate capture was not clean")
    if source.get("source_inputs") != list(SOURCE_INPUTS):
        errors.append("artifact.source.source_inputs: differs from the v1 source closure")
    for key in ("source_tree_sha256", "owner_yaml_sha256", "lockfile_sha256"):
        try:
            _artifact_sha(source.get(key), f"artifact.source.{key}")
        except ValueError as exc:
            errors.append(str(exc))
    if repo_root is None or not _COMMIT_RE.fullmatch(commit):
        return
    try:
        exists = _git(repo_root, "cat-file", "-e", f"{commit}^{{commit}}", check=False)
        if exists.returncode != 0:
            errors.append("artifact.source.commit: candidate commit is unavailable")
            return
        if commit == receipt.freeze_commit:
            errors.append("artifact.source.commit: candidate must differ from freeze commit")
        ancestor = _git(
            repo_root,
            "merge-base",
            "--is-ancestor",
            receipt.freeze_commit,
            commit,
            check=False,
        )
        if ancestor.returncode != 0:
            errors.append("artifact.source.commit: candidate does not descend from freeze commit")
        expected_tree = source_tree_sha256_at_commit(repo_root, SOURCE_INPUTS, commit)
        if source.get("source_tree_sha256") != expected_tree:
            errors.append("artifact.source.source_tree_sha256: differs from candidate commit")
        if source.get("owner_yaml_sha256") != _git_file_sha256(
            repo_root, commit, "conf/ppo/task/g1_walk_flat/mjwarp.yaml"
        ):
            errors.append("artifact.source.owner_yaml_sha256: differs from candidate commit")
        if source.get("lockfile_sha256") != _git_file_sha256(repo_root, commit, "uv.lock"):
            errors.append("artifact.source.lockfile_sha256: differs from candidate commit")
    except (OSError, subprocess.CalledProcessError, ValueError) as exc:
        errors.append(f"artifact.source: cannot verify candidate commit: {exc}")


def _validate_execution(
    artifact: Mapping[str, Any], specs: Sequence[DrPerformanceCaseSpec], errors: list[str]
) -> None:
    execution = _mapping(artifact.get("execution"), "artifact.execution", errors)
    _exact_artifact_keys(
        execution,
        {
            "started_at",
            "finished_at",
            "preflight_before",
            "preflight_after",
            "case_order",
            "outcomes",
        },
        "artifact.execution",
        errors,
    )
    for key in ("started_at", "finished_at"):
        try:
            _artifact_string(execution.get(key), f"artifact.execution.{key}")
        except ValueError as exc:
            errors.append(str(exc))
    expected_order = [spec.case_id for spec in specs]
    if execution.get("case_order") != expected_order:
        errors.append("artifact.execution.case_order: differs from canonical 300-case order")
    outcomes = _mapping(execution.get("outcomes"), "artifact.execution.outcomes", errors)
    expected_outcomes = {
        "completed": len(specs),
        "failed": 0,
        "skipped": 0,
        "xfailed": 0,
        "xpassed": 0,
        "filtered": 0,
    }
    if outcomes != expected_outcomes:
        errors.append("artifact.execution.outcomes: incomplete or filtered case matrix")
    for label in ("preflight_before", "preflight_after"):
        path = f"artifact.execution.{label}"
        preflight = _mapping(execution.get(label), path, errors)
        _exact_artifact_keys(
            preflight,
            {"timestamp", "gpu_compute_processes", "gpu_sample"},
            path,
            errors,
        )
        try:
            _artifact_string(preflight.get("timestamp"), f"{path}.timestamp")
        except ValueError as exc:
            errors.append(str(exc))
        processes = preflight.get("gpu_compute_processes")
        if not isinstance(processes, list):
            errors.append(f"{path}.gpu_compute_processes: expected list")
        elif processes:
            errors.append(f"{path}: foreign GPU compute processes present")
        sample = _mapping(preflight.get("gpu_sample"), f"{path}.gpu_sample", errors)
        _exact_artifact_keys(
            sample,
            {"utilization_percent", "memory_used_mib", "temperature_c", "pstate"},
            f"{path}.gpu_sample",
            errors,
        )
        try:
            utilization = _artifact_integer(
                sample.get("utilization_percent"), f"{path}.gpu_sample.utilization_percent"
            )
            if utilization > 100:
                errors.append(f"{path}.gpu_sample.utilization_percent: expected <= 100")
            _artifact_integer(sample.get("memory_used_mib"), f"{path}.gpu_sample.memory_used_mib")
            _artifact_integer(sample.get("temperature_c"), f"{path}.gpu_sample.temperature_c")
            pstate = _artifact_string(sample.get("pstate"), f"{path}.gpu_sample.pstate")
            if not _GPU_PSTATE_RE.fullmatch(pstate):
                errors.append(f"{path}.gpu_sample.pstate: expected NVIDIA P-state such as P0")
        except ValueError as exc:
            errors.append(str(exc))


def recompute_mjwarp_dr_performance_evidence(
    cases: Sequence[Mapping[str, Any]],
    *,
    plan: MjwarpDrPerformancePlan,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Recompute aggregate and gate payloads for artifact construction."""

    aggregates = build_mjwarp_dr_performance_aggregates(cases, plan=plan)
    gate = recompute_mjwarp_dr_performance_gate(cases, aggregates, plan=plan)
    return aggregates, gate


def validate_mjwarp_dr_performance_artifact(
    artifact: Mapping[str, Any],
    *,
    plan: MjwarpDrPerformancePlan,
    receipt: MjwarpDrPerformanceFreezeReceipt,
    repo_root: Path | None = None,
    require_passing_gate: bool = True,
) -> tuple[str, ...]:
    """Validate provenance and reconstruct all evidence from raw samples."""

    integrity_errors: list[str] = []
    expected_top_keys = {
        "schema_version",
        "artifact_kind",
        "benchmark_id",
        "issue",
        "parent_issue",
        "contract",
        "source",
        "hardware",
        "dependencies",
        "execution",
        "tier_d_eligibility",
        "cases",
        "aggregates",
        "gate",
    }
    _exact_artifact_keys(artifact, expected_top_keys, "artifact", integrity_errors)
    identity = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": ARTIFACT_KIND,
        "benchmark_id": BENCHMARK_ID,
        "issue": ISSUE,
        "parent_issue": PARENT_ISSUE,
    }
    for key, expected in identity.items():
        if artifact.get(key) != expected:
            integrity_errors.append(f"artifact.{key}: expected {expected!r}")
    _validate_artifact_binding(
        artifact,
        plan=plan,
        receipt=receipt,
        repo_root=repo_root,
        errors=integrity_errors,
    )
    specs = expected_mjwarp_dr_performance_cases(plan)
    _validate_execution(artifact, specs, integrity_errors)

    cases_raw = artifact.get("cases")
    cases = cases_raw if isinstance(cases_raw, list) else []
    if not isinstance(cases_raw, list):
        integrity_errors.append("artifact.cases: expected a list")
    if len(cases) != len(specs):
        integrity_errors.append(f"artifact.cases: expected {len(specs)} cases, got {len(cases)}")
    recomputed_cases: list[Mapping[str, Any]] = []
    run_ids: list[str] = []
    for index, spec in enumerate(specs):
        if index >= len(cases):
            break
        try:
            computed_summary, run_id = _validate_case_and_recompute(
                cases[index], spec=spec, plan=plan
            )
            case = cast(Mapping[str, Any], cases[index])
            recomputed_cases.append({**case, "summary": computed_summary})
            run_ids.append(run_id)
        except (KeyError, TypeError, ValueError) as exc:
            integrity_errors.append(str(exc))
    if len(set(run_ids)) != len(run_ids):
        integrity_errors.append("artifact.cases: process run_id receipts are not unique")

    threshold_errors: list[str] = []
    computed_aggregates: Mapping[str, Any] | None = None
    if len(recomputed_cases) == len(specs) and not integrity_errors:
        try:
            computed_aggregates, computed_gate = recompute_mjwarp_dr_performance_evidence(
                recomputed_cases, plan=plan
            )
            threshold_errors.extend(cast(list[str], computed_gate["errors"]))
            if artifact.get("aggregates") != computed_aggregates:
                integrity_errors.append(
                    "artifact.aggregates: differs from independently recomputed raw data"
                )
        except (KeyError, TypeError, ValueError) as exc:
            integrity_errors.append(f"artifact aggregate recomputation failed: {exc}")

    expected_gate_errors = [*integrity_errors, *threshold_errors]
    expected_gate = {"passed": not expected_gate_errors, "errors": expected_gate_errors}
    gate_mismatch: list[str] = []
    if artifact.get("gate") != expected_gate:
        gate_mismatch.append("artifact.gate: differs from independent validation")
    if require_passing_gate:
        return tuple((*integrity_errors, *threshold_errors, *gate_mismatch))
    return tuple((*integrity_errors, *gate_mismatch))


__all__ = [
    "ARTIFACT_KIND",
    "BENCHMARK_ID",
    "DEFAULT_ARTIFACT_PATH",
    "FREEZE_COMMIT",
    "FREEZE_RECEIPT_PATH",
    "ISSUE",
    "DrPerformanceCaseSpec",
    "MjwarpDrPerformanceContractError",
    "MjwarpDrPerformanceFreezeReceipt",
    "MjwarpDrPerformancePlan",
    "PLAN_PATH",
    "PLAN_SHA256",
    "RESET_PHASES",
    "SOURCE_INPUTS",
    "TRAIN_SCALAR_TAGS",
    "build_mjwarp_dr_performance_aggregates",
    "dependency_version_satisfies",
    "expected_mjwarp_dr_performance_cases",
    "load_mjwarp_dr_performance_freeze_receipt",
    "load_mjwarp_dr_performance_plan",
    "recompute_mjwarp_dr_performance_evidence",
    "recompute_mjwarp_dr_performance_gate",
    "summarize_memory_windows",
    "summarize_mjwarp_dr_performance_case",
    "validate_mjwarp_dr_performance_artifact",
]
