"""Capability-derived production task rollout contract for Issue #705.

The rollout plan is a cold-path release contract.  It does not replace the
public training run: it proves that each promoted combination is backed by the
owner, registry, compiled signatures, and fresh prerequisite phase receipts
that the real-CUDA integration test subsequently exercises.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from omegaconf import OmegaConf

from unilab.tools.claim_gap_audit import InventoryTestState, load_claim_gap_inventory
from unilab.tools.issue705_phase_evidence import sha256_file
from unilab.tools.issue705_support import (
    CLAIM_INVENTORY_PATH,
    SUPPORT_EVIDENCE_PATH,
    CompiledSignature,
    DeclaredEvidenceLevel,
    SupportCombination,
    SupportEvidenceError,
    audit_support_evidence,
    load_support_evidence,
    snapshot_registry_backends,
    test_node_exists,
)
from unilab.tools.phase_acceptance import (
    ClaimStatus,
    EvidenceResult,
    ManifestValidationError,
    load_phase_acceptance,
)

SCHEMA_VERSION = 1
ISSUE = 705
CLAIM_ID = "P7-TASK-ROLLOUT"
PLAN_FINGERPRINT = "mjwarp-task-rollout-v1"
ROLLOUT_PLAN_PATH = Path("tests/acceptance/issue_705/mjwarp_task_rollout_plan.yaml")

_MISSING = object()
_EXPECTED_KEY = ("ppo_torch", "g1_walk_flat", "mjwarp")
_EXPECTED_CAPABILITIES = (
    "device_resident_ppo",
    "flat_scene",
    "g1_managed_device",
    "g1_position_target_control",
    "no_observation_noise",
    "no_render",
    "owner_disabled_dr",
    "typed_state_reset",
)
_EXPECTED_DISABLED_DR = (
    "randomize_body_gravity_compensation",
    "randomize_dof_armature",
    "randomize_kd",
    "randomize_kp",
)
_EXPECTED_RUNTIME_IMPL = "mjwarp_device_v1"
_EXPECTED_RUNTIME_RESOLVER = "unilab.training.rsl_rl_device:resolve_mjwarp_device_ppo_runtime"

_EXPECTED_PREREQUISITES: dict[str, tuple[int, str]] = {
    "P2-BACKEND-IDENTITY": (
        2,
        "tests/base/test_mjwarp_identity.py::test_mjwarp_identity_is_independent_from_mujoco",
    ),
    "P2-GPU-CORRECTNESS": (
        2,
        "tests/base/test_mjwarp_backend.py::test_real_cuda_init_reset_step",
    ),
    "P2-RESET-ISOLATION": (
        2,
        "tests/base/test_mjwarp_backend.py::test_selected_row_reset_isolated",
    ),
    "P2-TRAJECTORY-DIFFERENTIAL": (
        2,
        "tests/base/test_mjwarp_differential.py::test_g1_short_trajectory_matches_mujoco",
    ),
    "P2-DR-OWNER-SEMANTICS": (
        2,
        "tests/dr/test_mjwarp_g1_dr.py::"
        "test_g1_kp_kd_owner_semantics_have_physics_effect_or_are_disabled",
    ),
    "P2-TRANSFER-ACCOUNTING": (
        2,
        "tests/base/test_mjwarp_transfers.py::test_host_profile_transfer_count_matches_bound_plan",
    ),
    "P2-UNSUPPORTED-FAIL-CLOSED": (
        2,
        "tests/base/test_mjwarp_capabilities.py::test_unsupported_matrix_fails_before_step",
    ),
    "P2-TRAIN-LIVENESS": (
        2,
        "tests/integration/test_mjwarp_train_smoke.py::"
        "test_g1_one_iteration_uses_production_mjwarp",
    ),
    "P3-TASK-COMPILER": (
        3,
        "tests/manager/test_task_compiler.py::test_compiler_binds_and_freezes_complete_plan",
    ),
    "P3-LIFECYCLE-PARITY": (
        3,
        "tests/manager/test_managed_lifecycle.py::test_terminal_and_autoreset_lifecycle_trace",
    ),
    "P3-G1-REFERENCE-DIFFERENTIAL": (
        3,
        "tests/manager/test_g1_reference_differential.py::"
        "test_g1_managed_reference_matches_handwritten_env",
    ),
    "P3-POLICY-ABI": (
        3,
        "tests/training/test_managed_policy_abi.py::test_managed_policy_abi_mismatch_fails_closed",
    ),
    "P3-CROSS-BACKEND-PLAN": (
        3,
        "tests/manager/test_cross_backend_plan.py::test_g1_plan_is_shared_by_mujoco_and_mjwarp",
    ),
    "P5-DEVICE-ABI": (
        5,
        "tests/training/test_device_transition_abi.py::test_dlpack_pointer_shape_dtype_and_lifetime",
    ),
    "P5-STREAM-LIFETIME": (
        5,
        "tests/training/test_device_stream_contract.py::test_missing_completion_event_is_detected",
    ),
    "P5-DEVICE-LIFECYCLE": (
        5,
        "tests/training/test_device_lifecycle.py::test_device_adapter_matches_host_terminal_contract",
    ),
    "P5-NO-HOST-ROUNDTRIP": (
        5,
        "tests/training/test_device_transfer_budget.py::test_rollout_has_no_per_step_host_roundtrip",
    ),
    "P5-GRAPH-CONTRACT": (
        5,
        "tests/base/test_mjwarp_graph_contract.py::test_graph_key_change_recaptures_or_fails_closed",
    ),
    "P5-TRAIN-PERFORMANCE": (
        5,
        "tests/benchmark/test_mjwarp_ppo_benchmark.py::test_device_profile_meets_end_to_end_gate",
    ),
    "P5-DEVICE-STABILITY": (
        5,
        "tests/training/test_device_runtime_stability.py::"
        "test_long_rollout_memory_and_addresses_are_stable",
    ),
    "P6-CAPABILITY-BIJECTION": (
        6,
        "tests/dr/test_mjwarp_capability_matrix.py::"
        "test_advertised_capabilities_equal_mandatory_parameter_cases",
    ),
    "P6-DR-SEMANTICS": (
        6,
        "tests/dr/test_mjwarp_mutation_semantics.py::"
        "test_operations_baselines_rows_and_persistence",
    ),
    "P6-PHYSICS-EFFECT": (
        6,
        "tests/dr/test_mjwarp_physics_effect.py::test_each_supported_mutation_has_next_step_effect",
    ),
    "P6-RNG-REPRODUCIBILITY": (
        6,
        "tests/dr/test_keyed_rng.py::test_rng_is_invariant_to_row_order_and_unrelated_terms",
    ),
    "P6-GRAPH-RECAPTURE": (
        6,
        "tests/dr/test_mjwarp_graph_mutation.py::"
        "test_field_expansion_invalidates_and_recaptures_all_graph_consumers",
    ),
    "P6-DR-PERFORMANCE": (
        6,
        "tests/benchmark/test_mjwarp_dr_benchmark.py::"
        "test_dr_profiles_meet_preregistered_density_gates",
    ),
    "P6-RECOMPUTE-AGGREGATION": (
        6,
        "tests/dr/test_mjwarp_recompute.py::test_strongest_recompute_runs_once_per_barrier",
    ),
}


@dataclass(frozen=True)
class RolloutPrerequisite:
    phase: int
    claim_id: str
    test_id: str


@dataclass(frozen=True)
class TaskRolloutEntry:
    entrypoint_id: str
    task_slug: str
    env_name: str
    backend: str
    execution_profile: str
    owner_yaml: Path
    owner_yaml_sha256: str
    seeds: tuple[int, ...]
    num_envs: int
    num_steps_per_env: int
    max_iterations: int
    runtime_impl: str
    runtime_resolver: str
    required_capabilities: tuple[str, ...]
    disabled_domain_rand: tuple[str, ...]
    expected_model_targets: tuple[str, ...]
    support_compiled_signature: CompiledSignature
    rollout_compiled_signature: CompiledSignature
    prerequisites: tuple[RolloutPrerequisite, ...]

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.entrypoint_id, self.task_slug, self.backend)


@dataclass(frozen=True)
class TaskRolloutPlan:
    schema_version: int
    issue: int
    claim_id: str
    plan_fingerprint: str
    entries: tuple[TaskRolloutEntry, ...]
    source: Path


@dataclass(frozen=True)
class TaskRolloutAuditReport:
    entries: int
    prerequisites: int
    errors: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.errors


class TaskRolloutPlanError(ValueError):
    def __init__(self, source: Path, errors: list[str] | tuple[str, ...]) -> None:
        self.source = source
        self.errors = tuple(errors)
        detail = "\n".join(f"- {error}" for error in self.errors)
        super().__init__(f"invalid Issue #705 task rollout plan {source}:\n{detail}")


class _Parser:
    def __init__(self) -> None:
        self.errors: list[str] = []

    def mapping(self, value: Any, path: str, keys: tuple[str, ...]) -> dict[str, Any]:
        if not isinstance(value, dict):
            self.errors.append(f"{path}: expected mapping")
            return {}
        expected = set(keys)
        actual = set(value)
        for key in sorted(expected - actual):
            self.errors.append(f"{path}: missing key `{key}`")
        for key in sorted(actual - expected, key=str):
            self.errors.append(f"{path}: unknown key `{key}`")
        return value

    def string(self, value: Any, path: str) -> str:
        if not isinstance(value, str) or not value.strip():
            self.errors.append(f"{path}: expected non-empty string")
            return ""
        if "${" in value:
            self.errors.append(f"{path}: interpolation is not allowed")
        return value

    def integer(self, value: Any, path: str, *, minimum: int = 0) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            self.errors.append(f"{path}: expected integer")
            return 0
        if value < minimum:
            self.errors.append(f"{path}: must be >= {minimum}")
        return int(value)

    def repo_path(self, value: Any, path: str) -> Path:
        result = Path(self.string(value, path))
        if result.is_absolute() or ".." in result.parts or "." in result.parts:
            self.errors.append(f"{path}: must stay within the repository")
        return result

    def sha256(self, value: Any, path: str) -> str:
        text = self.string(value, path)
        if len(text) != 71 or not text.startswith("sha256:"):
            self.errors.append(f"{path}: expected sha256:<64 lowercase hex characters>")
        else:
            digest = text.removeprefix("sha256:")
            if any(character not in "0123456789abcdef" for character in digest):
                self.errors.append(f"{path}: expected sha256:<64 lowercase hex characters>")
        return text

    def string_list(
        self,
        value: Any,
        path: str,
        *,
        allow_empty: bool,
        canonical: bool,
    ) -> tuple[str, ...]:
        if not isinstance(value, list) or (not allow_empty and not value):
            qualifier = "possibly empty" if allow_empty else "non-empty"
            self.errors.append(f"{path}: expected {qualifier} string list")
            return ()
        result = tuple(self.string(item, f"{path}[{index}]") for index, item in enumerate(value))
        if len(set(result)) != len(result):
            self.errors.append(f"{path}: duplicate values are not allowed")
        if canonical and result != tuple(sorted(result)):
            self.errors.append(f"{path}: values must be sorted")
        return result

    def integer_list(self, value: Any, path: str) -> tuple[int, ...]:
        if not isinstance(value, list) or not value:
            self.errors.append(f"{path}: expected non-empty integer list")
            return ()
        result = tuple(
            self.integer(item, f"{path}[{index}]", minimum=0) for index, item in enumerate(value)
        )
        if len(set(result)) != len(result):
            self.errors.append(f"{path}: duplicate values are not allowed")
        if result != tuple(sorted(result)):
            self.errors.append(f"{path}: values must be sorted")
        return result


_ROOT_KEYS = ("schema_version", "issue", "claim_id", "plan_fingerprint", "entries")
_ENTRY_KEYS = (
    "entrypoint_id",
    "task_slug",
    "env_name",
    "backend",
    "execution_profile",
    "owner_yaml",
    "owner_yaml_sha256",
    "seeds",
    "num_envs",
    "num_steps_per_env",
    "max_iterations",
    "runtime_impl",
    "runtime_resolver",
    "required_capabilities",
    "disabled_domain_rand",
    "expected_model_targets",
    "support_compiled_signature",
    "rollout_compiled_signature",
    "prerequisites",
)
_SIGNATURE_KEYS = (
    "task_key",
    "executor_key",
    "task_plan_fingerprint",
    "policy_abi_fingerprint",
    "backend_plan_fingerprint",
)
_PREREQUISITE_KEYS = ("phase", "claim_id", "test_id")


def _parse_signature(parser: _Parser, value: Any, path: str) -> CompiledSignature:
    raw = parser.mapping(value, path, _SIGNATURE_KEYS)
    return CompiledSignature(
        task_key=parser.string(raw.get("task_key", _MISSING), f"{path}.task_key"),
        executor_key=parser.string(raw.get("executor_key", _MISSING), f"{path}.executor_key"),
        task_plan_fingerprint=parser.string(
            raw.get("task_plan_fingerprint", _MISSING), f"{path}.task_plan_fingerprint"
        ),
        policy_abi_fingerprint=parser.string(
            raw.get("policy_abi_fingerprint", _MISSING), f"{path}.policy_abi_fingerprint"
        ),
        backend_plan_fingerprint=parser.string(
            raw.get("backend_plan_fingerprint", _MISSING), f"{path}.backend_plan_fingerprint"
        ),
    )


def _parse_prerequisites(parser: _Parser, value: Any, path: str) -> tuple[RolloutPrerequisite, ...]:
    if not isinstance(value, list) or not value:
        parser.errors.append(f"{path}: expected non-empty list")
        return ()
    result: list[RolloutPrerequisite] = []
    for index, item in enumerate(value):
        item_path = f"{path}[{index}]"
        raw = parser.mapping(item, item_path, _PREREQUISITE_KEYS)
        result.append(
            RolloutPrerequisite(
                phase=parser.integer(raw.get("phase", _MISSING), f"{item_path}.phase", minimum=1),
                claim_id=parser.string(raw.get("claim_id", _MISSING), f"{item_path}.claim_id"),
                test_id=parser.string(raw.get("test_id", _MISSING), f"{item_path}.test_id"),
            )
        )
    duplicates = sorted(
        claim_id
        for claim_id, count in Counter(item.claim_id for item in result).items()
        if count > 1
    )
    if duplicates:
        parser.errors.append(f"{path}: duplicate claim IDs {duplicates!r}")
    return tuple(result)


def parse_task_rollout_plan(raw: Any, *, source: Path = Path("<memory>")) -> TaskRolloutPlan:
    """Parse rollout metadata without consulting mutable repository state."""

    parser = _Parser()
    values = parser.mapping(raw, "rollout", _ROOT_KEYS)
    schema_version = parser.integer(
        values.get("schema_version", _MISSING), "schema_version", minimum=1
    )
    issue = parser.integer(values.get("issue", _MISSING), "issue", minimum=1)
    claim_id = parser.string(values.get("claim_id", _MISSING), "claim_id")
    plan_fingerprint = parser.string(values.get("plan_fingerprint", _MISSING), "plan_fingerprint")
    raw_entries = values.get("entries", _MISSING)
    if not isinstance(raw_entries, list) or not raw_entries:
        parser.errors.append("entries: expected non-empty list")
        raw_entries = []
    entries: list[TaskRolloutEntry] = []
    for index, item in enumerate(raw_entries):
        path = f"entries[{index}]"
        entry = parser.mapping(item, path, _ENTRY_KEYS)
        entries.append(
            TaskRolloutEntry(
                entrypoint_id=parser.string(
                    entry.get("entrypoint_id", _MISSING), f"{path}.entrypoint_id"
                ),
                task_slug=parser.string(entry.get("task_slug", _MISSING), f"{path}.task_slug"),
                env_name=parser.string(entry.get("env_name", _MISSING), f"{path}.env_name"),
                backend=parser.string(entry.get("backend", _MISSING), f"{path}.backend"),
                execution_profile=parser.string(
                    entry.get("execution_profile", _MISSING), f"{path}.execution_profile"
                ),
                owner_yaml=parser.repo_path(
                    entry.get("owner_yaml", _MISSING), f"{path}.owner_yaml"
                ),
                owner_yaml_sha256=parser.sha256(
                    entry.get("owner_yaml_sha256", _MISSING), f"{path}.owner_yaml_sha256"
                ),
                seeds=parser.integer_list(entry.get("seeds", _MISSING), f"{path}.seeds"),
                num_envs=parser.integer(
                    entry.get("num_envs", _MISSING), f"{path}.num_envs", minimum=1
                ),
                num_steps_per_env=parser.integer(
                    entry.get("num_steps_per_env", _MISSING),
                    f"{path}.num_steps_per_env",
                    minimum=1,
                ),
                max_iterations=parser.integer(
                    entry.get("max_iterations", _MISSING),
                    f"{path}.max_iterations",
                    minimum=1,
                ),
                runtime_impl=parser.string(
                    entry.get("runtime_impl", _MISSING), f"{path}.runtime_impl"
                ),
                runtime_resolver=parser.string(
                    entry.get("runtime_resolver", _MISSING), f"{path}.runtime_resolver"
                ),
                required_capabilities=parser.string_list(
                    entry.get("required_capabilities", _MISSING),
                    f"{path}.required_capabilities",
                    allow_empty=False,
                    canonical=True,
                ),
                disabled_domain_rand=parser.string_list(
                    entry.get("disabled_domain_rand", _MISSING),
                    f"{path}.disabled_domain_rand",
                    allow_empty=False,
                    canonical=True,
                ),
                expected_model_targets=parser.string_list(
                    entry.get("expected_model_targets", _MISSING),
                    f"{path}.expected_model_targets",
                    allow_empty=True,
                    canonical=True,
                ),
                support_compiled_signature=_parse_signature(
                    parser,
                    entry.get("support_compiled_signature", _MISSING),
                    f"{path}.support_compiled_signature",
                ),
                rollout_compiled_signature=_parse_signature(
                    parser,
                    entry.get("rollout_compiled_signature", _MISSING),
                    f"{path}.rollout_compiled_signature",
                ),
                prerequisites=_parse_prerequisites(
                    parser, entry.get("prerequisites", _MISSING), f"{path}.prerequisites"
                ),
            )
        )

    duplicates = sorted(
        key for key, count in Counter(item.key for item in entries).items() if count > 1
    )
    if duplicates:
        parser.errors.append(f"entries: duplicate keys {duplicates!r}")
    if schema_version != SCHEMA_VERSION:
        parser.errors.append(f"schema_version: expected {SCHEMA_VERSION}, got {schema_version}")
    if parser.errors:
        raise TaskRolloutPlanError(source, parser.errors)
    return TaskRolloutPlan(
        schema_version=schema_version,
        issue=issue,
        claim_id=claim_id,
        plan_fingerprint=plan_fingerprint,
        entries=tuple(entries),
        source=source,
    )


def load_task_rollout_plan(path: Path) -> TaskRolloutPlan:
    """Load a strict rollout plan from YAML."""

    try:
        raw = OmegaConf.to_container(OmegaConf.load(path), resolve=False)
    except Exception as exc:  # noqa: BLE001 - normalize malformed YAML for CLI callers
        raise TaskRolloutPlanError(
            path, [f"cannot load YAML: {type(exc).__name__}: {exc}"]
        ) from exc
    return parse_task_rollout_plan(raw, source=path)


def _load_owner(path: Path, *, label: str, errors: list[str]) -> Mapping[str, Any] | None:
    try:
        raw = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    except Exception as exc:  # noqa: BLE001 - collect all owner faults
        errors.append(f"{label}: cannot load owner YAML: {type(exc).__name__}: {exc}")
        return None
    if not isinstance(raw, Mapping):
        errors.append(f"{label}: owner YAML must contain a mapping")
        return None
    return raw


def _validate_entry_identity(
    entry: TaskRolloutEntry,
    support: SupportCombination,
    *,
    root: Path,
    registry_backends: Mapping[str, set[str]],
    errors: list[str],
) -> None:
    label = "/".join(entry.key)
    if support.evidence_level not in {
        DeclaredEvidenceLevel.BENCHMARKED,
        DeclaredEvidenceLevel.RECOMMENDED,
    }:
        errors.append(f"{label}: rollout requires benchmarked or recommended support evidence")
    for name in ("env_name", "execution_profile", "owner_yaml", "owner_yaml_sha256"):
        if getattr(entry, name) != getattr(support, name):
            errors.append(f"{label}: rollout {name} differs from support evidence")
    if support.compiled_signature is None:
        errors.append(f"{label}: benchmarked support evidence has no compiled signature")
    elif entry.support_compiled_signature != support.compiled_signature:
        errors.append(f"{label}: support_compiled_signature differs from support evidence")
    for name in ("task_key", "executor_key", "policy_abi_fingerprint"):
        if getattr(entry.support_compiled_signature, name) != getattr(
            entry.rollout_compiled_signature, name
        ):
            errors.append(f"{label}: rollout and support compiled {name} differ")
    if not entry.rollout_compiled_signature.task_plan_fingerprint.startswith(
        "manager-task-contract-v1:"
    ):
        errors.append(f"{label}: rollout task fingerprint has an unknown contract version")
    if not entry.rollout_compiled_signature.backend_plan_fingerprint.startswith(
        "mjwarp-device-batch-v1:"
    ):
        errors.append(f"{label}: rollout backend fingerprint has an unknown contract version")
    if entry.backend not in registry_backends.get(entry.env_name, set()):
        errors.append(f"{label}: env/backend identity is not registered")
    if entry.seeds != (0, 1):
        errors.append(f"{label}: frozen rollout seeds must be [0, 1]")
    if (entry.num_envs, entry.num_steps_per_env, entry.max_iterations) != (128, 2, 1):
        errors.append(f"{label}: frozen rollout budget must be envs=128 steps=2 iterations=1")
    if entry.runtime_impl != _EXPECTED_RUNTIME_IMPL:
        errors.append(f"{label}: runtime_impl differs from the production device owner")
    if entry.runtime_resolver != _EXPECTED_RUNTIME_RESOLVER:
        errors.append(f"{label}: runtime_resolver differs from the production device owner")
    if entry.required_capabilities != _EXPECTED_CAPABILITIES:
        errors.append(f"{label}: required capability set differs from the frozen G1 plan")
    if entry.disabled_domain_rand != _EXPECTED_DISABLED_DR:
        errors.append(f"{label}: disabled DR set differs from the frozen G1 owner contract")
    if entry.expected_model_targets:
        errors.append(f"{label}: owner-disabled DR rollout must have no Model targets")

    owner_path = root / entry.owner_yaml
    if not owner_path.is_file():
        errors.append(f"{label}: owner YAML is missing")
        return
    if sha256_file(owner_path) != entry.owner_yaml_sha256:
        errors.append(f"{label}: owner YAML hash does not match current content")
    owner = _load_owner(owner_path, label=label, errors=errors)
    if owner is None:
        return
    training = owner.get("training")
    algo = owner.get("algo")
    env = owner.get("env")
    play = owner.get("play_profile")
    if not isinstance(training, Mapping) or (
        training.get("task_name"),
        training.get("sim_backend"),
        training.get("execution_profile"),
    ) != (entry.env_name, entry.backend, entry.execution_profile):
        errors.append(f"{label}: owner training identity differs from rollout plan")
    if not isinstance(algo, Mapping) or (
        algo.get("runtime_impl"),
        algo.get("runtime_resolver"),
    ) != (entry.runtime_impl, entry.runtime_resolver):
        errors.append(f"{label}: owner runtime route differs from rollout plan")
    domain_rand = env.get("domain_rand") if isinstance(env, Mapping) else None
    if not isinstance(domain_rand, Mapping) or any(
        domain_rand.get(name) is not False for name in entry.disabled_domain_rand
    ):
        errors.append(f"{label}: owner does not explicitly disable every frozen DR field")
    noise = env.get("noise_config") if isinstance(env, Mapping) else None
    if not isinstance(noise, Mapping) or float(noise.get("level", -1.0)) != 0.0:
        errors.append(f"{label}: owner observation noise must be disabled")
    if (
        not isinstance(play, Mapping)
        or play.get("enabled") is not False
        or not isinstance(training, Mapping)
        or training.get("no_play") is not True
        or training.get("play_render_mode") != "none"
    ):
        errors.append(f"{label}: owner render/play path must remain explicitly disabled")


def _validate_prerequisites(
    entry: TaskRolloutEntry,
    *,
    root: Path,
    phase_manifests: Mapping[int, Any],
    existing_test_ids: set[str],
    errors: list[str],
) -> None:
    label = "/".join(entry.key)
    actual = {item.claim_id: (item.phase, item.test_id) for item in entry.prerequisites}
    if actual != _EXPECTED_PREREQUISITES:
        errors.append(f"{label}: prerequisite claim/test matrix differs from frozen plan")
    for prerequisite in entry.prerequisites:
        prefix = f"{label}/{prerequisite.claim_id}"
        manifest = phase_manifests.get(prerequisite.phase)
        if manifest is None:
            errors.append(f"{prefix}: prerequisite phase manifest is unavailable")
            continue
        claim = next(
            (item for item in manifest.claims if item.claim_id == prerequisite.claim_id), None
        )
        if claim is None:
            errors.append(f"{prefix}: claim is absent from Phase {prerequisite.phase}")
            continue
        if claim.status != ClaimStatus.VERIFIED or claim.evidence.result != EvidenceResult.PASS:
            errors.append(f"{prefix}: prerequisite is not verified PASS")
        if claim.required_test_ids != (prerequisite.test_id,):
            errors.append(f"{prefix}: required test differs from rollout plan")
        if claim.evidence.executed_test_ids != (prerequisite.test_id,):
            errors.append(f"{prefix}: executed test differs from rollout plan")
        if claim.evidence.skipped_test_ids or claim.evidence.xfailed_test_ids:
            errors.append(f"{prefix}: skipped or xfailed prerequisite is forbidden")
        if prerequisite.test_id not in existing_test_ids:
            errors.append(f"{prefix}: test is not existing in claim inventory")
        if not test_node_exists(root, prerequisite.test_id):
            errors.append(f"{prefix}: exact pytest node does not exist")


def audit_task_rollout_plan(
    plan: TaskRolloutPlan,
    *,
    root: Path,
    registry_backends: Mapping[str, set[str]] | None = None,
    validate_support_payloads: bool = True,
) -> TaskRolloutAuditReport:
    """Audit plan, support declaration, owners, and prerequisite receipts."""

    root = root.resolve()
    errors: list[str] = []
    if plan.schema_version != SCHEMA_VERSION:
        errors.append(f"schema_version: expected {SCHEMA_VERSION}")
    if plan.issue != ISSUE or plan.claim_id != CLAIM_ID:
        errors.append(f"identity: expected issue {ISSUE} claim {CLAIM_ID}")
    if plan.plan_fingerprint != PLAN_FINGERPRINT:
        errors.append(f"plan_fingerprint: expected {PLAN_FINGERPRINT}")

    try:
        support_manifest = load_support_evidence(root / SUPPORT_EVIDENCE_PATH)
        support_report = audit_support_evidence(
            support_manifest,
            root=root,
            registry_backends=registry_backends,
            validate_phase_payloads=validate_support_payloads,
            validate_benchmark_payloads=validate_support_payloads,
        )
        errors.extend(f"support audit: {error}" for error in support_report.errors)
    except (OSError, RuntimeError, SupportEvidenceError, ValueError) as exc:
        errors.append(f"support audit: {type(exc).__name__}: {exc}")
        return TaskRolloutAuditReport(
            entries=len(plan.entries),
            prerequisites=sum(len(item.prerequisites) for item in plan.entries),
            errors=tuple(errors),
        )

    promoted = {
        item.key: item
        for item in support_manifest.combinations
        if item.evidence_level
        in {DeclaredEvidenceLevel.BENCHMARKED, DeclaredEvidenceLevel.RECOMMENDED}
    }
    by_key = {item.key: item for item in plan.entries}
    if set(by_key) != set(promoted):
        errors.append("rollout entries and benchmarked/recommended support combinations differ")
    if set(by_key) != {_EXPECTED_KEY}:
        errors.append("mjwarp-task-rollout-v1 must contain exactly the G1 pilot combination")

    snapshot = registry_backends if registry_backends is not None else snapshot_registry_backends()
    for key, entry in by_key.items():
        support = promoted.get(key)
        if support is not None:
            _validate_entry_identity(
                entry,
                support,
                root=root,
                registry_backends=snapshot,
                errors=errors,
            )

    phase_manifests: dict[int, Any] = {}
    for gate in support_manifest.phase_gates:
        try:
            phase_manifests[gate.phase] = load_phase_acceptance(root / gate.manifest)
        except ManifestValidationError as exc:
            errors.extend(f"Phase {gate.phase} manifest: {error}" for error in exc.errors)
    try:
        inventory = load_claim_gap_inventory(root / CLAIM_INVENTORY_PATH)
        existing_test_ids = {
            entry.test_id
            for entry in inventory.entries
            if entry.state == InventoryTestState.EXISTING
        }
    except ValueError as exc:
        errors.append(f"claim inventory: {exc}")
        existing_test_ids = set()
    for entry in plan.entries:
        _validate_prerequisites(
            entry,
            root=root,
            phase_manifests=phase_manifests,
            existing_test_ids=existing_test_ids,
            errors=errors,
        )

    return TaskRolloutAuditReport(
        entries=len(plan.entries),
        prerequisites=sum(len(item.prerequisites) for item in plan.entries),
        errors=tuple(errors),
    )


__all__ = [
    "CLAIM_ID",
    "ISSUE",
    "PLAN_FINGERPRINT",
    "ROLLOUT_PLAN_PATH",
    "RolloutPrerequisite",
    "SCHEMA_VERSION",
    "TaskRolloutAuditReport",
    "TaskRolloutEntry",
    "TaskRolloutPlan",
    "TaskRolloutPlanError",
    "audit_task_rollout_plan",
    "load_task_rollout_plan",
    "parse_task_rollout_plan",
]
