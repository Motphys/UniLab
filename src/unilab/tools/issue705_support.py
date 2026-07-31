"""Fail-closed support evidence and bidirectional audit for Issue #705.

The generated support matrix is a view, not a source of truth.  This module
binds every high-grade ``mjwarp`` claim to an owner, registry identity, phase
gates, a validated benchmark, and the compiled signatures observed in that
benchmark.  Repository audits traverse those inputs independently so adding a
YAML file or registry decorator cannot silently advertise a new combination.
"""

from __future__ import annotations

import ast
import json
import re
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

from omegaconf import OmegaConf

from unilab.base import registry
from unilab.base.registry import ensure_registries
from unilab.tools.claim_gap_audit import (
    EvidenceRole,
    InventoryTestState,
    load_claim_gap_inventory,
)
from unilab.tools.issue705_phase_evidence import sha256_file
from unilab.tools.phase_acceptance import (
    ClaimStatus,
    EvidenceResult,
    ManifestValidationError,
    load_phase_acceptance,
)

SCHEMA_VERSION = 1
ISSUE = 705
CLAIM_ID = "P7-SUPPORT-MATRIX"
INTEGRATION_BRANCH = "feat/issue-705-manager-mjwarp"
SUPPORT_EVIDENCE_PATH = Path("tests/acceptance/issue_705/support_evidence.yaml")
CLAIM_INVENTORY_PATH = Path("tests/acceptance/issue_705/claim_test_inventory.yaml")
SUPPORT_AUDIT_TEST_ID = (
    "tests/scripts/test_issue705_support_audit.py::"
    "test_supported_combinations_have_fresh_bidirectional_evidence"
)
REQUIRED_PHASES = (2, 3, 5, 6)
BENCHMARK_VALIDATOR = "issue705_phase5_ppo_v1"
BENCHMARK_ARTIFACT_PATH = Path("tests/acceptance/issue_705/artifacts/phase_5_mjwarp_ppo.json")
BENCHMARK_TEST_ID = (
    "tests/benchmark/test_mjwarp_ppo_benchmark.py::test_device_profile_meets_end_to_end_gate"
)

_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_TEST_ID_RE = re.compile(
    r"^(?P<path>tests/[^:\s]+\.py)"
    r"(?P<nodes>(?:::[A-Za-z_][A-Za-z0-9_]*(?:\[[^\]\r\n]+\])?)*)$"
)
_MISSING = object()

_ENTRYPOINT_OWNER_ROOTS: dict[str, Path] = {
    "ppo_torch": Path("conf/ppo/task"),
    "ppo_mlx": Path("conf/ppo/task"),
    "appo_torch": Path("conf/appo/task"),
    "sac_torch": Path("conf/offpolicy/task/sac"),
    "td3_torch": Path("conf/offpolicy/task/td3"),
    "flashsac_torch": Path("conf/offpolicy/task/flashsac"),
}


class DeclaredEvidenceLevel(str, Enum):
    """Evidence levels that require an explicit per-combination declaration."""

    TESTED = "tested"
    BENCHMARKED = "benchmarked"
    RECOMMENDED = "recommended"


@dataclass(frozen=True)
class PhaseGateRef:
    phase: int
    manifest: Path
    gate_artifact: Path
    sha256: str


@dataclass(frozen=True)
class BenchmarkRef:
    path: Path
    sha256: str
    validator: str


@dataclass(frozen=True)
class CompiledSignature:
    task_key: str
    executor_key: str
    task_plan_fingerprint: str
    policy_abi_fingerprint: str
    backend_plan_fingerprint: str


@dataclass(frozen=True)
class SupportCombination:
    entrypoint_id: str
    task_slug: str
    env_name: str
    backend: str
    execution_profile: str
    evidence_level: DeclaredEvidenceLevel
    owner_yaml: Path
    owner_yaml_sha256: str
    required_phase: int
    mandatory_test_ids: tuple[str, ...]
    benchmark: BenchmarkRef | None
    compiled_signature: CompiledSignature | None

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.entrypoint_id, self.task_slug, self.backend)


@dataclass(frozen=True)
class SupportEvidenceManifest:
    schema_version: int
    issue: int
    claim_id: str
    integration_branch: str
    phase_gates: tuple[PhaseGateRef, ...]
    combinations: tuple[SupportCombination, ...]
    source: Path


@dataclass(frozen=True)
class SupportAuditReport:
    combinations: int
    benchmarked: int
    recommended: int
    phase_gates: int
    errors: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.errors


class SupportEvidenceError(ValueError):
    def __init__(self, source: Path, errors: list[str] | tuple[str, ...]) -> None:
        self.source = source
        self.errors = tuple(errors)
        detail = "\n".join(f"- {error}" for error in self.errors)
        super().__init__(f"invalid Issue #705 support evidence {source}:\n{detail}")


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
        if value is _MISSING:
            return ""
        if not isinstance(value, str):
            self.errors.append(f"{path}: expected string")
            return ""
        if not value.strip():
            self.errors.append(f"{path}: must not be empty")
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
        text = self.string(value, path)
        result = Path(text)
        if result.is_absolute() or ".." in result.parts or "." in result.parts:
            self.errors.append(f"{path}: must stay within the repository")
        return result

    def sha256(self, value: Any, path: str) -> str:
        text = self.string(value, path)
        if text and _SHA256_RE.fullmatch(text) is None:
            self.errors.append(f"{path}: expected sha256:<64 lowercase hex characters>")
        return text

    def enum(
        self,
        value: Any,
        path: str,
        enum_type: type[DeclaredEvidenceLevel],
    ) -> DeclaredEvidenceLevel:
        try:
            return enum_type(value)
        except (TypeError, ValueError):
            self.errors.append(f"{path}: expected one of {[item.value for item in enum_type]!r}")
            return DeclaredEvidenceLevel.TESTED

    def string_list(self, value: Any, path: str) -> tuple[str, ...]:
        if not isinstance(value, list) or not value:
            self.errors.append(f"{path}: expected non-empty string list")
            return ()
        result = tuple(self.string(item, f"{path}[{index}]") for index, item in enumerate(value))
        duplicates = sorted(item for item, count in Counter(result).items() if count > 1)
        if duplicates:
            self.errors.append(f"{path}: duplicate values {duplicates!r}")
        return result


_ROOT_KEYS = (
    "schema_version",
    "issue",
    "claim_id",
    "integration_branch",
    "phase_gates",
    "combinations",
)
_PHASE_GATE_KEYS = ("phase", "manifest", "gate_artifact", "sha256")
_COMBINATION_KEYS = (
    "entrypoint_id",
    "task_slug",
    "env_name",
    "backend",
    "execution_profile",
    "evidence_level",
    "owner_yaml",
    "owner_yaml_sha256",
    "required_phase",
    "mandatory_test_ids",
    "benchmark",
    "compiled_signature",
)
_BENCHMARK_KEYS = ("path", "sha256", "validator")
_SIGNATURE_KEYS = (
    "task_key",
    "executor_key",
    "task_plan_fingerprint",
    "policy_abi_fingerprint",
    "backend_plan_fingerprint",
)


def _parse_benchmark(parser: _Parser, raw: Any, path: str) -> BenchmarkRef | None:
    if raw is None:
        return None
    values = parser.mapping(raw, path, _BENCHMARK_KEYS)
    return BenchmarkRef(
        path=parser.repo_path(values.get("path", _MISSING), f"{path}.path"),
        sha256=parser.sha256(values.get("sha256", _MISSING), f"{path}.sha256"),
        validator=parser.string(values.get("validator", _MISSING), f"{path}.validator"),
    )


def _parse_signature(parser: _Parser, raw: Any, path: str) -> CompiledSignature | None:
    if raw is None:
        return None
    values = parser.mapping(raw, path, _SIGNATURE_KEYS)
    return CompiledSignature(
        task_key=parser.string(values.get("task_key", _MISSING), f"{path}.task_key"),
        executor_key=parser.string(values.get("executor_key", _MISSING), f"{path}.executor_key"),
        task_plan_fingerprint=parser.string(
            values.get("task_plan_fingerprint", _MISSING), f"{path}.task_plan_fingerprint"
        ),
        policy_abi_fingerprint=parser.string(
            values.get("policy_abi_fingerprint", _MISSING),
            f"{path}.policy_abi_fingerprint",
        ),
        backend_plan_fingerprint=parser.string(
            values.get("backend_plan_fingerprint", _MISSING),
            f"{path}.backend_plan_fingerprint",
        ),
    )


def parse_support_evidence(raw: Any, *, source: Path = Path("<memory>")) -> SupportEvidenceManifest:
    """Parse a versioned support declaration without consulting repository state."""

    parser = _Parser()
    values = parser.mapping(raw, "support", _ROOT_KEYS)
    schema_version = parser.integer(
        values.get("schema_version", _MISSING), "schema_version", minimum=1
    )
    if schema_version != SCHEMA_VERSION:
        parser.errors.append(f"schema_version: expected {SCHEMA_VERSION}, got {schema_version!r}")
    issue = parser.integer(values.get("issue", _MISSING), "issue", minimum=1)
    claim_id = parser.string(values.get("claim_id", _MISSING), "claim_id")
    integration_branch = parser.string(
        values.get("integration_branch", _MISSING), "integration_branch"
    )

    raw_gates = values.get("phase_gates", _MISSING)
    if not isinstance(raw_gates, list) or not raw_gates:
        parser.errors.append("phase_gates: expected non-empty list")
        raw_gates = []
    phase_gates: list[PhaseGateRef] = []
    for index, raw_gate in enumerate(raw_gates):
        path = f"phase_gates[{index}]"
        gate = parser.mapping(raw_gate, path, _PHASE_GATE_KEYS)
        phase_gates.append(
            PhaseGateRef(
                phase=parser.integer(gate.get("phase", _MISSING), f"{path}.phase"),
                manifest=parser.repo_path(gate.get("manifest", _MISSING), f"{path}.manifest"),
                gate_artifact=parser.repo_path(
                    gate.get("gate_artifact", _MISSING), f"{path}.gate_artifact"
                ),
                sha256=parser.sha256(gate.get("sha256", _MISSING), f"{path}.sha256"),
            )
        )

    raw_combinations = values.get("combinations", _MISSING)
    if not isinstance(raw_combinations, list) or not raw_combinations:
        parser.errors.append("combinations: expected non-empty list")
        raw_combinations = []
    combinations: list[SupportCombination] = []
    for index, raw_combination in enumerate(raw_combinations):
        path = f"combinations[{index}]"
        item = parser.mapping(raw_combination, path, _COMBINATION_KEYS)
        combinations.append(
            SupportCombination(
                entrypoint_id=parser.string(
                    item.get("entrypoint_id", _MISSING), f"{path}.entrypoint_id"
                ),
                task_slug=parser.string(item.get("task_slug", _MISSING), f"{path}.task_slug"),
                env_name=parser.string(item.get("env_name", _MISSING), f"{path}.env_name"),
                backend=parser.string(item.get("backend", _MISSING), f"{path}.backend"),
                execution_profile=parser.string(
                    item.get("execution_profile", _MISSING), f"{path}.execution_profile"
                ),
                evidence_level=parser.enum(
                    item.get("evidence_level", _MISSING),
                    f"{path}.evidence_level",
                    DeclaredEvidenceLevel,
                ),
                owner_yaml=parser.repo_path(item.get("owner_yaml", _MISSING), f"{path}.owner_yaml"),
                owner_yaml_sha256=parser.sha256(
                    item.get("owner_yaml_sha256", _MISSING), f"{path}.owner_yaml_sha256"
                ),
                required_phase=parser.integer(
                    item.get("required_phase", _MISSING), f"{path}.required_phase"
                ),
                mandatory_test_ids=parser.string_list(
                    item.get("mandatory_test_ids", _MISSING), f"{path}.mandatory_test_ids"
                ),
                benchmark=_parse_benchmark(
                    parser, item.get("benchmark", _MISSING), f"{path}.benchmark"
                ),
                compiled_signature=_parse_signature(
                    parser,
                    item.get("compiled_signature", _MISSING),
                    f"{path}.compiled_signature",
                ),
            )
        )

    duplicate_phases = sorted(
        phase for phase, count in Counter(gate.phase for gate in phase_gates).items() if count > 1
    )
    if duplicate_phases:
        parser.errors.append(f"phase_gates: duplicate phases {duplicate_phases!r}")
    duplicate_keys = sorted(
        key
        for key, count in Counter(combination.key for combination in combinations).items()
        if count > 1
    )
    if duplicate_keys:
        parser.errors.append(f"combinations: duplicate keys {duplicate_keys!r}")
    if parser.errors:
        raise SupportEvidenceError(source, parser.errors)
    return SupportEvidenceManifest(
        schema_version=schema_version,
        issue=issue,
        claim_id=claim_id,
        integration_branch=integration_branch,
        phase_gates=tuple(phase_gates),
        combinations=tuple(combinations),
        source=source,
    )


def load_support_evidence(path: Path) -> SupportEvidenceManifest:
    """Load strict support metadata from YAML."""

    try:
        raw = OmegaConf.to_container(OmegaConf.load(path), resolve=False)
    except Exception as exc:  # noqa: BLE001 - normalize malformed YAML for the CLI
        raise SupportEvidenceError(
            path, [f"cannot load YAML: {type(exc).__name__}: {exc}"]
        ) from exc
    return parse_support_evidence(raw, source=path)


def snapshot_registry_backends() -> dict[str, set[str]]:
    """Return the independently bootstrapped env/backend registry view."""

    ensure_registries()
    result: dict[str, set[str]] = {}
    for env_name, metadata in registry.list_registered_envs().items():
        backends = metadata.get("available_backends")
        if isinstance(backends, list) and all(isinstance(item, str) for item in backends):
            result[env_name] = set(backends)
    return result


def _load_json(path: Path, *, label: str, errors: list[str]) -> dict[str, Any] | None:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        errors.append(f"{label}: cannot load JSON: {type(exc).__name__}: {exc}")
        return None
    if not isinstance(raw, dict):
        errors.append(f"{label}: expected JSON object")
        return None
    return raw


def _validate_phase_payload(
    phase: int, report: Mapping[str, Any], *, root: Path
) -> tuple[str, ...]:
    if phase == 2:
        from unilab.tools.issue705_phase2_evidence import validate_phase2_evidence

        return validate_phase2_evidence(report, root=root)
    if phase == 3:
        from unilab.tools.issue705_phase3_evidence import validate_phase3_evidence

        return validate_phase3_evidence(report, root=root)
    if phase == 5:
        from unilab.tools.issue705_phase5_evidence import validate_phase5_evidence

        return validate_phase5_evidence(report, root=root)
    if phase == 6:
        from unilab.tools.issue705_phase6_evidence import validate_phase6_evidence

        return validate_phase6_evidence(report, root=root)
    raise ValueError(f"no Issue #705 phase evidence validator is registered for phase {phase}")


def _test_node_exists(root: Path, test_id: str) -> bool:
    match = _TEST_ID_RE.fullmatch(test_id)
    if match is None:
        return False
    path = root / match.group("path")
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, UnicodeError, SyntaxError):
        return False
    nodes = tuple(part for part in match.group("nodes").split("::") if part)
    if not nodes:
        return False
    body: list[ast.stmt] = tree.body
    for index, raw_name in enumerate(nodes):
        name = raw_name.split("[", maxsplit=1)[0]
        found = next(
            (
                node
                for node in body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                and node.name == name
            ),
            None,
        )
        if found is None:
            return False
        if index < len(nodes) - 1:
            if not isinstance(found, ast.ClassDef):
                return False
            body = found.body
    return True


def _owner_values(path: Path, *, label: str, errors: list[str]) -> dict[str, Any] | None:
    try:
        raw = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    except Exception as exc:  # noqa: BLE001 - collect every audit fault
        errors.append(f"{label}: cannot load owner YAML: {type(exc).__name__}: {exc}")
        return None
    if not isinstance(raw, dict):
        errors.append(f"{label}: owner YAML must contain a mapping")
        return None
    if not all(isinstance(key, str) for key in raw):
        errors.append(f"{label}: owner YAML keys must be strings")
        return None
    return {key: value for key, value in raw.items() if isinstance(key, str)}


def _discover_mjwarp_owner_paths(root: Path) -> set[Path]:
    return {
        path.relative_to(root) for path in (root / "conf").glob("**/mjwarp.yaml") if path.is_file()
    }


def _validate_phase_gates(
    support: SupportEvidenceManifest,
    *,
    root: Path,
    validate_payloads: bool,
    errors: list[str],
) -> dict[int, tuple[dict[str, Any], Any]]:
    by_phase = {gate.phase: gate for gate in support.phase_gates}
    if set(by_phase) != set(REQUIRED_PHASES):
        errors.append(
            f"phase_gates: expected exactly {list(REQUIRED_PHASES)!r}, got {sorted(by_phase)!r}"
        )
    loaded: dict[int, tuple[dict[str, Any], Any]] = {}
    for phase, gate in sorted(by_phase.items()):
        expected_manifest = Path(f"tests/acceptance/issue_705/manifests/phase_{phase}.yaml")
        expected_artifact = Path(f"tests/acceptance/issue_705/artifacts/phase_{phase}_gate.json")
        if gate.manifest != expected_manifest:
            errors.append(f"phase {phase}: manifest path is not canonical")
        if gate.gate_artifact != expected_artifact:
            errors.append(f"phase {phase}: gate artifact path is not canonical")
        artifact_path = root / gate.gate_artifact
        if not artifact_path.is_file():
            errors.append(f"phase {phase}: gate artifact is missing: {gate.gate_artifact}")
            continue
        if sha256_file(artifact_path) != gate.sha256:
            errors.append(f"phase {phase}: gate artifact hash does not match support metadata")
            continue
        report = _load_json(artifact_path, label=f"phase {phase} gate", errors=errors)
        if report is None:
            continue
        try:
            manifest = load_phase_acceptance(root / gate.manifest)
        except ManifestValidationError as exc:
            errors.extend(f"phase {phase} manifest: {error}" for error in exc.errors)
            continue
        loaded[phase] = (report, manifest)
        if manifest.issue != ISSUE or manifest.phase != phase:
            errors.append(f"phase {phase}: manifest issue/phase identity does not match")
        if manifest.integration_branch != INTEGRATION_BRANCH:
            errors.append(f"phase {phase}: manifest integration branch does not match")

        source = report.get("source")
        source_commit = source.get("commit_sha") if isinstance(source, dict) else None
        if (
            not isinstance(source_commit, str)
            or re.fullmatch(r"[0-9a-f]{40}", source_commit) is None
        ):
            errors.append(f"phase {phase}: gate source commit is missing or malformed")
            source_commit = None

        raw_artifact_claims = report.get("claims")
        artifact_claims: dict[str, Mapping[str, Any]] = {}
        if not isinstance(raw_artifact_claims, list):
            errors.append(f"phase {phase}: gate claims are missing")
        else:
            for index, raw_claim in enumerate(raw_artifact_claims):
                claim_id = raw_claim.get("claim_id") if isinstance(raw_claim, Mapping) else None
                if not isinstance(claim_id, str):
                    errors.append(f"phase {phase}: gate claim[{index}] is malformed")
                elif claim_id in artifact_claims:
                    errors.append(f"phase {phase}: duplicate gate claim {claim_id}")
                else:
                    artifact_claims[claim_id] = raw_claim

        manifest_claim_ids = {claim.claim_id for claim in manifest.claims}
        if set(artifact_claims) != manifest_claim_ids:
            errors.append(f"phase {phase}: manifest and gate claim sets differ")
        inputs = report.get("inputs")
        config_hashes = inputs.get("claim_config_hashes") if isinstance(inputs, dict) else None
        for claim in manifest.claims:
            if claim.status != ClaimStatus.VERIFIED or claim.evidence.result != EvidenceResult.PASS:
                errors.append(
                    f"phase {phase}/{claim.claim_id}: required evidence is not verified PASS"
                )
            if gate.gate_artifact.as_posix() not in claim.evidence.artifact_refs:
                errors.append(f"phase {phase}/{claim.claim_id}: manifest does not cite its gate")
            if claim.evidence.skipped_test_ids or claim.evidence.xfailed_test_ids:
                errors.append(
                    f"phase {phase}/{claim.claim_id}: skipped or xfailed evidence is forbidden"
                )
            if source_commit is not None and claim.evidence.commit_sha != source_commit:
                errors.append(
                    f"phase {phase}/{claim.claim_id}: manifest commit does not match gate source"
                )
            if claim.evidence.executed_test_ids != claim.required_test_ids:
                errors.append(
                    f"phase {phase}/{claim.claim_id}: executed tests do not match required tests"
                )
            artifact_claim = artifact_claims.get(claim.claim_id)
            if (
                artifact_claim is not None
                and (artifact_claim.get("required_test_id"),) != claim.required_test_ids
            ):
                errors.append(
                    f"phase {phase}/{claim.claim_id}: manifest test does not match gate claim"
                )
            if isinstance(config_hashes, dict):
                artifact_config_hash = config_hashes.get(claim.claim_id)
                if (
                    not isinstance(artifact_config_hash, str)
                    or claim.evidence.config_hash != artifact_config_hash
                ):
                    errors.append(
                        f"phase {phase}/{claim.claim_id}: manifest config hash does not match gate"
                    )
        commands = report.get("commands")
        if not isinstance(commands, list) or not commands:
            errors.append(f"phase {phase}: gate has no command receipts")
        else:
            for index, command in enumerate(commands):
                if not isinstance(command, dict):
                    errors.append(f"phase {phase}: command[{index}] is malformed")
                    continue
                counts = command.get("pytest")
                if not isinstance(counts, dict) or any(
                    counts.get(category) != 0 for category in ("skipped", "xfailed", "xpassed")
                ):
                    errors.append(
                        f"phase {phase}: command[{index}] has skipped/xfail/xpass evidence"
                    )
        if validate_payloads:
            try:
                validation_errors = _validate_phase_payload(phase, report, root=root)
            except Exception as exc:  # noqa: BLE001 - validator faults must fail closed
                validation_errors = (f"validator raised {type(exc).__name__}: {exc}",)
            errors.extend(f"phase {phase} validator: {error}" for error in validation_errors)
    return loaded


def _validate_owner_and_registry(
    support: SupportEvidenceManifest,
    *,
    root: Path,
    registry_backends: Mapping[str, set[str]],
    errors: list[str],
) -> None:
    declared_owner_paths = {combination.owner_yaml for combination in support.combinations}
    discovered_owner_paths = _discover_mjwarp_owner_paths(root)
    if discovered_owner_paths != declared_owner_paths:
        missing = sorted(discovered_owner_paths - declared_owner_paths, key=Path.as_posix)
        stale = sorted(declared_owner_paths - discovered_owner_paths, key=Path.as_posix)
        if missing:
            errors.append(
                "owner audit: unmodeled mjwarp owner YAML: "
                + ", ".join(path.as_posix() for path in missing)
            )
        if stale:
            errors.append(
                "owner audit: metadata references missing mjwarp owner YAML: "
                + ", ".join(path.as_posix() for path in stale)
            )

    declared_registry = {(item.env_name, item.backend) for item in support.combinations}
    discovered_registry = {
        (env_name, "mjwarp")
        for env_name, backends in registry_backends.items()
        if "mjwarp" in backends
    }
    if discovered_registry != declared_registry:
        missing_registry = sorted(discovered_registry - declared_registry)
        stale_registry = sorted(declared_registry - discovered_registry)
        if missing_registry:
            errors.append(f"registry audit: unmodeled mjwarp identities {missing_registry!r}")
        if stale_registry:
            errors.append(
                f"registry audit: declared identities are not registered {stale_registry!r}"
            )

    for item in support.combinations:
        label = "/".join(item.key)
        if item.backend != "mjwarp":
            errors.append(f"{label}: Issue #705 support metadata currently accepts only mjwarp")
        owner_root = _ENTRYPOINT_OWNER_ROOTS.get(item.entrypoint_id)
        if owner_root is None:
            errors.append(f"{label}: unknown entrypoint_id {item.entrypoint_id!r}")
        else:
            expected_owner = owner_root / item.task_slug / f"{item.backend}.yaml"
            if item.owner_yaml != expected_owner:
                errors.append(
                    f"{label}: owner path does not match entrypoint/task/backend identity"
                )
        owner_path = root / item.owner_yaml
        if not owner_path.is_file():
            continue
        if sha256_file(owner_path) != item.owner_yaml_sha256:
            errors.append(f"{label}: owner YAML hash does not match current content")
        owner = _owner_values(owner_path, label=label, errors=errors)
        if owner is None:
            continue
        training = owner.get("training")
        if not isinstance(training, dict):
            errors.append(f"{label}: owner has no training mapping")
            continue
        if training.get("task_name") != item.env_name:
            errors.append(f"{label}: owner task_name does not match env identity")
        if training.get("sim_backend") != item.backend:
            errors.append(f"{label}: owner sim_backend does not match declaration")
        if training.get("execution_profile") != item.execution_profile:
            errors.append(f"{label}: owner execution_profile does not match declaration")


def _validate_claim_inventory_and_tests(
    support: SupportEvidenceManifest,
    *,
    root: Path,
    phase_data: Mapping[int, tuple[dict[str, Any], Any]],
    errors: list[str],
) -> None:
    try:
        inventory = load_claim_gap_inventory(root / CLAIM_INVENTORY_PATH)
    except ValueError as exc:
        errors.append(f"claim inventory: {exc}")
        return
    support_entries = [entry for entry in inventory.entries if entry.claim_id == CLAIM_ID]
    acceptance = [entry for entry in support_entries if entry.role == EvidenceRole.ACCEPTANCE]
    if len(acceptance) != 1 or acceptance[0].test_id != SUPPORT_AUDIT_TEST_ID:
        errors.append("claim inventory: P7 support acceptance node is not exact")
    elif acceptance[0].state != InventoryTestState.EXISTING or acceptance[0].gap is not None:
        errors.append("claim inventory: P7 support acceptance node is not existing")
    if not _test_node_exists(root, SUPPORT_AUDIT_TEST_ID):
        errors.append("claim inventory: P7 support acceptance test node is missing")

    existing_ids = {
        entry.test_id for entry in inventory.entries if entry.state == InventoryTestState.EXISTING
    }
    for combination in support.combinations:
        label = "/".join(combination.key)
        phase_pair = phase_data.get(combination.required_phase)
        if phase_pair is None:
            errors.append(f"{label}: required phase gate is unavailable")
            continue
        gate_report, phase_manifest = phase_pair
        inputs = gate_report.get("inputs")
        config_hashes = inputs.get("claim_config_hashes") if isinstance(inputs, dict) else None
        if not isinstance(config_hashes, dict) or combination.owner_yaml_sha256 not in set(
            config_hashes.values()
        ):
            errors.append(f"{label}: required phase gate does not bind the current owner hash")
        manifest_test_ids = {
            test_id for claim in phase_manifest.claims for test_id in claim.required_test_ids
        }
        gate_test_ids = {
            test_id
            for claim in gate_report.get("claims", [])
            if isinstance(claim, dict)
            for test_id in [claim.get("required_test_id")]
            if isinstance(test_id, str)
        }
        for test_id in combination.mandatory_test_ids:
            if not _test_node_exists(root, test_id):
                errors.append(f"{label}: mandatory test node is missing: {test_id}")
            if test_id not in existing_ids:
                errors.append(f"{label}: mandatory test is not existing in claim inventory")
            if test_id not in manifest_test_ids or test_id not in gate_test_ids:
                errors.append(f"{label}: mandatory test is not bound by the required phase gate")


def _device_cases(
    artifact: Mapping[str, Any], *, label: str, errors: list[str]
) -> list[dict[str, Any]]:
    raw_cases = artifact.get("cases")
    if not isinstance(raw_cases, list):
        errors.append(f"{label}: benchmark cases are missing")
        return []
    cases = [
        case for case in raw_cases if isinstance(case, dict) and case.get("mode") == "mjwarp_device"
    ]
    if not cases:
        errors.append(f"{label}: benchmark has no mjwarp_device cases")
    return cases


def _nested_mapping(value: object, *keys: str) -> Mapping[str, Any] | None:
    current = value
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current if isinstance(current, Mapping) else None


def _validate_benchmark(
    combination: SupportCombination,
    *,
    root: Path,
    validate_payload: bool,
    errors: list[str],
) -> None:
    label = "/".join(combination.key)
    benchmark = combination.benchmark
    signature = combination.compiled_signature
    if combination.evidence_level == DeclaredEvidenceLevel.TESTED:
        if benchmark is not None or signature is not None:
            errors.append(f"{label}: tested evidence must not carry benchmark-only receipts")
        return
    if benchmark is None or signature is None:
        errors.append(f"{label}: benchmarked/recommended evidence requires artifact and signatures")
        return
    if benchmark.validator != BENCHMARK_VALIDATOR:
        errors.append(f"{label}: unknown benchmark validator {benchmark.validator!r}")
    if benchmark.path != BENCHMARK_ARTIFACT_PATH:
        errors.append(f"{label}: benchmark artifact path is not canonical")
    if combination.required_phase != 5:
        errors.append(f"{label}: {combination.evidence_level.value} evidence requires Phase 5")
    if combination.mandatory_test_ids != (BENCHMARK_TEST_ID,):
        errors.append(f"{label}: benchmark mandatory test mapping is not exact")
    artifact_path = root / benchmark.path
    if not artifact_path.is_file():
        errors.append(f"{label}: benchmark artifact is missing: {benchmark.path}")
        return
    if sha256_file(artifact_path) != benchmark.sha256:
        errors.append(f"{label}: benchmark artifact hash does not match support metadata")
        return
    artifact = _load_json(artifact_path, label=f"{label} benchmark", errors=errors)
    if artifact is None:
        return
    if validate_payload:
        from unilab.tools.issue705_phase5_evidence import validate_ppo_benchmark_payload

        validation_errors = validate_ppo_benchmark_payload(
            artifact, root=root, artifact_path=artifact_path
        )
        errors.extend(f"{label} benchmark validator: {error}" for error in validation_errors)
    if artifact.get("profile") != combination.execution_profile:
        errors.append(f"{label}: benchmark profile does not match declaration")
    gate = artifact.get("gate")
    if not isinstance(gate, dict) or gate.get("passed") is not True or gate.get("errors") != []:
        errors.append(f"{label}: benchmark does not record a clean passing gate")

    expected_policy = {
        "task_key": signature.task_key,
        "executor_key": signature.executor_key,
        "plan_fingerprint": signature.task_plan_fingerprint,
        "policy_abi_fingerprint": signature.policy_abi_fingerprint,
        "execution_profile": combination.execution_profile,
    }
    for index, case in enumerate(_device_cases(artifact, label=label, errors=errors)):
        raw = case.get("raw")
        config = _nested_mapping(raw, "run_config", "config")
        training = config.get("training") if isinstance(config, Mapping) else None
        run = _nested_mapping(raw, "run_config", "run")
        policy = _nested_mapping(raw, "run_config", "contract_snapshot", "manager.policy_abi")
        summary = _nested_mapping(raw, "run_summary")
        if not isinstance(training, Mapping) or not isinstance(run, Mapping):
            errors.append(f"{label}: device case {index} lacks owner/backend identity")
            continue
        if (
            training.get("task_name") != combination.env_name
            or training.get("sim_backend") != combination.backend
            or training.get("execution_profile") != combination.execution_profile
            or run.get("task") != combination.env_name
            or run.get("sim_backend") != combination.backend
            or case.get("expected_execution_profile") != combination.execution_profile
        ):
            errors.append(f"{label}: device case {index} owner/backend/profile identity differs")
        if not isinstance(policy, Mapping) or any(
            policy.get(key) != value for key, value in expected_policy.items()
        ):
            errors.append(f"{label}: device case {index} compiled policy signature differs")
        process = case.get("process")
        orchestrator = case.get("orchestrator_process")
        if (
            not isinstance(process, Mapping)
            or process.get("return_code") != 0
            or not isinstance(orchestrator, Mapping)
            or orchestrator.get("return_code") != 0
            or not isinstance(summary, Mapping)
            or summary.get("status") != "completed"
        ):
            errors.append(f"{label}: device case {index} is not a completed passing process")

    receipt = _nested_mapping(artifact, "device", "profiler_summary", "backend_receipt")
    expected_receipt = {
        "backend_type": combination.backend,
        "execution_profile": combination.execution_profile,
        "task_plan_fingerprint": signature.task_plan_fingerprint,
        "policy_abi_fingerprint": signature.policy_abi_fingerprint,
        "backend_plan_fingerprint": signature.backend_plan_fingerprint,
    }
    if not isinstance(receipt, Mapping) or any(
        receipt.get(key) != value for key, value in expected_receipt.items()
    ):
        errors.append(f"{label}: compiled backend signature receipt differs")


def _validate_matrix(support: SupportEvidenceManifest, *, root: Path, errors: list[str]) -> None:
    from unilab.utils.support_matrix import EvidenceLevel, build_support_rows

    rows = build_support_rows(root, support_evidence=support)
    row_by_key = {(row.entrypoint_id, row.task_slug): row for row in rows}
    declared_high_grade: set[tuple[str, str, str]] = set()
    for combination in support.combinations:
        label = "/".join(combination.key)
        row = row_by_key.get((combination.entrypoint_id, combination.task_slug))
        if row is None or combination.backend not in row.cells:
            errors.append(f"{label}: generated support matrix cell is missing")
            continue
        expected_level = {
            DeclaredEvidenceLevel.TESTED: EvidenceLevel.TESTED,
            DeclaredEvidenceLevel.BENCHMARKED: EvidenceLevel.BENCHMARKED,
            DeclaredEvidenceLevel.RECOMMENDED: EvidenceLevel.RECOMMENDED,
        }[combination.evidence_level]
        cell = row.cells[combination.backend]
        if cell.env_name != combination.env_name or cell.level != expected_level:
            errors.append(f"{label}: generated support matrix does not match declaration")
        if cell.execution_profile != combination.execution_profile:
            errors.append(f"{label}: matrix execution profile does not match declaration")
        if cell.level >= EvidenceLevel.BENCHMARKED:
            declared_high_grade.add(combination.key)
    matrix_high_grade = {
        (row.entrypoint_id, row.task_slug, backend)
        for row in rows
        for backend, cell in row.cells.items()
        if cell.level >= EvidenceLevel.BENCHMARKED
    }
    if matrix_high_grade != declared_high_grade:
        errors.append("support matrix: high-grade cells are not bidirectionally declared")


def _recommended_rollout_is_verified(root: Path) -> bool:
    try:
        manifest = load_phase_acceptance(root / "tests/acceptance/issue_705/manifests/phase_7.yaml")
    except ManifestValidationError:
        return False
    rollout = next(
        (claim for claim in manifest.claims if claim.claim_id == "P7-TASK-ROLLOUT"), None
    )
    return bool(
        rollout is not None
        and rollout.status == ClaimStatus.VERIFIED
        and rollout.evidence.result == EvidenceResult.PASS
        and not rollout.evidence.skipped_test_ids
        and not rollout.evidence.xfailed_test_ids
    )


def audit_support_evidence(
    support: SupportEvidenceManifest,
    *,
    root: Path,
    registry_backends: Mapping[str, set[str]] | None = None,
    validate_phase_payloads: bool = True,
    validate_benchmark_payloads: bool = True,
) -> SupportAuditReport:
    """Audit support metadata from owner, registry, evidence, and matrix roots."""

    root = root.resolve()
    errors: list[str] = []
    if support.schema_version != SCHEMA_VERSION:
        errors.append(f"schema_version: expected {SCHEMA_VERSION}")
    if support.issue != ISSUE or support.claim_id != CLAIM_ID:
        errors.append(f"identity: expected issue {ISSUE} claim {CLAIM_ID}")
    if support.integration_branch != INTEGRATION_BRANCH:
        errors.append("integration_branch: does not match Issue #705 integration branch")
    if len(support.combinations) != 1:
        errors.append("Phase 7A must declare exactly one evidence-backed combination")

    phase_data = _validate_phase_gates(
        support,
        root=root,
        validate_payloads=validate_phase_payloads,
        errors=errors,
    )
    snapshot = registry_backends if registry_backends is not None else snapshot_registry_backends()
    _validate_owner_and_registry(support, root=root, registry_backends=snapshot, errors=errors)
    _validate_claim_inventory_and_tests(support, root=root, phase_data=phase_data, errors=errors)
    for combination in support.combinations:
        _validate_benchmark(
            combination,
            root=root,
            validate_payload=validate_benchmark_payloads,
            errors=errors,
        )
        if (
            combination.evidence_level == DeclaredEvidenceLevel.RECOMMENDED
            and not _recommended_rollout_is_verified(root)
        ):
            errors.append(
                f"{'/'.join(combination.key)}: recommended evidence requires verified Phase 7 rollout"
            )
    _validate_matrix(support, root=root, errors=errors)
    return SupportAuditReport(
        combinations=len(support.combinations),
        benchmarked=sum(
            item.evidence_level == DeclaredEvidenceLevel.BENCHMARKED
            for item in support.combinations
        ),
        recommended=sum(
            item.evidence_level == DeclaredEvidenceLevel.RECOMMENDED
            for item in support.combinations
        ),
        phase_gates=len(support.phase_gates),
        errors=tuple(errors),
    )


__all__ = [
    "BENCHMARK_ARTIFACT_PATH",
    "BENCHMARK_TEST_ID",
    "BENCHMARK_VALIDATOR",
    "CLAIM_ID",
    "CLAIM_INVENTORY_PATH",
    "CompiledSignature",
    "DeclaredEvidenceLevel",
    "INTEGRATION_BRANCH",
    "ISSUE",
    "PhaseGateRef",
    "REQUIRED_PHASES",
    "SCHEMA_VERSION",
    "SUPPORT_AUDIT_TEST_ID",
    "SUPPORT_EVIDENCE_PATH",
    "SupportAuditReport",
    "SupportCombination",
    "SupportEvidenceError",
    "SupportEvidenceManifest",
    "audit_support_evidence",
    "load_support_evidence",
    "parse_support_evidence",
    "snapshot_registry_backends",
]
