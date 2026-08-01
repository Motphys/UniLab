"""Final, fail-closed release evidence for Issue #705.

The final gate composes owner validators instead of reimplementing their
physics, training, or benchmark rules.  It adds the release-level guarantees
that individual phase gates cannot provide: a complete source closure, direct
input hashes, cross-phase validator coverage, and a mandatory A/B/C/D command
matrix with no skipped outcomes.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import stat
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any, cast

from omegaconf import OmegaConf

from unilab.tools.backend_isolation import audit_backend_isolation
from unilab.tools.claim_gap_audit import (
    InventoryTestState,
    audit_claim_gaps,
    load_claim_gap_inventory,
    load_phase_manifests,
)
from unilab.tools.g1_baseline_provenance import (
    load_g1_baseline_artifact,
    load_g1_baseline_plan,
    verify_g1_baseline_source,
)
from unilab.tools.issue705_legacy_retirement import (
    EVIDENCE_PATH as LEGACY_EVIDENCE_PATH,
)
from unilab.tools.issue705_legacy_retirement import (
    PLAN_PATH as LEGACY_PLAN_PATH,
)
from unilab.tools.issue705_legacy_retirement import (
    ROLLBACK_PATH as LEGACY_ROLLBACK_PATH,
)
from unilab.tools.issue705_legacy_retirement import (
    audit_legacy_retirement,
    load_legacy_retirement_evidence,
    load_legacy_retirement_plan,
    load_rollback_receipt,
)
from unilab.tools.issue705_phase1_evidence import (
    load_phase1_evidence,
    validate_phase1_evidence,
)
from unilab.tools.issue705_phase2_evidence import (
    load_phase2_evidence,
    validate_phase2_evidence,
)
from unilab.tools.issue705_phase3_evidence import (
    load_phase3_evidence,
    validate_phase3_evidence,
)
from unilab.tools.issue705_phase4_evidence import (
    load_phase4_evidence,
    validate_phase4_evidence,
)
from unilab.tools.issue705_phase5_evidence import (
    load_phase5_evidence,
    validate_phase5_evidence,
)
from unilab.tools.issue705_phase6_evidence import (
    load_phase6_evidence,
    validate_phase6_evidence,
)
from unilab.tools.issue705_support import (
    SUPPORT_EVIDENCE_PATH,
    audit_support_evidence,
    load_support_evidence,
)
from unilab.tools.issue705_task_rollout import (
    ROLLOUT_PLAN_PATH,
    audit_task_rollout_plan,
    load_task_rollout_plan,
)
from unilab.tools.issue705_thresholds import (
    AMENDMENT_FREEZE_RECEIPT_PATH,
    AMENDMENT_MANIFEST_PATH,
    BASELINE_ARTIFACT_PATH,
    BASELINE_PLAN_PATH,
    FREEZE_RECEIPT_PATH,
    THRESHOLD_MANIFEST_PATH,
    load_amendment_freeze_receipt,
    load_freeze_receipt,
    load_threshold_amendment,
    load_threshold_manifest,
)
from unilab.tools.issue705_training_behavior import (
    ARTIFACT_PATH as TRAINING_ARTIFACT_PATH,
)
from unilab.tools.issue705_training_behavior import (
    load_training_behavior_artifact,
)
from unilab.tools.mjwarp_dr_inventory import (
    inventory_claim_gap_errors,
    load_mjwarp_dr_inventory,
)
from unilab.tools.phase_acceptance import (
    ClaimStatus,
    EvidenceResult,
    ManifestValidationError,
    load_phase_acceptance,
    phase_gate_errors,
)

ISSUE = 705
PHASE = 7
CHILD_ISSUE = 843
SCHEMA_VERSION = 1
ARTIFACT_KIND = "issue705-final-gate-evidence-v1"
PLAN_FINGERPRINT = "issue705-final-gate-v1"
INTEGRATION_BRANCH = "feat/issue-705-manager-mjwarp"

ACCEPTANCE_ROOT = Path("tests/acceptance/issue_705")
MANIFEST_DIR = ACCEPTANCE_ROOT / "manifests"
PLAN_PATH = ACCEPTANCE_ROOT / "final_gate_plan.yaml"
ARTIFACT_PATH = ACCEPTANCE_ROOT / "artifacts/phase_7_gate.json"
PHASE7_MANIFEST_PATH = MANIFEST_DIR / "phase_7.yaml"
CLAIM_INVENTORY_PATH = ACCEPTANCE_ROOT / "claim_test_inventory.yaml"
DR_INVENTORY_PATH = ACCEPTANCE_ROOT / "mjwarp_dr_inventory.yaml"

_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_PYTEST_COUNT_RE = re.compile(r"(?<![A-Za-z0-9_])(\d+) (passed|skipped|xfailed|xpassed|deselected)")
_MANDATORY_ZERO_COUNTS = ("skipped", "xfailed", "xpassed", "deselected")
_PHASE7_MUTABLE_CLAIM_FIELDS = ("evidence", "status")
_EXPECTED_PHASE_VALIDATORS = {
    1: "issue705_phase1_evidence",
    2: "issue705_phase2_evidence",
    3: "issue705_phase3_evidence",
    4: "issue705_phase4_evidence",
    5: "issue705_phase5_evidence",
    6: "issue705_phase6_evidence",
}
_EXPECTED_COMPONENTS = (
    "phase0_manifest",
    "phase0_baseline",
    "phase0_thresholds",
    "phase0_dr_inventory",
    "claim_inventory",
    "backend_isolation",
    "phase1_evidence",
    "phase2_evidence",
    "phase3_evidence",
    "phase4_evidence",
    "phase5_evidence",
    "phase6_evidence",
    "support_matrix",
    "task_rollout",
    "training_behavior",
    "entrypoint_legacy_retirement",
    "phase7_contract",
)


class FinalGateError(RuntimeError):
    """Raised when final evidence cannot be captured or loaded safely."""


class FinalGatePlanError(ValueError):
    """Raised when the frozen final-gate plan is malformed."""

    def __init__(self, source: Path, errors: Sequence[str]) -> None:
        self.source = source
        self.errors = tuple(errors)
        detail = "\n".join(f"- {error}" for error in self.errors)
        super().__init__(f"invalid Issue #705 final gate plan {source}:\n{detail}")


@dataclass(frozen=True)
class FinalPhaseEvidence:
    """One prior phase artifact and the owner validator that must accept it."""

    phase: int
    artifact: Path
    validator: str


@dataclass(frozen=True)
class FinalGateCommand:
    """One Phase 7 claim command and its mandatory clean repetitions."""

    claim_id: str
    name: str
    lane: str
    argv: tuple[str, ...]
    required_test_id: str
    repetitions: int
    manifest_command: str
    config_input: Path
    artifact_refs: tuple[str, ...]


@dataclass(frozen=True)
class FinalGatePlan:
    """Strict, immutable release-validation plan."""

    schema_version: int
    issue: int
    phase: int
    child_issue: int
    plan_fingerprint: str
    integration_branch: str
    source_roots: tuple[Path, ...]
    source_exclusions: tuple[Path, ...]
    phase_evidence: tuple[FinalPhaseEvidence, ...]
    direct_inputs: tuple[Path, ...]
    commands: tuple[FinalGateCommand, ...]
    rss_mode: str
    rss_reason: str
    source_path: Path


@dataclass(frozen=True)
class FinalGateComponent:
    """Deterministic result from one owner-level validation component."""

    name: str
    details: Mapping[str, Any]
    errors: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.ok,
            "details": dict(self.details),
            "errors": list(self.errors),
        }


@dataclass(frozen=True)
class FinalGateReport:
    """Complete owner-validation report for the current repository head."""

    components: tuple[FinalGateComponent, ...]
    errors: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.ok,
            "components": [component.to_dict() for component in self.components],
            "errors": list(self.errors),
        }


class _Parser:
    def __init__(self, source: Path) -> None:
        self.source = source
        self.errors: list[str] = []

    def mapping(self, value: object, path: str, keys: Sequence[str]) -> dict[str, Any]:
        if not isinstance(value, dict):
            self.errors.append(f"{path}: expected mapping")
            return {}
        expected = set(keys)
        actual = set(value)
        for key in sorted(expected - actual):
            self.errors.append(f"{path}: missing key {key!r}")
        for key in sorted(actual - expected, key=str):
            self.errors.append(f"{path}: unknown key {key!r}")
        return cast(dict[str, Any], value)

    def string(self, value: object, path: str) -> str:
        if not isinstance(value, str) or not value:
            self.errors.append(f"{path}: expected non-empty string")
            return ""
        if "${" in value:
            self.errors.append(f"{path}: interpolation is forbidden")
        return value

    def integer(self, value: object, path: str, *, minimum: int = 0) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            self.errors.append(f"{path}: expected integer")
            return -1
        if value < minimum:
            self.errors.append(f"{path}: expected >= {minimum}")
        return value

    def strings(self, value: object, path: str, *, allow_empty: bool = False) -> tuple[str, ...]:
        if not isinstance(value, list) or (not value and not allow_empty):
            self.errors.append(f"{path}: expected {'a' if allow_empty else 'a non-empty'} list")
            return ()
        result = tuple(self.string(item, f"{path}[{index}]") for index, item in enumerate(value))
        if len(set(result)) != len(result):
            self.errors.append(f"{path}: duplicate entries are forbidden")
        return result

    def relative_path(self, value: object, path: str) -> Path:
        raw = self.string(value, path)
        result = Path(raw)
        if result.is_absolute() or ".." in result.parts:
            self.errors.append(f"{path}: expected repository-relative path")
        return result

    def finish(self) -> None:
        if self.errors:
            raise FinalGatePlanError(self.source, self.errors)


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        raw = OmegaConf.to_container(OmegaConf.load(path), resolve=False)
    except Exception as exc:  # noqa: BLE001 - normalize configuration failures
        raise FinalGatePlanError(path, [f"cannot load YAML: {type(exc).__name__}: {exc}"]) from exc
    if not isinstance(raw, dict):
        raise FinalGatePlanError(path, ["root must be a mapping"])
    return cast(dict[str, Any], raw)


def load_final_gate_plan(path: Path) -> FinalGatePlan:
    """Load and semantically validate the frozen Phase 7F plan."""

    raw = _load_yaml(path)
    parser = _Parser(path)
    root = parser.mapping(
        raw,
        "root",
        (
            "schema_version",
            "issue",
            "phase",
            "child_issue",
            "plan_fingerprint",
            "integration_branch",
            "source_scope",
            "phase_evidence",
            "direct_inputs",
            "commands",
            "rss_policy",
        ),
    )
    source_scope = parser.mapping(
        root.get("source_scope"), "source_scope", ("roots", "excluded_paths")
    )
    source_roots = tuple(
        parser.relative_path(item, f"source_scope.roots[{index}]")
        for index, item in enumerate(
            parser.strings(source_scope.get("roots"), "source_scope.roots")
        )
    )
    source_exclusions = tuple(
        parser.relative_path(item, f"source_scope.excluded_paths[{index}]")
        for index, item in enumerate(
            parser.strings(source_scope.get("excluded_paths"), "source_scope.excluded_paths")
        )
    )

    raw_phases = root.get("phase_evidence")
    if not isinstance(raw_phases, list) or not raw_phases:
        parser.errors.append("phase_evidence: expected non-empty list")
        raw_phases = []
    phase_evidence: list[FinalPhaseEvidence] = []
    for index, value in enumerate(raw_phases):
        item = parser.mapping(value, f"phase_evidence[{index}]", ("phase", "artifact", "validator"))
        phase_evidence.append(
            FinalPhaseEvidence(
                phase=parser.integer(
                    item.get("phase"), f"phase_evidence[{index}].phase", minimum=1
                ),
                artifact=parser.relative_path(
                    item.get("artifact"), f"phase_evidence[{index}].artifact"
                ),
                validator=parser.string(
                    item.get("validator"), f"phase_evidence[{index}].validator"
                ),
            )
        )

    direct_inputs = tuple(
        parser.relative_path(item, f"direct_inputs[{index}]")
        for index, item in enumerate(parser.strings(root.get("direct_inputs"), "direct_inputs"))
    )
    raw_commands = root.get("commands")
    if not isinstance(raw_commands, list) or not raw_commands:
        parser.errors.append("commands: expected non-empty list")
        raw_commands = []
    commands: list[FinalGateCommand] = []
    for index, value in enumerate(raw_commands):
        prefix = f"commands[{index}]"
        item = parser.mapping(
            value,
            prefix,
            (
                "claim_id",
                "name",
                "lane",
                "argv",
                "required_test_id",
                "repetitions",
                "manifest_command",
                "config_input",
                "artifact_refs",
            ),
        )
        commands.append(
            FinalGateCommand(
                claim_id=parser.string(item.get("claim_id"), f"{prefix}.claim_id"),
                name=parser.string(item.get("name"), f"{prefix}.name"),
                lane=parser.string(item.get("lane"), f"{prefix}.lane"),
                argv=parser.strings(item.get("argv"), f"{prefix}.argv"),
                required_test_id=parser.string(
                    item.get("required_test_id"), f"{prefix}.required_test_id"
                ),
                repetitions=parser.integer(
                    item.get("repetitions"), f"{prefix}.repetitions", minimum=1
                ),
                manifest_command=parser.string(
                    item.get("manifest_command"), f"{prefix}.manifest_command"
                ),
                config_input=parser.relative_path(
                    item.get("config_input"), f"{prefix}.config_input"
                ),
                artifact_refs=parser.strings(item.get("artifact_refs"), f"{prefix}.artifact_refs"),
            )
        )
    rss = parser.mapping(root.get("rss_policy"), "rss_policy", ("mode", "reason"))
    plan = FinalGatePlan(
        schema_version=parser.integer(root.get("schema_version"), "schema_version", minimum=1),
        issue=parser.integer(root.get("issue"), "issue", minimum=1),
        phase=parser.integer(root.get("phase"), "phase", minimum=0),
        child_issue=parser.integer(root.get("child_issue"), "child_issue", minimum=1),
        plan_fingerprint=parser.string(root.get("plan_fingerprint"), "plan_fingerprint"),
        integration_branch=parser.string(root.get("integration_branch"), "integration_branch"),
        source_roots=source_roots,
        source_exclusions=source_exclusions,
        phase_evidence=tuple(phase_evidence),
        direct_inputs=direct_inputs,
        commands=tuple(commands),
        rss_mode=parser.string(rss.get("mode"), "rss_policy.mode"),
        rss_reason=parser.string(rss.get("reason"), "rss_policy.reason"),
        source_path=path,
    )
    parser.finish()
    semantic_errors = _plan_semantic_errors(plan)
    if semantic_errors:
        raise FinalGatePlanError(path, semantic_errors)
    return plan


def _plan_semantic_errors(plan: FinalGatePlan) -> list[str]:
    errors: list[str] = []
    expected_identity = {
        "schema_version": SCHEMA_VERSION,
        "issue": ISSUE,
        "phase": PHASE,
        "child_issue": CHILD_ISSUE,
        "plan_fingerprint": PLAN_FINGERPRINT,
        "integration_branch": INTEGRATION_BRANCH,
    }
    for key, expected in expected_identity.items():
        if getattr(plan, key) != expected:
            errors.append(f"{key}: expected {expected!r}")
    if plan.source_roots != (Path("."),):
        errors.append("source_scope.roots: v1 must close over the complete tracked repository")
    expected_exclusions = {ARTIFACT_PATH, PHASE7_MANIFEST_PATH}
    if set(plan.source_exclusions) != expected_exclusions:
        errors.append(
            "source_scope.excluded_paths: only the final artifact and Phase 7 promotion manifest "
            "may be excluded"
        )
    phase_map = {entry.phase: entry for entry in plan.phase_evidence}
    if len(phase_map) != len(plan.phase_evidence) or set(phase_map) != set(range(1, 7)):
        errors.append("phase_evidence: expected each Phase 1..6 exactly once")
    for phase, validator in _EXPECTED_PHASE_VALIDATORS.items():
        entry = phase_map.get(phase)
        expected_artifact = ACCEPTANCE_ROOT / f"artifacts/phase_{phase}_gate.json"
        if entry is not None and (
            entry.validator != validator or entry.artifact != expected_artifact
        ):
            errors.append(f"phase_evidence[{phase}]: artifact/validator binding differs from v1")
    if len(set(plan.direct_inputs)) != len(plan.direct_inputs):
        errors.append("direct_inputs: duplicate paths are forbidden")
    if PLAN_PATH not in set(plan.direct_inputs):
        errors.append("direct_inputs: final gate plan must hash itself")
    forbidden_inputs = {ARTIFACT_PATH, PHASE7_MANIFEST_PATH} & set(plan.direct_inputs)
    if forbidden_inputs:
        errors.append(
            f"direct_inputs: self/promotion inputs are forbidden: {sorted(forbidden_inputs)!r}"
        )
    command_names = [command.name for command in plan.commands]
    claim_ids = [command.claim_id for command in plan.commands]
    if len(set(command_names)) != len(command_names):
        errors.append("commands: names must be unique")
    if len(set(claim_ids)) != len(claim_ids):
        errors.append("commands: claim IDs must be unique")
    expected_claims = {
        "P7-SUPPORT-MATRIX",
        "P7-TASK-ROLLOUT",
        "P7-TRAINING-BEHAVIOR",
        "P7-ENTRYPOINT-MATRIX",
        "P7-FINAL-REGRESSION",
        "P7-LEGACY-RETIREMENT",
    }
    if set(claim_ids) != expected_claims:
        errors.append("commands: must bind every Phase 7 claim exactly once")
    if {command.lane for command in plan.commands} != {"A", "B", "C", "D"}:
        errors.append("commands: must cover exactly lanes A/B/C/D")
    for command in plan.commands:
        if command.argv[:2] != ("uv", "run") or "pytest" not in command.argv:
            errors.append(
                f"commands[{command.name}].argv: mandatory evidence must use uv run pytest"
            )
        if command.required_test_id not in command.argv:
            errors.append(f"commands[{command.name}].argv: exact required test ID is absent")
        if command.config_input not in set(plan.direct_inputs):
            errors.append(f"commands[{command.name}].config_input: must be a direct input")
        if ARTIFACT_PATH.as_posix() not in command.artifact_refs:
            errors.append(f"commands[{command.name}].artifact_refs: final artifact is required")
    if plan.rss_mode != "diagnostic_only":
        errors.append("rss_policy.mode: RSS is diagnostic-only and cannot block Phase 7")
    return errors


def sha256_file(path: Path) -> str:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def _sha256_bytes(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return _sha256_bytes(payload.encode("utf-8"))


def _run_checked(root: Path, argv: Sequence[str], *, context: str) -> str:
    result = subprocess.run(argv, cwd=root, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise FinalGateError(
            f"{context} failed with exit {result.returncode}:\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result.stdout.strip()


def _git(root: Path, *args: str) -> str:
    return _run_checked(root, ("git", *args), context=f"git {' '.join(args)}")


def _git_is_ancestor(root: Path, commit: str) -> bool:
    return (
        subprocess.run(
            ("git", "merge-base", "--is-ancestor", commit, "HEAD"),
            cwd=root,
            capture_output=True,
            check=False,
        ).returncode
        == 0
    )


def _assert_clean(root: Path) -> None:
    status = _git(root, "status", "--porcelain")
    if status:
        raise FinalGateError("final evidence requires a clean source tree:\n" + status)


def _path_is_excluded(path: Path, exclusions: Sequence[Path]) -> bool:
    return path in exclusions


def _tracked_paths(
    root: Path, plan: FinalGatePlan, *, commit: str | None = None
) -> tuple[Path, ...]:
    if commit is None:
        raw = subprocess.run(
            ("git", "ls-files", "-z", "--", *(path.as_posix() for path in plan.source_roots)),
            cwd=root,
            capture_output=True,
            check=True,
        ).stdout
    else:
        raw = subprocess.run(
            (
                "git",
                "ls-tree",
                "-r",
                "--name-only",
                "-z",
                commit,
                "--",
                *(path.as_posix() for path in plan.source_roots),
            ),
            cwd=root,
            capture_output=True,
            check=True,
        ).stdout
    paths = {
        Path(item.decode("utf-8"))
        for item in raw.split(b"\0")
        if item and not _path_is_excluded(Path(item.decode("utf-8")), plan.source_exclusions)
    }
    return tuple(sorted(paths, key=Path.as_posix))


def _working_tree_blob(root: Path, path: Path) -> bytes:
    source = root / path
    if source.is_symlink():
        return os.fsencode(os.readlink(source))
    return source.read_bytes()


def _working_tree_git_mode(root: Path, path: Path) -> str:
    source = root / path
    try:
        mode = source.lstat().st_mode
    except OSError as exc:
        raise FinalGateError(f"cannot stat tracked source path {path.as_posix()}: {exc}") from exc
    if stat.S_ISLNK(mode):
        return "120000"
    if stat.S_ISREG(mode):
        return "100755" if mode & 0o111 else "100644"
    raise FinalGateError(
        f"tracked source path {path.as_posix()} has unsupported file type {stat.S_IFMT(mode):o}"
    )


def _commit_tree_modes(root: Path, plan: FinalGatePlan, commit: str) -> dict[Path, str]:
    raw = subprocess.run(
        (
            "git",
            "ls-tree",
            "-r",
            "-z",
            commit,
            "--",
            *(path.as_posix() for path in plan.source_roots),
        ),
        cwd=root,
        capture_output=True,
        check=True,
    ).stdout
    modes: dict[Path, str] = {}
    for record in raw.split(b"\0"):
        if not record:
            continue
        metadata, separator, raw_path = record.partition(b"\t")
        fields = metadata.split(b" ", 2)
        if not separator or len(fields) != 3:
            raise FinalGateError("git ls-tree returned a malformed source entry")
        path = Path(raw_path.decode("utf-8"))
        if _path_is_excluded(path, plan.source_exclusions):
            continue
        modes[path] = fields[0].decode("ascii")
    return modes


def _source_snapshot(
    root: Path, plan: FinalGatePlan, *, commit: str | None = None
) -> dict[str, object]:
    paths = _tracked_paths(root, plan, commit=commit)
    modes = (
        {path: _working_tree_git_mode(root, path) for path in paths}
        if commit is None
        else _commit_tree_modes(root, plan, commit)
    )
    if set(modes) != set(paths):
        raise FinalGateError("tracked source paths and Git mode entries differ")
    digest = hashlib.sha256()
    for path in paths:
        relative = path.as_posix()
        if commit is None:
            value = _working_tree_blob(root, path)
        else:
            value = subprocess.run(
                ("git", "show", f"{commit}:{relative}"),
                cwd=root,
                capture_output=True,
                check=True,
            ).stdout
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(modes[path].encode("ascii"))
        digest.update(b"\0")
        digest.update(value)
        digest.update(b"\0")
    mode_rows = [{"path": path.as_posix(), "mode": modes[path]} for path in paths]
    return {
        "tracked_file_count": len(paths),
        "tracked_paths_sha256": _canonical_sha256([path.as_posix() for path in paths]),
        "tracked_modes_sha256": _canonical_sha256(mode_rows),
        "tree_sha256": f"sha256:{digest.hexdigest()}",
    }


def _load_phase7_manifest_mapping(root: Path, *, commit: str | None = None) -> dict[str, Any]:
    if commit is None:
        try:
            text = (root / PHASE7_MANIFEST_PATH).read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise FinalGateError(f"cannot read Phase 7 manifest: {exc}") from exc
    else:
        result = subprocess.run(
            ("git", "show", f"{commit}:{PHASE7_MANIFEST_PATH.as_posix()}"),
            cwd=root,
            capture_output=True,
            check=True,
        )
        try:
            text = result.stdout.decode("utf-8")
        except UnicodeError as exc:
            raise FinalGateError("Phase 7 manifest at source commit is not UTF-8") from exc
    try:
        raw = OmegaConf.to_container(OmegaConf.create(text), resolve=False)
    except Exception as exc:  # noqa: BLE001 - normalize semantic snapshot failures
        raise FinalGateError(
            f"cannot parse Phase 7 manifest semantics: {type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(raw, dict):
        raise FinalGateError("Phase 7 manifest semantic snapshot must be a mapping")
    return cast(dict[str, Any], raw)


def _phase7_immutable_semantics(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return every Phase 7 manifest field except promotion-owned claim state."""

    claims = value.get("claims")
    if not isinstance(claims, list):
        raise FinalGateError("Phase 7 manifest claims must be a list")
    immutable_claims: list[dict[str, Any]] = []
    for index, claim in enumerate(claims):
        if not isinstance(claim, Mapping):
            raise FinalGateError(f"Phase 7 manifest claim {index} must be a mapping")
        immutable_claims.append(
            {
                str(key): deepcopy(item)
                for key, item in claim.items()
                if key not in _PHASE7_MUTABLE_CLAIM_FIELDS
            }
        )
    snapshot = {str(key): deepcopy(item) for key, item in value.items() if key != "claims"}
    snapshot["claims"] = immutable_claims
    return snapshot


def _phase7_manifest_snapshot(root: Path, *, commit: str | None = None) -> dict[str, Any]:
    semantics = _phase7_immutable_semantics(_load_phase7_manifest_mapping(root, commit=commit))
    return {
        "path": PHASE7_MANIFEST_PATH.as_posix(),
        "mutable_claim_fields": list(_PHASE7_MUTABLE_CLAIM_FIELDS),
        "sha256": _canonical_sha256(semantics),
        "snapshot": semantics,
    }


def _direct_input_hashes(root: Path, plan: FinalGatePlan) -> dict[str, str]:
    missing = [path.as_posix() for path in plan.direct_inputs if not (root / path).is_file()]
    if missing:
        raise FinalGateError(f"final gate direct inputs are missing: {missing!r}")
    return {path.as_posix(): sha256_file(root / path) for path in plan.direct_inputs}


def _git_blob_hash(root: Path, commit: str, path: Path) -> str:
    result = subprocess.run(
        ("git", "show", f"{commit}:{path.as_posix()}"),
        cwd=root,
        capture_output=True,
        check=True,
    )
    return _sha256_bytes(result.stdout)


def _component(
    name: str, validator: Callable[[], tuple[Mapping[str, Any], Sequence[str]]]
) -> FinalGateComponent:
    try:
        details, errors = validator()
        return FinalGateComponent(name=name, details=details, errors=tuple(errors))
    except Exception as exc:  # noqa: BLE001 - aggregate owner failures into one report
        return FinalGateComponent(
            name=name,
            details={},
            errors=(f"{type(exc).__name__}: {exc}",),
        )


def _phase0_manifest(root: Path) -> tuple[Mapping[str, Any], Sequence[str]]:
    manifest = load_phase_acceptance(root / MANIFEST_DIR / "phase_0.yaml")
    errors = phase_gate_errors(manifest)
    return {
        "claims": len(manifest.claims),
        "required_lanes": [lane.value for lane in manifest.required_lanes],
    }, errors


def _phase0_baseline(root: Path) -> tuple[Mapping[str, Any], Sequence[str]]:
    plan = replace(load_g1_baseline_plan(root / BASELINE_PLAN_PATH), source_path=BASELINE_PLAN_PATH)
    artifact = load_g1_baseline_artifact(
        root / BASELINE_ARTIFACT_PATH,
        plan,
        repo_root=root,
    )
    verification = verify_g1_baseline_source(artifact, plan, root)
    return {
        "source_commit": artifact["source"]["commit"],
        "cases": len(artifact["cases"]),
        "git_history_verified": verification.git_history_verified,
    }, verification.errors


def _phase0_thresholds(root: Path) -> tuple[Mapping[str, Any], Sequence[str]]:
    manifest = load_threshold_manifest(root / THRESHOLD_MANIFEST_PATH, repo_root=root)
    receipt = load_freeze_receipt(
        root / FREEZE_RECEIPT_PATH,
        manifest=manifest,
        repo_root=root,
    )
    amendment = load_threshold_amendment(
        root / AMENDMENT_MANIFEST_PATH,
        base_manifest=manifest,
        base_receipt=receipt,
        repo_root=root,
    )
    amendment_receipt = load_amendment_freeze_receipt(
        root / AMENDMENT_FREEZE_RECEIPT_PATH,
        amendment=amendment,
        base_receipt=receipt,
        repo_root=root,
    )
    return {
        "threshold_set_id": manifest.data["threshold_set_id"],
        "freeze_commit": receipt.freeze_commit,
        "amendment_id": amendment.amendment_id,
        "amendment_freeze_commit": amendment_receipt.freeze_commit,
        "rss_policy": "diagnostic_only_at_final_gate",
    }, ()


def _phase0_dr_inventory(root: Path) -> tuple[Mapping[str, Any], Sequence[str]]:
    inventory = load_mjwarp_dr_inventory(root / DR_INVENTORY_PATH)
    claims = load_claim_gap_inventory(root / CLAIM_INVENTORY_PATH)
    errors = inventory_claim_gap_errors(inventory, claims)
    return {
        "capabilities": len(inventory.capabilities),
        "exclusions": len(inventory.exclusions),
    }, errors


def _claim_inventory(root: Path) -> tuple[Mapping[str, Any], Sequence[str]]:
    manifests, manifest_errors = load_phase_manifests(root / MANIFEST_DIR, tuple(range(8)))
    inventory = load_claim_gap_inventory(root / CLAIM_INVENTORY_PATH)
    report = audit_claim_gaps(inventory, manifests, repo_root=root, phases=tuple(range(8)))
    errors = [*manifest_errors, *report.errors]
    if report.targets != 0:
        errors.append(f"claim inventory: expected zero targets at final head, got {report.targets}")
    target_entries = [
        entry.test_id for entry in inventory.entries if entry.state is InventoryTestState.TARGET
    ]
    return {
        "claims": report.claims,
        "entries": report.entries,
        "existing": report.existing,
        "targets": report.targets,
        "target_test_ids": target_entries,
    }, errors


def _backend_isolation(root: Path) -> tuple[Mapping[str, Any], Sequence[str]]:
    report = audit_backend_isolation(root)
    return {
        "backend_packages": len(report.backend_packages),
        "runtime_modules": len(report.runtime_modules),
        "contract_files": len(report.contract_files),
    }, tuple(violation.format() for violation in report.violations)


_PHASE_LOADERS: dict[int, Callable[[Path], dict[str, Any]]] = {
    1: load_phase1_evidence,
    2: load_phase2_evidence,
    3: load_phase3_evidence,
    4: load_phase4_evidence,
    5: load_phase5_evidence,
    6: load_phase6_evidence,
}


def _validate_phase(
    root: Path, entry: FinalPhaseEvidence
) -> tuple[Mapping[str, Any], Sequence[str]]:
    report = _PHASE_LOADERS[entry.phase](root / entry.artifact)
    validators: dict[int, Callable[..., tuple[str, ...]]] = {
        1: validate_phase1_evidence,
        2: validate_phase2_evidence,
        3: validate_phase3_evidence,
        4: validate_phase4_evidence,
        5: validate_phase5_evidence,
        6: validate_phase6_evidence,
    }
    errors = list(validators[entry.phase](report, root=root))
    manifest = load_phase_acceptance(root / MANIFEST_DIR / f"phase_{entry.phase}.yaml")
    errors.extend(f"manifest: {error}" for error in phase_gate_errors(manifest))
    return {
        "phase": entry.phase,
        "validator": entry.validator,
        "source_commit": report.get("source", {}).get("commit_sha"),
        "claims": len(report.get("claims", [])),
        "commands": len(report.get("commands", [])),
    }, errors


def _support_matrix(root: Path) -> tuple[Mapping[str, Any], Sequence[str]]:
    support = load_support_evidence(root / SUPPORT_EVIDENCE_PATH)
    report = audit_support_evidence(support, root=root)
    return {
        "combinations": report.combinations,
        "benchmarked": report.benchmarked,
        "recommended": report.recommended,
        "phase_gates": report.phase_gates,
    }, report.errors


def _task_rollout(root: Path) -> tuple[Mapping[str, Any], Sequence[str]]:
    plan = load_task_rollout_plan(root / ROLLOUT_PLAN_PATH)
    report = audit_task_rollout_plan(plan, root=root)
    return {"entries": report.entries, "prerequisites": report.prerequisites}, report.errors


def _training_behavior(root: Path) -> tuple[Mapping[str, Any], Sequence[str]]:
    artifact, report = load_training_behavior_artifact(
        root / TRAINING_ARTIFACT_PATH,
        repo_root=root,
    )
    return {
        "source_commit": artifact.get("source", {}).get("commit"),
        "paired_seeds": len(artifact.get("pairs", [])),
        "rss_policy": "diagnostic_only",
    }, report.errors


def _legacy_retirement(root: Path) -> tuple[Mapping[str, Any], Sequence[str]]:
    plan = load_legacy_retirement_plan(root / LEGACY_PLAN_PATH)
    rollback = load_rollback_receipt(root / LEGACY_ROLLBACK_PATH)
    evidence = load_legacy_retirement_evidence(root / LEGACY_EVIDENCE_PATH)
    report = audit_legacy_retirement(plan, rollback, evidence, root=root)
    return {
        "source_commit": evidence.get("source", {}).get("commit_sha"),
        "changed_paths": report.changed_paths,
        "entrypoint_repetitions": report.entrypoint_repetitions,
        "retained_routes": report.retained_routes,
    }, report.errors


def _phase7_contract(
    root: Path,
    plan: FinalGatePlan,
    *,
    require_promotion: bool,
    artifact: Mapping[str, Any] | None,
) -> tuple[Mapping[str, Any], Sequence[str]]:
    try:
        manifest = load_phase_acceptance(root / PHASE7_MANIFEST_PATH)
    except ManifestValidationError as exc:
        return {}, tuple(exc.errors)
    errors: list[str] = []
    if manifest.issue != ISSUE or manifest.phase != PHASE:
        errors.append("Phase 7 manifest identity differs from Issue #705 final gate")
    if manifest.integration_branch != INTEGRATION_BRANCH:
        errors.append("Phase 7 integration branch differs from the final plan")
    if {lane.value for lane in manifest.required_lanes} != {"A", "B", "C", "D"}:
        errors.append("Phase 7 manifest must require lanes A/B/C/D")
    claims = {claim.claim_id: claim for claim in manifest.claims}
    commands = {command.claim_id: command for command in plan.commands}
    if set(claims) != set(commands):
        errors.append("Phase 7 manifest and final plan claim IDs differ")
    source_commit = None
    input_hashes: Mapping[str, Any] = {}
    if artifact is not None:
        source_commit = artifact.get("source", {}).get("commit_sha")
        raw_inputs = artifact.get("inputs", {})
        if isinstance(raw_inputs, Mapping) and isinstance(raw_inputs.get("files"), Mapping):
            input_hashes = cast(Mapping[str, Any], raw_inputs["files"])
    for claim_id, command in commands.items():
        claim = claims.get(claim_id)
        if claim is None:
            continue
        if claim.commands != (command.manifest_command,):
            errors.append(f"{claim_id}: manifest command differs from final plan")
        if claim.required_test_ids != (command.required_test_id,):
            errors.append(f"{claim_id}: required test differs from final plan")
        if claim.lane.value != command.lane:
            errors.append(f"{claim_id}: lane differs from final plan")
        if claim.acceptance.repetitions != command.repetitions:
            errors.append(f"{claim_id}: repetitions differ from final plan")
        if not require_promotion:
            continue
        if claim.status is not ClaimStatus.VERIFIED:
            errors.append(f"{claim_id}: final promotion status must be verified")
        if claim.evidence.result is not EvidenceResult.PASS:
            errors.append(f"{claim_id}: final promotion evidence must be PASS")
        if claim.evidence.executed_test_ids != (command.required_test_id,):
            errors.append(f"{claim_id}: executed test differs from final plan")
        if claim.evidence.skipped_test_ids or claim.evidence.xfailed_test_ids:
            errors.append(f"{claim_id}: skipped or xfailed evidence is forbidden")
        if tuple(claim.evidence.artifact_refs) != command.artifact_refs:
            errors.append(f"{claim_id}: artifact refs differ from final plan")
        if source_commit is None or claim.evidence.commit_sha != source_commit:
            errors.append(f"{claim_id}: evidence commit must equal final artifact source")
        expected_hash = input_hashes.get(command.config_input.as_posix())
        if expected_hash is None or claim.evidence.config_hash != expected_hash:
            errors.append(f"{claim_id}: config hash must equal final artifact input hash")
    if require_promotion:
        errors.extend(f"phase gate: {error}" for error in phase_gate_errors(manifest))
    return {
        "claims": len(manifest.claims),
        "status_counts": {
            status.value: sum(claim.status is status for claim in manifest.claims)
            for status in ClaimStatus
        },
        "promotion_required": require_promotion,
    }, errors


def validate_final_head(
    root: Path,
    plan: FinalGatePlan,
    *,
    require_promotion: bool = False,
    artifact: Mapping[str, Any] | None = None,
) -> FinalGateReport:
    """Recompute every owner gate required at the final integration head."""

    root = root.resolve()
    phase_entries = {entry.phase: entry for entry in plan.phase_evidence}
    components = [
        _component("phase0_manifest", lambda: _phase0_manifest(root)),
        _component("phase0_baseline", lambda: _phase0_baseline(root)),
        _component("phase0_thresholds", lambda: _phase0_thresholds(root)),
        _component("phase0_dr_inventory", lambda: _phase0_dr_inventory(root)),
        _component("claim_inventory", lambda: _claim_inventory(root)),
        _component("backend_isolation", lambda: _backend_isolation(root)),
    ]
    components.extend(
        _component(
            f"phase{phase}_evidence",
            partial(_validate_phase, root, phase_entries[phase]),
        )
        for phase in range(1, 7)
    )
    components.extend(
        (
            _component("support_matrix", lambda: _support_matrix(root)),
            _component("task_rollout", lambda: _task_rollout(root)),
            _component("training_behavior", lambda: _training_behavior(root)),
            _component("entrypoint_legacy_retirement", lambda: _legacy_retirement(root)),
            _component(
                "phase7_contract",
                lambda: _phase7_contract(
                    root,
                    plan,
                    require_promotion=require_promotion,
                    artifact=artifact,
                ),
            ),
        )
    )
    component_names = tuple(component.name for component in components)
    errors: list[str] = []
    if component_names != _EXPECTED_COMPONENTS:
        errors.append("final gate component set differs from the v1 release contract")
    for component in components:
        errors.extend(f"{component.name}: {error}" for error in component.errors)
    return FinalGateReport(components=tuple(components), errors=tuple(errors))


def _pytest_counts(output: str) -> dict[str, int]:
    counts = {
        "passed": 0,
        "skipped": 0,
        "xfailed": 0,
        "xpassed": 0,
        "deselected": 0,
    }
    for raw_count, category in _PYTEST_COUNT_RE.findall(output):
        counts[category] += int(raw_count)
    return counts


def _run_command(
    command: FinalGateCommand,
    *,
    root: Path,
    repetition: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    result = subprocess.run(
        command.argv,
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
        timeout=1800,
    )
    duration = time.perf_counter() - started
    combined = f"{result.stdout}\n{result.stderr}"
    counts = _pytest_counts(combined)
    failures: list[str] = []
    if result.returncode != 0:
        failures.append(f"exit_code={result.returncode}")
    if counts["passed"] <= 0:
        failures.append("pytest reported no passed tests")
    for category in _MANDATORY_ZERO_COUNTS:
        if counts[category] != 0:
            failures.append(f"pytest reported {counts[category]} {category}")
    if command.required_test_id not in combined:
        failures.append("required test ID is absent from pytest output")
    if failures:
        raise FinalGateError(
            f"{command.name} repetition {repetition} failed final evidence requirements "
            f"({'; '.join(failures)}):\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return {
        "name": f"{command.name}#{repetition}",
        "series": command.name,
        "claim_id": command.claim_id,
        "lane": command.lane,
        "repetition": repetition,
        "argv": list(command.argv),
        "required_test_id": command.required_test_id,
        "exit_code": result.returncode,
        "duration_sec": duration,
        "pytest": counts,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def capture_final_gate_evidence(root: Path, plan: FinalGatePlan) -> dict[str, Any]:
    """Run the final A/B/C/D matrix from one clean implementation commit."""

    root = root.resolve()
    _assert_clean(root)
    commit = _git(root, "rev-parse", "HEAD")
    if not _COMMIT_RE.fullmatch(commit):
        raise FinalGateError("git did not return a full source commit SHA")
    input_hashes = _direct_input_hashes(root, plan)
    source_scope = _source_snapshot(root, plan)
    phase7_manifest = _phase7_manifest_snapshot(root)
    head_report = validate_final_head(root, plan, require_promotion=False)
    if not head_report.ok:
        raise FinalGateError(
            "final owner validation failed before command capture:\n"
            + "\n".join(f"- {error}" for error in head_report.errors)
        )
    commands = [
        _run_command(command, root=root, repetition=repetition)
        for command in plan.commands
        for repetition in range(1, command.repetitions + 1)
    ]
    _assert_clean(root)
    if _direct_input_hashes(root, plan) != input_hashes:
        raise FinalGateError("a mandatory command changed a direct final-gate input")
    if _source_snapshot(root, plan) != source_scope:
        raise FinalGateError("a mandatory command changed the final source closure")
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": ARTIFACT_KIND,
        "issue": ISSUE,
        "phase": PHASE,
        "child_issue": CHILD_ISSUE,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "plan": {
            "path": PLAN_PATH.as_posix(),
            "sha256": sha256_file(root / PLAN_PATH),
            "fingerprint": PLAN_FINGERPRINT,
        },
        "source": {
            "commit_sha": commit,
            "branch": _git(root, "rev-parse", "--abbrev-ref", "HEAD"),
            "tree_clean": True,
            **source_scope,
        },
        "phase7_manifest": phase7_manifest,
        "inputs": {"files": input_hashes},
        "environment": {
            "platform": platform.platform(),
            "python": sys.version,
        },
        "head_validation": head_report.to_dict(),
        "commands": commands,
        "diagnostics": {
            "rss_policy": plan.rss_mode,
            "reason": plan.rss_reason,
            "source_artifacts": [
                "tests/acceptance/issue_705/artifacts/phase_5_mjwarp_ppo.json",
                TRAINING_ARTIFACT_PATH.as_posix(),
            ],
        },
        "summary": {
            "required_lanes": ["A", "B", "C", "D"],
            "observed_lanes": sorted({command["lane"] for command in commands}),
            "command_runs": len(commands),
            "passed": sum(command["pytest"]["passed"] for command in commands),
            "skipped": sum(command["pytest"]["skipped"] for command in commands),
            "xfailed": sum(command["pytest"]["xfailed"] for command in commands),
            "xpassed": sum(command["pytest"]["xpassed"] for command in commands),
            "deselected": sum(command["pytest"]["deselected"] for command in commands),
        },
        "gate": {"passed": True, "errors": []},
    }
    errors = validate_final_gate_evidence(
        payload,
        root=root,
        plan=plan,
        require_promotion=False,
    )
    if errors:
        raise FinalGateError(
            "captured final evidence failed independent validation:\n"
            + "\n".join(f"- {error}" for error in errors)
        )
    return payload


def _mapping(value: object, path: str, errors: list[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        errors.append(f"{path}: expected mapping")
        return {}
    return cast(Mapping[str, Any], value)


def _exact_keys(value: Mapping[str, Any], expected: set[str], path: str, errors: list[str]) -> None:
    if set(value) != expected:
        errors.append(f"{path}: keys do not exactly match the v1 schema")


def _validate_artifact_source(
    artifact: Mapping[str, Any], *, root: Path, plan: FinalGatePlan, errors: list[str]
) -> None:
    source = _mapping(artifact.get("source"), "source", errors)
    _exact_keys(
        source,
        {
            "commit_sha",
            "branch",
            "tree_clean",
            "tracked_file_count",
            "tracked_paths_sha256",
            "tracked_modes_sha256",
            "tree_sha256",
        },
        "source",
        errors,
    )
    commit = source.get("commit_sha")
    if not isinstance(commit, str) or not _COMMIT_RE.fullmatch(commit):
        errors.append("source.commit_sha: expected full lowercase commit SHA")
        return
    if not _git_is_ancestor(root, commit):
        errors.append("source.commit_sha: must be an ancestor of current HEAD")
    if source.get("tree_clean") is not True:
        errors.append("source.tree_clean: must be true")
    if not isinstance(source.get("branch"), str) or not source.get("branch"):
        errors.append("source.branch: expected non-empty string")
    try:
        committed = _source_snapshot(root, plan, commit=commit)
        current = _source_snapshot(root, plan)
    except (OSError, subprocess.CalledProcessError, FinalGateError) as exc:
        errors.append(f"source: cannot recompute tracked closure: {type(exc).__name__}: {exc}")
        return
    for key, expected in committed.items():
        if source.get(key) != expected:
            errors.append(f"source.{key}: differs from source commit")
    for key, expected in current.items():
        if source.get(key) != expected:
            errors.append(f"source.{key}: stale against current final head")


def _validate_phase7_manifest_snapshot(
    artifact: Mapping[str, Any], *, root: Path, errors: list[str]
) -> None:
    recorded = _mapping(artifact.get("phase7_manifest"), "phase7_manifest", errors)
    _exact_keys(
        recorded,
        {"path", "mutable_claim_fields", "sha256", "snapshot"},
        "phase7_manifest",
        errors,
    )
    if recorded.get("path") != PHASE7_MANIFEST_PATH.as_posix():
        errors.append("phase7_manifest.path: differs from the promotion manifest path")
    if recorded.get("mutable_claim_fields") != list(_PHASE7_MUTABLE_CLAIM_FIELDS):
        errors.append("phase7_manifest.mutable_claim_fields: only evidence/status may change")
    snapshot = _mapping(recorded.get("snapshot"), "phase7_manifest.snapshot", errors)
    observed_hash = recorded.get("sha256")
    if not isinstance(observed_hash, str) or not _SHA256_RE.fullmatch(observed_hash):
        errors.append("phase7_manifest.sha256: expected sha256 hash")
    elif observed_hash != _canonical_sha256(snapshot):
        errors.append("phase7_manifest.sha256: differs from recorded immutable semantics")

    source = _mapping(artifact.get("source"), "source", errors)
    commit = source.get("commit_sha")
    if not isinstance(commit, str) or not _COMMIT_RE.fullmatch(commit):
        return
    try:
        committed = _phase7_manifest_snapshot(root, commit=commit)
        current = _phase7_manifest_snapshot(root)
    except (OSError, subprocess.CalledProcessError, FinalGateError) as exc:
        errors.append(
            f"phase7_manifest: cannot recompute immutable semantics: {type(exc).__name__}: {exc}"
        )
        return
    if dict(recorded) != committed:
        errors.append("phase7_manifest: differs from source commit immutable semantics")
    if dict(recorded) != current:
        errors.append("phase7_manifest: stale against current immutable semantics")


def _validate_artifact_inputs(
    artifact: Mapping[str, Any], *, root: Path, plan: FinalGatePlan, errors: list[str]
) -> None:
    inputs = _mapping(artifact.get("inputs"), "inputs", errors)
    _exact_keys(inputs, {"files"}, "inputs", errors)
    files = _mapping(inputs.get("files"), "inputs.files", errors)
    expected_paths = {path.as_posix() for path in plan.direct_inputs}
    if set(files) != expected_paths:
        errors.append("inputs.files: paths do not exactly match frozen direct inputs")
    source = _mapping(artifact.get("source"), "source", errors)
    commit = source.get("commit_sha")
    for path in plan.direct_inputs:
        key = path.as_posix()
        observed = files.get(key)
        if not isinstance(observed, str) or not _SHA256_RE.fullmatch(observed):
            errors.append(f"inputs.files.{key}: expected sha256 hash")
            continue
        current = root / path
        if not current.is_file() or sha256_file(current) != observed:
            errors.append(f"inputs.files.{key}: stale against current input")
        if isinstance(commit, str) and _COMMIT_RE.fullmatch(commit):
            try:
                if _git_blob_hash(root, commit, path) != observed:
                    errors.append(f"inputs.files.{key}: differs from source commit")
            except subprocess.CalledProcessError as exc:
                errors.append(f"inputs.files.{key}: absent from source commit: {exc}")


def command_evidence_errors(
    commands: object,
    *,
    plan: FinalGatePlan,
) -> tuple[str, ...]:
    """Validate the mandatory command matrix independently of other artifact fields."""

    errors: list[str] = []
    if not isinstance(commands, list):
        return ("commands: expected list",)
    by_name = {command.name: command for command in plan.commands}
    observed: dict[str, list[Mapping[str, Any]]] = {}
    names: set[str] = set()
    lanes: set[str] = set()
    expected_keys = {
        "name",
        "series",
        "claim_id",
        "lane",
        "repetition",
        "argv",
        "required_test_id",
        "exit_code",
        "duration_sec",
        "pytest",
        "stdout",
        "stderr",
    }
    for index, raw in enumerate(commands):
        row = _mapping(raw, f"commands[{index}]", errors)
        _exact_keys(row, expected_keys, f"commands[{index}]", errors)
        name = row.get("name")
        series = row.get("series")
        repetition = row.get("repetition")
        if not isinstance(name, str) or not name:
            errors.append(f"commands[{index}].name: expected non-empty string")
        elif name in names:
            errors.append(f"commands[{index}].name: duplicate")
        else:
            names.add(name)
        if not isinstance(series, str) or series not in by_name:
            errors.append(f"commands[{index}].series: unknown command")
            continue
        expected = by_name[series]
        observed.setdefault(series, []).append(row)
        if isinstance(repetition, bool) or not isinstance(repetition, int) or repetition < 1:
            errors.append(f"commands[{index}].repetition: expected positive integer")
        elif name != f"{series}#{repetition}":
            errors.append(f"commands[{index}].name: differs from series/repetition")
        expected_values = {
            "claim_id": expected.claim_id,
            "lane": expected.lane,
            "argv": list(expected.argv),
            "required_test_id": expected.required_test_id,
            "exit_code": 0,
        }
        for key, wanted in expected_values.items():
            if row.get(key) != wanted:
                errors.append(f"commands[{index}].{key}: differs from frozen command")
        lane = row.get("lane")
        if isinstance(lane, str):
            lanes.add(lane)
        duration = row.get("duration_sec")
        if isinstance(duration, bool) or not isinstance(duration, (int, float)) or duration <= 0:
            errors.append(f"commands[{index}].duration_sec: expected positive number")
        counts = _mapping(row.get("pytest"), f"commands[{index}].pytest", errors)
        _exact_keys(
            counts,
            {"passed", "skipped", "xfailed", "xpassed", "deselected"},
            f"commands[{index}].pytest",
            errors,
        )
        passed = counts.get("passed")
        if isinstance(passed, bool) or not isinstance(passed, int) or passed <= 0:
            errors.append(f"commands[{index}].pytest.passed: expected > 0")
        for category in _MANDATORY_ZERO_COUNTS:
            if counts.get(category) != 0:
                errors.append(f"commands[{index}].pytest.{category}: expected 0")
        stdout = row.get("stdout")
        if not isinstance(stdout, str) or expected.required_test_id not in stdout:
            errors.append(f"commands[{index}].stdout: required test ID is absent")
        if not isinstance(row.get("stderr"), str):
            errors.append(f"commands[{index}].stderr: expected string")
    if lanes != {"A", "B", "C", "D"}:
        errors.append("commands: observed lanes must be exactly A/B/C/D")
    if set(observed) != set(by_name):
        errors.append("commands: command series do not exactly match frozen plan")
    for name, expected in by_name.items():
        repetitions = {
            row.get("repetition")
            for row in observed.get(name, [])
            if isinstance(row.get("repetition"), int)
        }
        if repetitions != set(range(1, expected.repetitions + 1)):
            errors.append(f"commands[{name}]: expected repetitions 1..{expected.repetitions}")
    return tuple(errors)


def validate_final_gate_evidence(
    artifact: object,
    *,
    root: Path,
    plan: FinalGatePlan,
    require_promotion: bool = True,
) -> tuple[str, ...]:
    """Recompute final evidence and return every stale or malformed condition."""

    if not isinstance(artifact, Mapping):
        return ("artifact: expected mapping",)
    root = root.resolve()
    value = cast(Mapping[str, Any], artifact)
    errors: list[str] = []
    _exact_keys(
        value,
        {
            "schema_version",
            "kind",
            "issue",
            "phase",
            "child_issue",
            "generated_at_utc",
            "plan",
            "source",
            "phase7_manifest",
            "inputs",
            "environment",
            "head_validation",
            "commands",
            "diagnostics",
            "summary",
            "gate",
        },
        "artifact",
        errors,
    )
    identity = {
        "schema_version": SCHEMA_VERSION,
        "kind": ARTIFACT_KIND,
        "issue": ISSUE,
        "phase": PHASE,
        "child_issue": CHILD_ISSUE,
    }
    for key, expected in identity.items():
        if value.get(key) != expected:
            errors.append(f"{key}: expected {expected!r}")
    if not isinstance(value.get("generated_at_utc"), str):
        errors.append("generated_at_utc: expected string")
    plan_data = _mapping(value.get("plan"), "plan", errors)
    if plan_data != {
        "path": PLAN_PATH.as_posix(),
        "sha256": sha256_file(root / PLAN_PATH),
        "fingerprint": PLAN_FINGERPRINT,
    }:
        errors.append("plan: path/hash/fingerprint differs from current frozen plan")
    _validate_artifact_source(value, root=root, plan=plan, errors=errors)
    _validate_phase7_manifest_snapshot(value, root=root, errors=errors)
    _validate_artifact_inputs(value, root=root, plan=plan, errors=errors)
    errors.extend(command_evidence_errors(value.get("commands"), plan=plan))

    head = _mapping(value.get("head_validation"), "head_validation", errors)
    captured_components = head.get("components")
    captured_names: tuple[object, ...] = ()
    if isinstance(captured_components, list):
        captured_names = tuple(
            item.get("name") if isinstance(item, Mapping) else None for item in captured_components
        )
        for index, item in enumerate(captured_components):
            if (
                not isinstance(item, Mapping)
                or item.get("passed") is not True
                or item.get("errors") != []
            ):
                errors.append(f"head_validation.components[{index}]: expected captured PASS")
    else:
        errors.append("head_validation.components: expected list")
    if captured_names != _EXPECTED_COMPONENTS:
        errors.append("head_validation.components: component order differs from v1")
    if head.get("passed") is not True or head.get("errors") != []:
        errors.append("head_validation: captured gate must be PASS")

    diagnostics = _mapping(value.get("diagnostics"), "diagnostics", errors)
    if diagnostics.get("rss_policy") != "diagnostic_only":
        errors.append("diagnostics.rss_policy: RSS cannot be a final blocker")
    summary = _mapping(value.get("summary"), "summary", errors)
    raw_command_rows = value.get("commands")
    command_rows = (
        [cast(Mapping[str, Any], row) for row in raw_command_rows if isinstance(row, Mapping)]
        if isinstance(raw_command_rows, list)
        else []
    )
    observed_lanes = sorted(
        lane for lane in {row.get("lane") for row in command_rows} if isinstance(lane, str)
    )
    passed = 0
    for row in command_rows:
        counts = row.get("pytest")
        count = counts.get("passed") if isinstance(counts, Mapping) else None
        if isinstance(count, int) and not isinstance(count, bool):
            passed += count
    expected_summary = {
        "required_lanes": ["A", "B", "C", "D"],
        "observed_lanes": observed_lanes,
        "command_runs": len(command_rows),
        "passed": passed,
        "skipped": 0,
        "xfailed": 0,
        "xpassed": 0,
        "deselected": 0,
    }
    if dict(summary) != expected_summary:
        errors.append("summary: differs from independently recomputed command summary")

    current = validate_final_head(
        root,
        plan,
        require_promotion=require_promotion,
        artifact=value,
    )
    errors.extend(f"current head: {error}" for error in current.errors)
    expected_gate = {"passed": not errors, "errors": list(errors)}
    if value.get("gate") != expected_gate:
        errors.append("gate: differs from independent final validation")
    return tuple(errors)


def load_final_gate_evidence(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FinalGateError(f"cannot load final gate evidence {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise FinalGateError(f"final gate evidence {path} must contain a JSON object")
    return cast(dict[str, Any], value)


def write_final_gate_evidence(artifact: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(artifact), indent=2, sort_keys=True) + "\n", encoding="utf-8")


__all__ = [
    "ARTIFACT_KIND",
    "ARTIFACT_PATH",
    "CHILD_ISSUE",
    "FinalGateCommand",
    "FinalGateComponent",
    "FinalGateError",
    "FinalGatePlan",
    "FinalGatePlanError",
    "FinalGateReport",
    "ISSUE",
    "PHASE",
    "PLAN_FINGERPRINT",
    "PLAN_PATH",
    "capture_final_gate_evidence",
    "command_evidence_errors",
    "load_final_gate_evidence",
    "load_final_gate_plan",
    "sha256_file",
    "validate_final_gate_evidence",
    "validate_final_head",
    "write_final_gate_evidence",
]
