"""Claim-to-test inventory contracts for Issue #705."""

from __future__ import annotations

import ast
import re
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

from omegaconf import OmegaConf

from unilab.tools.phase_acceptance import (
    ManifestValidationError,
    PhaseAcceptanceManifest,
    load_phase_acceptance,
)

SCHEMA_VERSION = 1
ISSUE = 705
INTEGRATION_BRANCH = "feat/issue-705-manager-mjwarp"
PHASES = tuple(range(8))

_CLAIM_ID_RE = re.compile(r"^P(?P<phase>[0-7])-[A-Z0-9]+(?:-[A-Z0-9]+)*$")
_TEST_ID_RE = re.compile(
    r"^(?P<path>(?:tests|benchmark)/[^:\s]+\.py)"
    r"(?P<nodes>(?:::[A-Za-z_][A-Za-z0-9_]*(?:\[[^\]\r\n]+\])?)*)$"
)
_MISSING = object()


class InventoryTestState(str, Enum):
    EXISTING = "existing"
    TARGET = "target"


class EvidenceRole(str, Enum):
    ACCEPTANCE = "acceptance"
    SUPPORTING = "supporting"


class EvidenceKind(str, Enum):
    CONTRACT = "contract"
    FAULT = "fault"
    DIFFERENTIAL = "differential"
    LIFECYCLE = "lifecycle"
    EFFECT = "effect"
    PERFORMANCE = "performance"
    TRAINING = "training"
    PROVENANCE = "provenance"
    LIVENESS = "liveness"
    SMOKE = "smoke"


@dataclass(frozen=True)
class ClaimTestEntry:
    claim_id: str
    test_id: str
    state: InventoryTestState
    role: EvidenceRole
    evidence_kind: EvidenceKind
    owner: str
    oracle: str
    gap: str | None

    @property
    def phase(self) -> int:
        match = _CLAIM_ID_RE.fullmatch(self.claim_id)
        if match is None:  # Parsed inventories already reject this case.
            return -1
        return int(match.group("phase"))


@dataclass(frozen=True)
class ClaimGapInventory:
    schema_version: int
    issue: int
    integration_branch: str
    entries: tuple[ClaimTestEntry, ...]
    source: Path


@dataclass(frozen=True)
class ClaimGapAuditReport:
    phases: tuple[int, ...]
    claims: int
    entries: int
    existing: int
    targets: int
    supporting: int
    errors: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.errors


class ClaimGapInventoryError(ValueError):
    def __init__(self, source: Path, errors: Iterable[str]) -> None:
        self.source = source
        self.errors = tuple(errors)
        detail = "\n".join(f"- {error}" for error in self.errors)
        super().__init__(f"invalid claim-to-test inventory {source}:\n{detail}")


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

    def nullable_string(self, value: Any, path: str) -> str | None:
        if value is _MISSING or value is None:
            return None
        return self.string(value, path)

    def integer(self, value: Any, path: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            self.errors.append(f"{path}: expected integer")
            return 0
        return int(value)

    def enum(self, value: Any, path: str, enum_type: type[Enum], default: Enum) -> Any:
        if value is _MISSING:
            return default
        try:
            return enum_type(value)
        except (TypeError, ValueError):
            allowed = [item.value for item in enum_type]
            self.errors.append(f"{path}: expected one of {allowed!r}")
            return default


_ROOT_KEYS = ("schema_version", "issue", "integration_branch", "entries")
_ENTRY_KEYS = (
    "claim_id",
    "test_id",
    "state",
    "role",
    "evidence_kind",
    "owner",
    "oracle",
    "gap",
)


def parse_claim_gap_inventory(raw: Any, *, source: Path = Path("<memory>")) -> ClaimGapInventory:
    parser = _Parser()
    values = parser.mapping(raw, "inventory", _ROOT_KEYS)
    schema_version = parser.integer(values.get("schema_version", _MISSING), "schema_version")
    if schema_version != SCHEMA_VERSION:
        parser.errors.append(f"schema_version: expected {SCHEMA_VERSION}, got {schema_version!r}")
    issue = parser.integer(values.get("issue", _MISSING), "issue")
    integration_branch = parser.string(
        values.get("integration_branch", _MISSING), "integration_branch"
    )

    raw_entries = values.get("entries", _MISSING)
    if not isinstance(raw_entries, list):
        parser.errors.append("entries: expected non-empty list")
        raw_entries = []
    elif not raw_entries:
        parser.errors.append("entries: must not be empty")

    entries: list[ClaimTestEntry] = []
    for index, raw_entry in enumerate(raw_entries):
        path = f"entries[{index}]"
        entry_values = parser.mapping(raw_entry, path, _ENTRY_KEYS)
        entry = ClaimTestEntry(
            claim_id=parser.string(entry_values.get("claim_id", _MISSING), f"{path}.claim_id"),
            test_id=parser.string(entry_values.get("test_id", _MISSING), f"{path}.test_id"),
            state=parser.enum(
                entry_values.get("state", _MISSING),
                f"{path}.state",
                InventoryTestState,
                InventoryTestState.TARGET,
            ),
            role=parser.enum(
                entry_values.get("role", _MISSING),
                f"{path}.role",
                EvidenceRole,
                EvidenceRole.ACCEPTANCE,
            ),
            evidence_kind=parser.enum(
                entry_values.get("evidence_kind", _MISSING),
                f"{path}.evidence_kind",
                EvidenceKind,
                EvidenceKind.CONTRACT,
            ),
            owner=parser.string(entry_values.get("owner", _MISSING), f"{path}.owner"),
            oracle=parser.string(entry_values.get("oracle", _MISSING), f"{path}.oracle"),
            gap=parser.nullable_string(entry_values.get("gap", _MISSING), f"{path}.gap"),
        )
        if entry.claim_id and _CLAIM_ID_RE.fullmatch(entry.claim_id) is None:
            parser.errors.append(f"{path}.claim_id: expected P0..P7 uppercase kebab-case ID")
        match = _TEST_ID_RE.fullmatch(entry.test_id)
        if entry.test_id and match is None:
            parser.errors.append(
                f"{path}.test_id: expected repository pytest ID under tests/ or benchmark/"
            )
        elif match is not None:
            test_path = Path(match.group("path"))
            nodes = tuple(part for part in match.group("nodes").split("::") if part)
            if test_path.is_absolute() or ".." in test_path.parts or "." in test_path.parts:
                parser.errors.append(f"{path}.test_id: path must stay within the repository")
            if entry.state == InventoryTestState.TARGET and not nodes:
                parser.errors.append(
                    f"{path}.test_id: target test requires an explicit pytest node"
                )
            if nodes and not _strip_parametrization(nodes[-1]).startswith("test_"):
                parser.errors.append(f"{path}.test_id: explicit pytest ID must end in a test node")
        if entry.state == InventoryTestState.EXISTING and entry.gap is not None:
            parser.errors.append(f"{path}.gap: existing test must not declare a gap")
        if entry.state == InventoryTestState.TARGET and entry.gap is None:
            parser.errors.append(f"{path}.gap: target test requires an explicit gap")
        if entry.role == EvidenceRole.ACCEPTANCE and entry.evidence_kind == EvidenceKind.SMOKE:
            parser.errors.append(f"{path}: smoke evidence cannot be an acceptance oracle")
        entries.append(entry)

    duplicate_pairs = sorted(
        pair
        for pair, count in Counter((entry.claim_id, entry.test_id) for entry in entries).items()
        if count > 1
    )
    if duplicate_pairs:
        parser.errors.append(f"entries: duplicate claim/test mappings {duplicate_pairs!r}")

    if parser.errors:
        raise ClaimGapInventoryError(source, parser.errors)
    return ClaimGapInventory(
        schema_version=schema_version,
        issue=issue,
        integration_branch=integration_branch,
        entries=tuple(entries),
        source=source,
    )


def load_claim_gap_inventory(path: Path) -> ClaimGapInventory:
    try:
        config = OmegaConf.load(path)
        raw = OmegaConf.to_container(config, resolve=False)
    except Exception as exc:  # noqa: BLE001 - normalize malformed YAML for the CLI
        raise ClaimGapInventoryError(
            path, [f"cannot load YAML: {type(exc).__name__}: {exc}"]
        ) from exc
    return parse_claim_gap_inventory(raw, source=path)


def _test_id_parts(test_id: str) -> tuple[Path, tuple[str, ...]] | None:
    match = _TEST_ID_RE.fullmatch(test_id)
    if match is None:
        return None
    nodes = tuple(part for part in match.group("nodes").split("::") if part)
    return Path(match.group("path")), nodes


def _strip_parametrization(node: str) -> str:
    return node.split("[", maxsplit=1)[0]


def _node_exists(path: Path, nodes: tuple[str, ...]) -> tuple[bool, str | None]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, UnicodeError, SyntaxError) as exc:
        return False, f"cannot parse {path}: {type(exc).__name__}: {exc}"
    if not nodes:
        has_test = any(
            (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name.startswith("test_")
            )
            or (
                isinstance(node, ast.ClassDef)
                and node.name.startswith("Test")
                and any(
                    isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and child.name.startswith("test_")
                    for child in node.body
                )
            )
            for node in tree.body
        )
        return has_test, None

    body: list[ast.stmt] = tree.body
    for index, raw_name in enumerate(nodes):
        name = _strip_parametrization(raw_name)
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
            return False, None
        if index < len(nodes) - 1:
            if not isinstance(found, ast.ClassDef):
                return False, None
            body = found.body
    return True, None


def load_phase_manifests(
    manifest_dir: Path, phases: Iterable[int]
) -> tuple[tuple[PhaseAcceptanceManifest, ...], tuple[str, ...]]:
    manifests: list[PhaseAcceptanceManifest] = []
    errors: list[str] = []
    for phase in phases:
        path = manifest_dir / f"phase_{phase}.yaml"
        if not path.is_file():
            errors.append(f"phase {phase}: missing manifest {path}")
            continue
        try:
            manifest = load_phase_acceptance(path)
        except ManifestValidationError as exc:
            errors.extend(f"phase {phase}: {error}" for error in exc.errors)
            continue
        if manifest.phase != phase:
            errors.append(f"phase {phase}: manifest declares phase {manifest.phase}")
        if manifest.issue != ISSUE:
            errors.append(f"phase {phase}: expected issue {ISSUE}, got {manifest.issue}")
        if manifest.integration_branch != INTEGRATION_BRANCH:
            errors.append(
                f"phase {phase}: expected integration branch {INTEGRATION_BRANCH!r}, "
                f"got {manifest.integration_branch!r}"
            )
        manifests.append(manifest)
    return tuple(manifests), tuple(errors)


def audit_claim_gaps(
    inventory: ClaimGapInventory,
    manifests: Iterable[PhaseAcceptanceManifest],
    *,
    repo_root: Path,
    phases: Iterable[int],
) -> ClaimGapAuditReport:
    selected_phases = tuple(sorted(set(phases)))
    selected = set(selected_phases)
    errors: list[str] = []
    if not selected:
        errors.append("phases: at least one phase must be selected")
    invalid_phases = sorted(selected - set(PHASES))
    if invalid_phases:
        errors.append(f"phases: unsupported phases {invalid_phases!r}")
    if inventory.issue != ISSUE:
        errors.append(f"inventory issue: expected {ISSUE}, got {inventory.issue}")
    if inventory.integration_branch != INTEGRATION_BRANCH:
        errors.append(
            f"inventory integration_branch: expected {INTEGRATION_BRANCH!r}, "
            f"got {inventory.integration_branch!r}"
        )

    selected_manifests = tuple(manifest for manifest in manifests if manifest.phase in selected)
    manifest_phases = [manifest.phase for manifest in selected_manifests]
    for phase in sorted(selected - set(manifest_phases)):
        errors.append(f"phase {phase}: no manifest supplied to claim audit")
    duplicate_manifest_phases = sorted(
        phase for phase, count in Counter(manifest_phases).items() if count > 1
    )
    if duplicate_manifest_phases:
        errors.append(f"manifests: duplicate phases {duplicate_manifest_phases!r}")

    claims = [claim for manifest in selected_manifests for claim in manifest.claims]
    duplicate_claim_ids = sorted(
        claim_id
        for claim_id, count in Counter(claim.claim_id for claim in claims).items()
        if count > 1
    )
    if duplicate_claim_ids:
        errors.append(f"manifests: duplicate claim IDs {duplicate_claim_ids!r}")
    claim_by_id = {claim.claim_id: claim for claim in claims}
    entries = tuple(entry for entry in inventory.entries if entry.phase in selected)
    entries_by_claim: dict[str, list[ClaimTestEntry]] = {}
    for entry in entries:
        entries_by_claim.setdefault(entry.claim_id, []).append(entry)

    for claim_id in sorted(set(entries_by_claim) - set(claim_by_id)):
        errors.append(f"{claim_id}: inventory entry has no matching manifest claim")

    for claim_id, claim in sorted(claim_by_id.items()):
        claim_entries = entries_by_claim.get(claim_id, [])
        acceptance_ids = {
            entry.test_id for entry in claim_entries if entry.role == EvidenceRole.ACCEPTANCE
        }
        required_ids = set(claim.required_test_ids)
        for test_id in sorted(required_ids - acceptance_ids):
            errors.append(
                f"{claim_id}: required test `{test_id}` has no acceptance inventory entry"
            )
        for test_id in sorted(acceptance_ids - required_ids):
            errors.append(
                f"{claim_id}: acceptance inventory test `{test_id}` is not required by manifest"
            )
        if not acceptance_ids:
            errors.append(f"{claim_id}: no acceptance oracle is mapped")

        for entry in claim_entries:
            if entry.owner != claim.owner:
                errors.append(
                    f"{claim_id}/{entry.test_id}: owner {entry.owner!r} does not match "
                    f"manifest owner {claim.owner!r}"
                )
            parts = _test_id_parts(entry.test_id)
            if parts is None:
                continue
            relative_path, nodes = parts
            resolved_root = repo_root.resolve()
            absolute_path = (repo_root / relative_path).resolve()
            if not absolute_path.is_relative_to(resolved_root):
                errors.append(f"{claim_id}/{entry.test_id}: test path escapes repository root")
                continue
            if entry.state == InventoryTestState.EXISTING:
                if not absolute_path.is_file():
                    errors.append(f"{claim_id}/{entry.test_id}: existing test file is missing")
                    continue
                exists, parse_error = _node_exists(absolute_path, nodes)
                if parse_error is not None:
                    errors.append(f"{claim_id}/{entry.test_id}: {parse_error}")
                elif not exists:
                    errors.append(f"{claim_id}/{entry.test_id}: existing pytest node is missing")
            elif absolute_path.is_file():
                exists, parse_error = _node_exists(absolute_path, nodes)
                if parse_error is not None:
                    errors.append(f"{claim_id}/{entry.test_id}: {parse_error}")
                elif exists:
                    errors.append(
                        f"{claim_id}/{entry.test_id}: target pytest node already exists; "
                        "promote it to existing"
                    )

    return ClaimGapAuditReport(
        phases=selected_phases,
        claims=len(claim_by_id),
        entries=len(entries),
        existing=sum(entry.state == InventoryTestState.EXISTING for entry in entries),
        targets=sum(entry.state == InventoryTestState.TARGET for entry in entries),
        supporting=sum(entry.role == EvidenceRole.SUPPORTING for entry in entries),
        errors=tuple(errors),
    )
