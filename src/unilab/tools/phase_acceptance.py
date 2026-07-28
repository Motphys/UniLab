"""Strict contracts for phased implementation acceptance manifests."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from math import isfinite
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from omegaconf import OmegaConf

SCHEMA_VERSION = 1

_CLAIM_ID_RE = re.compile(r"^[A-Z][A-Z0-9]*(?:-[A-Z0-9]+)+$")
_COMMIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_CONFIG_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_MISSING = object()


class AcceptanceLane(str, Enum):
    PR = "A"
    BACKEND = "B"
    GPU = "C"
    BENCHMARK = "D"


class ClaimStatus(str, Enum):
    PLANNED = "planned"
    IMPLEMENTED = "implemented"
    VERIFIED = "verified"
    PROMOTED = "promoted"


class EvidenceResult(str, Enum):
    NOT_RUN = "NOT_RUN"
    PASS = "PASS"
    FAIL = "FAIL"
    BLOCKED = "BLOCKED"


@dataclass(frozen=True)
class EnvironmentSpec:
    dependencies: tuple[str, ...]
    hardware: str
    owner_yaml: str | None
    seeds: tuple[int, ...]
    batch_sizes: tuple[int, ...]
    dtype: str | None
    plan_fingerprint: str | None


@dataclass(frozen=True)
class AcceptanceSpec:
    tolerance: tuple[tuple[str, float], ...]
    thresholds: tuple[tuple[str, float], ...]
    repetitions: int
    max_dispersion: float | None
    failure_semantics: str


@dataclass(frozen=True)
class EvidenceSpec:
    result: EvidenceResult
    artifact_refs: tuple[str, ...]
    commit_sha: str | None
    config_hash: str | None
    executed_test_ids: tuple[str, ...]
    skipped_test_ids: tuple[str, ...]
    xfailed_test_ids: tuple[str, ...]
    summary: str


@dataclass(frozen=True)
class InvalidationSpec:
    paths: tuple[str, ...]
    capabilities: tuple[str, ...]
    fingerprints: tuple[str, ...]


@dataclass(frozen=True)
class AcceptanceClaim:
    claim_id: str
    expected: str
    risk: str
    owner: str
    oracle: str
    commands: tuple[str, ...]
    lane: AcceptanceLane
    required: bool
    required_test_ids: tuple[str, ...]
    environment: EnvironmentSpec
    acceptance: AcceptanceSpec
    evidence: EvidenceSpec
    invalidation: InvalidationSpec
    status: ClaimStatus


@dataclass(frozen=True)
class PhaseAcceptanceManifest:
    schema_version: int
    issue: int
    phase: int
    integration_branch: str
    required_lanes: tuple[AcceptanceLane, ...]
    claims: tuple[AcceptanceClaim, ...]
    source: Path


class ManifestValidationError(ValueError):
    def __init__(self, source: Path, errors: list[str] | tuple[str, ...]) -> None:
        self.source = source
        self.errors = tuple(errors)
        detail = "\n".join(f"- {error}" for error in self.errors)
        super().__init__(f"invalid phase acceptance manifest {source}:\n{detail}")


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

    def string(self, value: Any, path: str, *, allow_empty: bool = False) -> str:
        if value is _MISSING:
            return ""
        if not isinstance(value, str):
            self.errors.append(f"{path}: expected string")
            return ""
        if not allow_empty and not value.strip():
            self.errors.append(f"{path}: must not be empty")
        if "${" in value:
            self.errors.append(f"{path}: interpolation is not allowed")
        return value

    def nullable_string(self, value: Any, path: str) -> str | None:
        if value is _MISSING or value is None:
            return None
        return self.string(value, path)

    def integer(self, value: Any, path: str, *, minimum: int | None = None) -> int:
        if value is _MISSING:
            return 0
        if isinstance(value, bool) or not isinstance(value, int):
            self.errors.append(f"{path}: expected integer")
            return 0
        if minimum is not None and value < minimum:
            self.errors.append(f"{path}: must be >= {minimum}")
        return int(value)

    def boolean(self, value: Any, path: str) -> bool:
        if value is _MISSING:
            return False
        if not isinstance(value, bool):
            self.errors.append(f"{path}: expected boolean")
            return False
        return value

    def number(self, value: Any, path: str, *, minimum: float | None = None) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            self.errors.append(f"{path}: expected number")
            return 0.0
        result = float(value)
        if not isfinite(result):
            self.errors.append(f"{path}: must be finite")
        if minimum is not None and result < minimum:
            self.errors.append(f"{path}: must be >= {minimum}")
        return result

    def nullable_number(self, value: Any, path: str, *, minimum: float = 0.0) -> float | None:
        if value is _MISSING or value is None:
            return None
        return self.number(value, path, minimum=minimum)

    def string_list(
        self,
        value: Any,
        path: str,
        *,
        allow_empty: bool = True,
    ) -> tuple[str, ...]:
        if value is _MISSING:
            return ()
        if not isinstance(value, list):
            self.errors.append(f"{path}: expected list of strings")
            return ()
        values = tuple(self.string(item, f"{path}[{index}]") for index, item in enumerate(value))
        if not allow_empty and not values:
            self.errors.append(f"{path}: must not be empty")
        duplicates = sorted(item for item, count in Counter(values).items() if count > 1)
        if duplicates:
            self.errors.append(f"{path}: duplicate values {duplicates!r}")
        return values

    def int_list(self, value: Any, path: str, *, minimum: int) -> tuple[int, ...]:
        if value is _MISSING:
            return ()
        if not isinstance(value, list):
            self.errors.append(f"{path}: expected list of integers")
            return ()
        values = tuple(
            self.integer(item, f"{path}[{index}]", minimum=minimum)
            for index, item in enumerate(value)
        )
        if len(set(values)) != len(values):
            self.errors.append(f"{path}: duplicate values")
        return values

    def number_mapping(
        self, value: Any, path: str, *, minimum: float | None = None
    ) -> tuple[tuple[str, float], ...]:
        if value is _MISSING:
            return ()
        if not isinstance(value, dict):
            self.errors.append(f"{path}: expected mapping of numeric values")
            return ()
        rows: list[tuple[str, float]] = []
        for key, item in sorted(value.items(), key=lambda row: str(row[0])):
            parsed_key = self.string(key, f"{path}.<key>")
            rows.append((parsed_key, self.number(item, f"{path}.{parsed_key}", minimum=minimum)))
        return tuple(rows)

    def enum(self, value: Any, path: str, enum_type: type[Enum], default: Enum) -> Any:
        if value is _MISSING:
            return default
        try:
            return enum_type(value)
        except (TypeError, ValueError):
            allowed = [item.value for item in enum_type]
            self.errors.append(f"{path}: expected one of {allowed!r}")
            return default

    def repo_path(self, value: Any, path: str, *, nullable: bool = False) -> str | None:
        if nullable and (value is _MISSING or value is None):
            return None
        parsed = self.string(value, path)
        candidate = Path(parsed)
        if candidate.is_absolute() or ".." in candidate.parts:
            self.errors.append(f"{path}: must be a repository-relative path or glob")
        return parsed

    def repo_path_list(self, value: Any, path: str) -> tuple[str, ...]:
        values = self.string_list(value, path, allow_empty=False)
        for index, item in enumerate(values):
            candidate = Path(item)
            if candidate.is_absolute() or ".." in candidate.parts:
                self.errors.append(f"{path}[{index}]: must be a repository-relative path or glob")
        return values


_ROOT_KEYS = (
    "schema_version",
    "issue",
    "phase",
    "integration_branch",
    "required_lanes",
    "claims",
)
_CLAIM_KEYS = (
    "claim_id",
    "expected",
    "risk",
    "owner",
    "oracle",
    "commands",
    "lane",
    "required",
    "required_test_ids",
    "environment",
    "acceptance",
    "evidence",
    "invalidation",
    "status",
)
_ENVIRONMENT_KEYS = (
    "dependencies",
    "hardware",
    "owner_yaml",
    "seeds",
    "batch_sizes",
    "dtype",
    "plan_fingerprint",
)
_ACCEPTANCE_KEYS = (
    "tolerance",
    "thresholds",
    "repetitions",
    "max_dispersion",
    "failure_semantics",
)
_EVIDENCE_KEYS = (
    "result",
    "artifact_refs",
    "commit_sha",
    "config_hash",
    "executed_test_ids",
    "skipped_test_ids",
    "xfailed_test_ids",
    "summary",
)
_INVALIDATION_KEYS = ("paths", "capabilities", "fingerprints")


def _parse_environment(parser: _Parser, raw: Any, path: str) -> EnvironmentSpec:
    values = parser.mapping(raw, path, _ENVIRONMENT_KEYS)
    return EnvironmentSpec(
        dependencies=parser.string_list(
            values.get("dependencies", _MISSING), f"{path}.dependencies", allow_empty=False
        ),
        hardware=parser.string(values.get("hardware", _MISSING), f"{path}.hardware"),
        owner_yaml=parser.repo_path(
            values.get("owner_yaml", _MISSING), f"{path}.owner_yaml", nullable=True
        ),
        seeds=parser.int_list(values.get("seeds", _MISSING), f"{path}.seeds", minimum=0),
        batch_sizes=parser.int_list(
            values.get("batch_sizes", _MISSING), f"{path}.batch_sizes", minimum=1
        ),
        dtype=parser.nullable_string(values.get("dtype", _MISSING), f"{path}.dtype"),
        plan_fingerprint=parser.nullable_string(
            values.get("plan_fingerprint", _MISSING), f"{path}.plan_fingerprint"
        ),
    )


def _parse_acceptance(parser: _Parser, raw: Any, path: str) -> AcceptanceSpec:
    values = parser.mapping(raw, path, _ACCEPTANCE_KEYS)
    return AcceptanceSpec(
        tolerance=parser.number_mapping(
            values.get("tolerance", _MISSING), f"{path}.tolerance", minimum=0.0
        ),
        thresholds=parser.number_mapping(values.get("thresholds", _MISSING), f"{path}.thresholds"),
        repetitions=parser.integer(
            values.get("repetitions", _MISSING), f"{path}.repetitions", minimum=1
        ),
        max_dispersion=parser.nullable_number(
            values.get("max_dispersion", _MISSING), f"{path}.max_dispersion"
        ),
        failure_semantics=parser.string(
            values.get("failure_semantics", _MISSING), f"{path}.failure_semantics"
        ),
    )


def _parse_evidence(parser: _Parser, raw: Any, path: str) -> EvidenceSpec:
    values = parser.mapping(raw, path, _EVIDENCE_KEYS)
    evidence = EvidenceSpec(
        result=parser.enum(
            values.get("result", _MISSING),
            f"{path}.result",
            EvidenceResult,
            EvidenceResult.NOT_RUN,
        ),
        artifact_refs=parser.string_list(
            values.get("artifact_refs", _MISSING), f"{path}.artifact_refs"
        ),
        commit_sha=parser.nullable_string(values.get("commit_sha", _MISSING), f"{path}.commit_sha"),
        config_hash=parser.nullable_string(
            values.get("config_hash", _MISSING), f"{path}.config_hash"
        ),
        executed_test_ids=parser.string_list(
            values.get("executed_test_ids", _MISSING), f"{path}.executed_test_ids"
        ),
        skipped_test_ids=parser.string_list(
            values.get("skipped_test_ids", _MISSING), f"{path}.skipped_test_ids"
        ),
        xfailed_test_ids=parser.string_list(
            values.get("xfailed_test_ids", _MISSING), f"{path}.xfailed_test_ids"
        ),
        summary=parser.string(values.get("summary", _MISSING), f"{path}.summary", allow_empty=True),
    )
    for index, artifact_ref in enumerate(evidence.artifact_refs):
        if artifact_ref.startswith("https://"):
            if not urlparse(artifact_ref).netloc:
                parser.errors.append(
                    f"{path}.artifact_refs[{index}]: HTTPS artifact URL requires a host"
                )
            continue
        if "://" in artifact_ref:
            parser.errors.append(
                f"{path}.artifact_refs[{index}]: only HTTPS artifact URLs are allowed"
            )
            continue
        candidate = Path(artifact_ref)
        if candidate.is_absolute() or ".." in candidate.parts:
            parser.errors.append(
                f"{path}.artifact_refs[{index}]: must be an HTTPS URL or repository-relative path"
            )
    return evidence


def _parse_invalidation(parser: _Parser, raw: Any, path: str) -> InvalidationSpec:
    values = parser.mapping(raw, path, _INVALIDATION_KEYS)
    return InvalidationSpec(
        paths=parser.repo_path_list(values.get("paths", _MISSING), f"{path}.paths"),
        capabilities=parser.string_list(
            values.get("capabilities", _MISSING), f"{path}.capabilities"
        ),
        fingerprints=parser.string_list(
            values.get("fingerprints", _MISSING), f"{path}.fingerprints"
        ),
    )


def _validate_evidence_state(parser: _Parser, claim: AcceptanceClaim, path: str) -> None:
    evidence = claim.evidence
    if evidence.commit_sha is not None and not _COMMIT_SHA_RE.fullmatch(evidence.commit_sha):
        parser.errors.append(f"{path}.evidence.commit_sha: expected full 40-character SHA")
    if evidence.config_hash is not None and not _CONFIG_HASH_RE.fullmatch(evidence.config_hash):
        parser.errors.append(f"{path}.evidence.config_hash: expected sha256:<64 hex>")

    if claim.status == ClaimStatus.PLANNED:
        if evidence.result != EvidenceResult.NOT_RUN:
            parser.errors.append(f"{path}.evidence.result: planned claim must be NOT_RUN")
        if (
            evidence.artifact_refs
            or evidence.commit_sha
            or evidence.config_hash
            or evidence.executed_test_ids
            or evidence.skipped_test_ids
            or evidence.xfailed_test_ids
        ):
            parser.errors.append(f"{path}.evidence: planned claim must not carry pass artifacts")

    if evidence.result == EvidenceResult.PASS and claim.status not in {
        ClaimStatus.VERIFIED,
        ClaimStatus.PROMOTED,
    }:
        parser.errors.append(f"{path}.status: PASS evidence requires verified or promoted status")

    if claim.status not in {ClaimStatus.VERIFIED, ClaimStatus.PROMOTED}:
        return
    if evidence.result != EvidenceResult.PASS:
        parser.errors.append(f"{path}.evidence.result: verified claim must be PASS")
    if not evidence.artifact_refs:
        parser.errors.append(f"{path}.evidence.artifact_refs: verified claim requires artifacts")
    if evidence.commit_sha is None:
        parser.errors.append(f"{path}.evidence.commit_sha: expected full 40-character SHA")
    if evidence.config_hash is None:
        parser.errors.append(f"{path}.evidence.config_hash: expected sha256:<64 hex>")
    if not evidence.summary.strip():
        parser.errors.append(f"{path}.evidence.summary: verified claim requires a summary")
    if evidence.skipped_test_ids:
        parser.errors.append(f"{path}.evidence.skipped_test_ids: verified claim cannot skip tests")
    if evidence.xfailed_test_ids:
        parser.errors.append(f"{path}.evidence.xfailed_test_ids: verified claim cannot xfail tests")

    executed = set(evidence.executed_test_ids)
    for test_id in claim.required_test_ids:
        if test_id not in executed:
            parser.errors.append(f"{path}.evidence: required test `{test_id}` was not executed")
        if test_id in evidence.skipped_test_ids:
            parser.errors.append(f"{path}.evidence: required test `{test_id}` was skipped")
        if test_id in evidence.xfailed_test_ids:
            parser.errors.append(f"{path}.evidence: required test `{test_id}` was xfailed")


def _parse_claim(parser: _Parser, raw: Any, index: int) -> AcceptanceClaim:
    path = f"claims[{index}]"
    values = parser.mapping(raw, path, _CLAIM_KEYS)
    claim = AcceptanceClaim(
        claim_id=parser.string(values.get("claim_id", _MISSING), f"{path}.claim_id"),
        expected=parser.string(values.get("expected", _MISSING), f"{path}.expected"),
        risk=parser.string(values.get("risk", _MISSING), f"{path}.risk"),
        owner=parser.string(values.get("owner", _MISSING), f"{path}.owner"),
        oracle=parser.string(values.get("oracle", _MISSING), f"{path}.oracle"),
        commands=parser.string_list(
            values.get("commands", _MISSING), f"{path}.commands", allow_empty=False
        ),
        lane=parser.enum(
            values.get("lane", _MISSING), f"{path}.lane", AcceptanceLane, AcceptanceLane.PR
        ),
        required=parser.boolean(values.get("required", _MISSING), f"{path}.required"),
        required_test_ids=parser.string_list(
            values.get("required_test_ids", _MISSING),
            f"{path}.required_test_ids",
            allow_empty=False,
        ),
        environment=_parse_environment(
            parser, values.get("environment", _MISSING), f"{path}.environment"
        ),
        acceptance=_parse_acceptance(
            parser, values.get("acceptance", _MISSING), f"{path}.acceptance"
        ),
        evidence=_parse_evidence(parser, values.get("evidence", _MISSING), f"{path}.evidence"),
        invalidation=_parse_invalidation(
            parser, values.get("invalidation", _MISSING), f"{path}.invalidation"
        ),
        status=parser.enum(
            values.get("status", _MISSING),
            f"{path}.status",
            ClaimStatus,
            ClaimStatus.PLANNED,
        ),
    )
    if claim.claim_id and not _CLAIM_ID_RE.fullmatch(claim.claim_id):
        parser.errors.append(f"{path}.claim_id: expected stable uppercase kebab-case ID")
    _validate_evidence_state(parser, claim, path)
    return claim


def parse_phase_acceptance(raw: Any, *, source: Path = Path("<memory>")) -> PhaseAcceptanceManifest:
    parser = _Parser()
    values = parser.mapping(raw, "manifest", _ROOT_KEYS)
    schema_version = parser.integer(values.get("schema_version", _MISSING), "schema_version")
    if schema_version != SCHEMA_VERSION:
        parser.errors.append(f"schema_version: expected {SCHEMA_VERSION}, got {schema_version!r}")
    issue = parser.integer(values.get("issue", _MISSING), "issue", minimum=1)
    phase = parser.integer(values.get("phase", _MISSING), "phase", minimum=0)
    if phase > 7:
        parser.errors.append("phase: expected value in range 0..7")
    integration_branch = parser.string(
        values.get("integration_branch", _MISSING), "integration_branch"
    )

    raw_lanes = parser.string_list(
        values.get("required_lanes", _MISSING), "required_lanes", allow_empty=False
    )
    required_lanes = tuple(
        parser.enum(lane, f"required_lanes[{index}]", AcceptanceLane, AcceptanceLane.PR)
        for index, lane in enumerate(raw_lanes)
    )

    raw_claims = values.get("claims", _MISSING)
    if not isinstance(raw_claims, list):
        parser.errors.append("claims: expected non-empty list")
        raw_claims = []
    elif not raw_claims:
        parser.errors.append("claims: must not be empty")
    claims = tuple(_parse_claim(parser, raw, index) for index, raw in enumerate(raw_claims))

    claim_ids = [claim.claim_id for claim in claims]
    duplicate_ids = sorted(item for item, count in Counter(claim_ids).items() if count > 1)
    if duplicate_ids:
        parser.errors.append(f"claims: duplicate claim IDs {duplicate_ids!r}")
    expected_prefix = f"P{phase}-"
    for index, claim in enumerate(claims):
        if claim.claim_id and not claim.claim_id.startswith(expected_prefix):
            parser.errors.append(
                f"claims[{index}].claim_id: phase {phase} claim must start with `{expected_prefix}`"
            )

    required_claim_lanes = {claim.lane for claim in claims if claim.required}
    if set(required_lanes) != required_claim_lanes:
        parser.errors.append(
            "required_lanes: must exactly match lanes used by required claims; "
            f"declared={sorted(lane.value for lane in set(required_lanes))!r}, "
            f"used={sorted(lane.value for lane in required_claim_lanes)!r}"
        )

    if parser.errors:
        raise ManifestValidationError(source, parser.errors)
    return PhaseAcceptanceManifest(
        schema_version=schema_version,
        issue=issue,
        phase=phase,
        integration_branch=integration_branch,
        required_lanes=required_lanes,
        claims=claims,
        source=source,
    )


def load_phase_acceptance(path: Path) -> PhaseAcceptanceManifest:
    try:
        config = OmegaConf.load(path)
        raw = OmegaConf.to_container(config, resolve=False)
    except Exception as exc:  # noqa: BLE001 - normalize parser failures for the CLI
        raise ManifestValidationError(
            path, [f"cannot load YAML: {type(exc).__name__}: {exc}"]
        ) from exc
    return parse_phase_acceptance(raw, source=path)


def phase_gate_errors(manifest: PhaseAcceptanceManifest) -> tuple[str, ...]:
    errors: list[str] = []
    for claim in manifest.claims:
        if not claim.required:
            continue
        if claim.status not in {ClaimStatus.VERIFIED, ClaimStatus.PROMOTED}:
            errors.append(
                f"{claim.claim_id}: required claim is {claim.status.value}, expected verified/promoted"
            )
    return tuple(errors)


def manifest_status_counts(manifest: PhaseAcceptanceManifest) -> dict[str, int]:
    counts = Counter(claim.status.value for claim in manifest.claims)
    return {status.value: counts.get(status.value, 0) for status in ClaimStatus}
