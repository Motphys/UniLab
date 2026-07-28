"""Reusable, fail-closed capture primitives for Issue #705 phase evidence.

The phase manifests intentionally keep the acceptance claims human-readable.
This module supplies the mechanically verifiable companion: a fixed command
matrix is captured from a clean commit, bound to input hashes and raw pytest
output, then revalidated against the current manifest.  It is deliberately
for validation/provenance only; it owns neither a simulator nor task runtime
behavior.
"""

from __future__ import annotations

import hashlib
import json
import platform
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Mapping, Sequence

from unilab.tools.phase_acceptance import ManifestValidationError, load_phase_acceptance

_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_PYTEST_COUNT_RE = re.compile(r"(?<![A-Za-z0-9_])(\d+) (passed|skipped|xfailed|xpassed)")


class PhaseEvidenceError(RuntimeError):
    """Raised when a phase capture cannot establish trustworthy evidence."""


@dataclass(frozen=True)
class PhaseEvidenceCommand:
    """One pre-registered command and its mandatory clean repetitions."""

    name: str
    lane: str
    argv: tuple[str, ...]
    required_test_ids: tuple[str, ...]
    repetitions: int

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("phase evidence command name must be non-empty")
        if self.lane not in {"A", "B", "C", "D"}:
            raise ValueError("phase evidence command lane must be A, B, C, or D")
        if not self.argv or any(not isinstance(item, str) or not item for item in self.argv):
            raise ValueError("phase evidence command argv must be non-empty strings")
        if not self.required_test_ids or any(
            not isinstance(item, str) or not item for item in self.required_test_ids
        ):
            raise ValueError("phase evidence command requires non-empty test IDs")
        if len(set(self.required_test_ids)) != len(self.required_test_ids):
            raise ValueError("phase evidence command test IDs must be unique")
        if isinstance(self.repetitions, bool) or not isinstance(self.repetitions, int):
            raise ValueError("phase evidence command repetitions must be an integer")
        if self.repetitions < 1:
            raise ValueError("phase evidence command repetitions must be positive")


@dataclass(frozen=True)
class PhaseEvidenceClaim:
    """One manifest claim mapped to its command and input provenance."""

    claim_id: str
    required_test_id: str
    command_name: str
    minimum_repetitions: int
    config_input: Path
    manifest_command: str

    def __post_init__(self) -> None:
        for name, value in (
            ("claim_id", self.claim_id),
            ("required_test_id", self.required_test_id),
            ("command_name", self.command_name),
            ("manifest_command", self.manifest_command),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"phase evidence claim {name} must be non-empty")
        if self.config_input.is_absolute() or ".." in self.config_input.parts:
            raise ValueError("phase evidence claim config input must be repository-relative")
        if (
            isinstance(self.minimum_repetitions, bool)
            or not isinstance(self.minimum_repetitions, int)
            or self.minimum_repetitions < 1
        ):
            raise ValueError("phase evidence claim repetitions must be positive")


@dataclass(frozen=True)
class PhaseEvidenceSpec:
    """Immutable capture contract for one Issue #705 phase."""

    issue: int
    phase: int
    artifact_kind: str
    manifest_path: Path
    required_lanes: tuple[str, ...]
    input_files: tuple[Path, ...]
    package_names: tuple[str, ...]
    commands: tuple[PhaseEvidenceCommand, ...]
    claims: tuple[PhaseEvidenceClaim, ...]
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.issue != 705:
            raise ValueError("Issue #705 phase evidence spec must use issue 705")
        if isinstance(self.phase, bool) or not isinstance(self.phase, int) or self.phase < 0:
            raise ValueError("phase evidence spec phase must be non-negative")
        if not isinstance(self.artifact_kind, str) or not self.artifact_kind:
            raise ValueError("phase evidence artifact kind must be non-empty")
        if self.manifest_path.is_absolute() or ".." in self.manifest_path.parts:
            raise ValueError("phase evidence manifest path must be repository-relative")
        if not self.required_lanes or any(
            lane not in {"A", "B", "C", "D"} for lane in self.required_lanes
        ):
            raise ValueError("phase evidence spec lanes must be unique A/B/C/D values")
        if len(set(self.required_lanes)) != len(self.required_lanes):
            raise ValueError("phase evidence spec lanes must be unique")
        if not self.input_files or any(
            path.is_absolute() or ".." in path.parts for path in self.input_files
        ):
            raise ValueError("phase evidence inputs must be repository-relative")
        if len(set(self.input_files)) != len(self.input_files):
            raise ValueError("phase evidence inputs must be unique")
        if len(set(self.package_names)) != len(self.package_names) or any(
            not isinstance(name, str) or not name for name in self.package_names
        ):
            raise ValueError("phase evidence packages must be unique non-empty strings")
        if not self.commands or not self.claims:
            raise ValueError("phase evidence spec requires commands and claims")
        command_names = tuple(command.name for command in self.commands)
        if len(set(command_names)) != len(command_names):
            raise ValueError("phase evidence command names must be unique")
        claim_ids = tuple(claim.claim_id for claim in self.claims)
        if len(set(claim_ids)) != len(claim_ids):
            raise ValueError("phase evidence claim IDs must be unique")
        command_by_name = {command.name: command for command in self.commands}
        expected_lanes = {command.lane for command in self.commands}
        if expected_lanes != set(self.required_lanes):
            raise ValueError("phase evidence commands must cover exactly the required lanes")
        for claim in self.claims:
            try:
                command = command_by_name[claim.command_name]
            except KeyError as exc:
                raise ValueError("phase evidence claim references an unknown command") from exc
            if claim.required_test_id not in command.required_test_ids:
                raise ValueError("phase evidence claim test is absent from its command")
            if claim.minimum_repetitions != command.repetitions:
                raise ValueError("phase evidence claim repetitions must match its command")


def sha256_file(path: Path) -> str:
    """Return the canonical hash representation used by acceptance manifests."""

    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def _registered_input_hashes(spec: PhaseEvidenceSpec, root: Path) -> dict[str, str]:
    return {path.as_posix(): sha256_file(root / path) for path in spec.input_files}


def _claim_config_hashes(spec: PhaseEvidenceSpec, root: Path) -> dict[str, str]:
    return {claim.claim_id: sha256_file(root / claim.config_input) for claim in spec.claims}


def _run_checked(argv: Sequence[str], *, root: Path, context: str) -> str:
    result = subprocess.run(argv, cwd=root, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise PhaseEvidenceError(
            f"{context} failed with exit {result.returncode}:\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result.stdout.strip()


def _git(root: Path, *args: str) -> str:
    return _run_checked(("git", *args), root=root, context=f"git {' '.join(args)}")


def _assert_clean_source(root: Path) -> None:
    dirty = _git(root, "status", "--porcelain")
    if dirty:
        raise PhaseEvidenceError(
            "phase evidence must be captured from a clean source tree; found:\n" + dirty
        )


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError as exc:
        raise PhaseEvidenceError(f"required package {name!r} is not installed") from exc


def _capture_environment(spec: PhaseEvidenceSpec) -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "python": sys.version,
        "packages": {name: _package_version(name) for name in spec.package_names},
    }


def _pytest_counts(output: str) -> dict[str, int]:
    counts = {"passed": 0, "skipped": 0, "xfailed": 0, "xpassed": 0}
    for raw_count, category in _PYTEST_COUNT_RE.findall(output):
        counts[category] += int(raw_count)
    return counts


def _run_evidence_command(
    command: PhaseEvidenceCommand, *, root: Path, repetition: int
) -> dict[str, Any]:
    started = time.perf_counter()
    result = subprocess.run(
        command.argv,
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
        timeout=600,
    )
    duration_sec = time.perf_counter() - started
    combined = f"{result.stdout}\n{result.stderr}"
    counts = _pytest_counts(combined)
    errors: list[str] = []
    if result.returncode != 0:
        errors.append(f"exit_code={result.returncode}")
    if counts["passed"] <= 0:
        errors.append("pytest reported no passed tests")
    for category in ("skipped", "xfailed", "xpassed"):
        if counts[category] != 0:
            errors.append(f"pytest reported {counts[category]} {category}")
    missing_nodes = [test_id for test_id in command.required_test_ids if test_id not in combined]
    if missing_nodes:
        errors.append(f"required test IDs absent from pytest output: {missing_nodes!r}")
    if errors:
        raise PhaseEvidenceError(
            f"{command.name} did not meet evidence requirements ({'; '.join(errors)}):\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return {
        "name": f"{command.name}#{repetition}",
        "series": command.name,
        "lane": command.lane,
        "repetition": repetition,
        "argv": list(command.argv),
        "required_test_ids": list(command.required_test_ids),
        "exit_code": int(result.returncode),
        "duration_sec": duration_sec,
        "pytest": counts,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def capture_phase_evidence(spec: PhaseEvidenceSpec, root: Path) -> dict[str, Any]:
    """Capture every pre-registered command from one clean source commit."""

    root = root.resolve()
    missing = [path.as_posix() for path in spec.input_files if not (root / path).is_file()]
    if missing:
        raise PhaseEvidenceError("phase evidence is missing required inputs: " + ", ".join(missing))
    _assert_clean_source(root)
    source_commit = _git(root, "rev-parse", "HEAD")
    if not _COMMIT_RE.fullmatch(source_commit):
        raise PhaseEvidenceError(f"git returned an invalid commit SHA: {source_commit!r}")
    input_hashes = _registered_input_hashes(spec, root)
    config_hashes = _claim_config_hashes(spec, root)
    environment = _capture_environment(spec)
    commands = [
        _run_evidence_command(command, root=root, repetition=repetition)
        for command in spec.commands
        for repetition in range(1, command.repetitions + 1)
    ]
    _assert_clean_source(root)
    if _registered_input_hashes(spec, root) != input_hashes:
        raise PhaseEvidenceError(
            "phase evidence commands changed a registered input during capture"
        )
    return {
        "schema_version": spec.schema_version,
        "kind": spec.artifact_kind,
        "issue": spec.issue,
        "phase": spec.phase,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "commit_sha": source_commit,
            "branch": _git(root, "rev-parse", "--abbrev-ref", "HEAD"),
            "tree_clean": True,
        },
        "inputs": {"files": input_hashes, "claim_config_hashes": config_hashes},
        "environment": environment,
        "claims": [
            {
                "claim_id": claim.claim_id,
                "required_test_id": claim.required_test_id,
                "command": claim.command_name,
                "minimum_repetitions": claim.minimum_repetitions,
                "config_input": claim.config_input.as_posix(),
            }
            for claim in spec.claims
        ],
        "commands": commands,
    }


def write_phase_evidence(report: Mapping[str, Any], output: Path) -> None:
    """Write an auditable JSON artifact only after a successful capture."""

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dict(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_phase_evidence(path: Path) -> dict[str, Any]:
    """Load one evidence artifact with a domain-specific error on malformed I/O."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PhaseEvidenceError(f"cannot load phase evidence {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise PhaseEvidenceError(f"phase evidence {path} must contain a JSON object")
    return payload


def _mapping(value: object, name: str, errors: list[str]) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    errors.append(f"{name}: expected object")
    return {}


def _string(value: object, name: str, errors: list[str]) -> str:
    if isinstance(value, str) and value:
        return value
    errors.append(f"{name}: expected non-empty string")
    return ""


def _integer(value: object, name: str, errors: list[str]) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    errors.append(f"{name}: expected integer")
    return -1


def _keys_match(
    mapping: Mapping[str, Any], *, expected: set[str], name: str, errors: list[str]
) -> None:
    actual = set(mapping)
    if actual != expected:
        errors.append(f"{name}: keys do not exactly match the registered schema")


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


def _validate_inputs(
    spec: PhaseEvidenceSpec,
    report: Mapping[str, Any],
    *,
    root: Path,
    errors: list[str],
) -> None:
    inputs = _mapping(report.get("inputs"), "inputs", errors)
    _keys_match(
        inputs,
        expected={"files", "claim_config_hashes"},
        name="inputs",
        errors=errors,
    )
    expected_paths = {path.as_posix() for path in spec.input_files}
    raw_files = inputs.get("files")
    if not isinstance(raw_files, dict):
        errors.append("inputs.files: expected object")
        raw_files = {}
    if set(raw_files) != expected_paths:
        errors.append("inputs.files: paths do not exactly match registered inputs")
    for path in spec.input_files:
        key = path.as_posix()
        observed = _string(raw_files.get(key), f"inputs.files.{key}", errors)
        current = root / path
        if not current.is_file():
            errors.append(f"inputs.files.{key}: current input is missing")
        elif observed and not _SHA256_RE.fullmatch(observed):
            errors.append(f"inputs.files.{key}: expected sha256 hash")
        elif observed and observed != sha256_file(current):
            errors.append(f"inputs.files.{key}: does not match current input")

    raw_config_hashes = inputs.get("claim_config_hashes")
    if not isinstance(raw_config_hashes, dict):
        errors.append("inputs.claim_config_hashes: expected object")
        raw_config_hashes = {}
    expected_claims = {claim.claim_id for claim in spec.claims}
    if set(raw_config_hashes) != expected_claims:
        errors.append(
            "inputs.claim_config_hashes: claim IDs do not exactly match registered claims"
        )
    for claim in spec.claims:
        key = claim.claim_id
        observed = _string(raw_config_hashes.get(key), f"inputs.claim_config_hashes.{key}", errors)
        current = root / claim.config_input
        if not current.is_file():
            errors.append(f"inputs.claim_config_hashes.{key}: current input is missing")
        elif observed and not _SHA256_RE.fullmatch(observed):
            errors.append(f"inputs.claim_config_hashes.{key}: expected sha256 hash")
        elif observed and observed != sha256_file(current):
            errors.append(f"inputs.claim_config_hashes.{key}: does not match current input")


def _validate_environment(
    spec: PhaseEvidenceSpec, report: Mapping[str, Any], *, errors: list[str]
) -> None:
    environment = _mapping(report.get("environment"), "environment", errors)
    _keys_match(
        environment,
        expected={"platform", "python", "packages"},
        name="environment",
        errors=errors,
    )
    _string(environment.get("platform"), "environment.platform", errors)
    _string(environment.get("python"), "environment.python", errors)
    packages = _mapping(environment.get("packages"), "environment.packages", errors)
    if set(packages) != set(spec.package_names):
        errors.append("environment.packages: keys do not exactly match registered packages")
    for package_name in spec.package_names:
        _string(packages.get(package_name), f"environment.packages.{package_name}", errors)


def _validate_commands(
    spec: PhaseEvidenceSpec, report: Mapping[str, Any], *, errors: list[str]
) -> dict[str, list[dict[str, Any]]]:
    raw_commands = report.get("commands")
    if not isinstance(raw_commands, list) or not raw_commands:
        errors.append("commands: expected non-empty list")
        raw_commands = []
    command_by_name = {command.name: command for command in spec.commands}
    by_series: dict[str, list[dict[str, Any]]] = {}
    names: set[str] = set()
    seen_lanes: set[str] = set()
    expected_keys = {
        "name",
        "series",
        "lane",
        "repetition",
        "argv",
        "required_test_ids",
        "exit_code",
        "duration_sec",
        "pytest",
        "stdout",
        "stderr",
    }
    for index, raw_command in enumerate(raw_commands):
        command = _mapping(raw_command, f"commands[{index}]", errors)
        _keys_match(command, expected=expected_keys, name=f"commands[{index}]", errors=errors)
        name = _string(command.get("name"), f"commands[{index}].name", errors)
        if name:
            if name in names:
                errors.append(f"commands[{index}].name: duplicate {name!r}")
            names.add(name)
        series = _string(command.get("series"), f"commands[{index}].series", errors)
        repetition = _integer(command.get("repetition"), f"commands[{index}].repetition", errors)
        if repetition < 1:
            errors.append(f"commands[{index}].repetition: expected >= 1")
        if series and name and name != f"{series}#{repetition}":
            errors.append(f"commands[{index}].name: must equal its series/repetition key")
        expected = command_by_name.get(series)
        if expected is None:
            errors.append(f"commands[{index}].series: unknown series {series!r}")
        else:
            if command.get("lane") != expected.lane:
                errors.append(f"commands[{index}].lane: does not match {series!r}")
            if command.get("argv") != list(expected.argv):
                errors.append(
                    f"commands[{index}].argv: does not match registered command {series!r}"
                )
            if command.get("required_test_ids") != list(expected.required_test_ids):
                errors.append(f"commands[{index}].required_test_ids: does not match {series!r}")
            by_series.setdefault(series, []).append(command)
        lane = _string(command.get("lane"), f"commands[{index}].lane", errors)
        if lane:
            seen_lanes.add(lane)
        if _integer(command.get("exit_code"), f"commands[{index}].exit_code", errors) != 0:
            errors.append(f"commands[{index}].exit_code: expected 0")
        duration = command.get("duration_sec")
        if (
            isinstance(duration, bool)
            or not isinstance(duration, (int, float))
            or float(duration) <= 0.0
        ):
            errors.append(f"commands[{index}].duration_sec: expected positive number")
        pytest_counts = _mapping(command.get("pytest"), f"commands[{index}].pytest", errors)
        _keys_match(
            pytest_counts,
            expected={"passed", "skipped", "xfailed", "xpassed"},
            name=f"commands[{index}].pytest",
            errors=errors,
        )
        if _integer(pytest_counts.get("passed"), f"commands[{index}].pytest.passed", errors) <= 0:
            errors.append(f"commands[{index}].pytest.passed: expected > 0")
        for category in ("skipped", "xfailed", "xpassed"):
            if (
                _integer(
                    pytest_counts.get(category), f"commands[{index}].pytest.{category}", errors
                )
                != 0
            ):
                errors.append(f"commands[{index}].pytest.{category}: expected 0")
        required_test_ids = command.get("required_test_ids")
        stdout = _string(command.get("stdout"), f"commands[{index}].stdout", errors)
        if not isinstance(required_test_ids, list) or not all(
            isinstance(test_id, str) for test_id in required_test_ids
        ):
            errors.append(f"commands[{index}].required_test_ids: expected string list")
        else:
            for test_id in required_test_ids:
                if test_id not in stdout:
                    errors.append(
                        f"commands[{index}].stdout: required test {test_id!r} is not present"
                    )
        if not isinstance(command.get("stderr"), str):
            errors.append(f"commands[{index}].stderr: expected string")
    if seen_lanes != set(spec.required_lanes):
        errors.append(
            "commands: expected exactly lanes "
            f"{sorted(spec.required_lanes)!r}, got {sorted(seen_lanes)!r}"
        )
    if set(by_series) != set(command_by_name):
        errors.append("commands: series do not exactly match registered commands")
    for expected in spec.commands:
        matching = by_series.get(expected.name, [])
        repetitions = {
            _integer(command.get("repetition"), "commands.repetition", errors)
            for command in matching
        }
        if repetitions != set(range(1, expected.repetitions + 1)):
            errors.append(
                f"commands[{expected.name}]: expected repetitions 1..{expected.repetitions}"
            )
    return by_series


def _validate_claims(
    spec: PhaseEvidenceSpec,
    report: Mapping[str, Any],
    *,
    by_series: Mapping[str, list[dict[str, Any]]],
    errors: list[str],
) -> None:
    raw_claims = report.get("claims")
    if not isinstance(raw_claims, list):
        errors.append("claims: expected list")
        raw_claims = []
    observed: dict[str, tuple[str, str, int, str]] = {}
    for index, raw_claim in enumerate(raw_claims):
        claim = _mapping(raw_claim, f"claims[{index}]", errors)
        _keys_match(
            claim,
            expected={
                "claim_id",
                "required_test_id",
                "command",
                "minimum_repetitions",
                "config_input",
            },
            name=f"claims[{index}]",
            errors=errors,
        )
        claim_id = _string(claim.get("claim_id"), f"claims[{index}].claim_id", errors)
        test_id = _string(
            claim.get("required_test_id"), f"claims[{index}].required_test_id", errors
        )
        command_name = _string(claim.get("command"), f"claims[{index}].command", errors)
        repetitions = _integer(
            claim.get("minimum_repetitions"), f"claims[{index}].minimum_repetitions", errors
        )
        config_input = _string(claim.get("config_input"), f"claims[{index}].config_input", errors)
        if claim_id:
            if claim_id in observed:
                errors.append(f"claims[{index}].claim_id: duplicate {claim_id!r}")
            observed[claim_id] = (test_id, command_name, repetitions, config_input)
    expected_claims = {claim.claim_id for claim in spec.claims}
    if set(observed) != expected_claims:
        errors.append("claims: claim IDs do not exactly match registered claims")
    for expected in spec.claims:
        test_id, command_name, repetitions, config_input = observed.get(
            expected.claim_id, ("", "", -1, "")
        )
        if test_id != expected.required_test_id:
            errors.append(f"claims[{expected.claim_id}]: required test mapping does not match")
        if command_name != expected.command_name:
            errors.append(f"claims[{expected.claim_id}]: command mapping does not match")
        if repetitions != expected.minimum_repetitions:
            errors.append(f"claims[{expected.claim_id}]: minimum repetitions do not match")
        if config_input != expected.config_input.as_posix():
            errors.append(f"claims[{expected.claim_id}]: config input does not match")
        matching = by_series.get(command_name, [])
        if len(matching) < expected.minimum_repetitions:
            errors.append(f"claims[{expected.claim_id}]: insufficient successful repetitions")
        for command in matching:
            test_ids = command.get("required_test_ids")
            if not isinstance(test_ids, list) or expected.required_test_id not in test_ids:
                errors.append(f"claims[{expected.claim_id}]: command does not declare its test")


def _validate_manifest_contract(spec: PhaseEvidenceSpec, root: Path, *, errors: list[str]) -> None:
    try:
        manifest = load_phase_acceptance(root / spec.manifest_path)
    except ManifestValidationError as exc:
        errors.extend(f"manifest: {error}" for error in exc.errors)
        return
    if manifest.issue != spec.issue or manifest.phase != spec.phase:
        errors.append(f"manifest: expected issue/phase {spec.issue}/{spec.phase}")
    if tuple(lane.value for lane in manifest.required_lanes) != spec.required_lanes:
        errors.append("manifest: required lanes do not match registered evidence spec")
    manifest_claims = {claim.claim_id: claim for claim in manifest.claims}
    if set(manifest_claims) != {claim.claim_id for claim in spec.claims}:
        errors.append("manifest: claim IDs do not exactly match registered claims")
        return
    command_by_name = {command.name: command for command in spec.commands}
    for expected in spec.claims:
        claim = manifest_claims[expected.claim_id]
        command = command_by_name[expected.command_name]
        if claim.required_test_ids != (expected.required_test_id,):
            errors.append(f"manifest[{expected.claim_id}]: required test mapping does not match")
        if claim.acceptance.repetitions != expected.minimum_repetitions:
            errors.append(f"manifest[{expected.claim_id}]: repetitions do not match")
        if claim.lane.value != command.lane:
            errors.append(f"manifest[{expected.claim_id}]: lane does not match")
        if claim.commands != (expected.manifest_command,):
            errors.append(f"manifest[{expected.claim_id}]: command does not match")


def validate_phase_evidence(
    spec: PhaseEvidenceSpec, report: Mapping[str, Any], *, root: Path
) -> tuple[str, ...]:
    """Return all artifact/manifest/provenance errors without trusting PASS JSON."""

    root = root.resolve()
    errors: list[str] = []
    expected_top_level = {
        "schema_version",
        "kind",
        "issue",
        "phase",
        "generated_at_utc",
        "source",
        "inputs",
        "environment",
        "claims",
        "commands",
    }
    _keys_match(report, expected=expected_top_level, name="artifact", errors=errors)
    if report.get("schema_version") != spec.schema_version:
        errors.append(f"schema_version: expected {spec.schema_version}")
    if report.get("kind") != spec.artifact_kind:
        errors.append(f"kind: expected {spec.artifact_kind!r}")
    if report.get("issue") != spec.issue or report.get("phase") != spec.phase:
        errors.append(f"issue/phase: expected {spec.issue}/{spec.phase}")
    _string(report.get("generated_at_utc"), "generated_at_utc", errors)

    source = _mapping(report.get("source"), "source", errors)
    _keys_match(
        source,
        expected={"commit_sha", "branch", "tree_clean"},
        name="source",
        errors=errors,
    )
    commit = _string(source.get("commit_sha"), "source.commit_sha", errors)
    if commit and not _COMMIT_RE.fullmatch(commit):
        errors.append("source.commit_sha: expected full 40-character SHA")
    elif commit and not _git_is_ancestor(root, commit):
        errors.append("source.commit_sha: must be an ancestor of the current gate commit")
    _string(source.get("branch"), "source.branch", errors)
    if source.get("tree_clean") is not True:
        errors.append("source.tree_clean: must be true")

    _validate_inputs(spec, report, root=root, errors=errors)
    _validate_environment(spec, report, errors=errors)
    commands = _validate_commands(spec, report, errors=errors)
    _validate_claims(spec, report, by_series=commands, errors=errors)
    _validate_manifest_contract(spec, root, errors=errors)
    return tuple(errors)


__all__ = [
    "PhaseEvidenceClaim",
    "PhaseEvidenceCommand",
    "PhaseEvidenceError",
    "PhaseEvidenceSpec",
    "capture_phase_evidence",
    "load_phase_evidence",
    "sha256_file",
    "validate_phase_evidence",
    "write_phase_evidence",
]
