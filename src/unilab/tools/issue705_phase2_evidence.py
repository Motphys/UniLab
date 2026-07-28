"""Capture and fail-closed validation for Issue #705 Phase 2 gate evidence.

The capture path runs the production A/C lane commands and records their raw
pytest output.  The validator deliberately treats the JSON as untrusted: it
checks provenance, current input hashes, command/test mappings, and mandatory
skip/xfail counts before a manifest can cite the artifact as PASS evidence.
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

from unilab.base.backend.mjwarp.dependencies import load_mjwarp_dependencies

ISSUE = 705
PHASE = 2
SCHEMA_VERSION = 1
ARTIFACT_KIND = "issue705-phase2-gate-v1"
OWNER_YAML = Path("conf/ppo/task/g1_walk_flat/mjwarp.yaml")
UV_LOCK = Path("uv.lock")

PHASE2_REQUIRED_TEST_IDS: dict[str, str] = {
    "P2-BACKEND-IDENTITY": (
        "tests/base/test_mjwarp_identity.py::test_mjwarp_identity_is_independent_from_mujoco"
    ),
    "P2-GPU-CORRECTNESS": "tests/base/test_mjwarp_backend.py::test_real_cuda_init_reset_step",
    "P2-RESET-ISOLATION": "tests/base/test_mjwarp_backend.py::test_selected_row_reset_isolated",
    "P2-TRAJECTORY-DIFFERENTIAL": (
        "tests/base/test_mjwarp_differential.py::test_g1_short_trajectory_matches_mujoco"
    ),
    "P2-DR-OWNER-SEMANTICS": (
        "tests/dr/test_mjwarp_g1_dr.py::test_g1_kp_kd_owner_semantics_have_physics_effect_or_are_disabled"
    ),
    "P2-TRANSFER-ACCOUNTING": (
        "tests/base/test_mjwarp_transfers.py::test_host_profile_transfer_count_matches_bound_plan"
    ),
    "P2-UNSUPPORTED-FAIL-CLOSED": (
        "tests/base/test_mjwarp_capabilities.py::test_unsupported_matrix_fails_before_step"
    ),
    "P2-TRAIN-LIVENESS": (
        "tests/integration/test_mjwarp_train_smoke.py::test_g1_one_iteration_uses_production_mjwarp"
    ),
}

PHASE2_MIN_REPETITIONS: dict[str, int] = {
    "P2-BACKEND-IDENTITY": 1,
    "P2-GPU-CORRECTNESS": 2,
    "P2-RESET-ISOLATION": 3,
    "P2-TRAJECTORY-DIFFERENTIAL": 3,
    "P2-DR-OWNER-SEMANTICS": 3,
    "P2-TRANSFER-ACCOUNTING": 2,
    "P2-UNSUPPORTED-FAIL-CLOSED": 1,
    "P2-TRAIN-LIVENESS": 1,
}


class Phase2EvidenceError(RuntimeError):
    """Raised when an evidence capture cannot establish a trustworthy PASS."""


@dataclass(frozen=True)
class Phase2EvidenceCommand:
    """One pre-registered command contributing evidence to Phase 2."""

    name: str
    lane: str
    argv: tuple[str, ...]
    required_test_ids: tuple[str, ...]
    repetitions: int


PHASE2_COMMANDS = (
    Phase2EvidenceCommand(
        name="lane_a_identity",
        lane="A",
        argv=(
            "uv",
            "run",
            "pytest",
            "tests/base/test_backend_imports.py",
            PHASE2_REQUIRED_TEST_IDS["P2-BACKEND-IDENTITY"],
            "-v",
            "-rsxX",
        ),
        required_test_ids=(PHASE2_REQUIRED_TEST_IDS["P2-BACKEND-IDENTITY"],),
        repetitions=1,
    ),
    Phase2EvidenceCommand(
        name="lane_c_production_cuda",
        lane="C",
        argv=(
            "uv",
            "run",
            "--extra",
            "mjwarp",
            "pytest",
            "-m",
            "slow",
            PHASE2_REQUIRED_TEST_IDS["P2-GPU-CORRECTNESS"],
            PHASE2_REQUIRED_TEST_IDS["P2-RESET-ISOLATION"],
            PHASE2_REQUIRED_TEST_IDS["P2-TRAJECTORY-DIFFERENTIAL"],
            PHASE2_REQUIRED_TEST_IDS["P2-TRANSFER-ACCOUNTING"],
            PHASE2_REQUIRED_TEST_IDS["P2-UNSUPPORTED-FAIL-CLOSED"],
            PHASE2_REQUIRED_TEST_IDS["P2-TRAIN-LIVENESS"],
            "-v",
            "-rsxX",
        ),
        required_test_ids=tuple(
            PHASE2_REQUIRED_TEST_IDS[claim_id]
            for claim_id in (
                "P2-GPU-CORRECTNESS",
                "P2-RESET-ISOLATION",
                "P2-TRAJECTORY-DIFFERENTIAL",
                "P2-TRANSFER-ACCOUNTING",
                "P2-UNSUPPORTED-FAIL-CLOSED",
                "P2-TRAIN-LIVENESS",
            )
        ),
        repetitions=3,
    ),
    Phase2EvidenceCommand(
        name="lane_c_dr_owner_semantics",
        lane="C",
        argv=(
            "uv",
            "run",
            "--extra",
            "mjwarp",
            "pytest",
            PHASE2_REQUIRED_TEST_IDS["P2-DR-OWNER-SEMANTICS"],
            "-v",
            "-rsxX",
        ),
        required_test_ids=(PHASE2_REQUIRED_TEST_IDS["P2-DR-OWNER-SEMANTICS"],),
        repetitions=3,
    ),
)

_PYTEST_COUNT_RE = re.compile(r"(?<![A-Za-z0-9_])(\d+) (passed|skipped|xfailed|xpassed)")
_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


def sha256_file(path: Path) -> str:
    """Return a stable repository-input hash with the manifest's prefix form."""
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return f"sha256:{digest}"


def _run_checked(argv: Sequence[str], *, root: Path, context: str) -> str:
    result = subprocess.run(argv, cwd=root, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise Phase2EvidenceError(
            f"{context} failed with exit {result.returncode}:\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result.stdout.strip()


def _git(root: Path, *args: str) -> str:
    return _run_checked(("git", *args), root=root, context=f"git {' '.join(args)}")


def _assert_clean_source(root: Path) -> None:
    dirty = _git(root, "status", "--porcelain")
    if dirty:
        raise Phase2EvidenceError(
            f"Phase 2 evidence must be captured from a clean source tree; found:\n{dirty}"
        )


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError as exc:
        raise Phase2EvidenceError(f"required package {name!r} is not installed") from exc


def _cuda_environment(root: Path) -> dict[str, Any]:
    dependencies = load_mjwarp_dependencies()
    device = dependencies.warp.get_device()
    if not bool(device.is_cuda):
        raise Phase2EvidenceError("Phase 2 capture requires an active CUDA Warp device")
    nvidia_smi = _run_checked(
        (
            "nvidia-smi",
            "--query-gpu=name,uuid,driver_version,memory.total",
            "--format=csv,noheader",
        ),
        root=root,
        context="nvidia-smi",
    )
    return {
        "platform": platform.platform(),
        "python": sys.version,
        "warp_device": str(device),
        "nvidia_smi": nvidia_smi.splitlines(),
        "packages": {
            "mujoco": _package_version("mujoco"),
            "mujoco-warp": _package_version("mujoco-warp"),
            "warp-lang": _package_version("warp-lang"),
            "torch": _package_version("torch"),
            "rsl-rl-lib": _package_version("rsl-rl-lib"),
        },
    }


def _pytest_counts(output: str) -> dict[str, int]:
    counts = {"passed": 0, "skipped": 0, "xfailed": 0, "xpassed": 0}
    for raw_count, category in _PYTEST_COUNT_RE.findall(output):
        counts[category] += int(raw_count)
    return counts


def _run_evidence_command(
    command: Phase2EvidenceCommand, *, root: Path, repetition: int
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
        raise Phase2EvidenceError(
            f"{command.name} did not meet Phase 2 evidence requirements ({'; '.join(errors)}):\n"
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


def capture_phase2_evidence(root: Path) -> dict[str, Any]:
    """Run all required A/C evidence commands and return an immutable report payload."""
    root = root.resolve()
    owner_yaml = root / OWNER_YAML
    lock_path = root / UV_LOCK
    if not owner_yaml.is_file() or not lock_path.is_file():
        raise Phase2EvidenceError(
            "Phase 2 capture requires owner YAML and uv.lock at repository root"
        )
    _assert_clean_source(root)
    source_commit = _git(root, "rev-parse", "HEAD")
    if not _COMMIT_RE.fullmatch(source_commit):
        raise Phase2EvidenceError(f"git returned an invalid commit SHA: {source_commit!r}")

    environment = _cuda_environment(root)
    commands = [
        _run_evidence_command(command, root=root, repetition=repetition)
        for command in PHASE2_COMMANDS
        for repetition in range(1, command.repetitions + 1)
    ]
    command_by_test = {
        test_id: command.name
        for command in PHASE2_COMMANDS
        for test_id in command.required_test_ids
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": ARTIFACT_KIND,
        "issue": ISSUE,
        "phase": PHASE,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "commit_sha": source_commit,
            "branch": _git(root, "rev-parse", "--abbrev-ref", "HEAD"),
            "tree_clean": True,
        },
        "inputs": {
            "owner_yaml": OWNER_YAML.as_posix(),
            "owner_yaml_sha256": sha256_file(owner_yaml),
            "uv_lock": UV_LOCK.as_posix(),
            "uv_lock_sha256": sha256_file(lock_path),
        },
        "environment": environment,
        "claims": [
            {
                "claim_id": claim_id,
                "required_test_id": test_id,
                "command": command_by_test[test_id],
                "minimum_repetitions": PHASE2_MIN_REPETITIONS[claim_id],
            }
            for claim_id, test_id in PHASE2_REQUIRED_TEST_IDS.items()
        ],
        "commands": commands,
    }


def write_phase2_evidence(report: Mapping[str, Any], output: Path) -> None:
    """Write one human-auditable JSON report after a successful capture."""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dict(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_phase2_evidence(path: Path) -> dict[str, Any]:
    """Load the JSON report, preserving a clear error boundary for test callers."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Phase2EvidenceError(f"cannot load Phase 2 evidence {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise Phase2EvidenceError(f"Phase 2 evidence {path} must contain a JSON object")
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


def _int(value: object, name: str, errors: list[str]) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    errors.append(f"{name}: expected integer")
    return -1


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


def validate_phase2_evidence(report: Mapping[str, Any], *, root: Path) -> tuple[str, ...]:
    """Return every provenance/coverage error instead of trusting a PASS JSON."""
    root = root.resolve()
    errors: list[str] = []
    if report.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version: expected {SCHEMA_VERSION}")
    if report.get("kind") != ARTIFACT_KIND:
        errors.append(f"kind: expected {ARTIFACT_KIND!r}")
    if report.get("issue") != ISSUE or report.get("phase") != PHASE:
        errors.append(f"issue/phase: expected {ISSUE}/{PHASE}")

    source = _mapping(report.get("source"), "source", errors)
    commit = _string(source.get("commit_sha"), "source.commit_sha", errors)
    if commit and not _COMMIT_RE.fullmatch(commit):
        errors.append("source.commit_sha: expected full 40-character SHA")
    elif commit and not _git_is_ancestor(root, commit):
        errors.append("source.commit_sha: must be an ancestor of the current gate commit")
    if source.get("tree_clean") is not True:
        errors.append("source.tree_clean: must be true")

    inputs = _mapping(report.get("inputs"), "inputs", errors)
    if inputs.get("owner_yaml") != OWNER_YAML.as_posix():
        errors.append("inputs.owner_yaml: unexpected owner YAML")
    if inputs.get("uv_lock") != UV_LOCK.as_posix():
        errors.append("inputs.uv_lock: unexpected lock path")
    for field, path in (
        ("owner_yaml_sha256", root / OWNER_YAML),
        ("uv_lock_sha256", root / UV_LOCK),
    ):
        observed = _string(inputs.get(field), f"inputs.{field}", errors)
        if observed and not _SHA256_RE.fullmatch(observed):
            errors.append(f"inputs.{field}: expected sha256 hash")
        elif observed and path.is_file() and observed != sha256_file(path):
            errors.append(f"inputs.{field}: does not match current {path.name}")

    environment = _mapping(report.get("environment"), "environment", errors)
    nvidia_smi = environment.get("nvidia_smi")
    if not isinstance(nvidia_smi, list) or not nvidia_smi:
        errors.append("environment.nvidia_smi: expected non-empty GPU provenance")
    warp_device = _string(environment.get("warp_device"), "environment.warp_device", errors)
    if warp_device and "cuda" not in warp_device.lower():
        errors.append("environment.warp_device: expected CUDA device")

    raw_commands = report.get("commands")
    if not isinstance(raw_commands, list) or not raw_commands:
        errors.append("commands: expected non-empty list")
        raw_commands = []
    command_by_name: dict[str, dict[str, Any]] = {}
    commands_by_series: dict[str, list[dict[str, Any]]] = {}
    seen_lanes: set[str] = set()
    for index, raw_command in enumerate(raw_commands):
        command = _mapping(raw_command, f"commands[{index}]", errors)
        name = _string(command.get("name"), f"commands[{index}].name", errors)
        if name:
            if name in command_by_name:
                errors.append(f"commands[{index}].name: duplicate {name!r}")
            command_by_name[name] = command
        series = _string(command.get("series"), f"commands[{index}].series", errors)
        if series:
            commands_by_series.setdefault(series, []).append(command)
        lane = _string(command.get("lane"), f"commands[{index}].lane", errors)
        if lane:
            seen_lanes.add(lane)
        if _int(command.get("exit_code"), f"commands[{index}].exit_code", errors) != 0:
            errors.append(f"commands[{index}].exit_code: expected 0")
        pytest_counts = _mapping(command.get("pytest"), f"commands[{index}].pytest", errors)
        if _int(pytest_counts.get("passed"), f"commands[{index}].pytest.passed", errors) <= 0:
            errors.append(f"commands[{index}].pytest.passed: expected > 0")
        for category in ("skipped", "xfailed", "xpassed"):
            if (
                _int(
                    pytest_counts.get(category),
                    f"commands[{index}].pytest.{category}",
                    errors,
                )
                != 0
            ):
                errors.append(f"commands[{index}].pytest.{category}: expected 0")
        required_test_ids = command.get("required_test_ids")
        stdout = _string(command.get("stdout"), f"commands[{index}].stdout", errors)
        if not isinstance(required_test_ids, list) or not all(
            isinstance(item, str) for item in required_test_ids
        ):
            errors.append(f"commands[{index}].required_test_ids: expected string list")
        else:
            for test_id in required_test_ids:
                if test_id not in stdout:
                    errors.append(
                        f"commands[{index}].stdout: required test {test_id!r} is not present"
                    )
    if seen_lanes != {"A", "C"}:
        errors.append(f"commands: expected exactly lanes A/C, got {sorted(seen_lanes)!r}")
    for expected_command in PHASE2_COMMANDS:
        matching = commands_by_series.get(expected_command.name, [])
        repetitions = {
            _int(command.get("repetition"), f"commands[{index}].repetition", errors)
            for index, command in enumerate(matching)
        }
        if repetitions != set(range(1, expected_command.repetitions + 1)):
            errors.append(
                f"commands[{expected_command.name}]: expected repetitions "
                f"1..{expected_command.repetitions}"
            )

    raw_claims = report.get("claims")
    if not isinstance(raw_claims, list):
        errors.append("claims: expected list")
        raw_claims = []
    observed_claims: dict[str, tuple[str, str, int]] = {}
    for index, raw_claim in enumerate(raw_claims):
        claim = _mapping(raw_claim, f"claims[{index}]", errors)
        claim_id = _string(claim.get("claim_id"), f"claims[{index}].claim_id", errors)
        test_id = _string(
            claim.get("required_test_id"), f"claims[{index}].required_test_id", errors
        )
        command_name = _string(claim.get("command"), f"claims[{index}].command", errors)
        minimum_repetitions = _int(
            claim.get("minimum_repetitions"), f"claims[{index}].minimum_repetitions", errors
        )
        if claim_id:
            if claim_id in observed_claims:
                errors.append(f"claims[{index}].claim_id: duplicate {claim_id!r}")
            observed_claims[claim_id] = (test_id, command_name, minimum_repetitions)
    if set(observed_claims) != set(PHASE2_REQUIRED_TEST_IDS):
        errors.append("claims: claim IDs do not exactly match Phase 2 required claims")
    for claim_id, expected_test_id in PHASE2_REQUIRED_TEST_IDS.items():
        test_id, command_name, minimum_repetitions = observed_claims.get(claim_id, ("", "", -1))
        if test_id != expected_test_id:
            errors.append(f"claims[{claim_id}]: required test mapping does not match")
        if minimum_repetitions != PHASE2_MIN_REPETITIONS[claim_id]:
            errors.append(f"claims[{claim_id}]: minimum repetitions do not match Phase 2 manifest")
        mapped_commands = commands_by_series.get(command_name, [])
        if not mapped_commands:
            errors.append(f"claims[{claim_id}]: references unknown command {command_name!r}")
            continue
        if len(mapped_commands) < minimum_repetitions:
            errors.append(f"claims[{claim_id}]: insufficient successful repetitions")
        for mapped_command in mapped_commands:
            command_ids = mapped_command.get("required_test_ids")
            if not isinstance(command_ids, list) or expected_test_id not in command_ids:
                errors.append(f"claims[{claim_id}]: command does not declare its required test")
    return tuple(errors)


__all__ = [
    "ARTIFACT_KIND",
    "ISSUE",
    "OWNER_YAML",
    "PHASE",
    "PHASE2_COMMANDS",
    "PHASE2_REQUIRED_TEST_IDS",
    "Phase2EvidenceError",
    "capture_phase2_evidence",
    "load_phase2_evidence",
    "sha256_file",
    "validate_phase2_evidence",
    "write_phase2_evidence",
]
