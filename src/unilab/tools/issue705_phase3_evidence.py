"""Capture and validate fail-closed evidence for the Issue #705 Phase 3 gate.

The artifact is deliberately more than a list of green test names.  It binds
every required manager-pilot claim to an immutable command/test/repetition
mapping, raw pytest output, source provenance, current input hashes, and the
CUDA environment used by the cross-backend lane.  The validator treats the
JSON as untrusted data and checks the current manifest independently.
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
from unilab.tools.phase_acceptance import ManifestValidationError, load_phase_acceptance

ISSUE = 705
PHASE = 3
SCHEMA_VERSION = 1
ARTIFACT_KIND = "issue705-phase3-gate-v1"
UV_LOCK = Path("uv.lock")
MANIFEST_PATH = Path("tests/acceptance/issue_705/manifests/phase_3.yaml")
MUJOCO_OWNER = Path("conf/ppo/task/g1_walk_flat/mujoco.yaml")
MJWARP_OWNER = Path("conf/ppo/task/g1_walk_flat/mjwarp.yaml")

PHASE3_REQUIRED_TEST_IDS: dict[str, str] = {
    "P3-TASK-COMPILER": "tests/manager/test_task_compiler.py::test_compiler_binds_and_freezes_complete_plan",
    "P3-LIFECYCLE-PARITY": "tests/manager/test_managed_lifecycle.py::test_terminal_and_autoreset_lifecycle_trace",
    "P3-G1-REFERENCE-DIFFERENTIAL": "tests/manager/test_g1_reference_differential.py::test_g1_managed_reference_matches_handwritten_env",
    "P3-POLICY-ABI": "tests/training/test_managed_policy_abi.py::test_managed_policy_abi_mismatch_fails_closed",
    "P3-CROSS-BACKEND-PLAN": "tests/manager/test_cross_backend_plan.py::test_g1_plan_is_shared_by_mujoco_and_mjwarp",
    "P3-GENERALITY-FIXTURE": "tests/manager/test_manipulation_compile_fixture.py::test_multi_entity_manipulation_fixture_compiles",
}

PHASE3_MIN_REPETITIONS: dict[str, int] = {
    "P3-TASK-COMPILER": 2,
    "P3-LIFECYCLE-PARITY": 2,
    "P3-G1-REFERENCE-DIFFERENTIAL": 3,
    "P3-POLICY-ABI": 1,
    "P3-CROSS-BACKEND-PLAN": 2,
    "P3-GENERALITY-FIXTURE": 1,
}

PHASE3_MANIFEST_COMMANDS: dict[str, str] = {
    "P3-TASK-COMPILER": "uv run pytest tests/manager/test_task_compiler.py -v",
    "P3-LIFECYCLE-PARITY": "uv run pytest tests/manager/test_managed_lifecycle.py -v",
    "P3-G1-REFERENCE-DIFFERENTIAL": "uv run pytest tests/manager/test_g1_reference_differential.py -v",
    "P3-POLICY-ABI": "uv run pytest tests/training/test_managed_policy_abi.py -v",
    "P3-CROSS-BACKEND-PLAN": "uv run --with mujoco-warp --with warp-lang pytest -m slow tests/manager/test_cross_backend_plan.py -v",
    "P3-GENERALITY-FIXTURE": "uv run pytest tests/manager/test_manipulation_compile_fixture.py -v",
}

_CLAIM_CONFIG_INPUTS: dict[str, Path] = {
    "P3-TASK-COMPILER": Path("tests/manager/test_task_compiler.py"),
    "P3-LIFECYCLE-PARITY": MUJOCO_OWNER,
    "P3-G1-REFERENCE-DIFFERENTIAL": MUJOCO_OWNER,
    "P3-POLICY-ABI": MUJOCO_OWNER,
    "P3-CROSS-BACKEND-PLAN": MJWARP_OWNER,
    "P3-GENERALITY-FIXTURE": Path("tests/manager/test_manipulation_compile_fixture.py"),
}

_INPUT_FILES = (
    UV_LOCK,
    MUJOCO_OWNER,
    MJWARP_OWNER,
    Path("src/unilab/tools/issue705_phase3_evidence.py"),
    Path("scripts/capture_issue705_phase3_evidence.py"),
    Path("tests/tools/test_issue705_phase3_evidence.py"),
    Path("tests/manager/test_task_compiler.py"),
    Path("tests/manager/test_managed_lifecycle.py"),
    Path("tests/manager/test_g1_reference_differential.py"),
    Path("tests/training/test_managed_policy_abi.py"),
    Path("tests/manager/test_cross_backend_plan.py"),
    Path("tests/manager/test_manipulation_compile_fixture.py"),
)

_PYTEST_COUNT_RE = re.compile(r"(?<![A-Za-z0-9_])(\d+) (passed|skipped|xfailed|xpassed)")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


class Phase3EvidenceError(RuntimeError):
    """Raised when evidence capture cannot establish a trustworthy Phase 3 PASS."""


@dataclass(frozen=True)
class Phase3EvidenceCommand:
    """One pre-registered command and its mandatory repetitions."""

    name: str
    lane: str
    argv: tuple[str, ...]
    required_test_ids: tuple[str, ...]
    repetitions: int


PHASE3_COMMANDS = (
    Phase3EvidenceCommand(
        name="lane_a_task_compiler",
        lane="A",
        argv=(
            "uv",
            "run",
            "pytest",
            PHASE3_REQUIRED_TEST_IDS["P3-TASK-COMPILER"],
            "-v",
            "-rsxX",
        ),
        required_test_ids=(PHASE3_REQUIRED_TEST_IDS["P3-TASK-COMPILER"],),
        repetitions=PHASE3_MIN_REPETITIONS["P3-TASK-COMPILER"],
    ),
    Phase3EvidenceCommand(
        name="lane_a_policy_abi",
        lane="A",
        argv=(
            "uv",
            "run",
            "pytest",
            PHASE3_REQUIRED_TEST_IDS["P3-POLICY-ABI"],
            "-v",
            "-rsxX",
        ),
        required_test_ids=(PHASE3_REQUIRED_TEST_IDS["P3-POLICY-ABI"],),
        repetitions=PHASE3_MIN_REPETITIONS["P3-POLICY-ABI"],
    ),
    Phase3EvidenceCommand(
        name="lane_b_lifecycle",
        lane="B",
        argv=(
            "uv",
            "run",
            "pytest",
            PHASE3_REQUIRED_TEST_IDS["P3-LIFECYCLE-PARITY"],
            "-v",
            "-rsxX",
        ),
        required_test_ids=(PHASE3_REQUIRED_TEST_IDS["P3-LIFECYCLE-PARITY"],),
        repetitions=PHASE3_MIN_REPETITIONS["P3-LIFECYCLE-PARITY"],
    ),
    Phase3EvidenceCommand(
        name="lane_b_g1_reference",
        lane="B",
        argv=(
            "uv",
            "run",
            "pytest",
            PHASE3_REQUIRED_TEST_IDS["P3-G1-REFERENCE-DIFFERENTIAL"],
            "-v",
            "-rsxX",
        ),
        required_test_ids=(PHASE3_REQUIRED_TEST_IDS["P3-G1-REFERENCE-DIFFERENTIAL"],),
        repetitions=PHASE3_MIN_REPETITIONS["P3-G1-REFERENCE-DIFFERENTIAL"],
    ),
    Phase3EvidenceCommand(
        name="lane_b_generality",
        lane="B",
        argv=(
            "uv",
            "run",
            "pytest",
            PHASE3_REQUIRED_TEST_IDS["P3-GENERALITY-FIXTURE"],
            "-v",
            "-rsxX",
        ),
        required_test_ids=(PHASE3_REQUIRED_TEST_IDS["P3-GENERALITY-FIXTURE"],),
        repetitions=PHASE3_MIN_REPETITIONS["P3-GENERALITY-FIXTURE"],
    ),
    Phase3EvidenceCommand(
        name="lane_c_cross_backend",
        lane="C",
        argv=(
            "uv",
            "run",
            "--with",
            "mujoco-warp",
            "--with",
            "warp-lang",
            "pytest",
            "-m",
            "slow",
            PHASE3_REQUIRED_TEST_IDS["P3-CROSS-BACKEND-PLAN"],
            "-v",
            "-rsxX",
        ),
        required_test_ids=(PHASE3_REQUIRED_TEST_IDS["P3-CROSS-BACKEND-PLAN"],),
        repetitions=PHASE3_MIN_REPETITIONS["P3-CROSS-BACKEND-PLAN"],
    ),
)

_COMMAND_BY_TEST_ID = {
    test_id: command.name for command in PHASE3_COMMANDS for test_id in command.required_test_ids
}
_COMMAND_BY_NAME = {command.name: command for command in PHASE3_COMMANDS}


def sha256_file(path: Path) -> str:
    """Return the canonical hash representation used by acceptance manifests."""

    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def _run_checked(argv: Sequence[str], *, root: Path, context: str) -> str:
    result = subprocess.run(argv, cwd=root, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise Phase3EvidenceError(
            f"{context} failed with exit {result.returncode}:\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result.stdout.strip()


def _git(root: Path, *args: str) -> str:
    return _run_checked(("git", *args), root=root, context=f"git {' '.join(args)}")


def _assert_clean_source(root: Path) -> None:
    dirty = _git(root, "status", "--porcelain")
    if dirty:
        raise Phase3EvidenceError(
            "Phase 3 evidence must be captured from a clean source tree; found:\n" + dirty
        )


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError as exc:
        raise Phase3EvidenceError(f"required package {name!r} is not installed") from exc


def _cuda_environment(root: Path) -> dict[str, Any]:
    dependencies = load_mjwarp_dependencies()
    device = dependencies.warp.get_device()
    if not bool(device.is_cuda):
        raise Phase3EvidenceError("Phase 3 capture requires an active CUDA Warp device")
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
        },
    }


def _pytest_counts(output: str) -> dict[str, int]:
    counts = {"passed": 0, "skipped": 0, "xfailed": 0, "xpassed": 0}
    for raw_count, category in _PYTEST_COUNT_RE.findall(output):
        counts[category] += int(raw_count)
    return counts


def _run_evidence_command(
    command: Phase3EvidenceCommand, *, root: Path, repetition: int
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
        raise Phase3EvidenceError(
            f"{command.name} did not meet Phase 3 evidence requirements "
            f"({'; '.join(errors)}):\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
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


def capture_phase3_evidence(root: Path) -> dict[str, Any]:
    """Run every required A/B/C lane from a clean commit and return raw evidence."""

    root = root.resolve()
    missing = [path.as_posix() for path in _INPUT_FILES if not (root / path).is_file()]
    if missing:
        raise Phase3EvidenceError(
            "Phase 3 capture is missing required inputs: " + ", ".join(missing)
        )
    _assert_clean_source(root)
    source_commit = _git(root, "rev-parse", "HEAD")
    if not _COMMIT_RE.fullmatch(source_commit):
        raise Phase3EvidenceError(f"git returned an invalid commit SHA: {source_commit!r}")

    environment = _cuda_environment(root)
    commands = [
        _run_evidence_command(command, root=root, repetition=repetition)
        for command in PHASE3_COMMANDS
        for repetition in range(1, command.repetitions + 1)
    ]
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
            "files": {path.as_posix(): sha256_file(root / path) for path in _INPUT_FILES},
            "claim_config_hashes": {
                claim_id: sha256_file(root / path)
                for claim_id, path in _CLAIM_CONFIG_INPUTS.items()
            },
        },
        "environment": environment,
        "claims": [
            {
                "claim_id": claim_id,
                "required_test_id": test_id,
                "command": _COMMAND_BY_TEST_ID[test_id],
                "minimum_repetitions": PHASE3_MIN_REPETITIONS[claim_id],
                "config_input": _CLAIM_CONFIG_INPUTS[claim_id].as_posix(),
            }
            for claim_id, test_id in PHASE3_REQUIRED_TEST_IDS.items()
        ],
        "commands": commands,
    }


def write_phase3_evidence(report: Mapping[str, Any], output: Path) -> None:
    """Write an auditable JSON artifact after a successful capture."""

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dict(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_phase3_evidence(path: Path) -> dict[str, Any]:
    """Load one artifact while normalizing I/O and JSON failures for callers."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Phase3EvidenceError(f"cannot load Phase 3 evidence {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise Phase3EvidenceError(f"Phase 3 evidence {path} must contain a JSON object")
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


def _validate_inputs(report: Mapping[str, Any], *, root: Path, errors: list[str]) -> None:
    inputs = _mapping(report.get("inputs"), "inputs", errors)
    expected_paths = {path.as_posix() for path in _INPUT_FILES}
    raw_files = inputs.get("files")
    if not isinstance(raw_files, dict):
        errors.append("inputs.files: expected object")
        raw_files = {}
    if set(raw_files) != expected_paths:
        errors.append("inputs.files: paths do not exactly match the registered Phase 3 inputs")
    for path in _INPUT_FILES:
        key = path.as_posix()
        observed = _string(raw_files.get(key), f"inputs.files.{key}", errors)
        if not (root / path).is_file():
            errors.append(f"inputs.files.{key}: current input is missing")
            continue
        if observed and not _SHA256_RE.fullmatch(observed):
            errors.append(f"inputs.files.{key}: expected sha256 hash")
        elif observed and observed != sha256_file(root / path):
            errors.append(f"inputs.files.{key}: does not match current input")

    raw_config_hashes = inputs.get("claim_config_hashes")
    if not isinstance(raw_config_hashes, dict):
        errors.append("inputs.claim_config_hashes: expected object")
        raw_config_hashes = {}
    if set(raw_config_hashes) != set(PHASE3_REQUIRED_TEST_IDS):
        errors.append("inputs.claim_config_hashes: claim IDs do not exactly match Phase 3")
    for claim_id, input_path in _CLAIM_CONFIG_INPUTS.items():
        observed = _string(
            raw_config_hashes.get(claim_id),
            f"inputs.claim_config_hashes.{claim_id}",
            errors,
        )
        if not (root / input_path).is_file():
            errors.append(f"inputs.claim_config_hashes.{claim_id}: current input is missing")
            continue
        if observed and not _SHA256_RE.fullmatch(observed):
            errors.append(f"inputs.claim_config_hashes.{claim_id}: expected sha256 hash")
        elif observed and observed != sha256_file(root / input_path):
            errors.append(f"inputs.claim_config_hashes.{claim_id}: does not match current input")


def _validate_environment(report: Mapping[str, Any], *, errors: list[str]) -> None:
    environment = _mapping(report.get("environment"), "environment", errors)
    nvidia_smi = environment.get("nvidia_smi")
    if (
        not isinstance(nvidia_smi, list)
        or not nvidia_smi
        or not all(isinstance(value, str) and value for value in nvidia_smi)
    ):
        errors.append("environment.nvidia_smi: expected non-empty GPU provenance")
    warp_device = _string(environment.get("warp_device"), "environment.warp_device", errors)
    if warp_device and "cuda" not in warp_device.lower():
        errors.append("environment.warp_device: expected CUDA device")
    packages = _mapping(environment.get("packages"), "environment.packages", errors)
    for package_name in ("mujoco", "mujoco-warp", "warp-lang"):
        _string(packages.get(package_name), f"environment.packages.{package_name}", errors)


def _validate_commands(
    report: Mapping[str, Any], *, errors: list[str]
) -> dict[str, list[dict[str, Any]]]:
    raw_commands = report.get("commands")
    if not isinstance(raw_commands, list) or not raw_commands:
        errors.append("commands: expected non-empty list")
        raw_commands = []
    by_series: dict[str, list[dict[str, Any]]] = {}
    names: set[str] = set()
    seen_lanes: set[str] = set()
    for index, raw_command in enumerate(raw_commands):
        command = _mapping(raw_command, f"commands[{index}]", errors)
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
        expected = _COMMAND_BY_NAME.get(series)
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
    if seen_lanes != {"A", "B", "C"}:
        errors.append(f"commands: expected exactly lanes A/B/C, got {sorted(seen_lanes)!r}")
    if set(by_series) != set(_COMMAND_BY_NAME):
        errors.append("commands: series do not exactly match the registered Phase 3 commands")
    for expected in PHASE3_COMMANDS:
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
    report: Mapping[str, Any], *, by_series: Mapping[str, list[dict[str, Any]]], errors: list[str]
) -> None:
    raw_claims = report.get("claims")
    if not isinstance(raw_claims, list):
        errors.append("claims: expected list")
        raw_claims = []
    observed: dict[str, tuple[str, str, int, str]] = {}
    for index, raw_claim in enumerate(raw_claims):
        claim = _mapping(raw_claim, f"claims[{index}]", errors)
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
    if set(observed) != set(PHASE3_REQUIRED_TEST_IDS):
        errors.append("claims: claim IDs do not exactly match Phase 3 required claims")
    for claim_id, expected_test_id in PHASE3_REQUIRED_TEST_IDS.items():
        test_id, command_name, repetitions, config_input = observed.get(claim_id, ("", "", -1, ""))
        if test_id != expected_test_id:
            errors.append(f"claims[{claim_id}]: required test mapping does not match")
        if repetitions != PHASE3_MIN_REPETITIONS[claim_id]:
            errors.append(f"claims[{claim_id}]: minimum repetitions do not match Phase 3 manifest")
        if config_input != _CLAIM_CONFIG_INPUTS[claim_id].as_posix():
            errors.append(f"claims[{claim_id}]: config input does not match registered mapping")
        matching = by_series.get(command_name, [])
        if len(matching) < PHASE3_MIN_REPETITIONS[claim_id]:
            errors.append(f"claims[{claim_id}]: insufficient successful repetitions")
        for command in matching:
            required_test_ids = command.get("required_test_ids")
            if not isinstance(required_test_ids, list) or expected_test_id not in required_test_ids:
                errors.append(f"claims[{claim_id}]: command does not declare its required test")


def _validate_manifest_contract(root: Path, *, errors: list[str]) -> None:
    try:
        manifest = load_phase_acceptance(root / MANIFEST_PATH)
    except ManifestValidationError as exc:
        errors.extend(f"manifest: {error}" for error in exc.errors)
        return
    if manifest.issue != ISSUE or manifest.phase != PHASE:
        errors.append(f"manifest: expected issue/phase {ISSUE}/{PHASE}")
    claims = {claim.claim_id: claim for claim in manifest.claims}
    if set(claims) != set(PHASE3_REQUIRED_TEST_IDS):
        errors.append("manifest: claim IDs do not exactly match registered Phase 3 claims")
        return
    command_by_claim = {
        test_id: _COMMAND_BY_NAME[_COMMAND_BY_TEST_ID[test_id]]
        for test_id in PHASE3_REQUIRED_TEST_IDS.values()
    }
    for claim_id, test_id in PHASE3_REQUIRED_TEST_IDS.items():
        claim = claims[claim_id]
        expected_command = command_by_claim[test_id]
        if claim.required_test_ids != (test_id,):
            errors.append(f"manifest[{claim_id}]: required test mapping does not match")
        if claim.acceptance.repetitions != PHASE3_MIN_REPETITIONS[claim_id]:
            errors.append(f"manifest[{claim_id}]: repetitions do not match registered mapping")
        if claim.lane.value != expected_command.lane:
            errors.append(f"manifest[{claim_id}]: lane does not match registered mapping")
        if claim.commands != (PHASE3_MANIFEST_COMMANDS[claim_id],):
            errors.append(f"manifest[{claim_id}]: command does not match registered mapping")


def validate_phase3_evidence(report: Mapping[str, Any], *, root: Path) -> tuple[str, ...]:
    """Return all artifact/manifest/provenance errors without trusting a PASS JSON."""

    root = root.resolve()
    errors: list[str] = []
    if report.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version: expected {SCHEMA_VERSION}")
    if report.get("kind") != ARTIFACT_KIND:
        errors.append(f"kind: expected {ARTIFACT_KIND!r}")
    if report.get("issue") != ISSUE or report.get("phase") != PHASE:
        errors.append(f"issue/phase: expected {ISSUE}/{PHASE}")
    _string(report.get("generated_at_utc"), "generated_at_utc", errors)

    source = _mapping(report.get("source"), "source", errors)
    commit = _string(source.get("commit_sha"), "source.commit_sha", errors)
    if commit and not _COMMIT_RE.fullmatch(commit):
        errors.append("source.commit_sha: expected full 40-character SHA")
    elif commit and not _git_is_ancestor(root, commit):
        errors.append("source.commit_sha: must be an ancestor of the current gate commit")
    if source.get("tree_clean") is not True:
        errors.append("source.tree_clean: must be true")

    _validate_inputs(report, root=root, errors=errors)
    _validate_environment(report, errors=errors)
    commands = _validate_commands(report, errors=errors)
    _validate_claims(report, by_series=commands, errors=errors)
    _validate_manifest_contract(root, errors=errors)
    return tuple(errors)


__all__ = [
    "ARTIFACT_KIND",
    "ISSUE",
    "MANIFEST_PATH",
    "PHASE",
    "PHASE3_COMMANDS",
    "PHASE3_MANIFEST_COMMANDS",
    "PHASE3_MIN_REPETITIONS",
    "PHASE3_REQUIRED_TEST_IDS",
    "Phase3EvidenceError",
    "capture_phase3_evidence",
    "load_phase3_evidence",
    "sha256_file",
    "validate_phase3_evidence",
    "write_phase3_evidence",
]
