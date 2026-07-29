"""Fail-closed evidence contract for the Issue #705 Phase 4 gate."""

from __future__ import annotations

import json
import sys
from importlib import import_module
from pathlib import Path
from typing import Any, Mapping, cast

from unilab.tools.issue705_phase_evidence import (
    PhaseEvidenceClaim,
    PhaseEvidenceCommand,
    PhaseEvidenceError,
    PhaseEvidenceSpec,
    capture_phase_evidence,
    load_phase_evidence,
    sha256_file,
    validate_phase_evidence,
    write_phase_evidence,
)

ISSUE = 705
PHASE = 4
ARTIFACT_KIND = "issue705-phase4-gate-v1"
MANIFEST_PATH = Path("tests/acceptance/issue_705/manifests/phase_4.yaml")
MUJOCO_OWNER = Path("conf/ppo/task/g1_walk_flat/mujoco.yaml")
HOST_BENCHMARK_ARTIFACT = Path("tests/acceptance/issue_705/artifacts/phase_4_host_benchmark.json")

PHASE4_REQUIRED_TEST_IDS: dict[str, str] = {
    "P4-FUSED-PARITY": (
        "tests/manager/test_fused_executor.py::"
        "test_fused_executor_matches_reference_generated_vectors"
    ),
    "P4-NO-FALLBACK": (
        "tests/manager/test_fused_executor.py::test_fused_executor_never_silently_falls_back"
    ),
    "P4-ALLOCATION-STABILITY": (
        "tests/manager/test_fused_runtime_stability.py::"
        "test_warm_loop_has_stable_addresses_and_allocations"
    ),
    "P4-HOST-PERFORMANCE": (
        "tests/benchmark/test_managed_g1_host_benchmark.py::"
        "test_fused_host_meets_preregistered_gate"
    ),
}

PHASE4_MIN_REPETITIONS: dict[str, int] = {
    "P4-FUSED-PARITY": 3,
    "P4-NO-FALLBACK": 1,
    "P4-ALLOCATION-STABILITY": 3,
    "P4-HOST-PERFORMANCE": 5,
}

PHASE4_MANIFEST_COMMANDS: dict[str, str] = {
    "P4-FUSED-PARITY": "uv run pytest tests/manager/test_fused_executor.py -v",
    "P4-NO-FALLBACK": (
        "uv run pytest tests/manager/test_fused_executor.py::"
        "test_fused_executor_never_silently_falls_back -v"
    ),
    "P4-ALLOCATION-STABILITY": (
        "uv run pytest -o addopts='' tests/manager/test_fused_runtime_stability.py -v"
    ),
    "P4-HOST-PERFORMANCE": ("uv run benchmark/env/benchmark_managed_g1.py --profile host_numpy"),
}

_CLAIM_CONFIG_INPUTS: dict[str, Path] = {
    "P4-FUSED-PARITY": MUJOCO_OWNER,
    "P4-NO-FALLBACK": Path("tests/manager/test_fused_executor.py"),
    "P4-ALLOCATION-STABILITY": MUJOCO_OWNER,
    "P4-HOST-PERFORMANCE": MUJOCO_OWNER,
}

PHASE4_COMMANDS = (
    PhaseEvidenceCommand(
        name="lane_a_no_fallback",
        lane="A",
        argv=(
            "uv",
            "run",
            "pytest",
            PHASE4_REQUIRED_TEST_IDS["P4-NO-FALLBACK"],
            "-v",
            "-rsxX",
        ),
        required_test_ids=(PHASE4_REQUIRED_TEST_IDS["P4-NO-FALLBACK"],),
        repetitions=PHASE4_MIN_REPETITIONS["P4-NO-FALLBACK"],
    ),
    PhaseEvidenceCommand(
        name="lane_b_fused_parity",
        lane="B",
        argv=(
            "uv",
            "run",
            "pytest",
            PHASE4_REQUIRED_TEST_IDS["P4-FUSED-PARITY"],
            "-v",
            "-rsxX",
        ),
        required_test_ids=(PHASE4_REQUIRED_TEST_IDS["P4-FUSED-PARITY"],),
        repetitions=PHASE4_MIN_REPETITIONS["P4-FUSED-PARITY"],
    ),
    PhaseEvidenceCommand(
        name="lane_b_allocation_stability",
        lane="B",
        argv=(
            "uv",
            "run",
            "pytest",
            "-o",
            "addopts=",
            PHASE4_REQUIRED_TEST_IDS["P4-ALLOCATION-STABILITY"],
            "-v",
            "-rsxX",
        ),
        required_test_ids=(PHASE4_REQUIRED_TEST_IDS["P4-ALLOCATION-STABILITY"],),
        repetitions=PHASE4_MIN_REPETITIONS["P4-ALLOCATION-STABILITY"],
    ),
    PhaseEvidenceCommand(
        name="lane_d_host_performance",
        lane="D",
        argv=(
            "uv",
            "run",
            "pytest",
            PHASE4_REQUIRED_TEST_IDS["P4-HOST-PERFORMANCE"],
            "-v",
            "-rsxX",
        ),
        required_test_ids=(PHASE4_REQUIRED_TEST_IDS["P4-HOST-PERFORMANCE"],),
        repetitions=PHASE4_MIN_REPETITIONS["P4-HOST-PERFORMANCE"],
    ),
)

_COMMAND_BY_CLAIM = {
    "P4-FUSED-PARITY": "lane_b_fused_parity",
    "P4-NO-FALLBACK": "lane_a_no_fallback",
    "P4-ALLOCATION-STABILITY": "lane_b_allocation_stability",
    "P4-HOST-PERFORMANCE": "lane_d_host_performance",
}

_INPUT_FILES = (
    Path("uv.lock"),
    MUJOCO_OWNER,
    Path("tests/acceptance/issue_705/g1_mujoco_baseline_plan.yaml"),
    Path("tests/acceptance/issue_705/g1_threshold_manifest.yaml"),
    Path("tests/acceptance/issue_705/g1_threshold_freeze_receipt.yaml"),
    HOST_BENCHMARK_ARTIFACT,
    Path("benchmark/env/benchmark_managed_g1.py"),
    Path("src/unilab/tools/issue705_phase_evidence.py"),
    Path("src/unilab/tools/issue705_phase4_evidence.py"),
    Path("scripts/capture_issue705_phase4_evidence.py"),
    Path("tests/tools/test_issue705_phase4_evidence.py"),
    Path("tests/acceptance/issue_705/test_phase4_evidence.py"),
    Path("tests/manager/test_fused_executor.py"),
    Path("tests/manager/test_fused_runtime_stability.py"),
    Path("tests/benchmark/test_managed_g1_host_benchmark.py"),
)

PHASE4_SPEC = PhaseEvidenceSpec(
    issue=ISSUE,
    phase=PHASE,
    artifact_kind=ARTIFACT_KIND,
    manifest_path=MANIFEST_PATH,
    required_lanes=("A", "B", "D"),
    input_files=_INPUT_FILES,
    package_names=("mujoco", "numba"),
    commands=PHASE4_COMMANDS,
    claims=tuple(
        PhaseEvidenceClaim(
            claim_id=claim_id,
            required_test_id=test_id,
            command_name=_COMMAND_BY_CLAIM[claim_id],
            minimum_repetitions=PHASE4_MIN_REPETITIONS[claim_id],
            config_input=_CLAIM_CONFIG_INPUTS[claim_id],
            manifest_command=PHASE4_MANIFEST_COMMANDS[claim_id],
        )
        for claim_id, test_id in PHASE4_REQUIRED_TEST_IDS.items()
    ),
)


def load_host_benchmark_artifact(root: Path) -> dict[str, Any]:
    """Load the committed raw host artifact as untrusted JSON."""

    path = root.resolve() / HOST_BENCHMARK_ARTIFACT
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PhaseEvidenceError(f"cannot load Phase 4 host benchmark {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PhaseEvidenceError("Phase 4 host benchmark must contain a JSON object")
    return value


def _host_benchmark_module(root: Path) -> Any:
    """Import the benchmark owner from a script entrypoint's cold path.

    ``uv run scripts/...`` places ``scripts/`` rather than the repository
    root on ``sys.path``.  The benchmark is intentionally not a runtime
    dependency of ``unilab``; this explicit, cold-path root insertion is only
    used by the evidence CLI and preserves the benchmark module as its owner.
    """

    root_text = str(root.resolve())
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    return import_module("benchmark.env.benchmark_managed_g1")


def validate_host_benchmark_payload(
    artifact: object,
    *,
    root: Path,
) -> tuple[str, ...]:
    """Recompute every frozen host gate, including candidate provenance."""

    root = root.resolve()
    try:
        host_benchmark = _host_benchmark_module(root)
    except ModuleNotFoundError as exc:
        return (f"host benchmark module could not be imported: {exc}",)
    if root != host_benchmark.ROOT_DIR.resolve():
        return ("host benchmark validator must run from the UniLab repository root",)
    try:
        plan = host_benchmark._load_plan(host_benchmark.DEFAULT_BASELINE_PLAN)
        binding = host_benchmark.load_threshold_binding()
    except (OSError, ValueError, RuntimeError) as exc:
        return (f"host benchmark contract could not be loaded: {exc}",)
    return cast(
        tuple[str, ...],
        host_benchmark.validate_artifact(
            artifact,
            binding=binding,
            plan=plan,
            repo_root=root,
        ),
    )


def validate_host_benchmark_artifact(*, root: Path) -> tuple[str, ...]:
    """Load and validate the committed Phase 4 raw performance artifact."""

    try:
        artifact = load_host_benchmark_artifact(root)
    except PhaseEvidenceError as exc:
        return (str(exc),)
    return validate_host_benchmark_payload(artifact, root=root)


def capture_phase4_evidence(root: Path) -> dict[str, Any]:
    """Run the registered A/B/D matrix after validating the raw benchmark."""

    host_errors = validate_host_benchmark_artifact(root=root)
    if host_errors:
        raise PhaseEvidenceError(
            "Phase 4 host benchmark is not valid:\n"
            + "\n".join(f"- {error}" for error in host_errors)
        )
    report = capture_phase_evidence(PHASE4_SPEC, root)
    errors = validate_phase_evidence(PHASE4_SPEC, report, root=root)
    if errors:
        raise PhaseEvidenceError(
            "captured Phase 4 evidence failed validation:\n"
            + "\n".join(f"- {error}" for error in errors)
        )
    return report


def load_phase4_evidence(path: Path) -> dict[str, Any]:
    """Load one Phase 4 gate artifact."""

    return load_phase_evidence(path)


def validate_phase4_evidence(report: Mapping[str, Any], *, root: Path) -> tuple[str, ...]:
    """Validate command evidence and independently recompute the host gate."""

    errors = list(validate_phase_evidence(PHASE4_SPEC, report, root=root))
    errors.extend(
        f"host benchmark: {error}" for error in validate_host_benchmark_artifact(root=root)
    )
    return tuple(errors)


def write_phase4_evidence(report: Mapping[str, Any], output: Path) -> None:
    """Persist a previously validated Phase 4 capture."""

    write_phase_evidence(report, output)


__all__ = [
    "ARTIFACT_KIND",
    "HOST_BENCHMARK_ARTIFACT",
    "ISSUE",
    "MANIFEST_PATH",
    "PHASE",
    "PHASE4_COMMANDS",
    "PHASE4_MIN_REPETITIONS",
    "PHASE4_REQUIRED_TEST_IDS",
    "PHASE4_SPEC",
    "PhaseEvidenceError",
    "capture_phase4_evidence",
    "load_host_benchmark_artifact",
    "load_phase4_evidence",
    "sha256_file",
    "validate_host_benchmark_artifact",
    "validate_host_benchmark_payload",
    "validate_phase4_evidence",
    "write_phase4_evidence",
]
