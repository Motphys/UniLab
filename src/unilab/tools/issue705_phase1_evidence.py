"""Fail-closed evidence contract for the Issue #705 Phase 1 gate."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from unilab.tools.issue705_phase_evidence import (
    PhaseEvidenceClaim,
    PhaseEvidenceCommand,
    PhaseEvidenceError,
    PhaseEvidenceSpec,
    capture_phase_evidence,
    load_phase_evidence,
    sha256_file,
    tracked_input_files,
    validate_phase_evidence,
    write_phase_evidence,
)

ISSUE = 705
PHASE = 1
ARTIFACT_KIND = "issue705-phase1-gate-v1"
MANIFEST_PATH = Path("tests/acceptance/issue_705/manifests/phase_1.yaml")
MUJOCO_OWNER = Path("conf/ppo/task/g1_walk_flat/mujoco.yaml")
ROOT_DIR = Path(__file__).resolve().parents[3]

PHASE1_REQUIRED_TEST_IDS: dict[str, str] = {
    "P1-BATCH-CONTRACT": "tests/base/test_backend_batch_contract.py::test_bound_state_batch_contract",
    "P1-MUJOCO-REFERENCE": "tests/base/test_backend_batch_contract.py::test_mujoco_batch_matches_getter_reference",
    "P1-MUTATION-CONTRACT": "tests/dr/test_mutation_contract.py::test_typed_mutation_conflicts_fail_closed",
    "P1-HOT-PATH-INSTRUMENTATION": "tests/base/test_backend_batch_contract.py::test_managed_hot_path_has_no_dynamic_getters",
    "P1-DR-COMPATIBILITY": "tests/dr/test_mutation_contract.py::test_manager_filters_unsupported_terms_and_warns_once",
    "P1-BACKEND-ISOLATION": "tests/base/test_backend_batch_contract.py::test_runtime_backends_share_only_cold_materialization_contract",
}

PHASE1_MIN_REPETITIONS: dict[str, int] = {
    "P1-BATCH-CONTRACT": 1,
    "P1-MUJOCO-REFERENCE": 3,
    "P1-MUTATION-CONTRACT": 2,
    "P1-HOT-PATH-INSTRUMENTATION": 2,
    "P1-DR-COMPATIBILITY": 1,
    "P1-BACKEND-ISOLATION": 1,
}

PHASE1_MANIFEST_COMMANDS: dict[str, str] = {
    "P1-BATCH-CONTRACT": "uv run pytest tests/base/test_backend_batch_contract.py -v",
    "P1-MUJOCO-REFERENCE": "uv run pytest tests/base/test_backend_batch_contract.py -k mujoco -v",
    "P1-MUTATION-CONTRACT": "uv run pytest tests/dr/test_mutation_contract.py -v",
    "P1-HOT-PATH-INSTRUMENTATION": (
        "uv run pytest tests/base/test_backend_batch_contract.py::"
        "test_managed_hot_path_has_no_dynamic_getters -v"
    ),
    "P1-DR-COMPATIBILITY": (
        "uv run pytest tests/dr/test_mutation_contract.py::"
        "test_manager_filters_unsupported_terms_and_warns_once -v"
    ),
    "P1-BACKEND-ISOLATION": (
        "uv run pytest tests/base/test_backend_batch_contract.py::"
        "test_runtime_backends_share_only_cold_materialization_contract -v"
    ),
}

_CLAIM_CONFIG_INPUTS: dict[str, Path] = {
    "P1-BATCH-CONTRACT": Path("tests/base/test_backend_batch_contract.py"),
    "P1-MUJOCO-REFERENCE": MUJOCO_OWNER,
    "P1-MUTATION-CONTRACT": Path("tests/dr/test_mutation_contract.py"),
    "P1-HOT-PATH-INSTRUMENTATION": MUJOCO_OWNER,
    "P1-DR-COMPATIBILITY": Path("tests/dr/test_mutation_contract.py"),
    "P1-BACKEND-ISOLATION": Path("tests/base/test_backend_batch_contract.py"),
}

PHASE1_COMMANDS = (
    PhaseEvidenceCommand(
        name="lane_a_batch_contract",
        lane="A",
        argv=(
            "uv",
            "run",
            "pytest",
            PHASE1_REQUIRED_TEST_IDS["P1-BATCH-CONTRACT"],
            "-v",
            "-rsxX",
        ),
        required_test_ids=(PHASE1_REQUIRED_TEST_IDS["P1-BATCH-CONTRACT"],),
        repetitions=PHASE1_MIN_REPETITIONS["P1-BATCH-CONTRACT"],
    ),
    PhaseEvidenceCommand(
        name="lane_a_backend_isolation",
        lane="A",
        argv=(
            "uv",
            "run",
            "pytest",
            PHASE1_REQUIRED_TEST_IDS["P1-BACKEND-ISOLATION"],
            "-v",
            "-rsxX",
        ),
        required_test_ids=(PHASE1_REQUIRED_TEST_IDS["P1-BACKEND-ISOLATION"],),
        repetitions=PHASE1_MIN_REPETITIONS["P1-BACKEND-ISOLATION"],
    ),
    PhaseEvidenceCommand(
        name="lane_b_mujoco_reference",
        lane="B",
        argv=(
            "uv",
            "run",
            "pytest",
            PHASE1_REQUIRED_TEST_IDS["P1-MUJOCO-REFERENCE"],
            "-v",
            "-rsxX",
        ),
        required_test_ids=(PHASE1_REQUIRED_TEST_IDS["P1-MUJOCO-REFERENCE"],),
        repetitions=PHASE1_MIN_REPETITIONS["P1-MUJOCO-REFERENCE"],
    ),
    PhaseEvidenceCommand(
        name="lane_b_mutation_contract",
        lane="B",
        argv=(
            "uv",
            "run",
            "pytest",
            PHASE1_REQUIRED_TEST_IDS["P1-MUTATION-CONTRACT"],
            "-v",
            "-rsxX",
        ),
        required_test_ids=(PHASE1_REQUIRED_TEST_IDS["P1-MUTATION-CONTRACT"],),
        repetitions=PHASE1_MIN_REPETITIONS["P1-MUTATION-CONTRACT"],
    ),
    PhaseEvidenceCommand(
        name="lane_b_hot_path",
        lane="B",
        argv=(
            "uv",
            "run",
            "pytest",
            PHASE1_REQUIRED_TEST_IDS["P1-HOT-PATH-INSTRUMENTATION"],
            "-v",
            "-rsxX",
        ),
        required_test_ids=(PHASE1_REQUIRED_TEST_IDS["P1-HOT-PATH-INSTRUMENTATION"],),
        repetitions=PHASE1_MIN_REPETITIONS["P1-HOT-PATH-INSTRUMENTATION"],
    ),
    PhaseEvidenceCommand(
        name="lane_b_dr_compatibility",
        lane="B",
        argv=(
            "uv",
            "run",
            "pytest",
            PHASE1_REQUIRED_TEST_IDS["P1-DR-COMPATIBILITY"],
            "-v",
            "-rsxX",
        ),
        required_test_ids=(PHASE1_REQUIRED_TEST_IDS["P1-DR-COMPATIBILITY"],),
        repetitions=PHASE1_MIN_REPETITIONS["P1-DR-COMPATIBILITY"],
    ),
)

_COMMAND_BY_CLAIM = {
    "P1-BATCH-CONTRACT": "lane_a_batch_contract",
    "P1-MUJOCO-REFERENCE": "lane_b_mujoco_reference",
    "P1-MUTATION-CONTRACT": "lane_b_mutation_contract",
    "P1-HOT-PATH-INSTRUMENTATION": "lane_b_hot_path",
    "P1-DR-COMPATIBILITY": "lane_b_dr_compatibility",
    "P1-BACKEND-ISOLATION": "lane_a_backend_isolation",
}

_SOURCE_INPUT_TREES = (
    Path("src/unilab/base"),
    Path("src/unilab/dr"),
    Path("src/unilab/envs"),
)
_STATIC_INPUTS = (
    Path("uv.lock"),
    MUJOCO_OWNER,
    Path("src/unilab/tools/backend_isolation.py"),
    Path("src/unilab/tools/issue705_phase_evidence.py"),
    Path("src/unilab/tools/issue705_phase1_evidence.py"),
    Path("scripts/audit_issue705_backend_isolation.py"),
    Path("scripts/capture_issue705_phase1_evidence.py"),
    Path("tests/tools/test_issue705_phase1_evidence.py"),
    Path("tests/tools/test_backend_isolation.py"),
    Path("tests/acceptance/issue_705/test_phase1_evidence.py"),
    Path("tests/base/test_backend_batch_contract.py"),
    Path("tests/dr/test_mutation_contract.py"),
)
_INPUT_FILES = tuple(
    sorted(
        {
            *_STATIC_INPUTS,
            *(
                path
                for path in tracked_input_files(ROOT_DIR, _SOURCE_INPUT_TREES)
                if path.suffix == ".py"
            ),
        },
        key=Path.as_posix,
    )
)

PHASE1_SPEC = PhaseEvidenceSpec(
    issue=ISSUE,
    phase=PHASE,
    artifact_kind=ARTIFACT_KIND,
    manifest_path=MANIFEST_PATH,
    required_lanes=("A", "B"),
    input_files=_INPUT_FILES,
    package_names=("mujoco",),
    commands=PHASE1_COMMANDS,
    claims=tuple(
        PhaseEvidenceClaim(
            claim_id=claim_id,
            required_test_id=test_id,
            command_name=_COMMAND_BY_CLAIM[claim_id],
            minimum_repetitions=PHASE1_MIN_REPETITIONS[claim_id],
            config_input=_CLAIM_CONFIG_INPUTS[claim_id],
            manifest_command=PHASE1_MANIFEST_COMMANDS[claim_id],
        )
        for claim_id, test_id in PHASE1_REQUIRED_TEST_IDS.items()
    ),
)


def capture_phase1_evidence(root: Path) -> dict[str, Any]:
    """Run every Phase 1 A/B evidence command from a clean source commit."""

    return capture_phase_evidence(PHASE1_SPEC, root)


def load_phase1_evidence(path: Path) -> dict[str, Any]:
    """Load one Phase 1 raw evidence artifact."""

    return load_phase_evidence(path)


def validate_phase1_evidence(report: Mapping[str, Any], *, root: Path) -> tuple[str, ...]:
    """Validate raw evidence and the independent Phase 1 manifest mapping."""

    return validate_phase_evidence(PHASE1_SPEC, report, root=root)


def write_phase1_evidence(report: Mapping[str, Any], output: Path) -> None:
    """Persist a previously validated Phase 1 capture."""

    write_phase_evidence(report, output)


__all__ = [
    "ARTIFACT_KIND",
    "ISSUE",
    "MANIFEST_PATH",
    "PHASE",
    "PHASE1_COMMANDS",
    "PHASE1_MIN_REPETITIONS",
    "PHASE1_REQUIRED_TEST_IDS",
    "PHASE1_SPEC",
    "PhaseEvidenceError",
    "capture_phase1_evidence",
    "load_phase1_evidence",
    "sha256_file",
    "validate_phase1_evidence",
    "write_phase1_evidence",
]
