"""Fail-closed evidence contract for the Issue #705 Phase 3 gate."""

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
PHASE = 3
SCHEMA_VERSION = 1
ARTIFACT_KIND = "issue705-phase3-gate-v1"
MANIFEST_PATH = Path("tests/acceptance/issue_705/manifests/phase_3.yaml")
MUJOCO_OWNER = Path("conf/ppo/task/g1_walk_flat/mujoco.yaml")
MJWARP_OWNER = Path("conf/ppo/task/g1_walk_flat/mjwarp.yaml")
ROOT_DIR = Path(__file__).resolve().parents[3]

PHASE3_REQUIRED_TEST_IDS: dict[str, str] = {
    "P3-TASK-COMPILER": (
        "tests/manager/test_task_compiler.py::test_compiler_binds_and_freezes_complete_plan"
    ),
    "P3-LIFECYCLE-PARITY": (
        "tests/manager/test_managed_lifecycle.py::test_terminal_and_autoreset_lifecycle_trace"
    ),
    "P3-G1-REFERENCE-DIFFERENTIAL": (
        "tests/manager/test_g1_reference_differential.py::"
        "test_g1_managed_reference_matches_handwritten_env"
    ),
    "P3-POLICY-ABI": (
        "tests/training/test_managed_policy_abi.py::test_managed_policy_abi_mismatch_fails_closed"
    ),
    "P3-CROSS-BACKEND-PLAN": (
        "tests/manager/test_cross_backend_plan.py::test_g1_plan_is_shared_by_mujoco_and_mjwarp"
    ),
    "P3-GENERALITY-FIXTURE": (
        "tests/manager/test_manipulation_compile_fixture.py::"
        "test_multi_entity_manipulation_fixture_compiles"
    ),
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
    "P3-G1-REFERENCE-DIFFERENTIAL": (
        "uv run pytest tests/manager/test_g1_reference_differential.py -v"
    ),
    "P3-POLICY-ABI": "uv run pytest tests/training/test_managed_policy_abi.py -v",
    "P3-CROSS-BACKEND-PLAN": (
        "uv run --with mujoco-warp --with warp-lang pytest -m slow "
        "tests/manager/test_cross_backend_plan.py -v"
    ),
    "P3-GENERALITY-FIXTURE": (
        "uv run pytest tests/manager/test_manipulation_compile_fixture.py -v"
    ),
}

_CLAIM_CONFIG_INPUTS: dict[str, Path] = {
    "P3-TASK-COMPILER": Path("tests/manager/test_task_compiler.py"),
    "P3-LIFECYCLE-PARITY": MUJOCO_OWNER,
    "P3-G1-REFERENCE-DIFFERENTIAL": MUJOCO_OWNER,
    "P3-POLICY-ABI": MUJOCO_OWNER,
    "P3-CROSS-BACKEND-PLAN": MJWARP_OWNER,
    "P3-GENERALITY-FIXTURE": Path("tests/manager/test_manipulation_compile_fixture.py"),
}


def _pytest_command(name: str, claim_id: str, *, lane: str) -> PhaseEvidenceCommand:
    return PhaseEvidenceCommand(
        name=name,
        lane=lane,
        argv=(
            "uv",
            "run",
            "pytest",
            PHASE3_REQUIRED_TEST_IDS[claim_id],
            "-v",
            "-rsxX",
        ),
        required_test_ids=(PHASE3_REQUIRED_TEST_IDS[claim_id],),
        repetitions=PHASE3_MIN_REPETITIONS[claim_id],
    )


PHASE3_COMMANDS = (
    _pytest_command("lane_a_task_compiler", "P3-TASK-COMPILER", lane="A"),
    _pytest_command("lane_a_policy_abi", "P3-POLICY-ABI", lane="A"),
    _pytest_command("lane_b_lifecycle", "P3-LIFECYCLE-PARITY", lane="B"),
    _pytest_command("lane_b_g1_reference", "P3-G1-REFERENCE-DIFFERENTIAL", lane="B"),
    _pytest_command("lane_b_generality", "P3-GENERALITY-FIXTURE", lane="B"),
    PhaseEvidenceCommand(
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

_COMMAND_BY_CLAIM = {
    claim_id: command.name
    for command in PHASE3_COMMANDS
    for claim_id, test_id in PHASE3_REQUIRED_TEST_IDS.items()
    if test_id in command.required_test_ids
}
_SOURCE_INPUT_TREES = (
    Path("src/unilab/base/backend"),
    Path("src/unilab/envs/locomotion/g1"),
    Path("src/unilab/manager"),
)
_STATIC_INPUTS = (
    Path("uv.lock"),
    MUJOCO_OWNER,
    MJWARP_OWNER,
    Path("src/unilab/base/final_observation.py"),
    Path("src/unilab/base/np_env.py"),
    Path("src/unilab/base/registry.py"),
    Path("src/unilab/base/scene.py"),
    Path("src/unilab/training/sim2sim.py"),
    Path("src/unilab/tools/issue705_phase_evidence.py"),
    Path("src/unilab/tools/issue705_phase3_evidence.py"),
    Path("scripts/capture_issue705_phase3_evidence.py"),
    Path("tests/tools/test_issue705_phase3_evidence.py"),
    Path("tests/acceptance/issue_705/test_phase3_evidence.py"),
    Path("tests/manager/test_task_compiler.py"),
    Path("tests/manager/test_managed_lifecycle.py"),
    Path("tests/manager/test_g1_reference_differential.py"),
    Path("tests/training/test_managed_policy_abi.py"),
    Path("tests/manager/test_cross_backend_plan.py"),
    Path("tests/manager/test_manipulation_compile_fixture.py"),
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

PHASE3_SPEC = PhaseEvidenceSpec(
    issue=ISSUE,
    phase=PHASE,
    artifact_kind=ARTIFACT_KIND,
    manifest_path=MANIFEST_PATH,
    required_lanes=("A", "B", "C"),
    input_files=_INPUT_FILES,
    package_names=("mujoco", "mujoco-warp", "warp-lang"),
    commands=PHASE3_COMMANDS,
    claims=tuple(
        PhaseEvidenceClaim(
            claim_id=claim_id,
            required_test_id=test_id,
            command_name=_COMMAND_BY_CLAIM[claim_id],
            minimum_repetitions=PHASE3_MIN_REPETITIONS[claim_id],
            config_input=_CLAIM_CONFIG_INPUTS[claim_id],
            manifest_command=PHASE3_MANIFEST_COMMANDS[claim_id],
        )
        for claim_id, test_id in PHASE3_REQUIRED_TEST_IDS.items()
    ),
    require_cuda_environment=True,
    schema_version=SCHEMA_VERSION,
)

Phase3EvidenceError = PhaseEvidenceError


def capture_phase3_evidence(root: Path) -> dict[str, Any]:
    """Run every registered A/B/C command from a clean CUDA source commit."""

    return capture_phase_evidence(PHASE3_SPEC, root)


def load_phase3_evidence(path: Path) -> dict[str, Any]:
    """Load one Phase 3 gate artifact."""

    return load_phase_evidence(path)


def validate_phase3_evidence(report: Mapping[str, Any], *, root: Path) -> tuple[str, ...]:
    """Validate raw evidence and the independent Phase 3 manifest mapping."""

    return validate_phase_evidence(PHASE3_SPEC, report, root=root)


def write_phase3_evidence(report: Mapping[str, Any], output: Path) -> None:
    """Persist a previously validated Phase 3 capture."""

    write_phase_evidence(report, output)


__all__ = [
    "ARTIFACT_KIND",
    "ISSUE",
    "MANIFEST_PATH",
    "PHASE",
    "PHASE3_COMMANDS",
    "PHASE3_MANIFEST_COMMANDS",
    "PHASE3_MIN_REPETITIONS",
    "PHASE3_REQUIRED_TEST_IDS",
    "PHASE3_SPEC",
    "Phase3EvidenceError",
    "capture_phase3_evidence",
    "load_phase3_evidence",
    "sha256_file",
    "validate_phase3_evidence",
    "write_phase3_evidence",
]
