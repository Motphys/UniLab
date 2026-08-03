"""Fail-closed evidence contract for the managed MuJoCo/MJWarp rollout Phase 2 gate."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from tooling.acceptance.phase_evidence import (
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
PHASE = 2
SCHEMA_VERSION = 1
ARTIFACT_KIND = "manager_mjwarp-phase2-gate-v1"
MANIFEST_PATH = Path("tests/acceptance/manager_mjwarp/manifests/phase_2.yaml")
OWNER_YAML = Path("conf/ppo/task/g1_walk_flat/mjwarp.yaml")
ROOT_DIR = Path(__file__).resolve().parents[2]

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
        "tests/dr/test_mjwarp_g1_dr.py::"
        "test_g1_kp_kd_owner_semantics_have_physics_effect_or_are_disabled"
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

PHASE2_MANIFEST_COMMANDS: dict[str, str] = {
    "P2-BACKEND-IDENTITY": (
        "uv run pytest tests/base/test_backend_imports.py tests/base/test_mjwarp_identity.py -v"
    ),
    "P2-GPU-CORRECTNESS": (
        "uv run --with mujoco-warp --with warp-lang pytest "
        "tests/base/test_mjwarp_backend.py::test_real_cuda_init_reset_step -v"
    ),
    "P2-RESET-ISOLATION": (
        "uv run --with mujoco-warp --with warp-lang pytest "
        "tests/base/test_mjwarp_backend.py::test_selected_row_reset_isolated -v"
    ),
    "P2-TRAJECTORY-DIFFERENTIAL": (
        "uv run --with mujoco-warp --with warp-lang pytest "
        "tests/base/test_mjwarp_differential.py -v"
    ),
    "P2-DR-OWNER-SEMANTICS": (
        "uv run --with mujoco-warp --with warp-lang pytest tests/dr/test_mjwarp_g1_dr.py -v"
    ),
    "P2-TRANSFER-ACCOUNTING": (
        "uv run --with mujoco-warp --with warp-lang pytest tests/base/test_mjwarp_transfers.py -v"
    ),
    "P2-UNSUPPORTED-FAIL-CLOSED": (
        "uv run --with mujoco-warp --with warp-lang pytest "
        "tests/base/test_mjwarp_capabilities.py -v"
    ),
    "P2-TRAIN-LIVENESS": (
        "uv run --with mujoco-warp --with warp-lang pytest "
        "tests/integration/test_mjwarp_train_smoke.py -v"
    ),
}

PHASE2_COMMANDS = (
    PhaseEvidenceCommand(
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
    PhaseEvidenceCommand(
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
    PhaseEvidenceCommand(
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

_COMMAND_BY_CLAIM = {
    claim_id: command.name
    for command in PHASE2_COMMANDS
    for claim_id, test_id in PHASE2_REQUIRED_TEST_IDS.items()
    if test_id in command.required_test_ids
}
_SOURCE_INPUT_TREES = (
    Path("src/unilab/base/backend"),
    Path("src/unilab/dr"),
    Path("src/unilab/training"),
)
_STATIC_INPUTS = (
    Path("uv.lock"),
    OWNER_YAML,
    Path("src/unilab/cli.py"),
    Path("src/unilab/structured_configs.py"),
    Path("scripts/train_rsl_rl.py"),
    Path("tooling/acceptance/phase_evidence.py"),
    Path("tooling/acceptance/phase2.py"),
    Path("scripts/capture_acceptance.py"),
    Path("tests/tools/test_manager_mjwarp_phase2_evidence.py"),
    Path("tests/acceptance/manager_mjwarp/test_phase2_evidence.py"),
    Path("tests/base/test_backend_imports.py"),
    Path("tests/base/test_mjwarp_identity.py"),
    Path("tests/base/test_mjwarp_backend.py"),
    Path("tests/base/test_mjwarp_differential.py"),
    Path("tests/dr/test_mjwarp_g1_dr.py"),
    Path("tests/base/test_mjwarp_transfers.py"),
    Path("tests/base/test_mjwarp_capabilities.py"),
    Path("tests/integration/test_mjwarp_train_smoke.py"),
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

PHASE2_SPEC = PhaseEvidenceSpec(
    issue=ISSUE,
    phase=PHASE,
    artifact_kind=ARTIFACT_KIND,
    manifest_path=MANIFEST_PATH,
    required_lanes=("A", "C"),
    input_files=_INPUT_FILES,
    package_names=("mujoco", "mujoco-warp", "warp-lang", "torch", "rsl-rl-lib"),
    commands=PHASE2_COMMANDS,
    claims=tuple(
        PhaseEvidenceClaim(
            claim_id=claim_id,
            required_test_id=test_id,
            command_name=_COMMAND_BY_CLAIM[claim_id],
            minimum_repetitions=PHASE2_MIN_REPETITIONS[claim_id],
            config_input=OWNER_YAML,
            manifest_command=PHASE2_MANIFEST_COMMANDS[claim_id],
        )
        for claim_id, test_id in PHASE2_REQUIRED_TEST_IDS.items()
    ),
    require_cuda_environment=True,
    schema_version=SCHEMA_VERSION,
)

Phase2EvidenceError = PhaseEvidenceError


def capture_phase2_evidence(root: Path) -> dict[str, Any]:
    """Run every registered A/C command from a clean CUDA source commit."""

    return capture_phase_evidence(PHASE2_SPEC, root)


def load_phase2_evidence(path: Path) -> dict[str, Any]:
    """Load one Phase 2 gate artifact."""

    return load_phase_evidence(path)


def validate_phase2_evidence(report: Mapping[str, Any], *, root: Path) -> tuple[str, ...]:
    """Validate raw evidence and the independent Phase 2 manifest mapping."""

    return validate_phase_evidence(PHASE2_SPEC, report, root=root)


def write_phase2_evidence(report: Mapping[str, Any], output: Path) -> None:
    """Persist a previously validated Phase 2 capture."""

    write_phase_evidence(report, output)


__all__ = [
    "ARTIFACT_KIND",
    "ISSUE",
    "MANIFEST_PATH",
    "OWNER_YAML",
    "PHASE",
    "PHASE2_COMMANDS",
    "PHASE2_MANIFEST_COMMANDS",
    "PHASE2_MIN_REPETITIONS",
    "PHASE2_REQUIRED_TEST_IDS",
    "PHASE2_SPEC",
    "Phase2EvidenceError",
    "capture_phase2_evidence",
    "load_phase2_evidence",
    "sha256_file",
    "validate_phase2_evidence",
    "write_phase2_evidence",
]
