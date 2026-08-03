"""Fail-closed evidence contract for the managed MuJoCo/MJWarp rollout Phase 6 gate.

Phase 6 combines the registered CUDA contract matrix with the frozen Issue
#829 DR performance artifact.  The latter is treated as untrusted input: its
300 process receipts, aggregates, provenance, and gates are independently
recomputed before a capture can be produced or accepted.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from tooling.acceptance.phase_evidence import (
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
from unilab.tools.mjwarp_dr_performance import (
    DEFAULT_ARTIFACT_PATH,
    FREEZE_RECEIPT_PATH,
    PLAN_PATH,
    MjwarpDrPerformanceContractError,
    load_mjwarp_dr_performance_freeze_receipt,
    load_mjwarp_dr_performance_plan,
    validate_mjwarp_dr_performance_artifact,
)

ISSUE = 705
PHASE = 6
ARTIFACT_KIND = "manager_mjwarp-phase6-gate-v1"
MANIFEST_PATH = Path("tests/acceptance/manager_mjwarp/manifests/phase_6.yaml")
MJWARP_OWNER = Path("conf/ppo/task/g1_walk_flat/mjwarp.yaml")
DR_PERFORMANCE_ARTIFACT = DEFAULT_ARTIFACT_PATH
DR_PERFORMANCE_PLAN = PLAN_PATH
DR_PERFORMANCE_RECEIPT = FREEZE_RECEIPT_PATH
ROOT_DIR = Path(__file__).resolve().parents[2]

PHASE6_REQUIRED_TEST_IDS: dict[str, str] = {
    "P6-CAPABILITY-BIJECTION": (
        "tests/dr/test_mjwarp_capability_matrix.py::"
        "test_advertised_capabilities_equal_mandatory_parameter_cases"
    ),
    "P6-DR-SEMANTICS": (
        "tests/dr/test_mjwarp_mutation_semantics.py::test_operations_baselines_rows_and_persistence"
    ),
    "P6-PHYSICS-EFFECT": (
        "tests/dr/test_mjwarp_physics_effect.py::test_each_supported_mutation_has_next_step_effect"
    ),
    "P6-RNG-REPRODUCIBILITY": (
        "tests/dr/test_keyed_rng.py::test_rng_is_invariant_to_row_order_and_unrelated_terms"
    ),
    "P6-GRAPH-RECAPTURE": (
        "tests/dr/test_mjwarp_graph_mutation.py::"
        "test_field_expansion_invalidates_and_recaptures_all_graph_consumers"
    ),
    "P6-CONTROLLER-CONTRACT": (
        "tests/base/test_device_controller_contract.py::"
        "test_device_controller_cadence_reads_and_host_rejection"
    ),
    "P6-DR-PERFORMANCE": (
        "tests/benchmark/test_mjwarp_dr_benchmark.py::"
        "test_dr_profiles_meet_preregistered_density_gates"
    ),
    "P6-RECOMPUTE-AGGREGATION": (
        "tests/dr/test_mjwarp_recompute.py::test_strongest_recompute_runs_once_per_barrier"
    ),
}

PHASE6_MIN_REPETITIONS: dict[str, int] = {
    "P6-CAPABILITY-BIJECTION": 1,
    "P6-DR-SEMANTICS": 3,
    "P6-PHYSICS-EFFECT": 5,
    "P6-RNG-REPRODUCIBILITY": 3,
    "P6-GRAPH-RECAPTURE": 3,
    "P6-CONTROLLER-CONTRACT": 3,
    "P6-DR-PERFORMANCE": 5,
    "P6-RECOMPUTE-AGGREGATION": 3,
}

PHASE6_MANIFEST_COMMANDS: dict[str, str] = {
    "P6-CAPABILITY-BIJECTION": (
        "uv run --with mujoco-warp --with warp-lang pytest -m slow "
        "tests/dr/test_mjwarp_capability_matrix.py -v"
    ),
    "P6-DR-SEMANTICS": (
        "uv run --with mujoco-warp --with warp-lang pytest "
        "tests/dr/test_mjwarp_mutation_semantics.py "
        "tests/dr/test_mjwarp_g1_event_dr.py::"
        "test_g1_reset_events_are_keyed_partial_and_stable -m slow -v"
    ),
    "P6-PHYSICS-EFFECT": (
        "uv run --with mujoco-warp --with warp-lang pytest "
        "tests/dr/test_mjwarp_physics_effect.py "
        "tests/dr/test_mjwarp_g1_event_dr.py::"
        "test_partial_event_changes_only_selected_world_physics_not_control -m slow -v"
    ),
    "P6-RNG-REPRODUCIBILITY": (
        "uv run --with mujoco-warp --with warp-lang pytest -m slow "
        "tests/dr/test_keyed_rng.py::test_rng_is_invariant_to_row_order_and_unrelated_terms "
        "tests/manager/test_device_event_runtime.py::"
        "test_runtime_composes_sparse_kernel_and_event_values_by_target_kind -v"
    ),
    "P6-GRAPH-RECAPTURE": (
        "uv run --with mujoco-warp --with warp-lang pytest -m slow "
        "tests/dr/test_mjwarp_graph_mutation.py -v"
    ),
    "P6-CONTROLLER-CONTRACT": (
        "uv run --with mujoco-warp --with warp-lang pytest -m slow "
        "tests/base/test_device_controller_contract.py -v"
    ),
    "P6-DR-PERFORMANCE": (
        "uv run --with mujoco-warp --with warp-lang benchmark/mjwarp/benchmark_dr_profiles.py"
    ),
    "P6-RECOMPUTE-AGGREGATION": (
        "uv run --with mujoco-warp --with warp-lang pytest -m slow "
        "tests/dr/test_mjwarp_recompute.py -v"
    ),
}

_COMMAND_BY_CLAIM = {
    "P6-CAPABILITY-BIJECTION": "lane_b_capability_bijection",
    "P6-DR-SEMANTICS": "lane_c_dr_semantics",
    "P6-PHYSICS-EFFECT": "lane_c_physics_effect",
    "P6-RNG-REPRODUCIBILITY": "lane_c_rng_reproducibility",
    "P6-GRAPH-RECAPTURE": "lane_c_graph_recapture",
    "P6-CONTROLLER-CONTRACT": "lane_c_controller_contract",
    "P6-DR-PERFORMANCE": "lane_d_dr_performance",
    "P6-RECOMPUTE-AGGREGATION": "lane_c_recompute_aggregation",
}


def _cuda_pytest_command(claim_id: str, *, lane: str, slow: bool) -> PhaseEvidenceCommand:
    argv = [
        "uv",
        "run",
        "--with",
        "mujoco-warp",
        "--with",
        "warp-lang",
        "pytest",
    ]
    if slow:
        argv.extend(("-m", "slow"))
    argv.extend((PHASE6_REQUIRED_TEST_IDS[claim_id], "-v", "-rsxX"))
    return PhaseEvidenceCommand(
        name=_COMMAND_BY_CLAIM[claim_id],
        lane=lane,
        argv=tuple(argv),
        required_test_ids=(PHASE6_REQUIRED_TEST_IDS[claim_id],),
        repetitions=PHASE6_MIN_REPETITIONS[claim_id],
    )


PHASE6_COMMANDS = (
    _cuda_pytest_command("P6-CAPABILITY-BIJECTION", lane="B", slow=True),
    _cuda_pytest_command("P6-DR-SEMANTICS", lane="C", slow=True),
    _cuda_pytest_command("P6-PHYSICS-EFFECT", lane="C", slow=True),
    _cuda_pytest_command("P6-RNG-REPRODUCIBILITY", lane="C", slow=True),
    _cuda_pytest_command("P6-GRAPH-RECAPTURE", lane="C", slow=True),
    _cuda_pytest_command("P6-CONTROLLER-CONTRACT", lane="C", slow=True),
    _cuda_pytest_command("P6-DR-PERFORMANCE", lane="D", slow=False),
    _cuda_pytest_command("P6-RECOMPUTE-AGGREGATION", lane="C", slow=True),
)

_PYTHON_INPUT_TREES = (
    Path("src/unilab/base/backend"),
    Path("src/unilab/dr"),
    Path("src/unilab/envs/locomotion/g1"),
    Path("src/unilab/manager"),
    Path("tests/dr"),
)
_ASSET_INPUT_TREES = (
    Path("src/unilab/assets/robots/g1"),
    Path("src/unilab/assets/robots/go2w"),
)
_STATIC_INPUTS = (
    Path("uv.lock"),
    Path("pyproject.toml"),
    MJWARP_OWNER,
    DR_PERFORMANCE_ARTIFACT,
    DR_PERFORMANCE_PLAN,
    DR_PERFORMANCE_RECEIPT,
    Path("tests/acceptance/manager_mjwarp/claim_test_inventory.yaml"),
    Path("benchmark/mjwarp/process_dr_evidence.py"),
    Path("benchmark/mjwarp/benchmark_dr_profiles.py"),
    Path("scripts/train_rsl_rl.py"),
    Path("src/unilab/tools/g1_baseline_provenance.py"),
    Path("tooling/acceptance/phase_evidence.py"),
    Path("tooling/acceptance/phase6.py"),
    Path("src/unilab/tools/mjwarp_dr_performance.py"),
    Path("src/unilab/training/rsl_rl_device.py"),
    Path("scripts/capture_acceptance.py"),
    Path("tests/tools/test_manager_mjwarp_phase6_evidence.py"),
    Path("tests/acceptance/manager_mjwarp/test_phase6_evidence.py"),
    Path("tests/base/test_device_controller_contract.py"),
    Path("tests/benchmark/test_mjwarp_dr_benchmark.py"),
)


def _tracked_files(root: Path, paths: tuple[Path, ...]) -> tuple[Path, ...]:
    try:
        raw = subprocess.run(
            ("git", "ls-files", "-z", "--", *(path.as_posix() for path in paths)),
            cwd=root,
            capture_output=True,
            check=True,
        ).stdout
        tracked = {Path(item.decode("utf-8")) for item in raw.split(b"\0") if item}
    except (OSError, subprocess.CalledProcessError, UnicodeDecodeError) as exc:
        raise PhaseEvidenceError(f"cannot enumerate tracked Phase 6 inputs: {exc}") from exc
    return tuple(sorted(tracked, key=Path.as_posix))


def _expanded_inputs(root: Path) -> tuple[Path, ...]:
    inputs = set(_STATIC_INPUTS)
    inputs.update(
        path for path in _tracked_files(root, _PYTHON_INPUT_TREES) if path.suffix == ".py"
    )
    inputs.update(_tracked_files(root, _ASSET_INPUT_TREES))
    return tuple(sorted(inputs, key=Path.as_posix))


PHASE6_SPEC = PhaseEvidenceSpec(
    issue=ISSUE,
    phase=PHASE,
    artifact_kind=ARTIFACT_KIND,
    manifest_path=MANIFEST_PATH,
    required_lanes=("B", "C", "D"),
    input_files=_expanded_inputs(ROOT_DIR),
    package_names=("torch", "mujoco-warp", "warp-lang"),
    commands=PHASE6_COMMANDS,
    claims=tuple(
        PhaseEvidenceClaim(
            claim_id=claim_id,
            required_test_id=test_id,
            command_name=_COMMAND_BY_CLAIM[claim_id],
            minimum_repetitions=PHASE6_MIN_REPETITIONS[claim_id],
            config_input=MJWARP_OWNER,
            manifest_command=PHASE6_MANIFEST_COMMANDS[claim_id],
        )
        for claim_id, test_id in PHASE6_REQUIRED_TEST_IDS.items()
    ),
)


def load_dr_performance_artifact(root: Path) -> dict[str, Any]:
    """Load the committed #829 artifact as untrusted JSON."""

    path = root.resolve() / DR_PERFORMANCE_ARTIFACT
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PhaseEvidenceError(
            f"cannot load Phase 6 DR performance artifact {path}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise PhaseEvidenceError("Phase 6 DR performance artifact must contain a JSON object")
    return value


def validate_dr_performance_payload(
    artifact: object,
    *,
    root: Path,
) -> tuple[str, ...]:
    """Independently validate provenance and recompute every frozen DR gate."""

    root = root.resolve()
    if not isinstance(artifact, Mapping):
        return ("DR performance artifact must contain a JSON object",)
    try:
        plan = load_mjwarp_dr_performance_plan(root / DR_PERFORMANCE_PLAN)
        receipt = load_mjwarp_dr_performance_freeze_receipt(
            root / DR_PERFORMANCE_RECEIPT,
            plan=plan,
            repo_root=root,
        )
        return cast(
            tuple[str, ...],
            validate_mjwarp_dr_performance_artifact(
                artifact,
                plan=plan,
                receipt=receipt,
                repo_root=root,
                require_passing_gate=True,
            ),
        )
    except (MjwarpDrPerformanceContractError, OSError, KeyError, TypeError, ValueError) as exc:
        return (f"DR performance contract could not be validated: {type(exc).__name__}: {exc}",)


def validate_dr_performance_artifact(*, root: Path) -> tuple[str, ...]:
    """Load and independently validate the committed #829 artifact."""

    try:
        artifact = load_dr_performance_artifact(root)
    except PhaseEvidenceError as exc:
        return (str(exc),)
    return validate_dr_performance_payload(artifact, root=root)


def capture_phase6_evidence(root: Path) -> dict[str, Any]:
    """Run all registered B/C/D commands after validating the DR artifact."""

    performance_errors = validate_dr_performance_artifact(root=root)
    if performance_errors:
        raise PhaseEvidenceError(
            "Phase 6 DR performance artifact is not valid:\n"
            + "\n".join(f"- {error}" for error in performance_errors)
        )
    report = capture_phase_evidence(PHASE6_SPEC, root)
    errors = validate_phase6_evidence(report, root=root)
    if errors:
        raise PhaseEvidenceError(
            "captured Phase 6 evidence failed validation:\n"
            + "\n".join(f"- {error}" for error in errors)
        )
    return report


def load_phase6_evidence(path: Path) -> dict[str, Any]:
    """Load one Phase 6 gate artifact."""

    return load_phase_evidence(path)


def validate_phase6_evidence(report: Mapping[str, Any], *, root: Path) -> tuple[str, ...]:
    """Validate command evidence and independently recompute the DR gate."""

    errors = list(validate_phase_evidence(PHASE6_SPEC, report, root=root))
    errors.extend(
        f"DR performance: {error}" for error in validate_dr_performance_artifact(root=root)
    )
    return tuple(errors)


def write_phase6_evidence(report: Mapping[str, Any], output: Path) -> None:
    """Persist a previously validated Phase 6 capture."""

    write_phase_evidence(report, output)


__all__ = [
    "ARTIFACT_KIND",
    "DR_PERFORMANCE_ARTIFACT",
    "DR_PERFORMANCE_PLAN",
    "DR_PERFORMANCE_RECEIPT",
    "ISSUE",
    "MANIFEST_PATH",
    "PHASE",
    "PHASE6_COMMANDS",
    "PHASE6_MIN_REPETITIONS",
    "PHASE6_REQUIRED_TEST_IDS",
    "PHASE6_SPEC",
    "PhaseEvidenceError",
    "capture_phase6_evidence",
    "load_dr_performance_artifact",
    "load_phase6_evidence",
    "sha256_file",
    "validate_dr_performance_artifact",
    "validate_dr_performance_payload",
    "validate_phase6_evidence",
    "write_phase6_evidence",
]
