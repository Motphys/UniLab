"""Fail-closed evidence contract for the managed MuJoCo/MJWarp rollout Phase 5 gate.

The committed PPO artifact remains the owner of end-to-end performance,
behavior, memory, transfer, and profiler evidence.  This module combines its
independent validator with the registered real-CUDA contract matrix used to
promote every Phase 5 claim.
"""

from __future__ import annotations

import json
import sys
from importlib import import_module
from pathlib import Path
from typing import Any, Mapping, cast

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

ISSUE = 705
PHASE = 5
ARTIFACT_KIND = "manager_mjwarp-phase5-gate-v1"
MANIFEST_PATH = Path("tests/acceptance/manager_mjwarp/manifests/phase_5.yaml")
MJWARP_OWNER = Path("conf/ppo/task/g1_walk_flat/mjwarp.yaml")
PPO_ARTIFACT = Path("tests/acceptance/manager_mjwarp/artifacts/phase_5_mjwarp_ppo.json")
PPO_TRACE = Path("tests/acceptance/manager_mjwarp/artifacts/phase_5_mjwarp_ppo_trace.json")
ROOT_DIR = Path(__file__).resolve().parents[2]

PHASE5_REQUIRED_TEST_IDS: dict[str, str] = {
    "P5-DEVICE-ABI": (
        "tests/training/test_device_transition_abi.py::test_dlpack_pointer_shape_dtype_and_lifetime"
    ),
    "P5-STREAM-LIFETIME": (
        "tests/training/test_device_stream_contract.py::test_missing_completion_event_is_detected"
    ),
    "P5-DEVICE-LIFECYCLE": (
        "tests/training/test_device_lifecycle.py::test_device_adapter_matches_host_terminal_contract"
    ),
    "P5-NO-HOST-ROUNDTRIP": (
        "tests/training/test_device_transfer_budget.py::test_rollout_has_no_per_step_host_roundtrip"
    ),
    "P5-GRAPH-CONTRACT": (
        "tests/base/test_mjwarp_graph_contract.py::test_graph_key_change_recaptures_or_fails_closed"
    ),
    "P5-TRAIN-PERFORMANCE": (
        "tests/benchmark/test_mjwarp_ppo_benchmark.py::test_device_profile_meets_end_to_end_gate"
    ),
    "P5-DEVICE-STABILITY": (
        "tests/training/test_device_runtime_stability.py::"
        "test_long_rollout_memory_and_addresses_are_stable"
    ),
}

PHASE5_MIN_REPETITIONS: dict[str, int] = {
    "P5-DEVICE-ABI": 3,
    "P5-STREAM-LIFETIME": 10,
    "P5-DEVICE-LIFECYCLE": 3,
    "P5-NO-HOST-ROUNDTRIP": 3,
    "P5-GRAPH-CONTRACT": 3,
    "P5-TRAIN-PERFORMANCE": 5,
    "P5-DEVICE-STABILITY": 3,
}

PHASE5_MANIFEST_COMMANDS: dict[str, str] = {
    "P5-DEVICE-ABI": (
        "uv run --with mujoco-warp --with warp-lang pytest -m slow "
        "tests/training/test_device_transition_abi.py -v"
    ),
    "P5-STREAM-LIFETIME": (
        "uv run --with mujoco-warp --with warp-lang pytest -m slow "
        "tests/training/test_device_stream_contract.py -v"
    ),
    "P5-DEVICE-LIFECYCLE": (
        "uv run --with mujoco-warp --with warp-lang pytest -m slow "
        "tests/training/test_device_lifecycle.py -v"
    ),
    "P5-NO-HOST-ROUNDTRIP": (
        "uv run --with mujoco-warp --with warp-lang pytest -m slow "
        "tests/training/test_device_transfer_budget.py -v"
    ),
    "P5-GRAPH-CONTRACT": (
        "uv run --with mujoco-warp --with warp-lang pytest -m slow "
        "tests/base/test_mjwarp_graph_contract.py -v"
    ),
    "P5-TRAIN-PERFORMANCE": (
        "uv run --with mujoco-warp --with warp-lang benchmark/rl/benchmark_mjwarp_ppo.py"
    ),
    "P5-DEVICE-STABILITY": (
        "uv run --with mujoco-warp --with warp-lang pytest -m slow "
        "tests/training/test_device_runtime_stability.py -v"
    ),
}


def _cuda_command(name: str, claim_id: str) -> PhaseEvidenceCommand:
    return PhaseEvidenceCommand(
        name=name,
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
            PHASE5_REQUIRED_TEST_IDS[claim_id],
            "-v",
            "-rsxX",
        ),
        required_test_ids=(PHASE5_REQUIRED_TEST_IDS[claim_id],),
        repetitions=PHASE5_MIN_REPETITIONS[claim_id],
    )


PHASE5_COMMANDS = (
    _cuda_command("lane_c_device_abi", "P5-DEVICE-ABI"),
    _cuda_command("lane_c_stream_lifetime", "P5-STREAM-LIFETIME"),
    _cuda_command("lane_c_device_lifecycle", "P5-DEVICE-LIFECYCLE"),
    _cuda_command("lane_c_transfer_budget", "P5-NO-HOST-ROUNDTRIP"),
    _cuda_command("lane_c_graph_contract", "P5-GRAPH-CONTRACT"),
    PhaseEvidenceCommand(
        name="lane_d_train_performance",
        lane="D",
        argv=(
            "uv",
            "run",
            "pytest",
            PHASE5_REQUIRED_TEST_IDS["P5-TRAIN-PERFORMANCE"],
            "-v",
            "-rsxX",
        ),
        required_test_ids=(PHASE5_REQUIRED_TEST_IDS["P5-TRAIN-PERFORMANCE"],),
        repetitions=PHASE5_MIN_REPETITIONS["P5-TRAIN-PERFORMANCE"],
    ),
    _cuda_command("lane_c_device_stability", "P5-DEVICE-STABILITY"),
)

_COMMAND_BY_CLAIM = {
    "P5-DEVICE-ABI": "lane_c_device_abi",
    "P5-STREAM-LIFETIME": "lane_c_stream_lifetime",
    "P5-DEVICE-LIFECYCLE": "lane_c_device_lifecycle",
    "P5-NO-HOST-ROUNDTRIP": "lane_c_transfer_budget",
    "P5-GRAPH-CONTRACT": "lane_c_graph_contract",
    "P5-TRAIN-PERFORMANCE": "lane_d_train_performance",
    "P5-DEVICE-STABILITY": "lane_c_device_stability",
}


def _ppo_benchmark_module(root: Path) -> Any:
    root_text = str(root.resolve())
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    return import_module("benchmark.rl.benchmark_mjwarp_ppo")


def _expanded_ppo_source_inputs(root: Path) -> tuple[Path, ...]:
    """Expand benchmark source directories into tracked-code freshness inputs."""

    benchmark = _ppo_benchmark_module(root)
    inputs: set[Path] = set()
    for raw_path in benchmark.SOURCE_INPUTS:
        relative = Path(raw_path)
        absolute = root / relative
        if absolute.is_dir():
            inputs.update(
                child.relative_to(root)
                for child in absolute.rglob("*.py")
                if child.is_file() and "__pycache__" not in child.parts
            )
        else:
            inputs.add(relative)
    return tuple(sorted(inputs, key=Path.as_posix))


_EVIDENCE_INPUTS = (
    PPO_ARTIFACT,
    PPO_TRACE,
    Path("tests/acceptance/manager_mjwarp/claim_test_inventory.yaml"),
    Path("tooling/acceptance/phase_evidence.py"),
    Path("tooling/acceptance/phase5.py"),
    Path("scripts/capture_acceptance.py"),
    Path("tests/tools/test_manager_mjwarp_phase5_evidence.py"),
    Path("tests/acceptance/manager_mjwarp/test_phase5_evidence.py"),
)

_INPUT_FILES = tuple(
    sorted(
        {*_expanded_ppo_source_inputs(ROOT_DIR), *_EVIDENCE_INPUTS},
        key=Path.as_posix,
    )
)

PHASE5_SPEC = PhaseEvidenceSpec(
    issue=ISSUE,
    phase=PHASE,
    artifact_kind=ARTIFACT_KIND,
    manifest_path=MANIFEST_PATH,
    required_lanes=("C", "D"),
    input_files=_INPUT_FILES,
    package_names=("torch", "mujoco-warp", "warp-lang"),
    commands=PHASE5_COMMANDS,
    claims=tuple(
        PhaseEvidenceClaim(
            claim_id=claim_id,
            required_test_id=test_id,
            command_name=_COMMAND_BY_CLAIM[claim_id],
            minimum_repetitions=PHASE5_MIN_REPETITIONS[claim_id],
            config_input=MJWARP_OWNER,
            manifest_command=PHASE5_MANIFEST_COMMANDS[claim_id],
        )
        for claim_id, test_id in PHASE5_REQUIRED_TEST_IDS.items()
    ),
)


def load_ppo_benchmark_artifact(root: Path) -> dict[str, Any]:
    """Load the committed raw PPO artifact as untrusted JSON."""

    path = root.resolve() / PPO_ARTIFACT
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PhaseEvidenceError(f"cannot load Phase 5 PPO artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PhaseEvidenceError("Phase 5 PPO artifact must contain a JSON object")
    return value


def validate_ppo_benchmark_payload(
    artifact: object,
    *,
    root: Path,
    artifact_path: Path | None = None,
    validate_live_hardware: bool = False,
) -> tuple[str, ...]:
    """Recompute the PPO contract, optionally probing live hardware provenance."""

    root = root.resolve()
    try:
        benchmark = _ppo_benchmark_module(root)
        if root != benchmark.ROOT_DIR.resolve():
            return ("PPO benchmark validator must run from the UniLab repository root",)
        binding = benchmark.load_binding()
        path = artifact_path or root / PPO_ARTIFACT
        return cast(
            tuple[str, ...],
            benchmark.validate_artifact(
                artifact,
                binding=binding,
                repo_root=root,
                artifact_path=path,
                validate_live_hardware=validate_live_hardware,
            ),
        )
    except (OSError, ValueError, RuntimeError) as exc:
        return (f"PPO benchmark contract could not be validated: {type(exc).__name__}: {exc}",)


def validate_ppo_benchmark_artifact(
    *, root: Path, validate_live_hardware: bool = False
) -> tuple[str, ...]:
    """Load and validate the committed raw PPO artifact and profiler sibling."""

    try:
        artifact = load_ppo_benchmark_artifact(root)
    except PhaseEvidenceError as exc:
        return (str(exc),)
    return validate_ppo_benchmark_payload(
        artifact,
        root=root,
        validate_live_hardware=validate_live_hardware,
    )


def capture_phase5_evidence(root: Path) -> dict[str, Any]:
    """Run the registered C/D matrix after validating the complete PPO artifact."""

    ppo_errors = validate_ppo_benchmark_artifact(root=root, validate_live_hardware=True)
    if ppo_errors:
        raise PhaseEvidenceError(
            "Phase 5 PPO artifact is not valid:\n" + "\n".join(f"- {error}" for error in ppo_errors)
        )
    report = capture_phase_evidence(PHASE5_SPEC, root)
    errors = validate_phase5_evidence(report, root=root)
    if errors:
        raise PhaseEvidenceError(
            "captured Phase 5 evidence failed validation:\n"
            + "\n".join(f"- {error}" for error in errors)
        )
    return report


def load_phase5_evidence(path: Path) -> dict[str, Any]:
    """Load one Phase 5 gate artifact."""

    return load_phase_evidence(path)


def validate_phase5_evidence(report: Mapping[str, Any], *, root: Path) -> tuple[str, ...]:
    """Validate command evidence and independently recompute the strict PPO gate."""

    errors = list(validate_phase_evidence(PHASE5_SPEC, report, root=root))
    errors.extend(f"PPO benchmark: {error}" for error in validate_ppo_benchmark_artifact(root=root))
    return tuple(errors)


def write_phase5_evidence(report: Mapping[str, Any], output: Path) -> None:
    """Persist a previously validated Phase 5 capture."""

    write_phase_evidence(report, output)


__all__ = [
    "ARTIFACT_KIND",
    "ISSUE",
    "MANIFEST_PATH",
    "PHASE",
    "PHASE5_COMMANDS",
    "PHASE5_MIN_REPETITIONS",
    "PHASE5_REQUIRED_TEST_IDS",
    "PHASE5_SPEC",
    "PPO_ARTIFACT",
    "PPO_TRACE",
    "PhaseEvidenceError",
    "capture_phase5_evidence",
    "load_phase5_evidence",
    "load_ppo_benchmark_artifact",
    "sha256_file",
    "validate_phase5_evidence",
    "validate_ppo_benchmark_artifact",
    "validate_ppo_benchmark_payload",
    "write_phase5_evidence",
]
