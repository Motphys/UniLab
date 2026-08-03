"""Synthetic fault coverage for the managed MuJoCo/MJWarp rollout Phase 3 evidence validator."""

from __future__ import annotations

import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
import tooling.acceptance.phase_evidence as generic_evidence
from tooling.acceptance.phase3 import (
    ARTIFACT_KIND,
    ISSUE,
    PHASE,
    PHASE3_COMMANDS,
    PHASE3_MIN_REPETITIONS,
    PHASE3_REQUIRED_TEST_IDS,
    PHASE3_SPEC,
    Phase3EvidenceError,
    capture_phase3_evidence,
    sha256_file,
    validate_phase3_evidence,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _head() -> str:
    return subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def _input_paths() -> tuple[Path, ...]:
    return PHASE3_SPEC.input_files


def _config_input(claim_id: str) -> Path:
    return next(claim.config_input for claim in PHASE3_SPEC.claims if claim.claim_id == claim_id)


def _valid_report() -> dict[str, Any]:
    command_rows: list[dict[str, Any]] = []
    for command in PHASE3_COMMANDS:
        for repetition in range(1, command.repetitions + 1):
            command_rows.append(
                {
                    "name": f"{command.name}#{repetition}",
                    "series": command.name,
                    "lane": command.lane,
                    "repetition": repetition,
                    "argv": list(command.argv),
                    "required_test_ids": list(command.required_test_ids),
                    "exit_code": 0,
                    "duration_sec": 0.1,
                    "pytest": {"passed": 1, "skipped": 0, "xfailed": 0, "xpassed": 0},
                    "stdout": "\n".join((*command.required_test_ids, "1 passed")),
                    "stderr": "",
                }
            )
    return {
        "schema_version": 1,
        "kind": ARTIFACT_KIND,
        "issue": ISSUE,
        "phase": PHASE,
        "generated_at_utc": "2026-07-28T00:00:00+00:00",
        "source": {"commit_sha": _head(), "branch": "test", "tree_clean": True},
        "inputs": {
            "files": {path.as_posix(): sha256_file(REPO_ROOT / path) for path in _input_paths()},
            "claim_config_hashes": {
                claim_id: sha256_file(REPO_ROOT / _config_input(claim_id))
                for claim_id in PHASE3_REQUIRED_TEST_IDS
            },
        },
        "environment": {
            "platform": "Linux",
            "python": "test",
            "warp_device": "cuda:0",
            "nvidia_smi": ["test-gpu"],
            "packages": {"mujoco": "3.10", "mujoco-warp": "3.10", "warp-lang": "1.14"},
        },
        "claims": [
            {
                "claim_id": claim_id,
                "required_test_id": test_id,
                "command": next(
                    command.name
                    for command in PHASE3_COMMANDS
                    if test_id in command.required_test_ids
                ),
                "minimum_repetitions": PHASE3_MIN_REPETITIONS[claim_id],
                "config_input": _config_input(claim_id).as_posix(),
            }
            for claim_id, test_id in PHASE3_REQUIRED_TEST_IDS.items()
        ],
        "commands": command_rows,
    }


def _capture_fixture_root(tmp_path: Path) -> Path:
    root = tmp_path / "capture"
    for path in _input_paths():
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(f"fixture for {path}\n", encoding="utf-8")
    subprocess.run(("git", "init", "--quiet"), cwd=root, check=True)
    subprocess.run(("git", "add", "."), cwd=root, check=True)
    subprocess.run(
        (
            "git",
            "-c",
            "user.email=manager_mjwarp-test@example.invalid",
            "-c",
            "user.name=Issue705 Test",
            "commit",
            "--quiet",
            "-m",
            "fixture",
        ),
        cwd=root,
        check=True,
    )
    return root


def test_phase3_evidence_validator_accepts_complete_registered_report() -> None:
    assert validate_phase3_evidence(_valid_report(), root=REPO_ROOT) == ()


def test_phase3_evidence_validator_fails_closed_for_provenance_and_execution_faults() -> None:
    report = _valid_report()

    altered_input = deepcopy(report)
    altered_input["inputs"]["files"]["uv.lock"] = f"sha256:{'0' * 64}"
    assert any(
        "does not match current input" in error
        for error in validate_phase3_evidence(altered_input, root=REPO_ROOT)
    )

    missing_cuda_repetition = deepcopy(report)
    missing_cuda_repetition["commands"] = [
        command
        for command in missing_cuda_repetition["commands"]
        if command["name"] != "lane_c_cross_backend#2"
    ]
    assert any(
        "lane_c_cross_backend" in error and "expected repetitions" in error
        for error in validate_phase3_evidence(missing_cuda_repetition, root=REPO_ROOT)
    )

    skipped = deepcopy(report)
    skipped["commands"][0]["pytest"]["skipped"] = 1
    assert any(
        "skipped: expected 0" in error
        for error in validate_phase3_evidence(skipped, root=REPO_ROOT)
    )

    stale = deepcopy(report)
    stale["source"]["commit_sha"] = "0" * 40
    assert any("ancestor" in error for error in validate_phase3_evidence(stale, root=REPO_ROOT))

    bad_device = deepcopy(report)
    bad_device["environment"]["warp_device"] = "cpu"
    assert any(
        "expected CUDA device" in error
        for error in validate_phase3_evidence(bad_device, root=REPO_ROOT)
    )

    altered_argv = deepcopy(report)
    altered_argv["commands"][0]["argv"] = ["echo", "not-the-registered-command"]
    assert any(
        "does not match registered command" in error
        for error in validate_phase3_evidence(altered_argv, root=REPO_ROOT)
    )


def test_capture_rejects_registered_input_mutated_by_evidence_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _capture_fixture_root(tmp_path)

    def fake_cuda_environment(_: object, *, root: Path) -> dict[str, Any]:
        del root
        return {
            "platform": "Linux",
            "python": "test",
            "warp_device": "cuda:0",
            "nvidia_smi": ["test-gpu"],
            "packages": {"mujoco": "3.10", "mujoco-warp": "3.10", "warp-lang": "1.14"},
        }

    def fake_evidence_command(command: Any, *, root: Path, repetition: int) -> dict[str, Any]:
        (root / "uv.lock").write_text("mutated during capture\n", encoding="utf-8")
        return {
            "name": f"{command.name}#{repetition}",
            "series": command.name,
            "lane": command.lane,
            "repetition": repetition,
            "argv": list(command.argv),
            "required_test_ids": list(command.required_test_ids),
            "exit_code": 0,
            "duration_sec": 0.1,
            "pytest": {"passed": 1, "skipped": 0, "xfailed": 0, "xpassed": 0},
            "stdout": "\n".join((*command.required_test_ids, "1 passed")),
            "stderr": "",
        }

    monkeypatch.setattr(generic_evidence, "_capture_environment", fake_cuda_environment)
    monkeypatch.setattr(generic_evidence, "_run_evidence_command", fake_evidence_command)

    with pytest.raises(Phase3EvidenceError, match="clean source"):
        capture_phase3_evidence(root)


def test_phase3_claim_mapping_and_freshness_inputs_cover_implementation() -> None:
    assert {claim.claim_id for claim in PHASE3_SPEC.claims} == set(PHASE3_REQUIRED_TEST_IDS)
    for path in (
        Path("src/unilab/base/backend/base.py"),
        Path("src/unilab/envs/locomotion/g1/managed_reference.py"),
        Path("src/unilab/manager/runtime.py"),
        Path("src/unilab/training/sim2sim.py"),
    ):
        assert path in PHASE3_SPEC.input_files
