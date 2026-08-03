"""Synthetic fault coverage for the managed MuJoCo/MJWarp rollout Phase 1 evidence validator."""

from __future__ import annotations

import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
import tooling.acceptance.phase_evidence as generic_evidence
from tooling.acceptance.phase1 import (
    ARTIFACT_KIND,
    ISSUE,
    PHASE,
    PHASE1_COMMANDS,
    PHASE1_MIN_REPETITIONS,
    PHASE1_REQUIRED_TEST_IDS,
    PHASE1_SPEC,
    PhaseEvidenceError,
    capture_phase1_evidence,
    sha256_file,
    validate_phase1_evidence,
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


def _valid_report() -> dict[str, Any]:
    command_rows: list[dict[str, Any]] = []
    for command in PHASE1_COMMANDS:
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
            "files": {
                path.as_posix(): sha256_file(REPO_ROOT / path) for path in PHASE1_SPEC.input_files
            },
            "claim_config_hashes": {
                claim.claim_id: sha256_file(REPO_ROOT / claim.config_input)
                for claim in PHASE1_SPEC.claims
            },
        },
        "environment": {
            "platform": "Linux",
            "python": "test",
            "packages": {"mujoco": "3.10"},
        },
        "claims": [
            {
                "claim_id": claim.claim_id,
                "required_test_id": claim.required_test_id,
                "command": claim.command_name,
                "minimum_repetitions": claim.minimum_repetitions,
                "config_input": claim.config_input.as_posix(),
            }
            for claim in PHASE1_SPEC.claims
        ],
        "commands": command_rows,
    }


def _capture_fixture_root(tmp_path: Path) -> Path:
    root = tmp_path / "capture"
    for path in PHASE1_SPEC.input_files:
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


def test_phase1_evidence_validator_accepts_complete_registered_report() -> None:
    assert validate_phase1_evidence(_valid_report(), root=REPO_ROOT) == ()


def test_phase1_evidence_validator_fails_closed_for_provenance_and_execution_faults() -> None:
    report = _valid_report()

    altered_input = deepcopy(report)
    altered_input["inputs"]["files"]["uv.lock"] = f"sha256:{'0' * 64}"
    assert any(
        "does not match current input" in error
        for error in validate_phase1_evidence(altered_input, root=REPO_ROOT)
    )

    missing_repetition = deepcopy(report)
    missing_repetition["commands"] = [
        command
        for command in missing_repetition["commands"]
        if command["name"] != "lane_b_mujoco_reference#3"
    ]
    assert any(
        "lane_b_mujoco_reference" in error and "expected repetitions" in error
        for error in validate_phase1_evidence(missing_repetition, root=REPO_ROOT)
    )

    skipped = deepcopy(report)
    skipped["commands"][0]["pytest"]["skipped"] = 1
    assert any(
        "skipped: expected 0" in error
        for error in validate_phase1_evidence(skipped, root=REPO_ROOT)
    )

    stale = deepcopy(report)
    stale["source"]["commit_sha"] = "0" * 40
    assert any("ancestor" in error for error in validate_phase1_evidence(stale, root=REPO_ROOT))

    altered_argv = deepcopy(report)
    altered_argv["commands"][0]["argv"] = ["echo", "not-the-registered-command"]
    assert any(
        "does not match registered command" in error
        for error in validate_phase1_evidence(altered_argv, root=REPO_ROOT)
    )


def test_capture_rejects_registered_input_mutated_by_evidence_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _capture_fixture_root(tmp_path)

    def fake_environment(_: object, *, root: Path) -> dict[str, Any]:
        del root
        return {"platform": "Linux", "python": "test", "packages": {"mujoco": "3.10"}}

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

    monkeypatch.setattr(generic_evidence, "_capture_environment", fake_environment)
    monkeypatch.setattr(generic_evidence, "_run_evidence_command", fake_evidence_command)

    with pytest.raises(PhaseEvidenceError, match="clean source"):
        capture_phase1_evidence(root)


def test_phase1_claim_mapping_and_freshness_inputs_cover_implementation() -> None:
    assert {claim.claim_id for claim in PHASE1_SPEC.claims} == set(PHASE1_REQUIRED_TEST_IDS)
    for path in (
        Path("src/unilab/base/backend/base.py"),
        Path("src/unilab/dr/manager.py"),
        Path("src/unilab/envs/locomotion/g1/base.py"),
    ):
        assert path in PHASE1_SPEC.input_files
