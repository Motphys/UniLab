"""Synthetic fault coverage for the Issue #705 Phase 2 evidence validator."""

from __future__ import annotations

import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Any

from unilab.tools.issue705_phase2_evidence import (
    ARTIFACT_KIND,
    ISSUE,
    PHASE,
    PHASE2_COMMANDS,
    PHASE2_REQUIRED_TEST_IDS,
    PHASE2_SPEC,
    sha256_file,
    validate_phase2_evidence,
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
    commands = [
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
        for command in PHASE2_COMMANDS
        for repetition in range(1, command.repetitions + 1)
    ]
    return {
        "schema_version": PHASE2_SPEC.schema_version,
        "kind": ARTIFACT_KIND,
        "issue": ISSUE,
        "phase": PHASE,
        "generated_at_utc": "2026-08-03T00:00:00+00:00",
        "source": {"commit_sha": _head(), "branch": "test", "tree_clean": True},
        "inputs": {
            "files": {
                path.as_posix(): sha256_file(REPO_ROOT / path) for path in PHASE2_SPEC.input_files
            },
            "claim_config_hashes": {
                claim.claim_id: sha256_file(REPO_ROOT / claim.config_input)
                for claim in PHASE2_SPEC.claims
            },
        },
        "environment": {
            "platform": "Linux",
            "python": "test",
            "warp_device": "cuda:0",
            "nvidia_smi": ["test-gpu"],
            "packages": {
                "mujoco": "3.10",
                "mujoco-warp": "3.10",
                "warp-lang": "1.15",
                "torch": "2.7",
                "rsl-rl-lib": "5.0",
            },
        },
        "claims": [
            {
                "claim_id": claim.claim_id,
                "required_test_id": claim.required_test_id,
                "command": claim.command_name,
                "minimum_repetitions": claim.minimum_repetitions,
                "config_input": claim.config_input.as_posix(),
            }
            for claim in PHASE2_SPEC.claims
        ],
        "commands": commands,
    }


def test_phase2_evidence_validator_accepts_complete_registered_report() -> None:
    assert validate_phase2_evidence(_valid_report(), root=REPO_ROOT) == ()


def test_phase2_evidence_validator_fails_closed_for_provenance_and_execution_faults() -> None:
    report = _valid_report()

    skipped = deepcopy(report)
    skipped["commands"][0]["pytest"]["skipped"] = 1
    assert any(
        "skipped: expected 0" in error
        for error in validate_phase2_evidence(skipped, root=REPO_ROOT)
    )

    missing_repetition = deepcopy(report)
    missing_repetition["commands"] = [
        command
        for command in missing_repetition["commands"]
        if command["name"] != "lane_c_production_cuda#3"
    ]
    assert any(
        "lane_c_production_cuda" in error and "expected repetitions" in error
        for error in validate_phase2_evidence(missing_repetition, root=REPO_ROOT)
    )

    stale = deepcopy(report)
    stale["source"]["commit_sha"] = "0" * 40
    assert any("ancestor" in error for error in validate_phase2_evidence(stale, root=REPO_ROOT))

    bad_device = deepcopy(report)
    bad_device["environment"]["warp_device"] = "cpu"
    assert any(
        "expected CUDA device" in error
        for error in validate_phase2_evidence(bad_device, root=REPO_ROOT)
    )


def test_phase2_claim_mapping_and_freshness_inputs_cover_implementation() -> None:
    assert {claim.claim_id for claim in PHASE2_SPEC.claims} == set(PHASE2_REQUIRED_TEST_IDS)
    for path in (
        Path("src/unilab/base/backend/mjwarp/backend.py"),
        Path("src/unilab/dr/manager.py"),
        Path("src/unilab/training/rsl_rl.py"),
    ):
        assert path in PHASE2_SPEC.input_files
