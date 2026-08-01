"""Synthetic fault coverage for the Issue #705 Phase 4 evidence gate."""

from __future__ import annotations

import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Any

from unilab.tools.issue705_phase4_evidence import (
    ARTIFACT_KIND,
    ISSUE,
    PHASE,
    PHASE4_COMMANDS,
    PHASE4_REQUIRED_TEST_IDS,
    PHASE4_SPEC,
    load_host_benchmark_artifact,
    sha256_file,
    validate_host_benchmark_payload,
    validate_phase4_evidence,
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
    for command in PHASE4_COMMANDS:
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
        "schema_version": PHASE4_SPEC.schema_version,
        "kind": ARTIFACT_KIND,
        "issue": ISSUE,
        "phase": PHASE,
        "generated_at_utc": "2026-07-29T00:00:00+00:00",
        "source": {"commit_sha": _head(), "branch": "test", "tree_clean": True},
        "inputs": {
            "files": {
                path.as_posix(): sha256_file(REPO_ROOT / path) for path in PHASE4_SPEC.input_files
            },
            "claim_config_hashes": {
                claim.claim_id: sha256_file(REPO_ROOT / claim.config_input)
                for claim in PHASE4_SPEC.claims
            },
        },
        "environment": {
            "platform": "Linux",
            "python": "test",
            "packages": {"mujoco": "3.10", "numba": "0.63"},
        },
        "claims": [
            {
                "claim_id": claim.claim_id,
                "required_test_id": claim.required_test_id,
                "command": claim.command_name,
                "minimum_repetitions": claim.minimum_repetitions,
                "config_input": claim.config_input.as_posix(),
            }
            for claim in PHASE4_SPEC.claims
        ],
        "commands": command_rows,
    }


def test_phase4_evidence_validator_accepts_complete_registered_report() -> None:
    assert validate_phase4_evidence(_valid_report(), root=REPO_ROOT) == ()


def test_phase4_evidence_validator_fails_closed_for_provenance_and_execution_faults() -> None:
    report = _valid_report()

    altered_input = deepcopy(report)
    altered_input["inputs"]["files"]["uv.lock"] = f"sha256:{'0' * 64}"
    assert any(
        "does not match current input" in error
        for error in validate_phase4_evidence(altered_input, root=REPO_ROOT)
    )

    missing_repetition = deepcopy(report)
    missing_repetition["commands"] = [
        command
        for command in missing_repetition["commands"]
        if command["name"] != "lane_b_allocation_stability#3"
    ]
    assert any(
        "allocation_stability" in error and "expected repetitions" in error
        for error in validate_phase4_evidence(missing_repetition, root=REPO_ROOT)
    )

    skipped = deepcopy(report)
    skipped["commands"][0]["pytest"]["skipped"] = 1
    assert any(
        "skipped: expected 0" in error
        for error in validate_phase4_evidence(skipped, root=REPO_ROOT)
    )

    stale = deepcopy(report)
    stale["source"]["commit_sha"] = "0" * 40
    assert any("ancestor" in error for error in validate_phase4_evidence(stale, root=REPO_ROOT))


def test_phase4_host_artifact_rejects_gate_matrix_and_candidate_tampering() -> None:
    artifact = load_host_benchmark_artifact(REPO_ROOT)
    assert validate_host_benchmark_payload(artifact, root=REPO_ROOT) == ()

    gate_tamper = deepcopy(artifact)
    gate_tamper["gate"]["passed"] = False
    assert any(
        "does not recompute" in error
        for error in validate_host_benchmark_payload(gate_tamper, root=REPO_ROOT)
    )

    missing_case = deepcopy(artifact)
    missing_case["cases"].pop()
    assert any(
        "incomplete" in error and "process matrix" in error
        for error in validate_host_benchmark_payload(missing_case, root=REPO_ROOT)
    )

    summary_tamper = deepcopy(artifact)
    summary_tamper["cases"][0]["summary"]["throughput_env_steps_per_sec"] *= 1.0 + 1e-12
    assert any(
        "summary: does not recompute" in error
        for error in validate_host_benchmark_payload(summary_tamper, root=REPO_ROOT)
    )

    candidate_tamper = deepcopy(artifact)
    candidate_tamper["candidate"]["candidate_commit"] = "0" * 40
    assert any(
        "candidate_commit" in error or "candidate source" in error
        for error in validate_host_benchmark_payload(candidate_tamper, root=REPO_ROOT)
    )


def test_phase4_claim_mapping_is_exact() -> None:
    assert {claim.claim_id for claim in PHASE4_SPEC.claims} == set(PHASE4_REQUIRED_TEST_IDS)
