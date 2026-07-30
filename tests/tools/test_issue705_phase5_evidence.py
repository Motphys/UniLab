"""Synthetic fault coverage for the Issue #705 Phase 5 evidence gate."""

from __future__ import annotations

import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

from unilab.tools.issue705_phase5_evidence import (
    ARTIFACT_KIND,
    ISSUE,
    PHASE,
    PHASE5_COMMANDS,
    PHASE5_REQUIRED_TEST_IDS,
    PHASE5_SPEC,
    PPO_ARTIFACT,
    PPO_TRACE,
    load_ppo_benchmark_artifact,
    sha256_file,
    validate_phase5_evidence,
    validate_ppo_benchmark_payload,
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
    for command in PHASE5_COMMANDS:
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
        "schema_version": PHASE5_SPEC.schema_version,
        "kind": ARTIFACT_KIND,
        "issue": ISSUE,
        "phase": PHASE,
        "generated_at_utc": "2026-07-30T00:00:00+00:00",
        "source": {"commit_sha": _head(), "branch": "test", "tree_clean": True},
        "inputs": {
            "files": {
                path.as_posix(): sha256_file(REPO_ROOT / path) for path in PHASE5_SPEC.input_files
            },
            "claim_config_hashes": {
                claim.claim_id: sha256_file(REPO_ROOT / claim.config_input)
                for claim in PHASE5_SPEC.claims
            },
        },
        "environment": {
            "platform": "Linux",
            "python": "test",
            "packages": {
                "torch": "2.9",
                "mujoco-warp": "3.10",
                "warp-lang": "1.14",
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
            for claim in PHASE5_SPEC.claims
        ],
        "commands": command_rows,
    }


def test_phase5_evidence_validator_accepts_complete_registered_report() -> None:
    assert validate_phase5_evidence(_valid_report(), root=REPO_ROOT) == ()


def test_phase5_evidence_validator_fails_closed_for_freshness_and_execution_faults() -> None:
    report = _valid_report()

    altered_input = deepcopy(report)
    altered_input["inputs"]["files"][PPO_ARTIFACT.as_posix()] = f"sha256:{'0' * 64}"
    assert any(
        "does not match current input" in error
        for error in validate_phase5_evidence(altered_input, root=REPO_ROOT)
    )

    missing_repetition = deepcopy(report)
    missing_repetition["commands"] = [
        command
        for command in missing_repetition["commands"]
        if command["name"] != "lane_c_stream_lifetime#10"
    ]
    assert any(
        "stream_lifetime" in error and "expected repetitions" in error
        for error in validate_phase5_evidence(missing_repetition, root=REPO_ROOT)
    )

    stale = deepcopy(report)
    stale["source"]["commit_sha"] = "0" * 40
    assert any("ancestor" in error for error in validate_phase5_evidence(stale, root=REPO_ROOT))


def _tamper_profile(artifact: dict[str, Any]) -> None:
    case = next(case for case in artifact["cases"] if case["mode"] == "mjwarp_device")
    case["expected_execution_profile"] = "host_numpy"


def _tamper_aggregate(artifact: dict[str, Any]) -> None:
    artifact["aggregates"]["throughput"]["128"]["mjwarp_device"]["iteration_p50_median_ms"] += 1.0


def test_phase5_ppo_source_matrix_aggregate_profile_trace_and_threshold_faults_fail_closed() -> (
    None
):
    artifact = load_ppo_benchmark_artifact(REPO_ROOT)
    assert validate_ppo_benchmark_payload(artifact, root=REPO_ROOT) == ()

    faults: tuple[tuple[Callable[[dict[str, Any]], None], str], ...] = (
        (lambda value: value["source"].update({"commit": "0" * 40}), "candidate"),
        (lambda value: value["cases"].pop(), "matrix is incomplete"),
        (_tamper_aggregate, "aggregates are not an exact recomputation"),
        (_tamper_profile, "expected_execution_profile differs"),
        (
            lambda value: value["device"]["profiler_trace"].update(
                {"sha256": f"sha256:{'0' * 64}"}
            ),
            "trace hash does not match sibling",
        ),
        (
            lambda value: value["threshold"]["amendment"].update(
                {"manifest_sha256": f"sha256:{'0' * 64}"}
            ),
            "threshold differs from frozen binding",
        ),
        (lambda value: value["gate"].update({"passed": False}), "recorded gate"),
    )
    for mutate, expected in faults:
        tampered = deepcopy(artifact)
        mutate(tampered)
        errors = validate_ppo_benchmark_payload(tampered, root=REPO_ROOT)
        assert any(expected in error for error in errors), (expected, errors)


def test_phase5_claim_mapping_and_freshness_inputs_are_exact() -> None:
    assert {claim.claim_id for claim in PHASE5_SPEC.claims} == set(PHASE5_REQUIRED_TEST_IDS)
    assert PPO_ARTIFACT in PHASE5_SPEC.input_files
    assert PPO_TRACE in PHASE5_SPEC.input_files
    assert Path("src/unilab/training/rsl_rl_device.py") in PHASE5_SPEC.input_files
    assert Path("src/unilab/base/backend/mjwarp/backend.py") in PHASE5_SPEC.input_files
    assert Path("benchmark/rl/benchmark_mjwarp_ppo.py") in PHASE5_SPEC.input_files
