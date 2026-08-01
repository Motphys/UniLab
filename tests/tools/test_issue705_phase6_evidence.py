"""Synthetic fault coverage for the Issue #705 Phase 6 evidence gate."""

from __future__ import annotations

import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Any

import unilab.tools.issue705_phase6_evidence as phase6_evidence
from unilab.tools.issue705_phase6_evidence import (
    ARTIFACT_KIND,
    DR_PERFORMANCE_ARTIFACT,
    DR_PERFORMANCE_PLAN,
    DR_PERFORMANCE_RECEIPT,
    ISSUE,
    PHASE,
    PHASE6_COMMANDS,
    PHASE6_REQUIRED_TEST_IDS,
    PHASE6_SPEC,
    load_dr_performance_artifact,
    sha256_file,
    validate_dr_performance_payload,
    validate_phase6_evidence,
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
    for command in PHASE6_COMMANDS:
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
        "schema_version": PHASE6_SPEC.schema_version,
        "kind": ARTIFACT_KIND,
        "issue": ISSUE,
        "phase": PHASE,
        "generated_at_utc": "2026-07-30T00:00:00+00:00",
        "source": {"commit_sha": _head(), "branch": "test", "tree_clean": True},
        "inputs": {
            "files": {
                path.as_posix(): sha256_file(REPO_ROOT / path) for path in PHASE6_SPEC.input_files
            },
            "claim_config_hashes": {
                claim.claim_id: sha256_file(REPO_ROOT / claim.config_input)
                for claim in PHASE6_SPEC.claims
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
            for claim in PHASE6_SPEC.claims
        ],
        "commands": command_rows,
    }


def test_phase6_evidence_validator_accepts_complete_registered_report() -> None:
    assert validate_phase6_evidence(_valid_report(), root=REPO_ROOT) == ()


def test_phase6_expanded_inputs_include_only_git_tracked_tree_files(tmp_path: Path) -> None:
    subprocess.run(("git", "init", "--quiet"), cwd=tmp_path, check=True)
    python_dir = tmp_path / "src/unilab/base/backend"
    asset_dir = tmp_path / "src/unilab/assets/robots/g1"
    python_dir.mkdir(parents=True)
    asset_dir.mkdir(parents=True)
    (tmp_path / ".gitignore").write_text("**/tmp*.xml\n", encoding="utf-8")
    (python_dir / "tracked.py").write_text("TRACKED = True\n", encoding="utf-8")
    (python_dir / "untracked.py").write_text("UNTRACKED = True\n", encoding="utf-8")
    (asset_dir / "scene.xml").write_text("<mujoco/>\n", encoding="utf-8")
    (asset_dir / "tmp-generated.xml").write_text("<mujoco/>\n", encoding="utf-8")
    subprocess.run(
        (
            "git",
            "add",
            ".gitignore",
            "src/unilab/base/backend/tracked.py",
            "src/unilab/assets/robots/g1/scene.xml",
        ),
        cwd=tmp_path,
        check=True,
    )

    expanded = set(phase6_evidence._expanded_inputs(tmp_path))

    assert Path("src/unilab/base/backend/tracked.py") in expanded
    assert Path("src/unilab/assets/robots/g1/scene.xml") in expanded
    assert Path("src/unilab/base/backend/untracked.py") not in expanded
    assert Path("src/unilab/assets/robots/g1/tmp-generated.xml") not in expanded


def test_phase6_evidence_validator_fails_closed_for_capture_faults() -> None:
    report = _valid_report()

    stale = deepcopy(report)
    stale["source"]["commit_sha"] = "0" * 40
    assert any("ancestor" in error for error in validate_phase6_evidence(stale, root=REPO_ROOT))

    altered_input = deepcopy(report)
    altered_input["inputs"]["files"][DR_PERFORMANCE_ARTIFACT.as_posix()] = f"sha256:{'0' * 64}"
    assert any(
        "does not match current input" in error
        for error in validate_phase6_evidence(altered_input, root=REPO_ROOT)
    )

    altered_command = deepcopy(report)
    altered_command["commands"][0]["argv"] = ["echo", "not-the-registered-command"]
    assert any(
        "does not match registered command" in error
        for error in validate_phase6_evidence(altered_command, root=REPO_ROOT)
    )

    missing_repetition = deepcopy(report)
    missing_repetition["commands"] = [
        command
        for command in missing_repetition["commands"]
        if command["name"] != "lane_c_physics_effect#5"
    ]
    assert any(
        "lane_c_physics_effect" in error and "expected repetitions" in error
        for error in validate_phase6_evidence(missing_repetition, root=REPO_ROOT)
    )

    xpassed = deepcopy(report)
    xpassed["commands"][0]["pytest"]["xpassed"] = 1
    assert any(
        "xpassed: expected 0" in error
        for error in validate_phase6_evidence(xpassed, root=REPO_ROOT)
    )


def test_phase6_dr_performance_payload_is_independently_recomputed() -> None:
    artifact = load_dr_performance_artifact(REPO_ROOT)
    assert validate_dr_performance_payload(artifact, root=REPO_ROOT) == ()

    altered_aggregates = {**artifact, "aggregates": {}}
    errors = validate_dr_performance_payload(altered_aggregates, root=REPO_ROOT)
    assert any("independently recomputed raw data" in error for error in errors)

    altered_gate = {**artifact, "gate": {"passed": False, "errors": []}}
    errors = validate_dr_performance_payload(altered_gate, root=REPO_ROOT)
    assert any("differs from independent validation" in error for error in errors)


def test_phase6_claim_mapping_and_freshness_inputs_are_exact() -> None:
    assert {claim.claim_id for claim in PHASE6_SPEC.claims} == set(PHASE6_REQUIRED_TEST_IDS)
    assert sum(command.repetitions for command in PHASE6_COMMANDS) == 26
    assert {command.lane for command in PHASE6_COMMANDS} == {"B", "C", "D"}
    controller = next(
        command for command in PHASE6_COMMANDS if command.name == "lane_c_controller_contract"
    )
    assert ("-m", "slow") == controller.argv[7:9]
    assert DR_PERFORMANCE_ARTIFACT in PHASE6_SPEC.input_files
    assert DR_PERFORMANCE_PLAN in PHASE6_SPEC.input_files
    assert DR_PERFORMANCE_RECEIPT in PHASE6_SPEC.input_files
    assert Path("uv.lock") in PHASE6_SPEC.input_files
    assert Path("conf/ppo/task/g1_walk_flat/mjwarp.yaml") in PHASE6_SPEC.input_files
    assert Path("src/unilab/base/backend/mjwarp/backend.py") in PHASE6_SPEC.input_files
    assert Path("src/unilab/manager/device_runtime.py") in PHASE6_SPEC.input_files
