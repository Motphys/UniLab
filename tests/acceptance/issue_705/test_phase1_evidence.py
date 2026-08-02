"""Freshness and tamper checks for the Issue #705 Phase 1 gate artifact."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.slow

from copy import deepcopy
from pathlib import Path

from scripts import validate_issue705_phase

from unilab.tools.issue705_phase1_evidence import (
    PHASE1_MIN_REPETITIONS,
    PHASE1_REQUIRED_TEST_IDS,
    load_phase1_evidence,
    validate_phase1_evidence,
)
from unilab.tools.phase_acceptance import ClaimStatus, EvidenceResult, load_phase_acceptance

REPO_ROOT = Path(__file__).resolve().parents[3]
ARTIFACT_PATH = REPO_ROOT / "tests/acceptance/issue_705/artifacts/phase_1_gate.json"
MANIFEST_PATH = REPO_ROOT / "tests/acceptance/issue_705/manifests/phase_1.yaml"
ARTIFACT_REF = "tests/acceptance/issue_705/artifacts/phase_1_gate.json"


def test_phase1_gate_artifact_and_manifest_are_fresh_complete_and_promoted() -> None:
    report = load_phase1_evidence(ARTIFACT_PATH)
    assert validate_phase1_evidence(report, root=REPO_ROOT) == ()

    manifest = load_phase_acceptance(MANIFEST_PATH)
    claims = {claim.claim_id: claim for claim in manifest.claims}
    artifact_claims = {claim["claim_id"]: claim for claim in report["claims"]}
    config_hashes = report["inputs"]["claim_config_hashes"]
    source_commit = report["source"]["commit_sha"]
    assert set(claims) == set(PHASE1_REQUIRED_TEST_IDS)
    assert set(artifact_claims) == set(PHASE1_REQUIRED_TEST_IDS)
    for claim_id, test_id in PHASE1_REQUIRED_TEST_IDS.items():
        claim = claims[claim_id]
        assert claim.status is ClaimStatus.VERIFIED
        assert claim.evidence.result is EvidenceResult.PASS
        assert claim.required_test_ids == (test_id,)
        assert claim.evidence.executed_test_ids == (test_id,)
        assert claim.evidence.artifact_refs == (ARTIFACT_REF,)
        assert claim.evidence.commit_sha == source_commit
        assert claim.evidence.config_hash == config_hashes[claim_id]
        assert claim.evidence.skipped_test_ids == ()
        assert claim.evidence.xfailed_test_ids == ()
        assert artifact_claims[claim_id]["minimum_repetitions"] == PHASE1_MIN_REPETITIONS[claim_id]

    assert validate_issue705_phase.main(["--phase", "1", "--mode", "gate"]) == 0


def test_phase1_gate_artifact_faults_fail_closed() -> None:
    report = load_phase1_evidence(ARTIFACT_PATH)

    skipped = deepcopy(report)
    skipped["commands"][0]["pytest"]["skipped"] = 1
    assert any(
        "skipped: expected 0" in error
        for error in validate_phase1_evidence(skipped, root=REPO_ROOT)
    )

    stale = deepcopy(report)
    stale["source"]["commit_sha"] = "0" * 40
    assert any("ancestor" in error for error in validate_phase1_evidence(stale, root=REPO_ROOT))

    altered_input = deepcopy(report)
    altered_input["inputs"]["files"]["uv.lock"] = f"sha256:{'0' * 64}"
    assert any(
        "does not match current input" in error
        for error in validate_phase1_evidence(altered_input, root=REPO_ROOT)
    )

    altered_command = deepcopy(report)
    altered_command["commands"][0]["argv"] = ["echo", "not-the-registered-command"]
    assert any(
        "does not match registered command" in error
        for error in validate_phase1_evidence(altered_command, root=REPO_ROOT)
    )

    incomplete_repetitions = deepcopy(report)
    incomplete_repetitions["commands"] = [
        command
        for command in incomplete_repetitions["commands"]
        if command["name"] != "lane_b_mujoco_reference#3"
    ]
    assert any(
        "lane_b_mujoco_reference" in error and "expected repetitions" in error
        for error in validate_phase1_evidence(incomplete_repetitions, root=REPO_ROOT)
    )
