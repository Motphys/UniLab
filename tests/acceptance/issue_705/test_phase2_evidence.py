"""Fail-closed provenance checks for the Issue #705 Phase 2 gate artifact."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.slow

from copy import deepcopy
from pathlib import Path

from scripts import validate_issue705_phase

from unilab.tools.issue705_phase2_evidence import (
    PHASE2_MIN_REPETITIONS,
    PHASE2_REQUIRED_TEST_IDS,
    load_phase2_evidence,
    validate_phase2_evidence,
)
from unilab.tools.phase_acceptance import ClaimStatus, EvidenceResult, load_phase_acceptance

REPO_ROOT = Path(__file__).resolve().parents[3]
ARTIFACT_PATH = REPO_ROOT / "tests/acceptance/issue_705/artifacts/phase_2_gate.json"
MANIFEST_PATH = REPO_ROOT / "tests/acceptance/issue_705/manifests/phase_2.yaml"
ARTIFACT_REF = "tests/acceptance/issue_705/artifacts/phase_2_gate.json"


def test_phase2_gate_artifact_and_manifest_are_fresh_and_complete() -> None:
    report = load_phase2_evidence(ARTIFACT_PATH)
    assert validate_phase2_evidence(report, root=REPO_ROOT) == ()

    manifest = load_phase_acceptance(MANIFEST_PATH)
    claims = {claim.claim_id: claim for claim in manifest.claims}
    artifact_claims = {claim["claim_id"]: claim for claim in report["claims"]}
    assert set(claims) == set(PHASE2_REQUIRED_TEST_IDS)
    assert set(artifact_claims) == set(PHASE2_REQUIRED_TEST_IDS)
    source_commit = report["source"]["commit_sha"]
    config_hash = report["inputs"]["owner_yaml_sha256"]
    for claim_id, test_id in PHASE2_REQUIRED_TEST_IDS.items():
        claim = claims[claim_id]
        assert claim.status is ClaimStatus.VERIFIED
        assert claim.evidence.result is EvidenceResult.PASS
        assert claim.required_test_ids == (test_id,)
        assert claim.evidence.executed_test_ids == (test_id,)
        assert claim.evidence.artifact_refs == (ARTIFACT_REF,)
        assert claim.evidence.commit_sha == source_commit
        assert claim.evidence.config_hash == config_hash
        assert claim.evidence.skipped_test_ids == ()
        assert claim.evidence.xfailed_test_ids == ()
        assert artifact_claims[claim_id]["minimum_repetitions"] == PHASE2_MIN_REPETITIONS[claim_id]

    assert validate_issue705_phase.main(["--phase", "2", "--mode", "gate"]) == 0


def test_phase2_artifact_faults_fail_closed() -> None:
    report = load_phase2_evidence(ARTIFACT_PATH)

    skipped = deepcopy(report)
    skipped["commands"][0]["pytest"]["skipped"] = 1
    assert any(
        "skipped: expected 0" in error
        for error in validate_phase2_evidence(skipped, root=REPO_ROOT)
    )

    stale = deepcopy(report)
    stale["source"]["commit_sha"] = "0" * 40
    assert any("ancestor" in error for error in validate_phase2_evidence(stale, root=REPO_ROOT))

    mismatched_claim = deepcopy(report)
    mismatched_claim["claims"][0]["required_test_id"] = "tests/does_not_exist.py::test_missing"
    assert any(
        "required test mapping" in error
        for error in validate_phase2_evidence(mismatched_claim, root=REPO_ROOT)
    )

    altered_command = deepcopy(report)
    altered_command["commands"][0]["argv"] = ["echo", "not-the-registered-command"]
    assert any(
        "does not match registered command" in error
        for error in validate_phase2_evidence(altered_command, root=REPO_ROOT)
    )

    incomplete_repetitions = deepcopy(report)
    incomplete_repetitions["commands"] = [
        command
        for command in incomplete_repetitions["commands"]
        if command["name"] != "lane_c_production_cuda#3"
    ]
    assert any(
        "expected repetitions" in error
        for error in validate_phase2_evidence(incomplete_repetitions, root=REPO_ROOT)
    )
