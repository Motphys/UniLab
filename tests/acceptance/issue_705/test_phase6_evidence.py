"""Freshness and tamper checks for the Issue #705 Phase 6 gate artifacts."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.slow

from copy import deepcopy
from pathlib import Path

from scripts import validate_issue705_phase

from unilab.tools.issue705_phase6_evidence import (
    DR_PERFORMANCE_ARTIFACT,
    DR_PERFORMANCE_PLAN,
    DR_PERFORMANCE_RECEIPT,
    PHASE6_MIN_REPETITIONS,
    PHASE6_REQUIRED_TEST_IDS,
    load_phase6_evidence,
    validate_dr_performance_artifact,
    validate_phase6_evidence,
)
from unilab.tools.phase_acceptance import ClaimStatus, EvidenceResult, load_phase_acceptance

REPO_ROOT = Path(__file__).resolve().parents[3]
ARTIFACT_PATH = REPO_ROOT / "tests/acceptance/issue_705/artifacts/phase_6_gate.json"
MANIFEST_PATH = REPO_ROOT / "tests/acceptance/issue_705/manifests/phase_6.yaml"
GATE_REF = "tests/acceptance/issue_705/artifacts/phase_6_gate.json"


def _expected_refs(claim_id: str) -> tuple[str, ...]:
    if claim_id == "P6-DR-PERFORMANCE":
        return (
            GATE_REF,
            DR_PERFORMANCE_ARTIFACT.as_posix(),
            DR_PERFORMANCE_PLAN.as_posix(),
            DR_PERFORMANCE_RECEIPT.as_posix(),
        )
    return (GATE_REF,)


def test_phase6_gate_artifacts_and_manifest_are_fresh_complete_and_promoted() -> None:
    assert validate_dr_performance_artifact(root=REPO_ROOT) == ()
    report = load_phase6_evidence(ARTIFACT_PATH)
    assert validate_phase6_evidence(report, root=REPO_ROOT) == ()

    manifest = load_phase_acceptance(MANIFEST_PATH)
    claims = {claim.claim_id: claim for claim in manifest.claims}
    artifact_claims = {claim["claim_id"]: claim for claim in report["claims"]}
    config_hashes = report["inputs"]["claim_config_hashes"]
    source_commit = report["source"]["commit_sha"]
    assert set(claims) == set(PHASE6_REQUIRED_TEST_IDS)
    assert set(artifact_claims) == set(PHASE6_REQUIRED_TEST_IDS)
    for claim_id, test_id in PHASE6_REQUIRED_TEST_IDS.items():
        claim = claims[claim_id]
        assert claim.status is ClaimStatus.VERIFIED
        assert claim.evidence.result is EvidenceResult.PASS
        assert claim.required_test_ids == (test_id,)
        assert claim.evidence.executed_test_ids == (test_id,)
        assert claim.evidence.artifact_refs == _expected_refs(claim_id)
        assert claim.evidence.commit_sha == source_commit
        assert claim.evidence.config_hash == config_hashes[claim_id]
        assert claim.evidence.skipped_test_ids == ()
        assert claim.evidence.xfailed_test_ids == ()
        assert artifact_claims[claim_id]["minimum_repetitions"] == PHASE6_MIN_REPETITIONS[claim_id]

    assert validate_issue705_phase.main(["--phase", "6", "--mode", "gate"]) == 0


def test_phase6_gate_artifact_faults_fail_closed() -> None:
    report = load_phase6_evidence(ARTIFACT_PATH)

    skipped = deepcopy(report)
    skipped["commands"][0]["pytest"]["skipped"] = 1
    assert any(
        "skipped: expected 0" in error
        for error in validate_phase6_evidence(skipped, root=REPO_ROOT)
    )

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

    incomplete_repetitions = deepcopy(report)
    incomplete_repetitions["commands"] = [
        command
        for command in incomplete_repetitions["commands"]
        if command["name"] != "lane_c_recompute_aggregation#3"
    ]
    assert any(
        "lane_c_recompute_aggregation" in error and "expected repetitions" in error
        for error in validate_phase6_evidence(incomplete_repetitions, root=REPO_ROOT)
    )
