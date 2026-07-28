from __future__ import annotations

from pathlib import Path

from scripts import audit_issue705_claims, validate_issue705_phase

from unilab.tools.claim_gap_audit import (
    PHASES,
    EvidenceKind,
    EvidenceRole,
    InventoryTestState,
    load_claim_gap_inventory,
    load_phase_manifests,
)
from unilab.tools.phase_acceptance import ClaimStatus, phase_gate_errors

EXPECTED_CLAIMS = {
    0: {
        "P0-WORKFLOW-CANARY",
        "P0-ACCEPTANCE-CONTRACT",
        "P0-CLAIM-GAP-AUDIT",
        "P0-DR-CAPABILITY-INVENTORY",
        "P0-BASELINE-PROVENANCE",
        "P0-THRESHOLD-FREEZE",
    },
    1: {
        "P1-BATCH-CONTRACT",
        "P1-MUJOCO-REFERENCE",
        "P1-MUTATION-CONTRACT",
        "P1-HOT-PATH-INSTRUMENTATION",
        "P1-DR-COMPATIBILITY",
        "P1-BACKEND-ISOLATION",
    },
    2: {
        "P2-BACKEND-IDENTITY",
        "P2-GPU-CORRECTNESS",
        "P2-RESET-ISOLATION",
        "P2-TRAJECTORY-DIFFERENTIAL",
        "P2-DR-OWNER-SEMANTICS",
        "P2-TRANSFER-ACCOUNTING",
        "P2-UNSUPPORTED-FAIL-CLOSED",
        "P2-TRAIN-LIVENESS",
    },
    3: {
        "P3-TASK-COMPILER",
        "P3-LIFECYCLE-PARITY",
        "P3-G1-REFERENCE-DIFFERENTIAL",
        "P3-POLICY-ABI",
        "P3-CROSS-BACKEND-PLAN",
        "P3-GENERALITY-FIXTURE",
    },
    4: {
        "P4-FUSED-PARITY",
        "P4-NO-FALLBACK",
        "P4-ALLOCATION-STABILITY",
        "P4-HOST-PERFORMANCE",
    },
    5: {
        "P5-DEVICE-ABI",
        "P5-STREAM-LIFETIME",
        "P5-DEVICE-LIFECYCLE",
        "P5-NO-HOST-ROUNDTRIP",
        "P5-GRAPH-CONTRACT",
        "P5-TRAIN-PERFORMANCE",
        "P5-DEVICE-STABILITY",
    },
    6: {
        "P6-CAPABILITY-BIJECTION",
        "P6-DR-SEMANTICS",
        "P6-PHYSICS-EFFECT",
        "P6-RNG-REPRODUCIBILITY",
        "P6-GRAPH-RECAPTURE",
        "P6-CONTROLLER-CONTRACT",
        "P6-DR-PERFORMANCE",
        "P6-RECOMPUTE-AGGREGATION",
    },
    7: {
        "P7-SUPPORT-MATRIX",
        "P7-TASK-ROLLOUT",
        "P7-TRAINING-BEHAVIOR",
        "P7-ENTRYPOINT-MATRIX",
        "P7-FINAL-REGRESSION",
        "P7-LEGACY-RETIREMENT",
    },
}

REQUIRED_EVIDENCE_KINDS = {
    "P2-TRAIN-LIVENESS": EvidenceKind.LIVENESS,
    "P4-HOST-PERFORMANCE": EvidenceKind.PERFORMANCE,
    "P5-NO-HOST-ROUNDTRIP": EvidenceKind.PERFORMANCE,
    "P5-TRAIN-PERFORMANCE": EvidenceKind.PERFORMANCE,
    "P6-PHYSICS-EFFECT": EvidenceKind.EFFECT,
    "P6-DR-PERFORMANCE": EvidenceKind.PERFORMANCE,
    "P7-TRAINING-BEHAVIOR": EvidenceKind.TRAINING,
}


def test_all_phase_manifests_exist_and_pass_schema() -> None:
    manifests, errors = load_phase_manifests(audit_issue705_claims.MANIFEST_DIR, PHASES)

    assert errors == ()
    assert tuple(manifest.phase for manifest in manifests) == PHASES
    assert {
        manifest.phase: {claim.claim_id for claim in manifest.claims} for manifest in manifests
    } == EXPECTED_CLAIMS
    assert all(
        claim.status == ClaimStatus.PLANNED
        for manifest in manifests[1:]
        for claim in manifest.claims
    )


def test_repository_claim_inventory_is_bidirectional_for_all_phases(capsys) -> None:
    exit_code = audit_issue705_claims.main(["--all"])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "PASS phases=[0, 1, 2, 3, 4, 5, 6, 7]" in output
    assert "targets=" in output


def test_phase0_claim_inventory_cli_passes_json(capsys) -> None:
    exit_code = audit_issue705_claims.main(["--phase", "0", "--json"])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert '"ok": true' in output
    assert '"phases": [' in output


def test_high_risk_claims_require_the_right_evidence_class() -> None:
    inventory = load_claim_gap_inventory(audit_issue705_claims.INVENTORY_PATH)
    acceptance_by_claim = {
        entry.claim_id: entry
        for entry in inventory.entries
        if entry.role == EvidenceRole.ACCEPTANCE
    }

    for claim_id, evidence_kind in REQUIRED_EVIDENCE_KINDS.items():
        assert acceptance_by_claim[claim_id].evidence_kind == evidence_kind

    # ``existing`` means that a concrete test node now exists; it does *not*
    # promote a phase or turn a raw test into fresh gate evidence.  Correctness
    # work may therefore add an effect/differential oracle before its Phase C
    # artifact is captured.  Promotion remains guarded by the phase manifest
    # and the artifact validator below, rather than by pretending the test is
    # still absent from the inventory.
    current_acceptance = [
        entry
        for entry in inventory.entries
        if entry.role == EvidenceRole.ACCEPTANCE and entry.state == InventoryTestState.EXISTING
    ]
    assert all(entry.evidence_kind != EvidenceKind.SMOKE for entry in current_acceptance)


def test_phase_zero_is_open_and_later_phase_gates_remain_closed() -> None:
    manifests, errors = load_phase_manifests(audit_issue705_claims.MANIFEST_DIR, PHASES)

    assert errors == ()
    assert not phase_gate_errors(manifests[0])
    assert all(phase_gate_errors(manifest) for manifest in manifests[1:])


def test_phase_schema_cli_accepts_each_frozen_manifest(capsys) -> None:
    for phase in PHASES:
        assert validate_issue705_phase.main(["--phase", str(phase), "--mode", "schema"]) == 0
    output = capsys.readouterr().out
    assert output.count("PASS issue=705") == len(PHASES)


def test_cli_fails_closed_when_manifest_is_missing(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(audit_issue705_claims, "MANIFEST_DIR", tmp_path / "manifests")

    exit_code = audit_issue705_claims.main(["--phase", "0"])

    assert exit_code == 1
    assert "missing manifest" in capsys.readouterr().out


def test_cli_fails_closed_when_inventory_is_malformed(tmp_path: Path, monkeypatch, capsys) -> None:
    inventory = tmp_path / "inventory.yaml"
    inventory.write_text("entries: [", encoding="utf-8")
    monkeypatch.setattr(audit_issue705_claims, "INVENTORY_PATH", inventory)

    exit_code = audit_issue705_claims.main(["--phase", "0"])

    assert exit_code == 1
    assert "cannot load YAML" in capsys.readouterr().out
