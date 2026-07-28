from __future__ import annotations

from pathlib import Path

from scripts import validate_issue705_phase

from unilab.tools.phase_acceptance import (
    AcceptanceLane,
    ClaimStatus,
    load_phase_acceptance,
    phase_gate_errors,
)


def test_phase0_manifest_covers_frozen_claims_and_passes_schema() -> None:
    manifest_path = (
        Path(__file__).resolve().parents[2]
        / "tests"
        / "acceptance"
        / "issue_705"
        / "manifests"
        / "phase_0.yaml"
    )

    manifest = load_phase_acceptance(manifest_path)
    statuses = {claim.claim_id: claim.status for claim in manifest.claims}

    assert {claim.claim_id for claim in manifest.claims} == {
        "P0-WORKFLOW-CANARY",
        "P0-ACCEPTANCE-CONTRACT",
        "P0-CLAIM-GAP-AUDIT",
        "P0-DR-CAPABILITY-INVENTORY",
        "P0-BASELINE-PROVENANCE",
        "P0-THRESHOLD-FREEZE",
    }
    assert manifest.claims[0].status == ClaimStatus.VERIFIED
    assert statuses["P0-BASELINE-PROVENANCE"] == ClaimStatus.VERIFIED
    assert statuses["P0-THRESHOLD-FREEZE"] == ClaimStatus.VERIFIED
    assert set(manifest.required_lanes) == {
        AcceptanceLane.PR,
        AcceptanceLane.BACKEND,
        AcceptanceLane.BENCHMARK,
    }
    assert not phase_gate_errors(manifest)


def test_phase0_schema_cli_passes(capsys) -> None:
    exit_code = validate_issue705_phase.main(["--phase", "0", "--mode", "schema"])

    assert exit_code == 0
    assert "PASS issue=705 phase=0 mode=schema" in capsys.readouterr().out


def test_phase0_gate_cli_passes_when_all_required_claims_are_verified(capsys) -> None:
    exit_code = validate_issue705_phase.main(["--phase", "0", "--mode", "gate"])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "PASS issue=705 phase=0 mode=gate" in output


def test_cli_rejects_requested_phase_mismatch(tmp_path: Path, capsys) -> None:
    manifest = tmp_path / "phase_1.yaml"
    source = validate_issue705_phase.MANIFEST_DIR / "phase_0.yaml"
    manifest.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")

    exit_code = validate_issue705_phase.main(
        ["--phase", "1", "--manifest", str(manifest), "--mode", "schema"]
    )

    assert exit_code == 1
    assert "requested 1, manifest declares 0" in capsys.readouterr().out
