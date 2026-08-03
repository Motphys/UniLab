from __future__ import annotations

from pathlib import Path

from tooling.acceptance.commands import dr_inventory as validate_manager_mjwarp_dr_inventory


def test_repository_mjwarp_dr_inventory_cli_passes(capsys) -> None:
    exit_code = validate_manager_mjwarp_dr_inventory.main([])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "PASS backend=mjwarp capabilities=16" in output
    assert "blocked_pending_evidence" in output


def test_repository_mjwarp_dr_inventory_cli_emits_json(capsys) -> None:
    exit_code = validate_manager_mjwarp_dr_inventory.main(["--json"])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert '"ok": true' in output
    assert '"source_commit": "f643d245303ff439a90f37151056ff987bdb95f7"' in output


def test_cli_fails_closed_on_malformed_inventory(tmp_path: Path, capsys) -> None:
    inventory = tmp_path / "inventory.yaml"
    inventory.write_text("capabilities: [", encoding="utf-8")

    exit_code = validate_manager_mjwarp_dr_inventory.main(["--inventory", str(inventory)])

    assert exit_code == 1
    assert "cannot load YAML" in capsys.readouterr().out


def test_cli_fails_closed_on_malformed_claim_inventory(tmp_path: Path, capsys) -> None:
    inventory = tmp_path / "claims.yaml"
    inventory.write_text("entries: [", encoding="utf-8")

    exit_code = validate_manager_mjwarp_dr_inventory.main(["--claim-inventory", str(inventory)])

    assert exit_code == 1
    assert "claim inventory: cannot load YAML" in capsys.readouterr().out
