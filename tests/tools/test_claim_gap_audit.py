from __future__ import annotations

import copy
from pathlib import Path

import pytest

from unilab.tools.claim_gap_audit import (
    ClaimGapInventoryError,
    EvidenceKind,
    InventoryTestState,
    audit_claim_gaps,
    load_claim_gap_inventory,
    parse_claim_gap_inventory,
)
from unilab.tools.phase_acceptance import parse_phase_acceptance


def _claim() -> dict:
    return {
        "claim_id": "P0-CONTRACT",
        "expected": "The contract is enforced.",
        "risk": "A missing behavior is reported as implemented.",
        "owner": "validation",
        "oracle": "Independent contract and effect tests.",
        "commands": ["uv run pytest tests/test_contract.py"],
        "lane": "A",
        "required": True,
        "required_test_ids": [
            "tests/test_contract.py::test_contract",
            "tests/test_contract.py::test_effect",
        ],
        "environment": {
            "dependencies": ["uv.lock"],
            "hardware": "any",
            "owner_yaml": None,
            "seeds": [],
            "batch_sizes": [],
            "dtype": None,
            "plan_fingerprint": "claim-gap-test-v1",
        },
        "acceptance": {
            "tolerance": {},
            "thresholds": {},
            "repetitions": 1,
            "max_dispersion": 0,
            "failure_semantics": "Any unmapped test is FAIL.",
        },
        "evidence": {
            "result": "NOT_RUN",
            "artifact_refs": [],
            "commit_sha": None,
            "config_hash": None,
            "executed_test_ids": [],
            "skipped_test_ids": [],
            "xfailed_test_ids": [],
            "summary": "",
        },
        "invalidation": {
            "paths": ["tests/**"],
            "capabilities": ["claim-gap-audit"],
            "fingerprints": ["claim-gap-test-v1"],
        },
        "status": "planned",
    }


def _manifest():
    return parse_phase_acceptance(
        {
            "schema_version": 1,
            "issue": 705,
            "phase": 0,
            "integration_branch": "feat/manager-mjwarp-manager-mjwarp",
            "required_lanes": ["A"],
            "claims": [_claim()],
        }
    )


def _entry(
    test_id: str,
    *,
    state: str,
    role: str = "acceptance",
    evidence_kind: str = "contract",
    gap: str | None = None,
) -> dict:
    return {
        "claim_id": "P0-CONTRACT",
        "test_id": test_id,
        "state": state,
        "role": role,
        "evidence_kind": evidence_kind,
        "owner": "validation",
        "oracle": "The test uses a fixture independent of the implementation result.",
        "gap": gap,
    }


def _inventory() -> dict:
    return {
        "schema_version": 1,
        "issue": 705,
        "integration_branch": "feat/manager-mjwarp-manager-mjwarp",
        "entries": [
            _entry("tests/test_contract.py::test_contract", state="existing"),
            _entry(
                "tests/test_contract.py::test_effect",
                state="target",
                evidence_kind="effect",
                gap="The independent effect oracle has not been implemented.",
            ),
            _entry(
                "tests/test_contract.py::test_smoke",
                state="existing",
                role="supporting",
                evidence_kind="smoke",
            ),
        ],
    }


def _write_test_module(repo_root: Path, *, include_effect: bool = False) -> None:
    path = repo_root / "tests" / "test_contract.py"
    path.parent.mkdir(parents=True)
    effect = "\ndef test_effect():\n    pass\n" if include_effect else ""
    path.write_text(
        f"def test_contract():\n    pass\n\ndef test_smoke():\n    pass\n{effect}",
        encoding="utf-8",
    )


def _errors(raw: dict) -> tuple[str, ...]:
    with pytest.raises(ClaimGapInventoryError) as exc_info:
        parse_claim_gap_inventory(raw)
    return exc_info.value.errors


def test_valid_inventory_audits_existing_target_and_supporting_evidence(
    tmp_path: Path,
) -> None:
    _write_test_module(tmp_path)
    inventory = parse_claim_gap_inventory(_inventory())

    report = audit_claim_gaps(inventory, [_manifest()], repo_root=tmp_path, phases=[0])

    assert report.ok
    assert report.claims == 1
    assert report.existing == 2
    assert report.targets == 1
    assert report.supporting == 1
    assert inventory.entries[0].state == InventoryTestState.EXISTING
    assert inventory.entries[1].evidence_kind == EvidenceKind.EFFECT


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda raw: raw.update({"unknown": True}), "inventory: unknown key `unknown`"),
        (lambda raw: raw.pop("entries"), "inventory: missing key `entries`"),
        (lambda raw: raw.update({"schema_version": 2}), "schema_version: expected 1"),
        (
            lambda raw: raw["entries"][0].update({"test_id": "acceptance:P0-CONTRACT"}),
            "expected repository pytest ID",
        ),
        (
            lambda raw: raw["entries"][1].update({"test_id": "tests/test_future.py"}),
            "target test requires an explicit pytest node",
        ),
        (
            lambda raw: raw["entries"][1].update({"test_id": "tests/../outside.py::test_escape"}),
            "path must stay within the repository",
        ),
        (
            lambda raw: raw["entries"][1].update({"test_id": "tests/test_future.py::helper"}),
            "must end in a test node",
        ),
        (lambda raw: raw["entries"][1].update({"gap": None}), "target test requires"),
        (
            lambda raw: raw["entries"][0].update({"gap": "not actually complete"}),
            "existing test must not declare a gap",
        ),
        (
            lambda raw: raw["entries"][0].update({"role": "acceptance", "evidence_kind": "smoke"}),
            "smoke evidence cannot be an acceptance oracle",
        ),
        (
            lambda raw: raw["entries"][0].update({"oracle": "${oc.env:UNSAFE}"}),
            "interpolation is not allowed",
        ),
    ],
)
def test_inventory_rejects_malformed_or_weak_contract(mutate, message: str) -> None:
    raw = _inventory()
    mutate(raw)

    assert any(message in error for error in _errors(raw))


def test_inventory_rejects_duplicate_claim_test_pair() -> None:
    raw = _inventory()
    raw["entries"].append(copy.deepcopy(raw["entries"][0]))

    assert any("duplicate claim/test mappings" in error for error in _errors(raw))


def test_audit_rejects_missing_and_extra_acceptance_mapping(tmp_path: Path) -> None:
    _write_test_module(tmp_path)
    raw = _inventory()
    raw["entries"][1]["role"] = "supporting"
    raw["entries"][2] = _entry(
        "tests/test_contract.py::test_unowned",
        state="target",
        gap="This extra target is not part of the manifest contract.",
    )
    inventory = parse_claim_gap_inventory(raw)

    errors = audit_claim_gaps(inventory, [_manifest()], repo_root=tmp_path, phases=[0]).errors

    assert any("test_effect` has no acceptance inventory entry" in error for error in errors)
    assert any("test_unowned` is not required by manifest" in error for error in errors)


def test_audit_rejects_owner_mismatch_and_orphan_claim(tmp_path: Path) -> None:
    _write_test_module(tmp_path)
    raw = _inventory()
    raw["entries"][0]["owner"] = "backend"
    orphan = _entry(
        "tests/test_orphan.py::test_orphan",
        state="target",
        gap="No manifest claim owns this target.",
    )
    orphan["claim_id"] = "P0-ORPHAN"
    raw["entries"].append(orphan)
    inventory = parse_claim_gap_inventory(raw)

    errors = audit_claim_gaps(inventory, [_manifest()], repo_root=tmp_path, phases=[0]).errors

    assert any("owner 'backend' does not match" in error for error in errors)
    assert any(
        "P0-ORPHAN: inventory entry has no matching manifest claim" in error for error in errors
    )


def test_audit_rejects_missing_existing_file_or_node(tmp_path: Path) -> None:
    _write_test_module(tmp_path)
    raw = _inventory()
    raw["entries"][0]["test_id"] = "tests/test_contract.py::test_missing"
    raw["entries"][2]["test_id"] = "tests/test_missing.py::test_smoke"
    inventory = parse_claim_gap_inventory(raw)

    errors = audit_claim_gaps(inventory, [_manifest()], repo_root=tmp_path, phases=[0]).errors

    assert any("existing pytest node is missing" in error for error in errors)
    assert any("existing test file is missing" in error for error in errors)


def test_audit_rejects_target_that_has_become_resolvable(tmp_path: Path) -> None:
    _write_test_module(tmp_path, include_effect=True)
    inventory = parse_claim_gap_inventory(_inventory())

    errors = audit_claim_gaps(inventory, [_manifest()], repo_root=tmp_path, phases=[0]).errors

    assert any("target pytest node already exists" in error for error in errors)


def test_audit_rejects_empty_phase_selection_and_missing_manifest(tmp_path: Path) -> None:
    _write_test_module(tmp_path)
    inventory = parse_claim_gap_inventory(_inventory())

    empty_errors = audit_claim_gaps(inventory, [], repo_root=tmp_path, phases=[]).errors
    missing_errors = audit_claim_gaps(inventory, [], repo_root=tmp_path, phases=[0]).errors

    assert "phases: at least one phase must be selected" in empty_errors
    assert "phase 0: no manifest supplied to claim audit" in missing_errors


def test_file_level_existing_mapping_requires_at_least_one_test(tmp_path: Path) -> None:
    path = tmp_path / "tests" / "test_contract.py"
    path.parent.mkdir(parents=True)
    path.write_text("def helper():\n    pass\n", encoding="utf-8")
    raw = _inventory()
    raw["entries"][0]["test_id"] = "tests/test_contract.py"
    raw["entries"][2]["state"] = "target"
    raw["entries"][2]["gap"] = "The smoke test is intentionally absent."
    inventory = parse_claim_gap_inventory(raw)

    errors = audit_claim_gaps(inventory, [_manifest()], repo_root=tmp_path, phases=[0]).errors

    assert any("existing pytest node is missing" in error for error in errors)


def test_load_inventory_normalizes_invalid_yaml(tmp_path: Path) -> None:
    path = tmp_path / "inventory.yaml"
    path.write_text("entries: [", encoding="utf-8")

    with pytest.raises(ClaimGapInventoryError, match="cannot load YAML"):
        load_claim_gap_inventory(path)
