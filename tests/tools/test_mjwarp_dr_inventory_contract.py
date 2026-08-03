from __future__ import annotations

import copy
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from unilab.tools.claim_gap_audit import parse_claim_gap_inventory
from unilab.tools.mjwarp_dr_inventory import (
    DrInventoryValidationError,
    inventory_claim_gap_errors,
    load_mjwarp_dr_inventory,
    parse_mjwarp_dr_inventory,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
INVENTORY_PATH = REPO_ROOT / "tests/acceptance/manager_mjwarp/mjwarp_dr_inventory.yaml"
CLAIM_INVENTORY_PATH = REPO_ROOT / "tests/acceptance/manager_mjwarp/claim_test_inventory.yaml"


def _raw() -> dict:
    return copy.deepcopy(OmegaConf.to_container(OmegaConf.load(INVENTORY_PATH), resolve=False))


def _claim_raw() -> dict:
    return copy.deepcopy(
        OmegaConf.to_container(OmegaConf.load(CLAIM_INVENTORY_PATH), resolve=False)
    )


def _capability(raw: dict, capability_id: str) -> dict:
    return next(
        capability
        for capability in raw["capabilities"]
        if capability["capability_id"] == capability_id
    )


def _errors(raw: dict) -> tuple[str, ...]:
    with pytest.raises(DrInventoryValidationError) as exc_info:
        parse_mjwarp_dr_inventory(raw)
    return exc_info.value.errors


def test_valid_inventory_parses_without_advertising_support() -> None:
    inventory = parse_mjwarp_dr_inventory(_raw())

    assert len(inventory.capabilities) == 16
    assert all(
        "supported" not in capability.support_state.value for capability in inventory.capabilities
    )
    assert inventory.source.commit == "f643d245303ff439a90f37151056ff987bdb95f7"


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda raw: raw.update({"unknown": True}), "inventory: unknown key `unknown`"),
        (lambda raw: raw.pop("source"), "inventory: missing key `source`"),
        (
            lambda raw: raw["source"].update({"commit": "short"}),
            "expected full 40-character SHA",
        ),
        (
            lambda raw: raw["source"].update({"version": "future"}),
            "source.version: expected",
        ),
        (
            lambda raw: raw["source"]["dependencies"].pop(),
            "does not match the frozen mjlab dependency baseline",
        ),
        (
            lambda raw: raw["source"]["inspected_files"].pop(),
            "does not match the frozen review surface",
        ),
        (
            lambda raw: raw["required_capability_ids"].pop(),
            "does not match the frozen managed MuJoCo/MJWarp rollout",
        ),
        (
            lambda raw: raw["derived_field_sets"]["set_const_0"].pop(),
            "does not match the frozen mjlab recompute contract",
        ),
        (
            lambda raw: _capability(raw, "joint.armature")["derived_fields"].pop(),
            "derived fields do not match set_const_0",
        ),
        (
            lambda raw: _capability(raw, "actuator.pd_gains")["graph"]["invalidations"].remove(
                "step_graph"
            ),
            "invalidations must be exact",
        ),
        (
            lambda raw: _capability(raw, "actuator.pd_gains")["graph"]["invalidations"].append(
                "unknown_graph"
            ),
            "invalidations must be exact",
        ),
        (
            lambda raw: _capability(raw, "state.qpos_qvel_reset")["graph"]["invalidations"].append(
                "step_graph"
            ),
            "native Data writes cannot declare graph invalidations",
        ),
        (
            lambda raw: _capability(raw, "global.gravity").update(
                {"direct_fields": ["opt.gravity"]}
            ),
            "unresolved capability cannot assert direct fields",
        ),
        (
            lambda raw: _capability(raw, "global.gravity")["storage"].update(
                {"bytes_formula": "nworld * 3 * 4"}
            ),
            "unresolved byte formula must be unknown",
        ),
        (
            lambda raw: _capability(raw, "body.coupled_inertia")["recompute"].update(
                {"scope": "selected_rows"}
            ),
            "recompute scope must be 'all_worlds'",
        ),
        (
            lambda raw: _capability(raw, "global.gravity").update(
                {"support_state": "phase6_candidate"}
            ),
            "support states do not match the frozen rollout decisions",
        ),
        (
            lambda raw: raw["exclusions"].pop(),
            "does not match the frozen deferred and unsupported scope",
        ),
        (
            lambda raw: _capability(raw, "geom.friction")["storage"].update(
                {"derived_elements_per_world": "ngeom"}
            ),
            "empty derived fields require a zero storage formula",
        ),
        (
            lambda raw: _capability(raw, "geom.friction")["required_tests"].update(
                {"physics_effect": "tests/dr/test_field_write.py"}
            ),
            "expected an explicit pytest node",
        ),
        (
            lambda raw: _capability(raw, "geom.friction")["storage"].update(
                {"bytes_formula": "${oc.env:UNSAFE}"}
            ),
            "interpolation is not allowed",
        ),
    ],
)
def test_inventory_rejects_incomplete_or_ambiguous_field_contract(mutate, message: str) -> None:
    raw = _raw()
    mutate(raw)

    assert any(message in error for error in _errors(raw))


def test_inventory_rejects_duplicate_capability_and_legacy_owner() -> None:
    raw = _raw()
    raw["capabilities"].append(copy.deepcopy(raw["capabilities"][0]))
    _capability(raw, "state.velocity_impulse")["legacy_terms"].append("kp")

    errors = _errors(raw)

    assert any("duplicate IDs" in error for error in errors)
    assert any("duplicate ownership" in error for error in errors)


def test_claim_cross_reference_rejects_field_write_as_physics_effect() -> None:
    inventory = parse_mjwarp_dr_inventory(_raw())
    claim_raw = _claim_raw()
    effect_entry = next(
        entry
        for entry in claim_raw["entries"]
        if entry["test_id"]
        == "tests/dr/test_mjwarp_physics_effect.py::test_each_supported_mutation_has_next_step_effect"
    )
    effect_entry["evidence_kind"] = "contract"
    claim_inventory = parse_claim_gap_inventory(claim_raw)

    errors = inventory_claim_gap_errors(inventory, claim_inventory)

    assert any("requires effect evidence" in error for error in errors)


def test_claim_cross_reference_rejects_missing_or_mismatched_test() -> None:
    raw = _raw()
    capability = _capability(raw, "geom.friction")
    capability["storage"]["measurement_test_id"] = (
        "tests/benchmark/test_other.py::test_other_memory"
    )
    capability["required_tests"]["physics_effect"] = "tests/dr/test_other.py::test_other_effect"
    inventory = parse_mjwarp_dr_inventory(raw)
    claim_inventory = parse_claim_gap_inventory(_claim_raw())

    errors = inventory_claim_gap_errors(inventory, claim_inventory)

    assert any("absent from the claim-gap" in error for error in errors)
    assert any("storage measurement test must equal" in error for error in errors)


def test_claim_cross_reference_requires_fault_oracle_for_exclusions() -> None:
    inventory = parse_mjwarp_dr_inventory(_raw())
    claim_raw = _claim_raw()
    unsupported_entry = next(
        entry
        for entry in claim_raw["entries"]
        if entry["test_id"]
        == "tests/base/test_mjwarp_capabilities.py::test_unsupported_matrix_fails_before_step"
    )
    unsupported_entry["evidence_kind"] = "contract"
    claim_inventory = parse_claim_gap_inventory(claim_raw)

    errors = inventory_claim_gap_errors(inventory, claim_inventory)

    assert any("unsupported boundary requires fault evidence" in error for error in errors)


def test_load_inventory_normalizes_invalid_yaml(tmp_path: Path) -> None:
    path = tmp_path / "inventory.yaml"
    path.write_text("capabilities: [", encoding="utf-8")

    with pytest.raises(DrInventoryValidationError, match="cannot load YAML"):
        load_mjwarp_dr_inventory(path)


def test_parser_does_not_execute_inventory_strings(tmp_path: Path) -> None:
    raw = _raw()
    marker = tmp_path / "must-not-exist"
    _capability(raw, "geom.friction")["notes"] = f"touch {marker}"

    parse_mjwarp_dr_inventory(raw)

    assert not marker.exists()
