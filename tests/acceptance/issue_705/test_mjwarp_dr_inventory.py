from __future__ import annotations

from pathlib import Path

from unilab.tools.claim_gap_audit import load_claim_gap_inventory
from unilab.tools.mjwarp_dr_inventory import (
    DERIVED_FIELD_SETS,
    EXPECTED_CAPABILITY_IDS,
    EXPECTED_LEGACY_TERMS,
    MODEL_EXPANSION_INVALIDATIONS,
    ExpansionKind,
    FieldResolution,
    GraphImpact,
    SupportState,
    inventory_claim_gap_errors,
    load_mjwarp_dr_inventory,
)

ACCEPTANCE_DIR = Path(__file__).resolve().parent
INVENTORY_PATH = ACCEPTANCE_DIR / "mjwarp_dr_inventory.yaml"
CLAIM_INVENTORY_PATH = ACCEPTANCE_DIR / "claim_test_inventory.yaml"


def test_inventory_covers_required_field_contracts() -> None:
    inventory = load_mjwarp_dr_inventory(INVENTORY_PATH)
    claim_inventory = load_claim_gap_inventory(CLAIM_INVENTORY_PATH)

    assert {capability.capability_id for capability in inventory.capabilities} == set(
        EXPECTED_CAPABILITY_IDS
    )
    assert {
        term for capability in inventory.capabilities for term in capability.legacy_terms
    } == set(EXPECTED_LEGACY_TERMS)
    assert dict(inventory.derived_field_sets) == DERIVED_FIELD_SETS
    assert inventory_claim_gap_errors(inventory, claim_inventory) == ()


def test_model_expansion_contract_includes_storage_graph_and_effect_oracles() -> None:
    inventory = load_mjwarp_dr_inventory(INVENTORY_PATH)
    expanded = [
        capability
        for capability in inventory.capabilities
        if capability.storage.expansion == ExpansionKind.MODEL_FIELD_EXPAND
    ]

    assert expanded
    for capability in expanded:
        assert capability.field_resolution == FieldResolution.RESOLVED
        assert capability.direct_fields
        assert capability.graph.impact == GraphImpact.RECAPTURE_REQUIRED
        assert set(capability.graph.invalidations) == set(MODEL_EXPANSION_INVALIDATIONS)
        assert capability.required_tests.physics_effect.endswith(
            "::test_each_supported_mutation_has_next_step_effect"
        )
        assert capability.storage.measurement_test_id == capability.required_tests.memory


def test_inventory_is_a_plan_and_not_a_support_advertisement() -> None:
    inventory = load_mjwarp_dr_inventory(INVENTORY_PATH)
    states = {capability.support_state for capability in inventory.capabilities}

    assert states == {
        SupportState.PHASE2_REQUIRED,
        SupportState.PHASE2_DECISION,
        SupportState.PHASE6_CANDIDATE,
        SupportState.COLD_PATH_ONLY,
        SupportState.BLOCKED_PENDING_EVIDENCE,
    }
    gravity = next(
        capability
        for capability in inventory.capabilities
        if capability.capability_id == "global.gravity"
    )
    assert gravity.field_resolution == FieldResolution.UNRESOLVED
    assert not gravity.direct_fields


def test_benchmark_silent_payload_drop_is_recorded_only_as_negative_evidence() -> None:
    inventory = load_mjwarp_dr_inventory(INVENTORY_PATH)
    gains = next(
        capability
        for capability in inventory.capabilities
        if capability.capability_id == "actuator.pd_gains"
    )

    assert gains.support_state == SupportState.PHASE2_DECISION
    assert "silently discards" in gains.notes
    assert any("benchmark/mjwarp/backend.py" in reference for reference in gains.evidence_refs)
