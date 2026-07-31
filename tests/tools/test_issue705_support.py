"""Schema and parser tests for Issue #705 support evidence."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from unilab.tools.issue705_support import (
    BENCHMARK_TEST_ID,
    CLAIM_ID,
    REQUIRED_PHASES,
    SUPPORT_EVIDENCE_PATH,
    DeclaredEvidenceLevel,
    SupportEvidenceError,
    load_support_evidence,
    parse_support_evidence,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _raw_manifest() -> dict[str, object]:
    raw = OmegaConf.to_container(OmegaConf.load(REPO_ROOT / SUPPORT_EVIDENCE_PATH), resolve=False)
    assert isinstance(raw, dict)
    return raw


def test_support_evidence_parser_loads_exact_phase7a_declaration() -> None:
    manifest = load_support_evidence(REPO_ROOT / SUPPORT_EVIDENCE_PATH)

    assert manifest.issue == 705
    assert manifest.claim_id == CLAIM_ID
    assert tuple(gate.phase for gate in manifest.phase_gates) == REQUIRED_PHASES
    assert len(manifest.combinations) == 1
    combination = manifest.combinations[0]
    assert combination.key == ("ppo_torch", "g1_walk_flat", "mjwarp")
    assert combination.env_name == "G1WalkFlat"
    assert combination.execution_profile == "device_resident"
    assert combination.evidence_level == DeclaredEvidenceLevel.BENCHMARKED
    assert combination.required_phase == 5
    assert combination.mandatory_test_ids == (BENCHMARK_TEST_ID,)
    assert combination.benchmark is not None
    assert combination.compiled_signature is not None
    assert combination.compiled_signature.task_key == "g1_walk_flat.managed_device"
    assert combination.compiled_signature.backend_plan_fingerprint.startswith(
        "mjwarp-device-batch-v1:"
    )


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda raw: raw.update({"unexpected": True}), "unknown key"),
        (
            lambda raw: raw["phase_gates"].append(deepcopy(raw["phase_gates"][0])),
            "duplicate phases",
        ),
        (
            lambda raw: raw["combinations"].append(deepcopy(raw["combinations"][0])),
            "duplicate keys",
        ),
        (
            lambda raw: raw["combinations"][0].update({"owner_yaml": "../owner.yaml"}),
            "stay within",
        ),
        (
            lambda raw: raw["combinations"][0].update({"owner_yaml_sha256": "bad"}),
            "expected sha256",
        ),
        (
            lambda raw: raw["combinations"][0].update({"evidence_level": "supported"}),
            "expected one of",
        ),
        (
            lambda raw: raw["combinations"][0].update({"mandatory_test_ids": []}),
            "non-empty string list",
        ),
    ],
)
def test_support_evidence_parser_rejects_schema_faults(mutation, match: str) -> None:
    raw = _raw_manifest()
    mutation(raw)

    with pytest.raises(SupportEvidenceError, match=match):
        parse_support_evidence(raw)


def test_support_evidence_loader_normalizes_malformed_yaml(tmp_path: Path) -> None:
    path = tmp_path / "support.yaml"
    path.write_text("combinations: [", encoding="utf-8")

    with pytest.raises(SupportEvidenceError, match="cannot load YAML"):
        load_support_evidence(path)
