from __future__ import annotations

import copy
from pathlib import Path

import pytest

from unilab.tools.phase_acceptance import (
    AcceptanceLane,
    ClaimStatus,
    ManifestValidationError,
    parse_phase_acceptance,
    phase_gate_errors,
)


def _claim() -> dict:
    return {
        "claim_id": "P0-CONTRACT",
        "expected": "The contract is enforced.",
        "risk": "Invalid evidence passes.",
        "owner": "tooling",
        "oracle": "Independent validator tests.",
        "commands": ["uv run pytest tests/tools/test_phase_acceptance.py"],
        "lane": "A",
        "required": True,
        "required_test_ids": ["test:contract"],
        "environment": {
            "dependencies": ["uv.lock"],
            "hardware": "any",
            "owner_yaml": None,
            "seeds": [],
            "batch_sizes": [],
            "dtype": None,
            "plan_fingerprint": "contract-v1",
        },
        "acceptance": {
            "tolerance": {},
            "thresholds": {},
            "repetitions": 1,
            "max_dispersion": 0,
            "failure_semantics": "Any mismatch is FAIL.",
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
            "paths": ["src/unilab/tools/phase_acceptance.py"],
            "capabilities": ["phase-acceptance"],
            "fingerprints": ["contract-v1"],
        },
        "status": "planned",
    }


def _manifest() -> dict:
    return {
        "schema_version": 1,
        "issue": 705,
        "phase": 0,
        "integration_branch": "feat/manager-mjwarp-manager-mjwarp",
        "required_lanes": ["A"],
        "claims": [_claim()],
    }


def _errors(raw: dict) -> tuple[str, ...]:
    with pytest.raises(ManifestValidationError) as exc_info:
        parse_phase_acceptance(raw)
    return exc_info.value.errors


def test_valid_planned_manifest_passes_schema_and_fails_gate() -> None:
    manifest = parse_phase_acceptance(_manifest())

    assert manifest.required_lanes == (AcceptanceLane.PR,)
    assert manifest.claims[0].status == ClaimStatus.PLANNED
    assert phase_gate_errors(manifest) == (
        "P0-CONTRACT: required claim is planned, expected verified/promoted",
    )


def test_verified_manifest_passes_gate_with_complete_evidence() -> None:
    raw = _manifest()
    claim = raw["claims"][0]
    claim["status"] = "verified"
    claim["evidence"] = {
        "result": "PASS",
        "artifact_refs": ["https://github.com/unilabsim/UniLab/pull/708"],
        "commit_sha": "a" * 40,
        "config_hash": f"sha256:{'b' * 64}",
        "executed_test_ids": ["test:contract"],
        "skipped_test_ids": [],
        "xfailed_test_ids": [],
        "summary": "Contract test passed.",
    }

    manifest = parse_phase_acceptance(raw)

    assert phase_gate_errors(manifest) == ()


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda raw: raw.update({"unknown": True}), "manifest: unknown key `unknown`"),
        (lambda raw: raw.pop("issue"), "manifest: missing key `issue`"),
        (lambda raw: raw.update({"schema_version": 2}), "schema_version: expected 1"),
        (lambda raw: raw.update({"required_lanes": ["Z"]}), "expected one of"),
        (lambda raw: raw["claims"][0].update({"status": "done"}), "expected one of"),
        (lambda raw: raw["claims"][0].update({"oracle": ""}), "oracle: must not be empty"),
        (lambda raw: raw["claims"][0].update({"commands": []}), "commands: must not be empty"),
        (
            lambda raw: raw["claims"][0]["invalidation"].update({"paths": ["../outside"]}),
            "must be a repository-relative path",
        ),
    ],
)
def test_manifest_rejects_invalid_contract(mutate, message: str) -> None:
    raw = _manifest()
    mutate(raw)

    assert any(message in error for error in _errors(raw))


def test_manifest_rejects_duplicate_claim_ids() -> None:
    raw = _manifest()
    raw["claims"].append(copy.deepcopy(raw["claims"][0]))

    assert any("duplicate claim IDs" in error for error in _errors(raw))


def test_verified_manifest_rejects_missing_or_skipped_required_test() -> None:
    raw = _manifest()
    claim = raw["claims"][0]
    claim["status"] = "verified"
    claim["evidence"] = {
        "result": "PASS",
        "artifact_refs": ["artifacts/result.json"],
        "commit_sha": "a" * 40,
        "config_hash": f"sha256:{'b' * 64}",
        "executed_test_ids": [],
        "skipped_test_ids": ["test:contract"],
        "xfailed_test_ids": [],
        "summary": "Incomplete run.",
    }

    errors = _errors(raw)

    assert any("was not executed" in error for error in errors)
    assert any("was skipped" in error for error in errors)


def test_implemented_manifest_rejects_malformed_evidence_hashes() -> None:
    raw = _manifest()
    claim = raw["claims"][0]
    claim["status"] = "implemented"
    claim["evidence"]["commit_sha"] = "short"
    claim["evidence"]["config_hash"] = "sha256:not-a-hash"

    errors = _errors(raw)

    assert any("full 40-character SHA" in error for error in errors)
    assert any("sha256:<64 hex>" in error for error in errors)


def test_planned_manifest_rejects_executed_evidence() -> None:
    raw = _manifest()
    raw["claims"][0]["evidence"]["executed_test_ids"] = ["test:contract"]

    assert any("planned claim must not carry pass artifacts" in error for error in _errors(raw))


def test_manifest_rejects_non_https_artifact_url() -> None:
    raw = _manifest()
    claim = raw["claims"][0]
    claim["status"] = "implemented"
    claim["evidence"]["artifact_refs"] = ["http://example.invalid/result.json"]

    assert any("only HTTPS artifact URLs" in error for error in _errors(raw))


def test_manifest_rejects_required_lane_mismatch() -> None:
    raw = _manifest()
    raw["required_lanes"] = ["A", "D"]

    assert any(
        "must exactly match lanes used by required claims" in error for error in _errors(raw)
    )


def test_manifest_rejects_nonfinite_or_negative_tolerance() -> None:
    raw = _manifest()
    raw["claims"][0]["acceptance"]["tolerance"] = {
        "nan": float("nan"),
        "negative": -1.0,
    }

    errors = _errors(raw)

    assert any("tolerance.nan: must be finite" in error for error in errors)
    assert any("tolerance.negative: must be >= 0.0" in error for error in errors)


def test_manifest_reports_non_string_numeric_key_without_crashing() -> None:
    raw = _manifest()
    raw["claims"][0]["acceptance"]["thresholds"] = {"valid": 1, 2: 3}

    assert any("thresholds.<key>: expected string" in error for error in _errors(raw))


def test_verified_manifest_rejects_any_claim_xfail() -> None:
    raw = _manifest()
    claim = raw["claims"][0]
    claim["status"] = "verified"
    claim["evidence"] = {
        "result": "PASS",
        "artifact_refs": ["https://github.com/unilabsim/UniLab/pull/708"],
        "commit_sha": "a" * 40,
        "config_hash": f"sha256:{'b' * 64}",
        "executed_test_ids": ["test:contract"],
        "skipped_test_ids": [],
        "xfailed_test_ids": ["test:unrelated"],
        "summary": "Claim test passed but another claim test xfailed.",
    }

    assert any("verified claim cannot xfail tests" in error for error in _errors(raw))


def test_manifest_rejects_interpolation_without_resolving_it() -> None:
    raw = _manifest()
    raw["claims"][0]["commands"] = ["${oc.env:UNSAFE_COMMAND}"]

    assert any("interpolation is not allowed" in error for error in _errors(raw))


def test_parser_does_not_execute_declared_command(tmp_path: Path) -> None:
    raw = _manifest()
    marker = tmp_path / "must-not-exist"
    raw["claims"][0]["commands"] = [f"touch {marker}"]

    parse_phase_acceptance(raw)

    assert not marker.exists()
