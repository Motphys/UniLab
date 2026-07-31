"""Repository and synthetic fault audits for Issue #705 support claims."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from scripts import audit_issue705_support

from unilab.tools import issue705_support
from unilab.tools.issue705_support import (
    SUPPORT_EVIDENCE_PATH,
    CompiledSignature,
    DeclaredEvidenceLevel,
    SupportAuditReport,
    SupportCombination,
    SupportEvidenceManifest,
    audit_support_evidence,
    load_support_evidence,
    snapshot_registry_backends,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _manifest() -> SupportEvidenceManifest:
    return load_support_evidence(REPO_ROOT / SUPPORT_EVIDENCE_PATH)


def _replace_combination(
    manifest: SupportEvidenceManifest, **changes: object
) -> SupportEvidenceManifest:
    combination = replace(manifest.combinations[0], **changes)
    return replace(manifest, combinations=(combination,))


def _fast_audit(
    manifest: SupportEvidenceManifest,
    *,
    registry_backends: dict[str, set[str]] | None = None,
) -> SupportAuditReport:
    return audit_support_evidence(
        manifest,
        root=REPO_ROOT,
        registry_backends=registry_backends,
        validate_phase_payloads=False,
        validate_benchmark_payloads=False,
    )


def _assert_error(report: SupportAuditReport, fragment: str) -> None:
    assert not report.ok
    assert any(fragment in error for error in report.errors), report.errors


def test_supported_combinations_have_fresh_bidirectional_evidence() -> None:
    report = audit_support_evidence(_manifest(), root=REPO_ROOT)

    assert report.ok, report.errors
    assert report.combinations == 1
    assert report.benchmarked == 1
    assert report.recommended == 0
    assert report.phase_gates == 4


def test_support_audit_rejects_owner_registry_backend_and_profile_faults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _manifest()
    combination = manifest.combinations[0]

    bad_owner = _replace_combination(manifest, owner_yaml_sha256="sha256:" + "0" * 64)
    _assert_error(_fast_audit(bad_owner), "owner YAML hash")

    registry_snapshot = snapshot_registry_backends()
    missing_registry = {name: set(backends) for name, backends in registry_snapshot.items()}
    missing_registry[combination.env_name].discard("mjwarp")
    _assert_error(
        _fast_audit(manifest, registry_backends=missing_registry),
        "declared identities are not registered",
    )

    extra_registry = {name: set(backends) for name, backends in registry_snapshot.items()}
    extra_registry["UnmodeledMjwarpEnv"] = {"mjwarp"}
    _assert_error(
        _fast_audit(manifest, registry_backends=extra_registry),
        "unmodeled mjwarp identities",
    )

    discovered = issue705_support._discover_mjwarp_owner_paths(REPO_ROOT)
    monkeypatch.setattr(
        issue705_support,
        "_discover_mjwarp_owner_paths",
        lambda root: discovered | {Path("conf/ppo/task/unmodeled/mjwarp.yaml")},
    )
    _assert_error(_fast_audit(manifest), "unmodeled mjwarp owner YAML")

    bad_backend = _replace_combination(manifest, backend="mujoco")
    _assert_error(_fast_audit(bad_backend), "currently accepts only mjwarp")

    bad_profile = _replace_combination(manifest, execution_profile="host_numpy")
    _assert_error(_fast_audit(bad_profile), "execution_profile")


def test_support_audit_rejects_phase_test_artifact_level_and_signature_faults() -> None:
    manifest = _manifest()
    combination = manifest.combinations[0]
    assert combination.benchmark is not None
    assert combination.compiled_signature is not None

    stale_gate = replace(
        manifest,
        phase_gates=(
            *manifest.phase_gates[:-1],
            replace(manifest.phase_gates[-1], sha256="sha256:" + "0" * 64),
        ),
    )
    _assert_error(_fast_audit(stale_gate), "gate artifact hash")

    wrong_phase = _replace_combination(manifest, required_phase=6)
    _assert_error(_fast_audit(wrong_phase), "requires Phase 5")

    wrong_test = _replace_combination(
        manifest,
        mandatory_test_ids=(
            "tests/training/test_device_transition_abi.py::"
            "test_dlpack_pointer_shape_dtype_and_lifetime",
        ),
    )
    _assert_error(_fast_audit(wrong_test), "mandatory test mapping is not exact")

    stale_benchmark = _replace_combination(
        manifest,
        benchmark=replace(combination.benchmark, sha256="sha256:" + "0" * 64),
    )
    _assert_error(_fast_audit(stale_benchmark), "benchmark artifact hash")

    copied_benchmark = _replace_combination(
        manifest,
        benchmark=replace(combination.benchmark, path=Path("copied_benchmark.json")),
    )
    _assert_error(_fast_audit(copied_benchmark), "benchmark artifact path is not canonical")

    signature: CompiledSignature = combination.compiled_signature
    stale_signature = _replace_combination(
        manifest,
        compiled_signature=replace(
            signature,
            task_plan_fingerprint="manager-task-contract-v1:" + "0" * 64,
        ),
    )
    _assert_error(_fast_audit(stale_signature), "compiled policy signature differs")


def test_support_audit_accepts_recommended_after_verified_phase7_rollout() -> None:
    recommended = _replace_combination(
        _manifest(), evidence_level=DeclaredEvidenceLevel.RECOMMENDED
    )

    report = _fast_audit(recommended)

    assert report.ok, report.errors
    assert report.recommended == 1


def test_support_audit_rejects_manifest_receipt_that_differs_from_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _manifest()
    load_manifest = issue705_support.load_phase_acceptance

    def load_with_stale_receipt(path: Path):
        phase_manifest = load_manifest(path)
        if phase_manifest.phase != 5:
            return phase_manifest
        claims = list(phase_manifest.claims)
        claims[0] = replace(
            claims[0],
            evidence=replace(claims[0].evidence, commit_sha="0" * 40),
        )
        return replace(phase_manifest, claims=tuple(claims))

    monkeypatch.setattr(issue705_support, "load_phase_acceptance", load_with_stale_receipt)
    _assert_error(_fast_audit(manifest), "manifest commit does not match gate source")


def test_support_audit_cli_reports_pass_and_fail(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    passing = SupportAuditReport(
        combinations=1,
        benchmarked=1,
        recommended=0,
        phase_gates=4,
        errors=(),
    )
    monkeypatch.setattr(audit_issue705_support, "audit_support_evidence", lambda *a, **k: passing)
    assert audit_issue705_support.main([]) == 0
    assert "PASS Issue #705 support audit" in capsys.readouterr().out

    failing = replace(passing, errors=("synthetic fault",))
    monkeypatch.setattr(audit_issue705_support, "audit_support_evidence", lambda *a, **k: failing)
    assert audit_issue705_support.main(["--json"]) == 1
    output = capsys.readouterr().out
    assert '"ok": false' in output
    assert "synthetic fault" in output
