"""Parser, provenance, and fault audits for Issue #705 legacy retirement."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import pytest
from omegaconf import OmegaConf
from scripts import audit_issue705_legacy_retirement

from unilab.tools import issue705_legacy_retirement
from unilab.tools.issue705_legacy_retirement import (
    EVIDENCE_PATH,
    PLAN_PATH,
    ROLLBACK_PATH,
    LegacyRetirementAuditReport,
    LegacyRetirementError,
    LegacyRetirementPlan,
    RollbackReceipt,
    audit_legacy_retirement,
    load_legacy_retirement_evidence,
    load_legacy_retirement_plan,
    load_rollback_receipt,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _plan() -> LegacyRetirementPlan:
    return load_legacy_retirement_plan(REPO_ROOT / PLAN_PATH)


def _rollback() -> RollbackReceipt:
    return load_rollback_receipt(REPO_ROOT / ROLLBACK_PATH)


def _evidence() -> dict[str, Any]:
    return load_legacy_retirement_evidence(REPO_ROOT / EVIDENCE_PATH)


def _audit(
    *,
    plan: LegacyRetirementPlan | None = None,
    rollback: RollbackReceipt | None = None,
    evidence: Mapping[str, Any] | None = None,
) -> LegacyRetirementAuditReport:
    return audit_legacy_retirement(
        plan or _plan(),
        rollback or _rollback(),
        evidence or _evidence(),
        root=REPO_ROOT,
    )


def _assert_error(report: LegacyRetirementAuditReport, fragment: str) -> None:
    assert not report.ok
    assert any(fragment in error for error in report.errors), report.errors


@pytest.mark.parametrize(
    ("mutation", "fragment"), [("missing", "missing key"), ("unknown", "unknown key")]
)
def test_plan_parser_rejects_missing_and_unknown_keys(
    tmp_path: Path, mutation: str, fragment: str
) -> None:
    raw = OmegaConf.to_container(OmegaConf.load(REPO_ROOT / PLAN_PATH), resolve=True)
    assert isinstance(raw, dict)
    if mutation == "missing":
        raw.pop("claim_id")
    else:
        raw["unregistered"] = True
    path = tmp_path / "plan.yaml"
    OmegaConf.save(config=OmegaConf.create(raw), f=path)

    with pytest.raises(LegacyRetirementError, match=fragment):
        load_legacy_retirement_plan(path)


def test_rollback_receipt_rejects_hash_tamper() -> None:
    receipt = _rollback()
    files = list(receipt.baseline_files)
    files[0] = (files[0][0], "sha256:" + "0" * 64)

    report = _audit(rollback=replace(receipt, baseline_files=tuple(files)))

    _assert_error(report, "rollback baseline file/hash set")
    _assert_error(report, "rollback baseline hash differs")


def test_evidence_rejects_source_and_input_freshness_tamper() -> None:
    source_tamper = deepcopy(_evidence())
    source_tamper["source"]["commit_sha"] = "0" * 40
    _assert_error(_audit(evidence=source_tamper), "does not identify a Git commit")

    input_tamper = deepcopy(_evidence())
    input_path = next(iter(input_tamper["inputs"]))
    input_tamper["inputs"][input_path] = "sha256:" + "0" * 64
    _assert_error(_audit(evidence=input_tamper), "evidence input is stale")


def test_evidence_rejects_command_repetition_and_nonpass_tamper() -> None:
    argv_tamper = deepcopy(_evidence())
    argv_tamper["commands"][0]["argv"].append("--maxfail=1")
    _assert_error(_audit(evidence=argv_tamper), "frozen entrypoint command")

    repetition_tamper = deepcopy(_evidence())
    repetition_tamper["commands"][1]["repetition"] = 1
    _assert_error(_audit(evidence=repetition_tamper), "repetition: expected 2")

    for category in ("skipped", "xfailed", "xpassed", "deselected"):
        outcome_tamper = deepcopy(_evidence())
        outcome_tamper["commands"][0]["pytest"]["passed"] = 0
        outcome_tamper["commands"][0]["pytest"][category] = 1
        _assert_error(_audit(evidence=outcome_tamper), "one pass and no non-pass outcome")


def test_audit_rejects_implementation_scope_tamper() -> None:
    plan = _plan()
    tampered = replace(plan, implementation_paths=plan.implementation_paths[:-1])

    report = _audit(plan=tampered)

    _assert_error(report, "implementation_paths differs from the frozen retirement scope")
    _assert_error(report, "implementation diff paths differ")


def test_audit_rejects_implementation_scope_commit_tamper() -> None:
    plan = _plan()
    tampered = replace(
        plan,
        implementation_scope_commit=plan.integration_base.commit,
    )

    report = _audit(plan=tampered)

    _assert_error(report, "implementation_scope_commit differs from the frozen implementation")
    _assert_error(report, "implementation diff paths differ")


@pytest.mark.parametrize(
    ("old", "new", "fragment"),
    [
        (
            'registry.register_env("G1WalkFlat", G1MjwarpManagedEnv, sim_backend="mjwarp")',
            'registry.register_env("G1WalkFlat", G1WalkEnv, sim_backend="mjwarp")',
            "does not bind mjwarp to the managed-only replacement",
        ),
        (
            "class G1MjwarpManagedEnv(NpEnv):",
            "class G1MjwarpManagedEnv(G1WalkEnv):",
            "replacement class base must be exactly",
        ),
        (
            'self._reject_legacy_lifecycle("step")',
            "self._backend.step(actions, 1)",
            "must only call the typed legacy rejection helper",
        ),
    ],
)
def test_audit_rejects_registry_inheritance_and_fallback_tamper(
    monkeypatch: pytest.MonkeyPatch,
    old: str,
    new: str,
    fragment: str,
) -> None:
    evidence = _evidence()
    implementation_commit = _plan().implementation_scope_commit
    original_git_blob = issue705_legacy_retirement._git_blob

    def tampered_git_blob(root: Path, commit: str, path: Path) -> bytes:
        blob = original_git_blob(root, commit, path)
        if commit == implementation_commit and path == issue705_legacy_retirement.OWNER_MODULE_PATH:
            source = blob.decode()
            assert old in source
            return source.replace(old, new, 1).encode()
        return blob

    monkeypatch.setattr(issue705_legacy_retirement, "_git_blob", tampered_git_blob)

    _assert_error(_audit(evidence=evidence), fragment)


def test_audit_rejects_owner_identity_tamper(monkeypatch: pytest.MonkeyPatch) -> None:
    original_load = issue705_legacy_retirement.OmegaConf.load

    def tampered_load(path: Path):
        config = original_load(path)
        if Path(path).resolve() == (REPO_ROOT / issue705_legacy_retirement.OWNER_PATH).resolve():
            config.entrypoints.identity.backend = "mujoco"
        return config

    monkeypatch.setattr(issue705_legacy_retirement.OmegaConf, "load", tampered_load)

    _assert_error(_audit(), "entrypoint identity differs")


def test_legacy_retirement_audit_cli_reports_pass_and_fail(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    passing = LegacyRetirementAuditReport(
        changed_paths=11,
        entrypoint_repetitions=2,
        retained_routes=3,
        errors=(),
    )
    monkeypatch.setattr(
        audit_issue705_legacy_retirement,
        "audit_legacy_retirement",
        lambda *args, **kwargs: passing,
    )
    assert audit_issue705_legacy_retirement.main([]) == 0
    assert "PASS Issue #705 legacy retirement audit" in capsys.readouterr().out

    failing = replace(passing, errors=("synthetic retirement fault",))
    monkeypatch.setattr(
        audit_issue705_legacy_retirement,
        "audit_legacy_retirement",
        lambda *args, **kwargs: failing,
    )
    assert audit_issue705_legacy_retirement.main(["--json"]) == 1
    output = capsys.readouterr().out
    assert '"ok": false' in output
    assert "synthetic retirement fault" in output
