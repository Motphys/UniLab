"""Repository and fault audits for the Issue #705 task rollout plan."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from scripts import audit_issue705_task_rollout

from unilab.tools import issue705_task_rollout
from unilab.tools.issue705_support import CompiledSignature, snapshot_registry_backends
from unilab.tools.issue705_task_rollout import (
    ROLLOUT_PLAN_PATH,
    TaskRolloutAuditReport,
    TaskRolloutEntry,
    TaskRolloutPlan,
    audit_task_rollout_plan,
    load_task_rollout_plan,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _plan() -> TaskRolloutPlan:
    return load_task_rollout_plan(REPO_ROOT / ROLLOUT_PLAN_PATH)


def _replace_entry(plan: TaskRolloutPlan, **changes: object) -> TaskRolloutPlan:
    entry = replace(plan.entries[0], **changes)
    return replace(plan, entries=(entry,))


def _audit(
    plan: TaskRolloutPlan,
    *,
    registry_backends: dict[str, set[str]] | None = None,
) -> TaskRolloutAuditReport:
    return audit_task_rollout_plan(
        plan,
        root=REPO_ROOT,
        registry_backends=registry_backends,
        validate_support_payloads=False,
    )


def _assert_error(report: TaskRolloutAuditReport, fragment: str) -> None:
    assert not report.ok
    assert any(fragment in error for error in report.errors), report.errors


def test_task_rollout_plan_has_fresh_bidirectional_prerequisites() -> None:
    report = _audit(_plan())

    assert report.ok, report.errors
    assert report.entries == 1
    assert report.prerequisites == 27


def test_task_rollout_audit_rejects_owner_registry_budget_and_signature_faults() -> None:
    plan = _plan()
    entry = plan.entries[0]

    bad_owner = _replace_entry(plan, owner_yaml_sha256="sha256:" + "0" * 64)
    _assert_error(_audit(bad_owner), "owner_yaml_sha256 differs from support evidence")

    registry_snapshot = snapshot_registry_backends()
    missing_registry = {name: set(backends) for name, backends in registry_snapshot.items()}
    missing_registry[entry.env_name].discard(entry.backend)
    _assert_error(
        _audit(plan, registry_backends=missing_registry),
        "env/backend identity is not registered",
    )

    bad_seeds = _replace_entry(plan, seeds=(0, 2))
    _assert_error(_audit(bad_seeds), "frozen rollout seeds")

    bad_budget = _replace_entry(plan, num_envs=64)
    _assert_error(_audit(bad_budget), "frozen rollout budget")

    signature: CompiledSignature = entry.support_compiled_signature
    bad_signature = _replace_entry(
        plan,
        support_compiled_signature=replace(
            signature,
            policy_abi_fingerprint="managed-policy-abi-v1:" + "0" * 64,
        ),
    )
    _assert_error(_audit(bad_signature), "support_compiled_signature differs from support evidence")

    rollout_signature = entry.rollout_compiled_signature
    bad_rollout_signature = _replace_entry(
        plan,
        rollout_compiled_signature=replace(
            rollout_signature,
            policy_abi_fingerprint="managed-policy-abi-v1:" + "0" * 64,
        ),
    )
    _assert_error(_audit(bad_rollout_signature), "rollout and support compiled policy_abi_fingerprint")


def test_task_rollout_audit_rejects_prerequisite_and_phase_receipt_faults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan()
    entry: TaskRolloutEntry = plan.entries[0]
    missing = _replace_entry(plan, prerequisites=entry.prerequisites[:-1])
    _assert_error(_audit(missing), "prerequisite claim/test matrix")

    wrong_test = replace(
        entry.prerequisites[0],
        test_id="tests/base/test_mjwarp_identity.py::missing_test_node",
    )
    tampered = _replace_entry(plan, prerequisites=(wrong_test, *entry.prerequisites[1:]))
    _assert_error(_audit(tampered), "prerequisite claim/test matrix")

    load_manifest = issue705_task_rollout.load_phase_acceptance

    def load_with_stale_execution(path: Path):
        manifest = load_manifest(path)
        if manifest.phase != 5:
            return manifest
        claims = list(manifest.claims)
        claims[0] = replace(
            claims[0],
            evidence=replace(claims[0].evidence, executed_test_ids=()),
        )
        return replace(manifest, claims=tuple(claims))

    monkeypatch.setattr(
        issue705_task_rollout,
        "load_phase_acceptance",
        load_with_stale_execution,
    )
    _assert_error(_audit(plan), "executed test differs from rollout plan")


def test_task_rollout_audit_rejects_missing_exact_pytest_node(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan()
    missing_id = plan.entries[0].prerequisites[0].test_id
    exists = issue705_task_rollout.test_node_exists
    monkeypatch.setattr(
        issue705_task_rollout,
        "test_node_exists",
        lambda root, test_id: False if test_id == missing_id else exists(root, test_id),
    )

    _assert_error(_audit(plan), "exact pytest node does not exist")


def test_task_rollout_audit_cli_reports_pass_and_fail(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    passing = TaskRolloutAuditReport(entries=1, prerequisites=27, errors=())
    monkeypatch.setattr(
        audit_issue705_task_rollout,
        "audit_task_rollout_plan",
        lambda *args, **kwargs: passing,
    )
    assert audit_issue705_task_rollout.main([]) == 0
    assert "PASS Issue #705 task rollout audit" in capsys.readouterr().out

    failing = replace(passing, errors=("synthetic rollout fault",))
    monkeypatch.setattr(
        audit_issue705_task_rollout,
        "audit_task_rollout_plan",
        lambda *args, **kwargs: failing,
    )
    assert audit_issue705_task_rollout.main(["--json"]) == 1
    output = capsys.readouterr().out
    assert '"ok": false' in output
    assert "synthetic rollout fault" in output
