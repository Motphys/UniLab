"""Release acceptance for the Issue #705 mjwarp legacy-route retirement."""

from __future__ import annotations

from pathlib import Path

from unilab.tools.issue705_legacy_retirement import (
    EVIDENCE_PATH,
    PLAN_PATH,
    ROLLBACK_PATH,
    audit_legacy_retirement,
    load_legacy_retirement_evidence,
    load_legacy_retirement_plan,
    load_rollback_receipt,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_legacy_removal_requires_full_entrypoint_and_rollback_evidence() -> None:
    plan = load_legacy_retirement_plan(REPO_ROOT / PLAN_PATH)
    rollback = load_rollback_receipt(REPO_ROOT / ROLLBACK_PATH)
    evidence = load_legacy_retirement_evidence(REPO_ROOT / EVIDENCE_PATH)

    report = audit_legacy_retirement(plan, rollback, evidence, root=REPO_ROOT)

    assert report.ok, report.errors
    assert report.changed_paths == len(plan.implementation_paths)
    assert report.entrypoint_repetitions == plan.entrypoint_prerequisite.repetitions == 2
    assert report.retained_routes == len(plan.retained_routes) == 3
