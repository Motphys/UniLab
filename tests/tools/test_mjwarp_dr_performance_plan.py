"""Contract and tamper tests for the frozen Issue #829 benchmark plan."""

from __future__ import annotations

from pathlib import Path

import pytest

from unilab.tools.mjwarp_dr_performance import (
    FREEZE_COMMIT,
    FREEZE_RECEIPT_PATH,
    PLAN_PATH,
    MjwarpDrPerformanceContractError,
    load_mjwarp_dr_performance_freeze_receipt,
    load_mjwarp_dr_performance_plan,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_frozen_plan_and_receipt_bind_the_complete_worker_matrix() -> None:
    plan = load_mjwarp_dr_performance_plan(REPO_ROOT / PLAN_PATH)
    receipt = load_mjwarp_dr_performance_freeze_receipt(
        REPO_ROOT / FREEZE_RECEIPT_PATH,
        plan=plan,
        repo_root=REPO_ROOT,
    )

    assert [profile.profile_id for profile in plan.profiles] == [
        "disabled",
        "tier_b_pd",
        "tier_c_armature",
        "tier_c_mixed",
    ]
    assert plan.reset_worker_count == 240
    assert plan.env_worker_count == 45
    assert plan.train_worker_count == 15
    assert receipt.freeze_commit == FREEZE_COMMIT
    assert receipt.git_history_verified


def test_plan_byte_tamper_is_rejected_before_receipt_validation(tmp_path: Path) -> None:
    tampered = tmp_path / "plan.yaml"
    source = (REPO_ROOT / PLAN_PATH).read_text(encoding="utf-8")
    tampered.write_text(source.replace("absolute_rss_ceiling: null", "absolute_rss_ceiling: 1"))

    with pytest.raises(MjwarpDrPerformanceContractError, match="SHA256 differs"):
        load_mjwarp_dr_performance_plan(tampered)


def test_receipt_tamper_is_rejected_even_without_git_lookup(tmp_path: Path) -> None:
    plan = load_mjwarp_dr_performance_plan(REPO_ROOT / PLAN_PATH)
    tampered = tmp_path / "receipt.yaml"
    source = (REPO_ROOT / FREEZE_RECEIPT_PATH).read_text(encoding="utf-8")
    tampered.write_text(source.replace(FREEZE_COMMIT, "0" * 40), encoding="utf-8")

    with pytest.raises(MjwarpDrPerformanceContractError, match="freeze_commit"):
        load_mjwarp_dr_performance_freeze_receipt(
            tampered,
            plan=plan,
            repo_root=REPO_ROOT,
            verify_git=False,
        )
