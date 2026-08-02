from __future__ import annotations

import pytest

pytestmark = pytest.mark.slow

from pathlib import Path

from unilab.tools.issue705_thresholds import (
    FREEZE_RECEIPT_PATH,
    THRESHOLD_MANIFEST_PATH,
    load_freeze_receipt,
    load_threshold_manifest,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_phase0_thresholds_are_frozen() -> None:
    manifest = load_threshold_manifest(
        REPO_ROOT / THRESHOLD_MANIFEST_PATH,
        repo_root=REPO_ROOT,
    )
    receipt = load_freeze_receipt(
        REPO_ROOT / FREEZE_RECEIPT_PATH,
        manifest=manifest,
        repo_root=REPO_ROOT,
    )

    assert manifest.data["state"] == "frozen"
    assert manifest.data["baseline"]["source_commit"] == (
        "aa0a8e723e73e18d8b1b850eef7adfb442ef1bbb"
    )
    assert manifest.data["measurement"]["batch_sizes"] == [128, 1024, 4096]
    assert manifest.data["measurement"]["ppo_seeds"] == [0, 1, 2, 3, 4]
    assert manifest.gates["performance"]["p50_latency_ratio_max"] == 1.05
    assert manifest.gates["transfer"]["h2d_per_policy_step_max"] == 0.0
    assert receipt.freeze_commit == "a2419b342b8663998b2e29cf20a4dce49b3127f5"
    assert receipt.data["final_merge_method"] == "merge_commit"
