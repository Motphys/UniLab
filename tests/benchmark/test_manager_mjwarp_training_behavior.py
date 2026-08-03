"""Acceptance gate for managed MuJoCo/MJWarp rollout's paired-seed training behavior evidence."""

from __future__ import annotations

from pathlib import Path

import pytest
from tooling.acceptance.training_behavior import (
    ARTIFACT_PATH,
    load_training_behavior_artifact,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.local_evidence
def test_all_paired_seeds_meet_frozen_behavior_gates() -> None:
    artifact, report = load_training_behavior_artifact(
        REPO_ROOT / ARTIFACT_PATH,
        repo_root=REPO_ROOT,
    )

    assert report.ok, report.errors
    assert artifact["gate"] == {"passed": True, "errors": []}
    assert [pair["seed"] for pair in artifact["pairs"]] == [0, 1, 2, 3, 4]
    assert all(pair["gate"]["passed"] for pair in artifact["pairs"])
