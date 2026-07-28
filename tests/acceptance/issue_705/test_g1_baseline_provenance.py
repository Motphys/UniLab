from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from unilab.tools.g1_baseline_provenance import (
    expected_case_ids,
    load_g1_baseline_artifact,
    load_g1_baseline_plan,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
PLAN_PATH = Path("tests/acceptance/issue_705/g1_mujoco_baseline_plan.yaml")
ARTIFACT_PATH = Path("tests/acceptance/issue_705/artifacts/g1_mujoco_phase0_baseline.json")


def test_baseline_artifacts_are_reproducible() -> None:
    absolute_plan = REPO_ROOT / PLAN_PATH
    plan = replace(
        load_g1_baseline_plan(absolute_plan),
        source_path=PLAN_PATH,
    )
    artifact = load_g1_baseline_artifact(
        REPO_ROOT / ARTIFACT_PATH,
        plan,
        repo_root=REPO_ROOT,
    )

    assert {case["case_id"] for case in artifact["cases"]} == expected_case_ids(plan)
    assert artifact["source"]["dirty"] is False
    assert artifact["aggregates"]["env"]["case_count"] == 15
    assert artifact["aggregates"]["dr"]["case_count"] == 30
    assert artifact["aggregates"]["ppo"]["case_count"] == 5
    assert {case["seed"] for case in artifact["cases"] if case["lane"] == "ppo"} == set(
        plan.ppo_lane.seeds
    )
