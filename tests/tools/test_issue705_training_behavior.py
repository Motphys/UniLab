"""Contract tests for the frozen Issue #837 paired-seed behavior plan."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Mapping, cast

import pytest
import yaml

from unilab.tools.issue705_training_behavior import (
    PLAN_PATH,
    TrainingBehaviorContractError,
    evaluate_training_behavior_cases,
    load_frozen_training_inputs,
    load_training_behavior_plan,
    summarize_training_behavior_raw,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
PHASE5_ARTIFACT = REPO_ROOT / "tests/acceptance/issue_705/artifacts/phase_5_mjwarp_ppo.json"


def _phase5_candidate_cases() -> list[dict[str, Any]]:
    plan = load_training_behavior_plan(REPO_ROOT / PLAN_PATH, repo_root=REPO_ROOT)
    raw = json.loads(PHASE5_ARTIFACT.read_text(encoding="utf-8"))
    selected = [case for case in raw["cases"] if case.get("lane") == "behavior"]
    result: list[dict[str, Any]] = []
    for index, case in enumerate(selected):
        candidate = {
            "case_id": case["case_id"],
            "seed": case["seed"],
            "sequence_index": index,
            "process_retries": 0,
            "batch_size": case["batch_size"],
            "num_steps_per_env": plan.measurement["num_steps_per_env"],
            "iterations": case["iterations"],
            "mode": case["mode"],
            "raw": copy.deepcopy(case["raw"]),
        }
        candidate["summary"] = summarize_training_behavior_raw(
            cast(Mapping[str, Any], candidate["raw"]),
            plan,
            label=f"phase5/seed={case['seed']}",
        )
        result.append(candidate)
    return result


def test_frozen_plan_matches_phase0_phase7_and_support_signature() -> None:
    plan = load_training_behavior_plan(REPO_ROOT / PLAN_PATH, repo_root=REPO_ROOT)

    assert plan.seeds == (0, 1, 2, 3, 4)
    assert plan.measurement["num_envs"] == 1024
    assert plan.measurement["num_steps_per_env"] == 24
    assert plan.measurement["max_iterations"] == 100
    assert plan.measurement["final_window_iterations"] == 10
    assert plan.measurement["success_metric"]["disposition"] == "not_applicable"


def test_existing_phase5_curves_satisfy_stricter_per_seed_oracle() -> None:
    plan = load_training_behavior_plan(REPO_ROOT / PLAN_PATH, repo_root=REPO_ROOT)
    threshold, baseline = load_frozen_training_inputs(plan, repo_root=REPO_ROOT)
    pairs, aggregates, errors = evaluate_training_behavior_cases(
        plan=plan,
        threshold=threshold,
        baseline=baseline,
        candidate_cases=_phase5_candidate_cases(),
    )

    assert errors == []
    assert [pair["seed"] for pair in pairs] == [0, 1, 2, 3, 4]
    assert all(pair["gate"]["passed"] for pair in pairs)
    assert aggregates["gate"]["passed"] is True


def test_pair_gate_rejects_one_degraded_seed_without_median_masking() -> None:
    plan = load_training_behavior_plan(REPO_ROOT / PLAN_PATH, repo_root=REPO_ROOT)
    threshold, baseline = load_frozen_training_inputs(plan, repo_root=REPO_ROOT)
    cases = _phase5_candidate_cases()
    degraded = cases[3]
    reward = degraded["raw"]["scalars"]["Train/mean_reward"]
    for point in reward:
        point["value"] -= 1.0
    degraded["summary"] = summarize_training_behavior_raw(
        degraded["raw"], plan, label="degraded/seed=3"
    )

    pairs, _, errors = evaluate_training_behavior_cases(
        plan=plan,
        threshold=threshold,
        baseline=baseline,
        candidate_cases=cases,
    )

    seed_three = next(pair for pair in pairs if pair["seed"] == 3)
    assert seed_three["gate"]["passed"] is False
    assert any("seed 3: reward_auc_drop" in error for error in errors)
    assert any("seed 3: final_window_reward_drop" in error for error in errors)


def test_pair_gate_rejects_omitted_reordered_and_duplicate_seed() -> None:
    plan = load_training_behavior_plan(REPO_ROOT / PLAN_PATH, repo_root=REPO_ROOT)
    threshold, baseline = load_frozen_training_inputs(plan, repo_root=REPO_ROOT)
    cases = _phase5_candidate_cases()

    _, _, omitted = evaluate_training_behavior_cases(
        plan=plan,
        threshold=threshold,
        baseline=baseline,
        candidate_cases=cases[:-1],
    )
    assert "candidate omitted or added a training run" in omitted

    reordered_cases = [cases[1], cases[0], *cases[2:]]
    _, _, reordered = evaluate_training_behavior_cases(
        plan=plan,
        threshold=threshold,
        baseline=baseline,
        candidate_cases=reordered_cases,
    )
    assert "candidate case order must exactly match frozen seeds" in reordered

    duplicate_cases = [*cases[:-1], copy.deepcopy(cases[-2])]
    _, _, duplicate = evaluate_training_behavior_cases(
        plan=plan,
        threshold=threshold,
        baseline=baseline,
        candidate_cases=duplicate_cases,
    )
    assert "candidate contains duplicate seeds" in duplicate


def test_raw_curve_rejects_missing_iteration_nan_and_fabricated_success() -> None:
    plan = load_training_behavior_plan(REPO_ROOT / PLAN_PATH, repo_root=REPO_ROOT)
    source = _phase5_candidate_cases()[0]["raw"]

    missing = copy.deepcopy(source)
    missing["scalars"]["Train/mean_reward"].pop()
    with pytest.raises(TrainingBehaviorContractError, match="every iteration"):
        summarize_training_behavior_raw(missing, plan, label="missing")

    non_finite = copy.deepcopy(source)
    non_finite["scalars"]["Train/mean_reward"][50]["value"] = float("nan")
    with pytest.raises(TrainingBehaviorContractError, match="finite value"):
        summarize_training_behavior_raw(non_finite, plan, label="non-finite")

    fabricated = copy.deepcopy(source)
    fabricated["scalars"]["Train/success_rate"] = copy.deepcopy(
        fabricated["scalars"]["Train/mean_reward"]
    )
    with pytest.raises(TrainingBehaviorContractError, match="success metric"):
        summarize_training_behavior_raw(fabricated, plan, label="fabricated")


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda raw: raw.update({"extra": True}), "unknown key `extra`"),
        (
            lambda raw: raw["measurement"].update({"num_envs": 4096}),
            "measurement.num_envs",
        ),
        (
            lambda raw: raw["measurement"]["seeds"].pop(),
            "measurement.seeds",
        ),
        (
            lambda raw: raw["measurement"]["success_metric"].update(
                {"disposition": "available", "scalar_tag": "Train/success_rate"}
            ),
            "success_metric.disposition",
        ),
        (
            lambda raw: raw["gates"].update({"maximum_candidate_fps_population_cv": 0.2}),
            "maximum_candidate_fps_population_cv",
        ),
    ],
)
def test_plan_rejects_schema_budget_success_and_threshold_tamper(
    tmp_path: Path,
    mutation: Any,
    message: str,
) -> None:
    raw = yaml.safe_load((REPO_ROOT / PLAN_PATH).read_text(encoding="utf-8"))
    mutation(raw)
    path = tmp_path / "plan.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(TrainingBehaviorContractError, match=message):
        load_training_behavior_plan(path)
