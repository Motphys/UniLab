"""Schema tests for the managed MuJoCo/MJWarp rollout capability-derived task rollout plan."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
from omegaconf import OmegaConf
from tooling.acceptance.task_rollout import (
    CLAIM_ID,
    PLAN_FINGERPRINT,
    ROLLOUT_PLAN_PATH,
    TaskRolloutPlanError,
    load_task_rollout_plan,
    parse_task_rollout_plan,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _raw_plan() -> dict[str, object]:
    raw = OmegaConf.to_container(OmegaConf.load(REPO_ROOT / ROLLOUT_PLAN_PATH), resolve=False)
    assert isinstance(raw, dict)
    return raw


def test_task_rollout_parser_loads_frozen_g1_matrix() -> None:
    plan = load_task_rollout_plan(REPO_ROOT / ROLLOUT_PLAN_PATH)

    assert plan.issue == 705
    assert plan.claim_id == CLAIM_ID
    assert plan.plan_fingerprint == PLAN_FINGERPRINT
    assert len(plan.entries) == 1
    entry = plan.entries[0]
    assert entry.key == ("ppo_torch", "g1_walk_flat", "mjwarp")
    assert entry.seeds == (0, 1)
    assert (entry.num_envs, entry.num_steps_per_env, entry.max_iterations) == (128, 2, 1)
    assert entry.expected_model_targets == ()
    assert len(entry.prerequisites) == 27
    assert entry.support_compiled_signature.task_key == "g1_walk_flat.managed_device"
    assert entry.rollout_compiled_signature.task_plan_fingerprint == (
        "manager-task-contract-v1:8172ed8bb3a3ac6ea1e1d011770b7df8d306e604d81333669baccb838ec89a57"
    )


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda raw: raw.update({"unexpected": True}), "unknown key"),
        (
            lambda raw: raw["entries"].append(deepcopy(raw["entries"][0])),
            "duplicate keys",
        ),
        (
            lambda raw: raw["entries"][0].update({"owner_yaml": "../mjwarp.yaml"}),
            "stay within",
        ),
        (
            lambda raw: raw["entries"][0].update({"owner_yaml_sha256": "sha256:bad"}),
            "expected sha256",
        ),
        (
            lambda raw: raw["entries"][0].update({"seeds": [1, 0]}),
            "must be sorted",
        ),
        (
            lambda raw: raw["entries"][0].update({"seeds": [0, 0]}),
            "duplicate values",
        ),
        (
            lambda raw: raw["entries"][0].update(
                {"required_capabilities": ["typed_state_reset", "flat_scene"]}
            ),
            "must be sorted",
        ),
        (
            lambda raw: raw["entries"][0]["prerequisites"].append(
                deepcopy(raw["entries"][0]["prerequisites"][0])
            ),
            "duplicate claim IDs",
        ),
    ],
)
def test_task_rollout_parser_rejects_schema_faults(mutation, match: str) -> None:
    raw = _raw_plan()
    mutation(raw)

    with pytest.raises(TaskRolloutPlanError, match=match):
        parse_task_rollout_plan(raw)


def test_task_rollout_loader_normalizes_malformed_yaml(tmp_path: Path) -> None:
    path = tmp_path / "rollout.yaml"
    path.write_text("entries: [", encoding="utf-8")

    with pytest.raises(TaskRolloutPlanError, match="cannot load YAML"):
        load_task_rollout_plan(path)
