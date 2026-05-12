from __future__ import annotations

import pytest
from benchmark import benchmark_replay_buffer_placement as bench


def test_sac_default_case_uses_effective_symmetry_batch() -> None:
    cfg = bench._compose_offpolicy_cfg("sac", "g1_walk_flat", "mujoco")
    shape = bench.ReplayShape(obs_dim=45, action_dim=29, critic_dim=48)

    case = bench._build_case(
        cfg,
        algo="sac",
        task="g1_walk_flat",
        sim="mujoco",
        shape=shape,
        symmetry_batch_multiplier=2,
        max_capacity_rows=None,
    )

    assert case.command == "uv run train --algo sac --task g1_walk_flat --sim mujoco"
    assert case.config_capacity_rows == case.num_envs * case.replay_buffer_n
    assert case.learner_batch_size == case.configured_batch_size // 2
    assert case.sample_count == case.learner_batch_size * case.updates_per_step
    assert case.incremental_rows == case.num_envs * case.env_steps_per_sync
    assert case.replay_pipeline == "cpu_pinned_double_buffer"


def test_flashsac_default_case_uses_configured_batch() -> None:
    cfg = bench._compose_offpolicy_cfg("flashsac", "g1_walk_flat", "mujoco")
    shape = bench.ReplayShape(obs_dim=45, action_dim=29, critic_dim=48)

    case = bench._build_case(
        cfg,
        algo="flashsac",
        task="g1_walk_flat",
        sim="mujoco",
        shape=shape,
        symmetry_batch_multiplier=1,
        max_capacity_rows=1024,
    )

    assert case.command == "uv run train --algo flashsac --task g1_walk_flat --sim mujoco"
    assert case.config_capacity_rows == case.num_envs * case.replay_buffer_n
    assert case.benchmark_capacity_rows == 1024
    assert case.learner_batch_size == case.configured_batch_size
    assert case.sample_count == case.configured_batch_size * case.updates_per_step
    assert case.replay_pipeline == "gpu_cache"


def test_replay_shape_packed_width_includes_critic_fields() -> None:
    shape = bench.ReplayShape(obs_dim=45, action_dim=29, critic_dim=48)

    assert shape.packed_width == 2 * 45 + 29 + 3 + 2 * 48


def test_wbt_owner_config_is_only_included_when_present() -> None:
    assert bench._owner_config_exists("sac", "g1_sac_wbt", "mujoco")
    assert not bench._owner_config_exists("flashsac", "g1_sac_wbt", "mujoco")


def test_default_discovery_includes_existing_offpolicy_mujoco_tasks() -> None:
    targets, skipped = bench._resolve_targets(
        algos=["sac", "flashsac", "td3"],
        tasks=["auto"],
        sim="mujoco",
    )

    assert skipped == []
    assert ("sac", "g1_walk_flat") in targets
    assert ("sac", "g1_walk_rough") in targets
    assert ("sac", "g1_sac_wbt") in targets
    assert ("flashsac", "g1_walk_flat") in targets
    assert ("flashsac", "go2_joystick_flat") in targets
    assert ("td3", "g1_walk_flat") in targets
    assert ("flashsac", "g1_sac_wbt") not in targets


def test_td3_is_an_allowed_benchmark_algo() -> None:
    assert "td3" in bench._parse_algos("sac,flashsac,td3")


def test_parse_tasks_deduplicates_ordered_values() -> None:
    assert bench._parse_tasks("g1_walk_flat,g1_sac_wbt,g1_walk_flat") == [
        "g1_walk_flat",
        "g1_sac_wbt",
    ]


def test_parse_tasks_auto_must_stand_alone() -> None:
    assert bench._parse_tasks("auto") == ["auto"]
    with pytest.raises(ValueError, match="cannot be combined"):
        bench._parse_tasks("auto,g1_walk_flat")
