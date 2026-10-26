from __future__ import annotations

from collections import defaultdict
from types import SimpleNamespace

import pytest

from unilab.rl.logging.metric_schema import (
    METRIC_FAMILIES,
    METRIC_SPECS,
    normalize_metric_map,
    reward_term_key,
    validate_metric_tags,
)
from unilab.rl.logging.offpolicy import (
    _COLLECTOR_TIMING_SPECS,
    _LEARNER_DETAIL_TIMING_PROFILES,
    _LEARNER_TIMING_PROFILES,
)
from unilab.rl.runtime.offpolicy.double_buffer_runner import DoubleBufferOffPolicyRunner


def test_rsl_rl_tags_are_aligned_verbatim() -> None:
    rsl_rl_tags = {
        "Train/mean_reward",
        "Train/mean_episode_length",
        "Loss/surrogate",
        "Loss/value",
        "Loss/entropy",
        "Loss/learning_rate",
        "Policy/mean_std",
        "Perf/total_fps",
        "Perf/collection_time",
        "Perf/learning_time",
    }
    assert rsl_rl_tags <= set(METRIC_SPECS)

    mean_reward = METRIC_SPECS["Train/mean_reward"]
    assert mean_reward.owner == "collector"
    assert "raw environment returns" in mean_reward.aggregation
    assert "timeout-bootstrap corrections" in mean_reward.description
    assert mean_reward.distributed_aggregation == "cross-rank mean of per-rank 100-episode means"

    total_fps = METRIC_SPECS["Perf/total_fps"]
    assert total_fps.unit == "env steps/s"
    assert total_fps.distributed_aggregation == "cross-rank sum"


def test_all_specs_declare_complete_semantics() -> None:
    for spec in (*METRIC_SPECS.values(), *METRIC_FAMILIES.values()):
        assert spec.owner
        assert spec.unit
        assert spec.aggregation
        assert spec.step_axis
        assert spec.description
        assert spec.backends == ("tensorboard", "wandb")


def test_source_metrics_must_already_be_canonical() -> None:
    metrics = {
        "Loss/surrogate": 1.0,
        "Policy/temperature": 2.0,
        "Train/target_q_max": 3.0,
    }
    assert normalize_metric_map(metrics) == {key: float(value) for key, value in metrics.items()}

    for retired in ("actor_loss", "train/actor_loss", "losses/policy_loss", "Loss/policy_loss"):
        with pytest.raises(ValueError, match="unregistered canonical training metric"):
            normalize_metric_map({retired: 1.0})


def test_replay_ingress_diagnostics_are_registered_as_cumulative_pipeline_metrics() -> None:
    tags = {
        "Train/replay_ingress_depth",
        "Train/replay_ingress_occupancy",
        "Train/replay_ingress_high_water",
        "Train/replay_ingress_backpressure_wait_ms",
        "Train/replay_ingress_dropped_batches",
    }

    assert tags <= set(METRIC_SPECS)
    for tag in tags:
        spec = METRIC_SPECS[tag]
        assert spec.owner == "ipc/replay"
        assert spec.distributed_aggregation == "cross-rank sum"


def test_schema_is_closed_except_reward_terms() -> None:
    validate_metric_tags(
        [
            "Loss/surrogate",
            "reward/term",
            "Curriculum/voc/stage",
            "Train/rollouts_read",
            "Perf/learner_inference_ms",
        ]
    )

    assert set(METRIC_FAMILIES) == {"reward/", "Curriculum/", "Loss/aux_"}
    assert "Perf/learner_other_ms" not in METRIC_SPECS
    assert "Train/staging_pool_capacity" not in METRIC_SPECS
    assert "Train/updates" not in METRIC_SPECS
    top_level_groups = {tag.split("/", 1)[0] for tag in METRIC_SPECS}
    assert top_level_groups <= {"Train", "Loss", "Policy", "Episode", "Perf"}
    with pytest.raises(ValueError, match="unregistered backend metric tags: timing/retired_ms"):
        validate_metric_tags(["timing/retired_ms"])


def test_timing_specs_are_exactly_the_logger_timing_profiles() -> None:
    learner_tags = {
        key for profile in _LEARNER_TIMING_PROFILES.values() for key, _, _ in profile
    } | {key for profile in _LEARNER_DETAIL_TIMING_PROFILES.values() for key, _, _ in profile}
    collector_tags = {
        f"Perf/collector_{key}" for key in _COLLECTOR_TIMING_SPECS if key != "rollout_ms"
    } | {"Perf/collection_time"}
    dp_tags = {"Perf/dp_gradient_sync_ms_per_rank"}
    timing_tags = {
        tag
        for tag, spec in METRIC_SPECS.items()
        if tag.startswith("Perf/")
        and (spec.unit == "ms" or tag in {"Perf/collection_time", "Perf/learning_time"})
    }

    assert (learner_tags | collector_tags | dp_tags) == timing_tags


def test_reward_terms_are_namespaced_and_reserved_names_fail_closed() -> None:
    assert reward_term_key("alive") == "reward/alive"

    for term in ("reward/alive", "", "mean", "mean_ep100"):
        with pytest.raises(ValueError):
            reward_term_key(term)

    reward_family = METRIC_FAMILIES["reward/"]
    assert reward_family.unit == "weighted reward units/s"
    assert "cross-rank" not in reward_family.aggregation
    assert reward_family.distributed_aggregation == "cross-rank mean of per-rank report means"
    for invalid_tag in ("reward/", "reward/mean"):
        with pytest.raises(ValueError, match=invalid_tag):
            validate_metric_tags([invalid_tag])


@pytest.mark.parametrize("graph_mode", [False, True])
@pytest.mark.parametrize("timing_enabled", [False, True])
def test_dp_metrics_use_explicit_per_rank_units(graph_mode, timing_enabled) -> None:
    runner = object.__new__(DoubleBufferOffPolicyRunner)
    runner.dp_sync = SimpleNamespace(take_gradient_sync_metrics=lambda: (0.25, 3))
    runner.learner = SimpleNamespace(use_update_cycle=graph_mode)
    runner.trace_enabled = timing_enabled
    runner.trace_cuda_events = True
    iter_metrics: defaultdict[str, list[float]] = defaultdict(list)

    runner._collect_dp_sync_metrics(iter_metrics)

    if not graph_mode or timing_enabled:
        assert iter_metrics["Perf/dp_gradient_sync_ms_per_rank"] == [250.0]
    else:
        assert "Perf/dp_gradient_sync_ms_per_rank" not in iter_metrics
    assert iter_metrics["Perf/dp_gradient_sync_calls_per_rank"] == [3.0]
