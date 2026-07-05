from __future__ import annotations

import pytest
from benchmark import benchmark_reset_done_attribution as bench


def test_stats_reports_distribution() -> None:
    stats = bench._stats([3.0, 1.0, 2.0])

    assert stats.mean_ms == pytest.approx(2.0)
    assert stats.median_ms == pytest.approx(2.0)
    assert stats.min_ms == pytest.approx(1.0)
    assert stats.max_ms == pytest.approx(3.0)
    assert stats.samples_ms == [3.0, 1.0, 2.0]


def test_sample_env_ids_is_sorted_unique_and_seeded() -> None:
    env_ids = bench._sample_env_ids(num_envs=16, reset_count=5, seed=673)

    assert env_ids.tolist() == sorted(env_ids.tolist())
    assert len(set(env_ids.tolist())) == 5
    assert env_ids.tolist() == bench._sample_env_ids(16, 5, 673).tolist()


def test_sample_env_ids_rejects_invalid_count() -> None:
    with pytest.raises(ValueError, match="reset-count must be > 0"):
        bench._sample_env_ids(num_envs=16, reset_count=0, seed=673)

    with pytest.raises(ValueError, match="reset-count must be <= num-envs"):
        bench._sample_env_ids(num_envs=16, reset_count=17, seed=673)


def test_parse_args_defaults_to_issue_scale() -> None:
    args = bench.parse_args([])

    assert args.num_envs == 8192
    assert args.reset_count == 256
    assert args.measure_repeats == 30
    assert args.body_pose_backends == "motrix,mujoco"
    assert args.set_state_profile_backends == "motrix,mujoco"
    assert not args.skip_set_state
