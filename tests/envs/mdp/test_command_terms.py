"""Contract tests for generic pose command terms."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from unilab.envs.mdp import UniformPoseCommand, UniformPoseCommandCfg


def _env(num_envs: int = 2) -> SimpleNamespace:
    return SimpleNamespace(
        num_envs=num_envs,
        rng=np.random.default_rng(11),
    )


def test_uniform_pose_command_samples_configured_width_and_zero_bucket() -> None:
    env = _env()
    cfg = UniformPoseCommandCfg(
        resampling_time_range=(1.0, 1.0),
        ranges=((-1.0, 1.0), (2.0, 2.0)),
        zero_command_prob=1.0,
    )
    term = cfg.build(env)
    assert isinstance(term, UniformPoseCommand)

    term.reset(np.asarray([0, 1], dtype=np.int32))
    assert term.command.shape == (2, 2)
    np.testing.assert_array_equal(term.command, 0.0)


def test_uniform_pose_command_reads_curriculum_updates_without_changing_width() -> None:
    env = _env()
    cfg = UniformPoseCommandCfg(
        resampling_time_range=(1.0, 1.0),
        ranges=[[0.0, 0.0], [1.0, 1.0]],
    )
    term = cfg.build(env)
    ids = np.asarray([0, 1], dtype=np.int32)

    cfg.ranges[0] = [2.0, 2.0]
    term.reset(ids)
    np.testing.assert_array_equal(term.command, [[2.0, 1.0], [2.0, 1.0]])

    cfg.ranges.append([3.0, 3.0])
    with pytest.raises(ValueError, match="ranges width changed"):
        term.reset(ids)
