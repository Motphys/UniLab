"""Contract tests for generic pose command terms."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from unilab.envs.mdp import UniformPoseCommand, UniformPoseCommandCfg
from unilab.managers.torch_rng import TorchManagerRng


def _env(num_envs: int = 2) -> SimpleNamespace:
    return SimpleNamespace(
        num_envs=num_envs,
        device=torch.device("cpu"),
        torch_rng=TorchManagerRng.seeded(11),
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

    term.reset(torch.tensor([0, 1], dtype=torch.int64))
    assert term.command.shape == (2, 2)
    np.testing.assert_array_equal(term.command, 0.0)


def test_uniform_pose_command_reads_curriculum_updates_without_changing_width() -> None:
    env = _env()
    cfg = UniformPoseCommandCfg(
        resampling_time_range=(1.0, 1.0),
        ranges=[[0.0, 0.0], [1.0, 1.0]],
    )
    term = cfg.build(env)
    ids = torch.tensor([0, 1], dtype=torch.int64)

    cfg.ranges[0] = [2.0, 2.0]
    term.reset(ids)
    np.testing.assert_array_equal(term.command, [[2.0, 1.0], [2.0, 1.0]])

    cfg.ranges.append([3.0, 3.0])
    with pytest.raises(ValueError, match="ranges width changed"):
        term.reset(ids)


def test_uniform_pose_command_fails_closed_without_torch_rng() -> None:
    env = _env()
    env.torch_rng = None
    term = UniformPoseCommandCfg(
        resampling_time_range=(1.0, 1.0),
        ranges=((0.0, 1.0),),
    ).build(env)

    with pytest.raises(RuntimeError, match="Manager-owned Torch generator"):
        term.reset(torch.tensor([0], dtype=torch.int64))
