"""Slow integration tests for APPORunner.

Requires MuJoCo to be installed. Run with:
    uv run pytest -m slow -v
"""

from __future__ import annotations

import os
import tempfile

import pytest

pytest.importorskip("mujoco")

from unilab.algos.torch.appo.runner import APPORunner
from unilab.config.locomotion_params import appo_config


@pytest.mark.slow
def test_appo_runner_init_no_crash(mock_env_name):
    cfg = appo_config(mock_env_name).to_dict()
    cfg["num_envs"] = 4
    cfg["steps_per_env"] = 4

    runner = APPORunner(
        env_name=mock_env_name,
        env_cfg_overrides={},
        rl_cfg=cfg,
        num_envs=4,
        steps_per_env=4,
    )
    runner.close()


@pytest.mark.slow
def test_appo_runner_learn_two_iterations(mock_env_name):
    cfg = appo_config(mock_env_name).to_dict()
    cfg["num_envs"] = 4
    cfg["steps_per_env"] = 4

    runner = APPORunner(
        env_name=mock_env_name,
        env_cfg_overrides={},
        rl_cfg=cfg,
        num_envs=4,
        steps_per_env=4,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        runner.learn(max_iterations=2, save_interval=0, log_dir=tmpdir)

    runner.close()


@pytest.mark.slow
def test_appo_runner_close_is_idempotent(mock_env_name):
    cfg = appo_config(mock_env_name).to_dict()
    cfg["num_envs"] = 4
    cfg["steps_per_env"] = 4

    runner = APPORunner(
        env_name=mock_env_name,
        env_cfg_overrides={},
        rl_cfg=cfg,
        num_envs=4,
        steps_per_env=4,
    )
    runner.close()
    runner.close()  # must not raise
