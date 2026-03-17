"""Slow integration tests for APPORunner.

Requires MuJoCo to be installed. Run with:
    uv run pytest -m slow -v           # init + close tests
    uv run pytest -m veryslow -v       # full training iteration
"""

from __future__ import annotations

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
@pytest.mark.veryslow
@pytest.mark.parametrize("env_name", ["Go2JoystickFlatTerrain"])
def test_appo_runner_learn_two_iterations(env_name):
    """APPO learn test must use a real env — DummyFlatTest is not registered in
    the collector subprocess (mp.spawn) so registry.make() would fail there."""
    cfg = appo_config(env_name).to_dict()
    cfg["num_envs"] = 128
    cfg["steps_per_env"] = 8
    # Small network for smoke test speed
    cfg["actor"]["hidden_dims"] = [64, 64]
    cfg["critic"]["hidden_dims"] = [64, 64]
    cfg["algorithm"]["num_learning_epochs"] = 1
    cfg["algorithm"]["num_mini_batches"] = 2

    runner = APPORunner(
        env_name=env_name,
        env_cfg_overrides={},
        rl_cfg=cfg,
        num_envs=128,
        steps_per_env=8,
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
