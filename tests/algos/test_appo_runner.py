"""Slow integration tests for APPORunner.

Requires MuJoCo to be installed. Run with:
    uv run pytest -m slow -v           # init + close + full training iteration
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

pytest.importorskip("mujoco")

from uni_rl.appo.runner import APPORunner

from unilab.base.config_adapter import BackendAdapter
from unilab.base.env_factory import registry_env_factory
from unilab.structured_configs import APPOConfig


@pytest.mark.slow
def test_appo_runner_init_no_crash(mock_env_name):
    cfg = APPOConfig().to_dict()
    cfg["num_envs"] = 4
    cfg["steps_per_env"] = 4

    runner = APPORunner(
        env_name=mock_env_name,
        env_factory=registry_env_factory(mock_env_name, "mujoco"),
        env_cfg_overrides={},
        rl_cfg=cfg,
        num_envs=4,
        steps_per_env=4,
    )
    runner.close()


@pytest.mark.slow
@pytest.mark.parametrize("env_name", ["Go2JoystickFlat"])
def test_appo_runner_learn_two_iterations(env_name):
    """APPO learn test must use a real env — DummyFlatTest is not registered in
    the collector subprocess (mp.spawn) so registry.make() would fail there."""
    cfg = APPOConfig().to_dict()
    cfg["num_envs"] = 128
    cfg["steps_per_env"] = 8
    # Small network for smoke test speed
    cfg["actor"]["hidden_dims"] = [64, 64]
    cfg["critic"]["hidden_dims"] = [64, 64]
    cfg["algorithm"]["num_learning_epochs"] = 1
    cfg["algorithm"]["num_mini_batches"] = 2

    root_dir = Path(__file__).parents[2]
    GlobalHydra.instance().clear()
    with initialize_config_dir(
        config_dir=str(root_dir / "src" / "unilab" / "conf" / "appo"), version_base="1.3"
    ):
        hydra_cfg = compose("config", overrides=["task=go2_joystick_flat/mujoco"])
    env_cfg_overrides = BackendAdapter(hydra_cfg, root_dir=root_dir).build_task_env_cfg_override()

    runner = APPORunner(
        env_name=env_name,
        env_factory=registry_env_factory(env_name, "mujoco"),
        env_cfg_overrides=env_cfg_overrides,
        rl_cfg=cfg,
        num_envs=128,
        steps_per_env=8,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        runner.learn(max_iterations=2, save_interval=0, log_dir=tmpdir)

    runner.close()


@pytest.mark.slow
def test_appo_runner_close_is_idempotent(mock_env_name):
    cfg = APPOConfig().to_dict()
    cfg["num_envs"] = 4
    cfg["steps_per_env"] = 4

    runner = APPORunner(
        env_name=mock_env_name,
        env_factory=registry_env_factory(mock_env_name, "mujoco"),
        env_cfg_overrides={},
        rl_cfg=cfg,
        num_envs=4,
        steps_per_env=4,
    )
    runner.close()
    runner.close()  # must not raise
