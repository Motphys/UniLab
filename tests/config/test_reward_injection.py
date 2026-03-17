"""Test reward config injection system."""

import pytest
from hydra import compose, initialize
from omegaconf import OmegaConf


def test_reward_config_loading_g1():
    """Test G1 SAC reward config loads correctly."""
    with initialize(config_path="../../conf/offpolicy", version_base="1.3"):
        cfg = compose(config_name="config", overrides=["task=g1_sac"])
        assert hasattr(cfg, "reward")
        assert cfg.reward.scales.tracking_lin_vel == 2.0
        assert cfg.reward.scales.alive == 10.0
        assert cfg.reward.base_height_target == 0.754


def test_reward_config_loading_go1():
    """Test Go1 reward config loads correctly."""
    with initialize(config_path="../../conf/offpolicy", version_base="1.3"):
        cfg = compose(config_name="config", overrides=["task=go1_joystick"])
        assert hasattr(cfg, "reward")
        assert cfg.reward.scales.tracking_lin_vel == 1.0
        assert cfg.reward.scales.contact == 0.24


def test_reward_config_conversion():
    """Test reward config converts to dataclasses via registry."""
    from unilab.base import registry
    from unilab.utils.algo_utils import ensure_registries

    ensure_registries()

    # Test G1 SAC config - registry auto-converts dict to RewardConfigSAC
    g1_dict = {
        "scales": {"tracking_lin_vel": 2.0, "alive": 10.0},
        "tracking_sigma": 0.25,
        "base_height_target": 0.754,
        "gait_frequency": 1.5,
        "feet_phase_swing_height": 0.09,
        "feet_phase_tracking_sigma": 0.008,
        "min_base_height": 0.3,
        "max_tilt_deg": 65.0,
        "close_feet_threshold": 0.15,
        "pose_weights": [0.01] * 29,
    }
    env = registry.make(
        "G1WalkTaskMjSAC",
        num_envs=1,
        sim_backend="mujoco",
        env_cfg_override={"reward_config": g1_dict},
    )
    assert hasattr(env._cfg.reward_config, "scales")
    assert env._cfg.reward_config.scales["tracking_lin_vel"] == 2.0
    env.close()

    # Test Go1 config - registry auto-converts dict to RewardConfig
    go1_dict = {
        "scales": {"tracking_lin_vel": 1.0, "base_height": -100.0},
        "tracking_sigma": 0.25,
        "base_height_target": 0.3,
    }
    env = registry.make(
        "Go1JoystickFlatTerrain",
        num_envs=1,
        sim_backend="mujoco",
        env_cfg_override={"reward_config": go1_dict},
    )
    assert hasattr(env._cfg.reward_config, "scales")
    assert env._cfg.reward_config.scales["tracking_lin_vel"] == 1.0
    env.close()
