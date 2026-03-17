"""Test APPO reward injection."""

import pytest


def test_appo_reward_override():
    """Test APPO with reward override."""
    from unilab.base import registry
    from unilab.utils.algo_utils import ensure_registries

    ensure_registries()

    reward_dict = {
        "scales": {"tracking_lin_vel": 888.0},
        "tracking_sigma": 0.3,
        "base_height_target": 0.35,
    }

    env = registry.make(
        "Go1JoystickFlatTerrain",
        num_envs=2,
        sim_backend="mujoco",
        env_cfg_override={"reward_config": reward_dict}
    )

    assert env._cfg.reward_config.scales["tracking_lin_vel"] == 888.0
    env.close()


def test_rsl_rl_reward_override():
    """Test RSL-RL with reward override."""
    from unilab.base import registry
    from unilab.utils.algo_utils import ensure_registries

    ensure_registries()

    reward_dict = {
        "scales": {"tracking_lin_vel": 777.0},
        "tracking_sigma": 0.2,
        "base_height_target": 0.32,
    }

    env = registry.make(
        "Go1JoystickFlatTerrain",
        num_envs=2,
        sim_backend="mujoco",
        env_cfg_override={"reward_config": reward_dict}
    )

    assert env._cfg.reward_config.scales["tracking_lin_vel"] == 777.0
    env.close()
