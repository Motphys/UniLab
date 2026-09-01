"""Test reward config injection system."""

from typing import Any, cast

import pytest
from hydra import compose, initialize
from omegaconf import OmegaConf


def test_reward_config_loading_g1():
    """Test G1 SAC reward config loads correctly."""
    with initialize(config_path="../../src/unilab/conf/sac", version_base="1.3"):
        cfg = compose(config_name="config", overrides=["task=g1_walk_flat/mujoco"])
        assert hasattr(cfg, "reward")
        assert cfg.reward.tracking_lin_vel.weight == 2.0
        assert cfg.reward.alive.weight == 10.0
        assert cfg.reward.feet_phase.params.swing_height == 0.09


def test_reward_config_loading_g1_motrix():
    """Test G1 Motrix reward config loads correctly."""
    with initialize(config_path="../../src/unilab/conf/sac", version_base="1.3"):
        cfg = compose(config_name="config", overrides=["task=g1_walk_flat/motrix"])
        assert hasattr(cfg, "reward")
        assert cfg.reward.tracking_lin_vel.weight == 2.2
        assert cfg.reward.alive.weight == 12.0


def test_resolve_reward_dict_reads_task_reward():
    """Task-backend configs should expose the final reward mapping directly."""
    from unilab.utils.reward import resolve_reward_dict

    with initialize(config_path="../../src/unilab/conf/ppo", version_base="1.3"):
        cfg = compose(
            config_name="config",
            overrides=["task=go2_joystick_flat/motrix"],
        )

    reward_dict = resolve_reward_dict(cfg)

    assert reward_dict["tracking_lin_vel"]["weight"] == 1.0
    assert reward_dict["tracking_ang_vel"]["weight"] == 0.2
    assert reward_dict["tracking_lin_vel"]["func"].endswith("track_lin_vel_xy_exp")


def test_reward_config_conversion():
    """Test reward config materializes into manager reward terms via registry."""
    from unilab.base import registry
    from unilab.base.config_materialization import apply_cfg_overrides
    from unilab.base.registry import ensure_registries
    from unilab.envs import ManagerBasedRlEnvCfg

    ensure_registries()

    env_cfg = registry.materialize_env_config("G1WalkFlat")
    assert isinstance(env_cfg, ManagerBasedRlEnvCfg)
    apply_cfg_overrides(
        env_cfg,
        {
            "rewards": {
                "tracking_lin_vel": {
                    "_target_": "unilab.managers.RewardTermCfg",
                    "func": "unilab.tasks.locomotion.common.sensor_reward_terms.track_lin_vel",
                    "weight": 2.0,
                    "params": {
                        "tracking_sigma": 0.25,
                        "command_name": "twist",
                        "sensor_name": "pelvis_local_linvel",
                    },
                },
                "alive": {
                    "_target_": "unilab.managers.RewardTermCfg",
                    "func": "unilab.tasks.locomotion.common.manager_terms.alive",
                    "weight": 10.0,
                },
            }
        },
    )
    assert env_cfg.rewards["tracking_lin_vel"].weight == 2.0
    assert env_cfg.rewards["tracking_lin_vel"].params["tracking_sigma"] == 0.25
    assert env_cfg.rewards["alive"].weight == 10.0
    assert OmegaConf.is_config(env_cfg.rewards["tracking_lin_vel"].func) is False
