"""Test reward config override through registry."""

import pytest

from unilab.base import registry
from unilab.utils.algo_utils import ensure_registries


def _require_mujoco() -> None:
    pytest.importorskip("mujoco", exc_type=ImportError)


def test_reward_override_go1():
    """Test Go1 reward config override."""
    _require_mujoco()
    ensure_registries()

    from unilab.envs.locomotion.go1.joystick import RewardConfig

    override_config = RewardConfig(
        scales={"tracking_lin_vel": 999.0},
        tracking_sigma=0.5,
        base_height_target=0.5,
    )

    env = registry.make(
        "Go1JoystickFlatTerrain",
        num_envs=1,
        sim_backend="mujoco",
        env_cfg_override={"reward_config": override_config},
    )

    assert env._cfg.reward_config.scales["tracking_lin_vel"] == 999.0
    env.close()


def test_reward_override_g1():
    """Test G1 reward config override."""
    _require_mujoco()
    ensure_registries()

    from unilab.envs.locomotion.g1.joystick_sac import RewardConfigSAC

    override_config = RewardConfigSAC(
        scales={"tracking_lin_vel": 888.0, "alive": 20.0},
        tracking_sigma=0.3,
        base_height_target=0.8,
        min_base_height=0.3,
        max_tilt_deg=65.0,
        gait_frequency=1.5,
        feet_phase_swing_height=0.09,
        feet_phase_tracking_sigma=0.008,
        close_feet_threshold=0.15,
        pose_weights=[0.01] * 29,
    )

    env = registry.make(
        "G1WalkTaskMjSAC",
        num_envs=1,
        sim_backend="mujoco",
        env_cfg_override={"reward_config": override_config},
    )

    assert env._cfg.reward_config.scales["tracking_lin_vel"] == 888.0
    assert env._cfg.reward_config.scales["alive"] == 20.0
    env.close()
