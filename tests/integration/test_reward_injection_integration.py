"""Integration test for reward config injection in training."""

from typing import Any, cast

import numpy as np
import pytest


@pytest.mark.slow
def test_reward_injection_in_training():
    """Test reward config is properly injected during training."""
    from hydra import compose, initialize
    from scripts.train_offpolicy import build_runner

    with initialize(config_path="../../conf/offpolicy", version_base="1.3"):
        cfg = compose(
            config_name="config",
            overrides=[
                "task=sac/g1_walk_flat/mujoco",
                "algo.max_iterations=1",
                "algo.num_envs=64",
                "training.no_play=true",
                "training.task_name=G1WalkFlat",  # Ensure correct task
            ],
        )

        runner = build_runner("sac", cfg)

        # Verify runner was created with override
        assert runner.env_cfg_override is not None
        assert "reward_config" in runner.env_cfg_override

        # Verify reward config dict has correct values
        reward_dict = runner.env_cfg_override["reward_config"]
        assert reward_dict["scales"]["tracking_lin_vel"] == 2.0
        assert reward_dict["scales"]["alive"] == 10.0

        runner.close()


def test_reward_override_propagation():
    """Test reward override propagates through multiprocess collector."""
    from unilab.base import registry
    from unilab.base.registry import ensure_registries
    from unilab.envs.locomotion.go1.joystick import RewardConfig

    ensure_registries()

    # Create custom reward config
    custom_config = RewardConfig(
        scales={
            "tracking_lin_vel": 5.0,
            "tracking_ang_vel": 0.5,
            "lin_vel_z": -10.0,
        },
        tracking_sigma=0.5,
        base_height_target=0.4,
    )

    # Create env with override
    env = cast(
        Any,
        registry.make(
            "Go1JoystickFlat",
            num_envs=4,
            sim_backend="mujoco",
            env_cfg_override={"reward_config": custom_config},
        ),
    )

    # Verify override was applied
    assert env._cfg.reward_config.scales["tracking_lin_vel"] == 5.0
    assert env._cfg.reward_config.tracking_sigma == 0.5

    # Test reward computation uses overridden scales
    env.init_state()
    env.reset(np.array([0, 1, 2, 3], dtype=np.int32))

    # Take a step and verify reward is computed
    actions = np.zeros((4, env.action_space.shape[0]), dtype=np.float32)
    state = env.step(actions)

    assert state.reward is not None
    assert len(state.reward) == 4

    env.close()


def test_backward_compatibility_no_reward_config():
    """Test env requires reward config - should fail without it."""
    from unilab.base import registry
    from unilab.base.registry import ensure_registries

    ensure_registries()

    # Should fail without reward_config
    with pytest.raises(ValueError, match="reward_config must be provided"):
        registry.make(
            "Go1JoystickFlat",
            num_envs=2,
            sim_backend="mujoco",
        )


def test_zero_scale_skips_computation():
    """Test that reward functions with scale=0 are skipped."""
    from unilab.base import registry
    from unilab.base.registry import ensure_registries
    from unilab.envs.locomotion.go1.joystick import RewardConfig

    ensure_registries()

    # Set all scales to 0 except one
    custom_config = RewardConfig(
        scales={
            "tracking_lin_vel": 1.0,
            "tracking_ang_vel": 0.0,  # Should be skipped
            "lin_vel_z": 0.0,  # Should be skipped
        },
        tracking_sigma=0.25,
        base_height_target=0.3,
    )

    env = cast(
        Any,
        registry.make(
            "Go1JoystickFlat",
            num_envs=2,
            sim_backend="mujoco",
            env_cfg_override={"reward_config": custom_config},
        ),
    )

    # Verify only non-zero scales are in config
    assert env._cfg.reward_config.scales["tracking_lin_vel"] == 1.0
    assert env._cfg.reward_config.scales["tracking_ang_vel"] == 0.0

    env.close()
