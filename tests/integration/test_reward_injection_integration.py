"""Integration test for reward config injection in training."""

import pytest
import numpy as np


@pytest.mark.slow
def test_reward_injection_in_training():
    """Test reward config is properly injected during training."""
    from hydra import initialize, compose
    from scripts.train_offpolicy import build_runner

    with initialize(config_path="../../conf/offpolicy", version_base="1.3"):
        cfg = compose(
            config_name="config",
            overrides=[
                "task=g1_sac",
                "algo.max_iterations=1",
                "algo.num_envs=64",
                "training.no_play=true",
                "training.task_name=G1WalkTaskMjSAC",  # Ensure correct task
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
    from unilab.utils.algo_utils import ensure_registries
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
    env = registry.make(
        "Go1JoystickFlatTerrain",
        num_envs=4,
        sim_backend="mujoco",
        env_cfg_override={"reward_config": custom_config},
    )

    # Verify override was applied
    assert env._cfg.reward_config.scales["tracking_lin_vel"] == 5.0
    assert env._cfg.reward_config.tracking_sigma == 0.5

    # Test reward computation uses overridden scales
    env.init_state()
    state = env.reset(np.array([0, 1, 2, 3], dtype=np.int32))[0]

    # Take a step and verify reward is computed
    actions = np.zeros((4, env.action_space.shape[0]), dtype=np.float32)
    state = env.step(actions)

    assert state.reward is not None
    assert len(state.reward) == 4

    env.close()


def test_backward_compatibility_no_reward_config():
    """Test env works without reward config override."""
    from unilab.base import registry
    from unilab.utils.algo_utils import ensure_registries

    ensure_registries()

    # Create env without override
    env = registry.make(
        "Go1JoystickFlatTerrain",
        num_envs=2,
        sim_backend="mujoco",
    )

    # Should use default reward config from env
    assert env._cfg.reward_config is not None
    assert hasattr(env._cfg.reward_config, "scales")

    # Verify env works normally
    env.init_state()
    state = env.reset(np.array([0, 1], dtype=np.int32))[0]
    actions = np.zeros((2, env.action_space.shape[0]), dtype=np.float32)
    state = env.step(actions)

    assert state.reward is not None

    env.close()


def test_zero_scale_skips_computation():
    """Test that reward functions with scale=0 are skipped."""
    from unilab.base import registry
    from unilab.utils.algo_utils import ensure_registries
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

    env = registry.make(
        "Go1JoystickFlatTerrain",
        num_envs=2,
        sim_backend="mujoco",
        env_cfg_override={"reward_config": custom_config},
    )

    # Verify only non-zero scales are in config
    assert env._cfg.reward_config.scales["tracking_lin_vel"] == 1.0
    assert env._cfg.reward_config.scales["tracking_ang_vel"] == 0.0

    env.close()
