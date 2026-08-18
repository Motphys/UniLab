"""Integration test for reward config injection in training."""


def test_reward_injection_in_offpolicy_env_override():
    """Test reward config is injected without requiring accelerator hardware."""
    from hydra import compose, initialize
    from scripts.train_offpolicy import build_offpolicy_env_cfg_override

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

        env_cfg_override = build_offpolicy_env_cfg_override("sac", cfg)

        assert env_cfg_override is not None
        assert "reward_config" in env_cfg_override

        # Verify reward config dict has correct values
        reward_dict = env_cfg_override["reward_config"]
        assert reward_dict["scales"]["tracking_lin_vel"] == 2.0
        assert reward_dict["scales"]["alive"] == 10.0
