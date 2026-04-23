"""Tests for structured configs and Hydra YAML loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest

CONF_DIR = Path(__file__).parent.parent.parent / "conf"


# ---------------------------------------------------------------------------
# structured_configs dataclass defaults
# ---------------------------------------------------------------------------


def test_sac_config_defaults():
    from unilab.config.structured_configs import SACAlgoParams, SACConfig

    cfg = SACConfig()
    assert cfg.algo == "sac"
    assert cfg.num_envs == 4096
    assert cfg.batch_size == 8192
    assert cfg.use_symmetry is False
    assert isinstance(cfg.algo_params, SACAlgoParams)
    assert cfg.algo_params.alpha_init == 0.01


def test_td3_config_defaults():
    from unilab.config.structured_configs import TD3Config

    cfg = TD3Config()
    assert cfg.algo == "td3"
    assert cfg.num_envs == 4096
    assert cfg.use_layer_norm is False
    assert cfg.algo_params.weight_decay == 0.1


def test_flashsac_config_defaults():
    from unilab.config.structured_configs import FlashSACAlgoParams, FlashSACConfig

    cfg = FlashSACConfig()
    assert cfg.algo == "flashsac"
    assert cfg.num_envs == 1024
    assert cfg.batch_size == 2048
    assert cfg.learning_starts == 98
    assert cfg.gamma == pytest.approx(0.97)
    assert cfg.obs_normalization is False
    assert isinstance(cfg.algo_params, FlashSACAlgoParams)
    assert cfg.algo_params.normalize_reward is True
    assert cfg.algo_params.use_compile is False


def test_ppo_config_defaults():
    from unilab.config.structured_configs import PPOConfig

    cfg = PPOConfig()
    assert cfg.algo == "ppo"
    assert cfg.max_iterations == 101
    assert cfg.algorithm.clip_param == 0.2
    assert cfg.algorithm.class_name == "unilab.algos.torch.rsl_rl_ppo:FinalObservationAwarePPO"
    assert cfg.policy.class_name == "ActorCritic"


def test_appo_config_defaults():
    from unilab.config.structured_configs import APPOConfig

    cfg = APPOConfig()
    assert cfg.algo == "appo"
    assert cfg.num_envs == 2048
    assert cfg.actor.class_name == "rsl_rl.models.MLPModel"


def test_base_config_to_dict():
    from unilab.config.structured_configs import SACConfig

    cfg = SACConfig()
    d = cfg.to_dict()
    assert isinstance(d, dict)
    assert d["algo"] == "sac"
    assert "algo_params" in d
    assert isinstance(d["algo_params"], dict)


# ---------------------------------------------------------------------------
# Hydra YAML loading — offpolicy
# ---------------------------------------------------------------------------


def test_offpolicy_sac_defaults():
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_DIR / "offpolicy"), version_base="1.3"):
        cfg = compose("config")
    assert cfg.algo.algo == "sac"
    assert cfg.algo.num_envs == 2048


def test_offpolicy_sac_g1_task_overrides():
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_DIR / "offpolicy"), version_base="1.3"):
        cfg = compose("config", overrides=["algo=sac", "task=sac/g1_walk_flat/mujoco"])
    assert cfg.algo.num_envs == 2048
    assert cfg.algo.max_iterations == 5000
    assert cfg.algo.use_symmetry is True
    assert cfg.algo.algo_params.target_entropy_ratio == pytest.approx(0.0)
    assert cfg.training.task_name == "G1WalkFlat"

    assert cfg.env.control_config.action_scale == pytest.approx(1.0)
    assert cfg.env.gait_phase_init_mode == "offset_phase"
    assert cfg.env.reset_base_qvel_limit == pytest.approx(0.5)


def test_offpolicy_td3_defaults():
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_DIR / "offpolicy"), version_base="1.3"):
        cfg = compose("config", overrides=["algo=td3"])
    assert cfg.algo.algo == "td3"
    assert cfg.algo.use_layer_norm is False
    assert cfg.algo.algo_params.weight_decay == pytest.approx(0.001)
    assert cfg.algo.tau == pytest.approx(0.01)
    assert cfg.algo.algo_params.policy_noise == pytest.approx(0.1)
    assert cfg.algo.algo_params.noise_clip == pytest.approx(0.2)
    assert cfg.algo.algo_params.log_std_min == pytest.approx(-5.0)


def test_offpolicy_td3_g1_task_overrides():
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_DIR / "offpolicy"), version_base="1.3"):
        cfg = compose("config", overrides=["algo=td3", "task=td3/g1_walk_flat/mujoco"])
    assert cfg.training.task_name == "G1WalkFlat"
    assert cfg.algo.max_iterations == 100000
    assert cfg.env.control_config.action_scale == pytest.approx(1.0)


def test_offpolicy_flashsac_g1_task_overrides():
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_DIR / "offpolicy"), version_base="1.3"):
        cfg = compose(
            "config",
            overrides=["algo=flashsac", "task=flashsac/g1_walk_flat/mujoco"],
        )
    assert cfg.algo.algo == "flashsac"
    assert cfg.training.task_name == "G1WalkFlat"
    assert cfg.training.sim_backend == "mujoco"
    assert cfg.algo.algo_params.actor_num_blocks == 2
    assert cfg.algo.algo_params.normalize_reward is True


def test_offpolicy_g1_rough_terrain_task_overrides():
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    from unilab.envs.locomotion.g1.joystick import G1WalkRoughCfg

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_DIR / "offpolicy"), version_base="1.3"):
        cfg = compose(
            "config",
            overrides=["algo=sac", "task=sac/g1_walk_rough/mujoco"],
        )
    assert cfg.algo.algo == "sac"
    assert cfg.training.task_name == "G1WalkRough"
    assert cfg.training.sim_backend == "mujoco"
    assert G1WalkRoughCfg().model_file.endswith("scene_rough.xml")


def test_offpolicy_flashsac_g1_walk_flat_amp_task_overrides():
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_DIR / "offpolicy"), version_base="1.3"):
        cfg = compose(
            "config",
            overrides=["algo=flashsac", "task=flashsac/g1_walk_flat_amp/mujoco"],
        )
    assert cfg.algo.algo == "flashsac"
    assert cfg.training.task_name == "G1WalkFlat"
    assert cfg.training.sim_backend == "mujoco"
    assert cfg.training.use_amp is True
    assert cfg.algo.num_envs == 1024
    assert cfg.algo.updates_per_step == 2
    assert cfg.algo.num_envs == 1024
    assert cfg.algo.replay_buffer_n == 9766
    assert cfg.algo.learning_starts == 98
    assert cfg.algo.updates_per_step == 2
    assert cfg.algo.gamma == pytest.approx(0.97)
    assert cfg.env.control_config.action_scale == pytest.approx(0.5)
    assert cfg.env.commands.vel_limit[0] == [-1.0, -0.5, -1.0]
    assert "obs_profile" not in cfg.env
    assert cfg.env.curriculum.enabled is False
    assert cfg.reward.scales.tracking_ang_vel == pytest.approx(0.75)
    assert cfg.reward.scales.base_height == pytest.approx(0.0)


def test_g1_task_owner_yamls_preserve_legacy_and_walk_observation_profiles():
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    from unilab.envs.locomotion.g1.joystick import G1WalkEnv

    def uses_walk_profile(config_group: str, overrides: list[str]) -> bool:
        GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=str(CONF_DIR / config_group), version_base="1.3"):
            cfg = compose("config", overrides=overrides)
        env = cast(Any, object.__new__(G1WalkEnv))
        env._cfg = cfg.env
        env._reward_cfg = cfg.reward
        return bool(env._uses_walk_observation_profile())

    assert uses_walk_profile("ppo", ["task=g1_walk_flat/mujoco"]) is False
    assert uses_walk_profile("appo", ["task=g1_walk_flat/mujoco"]) is False
    assert (
        uses_walk_profile(
            "offpolicy",
            ["algo=flashsac", "task=flashsac/g1_walk_flat_amp/mujoco"],
        )
        is False
    )

    assert uses_walk_profile("offpolicy", ["algo=sac", "task=sac/g1_walk_flat/mujoco"]) is True
    assert uses_walk_profile("offpolicy", ["algo=sac", "task=sac/g1_walk_flat/motrix"]) is True
    assert uses_walk_profile("offpolicy", ["algo=sac", "task=sac/g1_walk_rough/mujoco"]) is True
    assert uses_walk_profile("offpolicy", ["algo=td3", "task=td3/g1_walk_flat/mujoco"]) is True
    assert (
        uses_walk_profile("offpolicy", ["algo=flashsac", "task=flashsac/g1_walk_flat/mujoco"])
        is True
    )


# ---------------------------------------------------------------------------
# Hydra YAML loading — appo
# ---------------------------------------------------------------------------


def test_appo_defaults():
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_DIR / "appo"), version_base="1.3"):
        cfg = compose("config")
    assert cfg.algo.algo == "appo"
    assert cfg.algo.max_iterations == 150


def test_appo_g1_task_overrides():
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_DIR / "appo"), version_base="1.3"):
        cfg = compose("config", overrides=["task=g1_walk_flat/mujoco"])
    assert cfg.algo.max_iterations == 500
    assert cfg.algo.save_interval == 100
    assert cfg.training.task_name == "G1WalkFlat"
    assert "obs_profile" not in cfg.env
    assert cfg.env.curriculum.enabled is False


# ---------------------------------------------------------------------------
# Hydra YAML loading — ppo
# ---------------------------------------------------------------------------


def test_ppo_go1_max_iterations():
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_DIR / "ppo"), version_base="1.3"):
        cfg = compose("config", overrides=["task=go1_joystick_flat/mujoco"])
    assert cfg.algo.max_iterations == 151
    assert "actor" in cfg.algo.obs_groups


def test_ppo_g1_num_envs():
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_DIR / "ppo"), version_base="1.3"):
        cfg = compose("config", overrides=["task=g1_walk_flat/mujoco"])
    assert cfg.algo.num_envs == 2048
    assert cfg.algo.max_iterations == 220
    assert cfg.training.task_name == "G1WalkFlat"
    assert "obs_profile" not in cfg.env
    assert cfg.env.curriculum.enabled is False


def test_ppo_go2_num_envs():
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_DIR / "ppo"), version_base="1.3"):
        cfg = compose("config", overrides=["task=go2_joystick_flat/mujoco"])
    assert cfg.algo.num_envs == 1024
    assert cfg.algo.max_iterations == 151


def test_ppo_g1_motion_tracking():
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_DIR / "ppo"), version_base="1.3"):
        cfg = compose("config", overrides=["task=g1_motion_tracking/mujoco"])
    assert cfg.training.task_name == "G1MotionTracking"
    assert cfg.algo.max_iterations == 15000
    assert cfg.algo.algorithm.entropy_coef == pytest.approx(0.005)


def test_ppo_g1_flip_tracking():
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_DIR / "ppo"), version_base="1.3"):
        cfg = compose("config", overrides=["task=g1_flip_tracking/mujoco"])
    assert cfg.training.task_name == "G1FlipTracking"
    assert cfg.algo.max_iterations == 30000
