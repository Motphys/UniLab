"""Tests for structured configs and Hydra YAML loading."""

from __future__ import annotations

from pathlib import Path

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


def test_ppo_config_defaults():
    from unilab.config.structured_configs import PPOConfig

    cfg = PPOConfig()
    assert cfg.algo == "ppo"
    assert cfg.max_iterations == 101
    assert cfg.algorithm.clip_param == 0.2
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
    # go1_joystick task sets num_envs=4096 (motrix-aligned hyperparams folded into base)
    assert cfg.algo.num_envs == 4096


def test_offpolicy_sac_g1_task_overrides():
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_DIR / "offpolicy"), version_base="1.3"):
        cfg = compose("config", overrides=["algo=sac", "task=g1_sac"])
    assert cfg.algo.num_envs == 2048
    assert cfg.algo.max_iterations == 5000
    assert cfg.algo.use_symmetry is True
    assert cfg.algo.algo_params.target_entropy_ratio == pytest.approx(0.0)
    assert cfg.training.task_name == "G1WalkTaskMjSAC"


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


def test_offpolicy_go2_task_overrides():
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_DIR / "offpolicy"), version_base="1.3"):
        cfg = compose("config", overrides=["algo=sac", "task=go2_joystick"])
    assert cfg.algo.num_envs == 1024
    assert cfg.training.task_name == "Go2JoystickFlatTerrain"


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
        cfg = compose("config", overrides=["task=g1_joystick"])
    assert cfg.algo.max_iterations == 500
    assert cfg.algo.save_interval == 100
    assert cfg.training.task_name == "G1JoystickFlatTerrain"


# ---------------------------------------------------------------------------
# Hydra YAML loading — ppo
# ---------------------------------------------------------------------------


def test_ppo_go1_max_iterations():
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_DIR / "ppo"), version_base="1.3"):
        cfg = compose("config", overrides=["task=go1_joystick"])
    assert cfg.algo.max_iterations == 151
    assert "actor" in cfg.algo.obs_groups


def test_ppo_g1_num_envs():
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_DIR / "ppo"), version_base="1.3"):
        cfg = compose("config", overrides=["task=g1_joystick"])
    assert cfg.algo.num_envs == 2048
    # max_iterations=151 (motrix-aligned hyperparams folded into base)
    assert cfg.algo.max_iterations == 151


def test_ppo_go2_num_envs():
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_DIR / "ppo"), version_base="1.3"):
        cfg = compose("config", overrides=["task=go2_joystick"])
    assert cfg.algo.num_envs == 1024
    assert cfg.algo.max_iterations == 151


def test_ppo_g1_motion_tracking():
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_DIR / "ppo"), version_base="1.3"):
        cfg = compose("config", overrides=["task=g1_motion_tracking"])
    assert cfg.training.task_name == "G1MotionTracking"
    assert cfg.algo.max_iterations == 15000
    assert cfg.algo.algorithm.entropy_coef == pytest.approx(0.005)


def test_ppo_g1_flip_tracking():
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_DIR / "ppo"), version_base="1.3"):
        cfg = compose("config", overrides=["task=g1_flip_tracking"])
    assert cfg.training.task_name == "G1FlipTracking"
    assert cfg.algo.max_iterations == 30000
