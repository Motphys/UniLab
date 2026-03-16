"""Tests for locomotion_params config builders."""

from __future__ import annotations

from copy import deepcopy

import pytest

from unilab.config.locomotion_params import appo_config, offpolicy_config, ppo_config

# ---------------------------------------------------------------------------
# ppo_config
# ---------------------------------------------------------------------------


def test_ppo_config_required_keys():
    cfg = ppo_config("Go1JoystickFlatTerrain")
    d = cfg.to_dict()
    for key in (
        "algo",
        "seed",
        "num_envs",
        "num_steps_per_env",
        "obs_groups",
        "policy",
        "algorithm",
    ):
        assert key in d, f"ppo_config missing key: {key}"


def test_ppo_config_to_dict_no_raise():
    cfg = ppo_config("Go2JoystickFlatTerrain")
    d = cfg.to_dict()
    assert isinstance(d, dict)


def test_ppo_config_algo_value():
    cfg = ppo_config("Go1JoystickFlatTerrain")
    assert cfg.algo == "ppo"


# ---------------------------------------------------------------------------
# appo_config
# ---------------------------------------------------------------------------


def test_appo_config_required_keys():
    cfg = appo_config("Go1JoystickFlatTerrain")
    d = cfg.to_dict()
    for key in ("algo", "actor", "critic", "algorithm"):
        assert key in d, f"appo_config missing key: {key}"


def test_appo_config_actor_has_class_name():
    cfg = appo_config("Go1JoystickFlatTerrain")
    d = cfg.to_dict()
    assert "class_name" in d["actor"], "appo_config actor must contain class_name"


def test_appo_config_to_dict_no_raise():
    cfg = appo_config("G1JoystickFlatTerrain")
    d = cfg.to_dict()
    assert isinstance(d, dict)


def test_appo_config_algo_value():
    cfg = appo_config("Go2JoystickFlatTerrain")
    assert cfg.algo == "appo"


# ---------------------------------------------------------------------------
# offpolicy_config
# ---------------------------------------------------------------------------


def test_offpolicy_sac_required_keys():
    cfg = offpolicy_config("sac", "Go2JoystickFlatTerrain")
    d = cfg.to_dict()
    for key in ("algo", "num_envs", "batch_size", "replay_buffer_n"):
        assert key in d, f"offpolicy_config(sac) missing key: {key}"
    assert d["algo"] == "sac"


def test_offpolicy_td3_required_keys():
    cfg = offpolicy_config("td3", "Go2JoystickFlatTerrain")
    d = cfg.to_dict()
    for key in ("algo", "num_envs", "batch_size", "replay_buffer_n"):
        assert key in d, f"offpolicy_config(td3) missing key: {key}"
    assert d["algo"] == "td3"


def test_offpolicy_invalid_algo_raises():
    with pytest.raises(ValueError, match="Unsupported"):
        offpolicy_config("dqn", "Go1JoystickFlatTerrain")


# ---------------------------------------------------------------------------
# ppo_config — env-specific branches
# ---------------------------------------------------------------------------


def test_ppo_config_go1_has_actor_obs_group():
    """Go1 branch sets actor obs group in addition to default."""
    cfg = ppo_config("Go1JoystickFlatTerrain")
    d = cfg.to_dict()
    obs = d["obs_groups"]
    assert "actor" in obs, "Go1 ppo_config must have actor obs group"
    assert "default" in obs


def test_ppo_config_go1_max_iterations():
    cfg = ppo_config("Go1JoystickFlatTerrain")
    assert cfg.max_iterations == 151


def test_ppo_config_g1_num_envs():
    """G1 branch sets num_envs=2048."""
    cfg = ppo_config("G1JoystickFlatTerrain")
    assert cfg.num_envs == 2048


def test_ppo_config_g1_has_actor_obs_group():
    cfg = ppo_config("G1JoystickFlatTerrain")
    d = cfg.to_dict()
    assert "actor" in d["obs_groups"]


def test_ppo_config_go2_has_actor_obs_group():
    cfg = ppo_config("Go2JoystickFlatTerrain")
    d = cfg.to_dict()
    assert "actor" in d["obs_groups"]


def test_ppo_config_unknown_env_uses_defaults():
    """Unknown env_name falls through all branches → uses default obs_groups."""
    cfg = ppo_config("SomeOtherEnv")
    d = cfg.to_dict()
    assert "default" in d["obs_groups"]
    # no actor obs group for unknown env
    assert "actor" not in d["obs_groups"]


# ---------------------------------------------------------------------------
# appo_config — env-specific branches
# ---------------------------------------------------------------------------


def test_appo_config_g1_max_iterations():
    """G1JoystickFlatTerrain sets max_iterations=500."""
    cfg = appo_config("G1JoystickFlatTerrain")
    assert cfg.max_iterations == 500


def test_appo_config_g1_walk_max_iterations():
    cfg = appo_config("G1WalkTaskMjSAC")
    assert cfg.max_iterations == 500


def test_appo_config_go1_default_max_iterations():
    cfg = appo_config("Go1JoystickFlatTerrain")
    # Go1 uses the default (150)
    assert cfg.max_iterations == 150


# ---------------------------------------------------------------------------
# offpolicy_config — SAC env-specific branches
# ---------------------------------------------------------------------------


def test_offpolicy_sac_go1_num_envs():
    """Go1 SAC branch sets num_envs=2048 and longer max_iterations."""
    cfg = offpolicy_config("sac", "Go1JoystickFlatTerrain")
    assert cfg.num_envs == 2048
    assert cfg.max_iterations == 2000


def test_offpolicy_sac_go2_num_envs():
    """Go2 SAC branch reduces num_envs to 1024."""
    cfg = offpolicy_config("sac", "Go2JoystickFlatTerrain")
    assert cfg.num_envs == 1024


def test_offpolicy_sac_g1_raises():
    """G1JoystickFlatTerrain raises NotImplementedError for SAC."""
    with pytest.raises(NotImplementedError):
        offpolicy_config("sac", "G1JoystickFlatTerrain")


def test_offpolicy_sac_g1_walk_symmetry():
    """G1WalkTaskMjSAC enables symmetry augmentation."""
    cfg = offpolicy_config("sac", "G1WalkTaskMjSAC")
    d = cfg.to_dict()
    assert d.get("use_symmetry") is True


def test_offpolicy_sac_g1_walk_num_envs():
    cfg = offpolicy_config("sac", "G1WalkTaskMjSAC")
    assert cfg.num_envs == 2048


def test_offpolicy_sac_g1_walk_max_iterations():
    cfg = offpolicy_config("sac", "G1WalkTaskMjSAC")
    assert cfg.max_iterations == 5000


# ---------------------------------------------------------------------------
# offpolicy_config — TD3 env-specific branches
# ---------------------------------------------------------------------------


def test_offpolicy_td3_go2_max_iterations():
    """Go2 TD3 branch sets max_iterations=2000."""
    cfg = offpolicy_config("td3", "Go2JoystickFlatTerrain")
    assert cfg.max_iterations == 2000


def test_offpolicy_td3_g1_num_envs():
    """G1JoystickFlatTerrain TD3 branch sets num_envs=2048."""
    cfg = offpolicy_config("td3", "G1JoystickFlatTerrain")
    assert cfg.num_envs == 2048


def test_offpolicy_td3_default_env():
    """Unknown env falls through all branches → default values."""
    cfg = offpolicy_config("td3", "SomeOtherEnv")
    d = cfg.to_dict()
    assert d["algo"] == "td3"


def test_deepcopy_pop_does_not_mutate_original():
    """deepcopy + pop('class_name') must not affect the original config."""
    cfg = appo_config("Go1JoystickFlatTerrain")
    d = cfg.to_dict()

    actor_copy = deepcopy(d["actor"])
    actor_copy.pop("class_name", None)

    # Original config must still have class_name
    d2 = cfg.to_dict()
    assert "class_name" in d2["actor"], "deepcopy mutated original appo_config actor"
