"""Tests for manipulation_params config builders."""

from __future__ import annotations

from copy import deepcopy

import pytest

from unilab.config.manipulation_params import appo_config, offpolicy_config, ppo_config

# ---------------------------------------------------------------------------
# ppo_config
# ---------------------------------------------------------------------------


def test_ppo_config_required_keys():
    cfg = ppo_config("AllegroInhandRotation")
    d = cfg.to_dict()
    for key in ("seed", "num_envs", "obs_groups", "policy", "algorithm"):
        assert key in d, f"ppo_config missing key: {key}"


def test_ppo_config_obs_groups_has_actor():
    cfg = ppo_config("AllegroInhandRotation")
    d = cfg.to_dict()
    assert "actor" in d["obs_groups"], "obs_groups must contain 'actor'"
    assert "critic" in d["obs_groups"], "obs_groups must contain 'critic'"


def test_ppo_config_actor_obs_group_contains_actor():
    """actor obs group lists 'actor' key."""
    cfg = ppo_config("AllegroInhandRotation")
    d = cfg.to_dict()
    assert "actor" in d["obs_groups"]["actor"]


def test_ppo_config_allegro_num_envs():
    cfg = ppo_config("AllegroInhandRotation")
    assert cfg.num_envs == 16384


def test_ppo_config_allegro_max_iterations():
    cfg = ppo_config("AllegroInhandRotation")
    assert cfg.max_iterations == 501


def test_ppo_config_allegro_sac_raises():
    with pytest.raises(NotImplementedError):
        ppo_config("AllegroInhandRotationSac")


def test_ppo_config_to_dict_no_raise():
    cfg = ppo_config("AllegroInhandRotation")
    d = cfg.to_dict()
    assert isinstance(d, dict)


# ---------------------------------------------------------------------------
# appo_config
# ---------------------------------------------------------------------------


def test_appo_config_required_keys():
    cfg = appo_config("AllegroInhandRotation")
    d = cfg.to_dict()
    for key in ("algo", "actor", "critic", "algorithm"):
        assert key in d, f"appo_config missing key: {key}"


def test_appo_config_algo_value():
    cfg = appo_config("AllegroInhandRotation")
    assert cfg.algo == "appo"


def test_appo_config_allegro_num_envs():
    cfg = appo_config("AllegroInhandRotation")
    assert cfg.num_envs == 16384


def test_appo_config_allegro_max_iterations():
    cfg = appo_config("AllegroInhandRotation")
    assert cfg.max_iterations == 501


def test_appo_config_allegro_empirical_normalization():
    cfg = appo_config("AllegroInhandRotation")
    assert cfg.empirical_normalization is True


def test_appo_config_sac_raises():
    with pytest.raises(NotImplementedError):
        appo_config("AllegroInhandRotationSac")


def test_appo_config_to_dict_no_raise():
    cfg = appo_config("AllegroInhandRotation")
    d = cfg.to_dict()
    assert isinstance(d, dict)


def test_appo_config_deepcopy_safety():
    """deepcopy + pop('class_name') must not affect the original config."""
    cfg = appo_config("AllegroInhandRotation")
    d = cfg.to_dict()
    actor_copy = deepcopy(d["actor"])
    actor_copy.pop("class_name", None)
    d2 = cfg.to_dict()
    assert "class_name" in d2["actor"]


# ---------------------------------------------------------------------------
# offpolicy_config — SAC
# ---------------------------------------------------------------------------


def test_offpolicy_sac_required_keys():
    cfg = offpolicy_config("sac", "AllegroInhandRotationSac")
    d = cfg.to_dict()
    for key in ("algo", "num_envs", "batch_size", "replay_buffer_n"):
        assert key in d, f"offpolicy_config(sac) missing key: {key}"
    assert d["algo"] == "sac"


def test_offpolicy_sac_allegro_rotation_raises():
    with pytest.raises(NotImplementedError):
        offpolicy_config("sac", "AllegroInhandRotation")


def test_offpolicy_sac_allegro_rotation_sac_num_envs():
    cfg = offpolicy_config("sac", "AllegroInhandRotationSac")
    assert cfg.num_envs == 2048


def test_offpolicy_sac_allegro_rotation_sac_max_iterations():
    cfg = offpolicy_config("sac", "AllegroInhandRotationSac")
    assert cfg.max_iterations == 25000


# ---------------------------------------------------------------------------
# offpolicy_config — invalid algo
# ---------------------------------------------------------------------------


def test_offpolicy_invalid_algo_raises():
    with pytest.raises(ValueError, match="Unsupported"):
        offpolicy_config("dqn", "AllegroInhandRotation")
