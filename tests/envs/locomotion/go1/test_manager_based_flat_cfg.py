"""Hydra-owned production contract for the Go1 flat Manager-Based task."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

from unilab.base import registry
from unilab.base.config_adapter import BackendAdapter
from unilab.base.config_materialization import apply_cfg_overrides
from unilab.envs import (
    ManagerBasedRlEnv,
    ManagerBasedRlEnvCfg,
    make_manager_based_rl_env,
    mdp,
)
from unilab.tasks.locomotion.common import manager_terms

ROOT_DIR = Path(__file__).parents[4]
CONF_DIR = ROOT_DIR / "conf"

_JOINT_NAMES = (
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
)
_ACTUATOR_NAMES = tuple(name.removesuffix("_joint") for name in _JOINT_NAMES)
_HOME_JOINT_POS = np.array(
    [0.0, 0.9, -1.8, 0.0, 0.9, -1.8, 0.0, 1.0, -1.8, 0.0, 1.0, -1.8],
    dtype=np.float32,
)
_RESET_EVENTS = ("reset_scene_to_default", "reset_root_state_uniform")
_DR_EVENTS = ("base_mass", "base_com", "pd_gains")
_BASE_REWARDS = (
    "tracking_lin_vel",
    "tracking_ang_vel",
    "lin_vel_z",
    "ang_vel_xy",
    "base_height",
    "action_rate",
    "similar_to_default",
)

_OWNER_CASES = (
    pytest.param(
        "ppo",
        ("task=go1_joystick_flat/mujoco",),
        "mujoco",
        (*_RESET_EVENTS, *_DR_EVENTS, "push_robot"),
        (*_BASE_REWARDS, "contact", "swing_feet_z"),
        False,
        id="ppo-mujoco",
    ),
    pytest.param(
        "ppo",
        ("task=go1_joystick_flat/motrix",),
        "motrix",
        (*_RESET_EVENTS, *_DR_EVENTS),
        (*_BASE_REWARDS, "swing_feet_z"),
        True,
        id="ppo-motrix",
    ),
    pytest.param(
        "ppo",
        ("task=go1_joystick_flat/drake",),
        "drake",
        _RESET_EVENTS,
        (*_BASE_REWARDS, "swing_feet_z"),
        False,
        id="ppo-drake",
    ),
    pytest.param(
        "appo",
        ("task=go1_joystick_flat/mujoco",),
        "mujoco",
        (*_RESET_EVENTS, *_DR_EVENTS, "push_robot"),
        (*_BASE_REWARDS, "contact"),
        False,
        id="appo-mujoco",
    ),
    pytest.param(
        "appo",
        ("task=go1_joystick_flat/motrix",),
        "motrix",
        (*_RESET_EVENTS, *_DR_EVENTS),
        (*_BASE_REWARDS, "swing_feet_z", "action_smooth"),
        True,
        id="appo-motrix",
    ),
    pytest.param(
        "td3",
        ("task=go1_joystick_flat/motrix",),
        "motrix",
        (*_RESET_EVENTS, *_DR_EVENTS),
        (*_BASE_REWARDS, "swing_feet_z"),
        True,
        id="td3-motrix",
    ),
)


def _compose(config_group: str, overrides: Sequence[str]) -> DictConfig:
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_DIR / config_group), version_base="1.3"):
        return compose("config", overrides=list(overrides))


def _materialize(
    config_group: str, overrides: Sequence[str]
) -> tuple[DictConfig, ManagerBasedRlEnvCfg, dict[str, Any]]:
    hydra_cfg = _compose(config_group, overrides)
    env_override = BackendAdapter(hydra_cfg, root_dir=ROOT_DIR).build_task_env_cfg_override()
    env_cfg = registry.materialize_env_config("Go1JoystickFlat")
    assert isinstance(env_cfg, ManagerBasedRlEnvCfg)
    apply_cfg_overrides(env_cfg, env_override)
    env_cfg.validate()
    return hydra_cfg, env_cfg, env_override


def _assert_no_omegaconf(value: Any) -> None:
    assert not OmegaConf.is_config(value)
    if is_dataclass(value) and not isinstance(value, type):
        for item in fields(value):
            _assert_no_omegaconf(getattr(value, item.name))
    elif isinstance(value, Mapping):
        for key, item in value.items():
            _assert_no_omegaconf(key)
            _assert_no_omegaconf(item)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for item in value:
            _assert_no_omegaconf(item)


@pytest.mark.parametrize(
    "config_group,overrides,backend,expected_events,expected_rewards,fixed_command",
    _OWNER_CASES,
)
def test_go1_flat_owner_materializes_complete_plain_manager_cfg(
    config_group: str,
    overrides: tuple[str, ...],
    backend: str,
    expected_events: tuple[str, ...],
    expected_rewards: tuple[str, ...],
    fixed_command: bool,
) -> None:
    registry.ensure_registries()
    hydra_cfg, env_cfg, _ = _materialize(config_group, overrides)

    assert hydra_cfg.training.task_name == "Go1JoystickFlat"
    assert hydra_cfg.training.sim_backend == backend
    assert env_cfg.sim_dt == pytest.approx(0.01)
    assert env_cfg.ctrl_dt == pytest.approx(0.02)
    assert env_cfg.max_episode_seconds == pytest.approx(20.0)
    assert env_cfg.policy_observation_group == "policy"
    assert env_cfg.critic_observation_group == "critic"
    assert env_cfg.scale_rewards_by_dt is True

    assert env_cfg.scene is not None
    assert env_cfg.scene.model_file.endswith("robots/go1/scene_flat.xml")
    assert env_cfg.scene.default_keyframe_name == "home"
    robot = env_cfg.scene.entities["robot"]
    assert robot.root_body_name == "trunk"
    assert tuple(robot.joint_names or ()) == _JOINT_NAMES
    assert tuple(robot.actuator_names or ()) == _ACTUATOR_NAMES
    assert robot.body_names == ["trunk"]

    observation_terms = [
        "base_ang_vel",
        "projected_gravity",
        "joint_pos",
        "joint_vel",
        "actions",
        "command",
        "gait_phase",
    ]
    assert list(env_cfg.observations) == ["policy", "critic"]
    assert list(env_cfg.observations["policy"].terms) == observation_terms
    assert list(env_cfg.observations["critic"].terms) == [*observation_terms, "base_lin_vel"]
    assert list(env_cfg.actions) == ["joint_pos"]
    assert env_cfg.actions["joint_pos"].scale == pytest.approx(0.25)
    assert list(env_cfg.terminations) == ["time_out", "bad_orientation"]
    assert (
        tuple(name for name, term in env_cfg.events.items() if term is not None) == expected_events
    )
    assert tuple(name for name, term in env_cfg.rewards.items() if term is not None) == (
        expected_rewards
    )

    command = env_cfg.commands["twist"]
    assert command.resampling_time_range == [20.0, 20.0]
    ranges = command.ranges
    if fixed_command:
        assert tuple(ranges.lin_vel_x) == (0.5, 0.5)
        assert tuple(ranges.lin_vel_y) == (0.0, 0.0)
        assert tuple(ranges.ang_vel_z) == (0.0, 0.0)
    else:
        assert tuple(ranges.lin_vel_x) == (-0.6, 1.0)

    if env_cfg.events["base_mass"] is not None:
        mass = env_cfg.events["base_mass"]
        assert mass.func is mdp.randomize_rigid_body_mass
        assert mass.params["mass_distribution_params"] == [-1.5, 1.5]
        assert mass.params["recompute_inertia"] is False
        com = env_cfg.events["base_com"]
        assert com.func is mdp.randomize_rigid_body_com
        assert com.params["com_range"] == {
            "x": [-0.05, 0.05],
            "y": [0.0, 0.0],
            "z": [0.0, 0.0],
        }
        gains = env_cfg.events["pd_gains"]
        assert gains.func is mdp.pd_gains
        assert gains.params["kp_range"] == [35.0, 35.0]
        assert gains.params["kd_range"] == [0.5, 0.5]
        assert gains.params["operation"] == "abs"

    push = env_cfg.events["push_robot"]
    if push is not None:
        assert push.func is mdp.push_by_setting_velocity
        assert push.interval_range_s == [15.0, 15.0]
        assert push.is_global_time is True

    contact = env_cfg.rewards["contact"]
    if contact is not None:
        assert contact.func is manager_terms.feet_phase_contact
        # Legacy Go1 returned a four-foot sum; the community term returns a mean.
        assert contact.weight == pytest.approx(4.0 * 0.24)
    action_smooth = env_cfg.rewards.get("action_smooth")
    if action_smooth is not None:
        assert action_smooth.func is mdp.action_acc_l2
        assert action_smooth.weight == pytest.approx(-0.01)

    for manager_name in ("observations", "events", "rewards", "terminations"):
        for term in getattr(env_cfg, manager_name).values():
            if term is None:
                continue
            nested_terms = term.terms.values() if manager_name == "observations" else (term,)
            for nested in nested_terms:
                if nested is None:
                    continue
                module = nested.func.__module__
                assert ".backend." not in module
                assert not any(name in module for name in (".mujoco", ".motrix", ".drake"))

    _assert_no_omegaconf(env_cfg)


def test_go1_flat_and_rough_registries_are_manager_only() -> None:
    registry.ensure_registries()

    assert registry.list_registered_envs()["Go1JoystickFlat"] == {
        "config_factory": "ManagerBasedRlEnvCfg",
        "available_backends": ["mujoco", "motrix", "drake"],
    }
    assert registry.list_registered_envs()["Go1JoystickRough"] == {
        "config_factory": "ManagerBasedRlEnvCfg",
        "available_backends": ["mujoco", "motrix"],
    }

    for legacy_override in (
        {"reward_config": {}},
        {"domain_rand": {"randomize_base_mass": True}},
        {"control_config": {"Kp": 35.0}},
    ):
        with pytest.raises(ValueError, match="has no attribute"):
            apply_cfg_overrides(ManagerBasedRlEnvCfg(), legacy_override)


@pytest.mark.parametrize(
    ("backend", "owner", "expected_events"),
    (
        (
            "mujoco",
            "task=go1_joystick_flat/mujoco",
            {"reset": [*_RESET_EVENTS, *_DR_EVENTS], "interval": ["push_robot"]},
        ),
        (
            "motrix",
            "task=go1_joystick_flat/motrix",
            {"reset": [*_RESET_EVENTS, *_DR_EVENTS]},
        ),
    ),
)
def test_go1_flat_registry_executes_real_manager_runtime(
    backend: str,
    owner: str,
    expected_events: dict[str, list[str]],
) -> None:
    registry.ensure_registries()
    hydra_cfg, _, env_override = _materialize("ppo", (owner,))
    try:
        env = registry.make(
            str(hydra_cfg.training.task_name),
            sim_backend=backend,
            env_cfg_override=env_override,
            num_envs=2,
        )
    except ImportError as exc:
        pytest.skip(f"{backend} runtime unavailable: {exc}")

    try:
        assert isinstance(env, ManagerBasedRlEnv)
        assert env.obs_groups_spec == {"obs": 49, "critic": 52}
        assert env.action_space.shape == (12,)
        action = env.action_manager.get_term("joint_pos")
        assert action.target_names == list(_JOINT_NAMES)
        np.testing.assert_allclose(action.offset, np.broadcast_to(_HOME_JOINT_POS, (2, 12)))
        assert env.event_manager.active_terms == expected_events

        obs, info = env.reset(seed=7)
        assert {name: value.shape for name, value in obs.items()} == {
            "obs": (2, 49),
            "critic": (2, 52),
        }
        assert isinstance(info, dict)
        np.testing.assert_allclose(
            env.scene["robot"].data.default_joint_pos,
            np.broadcast_to(_HOME_JOINT_POS, (2, 12)),
        )
        for _ in range(5):
            state = env.step(np.zeros((2, 12), dtype=np.float32))
        for value in (*state.obs.values(), state.reward):
            assert isinstance(value, np.ndarray)
            assert np.isfinite(value).all()
    finally:
        env.close()


def test_go1_motrix_velocity_push_request_fails_closed() -> None:
    _, motrix_cfg, _ = _materialize("ppo", ("task=go1_joystick_flat/motrix",))
    _, mujoco_cfg, _ = _materialize("ppo", ("task=go1_joystick_flat/mujoco",))
    motrix_cfg.events["push_robot"] = deepcopy(mujoco_cfg.events["push_robot"])

    try:
        with pytest.raises(NotImplementedError, match="interval root velocity delta.*motrix"):
            make_manager_based_rl_env(motrix_cfg, num_envs=1, backend_type="motrix")
    except ImportError as exc:
        pytest.skip(f"motrix runtime unavailable: {exc}")
