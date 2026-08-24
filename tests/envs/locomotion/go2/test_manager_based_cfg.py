"""Hydra-owned production contract for the Go2 flat Manager-Based task."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
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
from unilab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg, mdp

ROOT_DIR = Path(__file__).parents[4]
CONF_DIR = ROOT_DIR / "conf"

_JOINT_NAMES = (
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
)
_ACTUATOR_NAMES = (
    "FR_hip",
    "FR_thigh",
    "FR_calf",
    "FL_hip",
    "FL_thigh",
    "FL_calf",
    "RR_hip",
    "RR_thigh",
    "RR_calf",
    "RL_hip",
    "RL_thigh",
    "RL_calf",
)
_HOME_JOINT_POS = np.array(
    [0.0, 0.8, -1.5, 0.0, 0.8, -1.5, 0.0, 1.0, -1.5, 0.0, 1.0, -1.5],
    dtype=np.float32,
)

_OWNER_CASES = (
    pytest.param(
        "ppo",
        ("task=go2_joystick_flat/mujoco",),
        "mujoco",
        0.25,
        True,
        False,
        False,
        id="ppo-mujoco",
    ),
    pytest.param(
        "ppo",
        ("task=go2_joystick_flat/motrix",),
        "motrix",
        0.25,
        False,
        False,
        True,
        id="ppo-motrix",
    ),
    pytest.param(
        "ppo",
        ("task=go2_joystick_flat/drake",),
        "drake",
        0.25,
        False,
        False,
        False,
        id="ppo-drake",
    ),
    pytest.param(
        "appo",
        ("task=go2_joystick_flat/mujoco",),
        "mujoco",
        0.25,
        True,
        True,
        False,
        id="appo-mujoco",
    ),
    pytest.param(
        "appo",
        ("task=go2_joystick_flat/motrix",),
        "motrix",
        0.25,
        False,
        True,
        False,
        id="appo-motrix",
    ),
    pytest.param(
        "flashsac",
        ("task=go2_joystick_flat/mujoco",),
        "mujoco",
        0.4,
        True,
        False,
        False,
        id="flashsac-mujoco",
    ),
    pytest.param(
        "td3",
        ("task=go2_joystick_flat/motrix",),
        "motrix",
        0.25,
        False,
        False,
        True,
        id="td3-motrix",
    ),
    pytest.param(
        "sac",
        ("task=go2_joystick_flat/drake",),
        "drake",
        0.25,
        False,
        False,
        False,
        id="sac-drake",
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
    env_cfg = registry.materialize_env_config("Go2JoystickFlat")
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
    "config_group,overrides,backend,action_scale,pd_enabled,alive_declared,fixed_command",
    _OWNER_CASES,
)
def test_go2_flat_owner_materializes_complete_plain_manager_cfg(
    config_group: str,
    overrides: tuple[str, ...],
    backend: str,
    action_scale: float,
    pd_enabled: bool,
    alive_declared: bool,
    fixed_command: bool,
) -> None:
    registry.ensure_registries()
    hydra_cfg, env_cfg, _ = _materialize(config_group, overrides)

    assert hydra_cfg.training.task_name == "Go2JoystickFlat"
    assert hydra_cfg.training.sim_backend == backend
    assert env_cfg.sim_dt == pytest.approx(0.01)
    assert env_cfg.ctrl_dt == pytest.approx(0.02)
    assert env_cfg.max_episode_seconds == pytest.approx(20.0)
    assert env_cfg.policy_observation_group == "policy"
    assert env_cfg.critic_observation_group == "critic"
    assert env_cfg.scene is not None
    assert env_cfg.scene.default_keyframe_name == "home"
    robot_cfg = env_cfg.scene.entities["robot"]
    assert robot_cfg.root_body_name == "base"
    assert tuple(robot_cfg.joint_names) == _JOINT_NAMES
    assert tuple(robot_cfg.actuator_names) == _ACTUATOR_NAMES

    expected_policy_terms = [
        "base_ang_vel",
        "projected_gravity",
        "joint_pos",
        "joint_vel",
        "actions",
        "command",
        "gait_phase",
    ]
    assert list(env_cfg.observations) == ["policy", "critic"]
    assert list(env_cfg.observations["policy"].terms) == expected_policy_terms
    assert list(env_cfg.observations["critic"].terms) == [
        *expected_policy_terms,
        "base_lin_vel",
    ]
    assert list(env_cfg.actions) == ["joint_pos"]
    assert env_cfg.actions["joint_pos"].scale == pytest.approx(action_scale)
    assert list(env_cfg.commands) == ["twist"]
    assert list(env_cfg.terminations) == ["time_out", "bad_orientation"]
    assert (env_cfg.events["pd_gains"] is not None) is pd_enabled
    assert ("alive" in env_cfg.rewards) is alive_declared

    if action_scale == 0.25:
        expected_weights = {
            "tracking_lin_vel": 1.0,
            "tracking_ang_vel": 0.2,
            "lin_vel_z": -5.0,
            "ang_vel_xy": -0.1,
            "base_height": -100.0,
            "action_rate": -0.005,
            "similar_to_default": -0.1,
            "contact": 0.24,
            "swing_feet_z": 4.0,
        }
        if alive_declared:
            expected_weights["alive"] = 0.0
        assert {name: term.weight for name, term in env_cfg.rewards.items()} == expected_weights

    if fixed_command:
        ranges = env_cfg.commands["twist"].ranges
        assert tuple(ranges.lin_vel_x) == (0.5, 0.5)
        assert tuple(ranges.lin_vel_y) == (0.0, 0.0)
        assert tuple(ranges.ang_vel_z) == (0.0, 0.0)

    for manager_name in ("observations", "events", "rewards", "terminations"):
        for term in getattr(env_cfg, manager_name).values():
            if term is None:
                continue
            terms = term.terms.values() if manager_name == "observations" else (term,)
            for nested_term in terms:
                if nested_term is None:
                    continue
                module = nested_term.func.__module__
                assert ".backend." not in module
                assert not any(name in module for name in (".mujoco", ".motrix", ".drake"))

    _assert_no_omegaconf(env_cfg)


def test_go2_flat_registry_has_no_legacy_config_fallback() -> None:
    registry.ensure_registries()
    bare_cfg = registry.materialize_env_config("Go2JoystickFlat")

    assert isinstance(bare_cfg, ManagerBasedRlEnvCfg)
    assert bare_cfg.observations == {}
    assert bare_cfg.actions == {}
    assert bare_cfg.rewards == {}
    assert registry.list_registered_envs()["Go2JoystickFlat"] == {
        "config_factory": "ManagerBasedRlEnvCfg",
        "available_backends": ["mujoco", "motrix", "drake"],
    }
    for legacy_override in (
        {"reward_config": {}},
        {"domain_rand": {"randomize_kp": True}},
        {"control_config": {"action_scale": 0.4}},
    ):
        with pytest.raises(ValueError, match="has no attribute"):
            apply_cfg_overrides(ManagerBasedRlEnvCfg(), legacy_override)


@pytest.mark.parametrize(
    ("backend", "owner"),
    (("mujoco", "task=go2_joystick_flat/mujoco"), ("motrix", "task=go2_joystick_flat/motrix")),
)
def test_go2_flat_registry_executes_real_manager_runtime(backend: str, owner: str) -> None:
    if backend == "motrix":
        pytest.importorskip("motrixsim", reason="motrixsim not installed")
    registry.ensure_registries()
    hydra_cfg, _, env_override = _materialize("ppo", (owner,))
    env = registry.make(
        str(hydra_cfg.training.task_name),
        sim_backend=backend,
        env_cfg_override=env_override,
        num_envs=2,
    )
    try:
        assert isinstance(env, ManagerBasedRlEnv)
        assert env.obs_groups_spec == {"obs": 49, "critic": 52}
        assert env.action_space.shape == (12,)
        action = env.action_manager.get_term("joint_pos")
        assert action.target_names == list(_JOINT_NAMES)
        np.testing.assert_allclose(action.offset, np.broadcast_to(_HOME_JOINT_POS, (2, 12)))

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
        np.testing.assert_allclose(
            env.scene["robot"].data.joint_pos,
            np.broadcast_to(_HOME_JOINT_POS, (2, 12)),
        )

        state = env.step(np.zeros((2, 12), dtype=np.float32))
        assert {name: value.shape for name, value in state.obs.items()} == {
            "obs": (2, 49),
            "critic": (2, 52),
        }
        for value in (*state.obs.values(), state.reward):
            assert isinstance(value, np.ndarray)
            assert np.isfinite(value).all()
    finally:
        env.close()


def test_go2_flat_flashsac_uses_canonical_manager_events_and_numpy_noise() -> None:
    registry.ensure_registries()
    _, env_cfg, _ = _materialize(
        "flashsac",
        ("task=go2_joystick_flat/mujoco",),
    )

    assert env_cfg.scene is not None
    assert env_cfg.scene.entities["robot"].body_names == ["base"]
    assert list(env_cfg.events) == [
        "reset_scene_to_default",
        "reset_root_state_uniform",
        "pd_gains",
        "randomize_rigid_body_mass",
        "randomize_rigid_body_com",
        "randomize_physics_scene_gravity",
        "push_by_setting_velocity",
    ]
    push = env_cfg.events["push_by_setting_velocity"]
    assert push.func is mdp.push_by_setting_velocity
    assert push.mode == "interval"
    assert push.interval_range_s == [15.0, 15.0]
    assert push.is_global_time is True

    for group_name in ("policy", "critic"):
        group = env_cfg.observations[group_name]
        assert group.enable_corruption is True
        assert type(group.terms["joint_pos"].noise).__name__ == "UniformNoiseCfg"
        assert group.terms["joint_pos"].noise.n_min == pytest.approx(-0.01)
        assert group.terms["joint_pos"].noise.n_max == pytest.approx(0.01)
        assert type(group.terms["joint_vel"].noise).__name__ == "UniformNoiseCfg"
        assert group.terms["joint_vel"].noise.n_min == pytest.approx(-0.1)
        assert group.terms["joint_vel"].noise.n_max == pytest.approx(0.1)

    assert env_cfg.rewards["tracking_lin_vel"].params["std"] == pytest.approx(0.4**0.5)
    assert env_cfg.rewards["base_height"].weight == pytest.approx(-20.0)
    assert env_cfg.rewards["contact"].weight == pytest.approx(1.5)


def test_go2_flat_drake_missing_dependency_is_explicit() -> None:
    from unilab.base.backend.drake.backend import ensure_drake_batch_available

    available, _ = ensure_drake_batch_available()
    if available:
        pytest.skip("DrakeUni is installed; missing-dependency behavior is not applicable")

    registry.ensure_registries()
    hydra_cfg, _, env_override = _materialize("ppo", ("task=go2_joystick_flat/drake",))
    with pytest.raises(ImportError, match="[Dd]rake"):
        registry.make(
            str(hydra_cfg.training.task_name),
            sim_backend="drake",
            env_cfg_override=env_override,
            num_envs=1,
        )
