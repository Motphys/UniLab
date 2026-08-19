"""Hydra-owned production contract for the Go2W flat Manager-Based task."""

from __future__ import annotations

import inspect
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
from unilab.base.config_materialization import apply_cfg_overrides
from unilab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg, make_manager_based_rl_env
from unilab.tasks.locomotion.go2w import manager_terms
from unilab.tasks.locomotion.go2w.manager_terms import (
    Go2WMixedAction,
    Go2WMixedActionCfg,
    Go2WVelocityCommandCfg,
)
from unilab.training.backend_adapter import BackendAdapter

ROOT_DIR = Path(__file__).parents[4]
CONF_DIR = ROOT_DIR / "conf"

_LEG_JOINT_NAMES = tuple(
    f"{leg}_{joint}_joint" for leg in ("FR", "FL", "RR", "RL") for joint in ("hip", "thigh", "calf")
)
_WHEEL_JOINT_NAMES = tuple(f"{leg}_wheel_joint" for leg in ("FR", "FL", "RR", "RL"))
_JOINT_NAMES = (*_LEG_JOINT_NAMES, *_WHEEL_JOINT_NAMES)
_ACTUATOR_NAMES = tuple(name.removesuffix("_joint") for name in _JOINT_NAMES)
_HOME_JOINT_POS = np.asarray([0.0, 0.8, -1.5] * 4 + [0.0] * 4, dtype=np.float32)


def _compose(backend: str) -> DictConfig:
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_DIR / "ppo"), version_base="1.3"):
        return compose("config", overrides=[f"task=go2w_joystick_flat/{backend}"])


def _materialize(
    backend: str,
) -> tuple[DictConfig, ManagerBasedRlEnvCfg, dict[str, Any]]:
    hydra_cfg = _compose(backend)
    env_override = BackendAdapter(hydra_cfg, root_dir=ROOT_DIR).build_task_env_cfg_override()
    env_cfg = registry.materialize_env_config("Go2WJoystickFlat")
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


@pytest.mark.parametrize("backend", ["mujoco", "motrix", "drake"])
def test_go2w_flat_owner_materializes_complete_plain_manager_cfg(backend: str) -> None:
    registry.ensure_registries()
    hydra_cfg, env_cfg, _ = _materialize(backend)

    assert hydra_cfg.training.task_name == "Go2WJoystickFlat"
    assert hydra_cfg.training.sim_backend == backend
    assert list(hydra_cfg.algo.obs_groups.actor) == ["actor"]
    assert list(hydra_cfg.algo.obs_groups.critic) == ["critic"]
    assert env_cfg.sim_dt == pytest.approx(0.005)
    assert env_cfg.ctrl_dt == pytest.approx(0.02)
    assert env_cfg.max_episode_seconds == pytest.approx(20.0)
    assert env_cfg.policy_observation_group == "policy"
    assert env_cfg.critic_observation_group == "critic"

    assert env_cfg.scene is not None
    assert env_cfg.scene.model_file.endswith("robots/go2w/scene_flat.xml")
    assert env_cfg.scene.default_keyframe_name == "home"
    robot = env_cfg.scene.entities["robot"]
    assert robot.root_body_name == "base_link"
    assert tuple(robot.joint_names or ()) == _JOINT_NAMES
    assert tuple(robot.actuator_names or ()) == _ACTUATOR_NAMES
    assert robot.body_names == ["base_link"]

    policy_terms = [
        "base_ang_vel",
        "projected_gravity",
        "leg_joint_pos",
        "leg_joint_vel",
        "wheel_joint_vel",
        "actions",
        "command",
    ]
    assert list(env_cfg.observations) == ["policy", "critic"]
    assert list(env_cfg.observations["policy"].terms) == policy_terms
    assert list(env_cfg.observations["critic"].terms) == [
        *policy_terms,
        "base_lin_vel",
        "motor_torque",
    ]

    assert list(env_cfg.actions) == ["motor"]
    action = env_cfg.actions["motor"]
    assert isinstance(action, Go2WMixedActionCfg)
    assert action.leg_action_scale == pytest.approx(0.5)
    assert action.wheel_action_scale == pytest.approx(10.0)
    assert action.leg_kp == pytest.approx(50.0)
    assert action.leg_kd == pytest.approx(1.5)
    assert action.wheel_kd == pytest.approx(0.5)

    command = env_cfg.commands["twist"]
    assert isinstance(command, Go2WVelocityCommandCfg)
    assert command.resampling_time_range == [20.0, 20.0]
    assert command.planar_dead_zone == pytest.approx(0.2)
    assert tuple(command.ranges.lin_vel_x) == (0.0, 1.0)
    assert tuple(command.ranges.lin_vel_y) == (0.0, 0.0)
    assert tuple(command.ranges.ang_vel_z) == (-1.0, 1.0)

    assert list(env_cfg.events) == [
        "reset_scene_to_default",
        "reset_root_state_uniform",
        "motor_gains",
    ]
    gains = env_cfg.events["motor_gains"]
    assert gains.func is manager_terms.randomize_motor_gains
    assert gains.params["kp_multiplier_range"] == [1.0, 1.0]
    assert gains.params["kd_multiplier_range"] == [1.0, 1.0]
    assert list(env_cfg.terminations) == ["time_out", "bad_orientation"]
    assert {name: term.weight for name, term in env_cfg.rewards.items()} == {
        "tracking_lin_vel": 1.0,
        "tracking_ang_vel": 0.75,
        "lin_vel_z": -5.0,
        "ang_vel_xy": -0.1,
        "base_height": -100.0,
        "orientation": -2.0,
        "action_rate": -0.005,
        "similar_to_default": -0.5,
        "torques": -0.0002,
        "wheel_vel": 0.0,
        "alive": 0.5,
        "upward": 1.0,
    }

    for manager_name in ("observations", "events", "rewards", "terminations"):
        for term in getattr(env_cfg, manager_name).values():
            if term is None:
                continue
            terms = term.terms.values() if manager_name == "observations" else (term,)
            for nested in terms:
                if nested is None:
                    continue
                module = nested.func.__module__
                assert ".backend." not in module
                assert not any(name in module for name in (".mujoco", ".motrix", ".drake"))

    _assert_no_omegaconf(env_cfg)


def test_go2w_sac_drake_owner_uses_the_same_manager_contract() -> None:
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_DIR / "offpolicy"), version_base="1.3"):
        hydra_cfg = compose(
            "config",
            overrides=["algo=sac", "task=sac/go2w_joystick_flat/drake"],
        )
    env_override = BackendAdapter(hydra_cfg, root_dir=ROOT_DIR).build_task_env_cfg_override()
    env_cfg = registry.materialize_env_config("Go2WJoystickFlat")
    apply_cfg_overrides(env_cfg, env_override)

    assert isinstance(env_cfg, ManagerBasedRlEnvCfg)
    assert hydra_cfg.training.task_name == "Go2WJoystickFlat"
    assert hydra_cfg.training.sim_backend == "drake"
    assert list(env_cfg.actions) == ["motor"]
    assert env_cfg.scene is not None
    assert env_cfg.scene.default_keyframe_name == "home"
    env_cfg.validate()


def test_go2w_flat_and_rough_registries_are_manager_only() -> None:
    registry.ensure_registries()

    assert registry.list_registered_envs()["Go2WJoystickFlat"] == {
        "config_factory": "ManagerBasedRlEnvCfg",
        "available_backends": ["mujoco", "motrix", "drake"],
    }
    assert registry.list_registered_envs()["Go2WJoystickRough"] == {
        "config_factory": "ManagerBasedRlEnvCfg",
        "available_backends": ["mujoco", "motrix"],
    }
    for legacy_override in (
        {"reward_config": {}},
        {"domain_rand": {"randomize_kp": False}},
        {"control_config": {"action_scale": 0.5}},
    ):
        with pytest.raises(ValueError, match="has no attribute"):
            apply_cfg_overrides(ManagerBasedRlEnvCfg(), legacy_override)


@pytest.mark.parametrize("backend", ["mujoco", "motrix"])
def test_go2w_flat_registry_executes_real_manager_runtime(backend: str) -> None:
    registry.ensure_registries()
    _, _, env_override = _materialize(backend)
    try:
        env = registry.make(
            "Go2WJoystickFlat",
            sim_backend=backend,
            env_cfg_override=env_override,
            num_envs=2,
        )
    except ImportError as exc:
        pytest.skip(f"{backend} runtime unavailable: {exc}")

    try:
        assert isinstance(env, ManagerBasedRlEnv)
        assert env.obs_groups_spec == {"obs": 53, "critic": 72}
        assert env.action_space.shape == (16,)
        action = env.action_manager.get_term("motor")
        assert isinstance(action, Go2WMixedAction)
        np.testing.assert_allclose(action.leg_kp, 50.0)
        np.testing.assert_allclose(action.leg_kd, 1.5)

        obs, info = env.reset(seed=7)
        assert {name: value.shape for name, value in obs.items()} == {
            "obs": (2, 53),
            "critic": (2, 72),
        }
        assert isinstance(info, dict)
        np.testing.assert_allclose(
            env.scene["robot"].data.default_joint_pos,
            np.broadcast_to(_HOME_JOINT_POS, (2, 16)),
        )

        state = env.step(np.full((2, 16), 2.0, dtype=np.float32))
        np.testing.assert_allclose(action.raw_action, 1.0)
        np.testing.assert_allclose(action.previous_raw_action, 0.0)
        np.testing.assert_allclose(
            action.processed_action[:, :12],
            np.broadcast_to(_HOME_JOINT_POS[:12] + 0.5, (2, 12)),
        )
        np.testing.assert_allclose(action.processed_action[:, 12:], 10.0)
        np.testing.assert_allclose(state.obs["critic"][:, -16:], action.motor_torque)
        for value in (*state.obs.values(), state.reward, action.motor_torque):
            assert np.isfinite(value).all()

        env.reset(np.asarray([0], dtype=np.int32))
        np.testing.assert_allclose(action.raw_action[0], 0.0)
        np.testing.assert_allclose(action.raw_action[1], 1.0)
        np.testing.assert_allclose(action.motor_torque[0], 0.0)
    finally:
        env.close()


def test_go2w_flat_dead_zone_and_motor_gain_overrides_are_manager_owned() -> None:
    _, env_cfg, _ = _materialize("mujoco")
    command = env_cfg.commands["twist"]
    command.ranges.lin_vel_x = (0.1, 0.1)
    command.ranges.lin_vel_y = (0.0, 0.0)
    command.ranges.ang_vel_z = (0.0, 0.0)
    gains = env_cfg.events["motor_gains"].params
    gains["kp_multiplier_range"] = (0.5, 0.5)
    gains["kd_multiplier_range"] = (2.0, 2.0)

    env = make_manager_based_rl_env(env_cfg, num_envs=2, backend_type="mujoco")
    try:
        env.reset(seed=11)
        np.testing.assert_allclose(env.command_manager.get_command("twist"), 0.0)
        action = env.action_manager.get_term("motor")
        assert isinstance(action, Go2WMixedAction)
        np.testing.assert_allclose(action.leg_kp, 25.0)
        np.testing.assert_allclose(action.leg_kd, 3.0)
    finally:
        env.close()


def test_go2w_flat_incomplete_motor_selection_fails_closed() -> None:
    _, env_cfg, _ = _materialize("mujoco")
    action = env_cfg.actions["motor"]
    assert isinstance(action, Go2WMixedActionCfg)
    action.actuator_names = ["FR_.*"]

    with pytest.raises(ValueError, match="requires exactly 16 actuators and target joints"):
        make_manager_based_rl_env(env_cfg, num_envs=1, backend_type="mujoco")


def test_go2w_manager_terms_do_not_leak_backend_or_physical_layout() -> None:
    source = inspect.getsource(manager_terms)
    for forbidden in ("._backend", "getattr(", "hasattr(", "qpos", "qvel", "ASSETS_ROOT_PATH"):
        assert forbidden not in source
