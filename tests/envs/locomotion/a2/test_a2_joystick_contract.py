"""Hydra-owned Manager-Based production contract for A2JoystickFlat."""

from __future__ import annotations

import importlib
from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base import registry
from unilab.base.config_adapter import BackendAdapter
from unilab.base.config_materialization import apply_cfg_overrides
from unilab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg, mdp
from unilab.tasks.locomotion.common import manager_terms

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
    "FL_hip",
    "FL_thigh",
    "FL_calf",
    "FR_hip",
    "FR_thigh",
    "FR_calf",
    "RL_hip",
    "RL_thigh",
    "RL_calf",
    "RR_hip",
    "RR_thigh",
    "RR_calf",
)
_HOME_JOINT_POS = np.array(
    [-0.1, 0.9, -1.8, 0.1, 0.9, -1.8, -0.1, 0.9, -1.8, 0.1, 0.9, -1.8],
    dtype=np.float32,
)
_KP = np.array([100.0, 100.0, 150.0] * 4)
_KD = np.array([4.0, 4.0, 6.0] * 4)


def _compose() -> DictConfig:
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_DIR / "ppo"), version_base="1.3"):
        return compose("config", overrides=["task=a2_joystick_flat/mujoco"])


def _materialize() -> tuple[DictConfig, ManagerBasedRlEnvCfg, dict[str, Any]]:
    hydra_cfg = _compose()
    env_override = BackendAdapter(hydra_cfg, root_dir=ROOT_DIR).build_task_env_cfg_override()
    env_cfg = registry.materialize_env_config("A2JoystickFlat")
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


def test_a2_asset_declares_home_pose_and_per_joint_pd_defaults() -> None:
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_path(str(ASSETS_ROOT_PATH / "robots" / "a2" / "scene_flat.xml"))

    actuator_names = tuple(
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, index) for index in range(model.nu)
    )
    assert actuator_names == _ACTUATOR_NAMES
    affine = int(mujoco.mjtBias.mjBIAS_AFFINE)
    assert all(int(value) == affine for value in model.actuator_biastype)
    np.testing.assert_allclose(model.actuator_gainprm[:, 0], _KP)
    np.testing.assert_allclose(model.actuator_biasprm[:, 1], -_KP)
    np.testing.assert_allclose(model.actuator_biasprm[:, 2], -_KD)

    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    assert key_id >= 0
    assert model.nq == 19
    assert model.key_qpos[key_id, 2] == pytest.approx(0.4)
    np.testing.assert_allclose(model.key_qpos[key_id, 7:19], _HOME_JOINT_POS)
    np.testing.assert_allclose(model.key_ctrl[key_id], _HOME_JOINT_POS)


def test_a2_asset_exposes_manager_sensor_and_floor_friction_contract() -> None:
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_path(str(ASSETS_ROOT_PATH / "robots" / "a2" / "scene_flat.xml"))
    sensors = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, index)
        for index in range(model.nsensor)
    }
    assert {
        "gyro",
        "local_linvel",
        "upvector",
        "FL_pos",
        "FR_pos",
        "RL_pos",
        "RR_pos",
        "FL_foot_contact",
        "FR_foot_contact",
        "RL_foot_contact",
        "RR_foot_contact",
    } <= sensors

    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    foot = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "FL")
    assert floor >= 0 and foot >= 0
    assert int(model.geom_priority[floor]) > int(model.geom_priority[foot])
    assert int(model.geom_condim[floor]) == 6


def test_a2_owner_materializes_complete_plain_manager_config() -> None:
    registry.ensure_registries()
    hydra_cfg, env_cfg, _ = _materialize()

    assert hydra_cfg.training.task_name == "A2JoystickFlat"
    assert hydra_cfg.training.sim_backend == "mujoco"
    assert hydra_cfg.algo.max_iterations == 500
    assert list(hydra_cfg.algo.obs_groups.actor) == ["actor"]
    assert list(hydra_cfg.algo.obs_groups.critic) == ["critic"]
    assert env_cfg.sim_dt == pytest.approx(0.01)
    assert env_cfg.ctrl_dt == pytest.approx(0.02)
    assert env_cfg.max_episode_seconds == pytest.approx(20.0)
    assert env_cfg.policy_observation_group == "policy"
    assert env_cfg.critic_observation_group == "critic"

    assert env_cfg.scene is not None
    assert env_cfg.scene.model_file.endswith("robots/a2/scene_flat.xml")
    assert env_cfg.scene.default_keyframe_name == "home"
    robot = env_cfg.scene.entities["robot"]
    assert robot.root_body_name == "base_link"
    assert tuple(robot.joint_names or ()) == _JOINT_NAMES
    assert tuple(robot.actuator_names or ()) == _ACTUATOR_NAMES
    assert robot.body_names == ["base_link"]
    assert robot.geom_names == ["floor"]

    policy_terms = [
        "base_ang_vel",
        "projected_gravity",
        "joint_pos",
        "joint_vel",
        "actions",
        "command",
        "gait_phase",
    ]
    assert list(env_cfg.observations) == ["policy", "critic"]
    assert list(env_cfg.observations["policy"].terms) == policy_terms
    assert list(env_cfg.observations["critic"].terms) == [*policy_terms, "base_lin_vel"]
    assert env_cfg.observations["policy"].terms["gait_phase"].params == {
        "frequency": 2.0,
        "command_name": "twist",
        "command_threshold": 0.1,
    }
    assert list(env_cfg.actions) == ["joint_pos"]
    assert env_cfg.actions["joint_pos"].scale == pytest.approx(0.25)
    command = env_cfg.commands["twist"]
    assert command.resampling_time_range == [5.0, 5.0]
    assert command.rel_standing_envs == pytest.approx(0.1)
    assert tuple(command.ranges.lin_vel_x) == (-0.6, 1.0)

    expected_weights = {
        "tracking_lin_vel": 1.0,
        "tracking_ang_vel": 0.4,
        "lin_vel_z": -5.0,
        "ang_vel_xy": -0.1,
        "base_height": -100.0,
        "action_rate": -0.02,
        "similar_to_default": -0.25,
        "contact": 0.5,
        "swing_feet_z": 4.0,
        "stand_still": -4.0,
        "hip_deviation": -1.0,
        "stand_feet_air": -1.0,
    }
    assert {name: term.weight for name, term in env_cfg.rewards.items()} == expected_weights
    assert env_cfg.rewards["stand_still"].func is manager_terms.stand_still_l1
    assert env_cfg.rewards["stand_feet_air"].func is manager_terms.feet_air_while_standing
    assert env_cfg.rewards["hip_deviation"].params["asset_cfg"].joint_names == ".*_hip_joint"
    for name in ("contact", "swing_feet_z"):
        assert env_cfg.rewards[name].params["command_name"] == "twist"
        assert env_cfg.rewards[name].params["command_threshold"] == pytest.approx(0.1)

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


def test_a2_owner_declares_all_randomization_as_manager_events() -> None:
    _, env_cfg, _ = _materialize()
    assert list(env_cfg.events) == [
        "reset_scene_to_default",
        "reset_root_state_uniform",
        "base_mass",
        "base_com",
        "foot_friction",
        "joint_armature",
        "pd_gains",
        "push_robot",
    ]

    mass = env_cfg.events["base_mass"]
    assert mass.func is mdp.randomize_rigid_body_mass
    assert mass.params["mass_distribution_params"] == [0.0, 8.0]
    assert mass.params["recompute_inertia"] is False
    com = env_cfg.events["base_com"]
    assert com.func is mdp.randomize_rigid_body_com
    assert com.params["com_range"] == {
        "x": [-0.08, 0.08],
        "y": [-0.08, 0.08],
        "z": [-0.08, 0.08],
    }
    friction = env_cfg.events["foot_friction"]
    assert friction.func is mdp.geom_friction
    assert friction.params["ranges"] == [0.3, 1.6]
    assert friction.params["operation"] == "scale"
    assert friction.params["shared_random"] is True
    armature = env_cfg.events["joint_armature"]
    assert armature.func is mdp.joint_armature
    assert armature.params["ranges"] == [0.9, 1.1]
    gains = env_cfg.events["pd_gains"]
    assert gains.func is mdp.pd_gains
    assert gains.params["kp_range"] == [0.9, 1.1]
    assert gains.params["kd_range"] == [0.9, 1.1]
    push = env_cfg.events["push_robot"]
    assert push.func is mdp.push_by_setting_velocity
    assert push.mode == "interval"
    assert push.interval_range_s == [8.0, 8.0]
    assert push.is_global_time is True


def test_a2_registry_has_no_legacy_config_or_runtime_fallback() -> None:
    registry.ensure_registries()
    module = importlib.import_module("unilab.tasks.locomotion.a2.joystick")
    assert not hasattr(module, "A2JoystickCfg")
    assert not hasattr(module, "A2JoystickFlatEnv")
    assert not hasattr(module, "A2JoystickDomainRandomizationProvider")
    assert registry.list_registered_envs()["A2JoystickFlat"] == {
        "config_factory": "ManagerBasedRlEnvCfg",
        "available_backends": ["mujoco"],
    }
    for legacy_override in (
        {"reward_config": {}},
        {"domain_rand": {"randomize_kp": True}},
        {"control_config": {"action_scale": 0.4}},
    ):
        with pytest.raises(ValueError, match="has no attribute"):
            apply_cfg_overrides(ManagerBasedRlEnvCfg(), legacy_override)


def test_a2_registry_executes_real_manager_runtime() -> None:
    pytest.importorskip("mujoco")
    try:
        from mujoco_uni.batch_env import BatchEnvPool  # noqa: F401
    except Exception:
        pytest.skip("mujoco_uni.batch_env not available")

    hydra_cfg, _, env_override = _materialize()
    env = registry.make(
        str(hydra_cfg.training.task_name),
        sim_backend="mujoco",
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
        assert env.event_manager.active_terms == {
            "reset": [
                "reset_scene_to_default",
                "reset_root_state_uniform",
                "base_mass",
                "base_com",
                "foot_friction",
                "joint_armature",
                "pd_gains",
            ],
            "interval": ["push_robot"],
        }

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
        for _ in range(10):
            state = env.step(np.zeros((2, 12), dtype=np.float32))
        assert state.reward.shape == (2,)
        for value in (*state.obs.values(), state.reward):
            assert isinstance(value, np.ndarray)
            assert np.isfinite(value).all()
    finally:
        env.close()
