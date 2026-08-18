"""Go2 flat Manager-Based config and real-runtime fixture tests."""

from __future__ import annotations

import math
from typing import Any, TypeVar, cast

import numpy as np
import pytest

from unilab.base.backend import create_backend, env_backend_kwargs
from unilab.base.entity import EntityCfg
from unilab.base.np_env import NpEnvState
from unilab.envs import ManagerBasedRlEnv, mdp
from unilab.envs.locomotion.common import manager_terms
from unilab.envs.locomotion.go2.manager_based_cfg import (
    make_go2_joystick_flat_manager_cfg,
)
from unilab.envs.mdp import JointPositionAction, JointPositionActionCfg
from unilab.managers import EventTermCfg, SceneEntityCfg

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
_TermCfg = TypeVar("_TermCfg")


def _active(terms: dict[str, _TermCfg | None]) -> dict[str, _TermCfg]:
    assert all(term is not None for term in terms.values())
    return cast(dict[str, _TermCfg], terms)


def test_go2_manager_factory_preserves_legacy_config_surface() -> None:
    cfg = make_go2_joystick_flat_manager_cfg()

    cfg.validate()
    assert cfg.sim_dt == pytest.approx(0.01)
    assert cfg.ctrl_dt == pytest.approx(0.02)
    assert cfg.max_episode_seconds == pytest.approx(20.0)
    assert cfg.policy_observation_group == "policy"
    assert cfg.critic_observation_group == "critic"
    assert cfg.scene is not None
    assert cfg.scene.default_keyframe_name == "home"

    robot = cfg.scene.entities["robot"]
    assert robot.root_body_name == "base"
    assert robot.joint_names == _JOINT_NAMES
    assert robot.actuator_names == _ACTUATOR_NAMES

    expected_policy = [
        "base_ang_vel",
        "projected_gravity",
        "joint_pos",
        "joint_vel",
        "actions",
        "command",
        "gait_phase",
    ]
    observations = _active(cfg.observations)
    assert list(observations) == ["policy", "critic"]
    policy_terms = _active(observations["policy"].terms)
    critic_terms = _active(observations["critic"].terms)
    expected_policy_terms = [
        ("base_ang_vel", mdp.builtin_sensor, {"sensor_name": "gyro"}),
        (
            "projected_gravity",
            mdp.projected_gravity_from_sensor,
            {"sensor_name": "upvector"},
        ),
        ("joint_pos", mdp.joint_pos_rel, {}),
        ("joint_vel", mdp.joint_vel_rel, {}),
        ("actions", mdp.last_action, {}),
        ("command", mdp.generated_commands, {"command_name": "twist"}),
        ("gait_phase", manager_terms.quadruped_gait_phase, {"frequency": 2.0}),
    ]
    assert list(policy_terms) == expected_policy
    assert [(name, term.func, term.params) for name, term in policy_terms.items()] == (
        expected_policy_terms
    )
    assert [(name, term.func, term.params) for name, term in critic_terms.items()] == [
        *expected_policy_terms,
        ("base_lin_vel", mdp.builtin_sensor, {"sensor_name": "local_linvel"}),
    ]

    actions = _active(cfg.actions)
    assert list(actions) == ["joint_pos"]
    action = actions["joint_pos"]
    assert isinstance(action, JointPositionActionCfg)
    assert action.entity_name == "robot"
    assert action.actuator_names == (".*",)
    assert action.scale == pytest.approx(0.25)
    assert action.use_default_offset is True

    commands = _active(cfg.commands)
    assert list(commands) == ["twist"]
    command = commands["twist"]
    assert isinstance(command, mdp.UniformVelocityCommandCfg)
    assert command.resampling_time_range == (20.0, 20.0)
    assert command.heading_command is False
    assert command.heading_control_stiffness == pytest.approx(0.5)
    assert command.rel_standing_envs == 0.0
    assert command.rel_heading_envs == 0.0
    assert command.rel_world_envs == 0.0
    assert command.rel_forward_envs == 0.0
    assert command.init_velocity_prob == 0.0
    assert command.ranges.lin_vel_x == (-0.6, 1.0)
    assert command.ranges.lin_vel_y == (-0.4, 0.4)
    assert command.ranges.ang_vel_z == (-0.8, 0.8)
    assert command.ranges.heading is None

    events = _active(cfg.events)
    assert list(events) == ["reset_scene_to_default", "reset_root_state_uniform", "pd_gains"]
    assert events["reset_scene_to_default"].func is mdp.reset_scene_to_default
    assert events["reset_scene_to_default"].mode == "reset"
    assert events["reset_scene_to_default"].params == {}
    root_reset = events["reset_root_state_uniform"]
    assert root_reset.func is mdp.reset_root_state_uniform
    assert root_reset.mode == "reset"
    assert root_reset.params["pose_range"] == {
        "x": (-0.5, 0.5),
        "y": (-0.5, 0.5),
        "z": (0.0, 0.0),
        "roll": (0.0, 0.0),
        "pitch": (0.0, 0.0),
        "yaw": (-math.pi, math.pi),
    }
    assert root_reset.params["velocity_range"] == {
        key: (-0.5, 0.5) for key in ("x", "y", "z", "roll", "pitch", "yaw")
    }
    pd_gains = events["pd_gains"]
    assert pd_gains.func is mdp.pd_gains
    assert pd_gains.mode == "reset"
    assert pd_gains.params == {
        "kp_range": (31.5, 38.5),
        "kd_range": (0.45, 0.55),
        "operation": "abs",
    }

    expected_rewards = {
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
    rewards = _active(cfg.rewards)
    assert list(rewards) == list(expected_rewards)
    assert {name: term.weight for name, term in rewards.items()} == expected_rewards
    assert [(name, term.func, term.params) for name, term in rewards.items()] == [
        (
            "tracking_lin_vel",
            manager_terms.track_lin_vel_xy_exp,
            {"std": 0.5, "command_name": "twist"},
        ),
        (
            "tracking_ang_vel",
            manager_terms.track_ang_vel_z_exp,
            {"std": 0.5, "command_name": "twist"},
        ),
        ("lin_vel_z", manager_terms.lin_vel_z_l2, {}),
        ("ang_vel_xy", manager_terms.ang_vel_xy_l2, {}),
        ("base_height", manager_terms.base_height_l2, {"target_height": 0.3}),
        ("action_rate", mdp.action_rate_l2, {}),
        ("similar_to_default", manager_terms.joint_deviation_l1, {}),
        (
            "contact",
            manager_terms.feet_phase_contact,
            {
                "frequency": 2.0,
                "sensor_names": (
                    "FL_foot_contact",
                    "FR_foot_contact",
                    "RL_foot_contact",
                    "RR_foot_contact",
                ),
                "contact_threshold": 0.1,
                "stance_threshold": 0.6,
            },
        ),
        (
            "swing_feet_z",
            manager_terms.feet_phase_swing_height,
            {
                "frequency": 2.0,
                "sensor_names": ("FL_pos", "FR_pos", "RL_pos", "RR_pos"),
                "target_height": 0.1,
                "kernel": 0.01,
                "swing_start": 0.6,
            },
        ),
    ]

    terminations = _active(cfg.terminations)
    assert list(terminations) == ["time_out", "bad_orientation"]
    assert terminations["time_out"].func is mdp.time_out
    assert terminations["time_out"].time_out is True
    assert terminations["time_out"].params == {}
    assert terminations["bad_orientation"].func is mdp.bad_orientation
    assert terminations["bad_orientation"].time_out is False
    assert terminations["bad_orientation"].params["limit_angle"] == pytest.approx(math.pi / 3.0)


def test_go2_manager_factory_executes_on_real_mujoco() -> None:
    cfg = make_go2_joystick_flat_manager_cfg()
    assert cfg.scene is not None
    backend = create_backend(
        "mujoco",
        cfg.scene,
        2,
        cfg.sim_dt,
        base_name="base",
        add_body_sensors=True,
        **env_backend_kwargs(cfg),
    )
    env = ManagerBasedRlEnv(cfg, backend, 2)
    try:
        assert env.obs_groups_spec == {"obs": 49, "critic": 52}
        assert env.action_space.shape == (12,)
        assert env.observation_manager.active_terms == {
            "policy": [
                "base_ang_vel",
                "projected_gravity",
                "joint_pos",
                "joint_vel",
                "actions",
                "command",
                "gait_phase",
            ],
            "critic": [
                "base_ang_vel",
                "projected_gravity",
                "joint_pos",
                "joint_vel",
                "actions",
                "command",
                "gait_phase",
                "base_lin_vel",
            ],
        }
        assert env.reward_manager.active_terms == list(cfg.rewards)
        assert env.termination_manager.active_terms == list(cfg.terminations)
        assert env.event_manager.active_terms["reset"] == list(cfg.events)

        action = env.action_manager.get_term("joint_pos")
        assert isinstance(action, JointPositionAction)
        assert action.target_names == list(_JOINT_NAMES)
        np.testing.assert_allclose(action.offset, np.broadcast_to(_HOME_JOINT_POS, (2, 12)))

        obs, info = env.reset(seed=7)
        assert set(obs) == {"obs", "critic"}
        assert obs["obs"].shape == (2, 49)
        assert obs["critic"].shape == (2, 52)
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
        assert isinstance(state, NpEnvState)
        assert state.obs["obs"].shape == (2, 49)
        assert state.obs["critic"].shape == (2, 52)
        for value in (*state.obs.values(), state.reward):
            assert isinstance(value, np.ndarray)
            assert np.isfinite(value).all()
        assert state.terminated.dtype == np.bool_
        assert state.truncated.dtype == np.bool_
    finally:
        env.close()


def test_go2_manager_reset_randomization_mutates_real_mujoco_payload() -> None:
    cfg = make_go2_joystick_flat_manager_cfg()
    assert cfg.scene is not None
    robot = cfg.scene.entities["robot"]
    cfg.scene.entities["robot"] = EntityCfg(
        root_body_name=robot.root_body_name,
        joint_names=robot.joint_names,
        body_names=("base",),
        actuator_names=robot.actuator_names,
    )
    asset_cfg = SceneEntityCfg("robot", body_names=("base",))
    cfg.events.update(
        {
            "mass": EventTermCfg(
                func=mdp.randomize_rigid_body_mass,
                mode="reset",
                params={
                    "asset_cfg": asset_cfg,
                    "mass_distribution_params": (1.25, 1.25),
                    "operation": "scale",
                    "recompute_inertia": False,
                },
            ),
            "com": EventTermCfg(
                func=mdp.randomize_rigid_body_com,
                mode="reset",
                params={"asset_cfg": asset_cfg, "com_range": {"x": (0.02, 0.02)}},
            ),
            "gravity": EventTermCfg(
                func=mdp.randomize_physics_scene_gravity,
                mode="reset",
                params={
                    "gravity_distribution_params": ([0.0, 0.0, -9.7],) * 2,
                    "operation": "abs",
                },
            ),
        }
    )
    backend = create_backend(
        "mujoco",
        cfg.scene,
        2,
        cfg.sim_dt,
        base_name="base",
        add_body_sensors=True,
        **env_backend_kwargs(cfg),
    )
    base_id = int(backend.get_body_ids(("base",))[0])
    default_mass = backend.get_body_mass()
    default_ipos = backend.get_body_ipos()
    env = ManagerBasedRlEnv(cfg, backend, 2)
    try:
        env.reset(seed=31)
        pool = cast(Any, backend)._pool
        assert pool is not None
        for env_id in range(2):
            mass = pool.get_field(env_id, "body_mass")
            ipos = pool.get_field(env_id, "body_ipos").reshape(-1, 3)
            gravity = pool.get_field(env_id, "gravity")
            assert mass[base_id] == pytest.approx(default_mass[base_id] * 1.25)
            np.testing.assert_allclose(ipos[base_id], default_ipos[base_id] + [0.02, 0.0, 0.0])
            np.testing.assert_allclose(gravity, [0.0, 0.0, -9.7])
    finally:
        env.close()


def _read_runtime_actuator_gains(backend_type: str, backend) -> tuple[np.ndarray, np.ndarray]:
    if backend_type == "mujoco":
        assert backend._pool is not None
        kp = np.stack([backend._pool.get_field(index, "kp") for index in range(backend.num_envs)])
        kd = np.stack([backend._pool.get_field(index, "kd") for index in range(backend.num_envs)])
        return kp, kd
    assert backend_type == "motrix"
    actuators = sorted(backend._position_actuators, key=lambda actuator: int(actuator.index))
    kp = np.column_stack(
        [np.asarray(actuator.get_kp_override(backend._data)).reshape(-1) for actuator in actuators]
    )
    kd = np.column_stack(
        [np.asarray(actuator.get_kd_override(backend._data)).reshape(-1) for actuator in actuators]
    )
    return kp, kd


@pytest.mark.parametrize("backend_type", ["mujoco", "motrix"])
def test_go2_manager_pd_gains_mutates_real_backend_on_reset(backend_type: str) -> None:
    cfg = make_go2_joystick_flat_manager_cfg()
    assert cfg.scene is not None
    backend = create_backend(
        backend_type,
        cfg.scene,
        2,
        cfg.sim_dt,
        base_name="base",
        add_body_sensors=True,
        **env_backend_kwargs(cfg),
    )
    env = ManagerBasedRlEnv(cfg, backend, 2)
    try:
        env.reset(seed=29)
        kp, kd = _read_runtime_actuator_gains(backend_type, backend)
        assert kp.shape == (2, 12)
        assert kd.shape == (2, 12)
        assert np.all((kp >= 31.5) & (kp <= 38.5))
        assert np.all((kd >= 0.45) & (kd <= 0.55))
        assert np.unique(np.round(kp, 6)).size > 1
        assert np.unique(np.round(kd, 6)).size > 1
    finally:
        env.close()
