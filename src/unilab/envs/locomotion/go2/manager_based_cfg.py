"""Unregistered Manager-Based configuration for the Go2 flat pilot."""

from __future__ import annotations

import math

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base.entity import EntityCfg
from unilab.base.scene import SceneCfg
from unilab.envs import ManagerBasedRlEnvCfg, mdp
from unilab.envs.locomotion.common import manager_terms
from unilab.managers import (
    EventTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)

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
_FOOT_CONTACT_SENSORS = (
    "FL_foot_contact",
    "FR_foot_contact",
    "RL_foot_contact",
    "RR_foot_contact",
)
_FOOT_POSITION_SENSORS = ("FL_pos", "FR_pos", "RL_pos", "RR_pos")
_GAIT_FREQUENCY = 2.0


def make_go2_joystick_flat_manager_cfg() -> ManagerBasedRlEnvCfg:
    """Build the NumPy Manager-Based equivalent of the legacy Go2 flat task.

    The factory is intentionally not registered.  It proves the task-owned community
    config surface without changing the production Go2 registry or Hydra owners.
    """
    policy_terms: dict[str, ObservationTermCfg | None] = {
        "base_ang_vel": ObservationTermCfg(
            func=mdp.builtin_sensor,
            params={"sensor_name": "gyro"},
        ),
        "projected_gravity": ObservationTermCfg(
            func=mdp.projected_gravity_from_sensor,
            params={"sensor_name": "upvector"},
        ),
        "joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel),
        "joint_vel": ObservationTermCfg(func=mdp.joint_vel_rel),
        "actions": ObservationTermCfg(func=mdp.last_action),
        "command": ObservationTermCfg(
            func=mdp.generated_commands,
            params={"command_name": "twist"},
        ),
        "gait_phase": ObservationTermCfg(
            func=manager_terms.quadruped_gait_phase,
            params={"frequency": _GAIT_FREQUENCY},
        ),
    }
    critic_terms: dict[str, ObservationTermCfg | None] = {
        **policy_terms,
        "base_lin_vel": ObservationTermCfg(
            func=mdp.builtin_sensor,
            params={"sensor_name": "local_linvel"},
        ),
    }

    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            model_file=str(ASSETS_ROOT_PATH / "robots" / "go2" / "scene_flat.xml"),
            entities={
                "robot": EntityCfg(
                    root_body_name="base",
                    joint_names=_JOINT_NAMES,
                    actuator_names=_ACTUATOR_NAMES,
                )
            },
            default_keyframe_name="home",
        ),
        sim_dt=0.01,
        ctrl_dt=0.02,
        max_episode_seconds=20.0,
        observations={
            "policy": ObservationGroupCfg(terms=policy_terms),
            "critic": ObservationGroupCfg(terms=critic_terms),
        },
        actions={
            "joint_pos": mdp.JointPositionActionCfg(
                entity_name="robot",
                actuator_names=(".*",),
                scale=0.25,
                use_default_offset=True,
            )
        },
        commands={
            "twist": mdp.UniformVelocityCommandCfg(
                entity_name="robot",
                resampling_time_range=(20.0, 20.0),
                heading_command=False,
                heading_control_stiffness=0.5,
                rel_standing_envs=0.0,
                rel_heading_envs=0.0,
                rel_world_envs=0.0,
                rel_forward_envs=0.0,
                init_velocity_prob=0.0,
                ranges=mdp.UniformVelocityCommandCfg.Ranges(
                    lin_vel_x=(-0.6, 1.0),
                    lin_vel_y=(-0.4, 0.4),
                    ang_vel_z=(-0.8, 0.8),
                ),
            )
        },
        events={
            "reset_scene_to_default": EventTermCfg(
                func=mdp.reset_scene_to_default,
                mode="reset",
            ),
            "reset_root_state_uniform": EventTermCfg(
                func=mdp.reset_root_state_uniform,
                mode="reset",
                params={
                    "pose_range": {
                        "x": (-0.5, 0.5),
                        "y": (-0.5, 0.5),
                        "z": (0.0, 0.0),
                        "roll": (0.0, 0.0),
                        "pitch": (0.0, 0.0),
                        "yaw": (-math.pi, math.pi),
                    },
                    "velocity_range": {
                        "x": (-0.5, 0.5),
                        "y": (-0.5, 0.5),
                        "z": (-0.5, 0.5),
                        "roll": (-0.5, 0.5),
                        "pitch": (-0.5, 0.5),
                        "yaw": (-0.5, 0.5),
                    },
                },
            ),
        },
        rewards={
            "tracking_lin_vel": RewardTermCfg(
                func=manager_terms.track_lin_vel_xy_exp,
                weight=1.0,
                params={"std": math.sqrt(0.25), "command_name": "twist"},
            ),
            "tracking_ang_vel": RewardTermCfg(
                func=manager_terms.track_ang_vel_z_exp,
                weight=0.2,
                params={"std": math.sqrt(0.25), "command_name": "twist"},
            ),
            "lin_vel_z": RewardTermCfg(func=manager_terms.lin_vel_z_l2, weight=-5.0),
            "ang_vel_xy": RewardTermCfg(func=manager_terms.ang_vel_xy_l2, weight=-0.1),
            "base_height": RewardTermCfg(
                func=manager_terms.base_height_l2,
                weight=-100.0,
                params={"target_height": 0.3},
            ),
            "action_rate": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.005),
            "similar_to_default": RewardTermCfg(
                func=manager_terms.joint_deviation_l1,
                weight=-0.1,
            ),
            "contact": RewardTermCfg(
                func=manager_terms.feet_phase_contact,
                weight=0.24,
                params={
                    "frequency": _GAIT_FREQUENCY,
                    "sensor_names": _FOOT_CONTACT_SENSORS,
                    "contact_threshold": 0.1,
                    "stance_threshold": 0.6,
                },
            ),
            "swing_feet_z": RewardTermCfg(
                func=manager_terms.feet_phase_swing_height,
                weight=4.0,
                params={
                    "frequency": _GAIT_FREQUENCY,
                    "sensor_names": _FOOT_POSITION_SENSORS,
                    "target_height": 0.1,
                    "kernel": 0.01,
                    "swing_start": 0.6,
                },
            ),
        },
        terminations={
            "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
            "bad_orientation": TerminationTermCfg(
                func=mdp.bad_orientation,
                params={"limit_angle": math.acos(0.5)},
            ),
        },
        policy_observation_group="policy",
        critic_observation_group="critic",
    )


__all__ = ["make_go2_joystick_flat_manager_cfg"]
