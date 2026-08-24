"""Frozen A2Arm position-force layout contracts."""

from __future__ import annotations

import numpy as np

LEG_JOINT_NAMES = (
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
ARM_JOINT_NAMES = ("joint1", "joint2", "joint4", "joint6", "joint7")
JOINT_NAMES = LEG_JOINT_NAMES + ARM_JOINT_NAMES
ACTUATOR_NAMES = (
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
) + ARM_JOINT_NAMES

NUM_LEG = 12
NUM_ARM = 5
NUM_ACTIONS = 17
NUM_COMMANDS = 15
ACTOR_STEP_DIM = 73
CRITIC_CSE_DIM = 12
CRITIC_STEP_DIM = 134
ACTOR_HISTORY = 32
CRITIC_HISTORY = 3

CMD_VEL = slice(0, 3)
CMD_EE_POS = slice(3, 6)
CMD_EE_ORN = slice(6, 9)
CMD_EE_FORCE = slice(9, 12)
CMD_BASE_FORCE = slice(12, 15)

SENSOR_NAMES = (
    "gyro",
    "local_linvel",
    "upvector",
    "global_angvel",
    "global_position",
    "global_linvel",
    "endpoint_pos",
    "endpoint_quat",
    "endpoint_vel",
    "armbasepoint_world_pos",
    "armbasepoint_world_quat",
    "FL_global_linvel",
    "FR_global_linvel",
    "RL_global_linvel",
    "RR_global_linvel",
    "FL_pos",
    "FR_pos",
    "RL_pos",
    "RR_pos",
    "FL_thigh_pos",
    "FR_thigh_pos",
    "RL_thigh_pos",
    "RR_thigh_pos",
    "FL_foot_contact",
    "FR_foot_contact",
    "RL_foot_contact",
    "RR_foot_contact",
    "FL_foot_force",
    "FR_foot_force",
    "RL_foot_force",
    "RR_foot_force",
    "base1_contact",
    "base2_contact",
    "base3_contact",
    "FL_thigh_contact",
    "FR_thigh_contact",
    "RL_thigh_contact",
    "RR_thigh_contact",
    "FL_calf_contact1",
    "FR_calf_contact1",
    "RL_calf_contact1",
    "RR_calf_contact1",
    "FL_calf_contact2",
    "FR_calf_contact2",
    "RL_calf_contact2",
    "RR_calf_contact2",
)


def validate_layout() -> None:
    assert len(LEG_JOINT_NAMES) == NUM_LEG
    assert len(ARM_JOINT_NAMES) == NUM_ARM
    assert len(JOINT_NAMES) == len(ACTUATOR_NAMES) == NUM_ACTIONS
    actor = 2 + 3 + NUM_ACTIONS * 3 + 2 + NUM_COMMANDS
    critic = 12 + NUM_LEG + 6 + NUM_ACTIONS + 4 + 4 + 3 + 3 + actor
    if actor != ACTOR_STEP_DIM or critic != CRITIC_STEP_DIM:
        raise RuntimeError(f"A2Arm layout drifted: actor={actor}, critic={critic}")


validate_layout()

__all__ = [name for name in globals() if name.isupper()]
