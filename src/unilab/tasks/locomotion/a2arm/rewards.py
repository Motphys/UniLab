"""A2Arm reward and termination terms ported as manager callables."""

from __future__ import annotations

import numpy as np

from unilab.utils.rotation import np_quat_apply, np_quat_apply_inverse, np_yaw_quat

from .constants import NUM_LEG
from .observations import _roll_pitch


def _parts(env):
    state = env.command_manager.get_term("task_state")
    action = env.action_manager.get_term("joint_pd")
    state.prepare_transition(int(env.common_step_counter))
    robot = env.scene["robot"]
    return state, action, robot


def a2arm_reward(env, name: str) -> np.ndarray:
    state, action, robot = _parts(env)
    q = robot.data.joint_pos
    qd = robot.data.joint_vel
    vel = robot.data.root_link_lin_vel_b
    ang = robot.data.root_link_ang_vel_b
    command = state.command
    moving = (
        (np.abs(command[:, 0]) > state.cfg.velocity_clip[0])
        | (np.abs(command[:, 1]) > state.cfg.velocity_clip[1])
        | (np.abs(command[:, 2]) > state.cfg.velocity_clip[2])
    )
    if name == "tracking_lin_vel_force_world":
        force = np_quat_apply_inverse(
            np_yaw_quat(robot.data.root_link_quat_w), state.force_base_world
        )
        target = (
            command[:, :2]
            + (force[:, :2] + state.force_base_command[:, :2]) / state.cfg.base_force_kd
        )
        target_moving = (
            (np.abs(target[:, 0]) > state.cfg.velocity_clip[0])
            | (np.abs(target[:, 1]) > state.cfg.velocity_clip[1])
            | (np.abs(command[:, 2]) > state.cfg.velocity_clip[2])
        )
        target *= target_moving[:, None]
        return np.exp(-np.sum((target - vel[:, :2]) ** 2, axis=1) / 0.25)
    if name == "tracking_ee_force_world":
        yaw = np_yaw_quat(robot.data.root_link_quat_w)
        goal = (
            state.current_goal_world
            + (state.force_ee_world + np_quat_apply(yaw, state.force_ee_command))
            / state.cfg.gripper_force_kp
        )
        error = np.sum(np.abs(state.ee_world_pos() - goal), axis=1)
        return np.exp(-error / 1.0 * 2.0)
    if name == "tracking_ang_vel":
        return np.exp(-((command[:, 2] - ang[:, 2]) ** 2) / 0.25)
    if name == "orientation":
        gravity = robot.data.projected_gravity_b
        return gravity[:, 0] ** 2 + gravity[:, 1] ** 2
    if name == "lin_vel_z":
        return vel[:, 2] ** 2
    if name == "ang_vel_xy":
        return np.sum(ang[:, :2] ** 2, axis=1)
    if name == "alive":
        return np.ones(env.num_envs, dtype=np.float32)
    if name == "ref_dof_leg":
        return np.exp(-np.sum(np.abs(q[:, :NUM_LEG] - state.reference_dof_pos), axis=1) * 0.1)
    if name == "action_rate":
        return np.sum(
            (action.raw_action[:, :NUM_LEG] - action.previous_raw_action[:, :NUM_LEG]) ** 2, axis=1
        )
    if name == "action_rate_arm":
        return np.sum(
            (action.raw_action[:, NUM_LEG:] - action.previous_raw_action[:, NUM_LEG:]) ** 2, axis=1
        )
    if name == "torques":
        return np.sum(action.applied_torque[:, :NUM_LEG] ** 2, axis=1)
    if name == "dof_vel":
        return np.sum(qd[:, :NUM_LEG] ** 2, axis=1)
    if name == "dof_vel_arm":
        return np.sum(qd[:, NUM_LEG:] ** 2, axis=1)
    if name == "dof_acc":
        return np.sum(
            ((state.last_dof_vel[:, :NUM_LEG] - qd[:, :NUM_LEG]) / state.cfg.ctrl_dt) ** 2, axis=1
        )
    if name == "dof_acc_arm":
        return np.sum(
            ((state.last_dof_vel[:, NUM_LEG:] - qd[:, NUM_LEG:]) / state.cfg.ctrl_dt) ** 2, axis=1
        )
    if name == "base_height":
        return (robot.data.root_link_pos_w[:, 2] - 0.435) ** 2
    if name == "hip_pos":
        return np.sum(
            (q[:, [0, 3, 6, 9]] - robot.data.default_joint_pos[:, [0, 3, 6, 9]]) ** 2, axis=1
        )
    if name == "torque_limits":
        return np.sum(
            np.clip(np.abs(action.applied_torque) - 0.9 * action.torque_limits, 0.0, None), axis=1
        )
    if name == "dof_pos_limits":
        limits = state.soft_dof_pos_limits
        return np.sum(
            np.clip(limits[:, 0] - q, 0.0, None) + np.clip(q - limits[:, 1], 0.0, None), axis=1
        )
    if name == "stand_still":
        return np.exp(
            -np.sum(np.abs(q[:, :NUM_LEG] - robot.data.default_joint_pos[:, :NUM_LEG]), axis=1)
            * 0.05
        ) * (~moving)
    if name == "collision":
        return np.sum(state.undesired_contacts() > 0.5, axis=1).astype(np.float32)
    if name == "feet_contact_number":
        return np.mean(np.where(state.foot_contact == (state.stance_mask > 0.5), 1.0, -0.3), axis=1)
    if name == "feet_air_time":
        return np.sum((state.air_time_snapshot - 0.5) * state.first_contact, axis=1) * moving
    if name == "feet_height":
        value = np.clip(np.max(state.foot_pos()[:, :2, 2], axis=1) - 0.12, None, 0.0)
        return np.where(moving, value, 0.0).astype(np.float32)
    if name == "feet_height_high":
        value = np.clip(np.max(state.foot_pos()[:, :, 2], axis=1) - 0.24, 0.0, None)
        return np.where(moving, value, 0.0).astype(np.float32)
    if name == "feet_pos_xy":
        return np.mean(
            np.linalg.norm(state.foot_pos()[:, :, :2] - state.thigh_pos()[:, :, :2], axis=2), axis=1
        ).astype(np.float32)
    if name == "feet_drag":
        return np.sum(
            np.linalg.norm(state.foot_force_vec(), axis=2)
            * np.sum(np.abs(state.foot_vel()), axis=2),
            axis=1,
        ).astype(np.float32)
    if name == "feet_contact_forces":
        return np.sum(
            np.clip(np.linalg.norm(state.foot_force_vec(), axis=2) - 200.0, 0.0, None), axis=1
        ).astype(np.float32)
    raise KeyError(f"Unknown A2Arm reward term {name!r}")


def a2arm_termination(env) -> np.ndarray:
    _, _, robot = _parts(env)
    roll_pitch = _roll_pitch(robot.data.root_link_quat_w)
    return (np.abs(roll_pitch[:, 1]) > 1.0) | (np.abs(roll_pitch[:, 0]) > 0.8)


__all__ = ["a2arm_reward", "a2arm_termination"]
