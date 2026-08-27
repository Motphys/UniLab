"""A2Arm reward and termination terms owned by the task manager boundary."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from unilab.tasks.locomotion.a2arm.actions import A2ArmPdAction
from unilab.tasks.locomotion.a2arm.state import A2ArmPosForceState
from unilab.utils.rotation import np_quat_apply, np_quat_apply_inverse, np_yaw_quat

from .constants import NUM_LEG
from .observations import _roll_pitch


@dataclass(frozen=True)
class _RewardContext:
    """Read-only snapshot for one reward-term evaluation."""

    state: A2ArmPosForceState
    action: A2ArmPdAction
    robot: Any
    q: np.ndarray
    qd: np.ndarray
    vel: np.ndarray
    ang: np.ndarray
    command: np.ndarray
    moving: np.ndarray


def _parts(env: Any) -> _RewardContext:
    state: A2ArmPosForceState = env.command_manager.get_term("task_state")
    action: A2ArmPdAction = env.action_manager.get_term("joint_pd")
    robot = env.scene["robot"]
    command = state.command
    moving = (
        (np.abs(command[:, 0]) > state.cfg.velocity_clip[0])
        | (np.abs(command[:, 1]) > state.cfg.velocity_clip[1])
        | (np.abs(command[:, 2]) > state.cfg.velocity_clip[2])
    )
    return _RewardContext(
        state=state,
        action=action,
        robot=robot,
        q=robot.data.joint_pos,
        qd=robot.data.joint_vel,
        vel=robot.data.root_link_lin_vel_b,
        ang=robot.data.root_link_ang_vel_b,
        command=command,
        moving=moving,
    )


def _tracking_lin_vel_force_world(ctx: _RewardContext, params: dict[str, Any]) -> np.ndarray:
    force = np_quat_apply_inverse(
        np_yaw_quat(ctx.robot.data.root_link_quat_w), ctx.state.force_base_world
    )
    target = (
        ctx.command[:, :2]
        + (force[:, :2] + ctx.state.force_base_command[:, :2]) / ctx.state.cfg.base_force_kd
    )
    target_moving = (
        (np.abs(target[:, 0]) > ctx.state.cfg.velocity_clip[0])
        | (np.abs(target[:, 1]) > ctx.state.cfg.velocity_clip[1])
        | (np.abs(ctx.command[:, 2]) > ctx.state.cfg.velocity_clip[2])
    )
    target *= target_moving[:, None]
    sigma = float(params.get("sigma", 0.25))
    return np.exp(-np.sum((target - ctx.vel[:, :2]) ** 2, axis=1) / sigma)


def _tracking_ee_force_world(ctx: _RewardContext, params: dict[str, Any]) -> np.ndarray:
    yaw = np_yaw_quat(ctx.robot.data.root_link_quat_w)
    goal = (
        ctx.state.current_goal_world
        + (ctx.state.force_ee_world + np_quat_apply(yaw, ctx.state.force_ee_command))
        / ctx.state.cfg.gripper_force_kp
    )
    error = np.sum(np.abs(ctx.state.ee_world_pos() - goal), axis=1)
    sigma = float(params.get("sigma", 1.0))
    return np.exp(-error / sigma * 2.0)


def _tracking_ang_vel(ctx: _RewardContext, params: dict[str, Any]) -> np.ndarray:
    sigma = float(params.get("sigma", 0.25))
    return np.exp(-((ctx.command[:, 2] - ctx.ang[:, 2]) ** 2) / sigma)


def _orientation(ctx: _RewardContext, params: dict[str, Any]) -> np.ndarray:
    del params
    gravity = ctx.robot.data.projected_gravity_b
    return gravity[:, 0] ** 2 + gravity[:, 1] ** 2


def _lin_vel_z(ctx: _RewardContext, params: dict[str, Any]) -> np.ndarray:
    del params
    return ctx.vel[:, 2] ** 2


def _ang_vel_xy(ctx: _RewardContext, params: dict[str, Any]) -> np.ndarray:
    del params
    return np.sum(ctx.ang[:, :2] ** 2, axis=1)


def _alive(ctx: _RewardContext, params: dict[str, Any]) -> np.ndarray:
    del params
    return np.ones(ctx.command.shape[0], dtype=np.float32)


def _ref_dof_leg(ctx: _RewardContext, params: dict[str, Any]) -> np.ndarray:
    scale = float(params.get("scale", 0.1))
    return np.exp(-np.sum(np.abs(ctx.q[:, :NUM_LEG] - ctx.state.reference_dof_pos), axis=1) * scale)


def _action_rate(ctx: _RewardContext, params: dict[str, Any]) -> np.ndarray:
    del params
    return np.sum(
        (ctx.action.raw_action[:, :NUM_LEG] - ctx.action.previous_raw_action[:, :NUM_LEG]) ** 2,
        axis=1,
    )


def _action_rate_arm(ctx: _RewardContext, params: dict[str, Any]) -> np.ndarray:
    del params
    return np.sum(
        (ctx.action.raw_action[:, NUM_LEG:] - ctx.action.previous_raw_action[:, NUM_LEG:]) ** 2,
        axis=1,
    )


def _torques(ctx: _RewardContext, params: dict[str, Any]) -> np.ndarray:
    del params
    return np.sum(ctx.action.applied_torque[:, :NUM_LEG] ** 2, axis=1)


def _dof_vel(ctx: _RewardContext, params: dict[str, Any]) -> np.ndarray:
    del params
    return np.sum(ctx.qd[:, :NUM_LEG] ** 2, axis=1)


def _dof_vel_arm(ctx: _RewardContext, params: dict[str, Any]) -> np.ndarray:
    del params
    return np.sum(ctx.qd[:, NUM_LEG:] ** 2, axis=1)


def _dof_acc(ctx: _RewardContext, params: dict[str, Any]) -> np.ndarray:
    dt = float(params.get("ctrl_dt", ctx.state.cfg.ctrl_dt))
    return np.sum(((ctx.state.last_dof_vel[:, :NUM_LEG] - ctx.qd[:, :NUM_LEG]) / dt) ** 2, axis=1)


def _dof_acc_arm(ctx: _RewardContext, params: dict[str, Any]) -> np.ndarray:
    dt = float(params.get("ctrl_dt", ctx.state.cfg.ctrl_dt))
    return np.sum(((ctx.state.last_dof_vel[:, NUM_LEG:] - ctx.qd[:, NUM_LEG:]) / dt) ** 2, axis=1)


def _base_height(ctx: _RewardContext, params: dict[str, Any]) -> np.ndarray:
    target = float(params.get("target", 0.435))
    return (ctx.robot.data.root_link_pos_w[:, 2] - target) ** 2


def _hip_pos(ctx: _RewardContext, params: dict[str, Any]) -> np.ndarray:
    del params
    hip_ids = [0, 3, 6, 9]
    return np.sum(
        (ctx.q[:, hip_ids] - ctx.robot.data.default_joint_pos[:, hip_ids]) ** 2,
        axis=1,
    )


def _torque_limits(ctx: _RewardContext, params: dict[str, Any]) -> np.ndarray:
    soft_limit = float(params.get("soft_limit", 0.9))
    return np.sum(
        np.clip(
            np.abs(ctx.action.applied_torque) - soft_limit * ctx.action.torque_limits, 0.0, None
        ),
        axis=1,
    )


def _dof_pos_limits(ctx: _RewardContext, params: dict[str, Any]) -> np.ndarray:
    del params
    limits = ctx.state.soft_dof_pos_limits
    return np.sum(
        np.clip(limits[:, 0] - ctx.q, 0.0, None) + np.clip(ctx.q - limits[:, 1], 0.0, None),
        axis=1,
    )


def _stand_still(ctx: _RewardContext, params: dict[str, Any]) -> np.ndarray:
    scale = float(params.get("scale", 0.05))
    return np.exp(
        -np.sum(np.abs(ctx.q[:, :NUM_LEG] - ctx.robot.data.default_joint_pos[:, :NUM_LEG]), axis=1)
        * scale
    ) * (~ctx.moving)


def _collision(ctx: _RewardContext, params: dict[str, Any]) -> np.ndarray:
    del params
    return np.sum(ctx.state.undesired_contacts() > 0.5, axis=1).astype(np.float32)


def _feet_contact_number(ctx: _RewardContext, params: dict[str, Any]) -> np.ndarray:
    del params
    return np.mean(
        np.where(ctx.state.foot_contact == (ctx.state.stance_mask > 0.5), 1.0, -0.3), axis=1
    )


def _feet_air_time(ctx: _RewardContext, params: dict[str, Any]) -> np.ndarray:
    threshold = float(params.get("threshold", 0.5))
    return (
        np.sum((ctx.state.air_time_snapshot - threshold) * ctx.state.first_contact, axis=1)
        * ctx.moving
    )


def _feet_height(ctx: _RewardContext, params: dict[str, Any]) -> np.ndarray:
    target = float(params.get("target", 0.12))
    value = np.clip(np.max(ctx.state.foot_pos()[:, :2, 2], axis=1) - target, None, 0.0)
    return np.where(ctx.moving, value, 0.0).astype(np.float32)


def _feet_height_high(ctx: _RewardContext, params: dict[str, Any]) -> np.ndarray:
    target = float(params.get("target", 0.24))
    value = np.clip(np.max(ctx.state.foot_pos()[:, :, 2], axis=1) - target, 0.0, None)
    return np.where(ctx.moving, value, 0.0).astype(np.float32)


def _feet_pos_xy(ctx: _RewardContext, params: dict[str, Any]) -> np.ndarray:
    del params
    return np.mean(
        np.linalg.norm(ctx.state.foot_pos()[:, :, :2] - ctx.state.thigh_pos()[:, :, :2], axis=2),
        axis=1,
    ).astype(np.float32)


def _feet_drag(ctx: _RewardContext, params: dict[str, Any]) -> np.ndarray:
    del params
    return np.sum(
        np.linalg.norm(ctx.state.foot_force_vec(), axis=2)
        * np.sum(np.abs(ctx.state.foot_vel()), axis=2),
        axis=1,
    ).astype(np.float32)


def _feet_contact_forces(ctx: _RewardContext, params: dict[str, Any]) -> np.ndarray:
    threshold = float(params.get("threshold", 200.0))
    return np.sum(
        np.clip(np.linalg.norm(ctx.state.foot_force_vec(), axis=2) - threshold, 0.0, None),
        axis=1,
    ).astype(np.float32)


_REWARD_TERMS: dict[str, Callable[[_RewardContext, dict[str, Any]], np.ndarray]] = {
    "tracking_lin_vel_force_world": _tracking_lin_vel_force_world,
    "tracking_ee_force_world": _tracking_ee_force_world,
    "tracking_ang_vel": _tracking_ang_vel,
    "orientation": _orientation,
    "lin_vel_z": _lin_vel_z,
    "ang_vel_xy": _ang_vel_xy,
    "alive": _alive,
    "ref_dof_leg": _ref_dof_leg,
    "action_rate": _action_rate,
    "action_rate_arm": _action_rate_arm,
    "torques": _torques,
    "dof_vel": _dof_vel,
    "dof_vel_arm": _dof_vel_arm,
    "dof_acc": _dof_acc,
    "dof_acc_arm": _dof_acc_arm,
    "base_height": _base_height,
    "hip_pos": _hip_pos,
    "torque_limits": _torque_limits,
    "dof_pos_limits": _dof_pos_limits,
    "stand_still": _stand_still,
    "collision": _collision,
    "feet_contact_number": _feet_contact_number,
    "feet_air_time": _feet_air_time,
    "feet_height": _feet_height,
    "feet_height_high": _feet_height_high,
    "feet_pos_xy": _feet_pos_xy,
    "feet_drag": _feet_drag,
    "feet_contact_forces": _feet_contact_forces,
}


def a2arm_reward(env: Any, name: str, **params: Any) -> np.ndarray:
    """Evaluate one registered A2Arm reward term without a branch dispatcher."""
    try:
        term = _REWARD_TERMS[name]
    except KeyError as exc:
        raise KeyError(f"Unknown A2Arm reward term {name!r}") from exc
    return term(_parts(env), params)


def a2arm_termination(env: Any) -> np.ndarray:
    robot = env.scene["robot"]
    roll_pitch = _roll_pitch(robot.data.root_link_quat_w)
    return (np.abs(roll_pitch[:, 1]) > 1.0) | (np.abs(roll_pitch[:, 0]) > 0.8)


__all__ = ["a2arm_reward", "a2arm_termination"]
