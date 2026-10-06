"""Device-resident mimic-lite observation terms for motion tracking."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from unilab.tasks.motion_tracking.common.manager_terms import (
    MotionJointPositionAction,
    TensorMotionCommand,
)
from unilab.tasks.motion_tracking.common.tensor_rotation import (
    quat_apply,
    quat_apply_inverse,
    quat_conjugate,
    quat_mul,
    quat_to_rot6,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from unilab.managers._types import ManagerBasedRlEnv
    from unilab.tasks.motion_tracking.common.manager_terms import TensorMotionObsFuture


def _command(env: ManagerBasedRlEnv, command_name: str) -> TensorMotionCommand:
    try:
        command = env.command_manager.get_term(command_name)
    except KeyError as exc:
        raise KeyError(f"Motion command term '{command_name}' not found") from exc
    if not isinstance(command, TensorMotionCommand):
        raise TypeError(
            f"Command term '{command_name}' is {type(command).__name__}, expected "
            "TensorMotionCommand"
        )
    return command


def _future_aux(command: TensorMotionCommand, future: TensorMotionObsFuture, key: str, compute):
    memo = getattr(command, "_obs_terms_aux_memo", None)
    if memo is None or len(memo) >= 8:
        memo = {}
        command._obs_terms_aux_memo = memo
    entry = memo.get(id(future))
    if entry is None or entry[0] is not future:
        entry = (future, {})
        memo[id(future)] = entry
    values: dict = entry[1]
    if key not in values:
        values[key] = compute()
    return values[key]


def _projected_yaw_quat(quat: torch.Tensor, x_axis_xy_threshold: float = 0.1) -> torch.Tensor:
    if quat.shape[-1] != 4:
        raise ValueError(f"Expected quaternion last dimension 4, got {tuple(quat.shape)}")
    basis_x = torch.zeros((*quat.shape[:-1], 3), dtype=quat.dtype, device=quat.device)
    basis_x[..., 0] = 1.0
    basis_z = torch.zeros_like(basis_x)
    basis_z[..., 2] = 1.0
    x_axis_w = quat_apply(quat, basis_x)
    z_axis_w = quat_apply(quat, basis_z)
    x_axis_xy = x_axis_w[..., :2]
    norm = torch.linalg.vector_norm(x_axis_xy, dim=-1, keepdim=True)
    heading_xy = torch.where(
        norm > x_axis_xy_threshold,
        x_axis_xy,
        torch.where(x_axis_w[..., 2:3] < 0.0, z_axis_w[..., :2], -z_axis_w[..., :2]),
    )
    yaw = torch.atan2(heading_xy[..., 1], heading_xy[..., 0])
    half_yaw = 0.5 * yaw
    return torch.stack(
        (
            torch.cos(half_yaw),
            torch.zeros_like(half_yaw),
            torch.zeros_like(half_yaw),
            torch.sin(half_yaw),
        ),
        dim=-1,
    )


def _z0ed(value: torch.Tensor) -> torch.Tensor:
    result = value.clone()
    result[..., 2] = 0.0
    return result


def _diff_body_frames(
    command: TensorMotionCommand, future: TensorMotionObsFuture
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if 0 not in future.future_steps:
        raise ValueError("diff body observations require step 0 in future_steps")
    frames = _future_aux(
        command,
        future,
        "diff_body_frames",
        lambda: _compute_diff_body_frames(command, future),
    )
    return cast(tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], frames)


def _compute_diff_body_frames(
    command: TensorMotionCommand, future: TensorMotionObsFuture
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the shared projected-yaw local frames on the device."""
    idx0 = future.future_steps.index(0)
    root_idx = command.obs_root_body_idx
    ref_anchor_pos_w = _z0ed(future.ref_body_pos_w[:, idx0, root_idx])
    ref_anchor_yaw = _projected_yaw_quat(future.ref_body_quat_w[:, idx0, root_idx])
    robot_anchor_pos_w = _z0ed(command.obs_robot_root_pos_w)
    robot_anchor_yaw = _future_aux(
        command,
        future,
        "robot_anchor_yaw",
        lambda: _projected_yaw_quat(command.obs_robot_root_quat_w),
    )
    ref_pos_local = quat_apply_inverse(
        ref_anchor_yaw[:, None, None, :], future.ref_body_pos_w - ref_anchor_pos_w[:, None, None, :]
    )
    ref_quat_local = quat_mul(
        quat_conjugate(ref_anchor_yaw[:, None, None, :]), future.ref_body_quat_w
    )
    robot_pos_local = quat_apply_inverse(
        robot_anchor_yaw[:, None, :],
        command.obs_robot_body_pos_w - robot_anchor_pos_w[:, None, :],
    )
    robot_quat_local = quat_mul(
        quat_conjugate(robot_anchor_yaw[:, None, :]), command.obs_robot_body_quat_w
    )
    return ref_pos_local, ref_quat_local, robot_pos_local, robot_quat_local


def ref_root_pos_future_local(
    env: ManagerBasedRlEnv, command_name: str, future_steps: Iterable[int]
) -> torch.Tensor:
    command = _command(env, command_name)
    future = command.obs_future(future_steps)
    ref_root_pos_w = future.ref_body_pos_w[:, :, command.obs_root_body_idx]
    robot_yaw = _future_aux(
        command,
        future,
        "robot_anchor_yaw",
        lambda: _projected_yaw_quat(command.obs_robot_root_quat_w),
    )
    local = quat_apply_inverse(
        robot_yaw[:, None, :], ref_root_pos_w - command.obs_robot_root_pos_w[:, None, :]
    )
    return local.reshape(env.num_envs, -1)


def ref_root_ori_future_b(
    env: ManagerBasedRlEnv, command_name: str, future_steps: Iterable[int]
) -> torch.Tensor:
    command = _command(env, command_name)
    future = command.obs_future(future_steps)
    ref_root_quat_w = future.ref_body_quat_w[:, :, command.obs_root_body_idx]
    rel_quat = quat_mul(quat_conjugate(command.obs_robot_root_quat_w[:, None, :]), ref_root_quat_w)
    return quat_to_rot6(rel_quat).reshape(env.num_envs, -1)


def ref_joint_pos_future(
    env: ManagerBasedRlEnv, command_name: str, future_steps: Iterable[int]
) -> torch.Tensor:
    return (
        _command(env, command_name).obs_future(future_steps).ref_joint_pos.reshape(env.num_envs, -1)
    )


def ref_root_pos_future_b(
    env: ManagerBasedRlEnv, command_name: str, future_steps: Iterable[int]
) -> torch.Tensor:
    command = _command(env, command_name)
    future = command.obs_future(future_steps)
    ref_root_pos_w = future.ref_body_pos_w[:, :, command.obs_root_body_idx]
    local = quat_apply_inverse(
        command.obs_robot_root_quat_w[:, None, :],
        ref_root_pos_w - command.obs_robot_root_pos_w[:, None, :],
    )
    return local.reshape(env.num_envs, -1)


def diff_body_pos_future_local(
    env: ManagerBasedRlEnv, command_name: str, future_steps: Iterable[int]
) -> torch.Tensor:
    command = _command(env, command_name)
    future = command.obs_future(future_steps)
    ref_pos_local, _, robot_pos_local, _ = _diff_body_frames(command, future)
    return (ref_pos_local - robot_pos_local[:, None, :, :]).reshape(env.num_envs, -1)


def diff_body_ori_future_local(
    env: ManagerBasedRlEnv, command_name: str, future_steps: Iterable[int]
) -> torch.Tensor:
    command = _command(env, command_name)
    future = command.obs_future(future_steps)
    _, ref_quat_local, _, robot_quat_local = _diff_body_frames(command, future)
    diff = quat_mul(quat_conjugate(robot_quat_local[:, None, :, :]), ref_quat_local)
    return quat_to_rot6(diff).reshape(env.num_envs, -1)


def diff_body_lin_vel_future(
    env: ManagerBasedRlEnv, command_name: str, future_steps: Iterable[int]
) -> torch.Tensor:
    command = _command(env, command_name)
    future = command.obs_future(future_steps)
    return (future.ref_body_lin_vel_w - command.obs_robot_body_lin_vel_w[:, None, :, :]).reshape(
        env.num_envs, -1
    )


def diff_body_ang_vel_future(
    env: ManagerBasedRlEnv, command_name: str, future_steps: Iterable[int]
) -> torch.Tensor:
    command = _command(env, command_name)
    future = command.obs_future(future_steps)
    return (future.ref_body_ang_vel_w - command.obs_robot_body_ang_vel_w[:, None, :, :]).reshape(
        env.num_envs, -1
    )


def motion_applied_action(env: ManagerBasedRlEnv, action_name: str = "joint_pos") -> torch.Tensor:
    try:
        term = env.action_manager.get_term(action_name)
    except KeyError as exc:
        raise KeyError(f"Action term '{action_name}' not found") from exc
    if not isinstance(term, MotionJointPositionAction):
        raise TypeError(
            f"Action term '{action_name}' is {type(term).__name__}, "
            "expected MotionJointPositionAction"
        )
    if isinstance(term.tensor_target, torch.Tensor):
        return term.tensor_target
    return torch.as_tensor(term.target, dtype=torch.float32, device=env.device)


def motion_joint_pos(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    command = _command(env, command_name)
    robot_joint_pos = getattr(command, "device_robot_joint_pos", command.robot_joint_pos)
    if not isinstance(robot_joint_pos, torch.Tensor):
        robot_joint_pos = torch.as_tensor(robot_joint_pos, dtype=torch.float32, device=env.device)
    return robot_joint_pos - cast(torch.Tensor, command.joint_default_bias)


def motion_joint_vel(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    command = _command(env, command_name)
    robot_joint_vel = getattr(command, "device_robot_joint_vel", command.robot_joint_vel)
    if not isinstance(robot_joint_vel, torch.Tensor):
        robot_joint_vel = torch.as_tensor(robot_joint_vel, dtype=torch.float32, device=env.device)
    return robot_joint_vel


__all__ = [
    "diff_body_ang_vel_future",
    "diff_body_lin_vel_future",
    "diff_body_ori_future_local",
    "diff_body_pos_future_local",
    "motion_applied_action",
    "motion_joint_pos",
    "motion_joint_vel",
    "ref_joint_pos_future",
    "ref_root_ori_future_b",
    "ref_root_pos_future_b",
    "ref_root_pos_future_local",
]
