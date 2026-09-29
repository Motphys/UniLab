"""mimic-lite-style observation terms for motion tracking.

These terms mirror the observation semantics of mimic-lite
(``mimic_lite/tasks/observations/track.py`` and ``command.py``) on UniLab's
NumPy manager runtime: future-step reference windows gathered from the motion
command's observation body set, robot-frame root diffs, and body-local diffs
in projected-yaw anchor frames. Noise is configured on the observation term
(GaussianNoiseCfg with ``clamp=3.0`` matches mimic-lite's
``randn.clamp(-3, 3) * std``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from unilab.utils.rotation import (
    np_matrix_from_quat,
    np_quat_apply_batched,
    np_quat_apply_inverse_batched,
    np_quat_conjugate_batched,
    np_quat_mul_batched,
    np_yaw_to_quat,
)

from .manager_terms import MotionCommand, MotionJointPositionAction

if TYPE_CHECKING:
    from collections.abc import Iterable

    from unilab.managers._types import ManagerBasedRlEnv

    from .manager_terms import MotionObsFuture


def _command(env: ManagerBasedRlEnv, command_name: str) -> MotionCommand:
    try:
        command = env.command_manager.get_term(command_name)
    except KeyError as exc:
        raise KeyError(f"Motion command term '{command_name}' not found") from exc
    if not isinstance(command, MotionCommand):
        raise TypeError(
            f"Command term '{command_name}' is {type(command).__name__}, expected MotionCommand"
        )
    return command


def _projected_yaw_quat(quat: np.ndarray, x_axis_xy_threshold: float = 0.1) -> np.ndarray:
    """Build a level yaw quaternion from horizontal axis projections.

    NumPy port of mimic-lite's ``projected_yaw_quat``: the heading comes from
    the anchor x-axis xy-projection when that projection is significant, else
    from the z-axis xy-projection sign-adjusted by the x-axis vertical
    direction so the heading stays continuous when the x-axis crosses between
    pointing upward and downward.

    Args:
        quat: Orientation(s) in (w, x, y, z) order, shape (..., 4).
        x_axis_xy_threshold: Minimum horizontal norm for using the projected
            x-axis.

    Returns:
        Yaw-only quaternions, same shape as ``quat``.
    """
    shape = quat.shape
    flat = quat.reshape(-1, 4)
    basis_x = np.zeros((flat.shape[0], 3), dtype=quat.dtype)
    basis_x[:, 0] = 1.0
    basis_z = np.zeros((flat.shape[0], 3), dtype=quat.dtype)
    basis_z[:, 2] = 1.0

    x_axis_w = np_quat_apply_batched(flat, basis_x)
    z_axis_w = np_quat_apply_batched(flat, basis_z)

    x_axis_xy = x_axis_w[:, :2]
    x_axis_xy_norm = np.linalg.norm(x_axis_xy, axis=-1, keepdims=True)
    z_axis_heading_xy = np.where(x_axis_w[:, 2:3] < 0.0, z_axis_w[:, :2], -z_axis_w[:, :2])
    heading_xy = np.where(x_axis_xy_norm > x_axis_xy_threshold, x_axis_xy, z_axis_heading_xy)

    yaw = np.arctan2(heading_xy[:, 1], heading_xy[:, 0])
    return np_yaw_to_quat(yaw).reshape(shape)


def _matrix_first_two_rows(quat: np.ndarray) -> np.ndarray:
    """Flattened first two rotation-matrix rows for quaternions (..., 4).

    Matches mimic-lite's ``matrix_from_quat(q)[..., :2, :].reshape(..., 6)``.
    """
    if quat.shape[-1] != 4:
        raise ValueError(f"Expected quaternion last dimension 4, got {quat.shape}")
    lead_shape = quat.shape[:-1]
    mats = np_matrix_from_quat(quat.reshape(-1, 4))
    return mats[:, :2, :].reshape(*lead_shape, 6)


def _z0ed(arr: np.ndarray) -> np.ndarray:
    """Copy with the world-z component zeroed (projected-yaw anchor origin)."""
    out = arr.copy()
    out[..., 2] = 0.0
    return out


def _diff_body_frames(
    command: MotionCommand, future: MotionObsFuture
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Shared projected-yaw local frames for the body-diff observations.

    Ports mimic-lite's ``_compute_body_diff_obs``: reference bodies are
    expressed in the reference obs-root projected-yaw frame anchored at the
    current reference frame (step 0 of the gather), robot bodies in the robot
    obs-root projected-yaw frame; both anchors are z0-ed.

    Returns ``(ref_pos_local, ref_quat_local, robot_pos_local,
    robot_quat_local)`` with shapes (E, S, B, 3/4) for the references and
    (E, B, 3/4) for the robot.
    """
    if 0 not in future.future_steps:
        raise ValueError(
            f"diff body observations require step 0 in future_steps, got {future.future_steps}"
        )
    idx0 = future.future_steps.index(0)
    root_idx = command.obs_root_body_idx

    ref_anchor_pos_w = _z0ed(future.ref_body_pos_w[:, idx0, root_idx])
    ref_anchor_yaw_quat_w = _projected_yaw_quat(future.ref_body_quat_w[:, idx0, root_idx])
    robot_anchor_pos_w = _z0ed(command.obs_robot_root_pos_w)
    robot_anchor_yaw_quat_w = _projected_yaw_quat(command.obs_robot_root_quat_w)

    ref_pos_local = np_quat_apply_inverse_batched(
        ref_anchor_yaw_quat_w[:, None, None, :],
        future.ref_body_pos_w - ref_anchor_pos_w[:, None, None, :],
    )
    ref_quat_local = np_quat_mul_batched(
        np_quat_conjugate_batched(ref_anchor_yaw_quat_w[:, None, None, :]),
        future.ref_body_quat_w,
    )
    robot_pos_local = np_quat_apply_inverse_batched(
        robot_anchor_yaw_quat_w[:, None, :],
        command.obs_robot_body_pos_w - robot_anchor_pos_w[:, None, :],
    )
    robot_quat_local = np_quat_mul_batched(
        np_quat_conjugate_batched(robot_anchor_yaw_quat_w[:, None, :]),
        command.obs_robot_body_quat_w,
    )
    return ref_pos_local, ref_quat_local, robot_pos_local, robot_quat_local


##
# Actor reference observations (future-step windows).
##


def ref_root_pos_future_local(
    env: ManagerBasedRlEnv, command_name: str, future_steps: Iterable[int]
) -> np.ndarray:
    """Reference root positions in the robot root projected-yaw frame.

    ``R_yaw(robot_root)^T @ (ref_root_pos[t + s] - robot_root_pos[t])`` per
    future step ``s``; the robot root yaw uses the mimic-lite projected-yaw
    construction. Note mimic-lite expresses this term in the *reference*
    anchor projected-yaw frame instead; the robot-frame variant is the
    configured semantics here.
    """
    command = _command(env, command_name)
    future = command.obs_future(future_steps)
    ref_root_pos_w = future.ref_body_pos_w[:, :, command.obs_root_body_idx]
    robot_yaw_quat_w = _projected_yaw_quat(command.obs_robot_root_quat_w)
    local = np_quat_apply_inverse_batched(
        robot_yaw_quat_w[:, None, :],
        ref_root_pos_w - command.obs_robot_root_pos_w[:, None, :],
    )
    return local.reshape(env.num_envs, -1)


def ref_root_ori_future_b(
    env: ManagerBasedRlEnv, command_name: str, future_steps: Iterable[int]
) -> np.ndarray:
    """Reference root orientations relative to the robot root frame.

    ``conj(robot_root_quat) * ref_root_quat[t + s]`` per future step, encoded
    as the first two rotation-matrix rows (6D per step).
    """
    command = _command(env, command_name)
    future = command.obs_future(future_steps)
    ref_root_quat_w = future.ref_body_quat_w[:, :, command.obs_root_body_idx]
    rel_quat = np_quat_mul_batched(
        np_quat_conjugate_batched(command.obs_robot_root_quat_w[:, None, :]),
        ref_root_quat_w,
    )
    return _matrix_first_two_rows(rel_quat).reshape(env.num_envs, -1)


def ref_joint_pos_future(
    env: ManagerBasedRlEnv, command_name: str, future_steps: Iterable[int]
) -> np.ndarray:
    """Raw reference joint positions at each future step (no default offset)."""
    command = _command(env, command_name)
    future = command.obs_future(future_steps)
    return future.ref_joint_pos.reshape(env.num_envs, -1)


##
# Critic privileged observations.
##


def ref_root_pos_future_b(
    env: ManagerBasedRlEnv, command_name: str, future_steps: Iterable[int]
) -> np.ndarray:
    """Reference root positions in the full robot root frame (not yaw-only)."""
    command = _command(env, command_name)
    future = command.obs_future(future_steps)
    ref_root_pos_w = future.ref_body_pos_w[:, :, command.obs_root_body_idx]
    local = np_quat_apply_inverse_batched(
        command.obs_robot_root_quat_w[:, None, :],
        ref_root_pos_w - command.obs_robot_root_pos_w[:, None, :],
    )
    return local.reshape(env.num_envs, -1)


def diff_body_pos_future_local(
    env: ManagerBasedRlEnv, command_name: str, future_steps: Iterable[int]
) -> np.ndarray:
    """Reference minus robot body positions, each in its projected-yaw frame."""
    command = _command(env, command_name)
    future = command.obs_future(future_steps)
    ref_pos_local, _, robot_pos_local, _ = _diff_body_frames(command, future)
    diff = ref_pos_local - robot_pos_local[:, None, :, :]
    return diff.reshape(env.num_envs, -1)


def diff_body_ori_future_local(
    env: ManagerBasedRlEnv, command_name: str, future_steps: Iterable[int]
) -> np.ndarray:
    """``conj(robot_local) * ref_local`` per body/step, 6D matrix encoding."""
    command = _command(env, command_name)
    future = command.obs_future(future_steps)
    _, ref_quat_local, _, robot_quat_local = _diff_body_frames(command, future)
    diff_quat = np_quat_mul_batched(
        np_quat_conjugate_batched(robot_quat_local[:, None, :, :]),
        ref_quat_local,
    )
    return _matrix_first_two_rows(diff_quat).reshape(env.num_envs, -1)


def diff_body_lin_vel_future(
    env: ManagerBasedRlEnv, command_name: str, future_steps: Iterable[int]
) -> np.ndarray:
    """Reference minus robot body linear velocities, world frame."""
    command = _command(env, command_name)
    future = command.obs_future(future_steps)
    diff = future.ref_body_lin_vel_w - command.obs_robot_body_lin_vel_w[:, None, :, :]
    return diff.reshape(env.num_envs, -1)


def diff_body_ang_vel_future(
    env: ManagerBasedRlEnv, command_name: str, future_steps: Iterable[int]
) -> np.ndarray:
    """Reference minus robot body angular velocities, world frame."""
    command = _command(env, command_name)
    future = command.obs_future(future_steps)
    diff = future.ref_body_ang_vel_w - command.obs_robot_body_ang_vel_w[:, None, :, :]
    return diff.reshape(env.num_envs, -1)


def motion_applied_action(env: ManagerBasedRlEnv, action_name: str = "joint_pos") -> np.ndarray:
    """Most recently applied physical PD joint target (radians).

    mimic-lite exposes the dimensionless smoothed action here; this term
    deliberately exposes the physical target actually written to the
    simulator.
    """
    try:
        term = env.action_manager.get_term(action_name)
    except KeyError as exc:
        raise KeyError(f"Action term '{action_name}' not found") from exc
    if not isinstance(term, MotionJointPositionAction):
        raise TypeError(
            f"Action term '{action_name}' is {type(term).__name__}, "
            "expected MotionJointPositionAction"
        )
    return term.target


def motion_applied_torque(env: ManagerBasedRlEnv, command_name: str) -> np.ndarray:
    """Applied joint actuator torques (per-actuator ``actuatorfrc`` sensors)."""
    return _command(env, command_name).robot_joint_torque


##
# Proprioceptive observations aligned with mimic-lite's single-step values.
##


def motion_joint_pos(env: ManagerBasedRlEnv, command_name: str) -> np.ndarray:
    """Raw joint positions minus the per-episode action offset.

    Matches mimic-lite's ``joint_pos`` observation (``joint_pos -
    action_manager.offset``): no static default-joint subtraction, only the
    per-env episode offset that the action term adds back onto the target.
    """
    command = _command(env, command_name)
    return command.robot_joint_pos - command.joint_default_bias


def motion_joint_vel(env: ManagerBasedRlEnv, command_name: str) -> np.ndarray:
    """Raw joint velocities (mimic-lite applies no offset to velocities)."""
    return _command(env, command_name).robot_joint_vel


__all__ = [
    "diff_body_ang_vel_future",
    "diff_body_lin_vel_future",
    "diff_body_ori_future_local",
    "diff_body_pos_future_local",
    "motion_applied_action",
    "motion_applied_torque",
    "motion_joint_pos",
    "motion_joint_vel",
    "ref_joint_pos_future",
    "ref_root_ori_future_b",
    "ref_root_pos_future_b",
    "ref_root_pos_future_local",
]
