"""Shared observation construction for motion tracking.

Holds the robot-agnostic observation builders. The environment classes keep a
thin polymorphic method surface (``_compute_obs`` / ``_build_actor_obs`` /
``_write_body_*``) that delegates here so subclasses can still override obs
layout (SAC critic tail, box object obs, deploy mimic actor) while the core
math lives in one place.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from unilab.dtype_config import get_global_dtype
from unilab.utils.geometry import np_write_relative_anchor_transform_pos_rot6d


def actor_obs_dim(n: int) -> int:
    return 3 + 6 + 3 + 3 + n * 5


def critic_base_obs_dim(n: int) -> int:
    return 3 + 6 + 3 + 3 + n * 5


def mimic_actor_obs_dim(n: int) -> int:
    # unitree_rl_lab mimic deploy actor input:
    # motion_command(2n), motion_anchor_ori_b(6), gyro(3), joints, actions.
    return 6 + 3 + n * 5


def obs_groups_spec(env: Any) -> dict[str, int]:
    # Actor: command(2n) + motion_anchor_pos_b(3) + motion_anchor_ori_b(6)
    #        + linvel(3) + gyro(3) + joint_pos(n) + joint_vel(n) + actions(n)
    # Critic mirrors BeyondMimic physical terms without actor observation noise:
    #        command, motion anchor, robot body pos/ori, linvel, gyro, joints, actions.
    n = env._num_action
    actor_width = getattr(env, "_actor_obs_width", env._actor_obs_dim(n))
    critic_width = getattr(
        env,
        "_critic_obs_width",
        env._critic_base_obs_dim(n) + len(env._cfg.body_names) * 9,
    )
    return {"obs": actor_width, "critic": critic_width}


def build_actor_obs(
    *,
    actor_obs_dim: int,
    command: np.ndarray,
    motion_anchor_pos_b: np.ndarray,
    motion_anchor_ori_b: np.ndarray,
    noisy_linvel: np.ndarray,
    noisy_gyro: np.ndarray,
    noisy_joint_pos_rel: np.ndarray,
    noisy_dof_vel: np.ndarray,
    last_actions: np.ndarray,
) -> np.ndarray:
    num_envs = command.shape[0]
    n_action = noisy_joint_pos_rel.shape[1]
    actor_obs = np.empty((num_envs, actor_obs_dim), dtype=get_global_dtype())
    offset = 0
    actor_obs[:, offset : offset + command.shape[1]] = command
    offset += command.shape[1]
    actor_obs[:, offset : offset + 3] = motion_anchor_pos_b
    offset += 3
    actor_obs[:, offset : offset + 6] = motion_anchor_ori_b
    offset += 6
    actor_obs[:, offset : offset + 3] = noisy_linvel
    offset += 3
    actor_obs[:, offset : offset + 3] = noisy_gyro
    offset += 3
    actor_obs[:, offset : offset + n_action] = noisy_joint_pos_rel
    offset += n_action
    actor_obs[:, offset : offset + n_action] = noisy_dof_vel
    offset += n_action
    actor_obs[:, offset : offset + n_action] = last_actions
    return actor_obs


def build_mimic_actor_obs(
    *,
    command: np.ndarray,
    motion_anchor_ori_b: np.ndarray,
    noisy_gyro: np.ndarray,
    noisy_joint_pos_rel: np.ndarray,
    noisy_dof_vel: np.ndarray,
    last_actions: np.ndarray,
) -> np.ndarray:
    """unitree_rl_lab mimic deploy actor layout: 2n + 6 + 3 + n + n + n."""
    return np.concatenate(
        [
            command,
            motion_anchor_ori_b,
            noisy_gyro,
            noisy_joint_pos_rel,
            noisy_dof_vel,
            last_actions,
        ],
        axis=1,
        dtype=get_global_dtype(),
    )


def write_body_pos_in_anchor_frame(
    anchor_pos: np.ndarray,
    anchor_quat: np.ndarray,
    body_pos: np.ndarray,
    out: np.ndarray,
    *,
    body_vec_error: np.ndarray,
) -> None:
    aw = anchor_quat[:, None, 0]
    ax = anchor_quat[:, None, 1]
    ay = anchor_quat[:, None, 2]
    az = anchor_quat[:, None, 3]

    num_envs, n_body = body_pos.shape[:2]
    rel_pos = body_vec_error[:num_envs, :n_body]

    vx = rel_pos[..., 0]
    vy = rel_pos[..., 1]
    vz = rel_pos[..., 2]
    np.subtract(body_pos[..., 0], anchor_pos[:, None, 0], out=vx)
    np.subtract(body_pos[..., 1], anchor_pos[:, None, 1], out=vy)
    np.subtract(body_pos[..., 2], anchor_pos[:, None, 2], out=vz)

    tx = 2 * (az * vy - ay * vz)
    ty = 2 * (ax * vz - az * vx)
    tz = 2 * (ay * vx - ax * vy)

    out[..., 0] = vx + aw * tx + az * ty - ay * tz
    out[..., 1] = vy + aw * ty + ax * tz - az * tx
    out[..., 2] = vz + aw * tz + ay * tx - ax * ty


def write_body_ori6_in_anchor_frame(
    anchor_quat: np.ndarray,
    body_quat: np.ndarray,
    out: np.ndarray,
) -> None:
    aw = anchor_quat[:, None, 0]
    ax = anchor_quat[:, None, 1]
    ay = anchor_quat[:, None, 2]
    az = anchor_quat[:, None, 3]
    bw = body_quat[..., 0]
    bx = body_quat[..., 1]
    by = body_quat[..., 2]
    bz = body_quat[..., 3]

    rw = aw * bw + ax * bx + ay * by + az * bz
    rx = aw * bx - ax * bw - ay * bz + az * by
    ry = aw * by + ax * bz - ay * bw - az * bx
    rz = aw * bz - ax * by + ay * bx - az * bw

    out[..., 0] = 1 - 2 * (ry * ry + rz * rz)
    out[..., 1] = 2 * (rx * ry - rw * rz)
    out[..., 2] = 2 * (rx * ry + rw * rz)
    out[..., 3] = 1 - 2 * (rx * rx + rz * rz)
    out[..., 4] = 2 * (rx * rz - rw * ry)
    out[..., 5] = 2 * (ry * rz + rw * rx)


def compute_obs(
    env: Any,
    info: dict,
    motion_data: Any,
    linvel: np.ndarray,
    gyro: np.ndarray,
    dof_pos: np.ndarray,
    dof_vel: np.ndarray,
    robot_body_pos_w: np.ndarray,
    robot_body_quat_w: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute observations as dict with actor and critic groups."""
    num_envs = linvel.shape[0]
    dtype = get_global_dtype()
    n_action = dof_pos.shape[1]
    n_body = env._n_motion_bodies

    # Get anchor states
    anchor_pos_w = motion_data.body_pos_w[:, env.anchor_body_idx]
    anchor_quat_w = motion_data.body_quat_w[:, env.anchor_body_idx]
    robot_anchor_pos_w = robot_body_pos_w[:, env.anchor_body_idx]
    robot_anchor_quat_w = robot_body_quat_w[:, env.anchor_body_idx]

    # Motion anchor pose in robot frame
    if num_envs == env._num_envs:
        motion_anchor_pos_b = env._motion_anchor_pos_b
        motion_anchor_ori_b = env._motion_anchor_ori_b
        joint_pos_rel = env._joint_pos_rel
        zero_actions = env._zero_actions
    else:
        motion_anchor_pos_b = np.empty((num_envs, 3), dtype=dtype)
        motion_anchor_ori_b = np.empty((num_envs, 6), dtype=dtype)
        joint_pos_rel = np.empty((num_envs, n_action), dtype=dtype)
        zero_actions = np.zeros((num_envs, n_action), dtype=dtype)
    np_write_relative_anchor_transform_pos_rot6d(
        robot_anchor_pos_w,
        robot_anchor_quat_w,
        anchor_pos_w,
        anchor_quat_w,
        motion_anchor_pos_b,
        motion_anchor_ori_b,
    )

    # Joint positions and velocities
    bias = info.get("default_dof_pos_bias")
    effective_default = env.default_angles + bias if bias is not None else env.default_angles
    np.subtract(dof_pos, effective_default, out=joint_pos_rel)
    last_actions = info.get("current_actions")
    if not isinstance(last_actions, np.ndarray):
        last_actions = zero_actions

    if num_envs == env._num_envs:
        command = env._motion_command
    else:
        command = np.empty((num_envs, n_action * 2), dtype=dtype)
    command[:, :n_action] = motion_data.joint_pos
    command[:, n_action : n_action * 2] = motion_data.joint_vel

    # Per-step observation noise on sensor channels (actor only).
    # Critic uses the clean originals — asymmetric actor–critic contract.
    noise_cfg = env._cfg.noise_config
    noise_enabled = noise_cfg.level > 0.0
    if noise_enabled:
        linvel_actor = env._obs_noise(linvel, noise_cfg.scale_linvel)
        gyro_actor = env._obs_noise(gyro, noise_cfg.scale_gyro)
        joint_pos_actor = env._obs_noise(joint_pos_rel, noise_cfg.scale_joint_angle)
        dof_vel_actor = env._obs_noise(dof_vel, noise_cfg.scale_joint_vel)
    else:
        linvel_actor = linvel
        gyro_actor = gyro
        joint_pos_actor = joint_pos_rel
        dof_vel_actor = dof_vel

    # Actor observations (noisy proprioception)
    actor_obs = env._build_actor_obs(
        command=command,
        motion_anchor_pos_b=motion_anchor_pos_b,
        motion_anchor_ori_b=motion_anchor_ori_b,
        noisy_linvel=linvel_actor,
        noisy_gyro=gyro_actor,
        noisy_joint_pos_rel=joint_pos_actor,
        noisy_dof_vel=dof_vel_actor,
        last_actions=last_actions,
    )

    # Critic observations (clean proprioception + privileged body transforms)
    critic_obs = np.empty((num_envs, env._critic_obs_width), dtype=dtype)
    offset = 0
    critic_obs[:, offset : offset + command.shape[1]] = command
    offset += command.shape[1]
    critic_obs[:, offset : offset + 3] = motion_anchor_pos_b
    offset += 3
    critic_obs[:, offset : offset + 6] = motion_anchor_ori_b
    offset += 6
    critic_obs[:, offset : offset + 3] = linvel
    offset += 3
    critic_obs[:, offset : offset + 3] = gyro
    offset += 3
    critic_obs[:, offset : offset + n_action] = joint_pos_rel
    offset += n_action
    critic_obs[:, offset : offset + n_action] = dof_vel
    offset += n_action
    critic_obs[:, offset : offset + n_action] = last_actions
    offset += n_action
    robot_body_pos_b = critic_obs[:, offset : offset + n_body * 3].reshape(num_envs, n_body, 3)
    env._write_body_pos_in_anchor_frame(
        robot_anchor_pos_w, robot_anchor_quat_w, robot_body_pos_w, robot_body_pos_b
    )
    offset += n_body * 3
    robot_body_ori_b = critic_obs[:, offset : offset + n_body * 6].reshape(num_envs, n_body, 6)
    env._write_body_ori6_in_anchor_frame(robot_anchor_quat_w, robot_body_quat_w, robot_body_ori_b)
    return {"obs": actor_obs, "critic": critic_obs}
