"""Reward configuration and reward functions for motion tracking.

Reward terms are plain module-level callables ``fn(ctx: RewardContext) -> np.ndarray``
mirroring :mod:`unilab.envs.locomotion.common.rewards`. Robot-specific terms that
live on env subclasses (box-object / joint-effort terms) are stored in the same
``_reward_fns`` dispatch table as bound methods and are called with the same
``ctx`` argument.

``RewardContext`` intentionally carries the environment's *preallocated scratch
buffers* (``body_vec_error``, ``joint_error``, ... ``undesired_contact_mask``).
These buffers are env-owned so the hot path and the optional numba kernel run
with zero per-step allocations; the term functions write into them in place. The
op order, constants, clip bounds, and in-place ``out=`` usage are load-bearing —
they must stay bit-identical to preserve numeric parity with the numba path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import numpy as np


@dataclass
class RewardConfig:
    """Reward configuration for motion tracking."""

    scales: dict[str, float] = field(
        default_factory=lambda: {
            "motion_global_root_pos": 0.5,
            "motion_global_root_ori": 0.5,
            "motion_body_pos": 1.0,
            "motion_body_ori": 1.0,
            "motion_body_lin_vel": 1.0,
            "motion_body_ang_vel": 1.0,
            "motion_ee_body_pos_z": 0.0,
            "motion_joint_pos": 0.0,
            "motion_joint_vel": 0.0,
            "action_rate_l2": -0.1,
            "joint_limit": -10.0,
        }
    )
    # Standard deviations for exponential rewards
    std_root_pos: float = 0.3
    std_root_ori: float = 0.4
    std_body_pos: float = 0.3
    std_body_ori: float = 0.4
    std_body_lin_vel: float = 1.0
    std_body_ang_vel: float = 3.14
    std_joint_pos: float = 0.2
    std_joint_vel: float = 1.0


@dataclass
class RewardContext:
    """Bundle of everything the motion-tracking reward functions may read.

    Built once per ``_compute_reward`` call. The array fields prefixed as
    buffers (``env_error``, ``reward_term``, ``weighted_reward``, ...) are the
    environment's preallocated scratch arrays; reward terms write into them in
    place to keep the hot path allocation-free and numba-parity exact.
    """

    # ── semantic inputs ──────────────────────────────────────────────
    info: dict
    motion_data: Any = None
    robot_body_pos_w: np.ndarray | None = None
    robot_body_quat_w: np.ndarray | None = None
    robot_body_lin_vel_w: np.ndarray | None = None
    robot_body_ang_vel_w: np.ndarray | None = None
    ref_body_pos_w: np.ndarray | None = None  # env.body_pos_relative_w
    ref_body_quat_w: np.ndarray | None = None  # env.body_quat_relative_w
    dof_pos: np.ndarray | None = None
    dof_vel: np.ndarray | None = None

    # ── config-derived scalars / indices ────────────────────────────
    reward_config: Any = None
    anchor_body_idx: int = 0
    ee_body_indices: np.ndarray | None = None
    undesired_contact_body_indices: np.ndarray | None = None
    joint_lower: np.ndarray | None = None
    joint_upper: np.ndarray | None = None
    undesired_contact_z_threshold: float = 0.0
    num_envs: int = 0

    # ── env-owned scratch buffers (zero-alloc hot path) ──────────────
    body_vec_error: np.ndarray | None = None
    joint_error: np.ndarray | None = None
    joint_error_upper: np.ndarray | None = None
    env_error: np.ndarray | None = None
    env_error2: np.ndarray | None = None
    reward_term: np.ndarray | None = None
    weighted_reward: np.ndarray | None = None
    quat_error_w: np.ndarray | None = None
    quat_error_x: np.ndarray | None = None
    ee_pos_error_z: np.ndarray | None = None
    undesired_contact_mask: np.ndarray | None = None


# ── buffered math helpers ────────────────────────────────────────────


def _mean_body_xyz_squared_error(
    ctx: RewardContext, reference: np.ndarray, actual: np.ndarray
) -> np.ndarray:
    vec_error = ctx.body_vec_error
    env_error = ctx.env_error
    tmp_error = ctx.reward_term
    np.subtract(reference[..., 0], actual[..., 0], out=vec_error[..., 0])
    np.square(vec_error[..., 0], out=vec_error[..., 0])
    np.sum(vec_error[..., 0], axis=1, out=env_error)
    np.subtract(reference[..., 1], actual[..., 1], out=vec_error[..., 1])
    np.square(vec_error[..., 1], out=vec_error[..., 1])
    np.sum(vec_error[..., 1], axis=1, out=tmp_error)
    env_error += tmp_error
    np.subtract(reference[..., 2], actual[..., 2], out=vec_error[..., 2])
    np.square(vec_error[..., 2], out=vec_error[..., 2])
    np.sum(vec_error[..., 2], axis=1, out=tmp_error)
    env_error += tmp_error
    env_error /= reference.shape[1]
    return env_error


def _quat_error_magnitude_squared_body(
    ctx: RewardContext, q1: np.ndarray, q2: np.ndarray
) -> np.ndarray:
    rel_w = ctx.quat_error_w
    rel_x = ctx.quat_error_x
    # Motion/backend quaternions are unit quaternions, so the relative
    # rotation angle only needs abs(dot(q1, q2)).
    np.multiply(q1[..., 0], q2[..., 0], out=rel_w)
    np.multiply(q1[..., 1], q2[..., 1], out=rel_x)
    rel_w += rel_x
    np.multiply(q1[..., 2], q2[..., 2], out=rel_x)
    rel_w += rel_x
    np.multiply(q1[..., 3], q2[..., 3], out=rel_x)
    rel_w += rel_x
    np.abs(rel_w, out=rel_w)
    np.clip(rel_w, 0.0, 1.0, out=rel_w)
    np.arccos(rel_w, out=rel_x)
    rel_x *= 2.0
    np.square(rel_x, out=rel_x)
    return rel_x


def _exp_reward_from_error(ctx: RewardContext, error: np.ndarray, std: float) -> np.ndarray:
    out = ctx.reward_term
    np.divide(error, -(std**2), out=out)
    np.exp(out, out=out)
    return out


# ── reward terms ─────────────────────────────────────────────────────


def motion_global_root_pos(ctx: RewardContext) -> np.ndarray:
    motion_data = ctx.motion_data
    robot_body_pos_w = ctx.robot_body_pos_w
    anchor_pos_w = motion_data.body_pos_w[:, ctx.anchor_body_idx]
    robot_anchor_pos_w = robot_body_pos_w[:, ctx.anchor_body_idx]
    error = ctx.env_error
    np.subtract(anchor_pos_w[:, 0], robot_anchor_pos_w[:, 0], out=error)
    np.square(error, out=error)
    np.subtract(anchor_pos_w[:, 1], robot_anchor_pos_w[:, 1], out=ctx.reward_term)
    np.square(ctx.reward_term, out=ctx.reward_term)
    error += ctx.reward_term
    np.subtract(anchor_pos_w[:, 2], robot_anchor_pos_w[:, 2], out=ctx.reward_term)
    np.square(ctx.reward_term, out=ctx.reward_term)
    error += ctx.reward_term
    return _exp_reward_from_error(ctx, error, ctx.reward_config.std_root_pos)


def motion_global_root_ori(ctx: RewardContext) -> np.ndarray:
    motion_data = ctx.motion_data
    robot_body_quat_w = ctx.robot_body_quat_w
    anchor_quat_w = motion_data.body_quat_w[:, ctx.anchor_body_idx]
    robot_anchor_quat_w = robot_body_quat_w[:, ctx.anchor_body_idx]
    np.multiply(anchor_quat_w[:, 0], robot_anchor_quat_w[:, 0], out=ctx.env_error)
    np.multiply(anchor_quat_w[:, 1], robot_anchor_quat_w[:, 1], out=ctx.reward_term)
    ctx.env_error += ctx.reward_term
    np.multiply(anchor_quat_w[:, 2], robot_anchor_quat_w[:, 2], out=ctx.reward_term)
    ctx.env_error += ctx.reward_term
    np.multiply(anchor_quat_w[:, 3], robot_anchor_quat_w[:, 3], out=ctx.reward_term)
    ctx.env_error += ctx.reward_term
    np.abs(ctx.env_error, out=ctx.env_error)
    np.clip(ctx.env_error, 0.0, 1.0, out=ctx.env_error)
    np.arccos(ctx.env_error, out=ctx.env_error)
    ctx.env_error *= 2.0
    np.square(ctx.env_error, out=ctx.env_error)
    return _exp_reward_from_error(ctx, ctx.env_error, ctx.reward_config.std_root_ori)


def motion_body_pos(ctx: RewardContext) -> np.ndarray:
    robot_body_pos_w = ctx.robot_body_pos_w
    error = _mean_body_xyz_squared_error(ctx, ctx.ref_body_pos_w, robot_body_pos_w)
    return _exp_reward_from_error(ctx, error, ctx.reward_config.std_body_pos)


def motion_body_ori(ctx: RewardContext) -> np.ndarray:
    robot_body_quat_w = ctx.robot_body_quat_w
    error = _quat_error_magnitude_squared_body(ctx, ctx.ref_body_quat_w, robot_body_quat_w)
    np.sum(error, axis=-1, out=ctx.env_error)
    ctx.env_error /= error.shape[1]
    return _exp_reward_from_error(ctx, ctx.env_error, ctx.reward_config.std_body_ori)


def motion_body_lin_vel(ctx: RewardContext) -> np.ndarray:
    motion_data = ctx.motion_data
    robot_body_lin_vel_w = ctx.robot_body_lin_vel_w
    error = _mean_body_xyz_squared_error(ctx, motion_data.body_lin_vel_w, robot_body_lin_vel_w)
    return _exp_reward_from_error(ctx, error, ctx.reward_config.std_body_lin_vel)


def motion_body_ang_vel(ctx: RewardContext) -> np.ndarray:
    motion_data = ctx.motion_data
    robot_body_ang_vel_w = ctx.robot_body_ang_vel_w
    error = _mean_body_xyz_squared_error(ctx, motion_data.body_ang_vel_w, robot_body_ang_vel_w)
    return _exp_reward_from_error(ctx, error, ctx.reward_config.std_body_ang_vel)


def motion_ee_body_pos_z(ctx: RewardContext) -> np.ndarray:
    robot_body_pos_w = ctx.robot_body_pos_w
    np.subtract(
        ctx.ref_body_pos_w[:, ctx.ee_body_indices, 2],
        robot_body_pos_w[:, ctx.ee_body_indices, 2],
        out=ctx.ee_pos_error_z,
    )
    np.square(ctx.ee_pos_error_z, out=ctx.ee_pos_error_z)
    np.sum(ctx.ee_pos_error_z, axis=-1, out=ctx.env_error)
    ctx.env_error /= ctx.ee_pos_error_z.shape[1]
    return _exp_reward_from_error(ctx, ctx.env_error, ctx.reward_config.std_body_pos)


def motion_joint_pos(ctx: RewardContext) -> np.ndarray:
    motion_data = ctx.motion_data
    dof_pos = ctx.dof_pos
    np.subtract(motion_data.joint_pos, dof_pos, out=ctx.joint_error)
    np.square(ctx.joint_error, out=ctx.joint_error)
    np.sum(ctx.joint_error, axis=1, out=ctx.env_error)
    ctx.env_error /= dof_pos.shape[1]
    return _exp_reward_from_error(ctx, ctx.env_error, ctx.reward_config.std_joint_pos)


def motion_joint_vel(ctx: RewardContext) -> np.ndarray:
    motion_data = ctx.motion_data
    dof_vel = ctx.dof_vel
    np.subtract(motion_data.joint_vel, dof_vel, out=ctx.joint_error)
    np.square(ctx.joint_error, out=ctx.joint_error)
    np.sum(ctx.joint_error, axis=1, out=ctx.env_error)
    ctx.env_error /= dof_vel.shape[1]
    return _exp_reward_from_error(ctx, ctx.env_error, ctx.reward_config.std_joint_vel)


def undesired_contacts(ctx: RewardContext) -> np.ndarray:
    robot_body_pos_w = ctx.robot_body_pos_w
    body_z = robot_body_pos_w[:, ctx.undesired_contact_body_indices, 2]
    np.less(
        body_z,
        ctx.undesired_contact_z_threshold,
        out=ctx.undesired_contact_mask,
    )
    np.sum(ctx.undesired_contact_mask, axis=-1, out=ctx.env_error)
    return ctx.env_error


def action_rate_l2(ctx: RewardContext) -> np.ndarray:
    info = ctx.info
    np.subtract(info["current_actions"], info["last_actions"], out=ctx.joint_error)
    np.square(ctx.joint_error, out=ctx.joint_error)
    np.sum(ctx.joint_error, axis=1, out=ctx.env_error)
    return ctx.env_error


def joint_limit(ctx: RewardContext) -> np.ndarray:
    dof_pos = ctx.dof_pos
    lower = ctx.joint_lower
    upper = ctx.joint_upper
    if lower is None or upper is None:
        ctx.reward_term.fill(0.0)
        return ctx.reward_term

    # Compute violation
    np.subtract(lower, dof_pos, out=ctx.joint_error)
    np.maximum(ctx.joint_error, 0, out=ctx.joint_error)
    np.subtract(dof_pos, upper, out=ctx.joint_error_upper)
    np.maximum(ctx.joint_error_upper, 0, out=ctx.joint_error_upper)
    ctx.joint_error += ctx.joint_error_upper
    np.square(ctx.joint_error, out=ctx.joint_error)
    np.sum(ctx.joint_error, axis=1, out=ctx.reward_term)
    return ctx.reward_term


def build_reward_functions() -> dict[str, Callable[[RewardContext], np.ndarray]]:
    """Return the robot-agnostic reward-term dispatch table.

    Keys match :data:`unilab.envs.motion_tracking.common.numba.TERM_ORDER`.
    """
    return {
        "motion_global_root_pos": motion_global_root_pos,
        "motion_global_root_ori": motion_global_root_ori,
        "motion_body_pos": motion_body_pos,
        "motion_body_ori": motion_body_ori,
        "motion_body_lin_vel": motion_body_lin_vel,
        "motion_body_ang_vel": motion_body_ang_vel,
        "motion_ee_body_pos_z": motion_ee_body_pos_z,
        "motion_joint_pos": motion_joint_pos,
        "motion_joint_vel": motion_joint_vel,
        "action_rate_l2": action_rate_l2,
        "joint_limit": joint_limit,
        "undesired_contacts": undesired_contacts,
    }


def compute_reward(
    ctx: RewardContext,
    *,
    active_reward_fns: Mapping[str, Callable[[RewardContext], np.ndarray]],
    all_reward_fns: Mapping[str, Callable[[RewardContext], np.ndarray]],
    scales: Mapping[str, float],
    ctrl_dt: float,
    enable_log: bool,
) -> np.ndarray:
    """Reduce ``scales × fns(ctx)`` into the per-env reward (in place, zero-alloc).

    Uses ``ctx.env_error2`` as the reward accumulator and ``ctx.weighted_reward``
    as the per-term scratch, matching the numba kernel's op order. Logs per-term
    means into ``ctx.info["log"]`` every 4th step, then scales by ``ctrl_dt``.
    """
    reward = ctx.env_error2
    reward.fill(0.0)

    info = ctx.info
    step_count = info.get("steps")
    should_log = enable_log and (
        int(step_count[0]) % 4 == 0 if isinstance(step_count, np.ndarray) else True
    )
    log = {} if should_log else info.get("log", {})

    for name, scale in scales.items():
        if scale == 0:
            continue
        reward_fn = active_reward_fns.get(name)
        if reward_fn is None:
            if should_log and name in all_reward_fns:
                log[f"reward/{name}"] = 0.0
            continue
        rew = reward_fn(ctx)
        weighted_rew = ctx.weighted_reward
        np.multiply(rew, scale, out=weighted_rew)
        reward += weighted_rew
        if should_log:
            log[f"reward/{name}"] = float(np.sum(weighted_rew) / weighted_rew.size)

    info["log"] = log
    reward *= ctrl_dt
    return reward
