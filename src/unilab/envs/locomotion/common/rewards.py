"""Shared reward functions for locomotion environments.

Introduces ``RewardContext`` — a dataclass that bundles all state any
reward function might need.  Shared reward functions are plain
module-level callables ``fn(ctx) -> np.ndarray`` so that each
joystick environment can reference them **directly** in its
``_reward_fns`` dispatch table without per-class wrapper methods.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from unilab.dtype_config import get_global_dtype


@dataclass
class RewardContext:
    """Immutable snapshot of everything reward functions may read.

    Built once per ``_compute_reward`` call.  Shared functions access
    only the fields they need; robot-specific methods that still live
    on the environment class receive the same context via ``self``.
    """

    # ── always populated ────────────────────────────────────────────
    info: dict
    linvel: np.ndarray  # (N, 3)
    gyro: np.ndarray  # (N, 3)
    dof_pos: np.ndarray  # (N, num_action)
    num_envs: int = 0
    default_angles: np.ndarray = field(default_factory=lambda: np.empty(0))
    tracking_sigma: float = 0.25
    base_height_target: float = 0.0
    base_height: np.ndarray = field(default_factory=lambda: np.empty(0))  # pre-fetched

    # ── G1-only (None for quadrupeds) ───────────────────────────────
    gravity: np.ndarray | None = None
    dof_vel: np.ndarray | None = None

    # ── optional weights (G1 pose rewards) ──────────────────────────
    pose_weights: np.ndarray | None = None

    # ── optional state populated for rough / biped tasks ────────────
    joint_range: np.ndarray | None = None  # (num_action, 2) — [lower, upper]
    linvel_yaw: np.ndarray | None = None  # (N, 3) — base linvel in yaw frame


# ── tracking rewards ─────────────────────────────────────────────────


def tracking_lin_vel(ctx: RewardContext) -> np.ndarray:
    """Exponential reward for tracking commanded xy linear velocity."""
    commands = ctx.info["commands"]
    lin_vel_error = np.sum(np.square(commands[:, :2] - ctx.linvel[:, :2]), axis=1)
    return np.exp(-lin_vel_error / ctx.tracking_sigma)  # type: ignore[no-any-return]


def tracking_ang_vel(ctx: RewardContext) -> np.ndarray:
    """Exponential reward for tracking commanded yaw angular velocity."""
    commands = ctx.info["commands"]
    ang_vel_error = np.square(commands[:, 2] - ctx.gyro[:, 2])
    return np.exp(-ang_vel_error / ctx.tracking_sigma)  # type: ignore[no-any-return]


def forward_progress(ctx: RewardContext) -> np.ndarray:
    """Reward for forward progress relative to commanded speed."""
    commands = ctx.info["commands"]
    commanded_speed = np.maximum(commands[:, 0], 1e-6)
    forward_speed = np.maximum(ctx.linvel[:, 0], 0.0)
    return np.asarray(np.minimum(forward_speed / commanded_speed, 1.0), dtype=get_global_dtype())


def under_speed(ctx: RewardContext) -> np.ndarray:
    """Penalty for being below commanded forward speed."""
    commands = ctx.info["commands"]
    commanded_speed = np.maximum(commands[:, 0], 1e-6)
    forward_speed = np.maximum(ctx.linvel[:, 0], 0.0)
    gap = np.maximum(commands[:, 0] - forward_speed, 0.0)
    return np.asarray(gap / commanded_speed, dtype=get_global_dtype())


# ── velocity / orientation penalties ─────────────────────────────────


def lin_vel_z(ctx: RewardContext) -> np.ndarray:
    """Penalty for vertical (z) linear velocity."""
    return np.square(ctx.linvel[:, 2])  # type: ignore[no-any-return]


def ang_vel_xy(ctx: RewardContext) -> np.ndarray:
    """Penalty for roll/pitch angular velocity."""
    return np.sum(np.square(ctx.gyro[:, :2]), axis=1)  # type: ignore[no-any-return]


def orientation(ctx: RewardContext) -> np.ndarray:
    """Penalty for deviation from upright orientation (roll/pitch)."""
    g = ctx.gravity
    assert g is not None
    return np.square(g[:, 0]) + np.square(g[:, 1])  # type: ignore[no-any-return]


def roll(ctx: RewardContext) -> np.ndarray:
    """Penalty for deviation from roll orientation."""
    g = ctx.gravity
    assert g is not None
    return np.square(g[:, 0])  # type: ignore[no-any-return]


def upright(ctx: RewardContext) -> np.ndarray:
    """Exponential reward for upright orientation."""
    g = ctx.gravity
    assert g is not None
    xy_squared = np.sum(np.square(g[:, :2]), axis=1)
    return np.exp(-xy_squared / 0.25)  # type: ignore[no-any-return]


# ── height / pose penalties ──────────────────────────────────────────


def base_height(ctx: RewardContext) -> np.ndarray:
    """Penalty for base height deviation from target."""
    return np.square(ctx.base_height - ctx.base_height_target)  # type: ignore[no-any-return]


def similar_to_default(ctx: RewardContext) -> np.ndarray:
    """Penalty for joint position deviation from default (L1 norm)."""
    return np.sum(np.abs(ctx.dof_pos - ctx.default_angles), axis=1)  # type: ignore[no-any-return]


def weighted_pose(ctx: RewardContext) -> np.ndarray:
    """Weighted L2 penalty for joint position deviation."""
    assert ctx.pose_weights is not None
    diff = ctx.dof_pos - ctx.default_angles
    return np.asarray(np.sum(ctx.pose_weights * np.square(diff), axis=1), dtype=get_global_dtype())


# ── action penalties ─────────────────────────────────────────────────


def action_rate(ctx: RewardContext) -> np.ndarray:
    """Penalty for change in actions between timesteps."""
    current = ctx.info["current_actions"]
    last = ctx.info["last_actions"]
    return np.sum(np.square(current - last), axis=1)  # type: ignore[no-any-return]


# ── effort penalties ─────────────────────────────────────────────────


def _get_torques(ctx: RewardContext) -> np.ndarray:
    fallback = np.zeros((ctx.num_envs, ctx.dof_pos.shape[1]), dtype=get_global_dtype())
    return ctx.info.get("torques", fallback)  # type: ignore[no-any-return]


def torques(ctx: RewardContext) -> np.ndarray:
    """Penalty for total torque magnitude (L1 norm)."""
    return np.sum(np.abs(_get_torques(ctx)), axis=1)  # type: ignore[no-any-return]


def energy(ctx: RewardContext) -> np.ndarray:
    """Penalty for mechanical energy consumption."""
    assert ctx.dof_vel is not None
    t = _get_torques(ctx)
    return np.sum(np.abs(ctx.dof_vel) * np.abs(t), axis=1)  # type: ignore[no-any-return]


def dof_acc(ctx: RewardContext) -> np.ndarray:
    """Penalty for joint acceleration magnitude."""
    fallback = np.zeros((ctx.num_envs, ctx.dof_pos.shape[1]), dtype=get_global_dtype())
    qacc = ctx.info.get("qacc", fallback)
    return np.sum(np.square(qacc), axis=1)  # type: ignore[no-any-return]


# ── survival ─────────────────────────────────────────────────────────


def alive(ctx: RewardContext) -> np.ndarray:
    """Constant reward for staying alive."""
    return np.ones((ctx.num_envs,), dtype=get_global_dtype())


# ── quadruped-rough helpers / penalties ──────────────────────────────


def upright_scale(gravity: np.ndarray | None, num_envs: int) -> np.ndarray:
    """Scalar gate in [0, 1] from the body-up projection of gravity.

    Used by quadruped rough tasks to suppress reward / penalty bookkeeping
    while the robot is tipping over. Returns 1.0 when the body is upright
    (gravity[:, 2] >= 0.7) and 0.0 when fully tipped.
    """
    if gravity is None:
        return np.ones((num_envs,), dtype=get_global_dtype())
    return np.asarray(np.clip(gravity[:, 2], 0.0, 0.7) / 0.7, dtype=get_global_dtype())


def dof_torques_l2(ctx: RewardContext) -> np.ndarray:
    """Penalty for joint torque magnitude (L2)."""
    torques = np.asarray(
        ctx.info.get("torques", np.zeros((ctx.num_envs, ctx.dof_pos.shape[1]))),
        dtype=get_global_dtype(),
    )
    return np.asarray(np.sum(np.square(torques), axis=1), dtype=get_global_dtype())


def dof_acc_l2(ctx: RewardContext) -> np.ndarray:
    """Penalty for joint acceleration magnitude (L2)."""
    qacc = np.asarray(
        ctx.info.get("qacc", np.zeros((ctx.num_envs, ctx.dof_pos.shape[1]))),
        dtype=get_global_dtype(),
    )
    return np.asarray(np.sum(np.square(qacc), axis=1), dtype=get_global_dtype())


def joint_pos_limits(ctx: RewardContext) -> np.ndarray:
    """Penalty for joint position over/under-shoot relative to backend limits."""
    if ctx.joint_range is None:
        return np.zeros((ctx.num_envs,), dtype=get_global_dtype())
    lower = ctx.joint_range[:, 0]
    upper = ctx.joint_range[:, 1]
    low_error = np.clip(lower - ctx.dof_pos, 0.0, None)
    high_error = np.clip(ctx.dof_pos - upper, 0.0, None)
    return np.asarray(np.sum(low_error + high_error, axis=1), dtype=get_global_dtype())


def joint_power(ctx: RewardContext) -> np.ndarray:
    """Penalty for joint mechanical power (|tau * dq|)."""
    assert ctx.dof_vel is not None
    torques = np.asarray(
        ctx.info.get("torques", np.zeros((ctx.num_envs, ctx.dof_pos.shape[1]))),
        dtype=get_global_dtype(),
    )
    return np.asarray(np.sum(np.abs(ctx.dof_vel * torques), axis=1), dtype=get_global_dtype())


def stand_still(ctx: RewardContext, command_threshold: float = 0.1) -> np.ndarray:
    """Penalty for joint deviation from default while command norm is below threshold."""
    stopped = np.linalg.norm(ctx.info["commands"], axis=1) < command_threshold
    dof_error = np.sum(np.abs(ctx.dof_pos - ctx.default_angles), axis=1)
    return np.asarray(dof_error * stopped, dtype=get_global_dtype())


def joint_pos_penalty(
    ctx: RewardContext,
    *,
    stand_still_scale: float = 5.0,
    velocity_threshold: float = 0.5,
    command_threshold: float = 0.1,
) -> np.ndarray:
    """Penalty for joint deviation that switches scale based on command/body motion."""
    command_norm = np.linalg.norm(ctx.info["commands"], axis=1)
    body_vel = np.linalg.norm(ctx.linvel[:, :2], axis=1)
    running_error = np.linalg.norm(ctx.dof_pos - ctx.default_angles, axis=1)
    moving = (command_norm > command_threshold) | (body_vel > velocity_threshold)
    return np.asarray(
        np.where(moving, running_error, stand_still_scale * running_error),
        dtype=get_global_dtype(),
    )


def upward(ctx: RewardContext) -> np.ndarray:
    """Reward favouring an upright body (no Go2 upright gate)."""
    assert ctx.gravity is not None
    return np.asarray(np.square(1.0 + ctx.gravity[:, 2]), dtype=get_global_dtype())


# ── biped-style rewards ─────────────────────


def track_lin_vel_xy_yaw_frame_exp(ctx: RewardContext) -> np.ndarray:
    """Exponential tracking of xy linear velocity in the gravity-aligned yaw frame.

    Requires ``ctx.linvel_yaw`` (base linvel rotated into yaw frame).
    """
    linvel = ctx.linvel_yaw if ctx.linvel_yaw is not None else ctx.linvel
    commands = ctx.info["commands"]
    lin_vel_error = np.sum(np.square(commands[:, :2] - linvel[:, :2]), axis=1)
    return np.asarray(np.exp(-lin_vel_error / ctx.tracking_sigma), dtype=get_global_dtype())


def track_ang_vel_z_world_exp(ctx: RewardContext) -> np.ndarray:
    """Exponential tracking of yaw angular velocity (world frame)."""
    commands = ctx.info["commands"]
    ang_vel_error = np.square(commands[:, 2] - ctx.gyro[:, 2])
    return np.asarray(np.exp(-ang_vel_error / ctx.tracking_sigma), dtype=get_global_dtype())


def feet_air_time_positive_biped(
    ctx: RewardContext,
    *,
    threshold: float = 0.4,
    command_threshold: float = 0.1,
) -> np.ndarray:
    """Biped foot air-time reward: only rewards single-stance phase.

    Reads ``ctx.info`` keys ``current_air_time``, ``current_contact_time`` (each
    shape (N, 2)); the environment populates them per step.
    """
    air = np.asarray(
        ctx.info.get("current_air_time", np.zeros((ctx.num_envs, 2))), dtype=get_global_dtype()
    )
    contact = np.asarray(
        ctx.info.get("current_contact_time", np.zeros((ctx.num_envs, 2))), dtype=get_global_dtype()
    )
    in_contact = contact > 0.0
    in_mode_time = np.where(in_contact, contact, air)
    single_stance = np.sum(in_contact.astype(np.int32), axis=1) == 1
    masked = np.where(single_stance[:, None], in_mode_time, 0.0)
    reward = np.min(masked, axis=1)
    reward = np.clip(reward, None, threshold)
    moving = np.linalg.norm(ctx.info["commands"][:, :2], axis=1) > command_threshold
    return np.asarray(reward * moving, dtype=get_global_dtype())


def joint_deviation_l1(ctx: RewardContext, joint_indices: np.ndarray | None = None) -> np.ndarray:
    """L1 penalty for joints deviating from their default positions."""
    diff = ctx.dof_pos - ctx.default_angles
    if joint_indices is not None:
        diff = diff[:, joint_indices]
    return np.asarray(np.sum(np.abs(diff), axis=1), dtype=get_global_dtype())
