"""Shared reward functions for locomotion tasks.

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


# ── velocity / orientation penalties ─────────────────────────────────


def lin_vel_z(ctx: RewardContext) -> np.ndarray:
    """Penalty for vertical (z) linear velocity."""
    return np.square(ctx.linvel[:, 2])  # type: ignore[no-any-return]


def ang_vel_xy(ctx: RewardContext) -> np.ndarray:
    """Penalty for roll/pitch angular velocity."""
    return np.sum(np.square(ctx.gyro[:, :2]), axis=1)  # type: ignore[no-any-return]


def roll(ctx: RewardContext) -> np.ndarray:
    """Penalty for deviation from roll orientation."""
    g = ctx.gravity
    assert g is not None
    return np.square(g[:, 0])  # type: ignore[no-any-return]


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
