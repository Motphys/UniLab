# Derived from mujocolab/mjlab v1.6.0 (0fb8a681),
# src/mjlab/envs/mdp/rewards.py and src/mjlab/tasks/velocity/mdp/rewards.py.
# Copyright 2025, The mjlab Developers.
# Modified by UniLab for NumPy and the base-owned entity facade; Apache-2.0.
"""Community-style reward terms for the NumPy manager runtime."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

from unilab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from unilab.base.entity import Entity
    from unilab.managers._types import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _positive_std(term_name: str, std: float) -> float:
    if isinstance(std, bool) or not isinstance(std, (int, float, np.number)):
        raise TypeError(f"{term_name} std must be a real number")
    value = float(std)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{term_name} std must be finite and positive")
    return value


def _command(env: ManagerBasedRlEnv, command_name: str) -> np.ndarray:
    try:
        command = env.command_manager.get_command(command_name)
    except KeyError as exc:
        raise KeyError(f"Command term '{command_name}' not found") from exc
    if command is None:
        raise KeyError(f"Command term '{command_name}' not found")
    return command


def is_alive(env: ManagerBasedRlEnv) -> np.ndarray:
    """Reward environments that have not reached a non-timeout termination."""
    return np.logical_not(env.termination_manager.terminated).astype(np.float32, copy=False)


def is_terminated(env: ManagerBasedRlEnv) -> np.ndarray:
    """Return one for non-timeout terminations."""
    return env.termination_manager.terminated.astype(np.float32, copy=False)


def joint_vel_l2(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> np.ndarray:
    """Penalize selected joint velocities with an L2-squared kernel."""
    asset = cast("Entity", env.scene[asset_cfg.name])
    return np.sum(np.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), axis=1)


def action_rate_l2(env: ManagerBasedRlEnv) -> np.ndarray:
    """Penalize the first difference of raw policy actions."""
    delta = env.action_manager.action - env.action_manager.prev_action
    return np.sum(np.square(delta), axis=1)


def action_acc_l2(env: ManagerBasedRlEnv) -> np.ndarray:
    """Penalize the second difference of raw policy actions."""
    action_acc = (
        env.action_manager.action
        - 2.0 * env.action_manager.prev_action
        + env.action_manager.prev_prev_action
    )
    return np.sum(np.square(action_acc), axis=1)


def flat_orientation_l2(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> np.ndarray:
    """Penalize non-flat base orientation."""
    asset = cast("Entity", env.scene[asset_cfg.name])
    return np.sum(np.square(asset.data.projected_gravity_b[:, :2]), axis=1)


def track_linear_velocity(
    env: ManagerBasedRlEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> np.ndarray:
    """Reward commanded base linear velocity, assuming commanded z is zero."""
    scale = _positive_std("track_linear_velocity", std)
    asset = cast("Entity", env.scene[asset_cfg.name])
    command = _command(env, command_name)
    actual = asset.data.root_link_lin_vel_b
    xy_error = np.sum(np.square(command[:, :2] - actual[:, :2]), axis=1)
    z_error = np.square(actual[:, 2])
    return np.exp(-(xy_error + z_error) / scale**2)


def track_angular_velocity(
    env: ManagerBasedRlEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> np.ndarray:
    """Reward commanded yaw rate while keeping roll/pitch rates near zero."""
    scale = _positive_std("track_angular_velocity", std)
    asset = cast("Entity", env.scene[asset_cfg.name])
    command = _command(env, command_name)
    actual = asset.data.root_link_ang_vel_b
    z_error = np.square(command[:, 2] - actual[:, 2])
    xy_error = np.sum(np.square(actual[:, :2]), axis=1)
    return np.exp(-(z_error + xy_error) / scale**2)


def body_angular_velocity_penalty(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> np.ndarray:
    """Penalize roll/pitch angular velocity of one selected body."""
    asset = cast("Entity", env.scene[asset_cfg.name])
    ang_vel = asset.data.body_link_ang_vel_w[:, asset_cfg.body_ids, :]
    if ang_vel.shape != (env.num_envs, 1, 3):
        raise ValueError(
            "body_angular_velocity_penalty requires exactly one body; "
            f"received state shape {ang_vel.shape}"
        )
    return np.sum(np.square(ang_vel[:, 0, :2]), axis=1)


__all__ = [
    "action_acc_l2",
    "action_rate_l2",
    "body_angular_velocity_penalty",
    "flat_orientation_l2",
    "is_alive",
    "is_terminated",
    "joint_vel_l2",
    "track_angular_velocity",
    "track_linear_velocity",
]
