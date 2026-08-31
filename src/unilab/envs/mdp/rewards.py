# Derived from mujocolab/mjlab v1.6.0 (0fb8a681),
# src/mjlab/envs/mdp/rewards.py and src/mjlab/tasks/velocity/mdp/rewards.py.
# Copyright 2025, The mjlab Developers.
# Modified by UniLab for NumPy and the base-owned entity facade; Apache-2.0.
"""Community-style reward terms for the NumPy manager runtime."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, ClassVar, cast

import numpy as np

from unilab.managers.manager_base import ManagerTermBase, ManagerTermBaseCfg
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


def _nonnegative_threshold(term_name: str, name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise TypeError(f"{term_name} {name} must be a real number")
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{term_name} {name} must be finite and non-negative")
    return result


def _resolve_std_dict(
    term_name: str, param: str, data: object, joint_names: list[str]
) -> np.ndarray:
    """Resolve a ``{regex: std}`` mapping into one positive std per joint name.

    Mirrors mjlab ``resolve_matching_names_values`` with ``preserve_order=False``:
    full-regex matching, every pattern must match at least one joint, and every
    joint must match exactly one pattern.
    """
    if not isinstance(data, dict) or not data:
        raise TypeError(f"{term_name} {param} must be a non-empty dict of regex patterns to std")
    values: list[float] = []
    matched_patterns: set[str] = set()
    for name in joint_names:
        matches: list[float] = []
        for pattern, std in data.items():
            if not isinstance(pattern, str):
                raise TypeError(f"{term_name} {param} keys must be regex strings")
            try:
                matched = re.fullmatch(pattern, name) is not None
            except re.error as exc:
                raise ValueError(f"{term_name} {param} invalid regex {pattern!r}: {exc}") from exc
            if matched:
                matches.append(_positive_std(f"{term_name} {param}[{pattern!r}]", std))
                matched_patterns.add(pattern)
        if len(matches) != 1:
            raise ValueError(
                f"{term_name} {param} must match joint '{name}' exactly once, "
                f"got {len(matches)} matches"
            )
        values.append(matches[0])
    unmatched = [pattern for pattern in data if pattern not in matched_patterns]
    if unmatched:
        raise ValueError(f"{term_name} {param} patterns matched no joints: {unmatched}")
    return np.asarray(values, dtype=np.float64)


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


def joint_pos_limits(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> np.ndarray:
    """Penalize joint positions if they cross the soft limits."""
    asset = cast("Entity", env.scene[asset_cfg.name])
    limits = np.asarray(asset.data.soft_joint_pos_limits)[asset_cfg.joint_ids]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    out_of_limits = -np.clip(joint_pos - limits[:, 0], min=None, max=0.0)
    out_of_limits += np.clip(joint_pos - limits[:, 1], min=0.0, max=None)
    return np.sum(out_of_limits, axis=1)


def _selected_joint_names(asset: Entity, asset_cfg: SceneEntityCfg) -> list[str]:
    if asset_cfg.joint_names is None:
        return list(asset.joint_names)
    if isinstance(asset_cfg.joint_names, str):
        return [asset_cfg.joint_names]
    return list(asset_cfg.joint_names)


class posture(ManagerTermBase):
    """Penalize joint deviation from default pose with a per-joint-std Gaussian kernel.

    ``params["std"]`` maps joint-name regexes to per-joint standard deviations;
    the reward is ``exp(-mean(error^2 / std^2))`` over the selected joints.
    """

    _allowed_params: ClassVar[frozenset[str]] = frozenset({"std", "asset_cfg"})

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        unexpected = set(cfg.params) - self._allowed_params
        if unexpected:
            raise TypeError(f"{self.name} received unsupported parameters: {sorted(unexpected)}")
        asset_cfg = cfg.params.get("asset_cfg", _DEFAULT_ASSET_CFG)
        if not isinstance(asset_cfg, SceneEntityCfg):
            raise TypeError(f"{self.name} asset_cfg must be a SceneEntityCfg")
        asset = cast("Entity", env.scene[asset_cfg.name])
        self._asset_cfg = asset_cfg
        self._std = _resolve_std_dict(
            self.name, "std", cfg.params.get("std"), _selected_joint_names(asset, asset_cfg)
        )
        self._default_joint_pos = np.asarray(asset.data.default_joint_pos)

    def __call__(self, env: ManagerBasedRlEnv, **params: object) -> np.ndarray:
        del params
        asset = cast("Entity", env.scene[self._asset_cfg.name])
        current = asset.data.joint_pos[:, self._asset_cfg.joint_ids]
        desired = self._default_joint_pos[:, self._asset_cfg.joint_ids]
        error_squared = np.square(current - desired)
        return np.exp(-np.mean(error_squared / np.square(self._std), axis=1))


class variable_posture(ManagerTermBase):
    """``posture`` with speed-dependent tolerance: standing/walking/running std maps.

    The per-joint std is selected from ``std_standing`` / ``std_walking`` /
    ``std_running`` by the total command speed (planar norm plus yaw magnitude)
    against ``walking_threshold`` and ``running_threshold``.
    """

    _allowed_params: ClassVar[frozenset[str]] = frozenset(
        {
            "std_standing",
            "std_walking",
            "std_running",
            "asset_cfg",
            "command_name",
            "walking_threshold",
            "running_threshold",
        }
    )

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        unexpected = set(cfg.params) - self._allowed_params
        if unexpected:
            raise TypeError(f"{self.name} received unsupported parameters: {sorted(unexpected)}")
        asset_cfg = cfg.params.get("asset_cfg", _DEFAULT_ASSET_CFG)
        if not isinstance(asset_cfg, SceneEntityCfg):
            raise TypeError(f"{self.name} asset_cfg must be a SceneEntityCfg")
        asset = cast("Entity", env.scene[asset_cfg.name])
        joint_names = _selected_joint_names(asset, asset_cfg)
        self._asset_cfg = asset_cfg
        self._std_standing = _resolve_std_dict(
            self.name, "std_standing", cfg.params.get("std_standing"), joint_names
        )
        self._std_walking = _resolve_std_dict(
            self.name, "std_walking", cfg.params.get("std_walking"), joint_names
        )
        self._std_running = _resolve_std_dict(
            self.name, "std_running", cfg.params.get("std_running"), joint_names
        )
        command_name = cfg.params.get("command_name")
        if not isinstance(command_name, str) or not command_name:
            raise ValueError(f"{self.name} command_name must be a non-empty string")
        self._command_name = command_name
        self._walking_threshold = _nonnegative_threshold(
            self.name, "walking_threshold", cfg.params.get("walking_threshold", 0.5)
        )
        self._running_threshold = _nonnegative_threshold(
            self.name, "running_threshold", cfg.params.get("running_threshold", 1.5)
        )
        if self._walking_threshold >= self._running_threshold:
            raise ValueError(
                f"{self.name} walking_threshold must be below running_threshold, got "
                f"{self._walking_threshold} >= {self._running_threshold}"
            )
        self._default_joint_pos = np.asarray(asset.data.default_joint_pos)

    def __call__(self, env: ManagerBasedRlEnv, **params: object) -> np.ndarray:
        del params
        asset = cast("Entity", env.scene[self._asset_cfg.name])
        command = _command(env, self._command_name)
        linear_speed = np.linalg.norm(command[:, :2], axis=1)
        total_speed = linear_speed + np.abs(command[:, 2])
        standing = total_speed < self._walking_threshold
        running = total_speed >= self._running_threshold
        walking = ~(standing | running)
        std = (
            self._std_standing[None, :] * standing[:, None]
            + self._std_walking[None, :] * walking[:, None]
            + self._std_running[None, :] * running[:, None]
        )
        current = asset.data.joint_pos[:, self._asset_cfg.joint_ids]
        desired = self._default_joint_pos[:, self._asset_cfg.joint_ids]
        error_squared = np.square(current - desired)
        return np.exp(-np.mean(error_squared / np.square(std), axis=1))


__all__ = [
    "action_acc_l2",
    "action_rate_l2",
    "body_angular_velocity_penalty",
    "flat_orientation_l2",
    "is_alive",
    "is_terminated",
    "joint_pos_limits",
    "joint_vel_l2",
    "posture",
    "track_angular_velocity",
    "track_linear_velocity",
    "variable_posture",
]
