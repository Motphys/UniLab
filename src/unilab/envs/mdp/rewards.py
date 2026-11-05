# Derived from mujocolab/mjlab v1.6.0 (0fb8a681),
# src/mjlab/envs/mdp/rewards.py and src/mjlab/tasks/velocity/mdp/rewards.py.
# Copyright 2025, The mjlab Developers.
# Modified by UniLab for Torch and the base-owned entity facade; Apache-2.0.
"""Community-style reward terms for the Manager runtime."""

from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING, Any, ClassVar, cast

import torch

from unilab.envs.mdp.observations import (
    joint_view,
    require_read_plan,
    root_body_view,
)
from unilab.managers.manager_base import ManagerTermBase, ManagerTermBaseCfg
from unilab.managers.scene_entity_config import SceneEntityCfg
from unilab.math.rotation_tensor import quat_apply_inverse

if TYPE_CHECKING:
    from unilab.base.entity import Entity
    from unilab.managers.types import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _positive_std(term_name: str, std: float) -> float:
    if isinstance(std, bool) or not isinstance(std, (int, float)):
        raise TypeError(f"{term_name} std must be a real number")
    value = float(std)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{term_name} std must be finite and positive")
    return value


def _nonnegative_threshold(term_name: str, name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{term_name} {name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{term_name} {name} must be finite and non-negative")
    return result


def _resolve_std_dict(
    term_name: str, param: str, data: object, joint_names: list[str]
) -> tuple[float, ...]:
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
    return tuple(values)


def _command(env: ManagerBasedRlEnv, command_name: str) -> Any:
    try:
        command = env.command_manager.get_command(command_name)
    except KeyError as exc:
        raise KeyError(f"Command term '{command_name}' not found") from exc
    if command is None:
        raise KeyError(f"Command term '{command_name}' not found")
    return command


def _reward(values: Any) -> torch.Tensor:
    """Publish a reward computation on the Torch float32 carrier."""
    return torch.as_tensor(values, dtype=torch.float32)


def is_alive(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Reward environments that have not reached a non-timeout termination."""
    return (~env.termination_manager.terminated).to(torch.float32)


def is_terminated(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Return one for non-timeout terminations."""
    return env.termination_manager.terminated.to(torch.float32)


def root_height(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Return the world-frame root height as a scalar reward or metric term."""
    _, view = root_body_view(env, asset_cfg, "root_height")
    return _reward(view.pos_w[:, 0, 2])


def joint_vel_l2(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize selected joint velocities with an L2-squared kernel."""
    joint_vel = joint_view(env, asset_cfg, "joint_vel_l2").joint_vel[:, asset_cfg.joint_ids]
    return torch.sum(torch.square(joint_vel), dim=1)


def action_rate_l2(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Penalize the first difference of raw policy actions."""
    delta = env.action_manager.action - env.action_manager.prev_action
    return torch.sum(torch.square(delta), dim=1)


def action_acc_l2(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Penalize the second difference of raw policy actions."""
    action_acc = (
        env.action_manager.action
        - 2.0 * env.action_manager.prev_action
        + env.action_manager.prev_prev_action
    )
    return torch.sum(torch.square(action_acc), dim=1)


def flat_orientation_l2(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize non-flat base orientation."""
    asset, view = root_body_view(env, asset_cfg, "flat_orientation_l2")
    gravity_w = asset.data.gravity_vec_w(getattr(env, "device", torch.device("cpu")))[0]
    gravity = quat_apply_inverse(view.quat_w[:, 0, :], gravity_w.expand((view.quat_w.shape[0], 3)))
    return torch.sum(torch.square(gravity[:, :2]), dim=1)


def upright(
    env: ManagerBasedRlEnv,
    std: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Gaussian reward for keeping one selected body upright (mjlab ``upright``).

    The reward is ``exp(-||pg_xy||^2 / std^2)`` where ``pg`` is the world
    gravity vector expressed in the selected body's link frame.  Flat-ground
    form only: mjlab's optional terrain-normal sensors are not ported.
    """
    scale = _positive_std("upright", std)
    asset = cast("Entity", env.scene[asset_cfg.name])
    selected_names = _selected_body_names(asset, asset_cfg)
    body_quat_w = require_read_plan(env, "upright").body_tensor_view(asset, selected_names).quat_w
    if body_quat_w.shape != (env.num_envs, 1, 4):
        raise ValueError(
            f"upright requires exactly one body; received state shape {body_quat_w.shape}"
        )
    quat = torch.as_tensor(
        body_quat_w[:, 0, :],
        dtype=torch.float32,
        device=getattr(env, "device", torch.device("cpu")),
    )
    gravity = asset.data.gravity_vec_w(getattr(env, "device", torch.device("cpu")))[0]
    gravity = gravity.expand((quat.shape[0], 3))
    quat_w, quat_xyz = quat[:, 0:1], quat[:, 1:4]
    cross = 2.0 * torch.linalg.cross(quat_xyz.expand_as(gravity), gravity, dim=-1)
    projected_gravity_b = gravity - (
        quat_w * cross + torch.linalg.cross(quat_xyz.expand(gravity.shape), cross, dim=-1)
    )
    xy_squared = torch.sum(torch.square(projected_gravity_b[:, :2]), dim=1)
    return torch.exp(-xy_squared / scale**2)


def track_linear_velocity(
    env: ManagerBasedRlEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward commanded base linear velocity, assuming commanded z is zero."""
    scale = _positive_std("track_linear_velocity", std)
    _, view = root_body_view(env, asset_cfg, "track_linear_velocity")
    command = _command(env, command_name)
    actual = quat_apply_inverse(view.quat_w[:, 0, :], view.lin_vel_w[:, 0, :])
    command_t = torch.as_tensor(
        command, dtype=torch.float32, device=getattr(env, "device", torch.device("cpu"))
    )
    xy_error = torch.sum(torch.square(command_t[:, :2] - actual[:, :2]), dim=1)
    z_error = torch.square(actual[:, 2])
    return torch.exp(-(xy_error + z_error) / scale**2)


def track_angular_velocity(
    env: ManagerBasedRlEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward commanded yaw rate while keeping roll/pitch rates near zero."""
    scale = _positive_std("track_angular_velocity", std)
    _, view = root_body_view(env, asset_cfg, "track_angular_velocity")
    command = _command(env, command_name)
    actual = quat_apply_inverse(view.quat_w[:, 0, :], view.ang_vel_w[:, 0, :])
    command_t = torch.as_tensor(
        command, dtype=torch.float32, device=getattr(env, "device", torch.device("cpu"))
    )
    z_error = torch.square(command_t[:, 2] - actual[:, 2])
    xy_error = torch.sum(torch.square(actual[:, :2]), dim=1)
    return torch.exp(-(z_error + xy_error) / scale**2)


def body_angular_velocity_penalty(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize roll/pitch angular velocity of one selected body."""
    asset = cast("Entity", env.scene[asset_cfg.name])
    selected_names = _selected_body_names(asset, asset_cfg)
    ang_vel = (
        require_read_plan(env, "body_angular_velocity_penalty")
        .body_tensor_view(asset, selected_names)
        .ang_vel_w
    )
    if ang_vel.shape != (env.num_envs, 1, 3):
        raise ValueError(
            "body_angular_velocity_penalty requires exactly one body; "
            f"received state shape {ang_vel.shape}"
        )
    velocity = torch.as_tensor(
        ang_vel[:, 0, :2], dtype=torch.float32, device=getattr(env, "device", torch.device("cpu"))
    )
    return torch.sum(torch.square(velocity), dim=1)


def joint_pos_limits(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize joint positions if they cross the soft limits."""
    asset = env.scene[asset_cfg.name]
    limits = asset.data.soft_joint_pos_limits(getattr(env, "device", torch.device("cpu")))[
        asset_cfg.joint_ids
    ]
    joint_pos = joint_view(env, asset_cfg, "joint_pos_limits").joint_pos[:, asset_cfg.joint_ids]
    lower_violation = torch.clamp_min(limits[None, :, 0] - joint_pos, 0.0)
    upper_violation = torch.clamp_min(joint_pos - limits[None, :, 1], 0.0)
    return torch.sum(lower_violation + upper_violation, dim=1)


def _selected_joint_names(asset: Entity, asset_cfg: SceneEntityCfg) -> list[str]:
    if asset_cfg.joint_names is None:
        return list(asset.joint_names)
    if isinstance(asset_cfg.joint_names, str):
        return [asset_cfg.joint_names]
    return list(asset_cfg.joint_names)


def _selected_body_names(asset: Entity, asset_cfg: SceneEntityCfg) -> tuple[str, ...]:
    body_ids = asset_cfg.body_ids
    indices = range(len(asset.body_names)) if isinstance(body_ids, slice) else body_ids
    return tuple(asset.body_names[int(index)] for index in indices)


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
        self._default_joint_pos = asset.data.default_joint_pos(
            getattr(env, "device", torch.device("cpu"))
        )
        self._std = torch.tensor(
            self._std, dtype=torch.float32, device=getattr(env, "device", torch.device("cpu"))
        )

    def __call__(self, env: ManagerBasedRlEnv, **params: object) -> torch.Tensor:
        del params
        current = torch.as_tensor(
            joint_view(env, self._asset_cfg, self.name).joint_pos[:, self._asset_cfg.joint_ids],
            dtype=torch.float32,
            device=getattr(env, "device", torch.device("cpu")),
        )
        desired = self._default_joint_pos[:, self._asset_cfg.joint_ids]
        error_squared = torch.square(current - desired)
        return torch.exp(-torch.mean(error_squared / torch.square(self._std), dim=1))


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
        std_standing = _resolve_std_dict(
            self.name, "std_standing", cfg.params.get("std_standing"), joint_names
        )
        std_walking = _resolve_std_dict(
            self.name, "std_walking", cfg.params.get("std_walking"), joint_names
        )
        std_running = _resolve_std_dict(
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
        self._default_joint_pos = asset.data.default_joint_pos(
            getattr(env, "device", torch.device("cpu"))
        )
        device = getattr(env, "device", torch.device("cpu"))
        self._std_standing = torch.tensor(std_standing, dtype=torch.float32, device=device)
        self._std_walking = torch.tensor(std_walking, dtype=torch.float32, device=device)
        self._std_running = torch.tensor(std_running, dtype=torch.float32, device=device)

    def __call__(self, env: ManagerBasedRlEnv, **params: object) -> torch.Tensor:
        del params
        command = torch.as_tensor(
            _command(env, self._command_name),
            dtype=torch.float32,
            device=getattr(env, "device", torch.device("cpu")),
        )
        linear_speed = torch.linalg.vector_norm(command[:, :2], dim=1)
        total_speed = linear_speed + torch.abs(command[:, 2])
        standing = total_speed < self._walking_threshold
        running = total_speed >= self._running_threshold
        walking = ~(standing | running)
        std = (
            self._std_standing[None, :] * standing[:, None]
            + self._std_walking[None, :] * walking[:, None]
            + self._std_running[None, :] * running[:, None]
        )
        current = joint_view(env, self._asset_cfg, self.name).joint_pos[
            :, self._asset_cfg.joint_ids
        ]
        desired = self._default_joint_pos[:, self._asset_cfg.joint_ids]
        error_squared = torch.square(current - desired)
        return torch.exp(-torch.mean(error_squared / torch.square(std), dim=1))


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
    "root_height",
    "track_angular_velocity",
    "track_linear_velocity",
    "upright",
    "variable_posture",
]
