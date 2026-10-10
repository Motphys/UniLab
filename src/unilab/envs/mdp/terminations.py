# Derived from mujocolab/mjlab v1.6.0 (0fb8a681),
# src/mjlab/envs/mdp/terminations.py.
# Copyright 2025, The mjlab Developers.
# Modified by UniLab for the Torch Manager runtime; Apache-2.0.
"""Community-style termination terms for the Torch manager runtime."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

import torch

from unilab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from unilab.base.entity import Entity
    from unilab.managers._types import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _state_tensor(term_name: str, value: object, num_envs: int, *, width: int) -> torch.Tensor:
    state = torch.as_tensor(value, dtype=torch.float32)
    expected = (num_envs, width)
    if tuple(state.shape) != expected:
        raise ValueError(
            f"Termination term '{term_name}' received entity state shape "
            f"{tuple(state.shape)}, expected {expected}"
        )
    if not bool(torch.isfinite(state).all()):
        invalid_rows = torch.nonzero(~torch.isfinite(state).all(dim=1)).flatten()[:10]
        raise ValueError(
            f"Termination term '{term_name}' received NaN or Inf entity state for "
            f"environments {invalid_rows.tolist()}"
        )
    return state


def time_out(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Terminate when the episode length reaches its maximum."""
    return env.episode_length_buf >= env.max_episode_length


def bad_orientation(
    env: ManagerBasedRlEnv,
    limit_angle: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Terminate when the asset orientation exceeds ``limit_angle``."""
    if isinstance(limit_angle, bool) or not isinstance(limit_angle, (int, float)):
        raise TypeError("bad_orientation limit_angle must be a real number")
    if not math.isfinite(limit_angle):
        raise ValueError("bad_orientation limit_angle must be finite")
    asset = cast("Entity", env.scene[asset_cfg.name])
    projected_gravity = _state_tensor(
        "bad_orientation", asset.data.projected_gravity_b, env.num_envs, width=3
    )
    angle = torch.acos((-projected_gravity[:, 2]).clamp(-1.0, 1.0)).abs()
    return angle > limit_angle


def root_height_below_minimum(
    env: ManagerBasedRlEnv,
    minimum_height: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Terminate when the asset root height is below ``minimum_height``."""
    if isinstance(minimum_height, bool) or not isinstance(minimum_height, (int, float)):
        raise TypeError("root_height_below_minimum minimum_height must be a real number")
    if not math.isfinite(minimum_height):
        raise ValueError("root_height_below_minimum minimum_height must be finite")
    asset = cast("Entity", env.scene[asset_cfg.name])
    root_pos_w = _state_tensor(
        "root_height_below_minimum", asset.data.root_link_pos_w, env.num_envs, width=3
    )
    return root_pos_w[:, 2] < minimum_height


def nan_detection(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Terminate environments whose entity physics state contains NaN or Inf.

    The equivalent entity state (root pose/velocity and joint position/velocity)
    is reduced on the Torch carrier so a non-finite physics state terminates the
    episode explicitly.
    """
    asset = cast("Entity", env.scene[asset_cfg.name])
    states = (
        asset.data.root_link_pos_w,
        asset.data.root_link_lin_vel_w,
        asset.data.root_link_ang_vel_w,
        asset.data.joint_pos,
        asset.data.joint_vel,
    )
    invalid = torch.zeros(env.num_envs, dtype=torch.bool, device=getattr(env, "device", "cpu"))
    for state_value in states:
        state = torch.as_tensor(state_value, dtype=torch.float32, device=invalid.device)
        if state.ndim < 1 or state.shape[0] != env.num_envs:
            raise ValueError(
                f"nan_detection received entity state shape {tuple(state.shape)}, "
                f"expected leading dimension {env.num_envs}"
            )
        invalid |= ~torch.isfinite(state).reshape(env.num_envs, -1).all(dim=1)
    return invalid


__all__ = ["bad_orientation", "nan_detection", "root_height_below_minimum", "time_out"]
