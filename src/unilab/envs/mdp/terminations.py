# Derived from mujocolab/mjlab v1.6.0 (0fb8a681),
# src/mjlab/envs/mdp/terminations.py.
# Copyright 2025, The mjlab Developers.
# Modified by UniLab for NumPy and the base-owned entity facade; Apache-2.0.
"""Community-style termination terms for the NumPy manager runtime."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

from unilab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from unilab.base.entity import Entity
    from unilab.managers._types import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _require_state(term_name: str, value: np.ndarray, num_envs: int) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(
            f"Termination term '{term_name}' expected an np.ndarray entity state, "
            f"got {type(value).__name__}"
        )
    if value.shape != (num_envs, 3):
        raise ValueError(
            f"Termination term '{term_name}' received entity state shape {value.shape}, "
            f"expected ({num_envs}, 3)"
        )
    if not np.isfinite(value).all():
        env_ids = np.flatnonzero(~np.isfinite(value).all(axis=1)).tolist()
        raise ValueError(
            f"Termination term '{term_name}' received NaN or Inf entity state for "
            f"environments {env_ids[:10]}"
        )
    return value


def time_out(env: ManagerBasedRlEnv) -> np.ndarray:
    """Terminate when the episode length reaches its maximum."""
    return env.episode_length_buf >= env.max_episode_length


def bad_orientation(
    env: ManagerBasedRlEnv,
    limit_angle: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> np.ndarray:
    """Terminate when the asset orientation exceeds ``limit_angle``."""
    if isinstance(limit_angle, bool) or not isinstance(limit_angle, (int, float, np.number)):
        raise TypeError("bad_orientation limit_angle must be a real number")
    if not np.isfinite(limit_angle):
        raise ValueError("bad_orientation limit_angle must be finite")
    asset = cast("Entity", env.scene[asset_cfg.name])
    projected_gravity = _require_state(
        "bad_orientation", asset.data.projected_gravity_b, env.num_envs
    )
    return np.abs(np.arccos(np.clip(-projected_gravity[:, 2], -1.0, 1.0))) > limit_angle


def root_height_below_minimum(
    env: ManagerBasedRlEnv,
    minimum_height: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> np.ndarray:
    """Terminate when the asset root height is below ``minimum_height``."""
    if isinstance(minimum_height, bool) or not isinstance(minimum_height, (int, float, np.number)):
        raise TypeError("root_height_below_minimum minimum_height must be a real number")
    if not np.isfinite(minimum_height):
        raise ValueError("root_height_below_minimum minimum_height must be finite")
    asset = cast("Entity", env.scene[asset_cfg.name])
    root_pos_w = _require_state(
        "root_height_below_minimum", asset.data.root_link_pos_w, env.num_envs
    )
    return root_pos_w[:, 2] < minimum_height


__all__ = ["bad_orientation", "root_height_below_minimum", "time_out"]
