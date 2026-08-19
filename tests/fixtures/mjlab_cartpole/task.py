# Derived from mujocolab/mjlab v1.6.0 (0fb8a681),
# src/mjlab/tasks/cartpole/cartpole_env_cfg.py.
# Copyright 2025, The mjlab Developers.
# Modified by UniLab for NumPy and fixture-only Hydra/entity adapters; Apache-2.0.
"""NumPy terms for the pinned mjlab Cartpole Balance fixture."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

import numpy as np

from unilab.base import registry
from unilab.envs import ManagerBasedRlEnvCfg, make_manager_based_rl_env
from unilab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from unilab.base.entity import Entity
    from unilab.managers._types import ManagerBasedRlEnv


FIXTURE_ENV_NAME = "MjlabCartpoleBalanceFixture"
_GAUSSIAN_SCALE = math.sqrt(-2.0 * math.log(0.1))
_QUADRATIC_SCALE = math.sqrt(1.0 - 0.1)


def pole_angle_cos_sin(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> np.ndarray:
    """Return cosine and sine of the selected pole angle."""
    asset = cast("Entity", env.scene[asset_cfg.name])
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids]
    return np.concatenate((np.cos(angle), np.sin(angle)), axis=-1)


def _gaussian_tolerance(x: np.ndarray, margin: float) -> np.ndarray:
    if margin == 0.0:
        return (x == 0.0).astype(np.float32)
    scaled = x / margin * _GAUSSIAN_SCALE
    return np.exp(-0.5 * np.square(scaled))


def _quadratic_tolerance(x: np.ndarray, margin: float) -> np.ndarray:
    if margin == 0.0:
        return (x == 0.0).astype(np.float32)
    scaled = x / margin * _QUADRATIC_SCALE
    return np.maximum(1.0 - np.square(scaled), 0.0)


def cartpole_smooth_reward(
    env: ManagerBasedRlEnv,
    cart_cfg: SceneEntityCfg,
    hinge_cfg: SceneEntityCfg,
) -> np.ndarray:
    """Port mjlab's dm_control-style smooth Cartpole reward to NumPy."""
    asset = cast("Entity", env.scene[cart_cfg.name])
    hinge_angle = asset.data.joint_pos[:, hinge_cfg.joint_ids].squeeze(-1)
    upright = (np.cos(hinge_angle) + 1.0) / 2.0
    cart_pos = asset.data.joint_pos[:, cart_cfg.joint_ids].squeeze(-1)
    centered = (1.0 + _gaussian_tolerance(cart_pos, margin=2.0)) / 2.0
    control = env.action_manager.action.squeeze(-1)
    small_control = (4.0 + _quadratic_tolerance(control, margin=1.0)) / 5.0
    hinge_vel = asset.data.joint_vel[:, hinge_cfg.joint_ids].squeeze(-1)
    small_velocity = (1.0 + _gaussian_tolerance(hinge_vel, margin=5.0)) / 2.0
    return upright * centered * small_control * small_velocity


def register_fixture() -> None:
    """Register this fixture without adding it to production bootstrap."""
    if registry.contains(FIXTURE_ENV_NAME):
        return
    registry.register_env_config(FIXTURE_ENV_NAME, ManagerBasedRlEnvCfg)
    registry.register_env(FIXTURE_ENV_NAME, make_manager_based_rl_env, sim_backend="mujoco")


register_fixture()


__all__ = [
    "FIXTURE_ENV_NAME",
    "cartpole_smooth_reward",
    "pole_angle_cos_sin",
    "register_fixture",
]
