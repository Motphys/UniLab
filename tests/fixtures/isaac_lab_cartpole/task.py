# Derived from Isaac Lab b0542fe2d45bf91c4e1d9ef6952b9c709c80b4e8,
# source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/cartpole.
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# Modified by UniLab for NumPy and the fixture-local MJCF/entity adapter; BSD-3-Clause.
"""NumPy terms for the Isaac Lab Cartpole migration fixture."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

import numpy as np

from tests.fixtures.cartpole_manager_adapters import (
    JointEffortAction,
    JointEffortActionCfg,
    finite_real,
    numeric_range,
    reset_joints_by_offset,
)
from unilab.base import registry
from unilab.envs import ManagerBasedRlEnvCfg, make_manager_based_rl_env
from unilab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from unilab.base.entity import Entity
    from unilab.managers._types import ManagerBasedRlEnv


FIXTURE_ENV_NAME = "IsaacLabCartpoleFixture"


def joint_pos_target_l2(
    env: ManagerBasedRlEnv,
    target: float,
    asset_cfg: SceneEntityCfg,
) -> np.ndarray:
    """Penalize wrapped joint-position deviation from a target value."""
    target_value = finite_real(target, label="joint_pos_target_l2 target")
    asset = cast("Entity", env.scene[asset_cfg.name])
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    wrapped = np.remainder(joint_pos + math.pi, 2.0 * math.pi) - math.pi
    return np.sum(np.square(wrapped - target_value), axis=1)


def joint_vel_l1(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
) -> np.ndarray:
    """Penalize the absolute velocity of selected joints."""
    asset = cast("Entity", env.scene[asset_cfg.name])
    return np.sum(np.abs(asset.data.joint_vel[:, asset_cfg.joint_ids]), axis=1)


def joint_pos_out_of_manual_limit(
    env: ManagerBasedRlEnv,
    bounds: tuple[float, float] | list[float],
    asset_cfg: SceneEntityCfg,
) -> np.ndarray:
    """Terminate when a selected joint leaves the configured manual bounds."""
    lower, upper = numeric_range(bounds, label="joint_pos_out_of_manual_limit bounds")
    asset = cast("Entity", env.scene[asset_cfg.name])
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    return np.any((joint_pos < lower) | (joint_pos > upper), axis=1)


def register_fixture() -> None:
    """Register the fixture without adding it to the production task package."""
    if registry.contains(FIXTURE_ENV_NAME):
        return
    registry.register_env_config(FIXTURE_ENV_NAME, ManagerBasedRlEnvCfg)
    registry.register_env(
        FIXTURE_ENV_NAME,
        make_manager_based_rl_env,
        sim_backend="mujoco",
    )


register_fixture()


__all__ = [
    "FIXTURE_ENV_NAME",
    "JointEffortAction",
    "JointEffortActionCfg",
    "joint_pos_out_of_manual_limit",
    "joint_pos_target_l2",
    "joint_vel_l1",
    "register_fixture",
    "reset_joints_by_offset",
]
