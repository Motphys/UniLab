# Derived from Isaac Lab b0542fe2d45bf91c4e1d9ef6952b9c709c80b4e8,
# source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/cartpole.
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# Modified by UniLab for NumPy and the fixture-local MJCF/entity adapter; BSD-3-Clause.
"""NumPy terms and adapters for the Isaac Lab Cartpole migration fixture."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import TYPE_CHECKING, cast

import numpy as np

from unilab.base import registry
from unilab.envs import ManagerBasedRlEnvCfg, make_manager_based_rl_env
from unilab.managers import ActionTerm, ActionTermCfg
from unilab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from unilab.base.entity import Entity
    from unilab.managers._types import ManagerBasedRlEnv


FIXTURE_ENV_NAME = "IsaacLabCartpoleFixture"


def _finite_real(value: Real, *, label: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{label} must be a real number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _range(value: tuple[float, float] | list[float], *, label: str) -> tuple[float, float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (tuple, list)):
        raise TypeError(f"{label} must be a two-value range")
    if len(value) != 2:
        raise ValueError(f"{label} must contain two values")
    lower = _finite_real(value[0], label=f"{label}[0]")
    upper = _finite_real(value[1], label=f"{label}[1]")
    if lower > upper:
        raise ValueError(f"{label} lower bound {lower} exceeds upper bound {upper}")
    return lower, upper


@dataclass(kw_only=True)
class JointEffortActionCfg(ActionTermCfg):
    """Fixture-local adapter for Isaac Lab's ``JointEffortActionCfg``."""

    actuator_names: tuple[str, ...] | list[str]
    scale: float = 1.0

    def build(self, env: ManagerBasedRlEnv) -> JointEffortAction:
        return JointEffortAction(self, env)


class JointEffortAction(ActionTerm):
    """Scale policy actions and write entity-local actuator efforts."""

    cfg: JointEffortActionCfg

    def __init__(self, cfg: JointEffortActionCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        if cfg.clip is not None:
            raise NotImplementedError(
                "IsaacLabCartpoleFixture JointEffortAction does not support clip"
            )
        if isinstance(cfg.actuator_names, (str, bytes)) or not isinstance(
            cfg.actuator_names, (tuple, list)
        ):
            raise TypeError("JointEffortActionCfg actuator_names must be an ordered sequence")
        actuator_ids, actuator_names = self._entity.find_actuators(
            cfg.actuator_names,
            preserve_order=True,
        )
        if not actuator_ids:
            raise ValueError(
                "JointEffortActionCfg actuator_names resolved no actuators; "
                f"patterns={list(cfg.actuator_names)}"
            )
        self._actuator_ids = np.asarray(actuator_ids, dtype=np.intp)
        self._actuator_ids.setflags(write=False)
        self._actuator_names = tuple(actuator_names)
        self._scale = _finite_real(cfg.scale, label="JointEffortActionCfg scale")
        self._raw_actions = np.zeros((self.num_envs, len(actuator_ids)), dtype=np.float32)
        self._processed_actions = np.zeros_like(self._raw_actions)

    @property
    def action_dim(self) -> int:
        return self._raw_actions.shape[1]

    @property
    def raw_action(self) -> np.ndarray:
        return self._raw_actions

    def process_actions(self, actions: np.ndarray) -> None:
        if not isinstance(actions, np.ndarray):
            raise TypeError(
                "IsaacLabCartpoleFixture JointEffortAction expected np.ndarray, "
                f"received {type(actions).__name__}"
            )
        if actions.shape != self._raw_actions.shape:
            raise ValueError(
                "IsaacLabCartpoleFixture JointEffortAction expected shape "
                f"{self._raw_actions.shape}, received {actions.shape}"
            )
        if not np.isfinite(actions).all():
            raise ValueError("IsaacLabCartpoleFixture JointEffortAction received NaN or Inf")
        np.copyto(self._raw_actions, actions)
        np.multiply(actions, self._scale, out=self._processed_actions)

    def apply_actions(self) -> None:
        self._entity.data.write_ctrl(
            self._processed_actions,
            actuator_ids=self._actuator_ids,
        )

    def reset(self, env_ids: np.ndarray | slice | None = None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        self._raw_actions[ids] = 0.0
        self._processed_actions[ids] = 0.0


def reset_joints_by_offset(
    env: ManagerBasedRlEnv,
    env_ids: np.ndarray | None,
    position_range: tuple[float, float] | list[float],
    velocity_range: tuple[float, float] | list[float],
    asset_cfg: SceneEntityCfg,
) -> None:
    """Port Isaac Lab's joint-offset reset through the entity reset transaction."""
    if env_ids is None:
        raise ValueError("reset_joints_by_offset requires concrete environment IDs")
    position_lower, position_upper = _range(position_range, label="position_range")
    velocity_lower, velocity_upper = _range(velocity_range, label="velocity_range")
    asset = cast("Entity", env.scene[asset_cfg.name])
    joint_ids = asset_cfg.joint_ids
    default_position = asset.data.default_joint_pos[env_ids][:, joint_ids]
    default_velocity = asset.data.default_joint_vel[env_ids][:, joint_ids]
    position = default_position + env.rng.uniform(
        position_lower,
        position_upper,
        default_position.shape,
    )
    velocity = default_velocity + env.rng.uniform(
        velocity_lower,
        velocity_upper,
        default_velocity.shape,
    )
    asset.write_joint_state_to_sim(
        position.astype(default_position.dtype, copy=False),
        velocity.astype(default_velocity.dtype, copy=False),
        joint_ids=joint_ids,
        env_ids=env_ids,
    )


def joint_pos_target_l2(
    env: ManagerBasedRlEnv,
    target: float,
    asset_cfg: SceneEntityCfg,
) -> np.ndarray:
    """Penalize wrapped joint-position deviation from a target value."""
    target_value = _finite_real(target, label="joint_pos_target_l2 target")
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
    lower, upper = _range(bounds, label="joint_pos_out_of_manual_limit bounds")
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
