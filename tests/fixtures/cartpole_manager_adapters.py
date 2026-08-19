# Derived from Isaac Lab b0542fe2d45bf91c4e1d9ef6952b9c709c80b4e8,
# source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/cartpole.
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# Modified by UniLab as test-only NumPy/entity adapters; BSD-3-Clause.
"""Test-only action and reset adapters shared by pinned Cartpole fixtures."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import TYPE_CHECKING, cast

import numpy as np

from unilab.managers import ActionTerm, ActionTermCfg
from unilab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from unilab.base.entity import Entity
    from unilab.managers._types import ManagerBasedRlEnv


def finite_real(value: Real, *, label: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{label} must be a real number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def numeric_range(value: tuple[float, float] | list[float], *, label: str) -> tuple[float, float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (tuple, list)):
        raise TypeError(f"{label} must be a two-value range")
    if len(value) != 2:
        raise ValueError(f"{label} must contain two values")
    lower = finite_real(value[0], label=f"{label}[0]")
    upper = finite_real(value[1], label=f"{label}[1]")
    if lower > upper:
        raise ValueError(f"{label} lower bound {lower} exceeds upper bound {upper}")
    return lower, upper


@dataclass(kw_only=True)
class JointEffortActionCfg(ActionTermCfg):
    """Fixture-only adapter for the community ``JointEffortActionCfg`` surface."""

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
            raise NotImplementedError("Cartpole fixture JointEffortAction does not support clip")
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
        self._scale = finite_real(cfg.scale, label="JointEffortActionCfg scale")
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
                "Cartpole fixture JointEffortAction expected np.ndarray, "
                f"received {type(actions).__name__}"
            )
        if actions.shape != self._raw_actions.shape:
            raise ValueError(
                "Cartpole fixture JointEffortAction expected shape "
                f"{self._raw_actions.shape}, received {actions.shape}"
            )
        if not np.isfinite(actions).all():
            raise ValueError("Cartpole fixture JointEffortAction received NaN or Inf")
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
    """Write uniformly offset joint defaults through the reset transaction."""
    if env_ids is None:
        raise ValueError("reset_joints_by_offset requires concrete environment IDs")
    position_lower, position_upper = numeric_range(position_range, label="position_range")
    velocity_lower, velocity_upper = numeric_range(velocity_range, label="velocity_range")
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


__all__ = [
    "JointEffortAction",
    "JointEffortActionCfg",
    "finite_real",
    "numeric_range",
    "reset_joints_by_offset",
]
