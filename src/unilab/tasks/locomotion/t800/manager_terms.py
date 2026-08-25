"""T800-specific terms for the Manager-Based locomotion environment."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import TYPE_CHECKING, Any

import numpy as np

from unilab.envs.mdp import JointPositionAction, JointPositionActionCfg
from unilab.managers.manager_base import ManagerTermBaseCfg
from unilab.tasks.locomotion.common.sensor_terms import SensorTermBase
from unilab.utils.rotation import np_quat_apply_inverse, np_yaw_quat

if TYPE_CHECKING:
    from collections.abc import Sequence

    from unilab.managers._types import ManagerBasedRlEnv


_LATERAL_FOOT_SENSORS = (
    "left_foot_pos",
    "right_foot_pos",
    "base_link_quaternion",
)


def _finite_real(value: Any, *, label: str, strict_positive: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{label} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    if result < 0.0 or (strict_positive and result == 0.0):
        relation = "greater than zero" if strict_positive else "non-negative"
        raise ValueError(f"{label} must be {relation}")
    return result


def compute_lateral_feet_penalty(
    left_foot: np.ndarray,
    right_foot: np.ndarray,
    base_quat: np.ndarray,
    min_width: float = 0.20,
    sigma: float = 0.04,
) -> np.ndarray:
    """Penalize signed foot separation below a minimum heading-frame width."""
    separation_w = left_foot - right_foot
    separation_heading = np_quat_apply_inverse(np_yaw_quat(base_quat), separation_w)
    signed_width = separation_heading[:, 1]
    deficit = np.maximum(min_width - signed_width, 0.0)
    return 1.0 - np.exp(-np.square(deficit / sigma))


def _validate_unique_selectors(label: str, selectors: Sequence[str]) -> None:
    if len(selectors) != len(set(selectors)):
        raise ValueError(f"T800JointPositionAction {label} selectors contain duplicates")


@dataclass(kw_only=True)
class T800JointPositionActionCfg(JointPositionActionCfg):
    """Control selected joints while holding the remaining T800 joints."""

    held_actuator_names: tuple[str, ...] | list[str]

    def build(self, env: ManagerBasedRlEnv) -> "T800JointPositionAction":
        return T800JointPositionAction(self, env)


class T800JointPositionAction(JointPositionAction):
    """Joint-position action with cold-path-resolved non-policy hold targets."""

    def __init__(self, cfg: T800JointPositionActionCfg, env: ManagerBasedRlEnv):
        _validate_unique_selectors("active", cfg.actuator_names)
        _validate_unique_selectors("held", cfg.held_actuator_names)

        super().__init__(cfg, env)
        active_ids, _ = self._entity.find_joints_by_actuator_names(cfg.actuator_names)
        held_ids, _ = self._entity.find_joints_by_actuator_names(cfg.held_actuator_names)
        if len(set(active_ids)) != len(active_ids) or len(set(held_ids)) != len(held_ids):
            raise ValueError("T800JointPositionAction resolved joint IDs must be unique")
        if set(active_ids) & set(held_ids):
            raise ValueError("T800JointPositionAction active and held joints overlap")
        if set(active_ids) | set(held_ids) != set(range(self._entity.num_joints)):
            raise ValueError(
                "T800JointPositionAction active and held joints must form a complete partition"
            )

        self._held_ids = np.asarray(held_ids, dtype=np.intp)
        self._held_ids.setflags(write=False)
        held_shape = (self.num_envs, self._held_ids.size)
        self._held_default_targets = np.empty(
            held_shape, dtype=self._entity.data.default_joint_pos.dtype
        )
        self._held_encoder_bias = np.empty(held_shape, dtype=self._entity.data.encoder_bias.dtype)

    def apply_actions(self) -> None:
        super().apply_actions()
        np.take(
            self._entity.data.default_joint_pos,
            self._held_ids,
            axis=1,
            out=self._held_default_targets,
        )
        np.take(
            self._entity.data.encoder_bias,
            self._held_ids,
            axis=1,
            out=self._held_encoder_bias,
        )
        np.subtract(
            self._held_default_targets,
            self._held_encoder_bias,
            out=self._held_default_targets,
        )
        self._entity.set_joint_position_target(
            self._held_default_targets,
            joint_ids=self._held_ids,
        )


class penalty_close_feet_lateral(SensorTermBase):
    """Penalty for insufficient signed lateral foot separation."""

    _allowed_params = frozenset({"min_width", "sigma"})

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self._min_width = _finite_real(
            cfg.params.get("min_width", 0.20), label=f"{self.name} min_width"
        )
        self._sigma = _finite_real(
            cfg.params.get("sigma", 0.04),
            label=f"{self.name} sigma",
            strict_positive=True,
        )
        self._sensor_view = self._bind(_LATERAL_FOOT_SENSORS)
        if self._sensor_view.dimensions != (3, 3, 4):
            raise ValueError(
                f"{self.name} sensor dimensions must be (3, 3, 4); received "
                f"{self._sensor_view.dimensions} on backend "
                f"'{self._sensor_view.backend_type}'"
            )

    def __call__(self, env: ManagerBasedRlEnv, **params: Any) -> np.ndarray:
        del params
        values = self._read(self._sensor_view, self.name)
        expected_shape = (env.num_envs, 10)
        if values.shape != expected_shape:
            raise ValueError(
                f"{self.name} sensor view must have shape {expected_shape}, got {values.shape}"
            )
        return compute_lateral_feet_penalty(
            values[:, 0:3],
            values[:, 3:6],
            values[:, 6:10],
            min_width=self._min_width,
            sigma=self._sigma,
        )


__all__ = [
    "T800JointPositionAction",
    "T800JointPositionActionCfg",
    "compute_lateral_feet_penalty",
    "penalty_close_feet_lateral",
]
