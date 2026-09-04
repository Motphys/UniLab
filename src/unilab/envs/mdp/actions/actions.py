# Derived from mujocolab/mjlab v1.6.0 (0fb8a681),
# src/mjlab/envs/mdp/actions/actions.py.
# Copyright 2025, The mjlab Developers.
# Modified by UniLab for NumPy and the SimBackend/entity contracts; Apache-2.0.
"""Joint transmission actions for the NumPy Manager-Based runtime."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from numbers import Real
from typing import TYPE_CHECKING, Any

import numpy as np

from unilab.dtype_config import get_global_dtype
from unilab.managers.action_manager import ActionTerm, ActionTermCfg

if TYPE_CHECKING:
    from unilab.base.entity import Entity
    from unilab.managers._types import ManagerBasedRlEnv


def _real(value: Any, *, label: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{label} must be a real number, got {type(value).__name__}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite, got {result}")
    return result


def _resolve_named_values(
    values: dict[str, Any], names: list[str], *, label: str
) -> tuple[list[int], list[Any]]:
    """Resolve regex-keyed values once, preserving target-name order."""
    if not isinstance(values, dict):
        raise TypeError(f"{label} must be a dict")
    patterns = list(values)
    matched_by_pattern = [False] * len(patterns)
    indices: list[int] = []
    resolved: list[Any] = []
    for index, name in enumerate(names):
        matches: list[int] = []
        for pattern_index, pattern in enumerate(patterns):
            try:
                matches_pattern = re.fullmatch(pattern, name) is not None
            except re.error as exc:
                raise ValueError(f"{label} contains invalid regex {pattern!r}: {exc}") from exc
            if matches_pattern:
                matches.append(pattern_index)
        if len(matches) > 1:
            rendered = [patterns[pattern_index] for pattern_index in matches]
            raise ValueError(f"{label} patterns {rendered} both match target '{name}'")
        if matches:
            pattern_index = matches[0]
            matched_by_pattern[pattern_index] = True
            indices.append(index)
            resolved.append(values[patterns[pattern_index]])
    missing = [pattern for pattern, matched in zip(patterns, matched_by_pattern) if not matched]
    if missing:
        raise ValueError(f"{label} patterns {missing} match no targets; available={names}")
    return indices, resolved


@dataclass(kw_only=True)
class BaseActionCfg(ActionTermCfg):
    """Configuration shared by entity joint actions."""

    actuator_names: tuple[str, ...] | list[str]
    scale: float | dict[str, float] = 1.0
    offset: float | dict[str, float] = 0.0
    preserve_order: bool = False


class BaseAction(ActionTerm):
    """Apply a cold-path-resolved affine transform to raw policy actions."""

    cfg: BaseActionCfg
    _entity: Entity

    def __init__(self, cfg: BaseActionCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg=cfg, env=env)
        target_ids, target_names = self._entity.find_joints_by_actuator_names(cfg.actuator_names)
        self._target_ids = np.asarray(target_ids, dtype=np.intp)
        self._target_ids.setflags(write=False)
        self._target_names = list(target_names)
        self._action_dim = len(target_ids)
        dtype = get_global_dtype()
        self._raw_actions = np.zeros((self.num_envs, self.action_dim), dtype=dtype)
        self._processed_actions = np.zeros_like(self._raw_actions)
        self._scale = self._resolve_affine(cfg.scale, default=1.0, label="scale")
        self._offset = self._resolve_affine(cfg.offset, default=0.0, label="offset")
        self._clip = self._resolve_clip(cfg.clip)

    def _resolve_affine(
        self, value: float | dict[str, float], *, default: float, label: str
    ) -> float | np.ndarray:
        if isinstance(value, dict):
            result = np.full_like(self._raw_actions, default)
            indices, resolved = _resolve_named_values(
                value, self._target_names, label=f"{type(self).__name__} {label}"
            )
            result[:, indices] = [
                _real(item, label=f"{type(self).__name__} {label}") for item in resolved
            ]
            return result
        return _real(value, label=f"{type(self).__name__} {label}")

    def _resolve_clip(self, value: dict[str, tuple] | None) -> np.ndarray | None:
        if value is None:
            return None
        result = np.empty((*self._raw_actions.shape, 2), dtype=self._raw_actions.dtype)
        result[..., 0] = -np.inf
        result[..., 1] = np.inf
        indices, bounds = _resolve_named_values(
            value, self._target_names, label=f"{type(self).__name__} clip"
        )
        for index, raw_bounds in zip(indices, bounds, strict=True):
            if not isinstance(raw_bounds, (tuple, list)) or len(raw_bounds) != 2:
                raise TypeError(
                    f"{type(self).__name__} clip for '{self._target_names[index]}' "
                    "must be a (min, max) pair"
                )
            lower = _real(raw_bounds[0], label=f"{type(self).__name__} clip lower")
            upper = _real(raw_bounds[1], label=f"{type(self).__name__} clip upper")
            if lower > upper:
                raise ValueError(
                    f"{type(self).__name__} clip lower {lower} exceeds upper {upper} "
                    f"for '{self._target_names[index]}'"
                )
            result[:, index, 0] = lower
            result[:, index, 1] = upper
        return result

    @property
    def scale(self) -> float | np.ndarray:
        return self._scale

    @property
    def offset(self) -> float | np.ndarray:
        return self._offset

    @property
    def raw_action(self) -> np.ndarray:
        return self._raw_actions

    @property
    def processed_action(self) -> np.ndarray:
        return self._processed_actions

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def target_ids(self) -> np.ndarray:
        return self._target_ids

    @property
    def target_names(self) -> list[str]:
        return list(self._target_names)

    def process_actions(self, actions: np.ndarray) -> None:
        if not isinstance(actions, np.ndarray):
            raise TypeError(
                f"{type(self).__name__} expected np.ndarray, got {type(actions).__name__}"
            )
        if actions.shape != self._raw_actions.shape:
            raise ValueError(
                f"{type(self).__name__} expected action shape {self._raw_actions.shape}, "
                f"got {actions.shape}"
            )
        if not np.isfinite(actions).all():
            raise ValueError(f"{type(self).__name__} received NaN or Inf actions")
        self._raw_actions[:] = actions
        np.multiply(self._raw_actions, self._scale, out=self._processed_actions)
        np.add(self._processed_actions, self._offset, out=self._processed_actions)
        if self._clip is not None:
            np.clip(
                self._processed_actions,
                self._clip[..., 0],
                self._clip[..., 1],
                out=self._processed_actions,
            )

    def reset(self, env_ids: np.ndarray | slice | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = 0.0


@dataclass(kw_only=True)
class JointPositionActionCfg(BaseActionCfg):
    """Configuration for joint-position control."""

    use_default_offset: bool = True

    def build(self, env: ManagerBasedRlEnv) -> JointPositionAction:
        return JointPositionAction(self, env)


class JointPositionAction(BaseAction):
    """Convert policy actions into entity joint-position targets."""

    cfg: JointPositionActionCfg

    def __init__(self, cfg: JointPositionActionCfg, env: ManagerBasedRlEnv):
        if not isinstance(cfg.use_default_offset, bool):
            raise TypeError("JointPositionActionCfg use_default_offset must be bool")
        super().__init__(cfg=cfg, env=env)
        if cfg.use_default_offset:
            self._offset = self._entity.data.default_joint_pos[:, self._target_ids].copy()
        self._target = np.empty_like(self._processed_actions)

    def apply_actions(self) -> None:
        encoder_bias = self._entity.data.encoder_bias[:, self._target_ids]
        np.subtract(self._processed_actions, encoder_bias, out=self._target)
        self._entity.set_joint_position_target(self._target, joint_ids=self._target_ids)


@dataclass(kw_only=True)
class RelativeJointPositionActionCfg(BaseActionCfg):
    """Joint position targets relative to the current measured position.

    ``target = current_joint_pos + action * scale``.  A fixed offset has no
    useful meaning for this transmission and is rejected during construction.
    """

    def __post_init__(self) -> None:
        if isinstance(self.offset, dict):
            resolved = [
                _real(value, label="RelativeJointPositionActionCfg offset")
                for value in self.offset.values()
            ]
            nonzero = [value for value in resolved if value != 0.0]
        else:
            value = _real(self.offset, label="RelativeJointPositionActionCfg offset")
            nonzero = [value] if value != 0.0 else []
        if nonzero:
            raise ValueError("RelativeJointPositionActionCfg does not support a non-zero offset")

    def build(self, env: ManagerBasedRlEnv) -> RelativeJointPositionAction:
        return RelativeJointPositionAction(self, env)


class RelativeJointPositionAction(BaseAction):
    """Control joints via position targets relative to current positions."""

    def apply_actions(self) -> None:
        current = self._entity.data.joint_pos[:, self._target_ids]
        target = current + self._processed_actions
        self._entity.set_joint_position_target(target, joint_ids=self._target_ids)


@dataclass(kw_only=True)
class JointVelocityActionCfg(BaseActionCfg):
    """Configuration for joint velocity control."""

    use_default_offset: bool = True

    def build(self, env: ManagerBasedRlEnv) -> JointVelocityAction:
        return JointVelocityAction(self, env)


class JointVelocityAction(BaseAction):
    """Control joints via velocity targets."""

    def __init__(self, cfg: JointVelocityActionCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg=cfg, env=env)
        if not isinstance(cfg.use_default_offset, bool):
            raise TypeError("JointVelocityActionCfg use_default_offset must be bool")
        if cfg.use_default_offset:
            self._offset = self._entity.data.default_joint_vel[:, self._target_ids].copy()

    def apply_actions(self) -> None:
        self._entity.set_joint_velocity_target(
            self._processed_actions,
            joint_ids=self._target_ids,
        )


@dataclass(kw_only=True)
class JointEffortActionCfg(BaseActionCfg):
    """Configuration for joint effort (torque) control."""

    def build(self, env: ManagerBasedRlEnv) -> JointEffortAction:
        return JointEffortAction(self, env)


class JointEffortAction(BaseAction):
    """Control joints via effort targets."""

    def apply_actions(self) -> None:
        self._entity.set_joint_effort_target(
            self._processed_actions,
            joint_ids=self._target_ids,
        )


__all__ = [
    "JointEffortAction",
    "JointEffortActionCfg",
    "JointPositionAction",
    "JointPositionActionCfg",
    "JointVelocityAction",
    "JointVelocityActionCfg",
    "RelativeJointPositionAction",
    "RelativeJointPositionActionCfg",
]
