# Derived from mujocolab/mjlab v1.6.0 (0fb8a681),
# src/mjlab/envs/mdp/observations.py.
# Copyright 2025, The mjlab Developers.
# Modified by UniLab for NumPy and the base-owned entity facade; Apache-2.0.
"""Community-style observation terms for the NumPy manager runtime."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

from unilab.managers.manager_base import ManagerTermBase, ManagerTermBaseCfg
from unilab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from unilab.base.entity import Entity
    from unilab.managers._types import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


class _NamedSensorObservation(ManagerTermBase):
    """Cold-path binding shared by pinned named-sensor observation terms."""

    _term_name = "named_sensor"

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        sensor_name = cfg.params.get("sensor_name")
        if not isinstance(sensor_name, str) or not sensor_name:
            raise ValueError(
                f"Observation term '{self._term_name}' capability 'named sensor' "
                "requires a non-empty sensor_name"
            )
        self._sensor_name = sensor_name
        try:
            self._view = env.scene.bind_sensor_data((sensor_name,))
        except (KeyError, TypeError, ValueError, NotImplementedError) as exc:
            raise type(exc)(
                f"Observation term '{self._term_name}' capability 'named sensor "
                f"{sensor_name}' could not be materialized: {exc}"
            ) from exc

    def _validate_call_name(self, sensor_name: str) -> None:
        if sensor_name != self._sensor_name:
            raise ValueError(
                f"Observation term '{self._term_name}' was bound to sensor "
                f"'{self._sensor_name}', received '{sensor_name}'"
            )

    def _read(self) -> np.ndarray:
        try:
            return self._view.read()
        except (KeyError, TypeError, ValueError, NotImplementedError) as exc:
            raise type(exc)(
                f"Observation term '{self._term_name}' capability 'named sensor "
                f"{self._sensor_name}' failed on backend '{self._view.backend_type}': {exc}"
            ) from exc


class builtin_sensor(_NamedSensorObservation):
    """Read one existing backend sensor through a cached NumPy view."""

    _term_name = "builtin_sensor"

    def __call__(self, env: ManagerBasedRlEnv, sensor_name: str) -> np.ndarray:
        del env
        self._validate_call_name(sensor_name)
        return self._read()


class projected_gravity_from_sensor(_NamedSensorObservation):
    """Negate a cached 3-D up-vector sensor to obtain projected gravity."""

    _term_name = "projected_gravity_from_sensor"

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        if self._view.dimensions != (3,):
            raise ValueError(
                "Observation term 'projected_gravity_from_sensor' capability "
                f"'3-D named sensor {self._sensor_name}' received dimensions "
                f"{self._view.dimensions} on backend '{self._view.backend_type}'"
            )

    def __call__(self, env: ManagerBasedRlEnv, sensor_name: str) -> np.ndarray:
        del env
        self._validate_call_name(sensor_name)
        return -self._read()


def base_lin_vel(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> np.ndarray:
    asset = cast("Entity", env.scene[asset_cfg.name])
    return asset.data.root_link_lin_vel_b


def base_ang_vel(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> np.ndarray:
    asset = cast("Entity", env.scene[asset_cfg.name])
    return asset.data.root_link_ang_vel_b


def projected_gravity(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> np.ndarray:
    asset = cast("Entity", env.scene[asset_cfg.name])
    return asset.data.projected_gravity_b


def joint_pos_rel(
    env: ManagerBasedRlEnv,
    biased: bool = False,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> np.ndarray:
    if not isinstance(biased, bool):
        raise TypeError(f"joint_pos_rel biased must be bool, got {type(biased).__name__}")
    asset = cast("Entity", env.scene[asset_cfg.name])
    joint_ids = asset_cfg.joint_ids
    joint_pos = asset.data.joint_pos_biased if biased else asset.data.joint_pos
    return joint_pos[:, joint_ids] - asset.data.default_joint_pos[:, joint_ids]


def joint_vel_rel(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> np.ndarray:
    asset = cast("Entity", env.scene[asset_cfg.name])
    joint_ids = asset_cfg.joint_ids
    return asset.data.joint_vel[:, joint_ids] - asset.data.default_joint_vel[:, joint_ids]


def last_action(env: ManagerBasedRlEnv, action_name: str | None = None) -> np.ndarray:
    if action_name is None:
        return env.action_manager.action
    try:
        return env.action_manager.get_term(action_name).raw_action
    except KeyError as exc:
        raise KeyError(f"Action term '{action_name}' not found") from exc


def generated_commands(env: ManagerBasedRlEnv, command_name: str) -> np.ndarray:
    try:
        command = env.command_manager.get_command(command_name)
    except KeyError as exc:
        raise KeyError(f"Command term '{command_name}' not found") from exc
    if command is None:
        raise KeyError(f"Command term '{command_name}' not found")
    return command


__all__ = [
    "base_ang_vel",
    "base_lin_vel",
    "builtin_sensor",
    "generated_commands",
    "joint_pos_rel",
    "joint_vel_rel",
    "last_action",
    "projected_gravity",
    "projected_gravity_from_sensor",
]
