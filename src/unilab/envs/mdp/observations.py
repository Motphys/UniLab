# Derived from mujocolab/mjlab v1.6.0 (0fb8a681),
# src/mjlab/envs/mdp/observations.py.
# Copyright 2025, The mjlab Developers.
# Modified by UniLab for NumPy and the base-owned entity facade; Apache-2.0.
"""Community-style observation terms for the NumPy manager runtime."""

from __future__ import annotations

import weakref
from typing import TYPE_CHECKING, cast

import numpy as np
import torch

from unilab.managers.manager_base import ManagerTermBase, ManagerTermBaseCfg
from unilab.managers.scene_entity_config import SceneEntityCfg
from unilab.managers.torch_rng import TorchManagerRng
from unilab.utils.rotation import np_quat_apply_batched, np_quat_from_angle_axis

if TYPE_CHECKING:
    from unilab.base.entity import Entity
    from unilab.managers._types import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


class _NamedSensorObservation(ManagerTermBase):
    """Cold-path binding shared by pinned named-sensor observation terms."""

    _term_name = "named_sensor"
    _tensor_sensor_names: tuple[str, ...] = ()

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        sensor_name = cfg.params.get("sensor_name")
        if not isinstance(sensor_name, str) or not sensor_name:
            raise ValueError(
                f"Observation term '{self._term_name}' capability 'named sensor' "
                "requires a non-empty sensor_name"
            )
        self._sensor_name = sensor_name
        self._entity_name = self._resolve_entity_name(cfg)
        self._tensor_sensor_names = (sensor_name,)
        try:
            self._view = env.scene.bind_sensor_data((sensor_name,))
        except (KeyError, TypeError, ValueError, NotImplementedError) as exc:
            raise type(exc)(
                f"Observation term '{self._term_name}' capability 'named sensor "
                f"{sensor_name}' could not be materialized: {exc}"
            ) from exc

    def _resolve_entity_name(self, cfg: ManagerTermBaseCfg) -> str:
        entity_name = cfg.params.get("entity_name")
        if not isinstance(entity_name, str) or not entity_name:
            return "robot"
        return entity_name

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

    def _read_tensor(self, env: ManagerBasedRlEnv) -> torch.Tensor:
        read_plan = getattr(env.scene, "_tensor_read_plan", None)
        if read_plan is None:
            device = getattr(env, "device", torch.device("cpu"))
            return torch.as_tensor(self._read(), device=device)
        entity = env.scene[self._entity_name]
        sensor_names = self._tensor_sensor_names
        compiled = set(read_plan.sensor_names.get(self._entity_name, ()))
        if read_plan.device.type == "cuda" and not set(sensor_names).issubset(compiled):
            # Narrow device-resident adapters still expose only their audited
            # scalar sensor subset. Keep the validated backend host/cache view
            # rather than fabricating a packed packet entry.
            device = getattr(env, "device", torch.device("cpu"))
            return torch.as_tensor(self._read(), device=device)
        views = read_plan.sensor_tensor_views(entity, sensor_names)
        return views.values[self._sensor_name]

    @property
    def tensor_sensor_names(self) -> tuple[str, ...]:
        return self._tensor_sensor_names


class builtin_sensor(_NamedSensorObservation):
    """Read one existing backend sensor through a cached NumPy view."""

    _term_name = "builtin_sensor"

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        sensor_name: str,
        entity_name: str | None = None,
    ) -> np.ndarray:
        del entity_name
        self._validate_call_name(sensor_name)
        return self._read_tensor(env)


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

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        sensor_name: str,
        entity_name: str | None = None,
    ) -> np.ndarray:
        del entity_name
        self._validate_call_name(sensor_name)
        return -self._read_tensor(env)


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


# Per-env constant IMU mounting-misalignment quaternions, keyed by env instance
# and max_angle_rad. Sampled once per (env, angle) at term construction and
# shared by every misaligned term of that env, so the gyroscope and gravity
# observations see the SAME mounting error (legacy microduck recipe,
# microduck_rl/src/mjlab_microduck/tasks/mdp.py:3618-3660). Weak keys keep the
# cache from outliving the env.
_IMU_MISALIGNMENT_QUATS: weakref.WeakKeyDictionary[ManagerBasedRlEnv, dict[float, np.ndarray]] = (
    weakref.WeakKeyDictionary()
)


def _imu_misalignment_quat(env: ManagerBasedRlEnv, max_angle_rad: float) -> np.ndarray:
    """Per-env constant IMU mounting-misalignment rotation, sampled once.

    Models a fixed small mounting/calibration error of the IMU on each robot:
    a rotation about an axis uniform on the sphere with magnitude uniform in
    [0, max_angle_rad]. Sampled from ``env.torch_rng`` on first use and cached for the
    whole run (a startup-style systematic per-robot bias, not per-step noise and
    not resampled on episode reset), matching the legacy semantics.

    Returns (num_envs, 4) unit quaternions (w, x, y, z) in float32.
    """
    per_env = _IMU_MISALIGNMENT_QUATS.get(env)
    if per_env is None:
        per_env = {}
        _IMU_MISALIGNMENT_QUATS[env] = per_env
    quat = per_env.get(max_angle_rad)
    if quat is None:
        rng_owner = getattr(env, "torch_rng", None)
        if not isinstance(rng_owner, TorchManagerRng):
            raise RuntimeError(
                "IMU misalignment sampling requires the Manager-owned Torch generator"
            )
        axis = rng_owner.standard_normal((env.num_envs, 3))
        angle = rng_owner.uniform(0.0, max_angle_rad, (env.num_envs,), dtype=torch.float32)
        axis = axis / torch.linalg.vector_norm(axis, dim=-1, keepdim=True)
        half = 0.5 * angle
        quat_torch = torch.cat((torch.cos(half)[:, None], axis * torch.sin(half)[:, None]), dim=1)
        quat = quat_torch.detach().cpu().numpy().astype(np.float32, copy=False)
        per_env[max_angle_rad] = quat
    return quat


class _ImuMisalignedObservation(ManagerTermBase):
    """Cold-path binding shared by IMU mounting-misalignment observation terms."""

    _term_name = "imu_misaligned"

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        max_angle_deg = cfg.params.get("max_angle_deg", 1.0)
        if isinstance(max_angle_deg, bool) or not isinstance(max_angle_deg, (int, float)):
            raise ValueError(
                f"Observation term '{self._term_name}' capability 'IMU misalignment' "
                f"requires a real max_angle_deg, got {max_angle_deg!r}"
            )
        max_angle_deg = float(max_angle_deg)
        if not np.isfinite(max_angle_deg) or max_angle_deg < 0.0:
            raise ValueError(
                f"Observation term '{self._term_name}' capability 'IMU misalignment' "
                f"requires a finite non-negative max_angle_deg, got {max_angle_deg}"
            )
        self._max_angle_deg = max_angle_deg
        self._quat = _imu_misalignment_quat(env, float(np.deg2rad(max_angle_deg)))

    def _validate_max_angle(self, max_angle_deg: float) -> None:
        if float(max_angle_deg) != self._max_angle_deg:
            raise ValueError(
                f"Observation term '{self._term_name}' was bound to max_angle_deg "
                f"{self._max_angle_deg}, received {max_angle_deg}"
            )

    def _rotate(self, values: np.ndarray) -> np.ndarray:
        return np_quat_apply_batched(self._quat, values)


class base_ang_vel_imu_misaligned(_ImuMisalignedObservation):
    """Base angular velocity rotated by the per-env constant IMU misalignment."""

    _term_name = "base_ang_vel_imu_misaligned"

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        max_angle_deg: float = 1.0,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    ) -> np.ndarray:
        self._validate_max_angle(max_angle_deg)
        asset = cast("Entity", env.scene[asset_cfg.name])
        return self._rotate(asset.data.root_link_ang_vel_b)


class projected_gravity_imu_misaligned(_ImuMisalignedObservation):
    """Projected gravity rotated by the SAME per-env IMU misalignment as the gyro."""

    _term_name = "projected_gravity_imu_misaligned"

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        max_angle_deg: float = 1.0,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    ) -> np.ndarray:
        self._validate_max_angle(max_angle_deg)
        asset = cast("Entity", env.scene[asset_cfg.name])
        return self._rotate(asset.data.projected_gravity_b)


def joint_pos_rel(
    env: ManagerBasedRlEnv,
    biased: bool = False,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> np.ndarray | torch.Tensor:
    if not isinstance(biased, bool):
        raise TypeError(f"joint_pos_rel biased must be bool, got {type(biased).__name__}")
    asset = cast("Entity", env.scene[asset_cfg.name])
    joint_ids = asset_cfg.joint_ids
    read_plan = getattr(env.scene, "_tensor_read_plan", None)
    if read_plan is not None:
        state = read_plan.joint_tensor_view(asset)
        return (
            state.joint_pos[:, joint_ids]
            - asset.data.default_joint_pos_torch(env.device)[:, joint_ids]
        )
    joint_pos = asset.data.joint_pos_biased if biased else asset.data.joint_pos
    return joint_pos[:, joint_ids] - asset.data.default_joint_pos[:, joint_ids]


def joint_vel_rel(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> np.ndarray | torch.Tensor:
    asset = cast("Entity", env.scene[asset_cfg.name])
    joint_ids = asset_cfg.joint_ids
    read_plan = getattr(env.scene, "_tensor_read_plan", None)
    if read_plan is not None:
        state = read_plan.joint_tensor_view(asset)
        return (
            state.joint_vel[:, joint_ids]
            - asset.data.default_joint_vel_torch(env.device)[:, joint_ids]
        )
    return asset.data.joint_vel[:, joint_ids] - asset.data.default_joint_vel[:, joint_ids]


def last_action(
    env: ManagerBasedRlEnv, action_name: str | None = None
) -> np.ndarray | torch.Tensor:
    if action_name is None:
        return env.action_manager.action
    try:
        action = env.action_manager.get_term(action_name).raw_action
    except KeyError as exc:
        raise KeyError(f"Action term '{action_name}' not found") from exc
    return action


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
    "base_ang_vel_imu_misaligned",
    "base_lin_vel",
    "builtin_sensor",
    "generated_commands",
    "joint_pos_rel",
    "joint_vel_rel",
    "last_action",
    "projected_gravity",
    "projected_gravity_from_sensor",
    "projected_gravity_imu_misaligned",
]
