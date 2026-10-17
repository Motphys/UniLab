"""Sensor-driven reward terms shared by Manager-Based locomotion families.

Biped and quadruped locomotion families port legacy reward equations that read
IMU-style XML sensors by name.  This module owns those equations once, on top
of the ``SensorTermBase`` cold-path binding contract from ``manager_terms.py``:
the robot's sensor name is a cfg parameter (``params={"sensor_name": ...}``),
so each family binds its own sensors from its owner YAML instead of
subclassing.  Plain action/state penalties and the shared term bases live in
``manager_terms.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import torch

from unilab.managers.manager_base import ManagerTermBase, ManagerTermBaseCfg
from unilab.managers.scene_entity_config import SceneEntityCfg

from .manager_terms import SensorTermBase, _command, _real

if TYPE_CHECKING:
    from unilab.managers._types import ManagerBasedRlEnv


def _torch_values(values: torch.Tensor) -> torch.Tensor:
    return values.to(dtype=torch.float32)


def _quat_apply_inverse(quat_wxyz: torch.Tensor, vector_w: torch.Tensor) -> torch.Tensor:
    quat = torch.cat((quat_wxyz[..., 0:1], -quat_wxyz[..., 1:4]), dim=-1)
    quat_xyz = quat[..., 1:4]
    cross = torch.linalg.cross(quat_xyz, vector_w, dim=-1)
    second_cross = torch.linalg.cross(quat_xyz, cross, dim=-1)
    return vector_w + 2.0 * quat[..., 0:1] * cross + 2.0 * second_cross


class _Vec3SensorTerm(SensorTermBase):
    """Bind one named vec3 sensor on the cold path and read it per step."""

    _allowed_params: ClassVar[frozenset[str]] = frozenset({"sensor_name"})

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        sensor_name = cfg.params.get("sensor_name")
        if not isinstance(sensor_name, str) or not sensor_name:
            raise ValueError(f"{self.name} sensor_name must be a non-empty string")
        self._sensor_name = sensor_name
        self._sensor = self._bind((sensor_name,))
        if self._sensor.dimensions != (3,):
            raise ValueError(
                f"{self.name} sensor '{sensor_name}' must expose 3 values; received "
                f"{self._sensor.dimensions} on backend '{self._sensor.backend_type}'"
            )

    @property
    def tensor_sensor_names(self) -> tuple[str, ...]:
        return (self._sensor_name,)

    def _read_sensor_tensor(self, env: ManagerBasedRlEnv) -> torch.Tensor:
        read_plan = getattr(env.scene, "_tensor_read_plan", None)
        owner_sensors = set(getattr(read_plan, "sensor_names", {}).get("robot", ()))
        if read_plan is None or not set(self.tensor_sensor_names).issubset(owner_sensors):
            raise NotImplementedError(
                f"Manager term '{self.name}' requires its named sensors "
                f"{self.tensor_sensor_names} in the packed Torch read plan"
            )
        entity = env.scene["robot"]
        values = read_plan.sensor_tensor_views(entity, self.tensor_sensor_names).values[
            self._sensor_name
        ]
        if tuple(values.shape) != (env.num_envs, 3):
            raise ValueError(
                f"{self.name} sensor '{self._sensor_name}' must have shape "
                f"(num_envs, 3); got {tuple(values.shape)}"
            )
        return values


class _CommandTrackingTerm(_Vec3SensorTerm):
    """``exp(-error / tracking_sigma)`` tracking against a command channel."""

    _allowed_params = frozenset({"sensor_name", "tracking_sigma", "command_name"})

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self._sigma = _real(
            self.name,
            "tracking_sigma",
            cfg.params.get("tracking_sigma", 0.25),
            minimum=0.0,
            strict_minimum=True,
        )
        self._command_name = cfg.params.get("command_name", "twist")
        _command(env, self.name, self._command_name)  # Fail closed at construction.


class track_lin_vel(_CommandTrackingTerm):
    """Exponential reward for tracking commanded xy linear velocity."""

    def __call__(self, env: ManagerBasedRlEnv, **params: Any) -> torch.Tensor:
        del params
        linvel = self._read_sensor_tensor(env)
        command = _command(env, self.name, self._command_name)
        linvel_tensor = _torch_values(linvel)
        command_tensor = command.to(device=linvel_tensor.device, dtype=linvel_tensor.dtype)
        error = (command_tensor[:, :2] - linvel_tensor[:, :2]).square().sum(dim=1)
        return torch.exp(-error / self._sigma)


class track_ang_vel(_CommandTrackingTerm):
    """Exponential reward for tracking commanded yaw angular velocity."""

    def __call__(self, env: ManagerBasedRlEnv, **params: Any) -> torch.Tensor:
        del params
        gyro = self._read_sensor_tensor(env)
        command = _command(env, self.name, self._command_name)
        gyro_tensor = _torch_values(gyro)
        command_tensor = command.to(device=gyro_tensor.device, dtype=gyro_tensor.dtype)
        error = (command_tensor[:, 2] - gyro_tensor[:, 2]).square()
        return torch.exp(-error / self._sigma)


class _RootBodyVelocityTerm(ManagerTermBase):
    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        asset_cfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        if not isinstance(asset_cfg, SceneEntityCfg):
            raise TypeError(f"{self.name} asset_cfg must be a SceneEntityCfg")
        entity_cfg = env.cfg.scene.entities.get(asset_cfg.name) if env.cfg.scene else None
        root = getattr(entity_cfg, "root_body_name", None) if entity_cfg is not None else None
        if not isinstance(root, str) or not root:
            raise ValueError(f"{self.name} requires an entity root_body_name")
        self._entity_name = asset_cfg.name
        self._root_body_name = root

    @property
    def tensor_body_names(self) -> tuple[str, ...]:
        return (self._root_body_name,)

    @property
    def entity_name(self) -> str:
        return self._entity_name

    def _root_velocity_b(self, env: ManagerBasedRlEnv) -> torch.Tensor:
        read_plan = getattr(env.scene, "_tensor_read_plan", None)
        if read_plan is None:
            raise NotImplementedError(
                f"Manager term '{self.name}' requires the Manager packed root-body read plan"
            )
        step_counter = getattr(env, "step_counter", None)
        cache = getattr(env, "_root_body_velocity_cache", None)
        if cache is not None and cache[0] == step_counter:
            return cache[1]
        body = read_plan.body_tensor_view(self._entity_name, (self._root_body_name,))
        velocity = _quat_apply_inverse(body.quat_w[:, 0, :], body.lin_vel_w[:, 0, :])
        setattr(env, "_root_body_velocity_cache", (step_counter, velocity))
        return velocity


class root_track_lin_vel(_RootBodyVelocityTerm):
    """Track commanded planar velocity at the root body origin."""

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self._sigma = _real(
            self.name,
            "tracking_sigma",
            cfg.params.get("tracking_sigma", 0.25),
            minimum=0.0,
            strict_minimum=True,
        )
        self._command_name = cfg.params.get("command_name", "twist")
        _command(env, self.name, self._command_name)

    def __call__(self, env: ManagerBasedRlEnv, **params: Any) -> torch.Tensor:
        del params
        linvel = self._root_velocity_b(env)
        command = _command(env, self.name, self._command_name)
        command_tensor = command.to(device=linvel.device, dtype=linvel.dtype)
        error = (command_tensor[:, :2] - linvel[:, :2]).square().sum(dim=1)
        return torch.exp(-error / self._sigma)


class root_lin_vel_z(_RootBodyVelocityTerm):
    """Penalize vertical root-body-origin velocity."""

    def __call__(self, env: ManagerBasedRlEnv, **params: Any) -> torch.Tensor:
        del params
        return _torch_values(self._root_velocity_b(env)[:, 2]).square()


class lin_vel_z(_Vec3SensorTerm):
    """Penalty for vertical (z) linear velocity."""

    def __call__(self, env: ManagerBasedRlEnv, **params: Any) -> torch.Tensor:
        del params
        values = self._read_sensor_tensor(env)[:, 2]
        return _torch_values(values).square()


class ang_vel_xy(_Vec3SensorTerm):
    """Penalty for roll/pitch angular velocity."""

    def __call__(self, env: ManagerBasedRlEnv, **params: Any) -> torch.Tensor:
        del params
        return _torch_values(self._read_sensor_tensor(env)[:, :2]).square().sum(dim=1)


class orientation(_Vec3SensorTerm):
    """Penalty for deviation from upright (horizontal upvector components)."""

    def __call__(self, env: ManagerBasedRlEnv, **params: Any) -> torch.Tensor:
        del params
        upvector = self._read_sensor_tensor(env)
        return _torch_values(upvector[:, :2]).square().sum(dim=1)


__all__ = [
    "ang_vel_xy",
    "lin_vel_z",
    "orientation",
    "root_lin_vel_z",
    "root_track_lin_vel",
    "track_ang_vel",
    "track_lin_vel",
]
