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

import numpy as np

from unilab.dtype_config import get_global_dtype
from unilab.managers.manager_base import ManagerTermBaseCfg

from .manager_terms import SensorTermBase, _command, _real, _state

if TYPE_CHECKING:
    from unilab.managers._types import ManagerBasedRlEnv


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

    def _read_sensor(self, env: ManagerBasedRlEnv) -> np.ndarray:
        return _state(
            self.name,
            f"sensor '{self._sensor_name}'",
            self._read(self._sensor, self.name),
            (env.num_envs, 3),
        )


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

    def __call__(self, env: ManagerBasedRlEnv, **params: Any) -> np.ndarray:
        del params
        linvel = self._read_sensor(env)
        command = _command(env, self.name, self._command_name)
        error = np.sum(np.square(command[:, :2] - linvel[:, :2]), axis=1)
        return np.asarray(np.exp(-error / self._sigma), dtype=get_global_dtype())


class track_ang_vel(_CommandTrackingTerm):
    """Exponential reward for tracking commanded yaw angular velocity."""

    def __call__(self, env: ManagerBasedRlEnv, **params: Any) -> np.ndarray:
        del params
        gyro = self._read_sensor(env)
        command = _command(env, self.name, self._command_name)
        error = np.square(command[:, 2] - gyro[:, 2])
        return np.asarray(np.exp(-error / self._sigma), dtype=get_global_dtype())


class lin_vel_z(_Vec3SensorTerm):
    """Penalty for vertical (z) linear velocity."""

    def __call__(self, env: ManagerBasedRlEnv, **params: Any) -> np.ndarray:
        del params
        return np.asarray(np.square(self._read_sensor(env)[:, 2]), dtype=get_global_dtype())


class ang_vel_xy(_Vec3SensorTerm):
    """Penalty for roll/pitch angular velocity."""

    def __call__(self, env: ManagerBasedRlEnv, **params: Any) -> np.ndarray:
        del params
        return np.asarray(
            np.sum(np.square(self._read_sensor(env)[:, :2]), axis=1), dtype=get_global_dtype()
        )


class orientation(_Vec3SensorTerm):
    """Penalty for deviation from upright (horizontal upvector components)."""

    def __call__(self, env: ManagerBasedRlEnv, **params: Any) -> np.ndarray:
        del params
        upvector = self._read_sensor(env)
        return np.asarray(
            np.square(upvector[:, 0]) + np.square(upvector[:, 1]), dtype=get_global_dtype()
        )


__all__ = [
    "ang_vel_xy",
    "lin_vel_z",
    "orientation",
    "track_ang_vel",
    "track_lin_vel",
]
