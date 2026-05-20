from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

import gymnasium as gym
import numpy as np

from unilab.base.backend import SimBackend
from unilab.base.base import EnvCfg
from unilab.base.np_env import NpEnv, NpEnvState
from unilab.dtype_config import get_global_dtype
from unilab.envs.locomotion.common.terrain_spawn import BaseSpawnManager


@dataclass
class Sensor:
    local_linvel = "local_linvel"
    gyro = "gyro"


@dataclass
class ControlConfigBase:
    action_scale: float = 0.25
    simulate_action_latency: bool = False


@dataclass
class BaseNoiseConfig:
    level: float = 0.0
    scale_joint_angle: float = 0.03
    scale_joint_vel: float = 0.5
    scale_gyro: float = 0.2
    scale_gravity: float = 0.05
    scale_linvel: float = 0.1


@dataclass
class LocomotionBaseCfg(EnvCfg):
    control_config: ControlConfigBase = field(default_factory=ControlConfigBase)
    noise_config: BaseNoiseConfig = field(default_factory=BaseNoiseConfig)
    sensor: Sensor = field(default_factory=Sensor)
    sim_dt: float = 0.01
    ctrl_dt: float = 0.02


class LocomotionBaseEnv(NpEnv):
    """Common base environment for locomotion tasks (G1, Go1, Go2, etc.)."""

    _cfg: LocomotionBaseCfg

    _keyframe_name: ClassVar[str] = "home"
    _use_global_dtype: ClassVar[bool] = True

    def __init__(self, cfg: LocomotionBaseCfg, backend: SimBackend, num_envs: int = 1):
        super().__init__(cfg, backend, num_envs)
        self._init_action_space()
        self._num_action = self._action_space.shape[0]
        self._init_buffers()
        self._spawn: BaseSpawnManager = BaseSpawnManager()

    def _init_action_space(self) -> None:
        ctrl_range = self._backend.get_actuator_ctrl_range()
        nu = self._backend.num_actuators
        self._action_space = gym.spaces.Box(ctrl_range[:, 0], ctrl_range[:, 1], (nu,), dtype=float)  # type: ignore[assignment, arg-type]

    @property
    def action_space(self) -> gym.spaces.Box:
        return self._action_space  # type: ignore[no-any-return]

    def _init_buffers(self) -> None:
        dtype = get_global_dtype() if self._use_global_dtype else np.float32
        raw_qpos = self._backend.get_keyframe_qpos(self._keyframe_name)
        self._init_qpos = (
            np.asarray(raw_qpos, dtype=dtype) if self._use_global_dtype else np.asarray(raw_qpos)
        )
        self.default_angles = np.asarray(self._init_qpos[-self._num_action :], dtype=dtype)
        raw_qvel = self._backend.get_init_qvel()
        self._init_qvel = (
            np.asarray(raw_qvel, dtype=dtype) if self._use_global_dtype else np.asarray(raw_qvel)
        )

    def apply_action(self, actions: np.ndarray, state: NpEnvState) -> np.ndarray:
        state.info["last_actions"] = state.info.get("current_actions", np.zeros_like(actions))
        state.info["current_actions"] = actions
        exec_actions = (
            state.info["last_actions"]
            if self._cfg.control_config.simulate_action_latency
            else actions
        )
        ctrl: np.ndarray = (
            exec_actions * self._cfg.control_config.action_scale + self.default_angles
        )
        return ctrl

    def _obs_noise(self, data: np.ndarray, scale: float) -> np.ndarray:
        """Apply per-step uniform observation noise scaled by ``noise_config.level``."""
        level = float(self._cfg.noise_config.level)
        if level <= 0.0:
            return data
        noise = np.random.uniform(-1.0, 1.0, data.shape).astype(data.dtype) * level * scale
        return data + noise

    def get_local_linvel(self) -> np.ndarray:
        local_linvel: np.ndarray = self._backend.get_sensor_data(self._cfg.sensor.local_linvel)
        return local_linvel

    def get_gyro(self) -> np.ndarray:
        gyro: np.ndarray = self._backend.get_sensor_data(self._cfg.sensor.gyro)
        return gyro

    def get_dof_pos(self) -> np.ndarray:
        dof_pos: np.ndarray = self._backend.get_dof_pos()
        return dof_pos

    def get_dof_vel(self) -> np.ndarray:
        dof_vel: np.ndarray = self._backend.get_dof_vel()
        return dof_vel
