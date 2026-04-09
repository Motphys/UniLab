from __future__ import annotations

from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np

from unilab.base.backend import SimBackend
from unilab.base.base import EnvCfg
from unilab.base.dtype_config import get_global_dtype
from unilab.base.np_env import NpEnv, NpEnvState


@dataclass
class NoiseConfig:
    level: float = 0.0
    scale_joint_angle: float = 0.03
    scale_joint_vel: float = 0.5
    scale_gyro: float = 0.2
    scale_gravity: float = 0.05
    scale_linvel: float = 0.1


@dataclass
class ControlConfig:
    action_scale: float = 0.25
    Kp: float = 35.0
    Kd: float = 0.5
    simulate_action_latency: bool = False


@dataclass
class Asset:
    base_name = "base"
    foot_name = "foot"
    ground = "floor"


@dataclass
class Sensor:
    local_linvel = "local_linvel"
    gyro = "gyro"


@dataclass
class Go2BaseCfg(EnvCfg):
    model_file: str = field(default=str(""))
    noise_config: NoiseConfig = field(default_factory=NoiseConfig)
    control_config: ControlConfig = field(default_factory=ControlConfig)
    asset: Asset = field(default_factory=Asset)
    sensor: Sensor = field(default_factory=Sensor)
    sim_dt: float = 0.01
    ctrl_dt: float = 0.02


class Go2BaseEnv(NpEnv):
    _cfg: Go2BaseCfg

    def __init__(self, cfg: Go2BaseCfg, backend: SimBackend, num_envs=1):
        super().__init__(cfg, backend, num_envs)

        self._init_action_space()
        self._num_action = self._action_space.shape[0]
        self._init_buffers()

    def _init_action_space(self):
        ctrl_range = self._backend.get_actuator_ctrl_range()
        nu = self._backend.num_actuators
        self._action_space = gym.spaces.Box(ctrl_range[:, 0], ctrl_range[:, 1], (nu,), dtype=float)  # type: ignore[assignment]

    @property
    def action_space(self) -> gym.spaces.Box:
        return self._action_space  # type: ignore[no-any-return]

    def _init_buffers(self):
        dtype = get_global_dtype()
        self.default_angles = np.zeros((self._num_action,), dtype=np.float32)
        self._init_qpos = np.array(self._backend.get_keyframe_qpos("home"), dtype=dtype)
        self.default_angles = self._init_qpos[7:]
        self._init_qvel = self._backend.get_init_qvel().astype(dtype)

    def apply_action(self, actions: np.ndarray, state: NpEnvState) -> np.ndarray:
        state.info["last_actions"] = state.info.get("current_actions", actions.copy())
        state.info["current_actions"] = actions

        exec_actions = (
            state.info["last_actions"]
            if self._cfg.control_config.simulate_action_latency
            else actions
        )
        return np.asarray(
            exec_actions * self._cfg.control_config.action_scale + self.default_angles
        )

    def get_local_linvel(self) -> np.ndarray:
        return np.asarray(self._backend.get_sensor_data(self._cfg.sensor.local_linvel))

    def get_gyro(self) -> np.ndarray:
        return np.asarray(self._backend.get_sensor_data(self._cfg.sensor.gyro))

    def get_dof_pos(self) -> np.ndarray:
        return np.asarray(self._backend.get_dof_pos())

    def get_dof_vel(self) -> np.ndarray:
        return np.asarray(self._backend.get_dof_vel())

    def get_foot_pos(self) -> np.ndarray:
        """Get foot positions. Returns shape (num_envs, 4, 3)"""
        foot_names = ["FL_pos", "FR_pos", "RL_pos", "RR_pos"]
        foot_pos = [self._backend.get_sensor_data(name) for name in foot_names]
        return np.stack(foot_pos, axis=1)

    def get_foot_contact(self) -> np.ndarray:
        """Get foot contact forces. Returns shape (num_envs, 4)"""
        contact_names = ["FL_foot_contact", "FR_foot_contact", "RL_foot_contact", "RR_foot_contact"]
        contacts = [self._backend.get_sensor_data(name)[:, 0] for name in contact_names]
        return np.stack(contacts, axis=1)
