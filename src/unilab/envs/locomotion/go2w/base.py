from __future__ import annotations

from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np

from unilab.envs.locomotion.common.base import (
    ControlConfigBase,
    LocomotionBaseCfg,
    LocomotionBaseEnv,
)

LEG_JOINT_SENSOR_PREFIXES: tuple[str, ...] = (
    "FR_hip",
    "FR_thigh",
    "FR_calf",
    "FL_hip",
    "FL_thigh",
    "FL_calf",
    "RR_hip",
    "RR_thigh",
    "RR_calf",
    "RL_hip",
    "RL_thigh",
    "RL_calf",
)
WHEEL_JOINT_SENSOR_PREFIXES: tuple[str, ...] = ("FR_wheel", "FL_wheel", "RR_wheel", "RL_wheel")
JOINT_SENSOR_PREFIXES: tuple[str, ...] = LEG_JOINT_SENSOR_PREFIXES + WHEEL_JOINT_SENSOR_PREFIXES

NUM_LEG_ACTIONS = len(LEG_JOINT_SENSOR_PREFIXES)
NUM_WHEEL_ACTIONS = len(WHEEL_JOINT_SENSOR_PREFIXES)
NUM_GO2W_ACTIONS = len(JOINT_SENSOR_PREFIXES)

DEFAULT_LEG_ANGLES = np.asarray(
    [
        0.0,
        0.8,
        -1.5,
        0.0,
        0.8,
        -1.5,
        0.0,
        0.8,
        -1.5,
        0.0,
        0.8,
        -1.5,
    ],
    dtype=np.float64,
)
DEFAULT_GO2W_ANGLES = np.concatenate(
    [DEFAULT_LEG_ANGLES, np.zeros((NUM_WHEEL_ACTIONS,), dtype=np.float64)]
)


@dataclass
class NoiseConfig:
    level: float = 0.0
    scale_joint_angle: float = 0.03
    scale_joint_vel: float = 0.5
    scale_gyro: float = 0.2
    scale_gravity: float = 0.05
    scale_linvel: float = 0.1
    scale_wheel_vel: float = 0.5


@dataclass
class ControlConfig(ControlConfigBase):
    action_scale: float = 0.25
    wheel_action_scale: float = 10.0
    Kp: float = 35.0
    Kd: float = 0.5
    wheel_Kd: float = 0.5  # noqa: N815 - Hydra config key kept for compatibility.
    clip_actions: float = 1.0


@dataclass
class Asset:
    base_name = "base_link"
    ground = "floor"


@dataclass
class Go2WBaseCfg(LocomotionBaseCfg):
    noise_config: NoiseConfig = field(default_factory=NoiseConfig)
    control_config: ControlConfig = field(default_factory=ControlConfig)  # type: ignore[assignment]
    asset: Asset = field(default_factory=Asset)
    sim_dt: float = 0.005
    ctrl_dt: float = 0.02


def _sensor_scalar(data: np.ndarray) -> np.ndarray:
    return data.reshape(data.shape[0], -1)[:, 0]


def stack_joint_sensors(backend, suffix: str, *, dtype: np.dtype | type) -> np.ndarray:
    values = [
        _sensor_scalar(backend.get_sensor_data(f"{prefix}_{suffix}"))
        for prefix in JOINT_SENSOR_PREFIXES
    ]
    return np.asarray(np.stack(values, axis=1), dtype=dtype)


def compute_go2w_motor_ctrl(
    policy_ctrl: np.ndarray,
    joint_pos: np.ndarray,
    joint_vel: np.ndarray,
    leg_kp: np.ndarray,
    leg_kd: np.ndarray,
    wheel_kd: np.ndarray,
    ctrl_lower: np.ndarray,
    ctrl_upper: np.ndarray,
    out: np.ndarray,
) -> np.ndarray:
    """Convert Go2W owner-level controls into motor actuator torques.

    Hot path: shapes/dtypes are validated by the owning env at init/reset.
    """
    leg_out = out[:, :NUM_LEG_ACTIONS]
    np.subtract(policy_ctrl[:, :NUM_LEG_ACTIONS], joint_pos[:, :NUM_LEG_ACTIONS], out=leg_out)
    np.multiply(leg_out, leg_kp, out=leg_out)
    leg_out -= leg_kd * joint_vel[:, :NUM_LEG_ACTIONS]
    wheel_out = out[:, NUM_LEG_ACTIONS:]
    np.subtract(policy_ctrl[:, NUM_LEG_ACTIONS:], joint_vel[:, NUM_LEG_ACTIONS:], out=wheel_out)
    np.multiply(wheel_out, wheel_kd, out=wheel_out)
    np.clip(out, ctrl_lower, ctrl_upper, out=out)
    return out


class Go2WBaseEnv(LocomotionBaseEnv):
    _cfg: Go2WBaseCfg

    def _init_action_space(self) -> None:
        self._action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(NUM_GO2W_ACTIONS,),
            dtype=np.float32,
        )

    def _init_buffers(self) -> None:
        super()._init_buffers()
        self.default_angles = np.asarray(DEFAULT_GO2W_ANGLES, dtype=self.default_angles.dtype)

    def _obs_noise(self, data: np.ndarray, scale: float) -> np.ndarray:
        noise_cfg = self._cfg.noise_config
        if noise_cfg.level <= 0.0:
            return data
        return data + (
            np.random.uniform(-1.0, 1.0, data.shape).astype(data.dtype) * noise_cfg.level * scale
        )

    def get_dof_pos(self) -> np.ndarray:
        return stack_joint_sensors(self._backend, "pos", dtype=self.default_angles.dtype)

    def get_dof_vel(self) -> np.ndarray:
        return stack_joint_sensors(self._backend, "vel", dtype=self.default_angles.dtype)
