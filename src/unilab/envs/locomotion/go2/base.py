from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from unilab.envs.locomotion.common.base import (
    ControlConfigBase,
    LocomotionBaseCfg,
    LocomotionBaseEnv,
)


@dataclass
class NoiseConfig:
    level: float = 0.0
    scale_joint_angle: float = 0.03
    scale_joint_vel: float = 0.5
    scale_gyro: float = 0.2
    scale_gravity: float = 0.05
    scale_linvel: float = 0.1


@dataclass
class ControlConfig(ControlConfigBase):
    Kp: float = 35.0
    Kd: float = 0.5


@dataclass
class Asset:
    base_name = "base"
    foot_name = "foot"
    ground = "floor"


@dataclass
class Go2BaseCfg(LocomotionBaseCfg):
    noise_config: NoiseConfig = field(default_factory=NoiseConfig)
    control_config: ControlConfig = field(default_factory=ControlConfig)  # type: ignore[assignment]
    asset: Asset = field(default_factory=Asset)
    sim_dt: float = 0.01
    ctrl_dt: float = 0.02


class Go2BaseEnv(LocomotionBaseEnv):
    _cfg: Go2BaseCfg

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
