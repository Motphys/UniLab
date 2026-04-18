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
    base_name = "trunk"
    foot_name = "foot"
    ground = "floor"


@dataclass
class Go1BaseCfg(LocomotionBaseCfg):
    noise_config: NoiseConfig = field(default_factory=NoiseConfig)
    control_config: ControlConfig = field(default_factory=ControlConfig)  # type: ignore[assignment]
    asset: Asset = field(default_factory=Asset)
    sim_dt: float = 0.01
    ctrl_dt: float = 0.02


class Go1BaseEnv(LocomotionBaseEnv):
    _cfg: Go1BaseCfg

    def _obs_noise(self, data: np.ndarray, scale: float) -> np.ndarray:
        """Apply per-step uniform observation noise scaled by ``noise_config.level``."""
        noise_cfg = self._cfg.noise_config
        if noise_cfg.level > 0.0:
            return data + (
                np.random.uniform(-1.0, 1.0, data.shape).astype(data.dtype)
                * noise_cfg.level
                * scale
            )
        return data
