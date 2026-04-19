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
    level: float = 1.0
    scale_joint_angle: float = 0.01
    scale_joint_vel: float = 1.5
    scale_gyro: float = 0.2
    scale_gravity: float = 0.05
    scale_linvel: float = 0.1


@dataclass
class ControlConfig(ControlConfigBase):
    action_scale: float | np.ndarray = 0.25  # type: ignore[assignment]


@dataclass
class Asset:
    base_name = "pelvis"
    foot_name = "ankle_roll_link"
    ground = "floor"


@dataclass
class G1BaseCfg(LocomotionBaseCfg):
    noise_config: NoiseConfig = field(default_factory=NoiseConfig)
    control_config: ControlConfig = field(default_factory=ControlConfig)  # type: ignore[assignment]
    asset: Asset = field(default_factory=Asset)
    sim_dt: float = 0.02 / 3.0
    ctrl_dt: float = 0.02


class G1BaseEnv(LocomotionBaseEnv):
    _cfg: G1BaseCfg
    _keyframe_name = "stand"
    _use_global_dtype = False

    def _obs_noise(self, data: np.ndarray, scale: float) -> np.ndarray:
        """Apply per-step uniform observation noise scaled by ``noise_config.level``."""
        noise_cfg = self._cfg.noise_config
        if noise_cfg.level > 0.0:
            return np.asarray(
                data
                + (
                    np.random.uniform(-1.0, 1.0, data.shape).astype(data.dtype)
                    * noise_cfg.level
                    * scale
                ),
                dtype=data.dtype,
            )
        return np.asarray(data)
