from __future__ import annotations

from dataclasses import dataclass, field

from unilab.envs.locomotion.common.base import (
    BaseNoiseConfig,
    LocomotionBaseCfg,
    LocomotionBaseEnv,
    PdControlConfig,
)


@dataclass
class NoiseConfig(BaseNoiseConfig):
    pass


@dataclass
class ControlConfig(PdControlConfig):
    pass


@dataclass
class Asset:
    base_name = "trunk"
    foot_name = "foot"
    ground = "floor"


@dataclass
class Go1BaseCfg(LocomotionBaseCfg):
    noise_config: NoiseConfig = field(default_factory=NoiseConfig)  # type: ignore[assignment]
    control_config: ControlConfig = field(default_factory=ControlConfig)  # type: ignore[assignment]
    asset: Asset = field(default_factory=Asset)
    sim_dt: float = 0.01
    ctrl_dt: float = 0.02


class Go1BaseEnv(LocomotionBaseEnv):
    _cfg: Go1BaseCfg
