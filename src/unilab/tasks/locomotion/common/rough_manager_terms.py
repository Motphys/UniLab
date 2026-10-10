"""Manager-Based terrain terms shared by the production rough quadrupeds."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from numbers import Real
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from unisim.terrain.generator import SubTerrainCfg, TerrainGeneratorCfg

from unilab.envs.mdp.actions.actions import JointPositionAction, JointPositionActionCfg
from unilab.envs.mdp.commands.velocity_command import (
    UniformVelocityCommand,
    UniformVelocityCommandCfg,
)
from unilab.terrains import (
    flat,
    hf_pyramid_slope,
    hf_pyramid_slope_inv,
    pyramid_stairs,
    pyramid_stairs_inv,
    random_rough,
    wave_terrain,
)

if TYPE_CHECKING:
    from unilab.managers._types import ManagerBasedRlEnv


def _rough_sub_terrains() -> dict[str, SubTerrainCfg]:
    return {
        "flat": flat(proportion=0.0),
        "pyramid_stairs": pyramid_stairs(
            proportion=0.1,
            step_height_range=(0.025, 0.10),
            step_width=0.4,
            platform_width=3.0,
            border_width=0.2,
        ),
        "pyramid_stairs_inv": pyramid_stairs_inv(
            proportion=0.1,
            step_height_range=(0.025, 0.10),
            step_width=0.4,
            platform_width=3.0,
            border_width=0.2,
        ),
        "hf_pyramid_slope": hf_pyramid_slope(
            proportion=0.2,
            slope_range=(0.0, 0.3),
            platform_width=2.0,
            border_width=0.2,
        ),
        "hf_pyramid_slope_inv": hf_pyramid_slope_inv(
            proportion=0.2,
            slope_range=(0.0, 0.3),
            platform_width=2.0,
            border_width=0.2,
        ),
        "random_rough": random_rough(
            proportion=0.3,
            noise_range=(0.01, 0.06),
            noise_step=0.01,
            border_width=0.2,
        ),
        "wave_terrain": wave_terrain(
            proportion=0.3,
            amplitude_range=(0.0, 0.12),
            num_waves=4,
            border_width=0.2,
        ),
    }


@dataclass(kw_only=True)
class QuadrupedRoughTerrainCfg(TerrainGeneratorCfg):
    """Shared seven-terrain production generator for quadruped rough owners."""

    seed: int | None = 42
    curriculum: bool = False
    size: tuple[float, float] = (8.0, 8.0)
    horizontal_scale: float = 0.2
    vertical_scale: float = 0.005
    border_width: float = 20.0
    num_rows: int = 6
    num_cols: int = 6
    add_lights: bool = True
    sub_terrains: dict[str, SubTerrainCfg] = field(default_factory=_rough_sub_terrains)


def _real(
    value: Any,
    *,
    label: str,
    minimum: float | None = None,
    strict_minimum: bool = False,
) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{label} must be a real number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{label} must be finite")
    if minimum is not None and (result <= minimum if strict_minimum else result < minimum):
        relation = "greater than" if strict_minimum else "at least"
        raise ValueError(f"{label} must be {relation} {minimum}")
    return result


@dataclass(kw_only=True)
class RoughJointPositionActionCfg(JointPositionActionCfg):
    """Joint-position action with legacy raw-action clipping."""

    clip_actions: float = 100.0

    def build(self, env: ManagerBasedRlEnv) -> RoughJointPositionAction:
        return RoughJointPositionAction(self, env)


class RoughJointPositionAction(JointPositionAction):
    cfg: RoughJointPositionActionCfg  # pyright: ignore[reportIncompatibleVariableOverride]

    def __init__(self, cfg: RoughJointPositionActionCfg, env: ManagerBasedRlEnv):
        self._clip_actions = _real(
            cfg.clip_actions,
            label="RoughJointPositionActionCfg clip_actions",
            minimum=0.0,
            strict_minimum=True,
        )
        super().__init__(cfg, env)
        self._clipped_input = torch.empty_like(self.raw_action)

    def process_actions(self, actions: torch.Tensor) -> None:
        if not isinstance(actions, torch.Tensor):
            raise TypeError(
                f"RoughJointPositionAction expected torch.Tensor, got {type(actions).__name__}"
            )
        if actions.shape != self._clipped_input.shape:
            raise ValueError(
                f"RoughJointPositionAction expected shape {self._clipped_input.shape}, got {actions.shape}"
            )
        torch.clamp(
            actions, min=-self._clip_actions, max=self._clip_actions, out=self._clipped_input
        )
        super().process_actions(self._clipped_input)


@dataclass(kw_only=True)
class RoughVelocityCommandCfg(UniformVelocityCommandCfg):
    """Rough-task velocity command with a planar-norm dead zone."""

    planar_dead_zone: float = 0.08

    def build(self, env: ManagerBasedRlEnv) -> RoughVelocityCommand:
        return RoughVelocityCommand(self, env)


class RoughVelocityCommand(UniformVelocityCommand):
    cfg: RoughVelocityCommandCfg  # pyright: ignore[reportIncompatibleVariableOverride]

    def __init__(self, cfg: RoughVelocityCommandCfg, env: ManagerBasedRlEnv):
        self._planar_dead_zone = _real(
            cfg.planar_dead_zone,
            label="RoughVelocityCommandCfg planar_dead_zone",
            minimum=0.0,
        )
        if cfg.heading_command and not np.isclose(cfg.rel_heading_envs, 1.0):
            raise ValueError(
                "RoughVelocityCommandCfg heading_command requires rel_heading_envs=1.0"
            )
        super().__init__(cfg, env)

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        super()._resample_command(env_ids)
        tensor = self._tensor_command
        assert tensor is not None
        rows = env_ids.to(dtype=torch.int64, device=tensor.device)
        planar = tensor[rows, :2]
        moving = torch.linalg.vector_norm(planar, dim=1) > self._planar_dead_zone
        tensor[rows, :2] = planar * moving[:, None]


__all__ = [
    "QuadrupedRoughTerrainCfg",
    "RoughJointPositionAction",
    "RoughJointPositionActionCfg",
    "RoughVelocityCommand",
    "RoughVelocityCommandCfg",
]
