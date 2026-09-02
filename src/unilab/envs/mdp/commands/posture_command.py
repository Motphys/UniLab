# Derived from mujocolab/mjlab v1.6.0 (0fb8a681),
# src/mjlab_microduck/tasks/mdp.py command terms.
# Copyright 2025, The mjlab Developers.
# Modified by UniLab for NumPy and the public ManagerBasedRlEnv contract;
# licensed under Apache-2.0.
"""Reusable phase and posture commands for Manager-Based tasks."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from unilab.dtype_config import get_global_dtype
from unilab.envs.mdp.commands.velocity_command import (
    UniformVelocityCommand,
    UniformVelocityCommandCfg,
)
from unilab.managers.command_manager import CommandTerm

if TYPE_CHECKING:
    from unilab.base.entity import Entity
    from unilab.managers._types import ManagerBasedRlEnv


def _bounded_probability(value: Any, *, label: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)):
        raise TypeError(f"{label} must be a real number")
    result = float(value)
    if not np.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{label} must be finite and within [0, 1]")
    return result


def _finite_real(value: Any, *, label: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)):
        raise TypeError(f"{label} must be a real number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


class GroundPickPhaseCommand(UniformVelocityCommand):
    """Continuous cyclic phase command encoded as ``[cos, sin, 0]``."""

    def __init__(self, cfg: GroundPickPhaseCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        period = _finite_real(cfg.period, label="GroundPickPhaseCommandCfg period")
        if period <= 0.0:
            raise ValueError("GroundPickPhaseCommandCfg period must be finite and positive")
        if not isinstance(cfg.randomize_phase, bool):
            raise TypeError("GroundPickPhaseCommandCfg randomize_phase must be bool")
        self._period = period
        self._randomize_phase = cfg.randomize_phase
        self._phase = np.zeros(self.num_envs, dtype=get_global_dtype())

    @property
    def phase(self) -> np.ndarray:
        return self._phase

    def reset(self, env_ids: np.ndarray | slice | None) -> dict[str, float]:
        ids = (
            np.arange(self.num_envs, dtype=np.int32)
            if env_ids is None
            else (
                np.arange(self.num_envs, dtype=np.int32)[env_ids]
                if isinstance(env_ids, slice)
                else np.asarray(env_ids, dtype=np.int32)
            )
        )
        self.time_left[ids] = 0.0
        self.command_counter[ids] = 0
        self.metrics.clear()
        if self._randomize_phase:
            self._phase[ids] = self._env.rng.uniform(0.0, 1.0, size=len(ids))
        else:
            self._phase[ids] = 0.0
        self._update_command(ids)
        return {}

    def compute(self, dt: float | np.ndarray, env_ids: np.ndarray | None = None) -> None:
        # This command is a continuous phase signal, not a sampled command.
        # Bypass CommandTerm's timer/resampling path so ``command_counter`` and
        # ``time_left`` do not create a second, invisible scheduling contract.
        if env_ids is not None:
            if isinstance(dt, np.ndarray):
                raise ValueError("GroundPickPhaseCommand reset compute expects scalar dt")
            if not np.isfinite(dt):
                raise ValueError("GroundPickPhaseCommand received non-finite dt")
            ids = np.asarray(env_ids, dtype=np.int32)
            self._update_command(ids)
            return
        delta = np.asarray(dt, dtype=get_global_dtype())
        if not np.isfinite(delta).all():
            raise ValueError("GroundPickPhaseCommand received non-finite dt")
        if delta.ndim == 0:
            self._phase[:] = (self._phase + float(delta) / self._period) % 1.0
        elif delta.shape == (self.num_envs,):
            self._phase[:] = (self._phase + delta / self._period) % 1.0
        else:
            raise ValueError(
                f"GroundPickPhaseCommand dt must be scalar or ({self.num_envs},), got {delta.shape}"
            )
        self._update_command(None)

    def _resample_command(self, env_ids: np.ndarray) -> None:
        del env_ids

    def _update_command(self, env_ids: np.ndarray | None = None) -> None:
        del env_ids
        self.vel_command_b[:, 0] = np.cos(2.0 * math.pi * self._phase)
        self.vel_command_b[:, 1] = np.sin(2.0 * math.pi * self._phase)
        self.vel_command_b[:, 2] = 0.0

    def _update_metrics(self, env_ids: np.ndarray | None = None) -> None:
        del env_ids


@dataclass(kw_only=True)
class GroundPickPhaseCommandCfg(UniformVelocityCommandCfg):
    # Retained for overlays that specialize the shared uniform-velocity
    # command. Phase commands intentionally ignore these knobs.
    turn_in_place_fraction: float = 0.0
    turn_in_place_ang_min: float = 0.4
    period: float = 4.0
    randomize_phase: bool = True

    def build(self, env: ManagerBasedRlEnv) -> GroundPickPhaseCommand:
        return GroundPickPhaseCommand(self, env)


class SitStandCommand(UniformVelocityCommand):
    """Binary posture command with a bounded-rate internal target blend."""

    def __init__(self, cfg: SitStandCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self._sit_prob = _bounded_probability(cfg.sit_prob, label="SitStandCommandCfg sit_prob")
        ramp_s = _finite_real(cfg.ramp_s, label="SitStandCommandCfg ramp_s")
        sit_z = _finite_real(cfg.sit_z, label="SitStandCommandCfg sit_z")
        stand_z = _finite_real(cfg.stand_z, label="SitStandCommandCfg stand_z")
        if ramp_s <= 0.0:
            raise ValueError("SitStandCommandCfg ramp_s must be finite and positive")
        if sit_z >= stand_z:
            raise ValueError("SitStandCommandCfg sit_z must be below stand_z")
        self._ramp_s = ramp_s
        self._sit_z = sit_z
        self._stand_z = stand_z
        self._alpha = np.zeros(self.num_envs, dtype=get_global_dtype())
        self.robot = cast("Entity", env.scene[cfg.entity_name])

    @property
    def alpha(self) -> np.ndarray:
        return self._alpha

    def _resample_command(self, env_ids: np.ndarray) -> None:
        self.vel_command_b[env_ids] = 0.0
        sit = self._env.rng.uniform(0.0, 1.0, size=len(env_ids)) < self._sit_prob
        self.vel_command_b[env_ids, 0] = sit.astype(get_global_dtype())

    def _update_command(self, env_ids: np.ndarray | None = None) -> None:
        del env_ids

    def compute(self, dt: float | np.ndarray, env_ids: np.ndarray | None = None) -> None:
        super().compute(dt, env_ids)
        if env_ids is not None:
            ids = np.asarray(env_ids, dtype=np.int32)
            delta = np.zeros(len(ids), dtype=get_global_dtype())
            fresh = self._env.episode_length_buf[ids] <= 1
            if np.any(fresh):
                delta[fresh] = self._alpha_from_height(ids[fresh])
            self._alpha[ids[fresh]] = delta[fresh]
            return
        dt_values = np.asarray(dt, dtype=get_global_dtype())
        if dt_values.ndim == 0:
            step = np.full(self.num_envs, float(dt_values), dtype=get_global_dtype())
        elif dt_values.shape == (self.num_envs,):
            step = dt_values
        else:
            raise ValueError(
                f"SitStandCommand dt must be scalar or ({self.num_envs},), got {dt_values.shape}"
            )
        fresh = self._env.episode_length_buf <= 1
        if np.any(fresh):
            self._alpha[fresh] = self._alpha_from_height(np.flatnonzero(fresh))
        target = self.vel_command_b[:, 0]
        self._alpha += np.clip(target - self._alpha, -step / self._ramp_s, step / self._ramp_s)

    def _alpha_from_height(self, ids: np.ndarray) -> np.ndarray:
        height = self.robot.data.root_link_pos_w[ids, 2]
        return np.clip(
            (self._stand_z - height) / max(self._stand_z - self._sit_z, 1e-6),
            0.0,
            1.0,
        ).astype(get_global_dtype(), copy=False)

    def _update_metrics(self, env_ids: np.ndarray | None = None) -> None:
        del env_ids


@dataclass(kw_only=True)
class SitStandCommandCfg(UniformVelocityCommandCfg):
    # Retained for overlays that specialize the shared uniform-velocity
    # command. Posture commands intentionally ignore these knobs.
    turn_in_place_fraction: float = 0.0
    turn_in_place_ang_min: float = 0.4
    sit_prob: float = 0.5
    ramp_s: float = 2.0
    sit_z: float = 0.060
    stand_z: float = 0.115

    def build(self, env: ManagerBasedRlEnv) -> SitStandCommand:
        return SitStandCommand(self, env)


__all__ = [
    "GroundPickPhaseCommand",
    "GroundPickPhaseCommandCfg",
    "SitStandCommand",
    "SitStandCommandCfg",
]
