# Derived from mujocolab/mjlab v1.6.0 (0fb8a681),
# src/mjlab_microduck/tasks/mdp.py command terms.
# Copyright 2025, The mjlab Developers.
# Modified by UniLab for NumPy and the public ManagerBasedRlEnv contract;
# licensed under Apache-2.0.
"""Generic pose and posture command terms for Manager-Based tasks."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import TYPE_CHECKING

import numpy as np

from unilab.dtype_config import get_global_dtype
from unilab.managers.command_manager import CommandTerm, CommandTermCfg

if TYPE_CHECKING:
    from unilab.managers._types import ManagerBasedRlEnv


def _real(value: object, *, label: str, minimum: float | None = None) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{label} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{label} must be at least {minimum}")
    return result


def _range_pair(value: object, *, label: str) -> tuple[float, float]:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise TypeError(f"{label} must be a two-value range")
    lower = _real(value[0], label=f"{label} lower")
    upper = _real(value[1], label=f"{label} upper")
    if lower > upper:
        raise ValueError(f"{label} lower {lower} exceeds upper {upper}")
    return lower, upper


@dataclass(kw_only=True)
class UniformPoseCommandCfg(CommandTermCfg):
    """Sample a fixed-width pose vector independently per dimension."""

    ranges: tuple[tuple[float, float], ...] | list[list[float]]
    zero_command_prob: float = 0.0

    def build(self, env: ManagerBasedRlEnv) -> UniformPoseCommand:
        return UniformPoseCommand(self, env)


class UniformPoseCommand(CommandTerm):
    """Uniformly sampled vector command held until the next resample."""

    cfg: UniformPoseCommandCfg

    def __init__(self, cfg: UniformPoseCommandCfg, env: ManagerBasedRlEnv):
        ranges = self._validated_ranges(cfg.ranges)
        probability = _real(
            cfg.zero_command_prob,
            label="UniformPoseCommandCfg zero_command_prob",
        )
        if probability > 1.0:
            raise ValueError("UniformPoseCommandCfg zero_command_prob must be within [0, 1]")
        self._zero_command_prob = probability
        super().__init__(cfg, env)
        self._command = np.zeros((self.num_envs, len(ranges)), dtype=get_global_dtype())

    @staticmethod
    def _validated_ranges(ranges: object) -> tuple[tuple[float, float], ...]:
        if not isinstance(ranges, (tuple, list)) or not ranges:
            raise ValueError("UniformPoseCommandCfg ranges must not be empty")
        return tuple(
            _range_pair(item, label=f"UniformPoseCommandCfg ranges[{index}]")
            for index, item in enumerate(ranges)
        )

    @property
    def command(self) -> np.ndarray:
        return self._command

    def _update_metrics(self, env_ids: np.ndarray | None = None) -> None:
        del env_ids

    def _resample_command(self, env_ids: np.ndarray) -> None:
        if len(env_ids) == 0:
            return
        ranges = self._validated_ranges(self.cfg.ranges)
        if len(ranges) != self._command.shape[1]:
            raise ValueError(
                "UniformPoseCommandCfg ranges width changed from "
                f"{self._command.shape[1]} to {len(ranges)}; curricula may only "
                "change per-axis bounds"
            )
        for column, (lower, upper) in enumerate(ranges):
            self._command[env_ids, column] = self._env.rng.uniform(
                lower,
                upper,
                size=len(env_ids),
            )
        if self._zero_command_prob > 0.0:
            zero = self._env.rng.uniform(0.0, 1.0, size=len(env_ids)) < self._zero_command_prob
            self._command[env_ids[zero]] = 0.0

    def _update_command(self, env_ids: np.ndarray | None) -> None:
        del env_ids


__all__ = ["UniformPoseCommand", "UniformPoseCommandCfg"]
