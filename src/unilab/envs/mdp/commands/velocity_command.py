# Derived from mujocolab/mjlab v1.6.0 (0fb8a681),
# src/mjlab/tasks/velocity/mdp/velocity_command.py.
# Copyright 2025, The mjlab Developers.
# Modified by UniLab for NumPy and the base-owned entity facade; Apache-2.0.
"""Uniform velocity commands for locomotion tasks."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from numbers import Real
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from unilab.dtype_config import get_global_dtype
from unilab.managers.command_manager import CommandTerm, CommandTermCfg
from unilab.utils.rotation import np_wrap_to_pi

if TYPE_CHECKING:
    from unilab.base.entity import Entity
    from unilab.managers._types import ManagerBasedRlEnv


def _real(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{label} must be a real number, got {type(value).__name__}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite, got {result}")
    return result


def _range_pair(value: Any, *, label: str) -> tuple[float, float]:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise TypeError(f"{label} must be a two-value range")
    lower = _real(value[0], label=f"{label} lower")
    upper = _real(value[1], label=f"{label} upper")
    if lower > upper:
        raise ValueError(f"{label} lower {lower} exceeds upper {upper}")
    return lower, upper


def _ratio(value: Any, *, label: str) -> float:
    result = _real(value, label=label)
    if result < 0.0 or result > 1.0:
        raise ValueError(f"{label} must be within [0, 1], got {result}")
    return result


class UniformVelocityCommand(CommandTerm):
    """Sample planar velocity commands and update frame-dependent components."""

    cfg: UniformVelocityCommandCfg

    def __init__(self, cfg: UniformVelocityCommandCfg, env: ManagerBasedRlEnv):
        self._validate_cfg(cfg)
        super().__init__(cfg, env)
        if cfg.init_velocity_prob > 0.0:
            raise NotImplementedError(
                "UniformVelocityCommand capability 'initial root velocity write' is "
                "unavailable in the UniLab entity facade; set init_velocity_prob=0"
            )

        self.robot = cast("Entity", env.scene[cfg.entity_name])
        dtype = get_global_dtype()
        self.vel_command_b = np.zeros((self.num_envs, 3), dtype=dtype)
        self.vel_command_w = np.zeros_like(self.vel_command_b)
        self.heading_target = np.zeros(self.num_envs, dtype=dtype)
        self.heading_error = np.zeros(self.num_envs, dtype=dtype)
        self.is_heading_env = np.zeros(self.num_envs, dtype=np.bool_)
        self.is_standing_env = np.zeros(self.num_envs, dtype=np.bool_)
        self.is_world_env = np.zeros(self.num_envs, dtype=np.bool_)
        self.is_forward_env = np.zeros(self.num_envs, dtype=np.bool_)
        self.metrics["error_vel_xy"] = np.zeros(self.num_envs, dtype=dtype)
        self.metrics["error_vel_yaw"] = np.zeros(self.num_envs, dtype=dtype)

    @staticmethod
    def _validate_cfg(cfg: UniformVelocityCommandCfg) -> None:
        if not isinstance(cfg.entity_name, str) or not cfg.entity_name:
            raise ValueError("UniformVelocityCommandCfg entity_name must be non-empty")
        if not isinstance(cfg.heading_command, bool):
            raise TypeError("UniformVelocityCommandCfg heading_command must be bool")
        _real(
            cfg.heading_control_stiffness,
            label="UniformVelocityCommandCfg heading_control_stiffness",
        )
        if cfg.heading_control_stiffness < 0.0:
            raise ValueError("heading_control_stiffness must be non-negative")
        for name in (
            "rel_standing_envs",
            "rel_heading_envs",
            "rel_world_envs",
            "rel_forward_envs",
            "init_velocity_prob",
        ):
            _ratio(getattr(cfg, name), label=f"UniformVelocityCommandCfg {name}")
        if not isinstance(cfg.ranges, UniformVelocityCommandCfg.Ranges):
            raise TypeError("UniformVelocityCommandCfg ranges must be a Ranges instance")
        _range_pair(cfg.ranges.lin_vel_x, label="ranges.lin_vel_x")
        _range_pair(cfg.ranges.lin_vel_y, label="ranges.lin_vel_y")
        _range_pair(cfg.ranges.ang_vel_z, label="ranges.ang_vel_z")
        if cfg.ranges.heading is not None:
            _range_pair(cfg.ranges.heading, label="ranges.heading")
        if cfg.heading_command and cfg.ranges.heading is None:
            raise ValueError("heading_command=True but ranges.heading is None")
        if cfg.ranges.heading is not None and not cfg.heading_command:
            raise ValueError("ranges.heading is set but heading_command=False")
        _, upper = _range_pair(
            cfg.resampling_time_range,
            label="UniformVelocityCommandCfg resampling_time_range",
        )
        if upper <= 0.0:
            raise ValueError("resampling_time_range upper bound must be positive")

    @property
    def command(self) -> np.ndarray:
        return self.vel_command_b

    def _update_metrics(self, env_ids: np.ndarray | None = None) -> None:
        del env_ids  # Metrics accumulate over all rows on every compute.
        max_command_steps = self.cfg.resampling_time_range[1] / self._env.step_dt
        self.metrics["error_vel_xy"] += (
            np.linalg.norm(
                self.vel_command_b[:, :2] - self.robot.data.root_link_lin_vel_b[:, :2],
                axis=-1,
            )
            / max_command_steps
        )
        self.metrics["error_vel_yaw"] += (
            np.abs(self.vel_command_b[:, 2] - self.robot.data.root_link_ang_vel_b[:, 2])
            / max_command_steps
        )

    def _resample_command(self, env_ids: np.ndarray) -> None:
        count = len(env_ids)
        rng = self._env.rng
        ranges = self.cfg.ranges
        self.vel_command_b[env_ids, 0] = rng.uniform(*ranges.lin_vel_x, size=count)
        self.vel_command_b[env_ids, 1] = rng.uniform(*ranges.lin_vel_y, size=count)
        self.vel_command_b[env_ids, 2] = rng.uniform(*ranges.ang_vel_z, size=count)

        if self.cfg.heading_command:
            assert ranges.heading is not None
            self.heading_target[env_ids] = rng.uniform(*ranges.heading, size=count)
            self.is_heading_env[env_ids] = (
                rng.uniform(0.0, 1.0, size=count) <= self.cfg.rel_heading_envs
            )
        self.is_standing_env[env_ids] = (
            rng.uniform(0.0, 1.0, size=count) <= self.cfg.rel_standing_envs
        )
        self.is_world_env[env_ids] = rng.uniform(0.0, 1.0, size=count) <= self.cfg.rel_world_envs
        self.vel_command_w[env_ids] = self.vel_command_b[env_ids]
        self.is_forward_env[env_ids] = (
            rng.uniform(0.0, 1.0, size=count) <= self.cfg.rel_forward_envs
        )
        forward_ids = env_ids[self.is_forward_env[env_ids]]
        if len(forward_ids) > 0:
            self.vel_command_b[forward_ids, 0] = np.maximum(
                np.abs(self.vel_command_b[forward_ids, 0]), 0.3
            )
            self.vel_command_b[forward_ids, 1:] = 0.0

    def _update_command(self, env_ids: np.ndarray | None = None) -> None:
        del env_ids
        if self.cfg.heading_command:
            self.heading_error[:] = np_wrap_to_pi(self.heading_target - self.robot.data.heading_w)
            heading_ids = np.flatnonzero(self.is_heading_env)
            self.vel_command_b[heading_ids, 2] = np.clip(
                self.cfg.heading_control_stiffness * self.heading_error[heading_ids],
                self.cfg.ranges.ang_vel_z[0],
                self.cfg.ranges.ang_vel_z[1],
            )

        world_ids = np.flatnonzero(self.is_world_env)
        if len(world_ids) > 0:
            heading = self.robot.data.heading_w[world_ids]
            cos_heading = np.cos(heading)
            sin_heading = np.sin(heading)
            velocity_x_w = self.vel_command_w[world_ids, 0]
            velocity_y_w = self.vel_command_w[world_ids, 1]
            self.vel_command_b[world_ids, 0] = (
                cos_heading * velocity_x_w + sin_heading * velocity_y_w
            )
            self.vel_command_b[world_ids, 1] = (
                -sin_heading * velocity_x_w + cos_heading * velocity_y_w
            )

        standing_ids = np.flatnonzero(self.is_standing_env)
        self.vel_command_b[standing_ids] = 0.0
        self.vel_command_w[standing_ids] = 0.0


@dataclass(kw_only=True)
class UniformVelocityCommandCfg(CommandTermCfg):
    """Configuration for uniformly sampled planar velocity commands."""

    entity_name: str
    heading_command: bool = False
    heading_control_stiffness: float = 1.0
    rel_standing_envs: float = 0.0
    rel_heading_envs: float = 1.0
    rel_world_envs: float = 0.0
    rel_forward_envs: float = 0.0
    init_velocity_prob: float = 0.0

    @dataclass
    class Ranges:
        lin_vel_x: tuple[float, float]
        lin_vel_y: tuple[float, float]
        ang_vel_z: tuple[float, float]
        heading: tuple[float, float] | None = None

    ranges: Ranges

    @dataclass
    class VizCfg:
        z_offset: float = 0.2
        scale: float = 0.5

    viz: VizCfg = field(default_factory=VizCfg)

    def build(self, env: ManagerBasedRlEnv) -> UniformVelocityCommand:
        return UniformVelocityCommand(self, env)


__all__ = ["UniformVelocityCommand", "UniformVelocityCommandCfg"]
