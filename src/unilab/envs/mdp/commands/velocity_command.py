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
import torch

from unilab.managers.command_manager import CommandTerm, CommandTermCfg
from unilab.managers.torch_rng import TorchManagerRng

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
        self._tensor_command = torch.zeros(
            (self.num_envs, 3), dtype=torch.float32, device=self._device
        )
        self._world_command = torch.zeros_like(self._tensor_command)
        self._heading_target = torch.zeros(self.num_envs, dtype=torch.float32, device=self._device)
        self._is_heading_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self._device)
        self._is_standing_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self._device)
        self._is_world_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self._device)
        self._is_forward_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self._device)
        self.metrics["error_vel_xy"] = torch.zeros(self.num_envs, device=self._device)
        self.metrics["error_vel_yaw"] = torch.zeros(self.num_envs, device=self._device)

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
    def command(self) -> torch.Tensor:
        tensor = self._tensor_command
        assert tensor is not None
        return tensor

    @property
    def tensor_sensor_names(self) -> tuple[str, ...]:
        """IMU sensors used by per-step command tracking metrics."""
        return ("pelvis_local_linvel", "torso_gyro")

    # Backend-local aliases for the optional packed metric read. Go2 scenes
    # expose ``local_linvel``/``gyro`` while G1 exposes the canonical names.
    packed_sensor_aliases = {
        "pelvis_local_linvel": ("pelvis_local_linvel", "local_linvel"),
        "torso_gyro": ("torso_gyro", "gyro"),
    }

    def _metric_velocities(self) -> tuple[torch.Tensor, torch.Tensor]:
        read_plan = getattr(self._env.scene, "_tensor_read_plan", None)
        names = self.tensor_sensor_names
        packed = read_plan.sensor_names.get(self.cfg.entity_name, ()) if read_plan else ()
        # The scene pack negotiates backend-local aliases (Go2 ``local_linvel``/
        # ``gyro``) but the semantic metric remains the canonical IMU contract.
        if len(names) == 2 and names[0] in packed and names[1] in packed:
            entity = self._env.scene[self.cfg.entity_name]
            assert read_plan is not None
            views = read_plan.sensor_tensor_views(entity, names).values
            return views[names[0]], views[names[1]]
        # HOST_BRIDGE owners use their public entity facade. This is an
        # explicit carrier choice, not a hidden device transfer.
        device = getattr(self._env, "device", torch.device("cpu"))
        return (
            torch.as_tensor(self.robot.data.root_link_lin_vel_b, device=device),
            torch.as_tensor(self.robot.data.root_link_ang_vel_b, device=device),
        )

    def _update_metrics(self, env_ids: torch.Tensor | None = None) -> None:
        del env_ids  # Metrics accumulate over all rows on every compute.
        max_command_steps = self.cfg.resampling_time_range[1] / self._env.step_dt
        lin_vel, ang_vel = self._metric_velocities()
        command = torch.as_tensor(self.command, device=lin_vel.device)
        errors = torch.stack(
            (
                torch.linalg.vector_norm(command[:, :2] - lin_vel[:, :2], dim=-1),
                torch.abs(command[:, 2] - ang_vel[:, 2]),
            ),
            dim=1,
        )
        errors.div_(max_command_steps)
        self.metrics["error_vel_xy"] += errors[:, 0]
        self.metrics["error_vel_yaw"] += errors[:, 1]

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        self._resample_tensor_command(env_ids)

    def _resample_tensor_command(self, env_ids: torch.Tensor) -> None:
        count = env_ids.numel()
        if count == 0:
            return
        tensor = self._tensor_command
        assert tensor is not None
        rows = env_ids.to(dtype=torch.int64, device=self._device)
        ranges = self.cfg.ranges
        samples = torch.empty((count, 3), dtype=torch.float32, device=self._device)
        for column, bounds in enumerate((ranges.lin_vel_x, ranges.lin_vel_y, ranges.ang_vel_z)):
            samples[:, column] = self._sample_uniform(bounds[0], bounds[1], count)

        if self.cfg.heading_command:
            assert ranges.heading is not None
            self._heading_target[rows] = self._sample_uniform(*ranges.heading, count)
            self._is_heading_env[rows] = self._sample_uniform(0.0, 1.0, count) <= (
                self.cfg.rel_heading_envs
            )
        self._is_standing_env[rows] = self._sample_uniform(0.0, 1.0, count) <= (
            self.cfg.rel_standing_envs
        )
        self._is_world_env[rows] = self._sample_uniform(0.0, 1.0, count) <= self.cfg.rel_world_envs
        self._is_forward_env[rows] = self._sample_uniform(0.0, 1.0, count) <= (
            self.cfg.rel_forward_envs
        )
        forward = self._is_forward_env.index_select(0, rows)
        if bool(forward.any()):
            forward_rows = rows[forward]
            tensor[forward_rows, 0] = torch.clamp(samples[forward, 0].abs(), min=0.3)
            tensor[forward_rows, 1:] = 0.0
            self._world_command[forward_rows] = tensor[forward_rows]
            return
        tensor.index_copy_(0, rows, samples)
        self._world_command.index_copy_(0, rows, samples)

    def _sample_uniform(self, lower: float, upper: float, count: int) -> torch.Tensor:
        rng_owner = getattr(self._env, "torch_rng", None)
        if not isinstance(rng_owner, TorchManagerRng):
            raise RuntimeError(
                "UniformVelocityCommand sampling requires the Manager-owned Torch generator"
            )
        return rng_owner.uniform(lower, upper, (count,), dtype=torch.float32)

    def _heading(self) -> torch.Tensor:
        heading = self.robot.data.heading_w
        if isinstance(heading, torch.Tensor):
            return heading.to(device=self._device, dtype=torch.float32)
        return torch.as_tensor(heading, dtype=torch.float32, device=self._device)

    def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
        del env_ids
        tensor = self._tensor_command
        assert tensor is not None

        if self.cfg.heading_command:
            heading_error = (
                torch.remainder(
                    self._heading_target - self._heading() + torch.pi,
                    2.0 * torch.pi,
                )
                - torch.pi
            )
            heading_ids = self._is_heading_env.nonzero(as_tuple=False).flatten()
            if heading_ids.numel():
                tensor[heading_ids, 2] = torch.clamp(
                    self.cfg.heading_control_stiffness * heading_error[heading_ids],
                    min=self.cfg.ranges.ang_vel_z[0],
                    max=self.cfg.ranges.ang_vel_z[1],
                )

        world_ids = self._is_world_env.nonzero(as_tuple=False).flatten()
        if world_ids.numel():
            heading = self._heading()[world_ids]
            cos_heading = torch.cos(heading)
            sin_heading = torch.sin(heading)
            velocity_x_w = self._world_command[world_ids, 0]
            velocity_y_w = self._world_command[world_ids, 1]
            tensor[world_ids, 0] = cos_heading * velocity_x_w + sin_heading * velocity_y_w
            tensor[world_ids, 1] = -sin_heading * velocity_x_w + cos_heading * velocity_y_w

        standing_ids = self._is_standing_env.nonzero(as_tuple=False).flatten()
        if standing_ids.numel():
            tensor[standing_ids] = 0.0
            self._world_command[standing_ids] = 0.0


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
