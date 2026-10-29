# Derived from mujocolab/mjlab v1.6.0 (0fb8a681),
# src/mjlab/tasks/velocity/mdp/velocity_command.py.
# Copyright 2025, The mjlab Developers.
# Modified by UniLab for Torch and the base-owned entity facade; Apache-2.0.
"""Uniform velocity commands for locomotion tasks."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from numbers import Real
from typing import TYPE_CHECKING, Any, cast

import torch

from unilab.managers.command_manager import CommandTerm, CommandTermCfg
from unilab.managers.rng import ManagerRng

if TYPE_CHECKING:
    from unilab.base.entity import Entity
    from unilab.managers.types import ManagerBasedRlEnv


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
        read_plan = getattr(self._env.scene, "read_plan", None)
        if read_plan is None:
            raise RuntimeError(
                "UniformVelocityCommand metrics require the scene's compiled packed "
                "tensor read plan. The lifecycle order is manager construction, then "
                "ManagerBasedRlEnv read-plan compilation, then per-step command "
                "compute; consuming metrics before the plan exists is an ordering bug"
            )
        names = self.tensor_sensor_names
        packed = read_plan.sensor_names.get(self.cfg.entity_name, ())
        # The scene pack negotiates backend-local aliases (Go2 ``local_linvel``/
        # ``gyro``) but the semantic metric remains the canonical IMU contract.
        aliases = self.packed_sensor_aliases
        resolved_names = tuple(
            next((local for local in aliases.get(name, (name,)) if local in packed), name)
            for name in names
        )
        if len(names) == 2 and set(resolved_names).issubset(packed):
            entity = self._env.scene[self.cfg.entity_name]
            views = read_plan.sensor_tensor_views(entity, resolved_names).values
            return views[resolved_names[0]], views[resolved_names[1]]
        raise NotImplementedError(
            "UniformVelocityCommand term 'twist' requires sensors "
            f"{names} in the Manager packed Torch read plan; "
            f"compiled={tuple(packed)}"
        )

    def _update_metrics(self) -> None:
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

    def _resample_command(self, reset_mask: torch.Tensor) -> None:
        """Sample at full width and merge only selected rows."""
        ranges = self.cfg.ranges
        samples = self._sample_full_width()
        if self.cfg.heading_command:
            assert ranges.heading is not None
            self._heading_target.copy_(
                torch.where(
                    reset_mask,
                    self._sample_uniform(ranges.heading[0], ranges.heading[1], self.num_envs),
                    self._heading_target,
                )
            )
            self._is_heading_env.copy_(
                torch.where(
                    reset_mask,
                    self._sample_uniform(0.0, 1.0, self.num_envs) < self.cfg.rel_heading_envs,
                    self._is_heading_env,
                )
            )
        self._is_standing_env.copy_(
            torch.where(
                reset_mask,
                self._sample_uniform(0.0, 1.0, self.num_envs) < self.cfg.rel_standing_envs,
                self._is_standing_env,
            )
        )
        self._is_world_env.copy_(
            torch.where(
                reset_mask,
                self._sample_uniform(0.0, 1.0, self.num_envs) < self.cfg.rel_world_envs,
                self._is_world_env,
            )
        )
        self._is_forward_env.copy_(
            torch.where(
                reset_mask,
                self._sample_uniform(0.0, 1.0, self.num_envs) < self.cfg.rel_forward_envs,
                self._is_forward_env,
            )
        )
        forward = self._is_forward_env & reset_mask
        samples[:, 0] = torch.where(forward, samples[:, 0].abs().clamp(min=0.3), samples[:, 0])
        samples[:, 1:] = torch.where(
            forward[:, None], torch.zeros_like(samples[:, 1:]), samples[:, 1:]
        )
        self._merge_samples_masked(samples, reset_mask)

    def _sample_full_width(self) -> torch.Tensor:
        ranges = self.cfg.ranges
        samples = torch.empty((self.num_envs, 3), dtype=torch.float32, device=self._device)
        for column, bounds in enumerate((ranges.lin_vel_x, ranges.lin_vel_y, ranges.ang_vel_z)):
            samples[:, column] = self._sample_uniform(bounds[0], bounds[1], self.num_envs)
        return samples

    def _merge_samples_masked(self, samples: torch.Tensor, reset_mask: torch.Tensor) -> None:
        tensor = self._tensor_command
        assert tensor is not None
        expanded = reset_mask.reshape(-1, *([1] * (tensor.ndim - 1)))
        tensor.copy_(torch.where(expanded, samples, tensor))
        self._world_command.copy_(torch.where(expanded, samples, self._world_command))

    def refresh_reset(self, reset_mask: torch.Tensor) -> None:
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
            heading_command = torch.clamp(
                self.cfg.heading_control_stiffness * heading_error,
                min=self.cfg.ranges.ang_vel_z[0],
                max=self.cfg.ranges.ang_vel_z[1],
            )
            tensor[:, 2] = torch.where(self._is_heading_env, heading_command, tensor[:, 2])

        if self.cfg.rel_world_envs > 0.0:
            heading = self._heading()
            cos_heading = torch.cos(heading)
            sin_heading = torch.sin(heading)
            velocity_x_w = self._world_command[:, 0]
            velocity_y_w = self._world_command[:, 1]
            world_x = cos_heading * velocity_x_w + sin_heading * velocity_y_w
            world_y = -sin_heading * velocity_x_w + cos_heading * velocity_y_w
            tensor[:, 0] = torch.where(self._is_world_env, world_x, tensor[:, 0])
            tensor[:, 1] = torch.where(self._is_world_env, world_y, tensor[:, 1])

        if self.cfg.rel_standing_envs > 0.0:
            tensor.copy_(
                torch.where(
                    self._is_standing_env[:, None],
                    torch.zeros_like(tensor),
                    tensor,
                )
            )
            self._world_command.copy_(
                torch.where(
                    self._is_standing_env[:, None],
                    torch.zeros_like(self._world_command),
                    self._world_command,
                )
            )

    def post_compute_reset(self) -> None:
        self.post_compute()

    def _sample_uniform(self, lower: float, upper: float, count: int) -> torch.Tensor:
        rng_owner = getattr(self._env, "rng", None)
        if not isinstance(rng_owner, ManagerRng):
            raise RuntimeError(
                "UniformVelocityCommand sampling requires the Manager-owned Torch generator"
            )
        return rng_owner.uniform(lower, upper, (count,), dtype=torch.float32)

    def _heading(self) -> torch.Tensor:
        read_plan = getattr(self._env.scene, "read_plan", None)
        root_body_name = getattr(self.robot, "root_body_name", None)
        if read_plan is None or not isinstance(root_body_name, str) or not root_body_name:
            # Command construction/reset does not consume heading; return the
            # cold neutral value until the packed read plan is compiled.
            return torch.zeros((self.num_envs,), dtype=torch.float32, device=self._device)
        quat = read_plan.body_tensor_view(self.robot, (root_body_name,)).quat_w[:, 0, :]
        w, x, y, z = quat.unbind(dim=-1)
        return torch.atan2(
            2.0 * (w * z + x * y),
            1.0 - 2.0 * (y * y + z * z),
        ).to(dtype=torch.float32)

    def _update_command(self) -> None:
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
            controlled_yaw = torch.clamp(
                self.cfg.heading_control_stiffness * heading_error,
                min=self.cfg.ranges.ang_vel_z[0],
                max=self.cfg.ranges.ang_vel_z[1],
            )
            tensor[:, 2] = torch.where(self._is_heading_env, controlled_yaw, tensor[:, 2])

        heading = self._heading()
        cos_heading = torch.cos(heading)
        sin_heading = torch.sin(heading)
        velocity_x_w = self._world_command[:, 0]
        velocity_y_w = self._world_command[:, 1]
        velocity_x_b = cos_heading * velocity_x_w + sin_heading * velocity_y_w
        velocity_y_b = -sin_heading * velocity_x_w + cos_heading * velocity_y_w
        tensor[:, 0] = torch.where(self._is_world_env, velocity_x_b, tensor[:, 0])
        tensor[:, 1] = torch.where(self._is_world_env, velocity_y_b, tensor[:, 1])

        standing = self._is_standing_env
        tensor.copy_(torch.where(standing[:, None], torch.zeros_like(tensor), tensor))
        self._world_command.copy_(
            torch.where(
                standing[:, None], torch.zeros_like(self._world_command), self._world_command
            )
        )


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
