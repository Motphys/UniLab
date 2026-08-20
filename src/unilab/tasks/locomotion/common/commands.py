"""Shared command helpers for locomotion tasks."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Commands:
    vel_limit: list[list[float]] = field(
        default_factory=lambda: [
            [-0.6, -0.4, -0.8],  # [vx_min, vy_min, vyaw_min]
            [1.0, 0.4, 0.8],  # [vx_max, vy_max, vyaw_max]
        ]
    )
    resampling_time: float = 0.0
    heading_command: bool = False
    heading_range: list[float] = field(default_factory=lambda: [-3.14, 3.14])
    heading_control_stiffness: float = 0.5
    rel_standing_envs: float = 0.0
