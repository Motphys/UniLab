from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from unilab.dtype_config import get_global_dtype


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


def sample_velocity_commands(
    rng: np.random.Generator, num_samples: int, low: np.ndarray, high: np.ndarray
) -> np.ndarray:
    return np.asarray(
        rng.uniform(low=low, high=high, size=(num_samples, 3)), dtype=get_global_dtype()
    )


def zero_small_xy_commands(commands: np.ndarray, *, threshold: float = 0.2) -> None:
    """Zero ``commands[:, :2]`` in-place wherever its norm is below ``threshold``."""
    moving = np.linalg.norm(commands[:, :2], axis=1) > threshold
    commands[:, :2] *= moving[:, None]
