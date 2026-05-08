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


def sample_velocity_commands(
    rng: np.random.Generator, num_samples: int, low: np.ndarray, high: np.ndarray
) -> np.ndarray:
    return np.asarray(
        rng.uniform(low=low, high=high, size=(num_samples, 3)), dtype=get_global_dtype()
    )
