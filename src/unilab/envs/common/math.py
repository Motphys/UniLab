from __future__ import annotations

import numpy as np


def np_sample_uniform(
    lower: float | np.ndarray,
    upper: float | np.ndarray,
    size: tuple[int, ...],
    dtype=np.float32,
) -> np.ndarray:
    """Sample uniformly from [lower, upper] with output dtype."""
    return np.random.uniform(lower, upper, size).astype(dtype)
