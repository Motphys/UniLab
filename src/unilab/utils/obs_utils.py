from __future__ import annotations

import numpy as np


def flatten_obs_dict(obs: dict[str, np.ndarray]) -> np.ndarray:
    """Concatenate obs groups in insertion order -> flat (N, total_dim) array."""
    return np.concatenate(list(obs.values()), axis=1)
