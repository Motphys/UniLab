"""Global dtype configuration for environments."""

from __future__ import annotations

from typing import Any

import numpy as np

# Global dtype for all environment computations
GLOBAL_DTYPE = np.float32


def get_global_dtype() -> np.dtype[Any]:
    """Get the global dtype for environment computations."""
    return np.dtype(GLOBAL_DTYPE)
