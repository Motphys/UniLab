"""Global dtype configuration for environments."""

import numpy as np

# Global dtype for all environment computations
GLOBAL_DTYPE = np.float32


def get_global_dtype() -> np.dtype:
    """Get the global dtype for environment computations."""
    return GLOBAL_DTYPE
