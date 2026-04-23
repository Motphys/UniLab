from __future__ import annotations

import warnings

from unilab.utils.tensor import to_numpy, to_torch

warnings.warn(
    "`unilab.utils.torch_utils` is deprecated and will be removed in 0.2.0; "
    "use `unilab.utils.tensor` instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["to_numpy", "to_torch"]
