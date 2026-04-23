from __future__ import annotations

import warnings

from unilab.algos.torch.common.device import get_env_dims
from unilab.utils.device import get_default_device

warnings.warn(
    "`unilab.utils.device_utils` is deprecated and will be removed in 0.2.0; "
    "use `unilab.utils.device` for `get_default_device` "
    "and `unilab.algos.torch.common.device` for `get_env_dims` instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["get_default_device", "get_env_dims"]
