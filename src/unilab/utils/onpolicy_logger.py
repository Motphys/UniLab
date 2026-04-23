from __future__ import annotations

import warnings

from unilab.training.logging.onpolicy import OnPolicyLogger

warnings.warn(
    "`unilab.utils.onpolicy_logger` is deprecated and will be removed in 0.2.0; "
    "use `unilab.training.logging.onpolicy` instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["OnPolicyLogger"]
