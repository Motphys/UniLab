from __future__ import annotations

import warnings

from unilab.training.logging.offpolicy import OffPolicyLogger

warnings.warn(
    "`unilab.algos.torch.offpolicy.logging` is deprecated and will be removed in 0.2.0; "
    "use `unilab.training.logging.offpolicy` instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["OffPolicyLogger"]
