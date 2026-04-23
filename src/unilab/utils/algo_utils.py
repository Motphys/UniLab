from __future__ import annotations

import warnings

from unilab.algos.torch.common.actor_factory import build_actor
from unilab.base.registry import ensure_registries

warnings.warn(
    "`unilab.utils.algo_utils` is deprecated and will be removed in 0.2.0; "
    "use `unilab.base.registry` and `unilab.algos.torch.common.actor_factory` instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["build_actor", "ensure_registries"]
