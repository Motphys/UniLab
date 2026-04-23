from __future__ import annotations

import warnings

from unilab.algos.torch.rsl_rl.compat import *  # noqa: F403

warnings.warn(
    "`unilab.utils.rsl_rl_compat` is deprecated and will be removed in 0.2.0; "
    "use `unilab.algos.torch.rsl_rl.compat` instead.",
    DeprecationWarning,
    stacklevel=2,
)
