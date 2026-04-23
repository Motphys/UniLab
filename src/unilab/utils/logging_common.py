from __future__ import annotations

import warnings

from unilab.training.logging.common import BaseTrainingLogger, _fmt_number, _fmt_time, _load_wandb

warnings.warn(
    "`unilab.utils.logging_common` is deprecated and will be removed in 0.2.0; "
    "use `unilab.training.logging.common` instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["BaseTrainingLogger", "_fmt_number", "_fmt_time", "_load_wandb"]
