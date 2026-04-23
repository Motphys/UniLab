from __future__ import annotations

import warnings

from unilab.training.monitoring import HardwareMonitor

warnings.warn(
    "`unilab.utils.hardware_monitor` is deprecated and will be removed in 0.2.0; "
    "use `unilab.training.monitoring` instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["HardwareMonitor"]
