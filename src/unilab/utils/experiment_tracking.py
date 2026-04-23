from __future__ import annotations

import warnings

from unilab.training.logging.experiment import (
    ExperimentTracker,
    build_wandb_run_name,
    build_wandb_settings,
    get_device_info_dict,
    get_git_info,
    patch_rsl_rl_wandb_writer,
)

warnings.warn(
    "`unilab.utils.experiment_tracking` is deprecated and will be removed in 0.2.0; "
    "use `unilab.training.logging.experiment` instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "ExperimentTracker",
    "build_wandb_run_name",
    "build_wandb_settings",
    "get_device_info_dict",
    "get_git_info",
    "patch_rsl_rl_wandb_writer",
]
