from __future__ import annotations

import warnings

from unilab.training.run import (
    get_entrypoint_log_root,
    get_latest_checkpoint,
    get_latest_run,
    get_log_root,
    parse_checkpoint_path,
    resolve_checkpoint_path,
    resolve_task_checkpoint_path,
)

warnings.warn(
    "`unilab.utils.run_utils` is deprecated and will be removed in 0.2.0; "
    "use `unilab.training.run` instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "get_entrypoint_log_root",
    "get_latest_checkpoint",
    "get_latest_run",
    "get_log_root",
    "parse_checkpoint_path",
    "resolve_checkpoint_path",
    "resolve_task_checkpoint_path",
]
