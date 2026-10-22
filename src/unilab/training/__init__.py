"""Shared training helpers for entrypoint scripts.

The re-exports below keep the owner-facing ``unilab.training`` import path
clear. New code should import from canonical owner modules instead
(``unilab.training.*``, ``unilab.base.config_adapter``, and
``unisim.backend.base``).
"""

from unisim.backend.base import log_playback_plan

from unilab.base.config_adapter import EnvConfigAdapter, create_env
from unilab.interruption import graceful_keyboard_interrupt
from unilab.training.common import (
    assert_offpolicy_task_choice_matches_algo,
    ensure_registries,
    get_hydra_runtime_choice,
    setup_logger,
)
from unilab.training.experiment import ExperimentTracker
from unilab.training.run import (
    algo_config_dict,
    apply_env_nan_guard,
    build_run_dir_name,
    format_play_checkpoint_error,
    get_log_root,
    is_viser_play_render_mode,
    nonfatal_play_step,
    resolve_configured_checkpoint_path,
    resolve_nan_guard_cfg,
    resolve_run_log_dir,
    should_run_playback,
)
from unilab.training.seed import (
    TrainingSeedInfo,
    apply_configured_training_seed,
    apply_training_seed,
    resolve_training_seed,
)
from unilab.utils.checkpoint import (
    get_latest_checkpoint,
    get_latest_run,
    resolve_checkpoint_path,
)

__all__ = [
    "EnvConfigAdapter",
    "ExperimentTracker",
    "algo_config_dict",
    "apply_env_nan_guard",
    "assert_offpolicy_task_choice_matches_algo",
    "build_run_dir_name",
    "create_env",
    "ensure_registries",
    "format_play_checkpoint_error",
    "get_hydra_runtime_choice",
    "get_latest_checkpoint",
    "get_latest_run",
    "get_log_root",
    "graceful_keyboard_interrupt",
    "is_viser_play_render_mode",
    "log_playback_plan",
    "nonfatal_play_step",
    "resolve_checkpoint_path",
    "resolve_configured_checkpoint_path",
    "resolve_nan_guard_cfg",
    "resolve_run_log_dir",
    "should_run_playback",
    "TrainingSeedInfo",
    "apply_configured_training_seed",
    "apply_training_seed",
    "resolve_training_seed",
    "setup_logger",
]
