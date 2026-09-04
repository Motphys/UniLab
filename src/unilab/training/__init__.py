"""Shared training helpers for entrypoint scripts.

Deprecated compatibility layer: the re-exports below are kept for existing
callers only. New code should import from the canonical locations instead
(``unilab.utils.*`` for layer-0 helpers, ``unilab.base.config_adapter`` /
``unisim.backend.base`` for base-layer helpers).
"""

from unisim.backend.base import log_playback_plan

from unilab.base.config_adapter import BackendAdapter, create_env
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
    format_hora_stage2_checkpoint_error,
    format_play_checkpoint_error,
    get_log_root,
    parse_checkpoint_path,
    resolve_hora_stage2_checkpoint_path,
    resolve_nan_guard_cfg,
    should_run_playback,
)
from unilab.utils.checkpoint import (
    get_entrypoint_log_root,
    get_latest_checkpoint,
    get_latest_run,
    resolve_appo_checkpoint_path,
    resolve_checkpoint_path,
    resolve_offpolicy_checkpoint_path,
    resolve_task_checkpoint_path,
)
from unilab.utils.monitoring import HardwareMonitor
from unilab.utils.seed import (
    TrainingSeedInfo,
    apply_configured_training_seed,
    apply_training_seed,
    derive_worker_seed,
    resolve_training_seed,
)

__all__ = [
    "BackendAdapter",
    "ExperimentTracker",
    "HardwareMonitor",
    "algo_config_dict",
    "apply_env_nan_guard",
    "assert_offpolicy_task_choice_matches_algo",
    "build_run_dir_name",
    "create_env",
    "ensure_registries",
    "format_play_checkpoint_error",
    "get_entrypoint_log_root",
    "get_hydra_runtime_choice",
    "get_latest_checkpoint",
    "get_latest_run",
    "get_log_root",
    "log_playback_plan",
    "parse_checkpoint_path",
    "resolve_checkpoint_path",
    "resolve_nan_guard_cfg",
    "resolve_task_checkpoint_path",
    "should_run_playback",
    "TrainingSeedInfo",
    "apply_configured_training_seed",
    "apply_training_seed",
    "derive_worker_seed",
    "format_hora_stage2_checkpoint_error",
    "resolve_appo_checkpoint_path",
    "resolve_hora_stage2_checkpoint_path",
    "resolve_offpolicy_checkpoint_path",
    "resolve_training_seed",
    "setup_logger",
]
