"""Shared training helpers for entrypoint scripts."""

from unilab.training.backend_adapter import BackendAdapter
from unilab.training.common import (
    assert_offpolicy_task_choice_matches_algo,
    create_env,
    ensure_registries,
    get_hydra_runtime_choice,
    setup_logger,
)
from unilab.training.experiment import ExperimentTracker
from unilab.training.monitoring import HardwareMonitor
from unilab.training.run import (
    algo_config_dict,
    apply_env_nan_guard,
    build_run_dir_name,
    format_hora_stage2_checkpoint_error,
    format_play_checkpoint_error,
    get_entrypoint_log_root,
    get_latest_checkpoint,
    get_latest_run,
    get_log_root,
    log_playback_plan,
    parse_checkpoint_path,
    resolve_appo_checkpoint_path,
    resolve_checkpoint_path,
    resolve_hora_stage2_checkpoint_path,
    resolve_nan_guard_cfg,
    resolve_offpolicy_checkpoint_path,
    resolve_task_checkpoint_path,
    should_run_playback,
)
from unilab.training.seed import (
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
