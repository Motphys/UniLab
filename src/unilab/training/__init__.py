"""Shared training helpers for entrypoint scripts."""

from unilab.training.backend_adapter import BackendAdapter
from unilab.training.common import (
    assert_offpolicy_task_choice_matches_algo,
    create_env,
    ensure_registries,
    get_entrypoint_log_root,
    get_hydra_runtime_choice,
    get_latest_checkpoint,
    get_latest_run,
    get_log_root,
    parse_checkpoint_path,
    render_play_mode,
    resolve_checkpoint_path,
    resolve_task_checkpoint_path,
    setup_logger,
)

__all__ = [
    "BackendAdapter",
    "assert_offpolicy_task_choice_matches_algo",
    "create_env",
    "ensure_registries",
    "get_entrypoint_log_root",
    "get_hydra_runtime_choice",
    "get_latest_checkpoint",
    "get_latest_run",
    "get_log_root",
    "parse_checkpoint_path",
    "render_play_mode",
    "resolve_checkpoint_path",
    "resolve_task_checkpoint_path",
    "setup_logger",
]
