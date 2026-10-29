"""Run directory and checkpoint resolution helpers."""

from __future__ import annotations

import datetime
import os
import traceback
from collections.abc import Iterator
from contextlib import contextmanager
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from omegaconf import DictConfig, OmegaConf
from unisim.backend.base import normalize_play_render_mode

from unilab.rl.runtime.distributed.context import UNILAB_DP_LOG_DIR
from unilab.utils.checkpoint import (
    TEST_LOG_ROOT_ENV,
    get_latest_checkpoint,
    get_latest_run,
    normalize_load_run,
    resolve_checkpoint_path,
)

if TYPE_CHECKING:
    from unilab.training.tensor_diagnostics import NanGuardCfg


def build_run_dir_name(timestamp: str, sim_backend: str, *, world_size: int = 1) -> str:
    """Return the canonical run directory name shared by all training entries."""
    gpu_suffix = f"_gpux{world_size}" if world_size > 1 else ""
    return f"{timestamp}_{sim_backend}{gpu_suffix}"


def algo_config_dict(cfg: DictConfig) -> dict[str, Any]:
    """Resolve the composed ``cfg.algo`` subtree into a plain mutable dict."""
    raw = OmegaConf.to_container(cfg.algo, resolve=True)
    if not isinstance(raw, dict):
        raise TypeError("cfg.algo must resolve to a dict")
    return cast(dict[str, Any], raw)


def format_play_checkpoint_error(
    cfg: DictConfig,
    *,
    task_log_root: Path,
    load_path: Path | None,
    load_path_dir: Path | None,
) -> str:
    """Build the user-facing diagnostic for an unresolvable play checkpoint."""
    selected_checkpoint = OmegaConf.select(cfg, "algo.checkpoint", default=-1)
    checkpoint_hint = (
        f" algo.checkpoint={selected_checkpoint!r}"
        if selected_checkpoint not in (None, "", -1, "-1")
        else ""
    )

    if load_path_dir is not None and load_path is None and checkpoint_hint:
        reason = f"Requested checkpoint was not found under resolved_run={load_path_dir}."
    elif not task_log_root.exists():
        reason = "Task log root does not exist."
    else:
        latest_run = get_latest_run(task_log_root)
        if latest_run is None:
            reason = "No run directories were found under the task log root."
        elif get_latest_checkpoint(latest_run) is None:
            reason = f"Resolved latest run has no model_*.pt checkpoint files: {latest_run}."
        else:
            reason = "Requested run or checkpoint could not be resolved."

    return (
        "Could not resolve a checkpoint for play mode. "
        f"{reason} task={cfg.training.task_name} task_log_root={task_log_root} "
        f"algo.load_run={cfg.algo.load_run!r}{checkpoint_hint}."
        " Use algo.load_run=<run-dir-or-checkpoint-path> "
        "and optionally algo.checkpoint=<iteration-or-filename>."
    )


def resolve_nan_guard_cfg(training_cfg: Any) -> NanGuardCfg | None:
    """Build the shared tensor diagnostics config, or ``None``."""
    nan_guard_cfg = getattr(training_cfg, "nan_guard", None)
    if nan_guard_cfg is None or not getattr(nan_guard_cfg, "enabled", False):
        return None
    from unilab.training.tensor_diagnostics import NanGuardCfg

    return NanGuardCfg(
        enabled=True,
        buffer_size=int(getattr(nan_guard_cfg, "buffer_size", 100)),
        max_envs_to_dump=int(getattr(nan_guard_cfg, "max_envs_to_dump", 5)),
        output_dir=getattr(nan_guard_cfg, "output_dir", None),
        capture_interval=getattr(nan_guard_cfg, "capture_interval", 1),
        check_interval=getattr(nan_guard_cfg, "check_interval", 1),
        defer_validation=getattr(nan_guard_cfg, "defer_validation", False),
    )


def finite_validation_env_overrides(cfg: DictConfig) -> dict[str, Any]:
    """Route training's finite-check policy to the environment owner."""
    deferred = OmegaConf.select(cfg, "training.defer_finite_validation", default=False)
    interval = OmegaConf.select(cfg, "training.finite_validation_interval", default=1)
    if not isinstance(deferred, bool):
        raise TypeError("training.defer_finite_validation must be bool")
    if isinstance(interval, bool) or not isinstance(interval, int) or interval <= 0:
        raise ValueError("training.finite_validation_interval must be a positive integer")
    return {
        "defer_finite_validation": deferred,
        "finite_validation_interval": interval,
    }


def apply_env_nan_guard(env: Any, training_cfg: Any) -> None:
    """Attach a ``NanGuard`` to ``env`` when diagnostics are enabled."""
    nan_guard_cfg = resolve_nan_guard_cfg(training_cfg)
    if nan_guard_cfg is None:
        return
    from unilab.training.tensor_diagnostics import NanGuard

    env.set_nan_guard(
        NanGuard(
            nan_guard_cfg,
            num_envs=env.num_envs,
            supports_state_playback=env.play_capabilities.supports_physics_state_playback,
        )
    )


def is_viser_play_render_mode(play_render_mode: str | None) -> bool:
    """Return whether the configured mode selects the browser-based viser viewer.

    ``viser`` is a UniLab-level playback mode layered on top of the unisim
    ``PLAY_RENDER_MODES`` contract; it renders through the shared MuJoCo
    playback shell instead of a backend-native renderer, so it never reaches
    ``SimBackend.resolve_play_render_plan``.
    """
    return play_render_mode is not None and str(play_render_mode).strip().lower() == "viser"


def should_run_playback(*, play_only: bool, no_play: bool, play_render_mode: str | None) -> bool:
    """Return whether train/eval should enter playback for the configured mode."""
    if is_viser_play_render_mode(play_render_mode):
        return bool(play_only) or not bool(no_play)
    if normalize_play_render_mode(play_render_mode) == "none":
        return False
    return bool(play_only) or not bool(no_play)


@contextmanager
def nonfatal_play_step(step: str) -> Iterator[None]:
    """Keep independent post-training play artifacts (ONNX export, video render)
    from blocking each other: log the failure and continue with the rest."""
    try:
        yield
    except Exception:
        print(f"WARNING: {step} failed; continuing with the remaining play steps.")
        traceback.print_exc()


def get_log_root(
    root_dir: str | Path,
    cfg: DictConfig | None = None,
    *,
    algo_log_name: str | None = None,
    log_root: str | Path | None = None,
) -> Path:
    """Resolve the algorithm log root.

    The configured ``training.log_root`` (or the explicit ``log_root``
    argument) wins; otherwise the ``UNILAB_TEST_LOG_ROOT`` override applies;
    otherwise the default is ``<root_dir>/logs/<algo_log_name>``. Pass either
    a composed Hydra ``cfg`` or an explicit ``algo_log_name``.
    """
    if cfg is not None:
        if log_root is None:
            log_root = OmegaConf.select(cfg, "training.log_root")
        if algo_log_name is None:
            selected = OmegaConf.select(cfg, "algo.algo_log_name")
            algo_log_name = str(selected) if selected is not None else None
    if algo_log_name is None:
        raise ValueError("get_log_root requires a Hydra cfg or an explicit algo_log_name")
    if log_root:
        configured_root = Path(log_root)
        return (
            configured_root if configured_root.is_absolute() else Path(root_dir) / configured_root
        )
    test_log_root = os.environ.get(TEST_LOG_ROOT_ENV)
    if test_log_root:
        return Path(test_log_root) / algo_log_name
    return Path(root_dir) / "logs" / algo_log_name


def resolve_run_log_dir(
    cfg: DictConfig,
    *,
    world_size: int = 1,
    timestamp: str | None = None,
    root_dir: str | Path | None = None,
) -> str:
    """Resolve one canonical run directory shared by all distributed ranks."""
    distributed_log_dir = os.environ.get(UNILAB_DP_LOG_DIR)
    if distributed_log_dir:
        return distributed_log_dir
    configured_log_dir = OmegaConf.select(cfg, "training.log_dir", default=None)
    if configured_log_dir:
        return str(configured_log_dir)
    timestamp = timestamp or datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return str(
        get_log_root(Path.cwd() if root_dir is None else root_dir, cfg)
        / str(cfg.training.task_name)
        / build_run_dir_name(
            timestamp,
            str(cfg.training.sim_backend),
            world_size=world_size,
        )
    )


def resolve_configured_checkpoint_path(
    cfg: DictConfig,
    *,
    root_dir: str | Path,
    load_run: str | int | PathLike[str] | None = None,
    task_name: str | None = None,
    checkpoint: str | int | None = None,
    suffix: str = ".pt",
) -> tuple[Path | None, Path | None]:
    """Resolve a checkpoint path from Hydra config and repository root."""
    selected_task = task_name or str(OmegaConf.select(cfg, "training.task_name"))
    selected_run = (
        normalize_load_run(load_run)
        if load_run is not None
        else str(OmegaConf.select(cfg, "algo.load_run", default="-1"))
    )
    selected_checkpoint = checkpoint
    if selected_checkpoint is None:
        selected_checkpoint = OmegaConf.select(cfg, "algo.checkpoint", default=-1)
    if selected_checkpoint in (None, "", -1, "-1"):
        selected_checkpoint = None

    task_log_root = get_log_root(root_dir, cfg) / selected_task
    return resolve_checkpoint_path(
        task_log_root,
        selected_run,
        checkpoint=str(selected_checkpoint) if selected_checkpoint is not None else None,
        suffix=suffix,
    )
