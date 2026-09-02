"""Run directory and checkpoint resolution helpers."""

from __future__ import annotations

import os
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from omegaconf import DictConfig, OmegaConf
from unisim.backend.base import normalize_play_render_mode

from unilab.utils.checkpoint import (
    _TEST_LOG_ROOT_ENV,
    _normalize_load_run,
    get_latest_checkpoint,
    get_latest_run,
    resolve_task_checkpoint_path,
)

if TYPE_CHECKING:
    from unilab.utils.nan_guard import NanGuardCfg


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
    """Build the shared ``NanGuardCfg`` from ``training.nan_guard``, or ``None``."""
    nan_guard_cfg = getattr(training_cfg, "nan_guard", None)
    if nan_guard_cfg is None or not getattr(nan_guard_cfg, "enabled", False):
        return None
    from unilab.utils.nan_guard import NanGuardCfg

    return NanGuardCfg(
        enabled=True,
        buffer_size=int(getattr(nan_guard_cfg, "buffer_size", 100)),
        max_envs_to_dump=int(getattr(nan_guard_cfg, "max_envs_to_dump", 5)),
        output_dir=getattr(nan_guard_cfg, "output_dir", None),
    )


def apply_env_nan_guard(env: Any, training_cfg: Any) -> None:
    """Attach a ``NanGuard`` to ``env`` when ``training.nan_guard`` is enabled."""
    nan_guard_cfg = resolve_nan_guard_cfg(training_cfg)
    if nan_guard_cfg is None:
        return
    from unilab.utils.nan_guard import NanGuard

    env.set_nan_guard(
        NanGuard(
            nan_guard_cfg,
            num_envs=env.num_envs,
            supports_state_playback=env.play_capabilities.supports_physics_state_playback,
        )
    )


def should_run_playback(*, play_only: bool, no_play: bool, play_render_mode: str | None) -> bool:
    """Return whether train/eval should enter playback for the configured mode."""
    if normalize_play_render_mode(play_render_mode) == "none":
        return False
    return bool(play_only) or not bool(no_play)


def get_log_root(root_dir: str | Path, cfg: DictConfig) -> Path:
    """Resolve the algorithm log root, honoring optional training.log_root overrides."""
    configured_root = OmegaConf.select(cfg, "training.log_root")
    if configured_root:
        log_root = Path(str(configured_root))
        return log_root if log_root.is_absolute() else Path(root_dir) / log_root
    test_log_root = os.environ.get(_TEST_LOG_ROOT_ENV)
    if test_log_root:
        return Path(test_log_root) / str(OmegaConf.select(cfg, "algo.algo_log_name"))
    return Path(root_dir) / "logs" / str(OmegaConf.select(cfg, "algo.algo_log_name"))


def parse_checkpoint_path(
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
        _normalize_load_run(load_run)
        if load_run is not None
        else str(OmegaConf.select(cfg, "algo.load_run", default="-1"))
    )
    selected_checkpoint = checkpoint
    if selected_checkpoint is None:
        selected_checkpoint = OmegaConf.select(cfg, "algo.checkpoint", default=-1)
    if selected_checkpoint in (None, "", -1, "-1"):
        selected_checkpoint = None

    return resolve_task_checkpoint_path(
        root_dir,
        task_name=selected_task,
        load_run=selected_run,
        algo_log_name=str(OmegaConf.select(cfg, "algo.algo_log_name")),
        checkpoint=str(selected_checkpoint) if selected_checkpoint is not None else None,
        suffix=suffix,
        log_root=OmegaConf.select(cfg, "training.log_root"),
    )


def resolve_hora_stage2_checkpoint_path(
    cfg: DictConfig,
    *,
    root_dir: str | Path,
) -> tuple[Path | None, Path | None]:
    """Resolve the HORA stage-2 distillation checkpoint selected by ``cfg.algo``."""
    task_log_root = get_log_root(root_dir, cfg) / str(OmegaConf.select(cfg, "training.task_name"))
    load_run = str(OmegaConf.select(cfg, "algo.load_run", default="-1"))
    selected_checkpoint = OmegaConf.select(cfg, "algo.checkpoint", default=-1)

    run_dir: Path | None
    if load_run == "-1":
        run_dir = get_latest_run(task_log_root)
    else:
        candidate = Path(load_run)
        if not candidate.exists():
            candidate = task_log_root / load_run
        if candidate.is_file():
            return candidate, candidate.parent
        run_dir = candidate if candidate.is_dir() else None

    if run_dir is None:
        return None, None

    if selected_checkpoint not in (None, "", -1, "-1"):
        checkpoint_name = (
            f"hora_stage2_{selected_checkpoint}.pt"
            if str(selected_checkpoint).isdigit()
            else str(selected_checkpoint)
        )
        checkpoint_path = run_dir / checkpoint_name
        return (checkpoint_path, run_dir) if checkpoint_path.exists() else (None, run_dir)

    last_path = run_dir / "hora_stage2_last.pt"
    if last_path.exists():
        return last_path, run_dir

    numbered = [
        path for path in run_dir.glob("hora_stage2_*.pt") if path.stem.split("_")[-1].isdigit()
    ]
    if not numbered:
        return None, run_dir
    return max(numbered, key=lambda path: int(path.stem.split("_")[-1])), run_dir


def format_hora_stage2_checkpoint_error(
    cfg: DictConfig,
    *,
    task_log_root: Path,
    load_path: Path | None,
    load_path_dir: Path | None,
) -> str:
    """Build the user-facing diagnostic for an unresolvable stage-2 checkpoint."""
    selected_checkpoint = OmegaConf.select(cfg, "algo.checkpoint", default=-1)
    checkpoint_hint = (
        f" algo.checkpoint={selected_checkpoint!r}"
        if selected_checkpoint not in (None, "", -1, "-1")
        else ""
    )
    if load_path_dir is not None and load_path is None and checkpoint_hint:
        reason = f"Requested stage-2 checkpoint was not found under resolved_run={load_path_dir}."
    elif not task_log_root.exists():
        reason = "Task log root does not exist."
    else:
        latest_run = get_latest_run(task_log_root)
        if latest_run is None:
            reason = "No run directories were found under the task log root."
        else:
            reason = "Requested run or stage-2 checkpoint could not be resolved."
    return (
        "Could not resolve a stage-2 HORA checkpoint for play mode. "
        f"{reason} task={cfg.training.task_name} task_log_root={task_log_root} "
        f"algo.load_run={cfg.algo.load_run!r}{checkpoint_hint}. "
        "Use algo.load_run=<run-dir-or-checkpoint-path> and optionally "
        "algo.checkpoint=<iteration-or-filename>."
    )
