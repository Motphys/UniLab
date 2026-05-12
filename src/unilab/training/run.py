"""Run directory and checkpoint resolution helpers."""

from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from unilab.base.backend.base import BackendPlayRenderPlan, normalize_play_render_mode


def should_run_playback(*, play_only: bool, no_play: bool, play_render_mode: str | None) -> bool:
    """Return whether train/eval should enter playback for the configured mode."""
    if normalize_play_render_mode(play_render_mode) == "none":
        return False
    return bool(play_only) or not bool(no_play)


def log_playback_plan(plan: BackendPlayRenderPlan, *, prefix: str = "") -> None:
    """Print user-facing playback status for a resolved backend plan."""
    if plan.mode == "none":
        print(f"{prefix}Skipping playback because training.play_render_mode=none.")
        return
    if plan.record_video:
        print(f"{prefix}Rendering video to {plan.output_video}...")
    elif plan.mode == "interactive":
        print(f"{prefix}Starting interactive visualization (motrix native renderer)...")
        print(f"{prefix}Close the render window to exit.")
    else:
        print(f"{prefix}Running playback without video recording...")
    print(f"{prefix}Rendering playback frames...")


def get_log_root(root_dir: str | Path, cfg: DictConfig) -> Path:
    """Resolve the algorithm log root, honoring optional training.log_root overrides."""
    configured_root = OmegaConf.select(cfg, "training.log_root")
    if configured_root:
        log_root = Path(str(configured_root))
        return log_root if log_root.is_absolute() else Path(root_dir) / log_root
    return Path(root_dir) / "logs" / str(OmegaConf.select(cfg, "algo.algo_log_name"))


def get_entrypoint_log_root(
    root_dir: str | Path,
    *,
    algo_log_name: str,
    log_root: str | Path | None = None,
) -> Path:
    """Resolve the log root for non-Hydra entrypoints using training helper semantics."""
    if log_root is not None:
        configured_root = Path(log_root)
        return (
            configured_root if configured_root.is_absolute() else Path(root_dir) / configured_root
        )
    return Path(root_dir) / "logs" / algo_log_name


def get_latest_run(log_dir: str | Path) -> Path | None:
    """Return the lexicographically latest run directory under a task log root."""
    base_dir = Path(log_dir)
    if not base_dir.exists():
        return None
    runs = sorted(path for path in base_dir.iterdir() if path.is_dir())
    return runs[-1] if runs else None


def get_latest_checkpoint(run_dir: str | Path, *, suffix: str = ".pt") -> Path | None:
    """Return the latest model checkpoint inside a run directory."""
    run_path = Path(run_dir)
    if not run_path.exists():
        return None

    def _iteration(path: Path) -> int:
        stem_parts = path.stem.split("_", 1)
        if len(stem_parts) != 2:
            return -1
        try:
            return int(stem_parts[1])
        except ValueError:
            return -1

    model_files = [
        path
        for path in run_path.iterdir()
        if path.is_file() and path.name.startswith("model_") and path.suffix == suffix
    ]
    if not model_files:
        return None
    return max(model_files, key=_iteration)


def resolve_checkpoint_path(
    base_log_dir: str | Path,
    load_run: str,
    *,
    suffix: str = ".pt",
) -> tuple[Path | None, Path | None]:
    """Resolve a latest or explicit checkpoint path from a task log root."""
    base_dir = Path(base_log_dir)
    if load_run == "-1":
        run_dir = get_latest_run(base_dir)
        if run_dir is None:
            return None, None
        checkpoint = get_latest_checkpoint(run_dir, suffix=suffix)
        return (checkpoint, run_dir) if checkpoint is not None else (None, None)

    candidate = Path(load_run)
    if not candidate.exists():
        candidate = base_dir / load_run
    if candidate.is_file():
        return candidate, candidate.parent
    if candidate.is_dir():
        checkpoint = get_latest_checkpoint(candidate, suffix=suffix)
        return (checkpoint, candidate) if checkpoint is not None else (None, None)
    return None, None


def parse_checkpoint_path(
    cfg: DictConfig,
    *,
    root_dir: str | Path,
    load_run: str | None = None,
    task_name: str | None = None,
    checkpoint: str | int | None = None,
    suffix: str = ".pt",
) -> tuple[Path | None, Path | None]:
    """Resolve a checkpoint path from Hydra config and repository root."""
    selected_task = task_name or str(OmegaConf.select(cfg, "training.task_name"))
    selected_run = load_run or str(OmegaConf.select(cfg, "algo.load_run", default="-1"))
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


def resolve_task_checkpoint_path(
    root_dir: str | Path,
    *,
    task_name: str,
    load_run: str,
    algo_log_name: str,
    checkpoint: str | None = None,
    suffix: str = ".pt",
    log_root: str | Path | None = None,
) -> tuple[Path | None, Path | None]:
    """Resolve checkpoint paths for auxiliary entrypoints through shared training semantics."""
    task_log_root = (
        get_entrypoint_log_root(
            root_dir,
            algo_log_name=algo_log_name,
            log_root=log_root,
        )
        / task_name
    )

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

    checkpoint_path: Path | None
    if checkpoint is not None:
        checkpoint_name = (
            f"model_{checkpoint}{suffix}" if str(checkpoint).isdigit() else str(checkpoint)
        )
        checkpoint_path = run_dir / checkpoint_name
        return (checkpoint_path, run_dir) if checkpoint_path.exists() else (None, run_dir)

    checkpoint_path = get_latest_checkpoint(run_dir, suffix=suffix)
    return (checkpoint_path, run_dir) if checkpoint_path is not None else (None, run_dir)
