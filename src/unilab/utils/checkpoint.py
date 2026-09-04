"""Pure-filesystem log root and checkpoint resolution helpers."""

from __future__ import annotations

import os
from os import PathLike
from pathlib import Path

_TEST_LOG_ROOT_ENV = "UNILAB_TEST_LOG_ROOT"


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
    test_log_root = os.environ.get(_TEST_LOG_ROOT_ENV)
    if test_log_root:
        return Path(test_log_root) / algo_log_name
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


def _normalize_load_run(load_run: str | int | PathLike[str]) -> str:
    return str(load_run)


def resolve_checkpoint_path(
    base_log_dir: str | Path,
    load_run: str | int | PathLike[str],
    *,
    suffix: str = ".pt",
) -> tuple[Path | None, Path | None]:
    """Resolve a latest or explicit checkpoint path from a task log root."""
    base_dir = Path(base_log_dir)
    selected_run = _normalize_load_run(load_run)
    if selected_run == "-1":
        run_dir = get_latest_run(base_dir)
        if run_dir is None:
            return None, None
        checkpoint = get_latest_checkpoint(run_dir, suffix=suffix)
        return (checkpoint, run_dir) if checkpoint is not None else (None, None)

    candidate = Path(selected_run)
    if not candidate.exists():
        candidate = base_dir / selected_run
    if candidate.is_file():
        return candidate, candidate.parent
    if candidate.is_dir():
        checkpoint = get_latest_checkpoint(candidate, suffix=suffix)
        return (checkpoint, candidate) if checkpoint is not None else (None, None)
    return None, None


def resolve_task_checkpoint_path(
    root_dir: str | Path,
    *,
    task_name: str,
    load_run: str | int | PathLike[str],
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
    selected_run = _normalize_load_run(load_run)
    if selected_run == "-1":
        run_dir = get_latest_run(task_log_root)
    else:
        candidate = Path(selected_run)
        if not candidate.exists():
            candidate = task_log_root / selected_run
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


def resolve_appo_checkpoint_path(
    base_log_dir: str | Path,
    load_run: str | int | PathLike[str],
) -> tuple[str | None, str | None]:
    """Resolve an APPO checkpoint under a task log root, returning string paths."""
    checkpoint_path, checkpoint_dir = resolve_checkpoint_path(
        base_log_dir,
        str(load_run),
        suffix=".pt",
    )
    return (
        str(checkpoint_path) if checkpoint_path is not None else None,
        str(checkpoint_dir) if checkpoint_dir is not None else None,
    )


def resolve_offpolicy_checkpoint_path(
    root_dir: str | Path,
    algo_log_name: str,
    task: str,
    load_run: str | int | PathLike[str],
) -> tuple[str | None, str | None]:
    """Resolve an off-policy checkpoint from the repo-rooted log tree."""
    checkpoint_path, checkpoint_dir = resolve_checkpoint_path(
        Path(root_dir) / "logs" / algo_log_name / task,
        load_run,
        suffix=".pt",
    )
    return (
        str(checkpoint_path) if checkpoint_path is not None else None,
        str(checkpoint_dir) if checkpoint_dir is not None else None,
    )
