"""Shared helpers for training entrypoints."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Callable, TypeVar

import numpy as np
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from unilab.utils.algo_utils import ensure_registries as _ensure_registries

ObsT = TypeVar("ObsT")


def ensure_registries() -> None:
    """Import env modules so registry-based entrypoints can instantiate tasks."""
    _ensure_registries()


def get_hydra_runtime_choice(cfg: DictConfig, group: str) -> str | None:
    """Return a selected Hydra config-group choice when runtime metadata is available."""
    cfg_choice = OmegaConf.select(cfg, f"hydra.runtime.choices.{group}")
    if cfg_choice is not None:
        return str(cfg_choice)

    if not HydraConfig.initialized():
        return None

    try:
        runtime_choice = HydraConfig.get().runtime.choices.get(group)
    except Exception:
        return None
    return str(runtime_choice) if runtime_choice is not None else None


def assert_offpolicy_task_choice_matches_algo(
    cfg: DictConfig,
    *,
    algo_name: str | None = None,
) -> None:
    """Reject offpolicy configs whose task owner path does not match the selected algo."""
    cfg_algo_name = str(OmegaConf.select(cfg, "algo.algo"))
    if algo_name is not None and cfg_algo_name != algo_name:
        raise ValueError(
            f"Off-policy algo argument {algo_name!r} is inconsistent with cfg.algo.algo={cfg_algo_name!r}"
        )

    selected_algo = algo_name or cfg_algo_name
    task_choice = get_hydra_runtime_choice(cfg, "task")
    if task_choice is None:
        return

    task_algo, sep, _ = task_choice.partition("/")
    if not sep:
        raise ValueError(
            f"Off-policy task choice must use task=<algo>/<task>/<backend>; got task={task_choice}"
        )
    if task_algo != selected_algo:
        raise ValueError(
            f"Off-policy algo/task mismatch: algo={selected_algo} is inconsistent with task={task_choice}. "
            "Use task=<algo>/<task>/<backend> with the same algo prefix."
        )


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
    suffix: str = ".pt",
) -> tuple[Path | None, Path | None]:
    """Resolve a checkpoint path from Hydra config and repository root."""
    selected_task = task_name or str(OmegaConf.select(cfg, "training.task_name"))
    selected_run = load_run or str(OmegaConf.select(cfg, "algo.load_run", default="-1"))
    base_log_dir = get_log_root(root_dir, cfg) / selected_task
    return resolve_checkpoint_path(base_log_dir, selected_run, suffix=suffix)


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


def setup_logger(
    log_dir: str | Path,
    algo_name: str,
    *,
    echo: bool = True,
    filename: str = "train.log",
) -> logging.Logger:
    """Create a simple file-backed logger for script-local progress messages."""
    path = Path(log_dir)
    path.mkdir(parents=True, exist_ok=True)

    logger_name = f"unilab.training.{algo_name}.{path.resolve()}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter("%(message)s")

    file_handler = logging.FileHandler(path / filename, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if echo:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger


def create_env(
    cfg: DictConfig,
    *,
    num_envs: int,
    env_cfg_override: dict[str, Any] | None = None,
    sim_backend: str | None = None,
    task_name: str | None = None,
):
    """Construct an environment via the registry using the current Hydra config."""
    from unilab.base import registry

    return registry.make(
        task_name or str(OmegaConf.select(cfg, "training.task_name")),
        num_envs=num_envs,
        sim_backend=sim_backend or str(OmegaConf.select(cfg, "training.sim_backend")),
        env_cfg_override=env_cfg_override,
    )


def render_play_mode(
    env,
    *,
    sim_backend: str,
    initialize: Callable[[], ObsT],
    step: Callable[[ObsT], ObsT],
    num_steps: int | None,
    output_video: str | Path | None = None,
    render_spacing: float | None = None,
    frame_state_getter: Callable[[], np.ndarray] | None = None,
    camera_kwargs: dict[str, Any] | None = None,
) -> str | None:
    """Render interactive Motrix play or MuJoCo video generation through shared callbacks."""
    if sim_backend == "motrix":
        env.init_play_renderer(render_spacing=render_spacing)

        obs = initialize()
        last_render_time = time.perf_counter()
        render_dt = 1.0 / 60.0
        steps_run = 0

        while num_steps is None or steps_run < num_steps:
            obs = step(obs)
            current_time = time.perf_counter()
            elapsed = current_time - last_render_time
            if elapsed < render_dt:
                time.sleep(render_dt - elapsed)
            last_render_time = time.perf_counter()
            env.render_play_frame()
            steps_run += 1
        return None

    if num_steps is None:
        raise ValueError("MuJoCo play rendering requires a finite num_steps value.")
    if output_video is None:
        raise ValueError("MuJoCo play rendering requires an output_video path.")
    if frame_state_getter is None:
        frame_state_getter = env.get_physics_state_snapshot
    assert frame_state_getter is not None

    obs = initialize()
    state_list = []
    for _ in range(num_steps):
        obs = step(obs)
        state_list.append(np.asarray(frame_state_getter(), dtype=np.float32).copy())

    from unilab.utils import render_many

    frames = render_many.render_states_get_frames(
        state_list,
        env.cfg.model_file,
        width=1280,
        height=720,
        camera_id=-1,
        **(camera_kwargs or {}),
    )

    import mediapy as media

    media.write_video(str(output_video), frames, fps=int(1.0 / env.cfg.ctrl_dt))
    return str(output_video)
