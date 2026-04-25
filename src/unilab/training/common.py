"""Shared helpers for training entrypoints."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from unilab.base.registry import ensure_registries as _ensure_registries


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
