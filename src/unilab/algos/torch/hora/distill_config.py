"""HORA distillation config and teacher-owner resolution helpers."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, cast

from omegaconf import DictConfig, OmegaConf

from unilab.training.run import resolve_task_checkpoint_path

_REPO_ROOT = Path(__file__).resolve().parents[5]


def _root(root_dir: str | Path | None) -> Path:
    return Path(root_dir) if root_dir is not None else _REPO_ROOT


def _load_yaml_config(path: Path) -> DictConfig:
    loaded = OmegaConf.load(path)
    if not isinstance(loaded, DictConfig):
        raise TypeError(f"Expected DictConfig from {path}, got {type(loaded)!r}")
    return loaded


def _sanitize_path_token(value: str, *, fallback: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value)).strip("-._")
    return sanitized or fallback


def load_teacher_owner_config(
    algo_family: str,
    task: str,
    *,
    root_dir: str | Path | None = None,
) -> DictConfig:
    """Load a HORA teacher owner config and its direct owner defaults."""
    root = _root(root_dir)
    owner_path = root / "conf" / str(algo_family) / "task" / f"{task}.yaml"
    owner_cfg = _load_yaml_config(owner_path)
    merged_cfg = OmegaConf.create()
    for default_entry in owner_cfg.get("defaults", []):
        if not isinstance(default_entry, str) or default_entry == "_self_":
            continue
        include_path = root / "conf" / str(algo_family) / f"{default_entry.lstrip('/')}.yaml"
        merged_cfg = OmegaConf.merge(merged_cfg, _load_yaml_config(include_path))
    return cast(DictConfig, OmegaConf.merge(merged_cfg, owner_cfg))


def get_teacher_owner_spec(cfg: DictConfig) -> tuple[str | None, str | None]:
    """Resolve the teacher algo family and task owner from distillation config."""
    algo_family = OmegaConf.select(cfg, "teacher.algo_family")
    task = OmegaConf.select(cfg, "teacher.task")
    if algo_family in (None, "") or task in (None, ""):
        return None, None
    return str(algo_family), str(task)


def teacher_default_cfg(
    cfg: DictConfig,
    *,
    root_dir: str | Path | None = None,
) -> DictConfig:
    """Build HORA student defaults from the selected teacher owner YAML."""
    teacher_algo_family, teacher_task = get_teacher_owner_spec(cfg)
    if teacher_algo_family is None or teacher_task is None:
        return OmegaConf.create()

    teacher_cfg = load_teacher_owner_config(
        teacher_algo_family,
        teacher_task,
        root_dir=root_dir,
    )
    actor_cfg = OmegaConf.to_container(OmegaConf.select(teacher_cfg, "algo.actor"), resolve=True)
    if not isinstance(actor_cfg, dict):
        actor_cfg = {}
    actor_cfg = dict(actor_cfg)
    actor_class_name = str(actor_cfg.get("class_name", ""))
    if "HoraActorModel" not in actor_class_name:
        raise ValueError(
            "HORA distillation teacher owner must resolve to HoraActorModel. "
            f"Got algo_family={teacher_algo_family} task={teacher_task} "
            f"actor.class_name={actor_class_name!r}."
        )
    actor_cfg.pop("class_name", None)
    distribution_cfg = actor_cfg.get("distribution_cfg")
    if isinstance(distribution_cfg, dict):
        distribution_cfg = {
            key: value for key, value in distribution_cfg.items() if key != "class_name"
        }

    return OmegaConf.create(
        {
            "training": OmegaConf.select(teacher_cfg, "training"),
            "reward": OmegaConf.select(teacher_cfg, "reward"),
            "env": OmegaConf.select(teacher_cfg, "env"),
            "algo": {
                "model": {
                    "hidden_dims": actor_cfg.get("hidden_dims"),
                    "activation": actor_cfg.get("activation"),
                    "obs_normalization": actor_cfg.get("obs_normalization"),
                    "priv_info_embed_dim": actor_cfg.get("priv_info_embed_dim"),
                    "priv_mlp_hidden_dims": actor_cfg.get("priv_mlp_hidden_dims"),
                    "distribution_cfg": distribution_cfg,
                }
            },
        }
    )


def apply_teacher_defaults(
    cfg: DictConfig,
    *,
    root_dir: str | Path | None = None,
) -> DictConfig:
    """Merge teacher-owner defaults under the user distillation config."""
    return cast(DictConfig, OmegaConf.merge(teacher_default_cfg(cfg, root_dir=root_dir), cfg))


def resolved_distill_runtime_cfg(cfg: DictConfig) -> DictConfig:
    """Return stage-2 playback fields that do not depend on teacher algorithm."""
    model_cfg = OmegaConf.select(cfg, "algo.model")
    return OmegaConf.create(
        {
            "training": {
                "task_name": OmegaConf.select(cfg, "training.task_name"),
                "sim_backend": OmegaConf.select(cfg, "training.sim_backend"),
                "render_spacing": OmegaConf.select(cfg, "training.render_spacing"),
                "cam_distance": OmegaConf.select(cfg, "training.cam_distance"),
                "cam_elevation": OmegaConf.select(cfg, "training.cam_elevation"),
                "cam_azimuth": OmegaConf.select(cfg, "training.cam_azimuth"),
                "cam_lookat": OmegaConf.select(cfg, "training.cam_lookat"),
                "cam_tracking": OmegaConf.select(cfg, "training.cam_tracking"),
                "cam_tracking_env_idx": OmegaConf.select(cfg, "training.cam_tracking_env_idx"),
                "cam_tracking_extra_envs": OmegaConf.select(
                    cfg, "training.cam_tracking_extra_envs"
                ),
            },
            "reward": OmegaConf.select(cfg, "reward"),
            "env": OmegaConf.select(cfg, "env"),
            "algo": {
                "model": (
                    OmegaConf.to_container(model_cfg, resolve=True) if model_cfg is not None else {}
                )
            },
        }
    )


def teacher_run_metadata(
    cfg: DictConfig,
    *,
    teacher_algo_family: str,
    teacher_checkpoint: Path,
    root_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Build explicit teacher provenance metadata for distillation outputs."""
    teacher_task = OmegaConf.select(cfg, "teacher.task")
    root = _root(root_dir).resolve()
    checkpoint_path = teacher_checkpoint.resolve()
    try:
        checkpoint_display = str(checkpoint_path.relative_to(root))
    except ValueError:
        checkpoint_display = str(checkpoint_path)

    checkpoint_name = checkpoint_path.name
    return {
        "algo_family": str(teacher_algo_family),
        "task": None if teacher_task in (None, "") else str(teacher_task),
        "checkpoint_path": checkpoint_display,
        "checkpoint_name": checkpoint_name,
        "checkpoint_stem": checkpoint_path.stem,
        "run_name": checkpoint_path.parent.name,
        "run_slug": f"teacher-{_sanitize_path_token(teacher_algo_family, fallback='teacher')}",
    }


def resolve_teacher_checkpoint_path(
    cfg: DictConfig,
    *,
    root_dir: str | Path | None = None,
) -> tuple[Path | None, Path | None]:
    """Resolve the selected HORA teacher checkpoint through owner metadata."""
    teacher_algo_family, teacher_task = get_teacher_owner_spec(cfg)
    if teacher_algo_family is None or teacher_task is None:
        return None, None

    root = _root(root_dir)
    teacher_cfg = load_teacher_owner_config(
        teacher_algo_family,
        teacher_task,
        root_dir=root,
    )
    teacher_task_name = OmegaConf.select(teacher_cfg, "training.task_name")
    teacher_algo_log_name = OmegaConf.select(teacher_cfg, "algo.algo_log_name")
    if teacher_task_name in (None, "") or teacher_algo_log_name in (None, ""):
        raise ValueError(
            "Teacher owner config must define training.task_name and algo.algo_log_name. "
            f"Got algo_family={teacher_algo_family} task={teacher_task}."
        )

    selected_checkpoint = OmegaConf.select(cfg, "algo.checkpoint", default=-1)
    return resolve_task_checkpoint_path(
        root,
        task_name=str(teacher_task_name),
        load_run=str(OmegaConf.select(cfg, "algo.load_run", default="-1")),
        algo_log_name=str(teacher_algo_log_name),
        checkpoint=(
            str(selected_checkpoint)
            if selected_checkpoint not in (None, "", -1, "-1")
            else None
        ),
        suffix=".pt",
        log_root=OmegaConf.select(cfg, "training.log_root"),
    )
