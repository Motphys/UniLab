"""HORA distillation config and teacher-owner resolution helpers."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, cast

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

from unilab.training.run import resolve_task_checkpoint_path

_REPO_ROOT = Path(__file__).resolve().parents[4]

# Teacher owner configs are Hydra-composed from their family config tree.
# SAC teachers live in their own per-algo tree; there is no `algo` group anymore.
_TEACHER_TREE_BY_FAMILY = {"sac": "sac"}

# Teacher -> student `algo.model` mappings expressed in YAML; see
# conf/hora_distill/student_model/*.yaml. The mapping files interpolate against
# the composed teacher owner config mounted at `teacher_owner`.
_STUDENT_MODEL_MAPPING_DIR = Path("conf") / "hora_distill" / "student_model"


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
    """Compose a HORA teacher owner config with standard Hydra semantics.

    Uses ``initialize_config_dir`` + ``compose`` (same pattern as
    ``scripts/audit_sim2sim_contracts.py``), so package directives, nested
    ``defaults`` lists, and interpolations in the teacher tree all resolve.
    """
    root = _root(root_dir)
    algo_family = str(algo_family)
    conf_dir = root / "conf" / _TEACHER_TREE_BY_FAMILY.get(algo_family, algo_family)
    overrides = [f"task={task}"]
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(conf_dir.absolute()), version_base="1.3"):
        return compose("config", overrides=overrides)


def get_teacher_owner_spec(cfg: DictConfig) -> tuple[str | None, str | None]:
    """Resolve the teacher algo family and task owner from distillation config."""
    algo_family = OmegaConf.select(cfg, "teacher.algo_family")
    task = OmegaConf.select(cfg, "teacher.task")
    if algo_family in (None, "") or task in (None, ""):
        return None, None
    return str(algo_family), str(task)


def _teacher_contract_mapping_name(
    teacher_algo_family: str,
    teacher_task: str,
    teacher_cfg: DictConfig,
) -> str:
    """Validate the teacher owner contract and select the YAML mapping name."""
    if teacher_algo_family == "sac":
        runtime_impl = OmegaConf.select(teacher_cfg, "algo.runtime_impl")
        if runtime_impl != "hora_sac":
            raise ValueError(
                "HORA distillation SAC teacher owner must select runtime_impl='hora_sac'. "
                f"Got task={teacher_task} runtime_impl={runtime_impl!r}."
            )
        return "hora_sac"

    actor_class_name = str(OmegaConf.select(teacher_cfg, "algo.actor.class_name") or "")
    if "HoraActorModel" not in actor_class_name:
        raise ValueError(
            "HORA distillation teacher owner must resolve to HoraActorModel. "
            f"Got algo_family={teacher_algo_family} task={teacher_task} "
            f"actor.class_name={actor_class_name!r}."
        )
    return "hora_actor"


def _student_model_defaults(
    mapping_name: str,
    teacher_cfg: DictConfig,
    *,
    root: Path,
) -> dict[str, Any]:
    """Resolve the YAML-expressed teacher -> student model mapping.

    The mapping YAML is merged next to the composed teacher owner config
    (mounted at ``teacher_owner``) so its interpolations resolve against the
    Hydra-composed teacher hyperparameters. The mapping file owns all fallback
    defaults; this function only mounts and resolves it.
    """
    mapping_path = root / _STUDENT_MODEL_MAPPING_DIR / f"{mapping_name}.yaml"
    merged = OmegaConf.merge(
        {"teacher_owner": teacher_cfg},
        _load_yaml_config(mapping_path),
    )
    model_cfg = OmegaConf.to_container(OmegaConf.select(merged, "model"), resolve=True)
    if not isinstance(model_cfg, dict):
        raise TypeError(
            f"Expected mapping 'model' dict from {mapping_path}, got {type(model_cfg)!r}"
        )
    model_cfg = cast(dict[str, Any], model_cfg)
    distribution_cfg = model_cfg.get("distribution_cfg")
    if isinstance(distribution_cfg, dict):
        # The student re-binds its own distribution class; only the teacher's
        # distribution hyperparameters carry over, not its class binding.
        distribution_cfg.pop("class_name", None)
    return model_cfg


def teacher_default_cfg(
    cfg: DictConfig,
    *,
    root_dir: str | Path | None = None,
) -> DictConfig:
    """Build HORA student defaults from the selected teacher owner YAML."""
    teacher_algo_family, teacher_task = get_teacher_owner_spec(cfg)
    if teacher_algo_family is None or teacher_task is None:
        return OmegaConf.create()

    root = _root(root_dir)
    teacher_cfg = load_teacher_owner_config(
        teacher_algo_family,
        teacher_task,
        root_dir=root,
    )
    mapping_name = _teacher_contract_mapping_name(
        teacher_algo_family,
        teacher_task,
        teacher_cfg,
    )
    model_cfg = _student_model_defaults(mapping_name, teacher_cfg, root=root)
    return OmegaConf.create(
        {
            "training": OmegaConf.select(teacher_cfg, "training"),
            "reward": OmegaConf.select(teacher_cfg, "reward"),
            "env": OmegaConf.select(teacher_cfg, "env"),
            "algo": {"model": model_cfg},
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
    """Return checkpoint runtime fields needed to rebuild the student model.

    Stage-2 checkpoints intentionally do not persist owner runtime settings such
    as env, reward, or domain randomization. Replay should use the currently
    composed owner config for those fields.
    """
    model_cfg = OmegaConf.select(cfg, "algo.model")
    return OmegaConf.create(
        {
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
            str(selected_checkpoint) if selected_checkpoint not in (None, "", -1, "-1") else None
        ),
        suffix=".pt",
        log_root=OmegaConf.select(cfg, "training.log_root"),
    )
