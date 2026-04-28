import datetime
import json
import re
import sys
from pathlib import Path
from typing import Any, cast

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict

ROOT_DIR = Path(__file__).parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from unilab.algos.torch.hora import HoraDistillationTrainer
from unilab.algos.torch.hora.distill import (
    build_student_actor_and_normalizer,
    load_distilled_checkpoint,
)
from unilab.algos.torch.hora.rsl_rl import HoraRslRlVecEnvWrapper as RslRlVecEnvWrapper
from unilab.base.backend.xml import materialize_scene_visual_override
from unilab.training import (
    BackendAdapter,
    create_env,
    ensure_registries,
    get_latest_run,
    get_log_root,
    resolve_task_checkpoint_path,
    setup_logger,
)
from unilab.visualization import render_play_mode


def _load_yaml_config(path: Path) -> DictConfig:
    loaded = OmegaConf.load(path)
    if not isinstance(loaded, DictConfig):
        raise TypeError(f"Expected DictConfig from {path}, got {type(loaded)!r}")
    return loaded


def _sanitize_path_token(value: str, *, fallback: str) -> str:
    """Convert an arbitrary identifier into a filesystem-safe path token.

    Args:
        value: Raw identifier to sanitize.
        fallback: Token to use when the sanitized value becomes empty.

    Returns:
        ASCII-safe token suitable for a log-directory name.
    """
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value)).strip("-._")
    return sanitized or fallback


def _load_teacher_owner_config(algo_family: str, task: str) -> DictConfig:
    owner_path = ROOT_DIR / "conf" / str(algo_family) / "task" / f"{task}.yaml"
    owner_cfg = _load_yaml_config(owner_path)
    merged_cfg = OmegaConf.create()
    for default_entry in owner_cfg.get("defaults", []):
        if not isinstance(default_entry, str) or default_entry == "_self_":
            continue
        include_path = ROOT_DIR / "conf" / str(algo_family) / f"{default_entry.lstrip('/')}.yaml"
        merged_cfg = OmegaConf.merge(merged_cfg, _load_yaml_config(include_path))
    return cast(DictConfig, OmegaConf.merge(merged_cfg, owner_cfg))


def _get_teacher_owner_spec(cfg: DictConfig) -> tuple[str | None, str | None]:
    algo_family = OmegaConf.select(cfg, "teacher.algo_family")
    task = OmegaConf.select(cfg, "teacher.task")
    if algo_family in (None, "") or task in (None, ""):
        return None, None
    return str(algo_family), str(task)


def _teacher_default_cfg(cfg: DictConfig) -> DictConfig:
    teacher_algo_family, teacher_task = _get_teacher_owner_spec(cfg)
    if teacher_algo_family is None or teacher_task is None:
        return OmegaConf.create()

    teacher_cfg = _load_teacher_owner_config(teacher_algo_family, teacher_task)
    actor_cfg = OmegaConf.to_container(OmegaConf.select(teacher_cfg, "algo.actor"), resolve=True)
    if not isinstance(actor_cfg, dict):
        actor_cfg = {}
    actor_cfg = dict(actor_cfg)
    actor_class_name = str(actor_cfg.get("class_name", ""))
    if "HoraActorModel" not in actor_class_name:
        raise ValueError(
            "HORA distillation teacher owner must resolve to HoraActorModel. "
            f"Got algo_family={teacher_algo_family} task={teacher_task} actor.class_name={actor_class_name!r}."
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


def _apply_teacher_defaults(cfg: DictConfig) -> DictConfig:
    return cast(DictConfig, OmegaConf.merge(_teacher_default_cfg(cfg), cfg))


def _resolved_distill_runtime_cfg(cfg: DictConfig) -> DictConfig:
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


def _teacher_run_metadata(
    cfg: DictConfig,
    *,
    teacher_algo_family: str,
    teacher_checkpoint: Path,
) -> dict[str, Any]:
    """Build explicit teacher provenance metadata for distillation outputs.

    Args:
        cfg: Resolved distillation config.
        teacher_algo_family: Teacher algorithm family selected by the owner config.
        teacher_checkpoint: Resolved checkpoint path used to initialize the teacher actor.

    Returns:
        Dict containing teacher owner and checkpoint fields for log naming and metadata files.
    """
    teacher_task = OmegaConf.select(cfg, "teacher.task")
    checkpoint_path = teacher_checkpoint.resolve()
    try:
        checkpoint_display = str(checkpoint_path.relative_to(ROOT_DIR))
    except ValueError:
        checkpoint_display = str(checkpoint_path)

    run_name = checkpoint_path.parent.name
    checkpoint_name = checkpoint_path.name
    algo_token = _sanitize_path_token(teacher_algo_family, fallback="teacher")
    run_slug = f"teacher-{algo_token}"

    return {
        "algo_family": str(teacher_algo_family),
        "task": None if teacher_task in (None, "") else str(teacher_task),
        "checkpoint_path": checkpoint_display,
        "checkpoint_name": checkpoint_name,
        "checkpoint_stem": checkpoint_path.stem,
        "run_name": run_name,
        "run_slug": run_slug,
    }


def _write_distill_run_config(
    log_dir: Path,
    *,
    cfg: DictConfig,
    teacher_metadata: dict[str, Any],
) -> None:
    """Persist distillation run config plus teacher provenance near the checkpoints.

    Args:
        log_dir: Run directory where the metadata file should be written.
        cfg: Resolved distillation config for this run.
        teacher_metadata: Explicit teacher provenance dictionary for this run.

    Returns:
        None. Writes `distill_run_config.json` into `log_dir`.
    """
    payload = {
        "run": {
            "algo": "hora_distill",
            "task": str(OmegaConf.select(cfg, "training.task_name")),
            "sim_backend": str(OmegaConf.select(cfg, "training.sim_backend")),
            "log_dir": str(log_dir),
            "teacher": teacher_metadata,
        },
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    with (log_dir / "distill_run_config.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)
        f.write("\n")


def _resolve_teacher_checkpoint_path(cfg: DictConfig) -> tuple[Path | None, Path | None]:
    teacher_algo_family, teacher_task = _get_teacher_owner_spec(cfg)
    if teacher_algo_family is None or teacher_task is None:
        return None, None

    teacher_cfg = _load_teacher_owner_config(teacher_algo_family, teacher_task)
    teacher_task_name = OmegaConf.select(teacher_cfg, "training.task_name")
    teacher_algo_log_name = OmegaConf.select(teacher_cfg, "algo.algo_log_name")
    if teacher_task_name in (None, "") or teacher_algo_log_name in (None, ""):
        raise ValueError(
            "Teacher owner config must define training.task_name and algo.algo_log_name. "
            f"Got algo_family={teacher_algo_family} task={teacher_task}."
        )

    return resolve_task_checkpoint_path(
        ROOT_DIR,
        task_name=str(teacher_task_name),
        load_run=str(OmegaConf.select(cfg, "algo.load_run", default="-1")),
        algo_log_name=str(teacher_algo_log_name),
        checkpoint=(
            str(selected_checkpoint)
            if (selected_checkpoint := OmegaConf.select(cfg, "algo.checkpoint", default=-1))
            not in (None, "", -1, "-1")
            else None
        ),
        suffix=".pt",
        log_root=OmegaConf.select(cfg, "training.log_root"),
    )


def _build_env_cfg_override(cfg: DictConfig) -> dict[str, Any]:
    adapter = BackendAdapter(
        cfg,
        root_dir=ROOT_DIR,
        algo_name="hora_distill",
        scene_materializer=materialize_scene_visual_override,
    )
    return cast(dict[str, Any], adapter.build_task_env_cfg_override())


def _build_play_env_cfg_override(cfg: DictConfig) -> dict[str, Any]:
    adapter = BackendAdapter(
        cfg,
        root_dir=ROOT_DIR,
        algo_name="hora_distill",
        scene_materializer=materialize_scene_visual_override,
    )
    return cast(dict[str, Any], adapter.build_play_env_cfg_override())


def _resolve_stage2_checkpoint_path(cfg: DictConfig) -> tuple[Path | None, Path | None]:
    task_log_root = get_log_root(ROOT_DIR, cfg) / str(cfg.training.task_name)
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


def _format_stage2_play_checkpoint_error(
    cfg: DictConfig,
    *,
    task_log_root: Path,
    load_path: Path | None,
    load_path_dir: Path | None,
) -> str:
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


def _student_policy(
    actor, hist_normalizer, obs: TensorDict, *, device: torch.device
) -> torch.Tensor:
    proprio_hist = hist_normalizer(obs["proprio_hist"].to(device), update=False)
    policy_obs = TensorDict(
        {
            "actor": obs["actor"].to(device),
            "proprio_hist": proprio_hist,
        },
        batch_size=obs.batch_size,
        device=device,
    )
    return actor(policy_obs, stochastic_output=False).clamp_(-1.0, 1.0)


def _play_camera_kwargs(cfg: DictConfig) -> dict[str, Any]:
    camera_kwargs = {
        "cam_tracking": getattr(cfg.training, "cam_tracking", False),
        "cam_tracking_env_idx": getattr(cfg.training, "cam_tracking_env_idx", 0),
        "cam_tracking_extra_envs": getattr(cfg.training, "cam_tracking_extra_envs", 2),
    }
    for key in ("cam_distance", "cam_elevation", "cam_azimuth", "cam_lookat"):
        value = getattr(cfg.training, key, None)
        if value is not None:
            camera_kwargs[key] = value
    return camera_kwargs


def _cfg_with_checkpoint_runtime(cfg: DictConfig, checkpoint: dict[str, Any]) -> DictConfig:
    """Merge teacher-independent runtime config stored in a stage-2 checkpoint.

    Args:
        cfg: Hydra-composed distillation config supplied to play mode.
        checkpoint: Loaded stage-2 checkpoint dictionary.

    Returns:
        Config with checkpoint runtime fields restored for environment and model construction.
    """
    runtime_cfg = checkpoint.get("distill_runtime_cfg")
    if runtime_cfg is None:
        # Backward compatibility for older stage-2 checkpoints that did not
        # persist teacher-independent playback config.
        return _apply_teacher_defaults(cfg)
    # Hydra keeps the distillation root config structured, but runtime playback
    # metadata legitimately restores owner fields such as reward/env that are
    # absent from the bare distillation config.
    cfg_clone = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    return cast(DictConfig, OmegaConf.merge(cfg_clone, OmegaConf.create(runtime_cfg)))


def play_hora_distill(cfg: DictConfig, device: str) -> str | None:
    task_log_root = get_log_root(ROOT_DIR, cfg) / str(cfg.training.task_name)
    load_path, load_path_dir = _resolve_stage2_checkpoint_path(cfg)
    if load_path is None or load_path_dir is None or not load_path.exists():
        print(
            _format_stage2_play_checkpoint_error(
                cfg,
                task_log_root=task_log_root,
                load_path=load_path,
                load_path_dir=load_path_dir,
            )
        )
        return None

    print(f"Loading distilled model: {load_path}")
    checkpoint = torch.load(load_path, map_location="cpu", weights_only=False)
    if "model_state_dict" not in checkpoint:
        print(
            f"Checkpoint at {load_path} is not a HORA distillation checkpoint "
            f"(found keys: {set(checkpoint.keys())}). Aborting play."
        )
        return None

    cfg = _cfg_with_checkpoint_runtime(cfg, checkpoint)
    env = create_env(
        cfg,
        num_envs=int(cfg.training.play_env_num),
        env_cfg_override=_build_play_env_cfg_override(cfg),
    )
    wrapped_env = RslRlVecEnvWrapper(env, device=device, policy_obs_mode="actor")
    torch_device = torch.device(device)
    actor, hist_normalizer = build_student_actor_and_normalizer(
        wrapped_env,
        cfg,
        device=torch_device,
    )
    load_distilled_checkpoint(actor, hist_normalizer, load_path, device=torch_device)
    actor.eval()
    hist_normalizer.eval()

    if cfg.training.sim_backend == "motrix":
        raise NotImplementedError(
            "HORA distillation play_only currently supports offline MuJoCo video rendering only."
        )

    output_video = Path(load_path_dir) / "play_video_stage2.mp4"
    print(f"Rendering video to {output_video}...")
    print("Collecting physics states...")
    with torch.inference_mode():
        render_play_mode(
            env,
            sim_backend=cfg.training.sim_backend,
            render_spacing=float(
                getattr(cfg.training, "render_spacing", getattr(env.cfg, "render_spacing", 1.0))
            ),
            num_steps=int(cfg.training.play_steps),
            output_video=output_video,
            initialize=lambda: wrapped_env.reset()[0],
            step=lambda obs: wrapped_env.step(
                _student_policy(actor, hist_normalizer, obs, device=torch_device)
            )[0],
            camera_kwargs=_play_camera_kwargs(cfg),
        )
    print("Done.")
    return str(output_video)


@hydra.main(version_base="1.3", config_path="../conf/hora_distill", config_name="config")
def main(cfg: DictConfig) -> None:
    ensure_registries()

    if cfg.training.device:
        device = str(cfg.training.device)
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    if cfg.training.play_only:
        play_hora_distill(cfg, device)
        return

    cfg = _apply_teacher_defaults(cfg)
    teacher_algo_family, teacher_task = _get_teacher_owner_spec(cfg)
    if teacher_algo_family is None or teacher_task is None:
        raise ValueError("HORA distillation requires teacher.algo_family and teacher.task.")

    teacher_checkpoint, _ = _resolve_teacher_checkpoint_path(cfg)
    if teacher_checkpoint is None:
        raise FileNotFoundError(
            "Could not resolve HORA teacher checkpoint. "
            f"teacher.algo_family={teacher_algo_family!r} "
            f"teacher.task={teacher_task!r}. "
            "Set algo.load_run and optionally algo.checkpoint."
        )

    teacher_metadata = _teacher_run_metadata(
        cfg,
        teacher_algo_family=teacher_algo_family,
        teacher_checkpoint=teacher_checkpoint,
    )
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_root = Path(cfg.training.log_dir) if cfg.training.log_dir else get_log_root(ROOT_DIR, cfg)
    run_name = f"{timestamp}_{cfg.training.sim_backend}_{teacher_metadata['run_slug']}"
    log_dir = log_root / str(cfg.training.task_name) / run_name
    logger = setup_logger(log_dir, "hora_distill", echo=str(cfg.training.logger) != "no_print")
    _write_distill_run_config(log_dir, cfg=cfg, teacher_metadata=teacher_metadata)
    logger.info(
        "teacher_algo=%s teacher_task=%s teacher_checkpoint=%s",
        teacher_metadata["algo_family"],
        teacher_metadata["task"],
        teacher_metadata["checkpoint_path"],
    )

    env = create_env(
        cfg,
        num_envs=int(cfg.algo.num_envs),
        env_cfg_override=_build_env_cfg_override(cfg),
    )
    wrapped_env = RslRlVecEnvWrapper(env, device=device, policy_obs_mode="actor")
    trainer = HoraDistillationTrainer(
        wrapped_env,
        cfg,
        device=device,
        log_dir=log_dir,
        teacher_checkpoint=teacher_checkpoint,
        teacher_algo_family=teacher_algo_family,
        teacher_metadata=teacher_metadata,
        distill_runtime_cfg=_resolved_distill_runtime_cfg(cfg),
        logger=logger,
    )
    trainer.train()


if __name__ == "__main__":
    main()
