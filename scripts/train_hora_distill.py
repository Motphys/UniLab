import datetime
import sys
from pathlib import Path
from typing import Any, cast

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

ROOT_DIR = Path(__file__).parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from uni_rl.hora import HoraDistillationTrainer
from uni_rl.hora.distill import (
    build_student_actor_and_normalizer,
    cfg_with_checkpoint_runtime,
    load_distilled_checkpoint,
    student_policy,
)
from uni_rl.hora.rsl_rl import HoraRslRlVecEnvWrapper as RslRlVecEnvWrapper
from unisim.backend.base import log_playback_plan
from unisim.backend.mujoco.xml import materialize_scene_visual_override

from unilab.base.config_adapter import (
    BackendAdapter,
    create_env,
)
from unilab.training import (
    ensure_registries,
    format_hora_stage2_checkpoint_error,
    get_log_root,
    resolve_hora_stage2_checkpoint_path,
    setup_logger,
    should_run_playback,
)
from unilab.training.experiment import get_device_info_dict, write_run_config_snapshot
from unilab.training.hora_distill_config import (
    apply_teacher_defaults as _apply_teacher_defaults,
)
from unilab.training.hora_distill_config import (
    get_teacher_owner_spec as _get_teacher_owner_spec,
)
from unilab.training.hora_distill_config import (
    resolve_teacher_checkpoint_path as _resolve_teacher_checkpoint_path,
)
from unilab.training.hora_distill_config import (
    resolved_distill_runtime_cfg as _resolved_distill_runtime_cfg,
)
from unilab.training.hora_distill_config import (
    teacher_run_metadata as _teacher_run_metadata,
)


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
    write_run_config_snapshot(
        log_dir,
        run_metadata={
            "algo": "hora_distill",
            "task": str(OmegaConf.select(cfg, "training.task_name")),
            "sim_backend": str(OmegaConf.select(cfg, "training.sim_backend")),
            "log_dir": str(log_dir),
            "hardware": get_device_info_dict(),
            "teacher": teacher_metadata,
        },
        full_cfg=cfg,
        filename="distill_run_config.json",
        trailing_newline=True,
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


def play_hora_distill(cfg: DictConfig, device: str) -> str | None:
    task_log_root = get_log_root(ROOT_DIR, cfg) / str(cfg.training.task_name)
    load_path, load_path_dir = resolve_hora_stage2_checkpoint_path(cfg, root_dir=ROOT_DIR)
    if load_path is None or load_path_dir is None or not load_path.exists():
        print(
            format_hora_stage2_checkpoint_error(
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

    # uni_rl's cfg_with_checkpoint_runtime no longer composes teacher defaults;
    # the caller owns that composition (issue #1480).
    cfg = cfg_with_checkpoint_runtime(_apply_teacher_defaults(cfg), checkpoint)
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

    with torch.inference_mode():
        play_video_path = env.run_playback_mode(
            play_render_mode=getattr(cfg.training, "play_render_mode", "auto"),
            play_steps=getattr(cfg.training, "play_steps", None),
            output_video=Path(load_path_dir) / "play_video_stage2.mp4",
            render_spacing=float(
                getattr(cfg.training, "render_spacing", getattr(env.cfg, "render_spacing", 1.0))
            ),
            initialize=lambda: wrapped_env.reset()[0],
            step=lambda obs: wrapped_env.step(
                student_policy(actor, hist_normalizer, obs, device=torch_device)
            )[0],
            camera_kwargs=_play_camera_kwargs(cfg),
            on_plan=log_playback_plan,
        )
    print("Done.")
    return play_video_path


@hydra.main(version_base="1.3", config_path="../src/unilab/conf/hora_distill", config_name="config")
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

    if should_run_playback(
        play_only=cfg.training.play_only,
        no_play=True,
        play_render_mode=getattr(cfg.training, "play_render_mode", "auto"),
    ):
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
