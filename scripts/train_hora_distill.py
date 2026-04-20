import datetime
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
from unilab.algos.torch.hora.distill import build_student_actor_and_normalizer, load_distilled_checkpoint
from unilab.training import (
    BackendAdapter,
    create_env,
    ensure_registries,
    get_latest_run,
    get_log_root,
    parse_checkpoint_path,
    render_play_mode,
    setup_logger,
)
from unilab.utils.rsl_rl_vec_env_wrapper import RslRlVecEnvWrapper
from unilab.utils.xml_utils import materialize_scene_visual_override


def _load_yaml_config(path: Path) -> DictConfig:
    loaded = OmegaConf.load(path)
    if not isinstance(loaded, DictConfig):
        raise TypeError(f"Expected DictConfig from {path}, got {type(loaded)!r}")
    return loaded


def _load_teacher_owner_config(ppo_task: str) -> DictConfig:
    owner_path = ROOT_DIR / "conf" / "ppo" / "task" / f"{ppo_task}.yaml"
    owner_cfg = _load_yaml_config(owner_path)
    merged_cfg = OmegaConf.create()
    for default_entry in owner_cfg.get("defaults", []):
        if not isinstance(default_entry, str) or default_entry == "_self_":
            continue
        include_path = ROOT_DIR / "conf" / "ppo" / f"{default_entry.lstrip('/')}.yaml"
        merged_cfg = OmegaConf.merge(merged_cfg, _load_yaml_config(include_path))
    return cast(DictConfig, OmegaConf.merge(merged_cfg, owner_cfg))


def _teacher_default_cfg(cfg: DictConfig) -> DictConfig:
    ppo_task = OmegaConf.select(cfg, "teacher_defaults.ppo_task")
    if not ppo_task:
        return OmegaConf.create()

    teacher_cfg = _load_teacher_owner_config(str(ppo_task))
    actor_cfg = OmegaConf.to_container(OmegaConf.select(teacher_cfg, "algo.actor"), resolve=True)
    if not isinstance(actor_cfg, dict):
        actor_cfg = {}
    actor_cfg = dict(actor_cfg)
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
        path
        for path in run_dir.glob("hora_stage2_*.pt")
        if path.stem.split("_")[-1].isdigit()
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


def _student_policy(actor, hist_normalizer, obs: TensorDict, *, device: torch.device) -> torch.Tensor:
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
    cfg = _apply_teacher_defaults(cfg)

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

    teacher_checkpoint, _ = parse_checkpoint_path(cfg, root_dir=ROOT_DIR)
    if teacher_checkpoint is None:
        raise FileNotFoundError(
            "Could not resolve HORA teacher checkpoint. Set algo.load_run and optionally algo.checkpoint."
        )

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_root = Path(cfg.training.log_dir) if cfg.training.log_dir else get_log_root(ROOT_DIR, cfg)
    log_dir = log_root / str(cfg.training.task_name) / f"{timestamp}_{cfg.training.sim_backend}"
    logger = setup_logger(log_dir, "hora_distill", echo=str(cfg.training.logger) != "no_print")

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
        logger=logger,
    )
    trainer.train()


if __name__ == "__main__":
    main()
