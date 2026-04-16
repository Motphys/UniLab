import datetime
import statistics
import sys
import time
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

ROOT_DIR = Path(__file__).parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from unilab.training import (
    BackendAdapter,
    create_env,
    ensure_registries,
    get_log_root,
    parse_checkpoint_path,
    render_play_mode,
)
from unilab.utils.experiment_tracking import ExperimentTracker, patch_rsl_rl_wandb_writer
from unilab.utils.rsl_rl_vec_env_wrapper import RslRlVecEnvWrapper
from unilab.utils.xml_utils import materialize_scene_visual_override

try:
    from rsl_rl.runners import OnPolicyRunner
except ImportError:
    print("Could not import rsl_rl. Please ensure it is installed.")
    sys.exit(1)

from unilab.utils.rsl_rl_compat import convert_config_v5, is_rsl_rl_v5


def _backend_adapter(cfg: DictConfig) -> BackendAdapter:
    return BackendAdapter(
        cfg,
        root_dir=ROOT_DIR,
        algo_name="ppo",
        scene_materializer=materialize_scene_visual_override,
    )


def build_ppo_env_cfg_override(cfg: DictConfig) -> dict:
    return _backend_adapter(cfg).build_task_env_cfg_override()


def build_ppo_play_env_cfg_override(cfg: DictConfig) -> dict:
    return _backend_adapter(cfg).build_play_env_cfg_override()


def run_motrix_rsl_play_loop(
    wrapped_env,
    policy,
    *,
    render_spacing: float,
    num_steps: int | None = None,
) -> None:
    env = wrapped_env.env

    with torch.inference_mode():
        render_play_mode(
            env,
            sim_backend="motrix",
            render_spacing=render_spacing,
            num_steps=num_steps,
            initialize=lambda: wrapped_env.reset()[0],
            step=lambda obs: wrapped_env.step(policy(obs))[0],
        )


def _get_log_root(cfg: DictConfig) -> str:
    return str(get_log_root(ROOT_DIR, cfg))


def play_rsl_rl(cfg: DictConfig, device: str) -> str | None:
    """Play mode for RSL-RL."""

    env_cfg_override = build_ppo_play_env_cfg_override(cfg)

    env = create_env(
        cfg,
        num_envs=cfg.training.play_env_num,
        env_cfg_override=env_cfg_override,
    )
    wrapped_env = RslRlVecEnvWrapper(env, device=device)
    train_cfg = OmegaConf.to_container(cfg.algo, resolve=True)
    if "runner" not in train_cfg:
        train_cfg["runner"] = {}
    train_cfg["runner"]["logger"] = "none"
    if is_rsl_rl_v5():
        train_cfg = convert_config_v5(train_cfg)

    load_path, load_path_dir = parse_checkpoint_path(cfg, root_dir=ROOT_DIR)
    if load_path is None or load_path_dir is None or not load_path.exists():
        print(f"Could not find run to load at {load_path}")
        return None

    print(f"Loading latest model: {load_path}")
    _ckpt_keys = set(torch.load(load_path, map_location="cpu", weights_only=True).keys())
    if "actor_state_dict" not in _ckpt_keys:
        print(
            f"Checkpoint at {load_path} is not an rsl-rl checkpoint "
            f"(found keys: {_ckpt_keys}). Aborting play."
        )
        return None

    runner = OnPolicyRunner(wrapped_env, train_cfg, log_dir=None, device=device)
    runner.load(str(load_path))
    policy = runner.get_inference_policy(device=device)
    if EXPORT_POLICY:
        runner.export_policy_to_onnx(path=load_path_dir)
        runner.export_policy_to_jit(path=load_path_dir)
    if cfg.training.sim_backend == "motrix":
        print("Starting interactive visualization (motrix native renderer)...")
        print("Close the render window to exit.")
        with torch.inference_mode():
            try:
                run_motrix_rsl_play_loop(
                    wrapped_env=wrapped_env,
                    policy=policy,
                    render_spacing=float(
                        getattr(
                            cfg.training, "render_spacing", getattr(env.cfg, "render_spacing", 1.0)
                        )
                    ),
                )
            except Exception as e:
                if "RenderClosedError" in str(type(e).__name__):
                    print("Render window closed.")
                else:
                    raise
    else:
        output_video = Path(load_path_dir) / "play_video.mp4"
        print(f"Rendering video to {output_video}...")

        print("Collecting physics states...")
        with torch.inference_mode():
            render_play_mode(
                env,
                sim_backend=cfg.training.sim_backend,
                render_spacing=float(
                    getattr(cfg.training, "render_spacing", getattr(env.cfg, "render_spacing", 1.0))
                ),
                num_steps=cfg.training.play_steps,
                output_video=output_video,
                initialize=lambda: wrapped_env.reset()[0],
                step=lambda obs: wrapped_env.step(policy(obs))[0],
                camera_kwargs={
                    "cam_distance": cfg.training.cam_distance,
                    "cam_elevation": cfg.training.cam_elevation,
                    "cam_azimuth": cfg.training.cam_azimuth,
                    "cam_lookat": getattr(cfg.training, "cam_lookat", None),
                },
            )
        print("Done.")
        return str(output_video)

    return None


@hydra.main(version_base="1.3", config_path="../conf/ppo", config_name="config")
def main(cfg: DictConfig) -> None:
    ensure_registries()

    env_cfg_override = build_ppo_env_cfg_override(cfg)

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    # Compute effective max_iterations (supports num_timesteps override)
    max_iterations = cfg.algo.max_iterations
    if cfg.training.num_timesteps:
        n_steps_per_iter = cfg.algo.num_steps_per_env * cfg.algo.num_envs
        max_iterations = max(1, int(cfg.training.num_timesteps / n_steps_per_iter))
        print(
            f"Overriding max_iterations to {max_iterations} based on "
            f"num_timesteps {cfg.training.num_timesteps}"
        )

    if not cfg.training.play_only:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_root = _get_log_root(cfg)
        log_dir = str(
            Path(log_root) / cfg.training.task_name / f"{timestamp}_{cfg.training.sim_backend}"
        )
    else:
        log_dir = None

    tracker = None
    if not cfg.training.play_only and log_dir is not None:
        tracker = ExperimentTracker(
            root_dir=ROOT_DIR,
            log_dir=log_dir,
            algo_name="ppo",
            task_name=cfg.training.task_name,
            sim_backend=cfg.training.sim_backend,
            training_cfg=cfg.training,
            full_cfg=cfg,
            device=device,
        )
        tracker.start()

    try:
        if not cfg.training.play_only:
            env = create_env(
                cfg,
                num_envs=cfg.algo.num_envs,
                env_cfg_override=env_cfg_override,
            )
            wrapped_env = RslRlVecEnvWrapper(env, device=device)

            train_cfg = OmegaConf.to_container(cfg.algo, resolve=True)
            if "runner" not in train_cfg:
                train_cfg["runner"] = {}

            logger_type = (
                cfg.training.logger if cfg.training.logger in ["tensorboard", "wandb"] else "none"
            )
            train_cfg["runner"]["logger"] = logger_type
            train_cfg["logger"] = logger_type

            if tracker is not None and logger_type == "wandb":
                patch_rsl_rl_wandb_writer()
                wandb_settings = tracker.wandb_settings
                train_cfg["wandb_project"] = wandb_settings["project"]
                train_cfg["wandb_entity"] = wandb_settings["entity"]
                train_cfg["wandb_group"] = wandb_settings["group"]
                train_cfg["wandb_job_type"] = wandb_settings["job_type"]
                train_cfg["wandb_tags"] = wandb_settings["tags"]
                train_cfg["wandb_notes"] = wandb_settings["notes"]
                train_cfg["wandb_mode"] = wandb_settings["mode"]

            if is_rsl_rl_v5():
                train_cfg = convert_config_v5(train_cfg)

            runner = OnPolicyRunner(wrapped_env, train_cfg, log_dir=log_dir, device=device)

            if cfg.algo.load_run != "-1":
                resume_path, _ = parse_checkpoint_path(cfg, root_dir=ROOT_DIR)
                if resume_path:
                    print(f"Resuming from {resume_path}")
                    runner.load(str(resume_path))

            train_start_wall = time.time()
            runner.learn(num_learning_iterations=max_iterations, init_at_random_ep_len=True)
            train_summary = {
                "status": "completed",
                "completed_iterations": int(runner.current_learning_iteration),
                "total_env_steps": int(getattr(runner.logger, "tot_timesteps", 0)),
                "final_mean_reward": (
                    float(statistics.mean(runner.logger.rewbuffer))
                    if len(getattr(runner.logger, "rewbuffer", [])) > 0
                    else None
                ),
                "best_mean_reward": (
                    float(max(runner.logger.rewbuffer))
                    if len(getattr(runner.logger, "rewbuffer", [])) > 0
                    else None
                ),
                "mean_episode_length": (
                    float(statistics.mean(runner.logger.lenbuffer))
                    if len(getattr(runner.logger, "lenbuffer", [])) > 0
                    else None
                ),
                "last_checkpoint": str(
                    Path(log_dir) / f"model_{int(runner.current_learning_iteration)}.pt"
                ),
                "training_wall_time_sec": time.time() - train_start_wall,
            }
            if tracker is not None:
                tracker.update_summary(train_summary)
            env.close()

        if cfg.training.play_only or not cfg.training.no_play:
            play_video_path = play_rsl_rl(cfg, device)
            if tracker is not None:
                tracker.log_video(play_video_path)
    finally:
        if tracker is not None:
            tracker.finish()


if __name__ == "__main__":
    EXPORT_POLICY = True
    main()
