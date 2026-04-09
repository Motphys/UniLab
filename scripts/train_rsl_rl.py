import datetime
import importlib
import os
import pkgutil
import statistics
import sys
import time
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict

ROOT_DIR = Path(__file__).parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from unilab.utils.experiment_tracking import ExperimentTracker, patch_rsl_rl_wandb_writer
from unilab.utils.obs_utils import flatten_obs_dict
from unilab.utils.reward_utils import resolve_reward_dict
from unilab.utils.rsl_rl_vec_env_wrapper import RslRlVecEnvWrapper
from unilab.utils.torch_utils import to_numpy, to_torch
from unilab.utils.xml_utils import materialize_scene_visual_override

try:
    from rsl_rl.runners import OnPolicyRunner
except ImportError:
    print("Could not import rsl_rl. Please ensure it is installed.")
    sys.exit(1)

from unilab.utils.rsl_rl_compat import (
    convert_config_v3_to_v4,
    convert_config_v5,
    is_rsl_rl_v4,
    is_rsl_rl_v5,
)
from unilab.utils.run_utils import get_latest_run


def ensure_registries():
    for pkg_name in (
        "unilab.envs.locomotion",
        "unilab.envs.manipulation",
        "unilab.envs.motion_tracking",
    ):
        try:
            package = importlib.import_module(pkg_name)
            if hasattr(package, "__path__"):
                for _, name, _ in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
                    try:
                        importlib.import_module(name)
                    except Exception:
                        pass
        except ImportError:
            pass


def _explicit_cli_override_keys() -> set[str]:
    return {arg.split("=", 1)[0] for arg in sys.argv[1:] if "=" in arg}


def _apply_task_algo_overrides(
    target, overrides, *, base_path: str, explicit_keys: set[str]
) -> None:
    if OmegaConf.is_config(overrides):
        override_dict = OmegaConf.to_container(overrides, resolve=True)
    elif isinstance(overrides, dict):
        override_dict = overrides
    else:
        return

    for key, value in override_dict.items():
        path = f"{base_path}.{key}"
        if isinstance(value, dict):
            if path in explicit_keys:
                continue
            if key not in target or target[key] is None:
                target[key] = {}
            _apply_task_algo_overrides(
                target[key], value, base_path=path, explicit_keys=explicit_keys
            )
            continue
        if path not in explicit_keys:
            target[key] = value


def _apply_task_env_profile(
    env_cfg_override: dict, env_profile, *, explicit_keys: set[str]
) -> None:
    if OmegaConf.is_config(env_profile):
        env_profile_dict = OmegaConf.to_container(env_profile, resolve=True)
    elif isinstance(env_profile, dict):
        env_profile_dict = env_profile
    else:
        return

    reward_explicit = "reward" in explicit_keys or any(
        k.startswith("reward.") for k in explicit_keys
    )
    for key, value in env_profile_dict.items():
        if key == "reward_config" and reward_explicit:
            continue
        env_cfg_override[key] = value


def build_task_motrix_ppo_env_cfg_override(cfg: DictConfig) -> dict:
    env_cfg_override = {}
    motrix_legacy = getattr(cfg, "motrix_legacy", None)
    explicit_keys = _explicit_cli_override_keys()
    env_cfg_override["reward_config"] = resolve_reward_dict(cfg)
    if motrix_legacy is None or not motrix_legacy.enabled or cfg.training.sim_backend != "motrix":
        return env_cfg_override

    algo_overrides = getattr(motrix_legacy, "algo_overrides", None)
    if algo_overrides is not None:
        _apply_task_algo_overrides(
            cfg.algo,
            algo_overrides,
            base_path="algo",
            explicit_keys=explicit_keys,
        )

    env_profile = getattr(motrix_legacy, "env_cfg_override", None)
    if env_profile is not None:
        _apply_task_env_profile(
            env_cfg_override,
            env_profile,
            explicit_keys=explicit_keys,
        )
    return env_cfg_override


def _resolve_root_relative_path(path_value: str) -> str:
    candidate = Path(path_value)
    if candidate.is_absolute():
        return str(candidate)
    return str((ROOT_DIR / candidate).resolve())


def build_motrix_play_ppo_env_cfg_override(cfg: DictConfig) -> dict:
    env_cfg_override = build_task_motrix_ppo_env_cfg_override(cfg)
    explicit_keys = _explicit_cli_override_keys()
    play_profile = getattr(cfg, "motrix_play_only", None)
    if (
        play_profile is None
        or not getattr(play_profile, "enabled", False)
        or cfg.training.sim_backend != "motrix"
        or not cfg.training.play_only
    ):
        return env_cfg_override

    training_overrides = getattr(play_profile, "training_overrides", None)
    if training_overrides is not None:
        _apply_task_algo_overrides(
            cfg.training,
            training_overrides,
            base_path="training",
            explicit_keys=explicit_keys,
        )

    env_profile = getattr(play_profile, "env_cfg_override", None)
    if env_profile is not None:
        _apply_task_env_profile(
            env_cfg_override,
            env_profile,
            explicit_keys=explicit_keys,
        )

    scene_override = getattr(play_profile, "scene_override", None)
    if scene_override is None or not getattr(scene_override, "enabled", False):
        return env_cfg_override

    source_model_file = getattr(scene_override, "source_model_file", None)
    if not source_model_file:
        raise ValueError("motrix_play_only.scene_override.source_model_file must be configured")

    env_cfg_override["model_file"] = materialize_scene_visual_override(
        _resolve_root_relative_path(str(source_model_file)),
        ground_texture_file=(
            _resolve_root_relative_path(str(scene_override.ground_texture_file))
            if getattr(scene_override, "ground_texture_file", None)
            else None
        ),
        ground_texrepeat=getattr(scene_override, "ground_texrepeat", None),
        skybox_rgb1=getattr(scene_override, "skybox_rgb1", None),
        skybox_rgb2=getattr(scene_override, "skybox_rgb2", None),
    )
    return env_cfg_override


def run_motrix_rsl_play_loop(
    wrapped_env,
    policy,
    *,
    render_spacing: float,
    num_steps: int | None = None,
) -> None:
    import time

    env = wrapped_env.env
    env._backend.init_renderer(spacing=render_spacing)
    obs, _ = wrapped_env.reset()

    last_render_time = time.perf_counter()
    render_dt = 1.0 / 60.0
    steps_run = 0

    with torch.inference_mode():
        while num_steps is None or steps_run < num_steps:
            actions = policy(obs)
            obs, _, _, _ = wrapped_env.step(actions)

            current_time = time.perf_counter()
            elapsed = current_time - last_render_time
            if elapsed < render_dt:
                time.sleep(render_dt - elapsed)
            last_render_time = time.perf_counter()

            env._backend.render()
            steps_run += 1


def play_rsl_rl(cfg: DictConfig, device: str) -> str | None:
    """Play mode for RSL-RL."""

    from unilab.base import registry

    env_cfg_override = build_motrix_play_ppo_env_cfg_override(cfg)

    env = registry.make(
        cfg.training.task_name,
        num_envs=cfg.training.play_env_num,
        sim_backend=cfg.training.sim_backend,
        env_cfg_override=env_cfg_override,
    )
    wrapped_env = RslRlVecEnvWrapper(env, device=device)
    train_cfg = OmegaConf.to_container(cfg.algo, resolve=True)
    if "runner" not in train_cfg:
        train_cfg["runner"] = {}
    train_cfg["runner"]["logger"] = "none"
    if is_rsl_rl_v5():
        train_cfg = convert_config_v5(train_cfg)

    base_log_dir = ROOT_DIR / "logs" / "rsl_rl_train" / cfg.training.task_name
    load_path = None

    load_run = cfg.training.load_run
    if load_run == "-1":
        load_path = get_latest_run(str(base_log_dir))
    else:
        if os.path.exists(load_run):
            load_path = load_run
        else:
            load_path = str(base_log_dir / load_run)

    if not load_path or not os.path.exists(load_path):
        print(f"Could not find run to load at {load_path}")
        return None

    if os.path.isdir(load_path):
        model_files = [
            f for f in os.listdir(load_path) if f.startswith("model_") and f.endswith(".pt")
        ]
        if len(model_files) > 0:
            model_files.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))
            load_path_dir = load_path
            load_path = os.path.join(load_path, model_files[-1])
            print(f"Loading latest model: {load_path}")
        else:
            print(f"No model files found in {load_path}")
            return None
    else:
        load_path_dir = os.path.dirname(load_path)

    _ckpt_keys = set(torch.load(load_path, map_location="cpu", weights_only=True).keys())
    if "actor_state_dict" not in _ckpt_keys:
        print(
            f"Checkpoint at {load_path} is not an rsl-rl checkpoint "
            f"(found keys: {_ckpt_keys}). Aborting play."
        )
        return None

    runner = OnPolicyRunner(wrapped_env, train_cfg, log_dir=None, device=device)
    runner.load(load_path)
    policy = runner.get_inference_policy(device=device)

    if cfg.training.sim_backend == "motrix":
        print("Starting interactive visualization (motrix native renderer)...")
        print("Close the render window to exit.")
        with torch.inference_mode():
            try:
                run_motrix_rsl_play_loop(
                    wrapped_env=wrapped_env,
                    policy=policy,
                    render_spacing=float(getattr(env.cfg, "render_spacing", 1.0)),
                )
            except Exception as e:
                if "RenderClosedError" in str(type(e).__name__):
                    print("Render window closed.")
                else:
                    raise
    else:
        import mediapy as media

        from unilab.utils import render_many

        output_video = Path(load_path_dir) / "play_video.mp4"
        print(f"Rendering video to {output_video}...")

        obs, _ = wrapped_env.reset()
        state_list = []

        print("Collecting physics states...")
        with torch.inference_mode():
            for _ in range(cfg.training.play_steps):
                actions = policy(obs)
                obs, _, _, _ = wrapped_env.step(actions)
                state_list.append(to_numpy(env._backend.get_physics_state()).copy())

        print("Rendering frames...")
        frames = render_many.render_states_get_frames(
            state_list,
            env.cfg.model_file,
            width=1280,
            height=720,
            camera_id=-1,
            cam_distance=cfg.training.cam_distance,
            cam_elevation=cfg.training.cam_elevation,
            cam_azimuth=cfg.training.cam_azimuth,
        )

        print(f"Saving video to {output_video} with mediapy...")
        media.write_video(str(output_video), frames, fps=int(1.0 / env.cfg.ctrl_dt))
        print("Done.")
        return str(output_video)

    return None


@hydra.main(version_base="1.3", config_path="../conf/ppo", config_name="config")
def main(cfg: DictConfig) -> None:
    ensure_registries()

    from omegaconf import OmegaConf

    from unilab.base import registry

    env_cfg_override = build_task_motrix_ppo_env_cfg_override(cfg)

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
        log_dir = str(
            ROOT_DIR
            / "logs"
            / "rsl_rl_train"
            / cfg.training.task_name
            / f"{timestamp}_{cfg.training.sim_backend}"
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
            env = registry.make(
                cfg.training.task_name,
                num_envs=cfg.algo.num_envs,
                sim_backend=cfg.training.sim_backend,
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

            if cfg.training.load_run != "-1":
                if os.path.exists(cfg.training.load_run):
                    resume_path = cfg.training.load_run
                else:
                    base_log_dir = ROOT_DIR / "logs" / "rsl_rl_train" / cfg.training.task_name
                    run_path = base_log_dir / cfg.training.load_run
                    resume_path = str(run_path) if run_path.exists() else None

                if resume_path:
                    print(f"Resuming from {resume_path}")
                    runner.load(resume_path)

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
    main()
