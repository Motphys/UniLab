"""Train APPO agent — native multiprocessing."""

from __future__ import annotations

import datetime
import importlib
import os
import pkgutil
import sys
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR))

from unilab.utils.experiment_tracking import ExperimentTracker


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


def build_appo_runner_kwargs(
    cfg: DictConfig,
    env_cfg_override: dict | None,
    collector_device: str | None,
    rl_cfg: dict | None = None,
) -> dict:
    if rl_cfg is None:
        rl_cfg = OmegaConf.to_container(cfg.algo, resolve=True)

    runner_kwargs = {
        "env_name": cfg.training.task_name,
        "env_cfg_overrides": env_cfg_override,
        "rl_cfg": rl_cfg,
        "device": cfg.training.device,
        "collector_device": collector_device,
        "num_envs": cfg.algo.num_envs,
        "steps_per_env": cfg.algo.steps_per_env,
        "sim_backend": cfg.training.sim_backend,
    }
    if cfg.training.replay_queue_size is not None:
        runner_kwargs["replay_queue_size"] = cfg.training.replay_queue_size
    return runner_kwargs


def run_motrix_play_loop(
    env,
    actor,
    device: str,
    play_env_num: int,
    num_steps: int | None = None,
) -> None:
    import time

    import numpy as np
    from tensordict import TensorDict

    if env.state is None:
        env.init_state()
    env._backend.init_renderer()

    env_indices = np.arange(play_env_num, dtype=np.int32)
    obs_out, _ = env.reset(env_indices)
    obs_np = np.asarray(obs_out["obs"], dtype=np.float32)

    last_render_time = time.perf_counter()
    render_dt = 1.0 / 60.0
    steps_run = 0

    with torch.inference_mode():
        while num_steps is None or steps_run < num_steps:
            obs_torch = torch.from_numpy(obs_np).to(device)
            td = TensorDict({"policy": obs_torch}, batch_size=play_env_num)
            actions_torch = actor(td)
            actions_np = actions_torch.cpu().numpy().astype(np.float32)
            state = env.step(actions_np)
            obs_np = np.asarray(state.obs["obs"], dtype=np.float32)

            current_time = time.perf_counter()
            elapsed = current_time - last_render_time
            if elapsed < render_dt:
                time.sleep(render_dt - elapsed)
            last_render_time = time.perf_counter()

            env._backend.render()
            steps_run += 1


def resolve_appo_checkpoint_path(
    base_log_dir: str | Path,
    load_run: str,
) -> tuple[str | None, str | None]:
    base_log_dir = str(base_log_dir)

    if load_run == "-1":
        if not os.path.exists(base_log_dir):
            return None, None
        all_runs = sorted(
            [d for d in os.listdir(base_log_dir) if os.path.isdir(os.path.join(base_log_dir, d))]
        )
        if not all_runs:
            return None, None
        candidate_dir = os.path.join(base_log_dir, all_runs[-1])
    elif os.path.isdir(load_run):
        candidate_dir = load_run
    elif os.path.isfile(load_run):
        return load_run, os.path.dirname(load_run)
    else:
        candidate_dir = os.path.join(base_log_dir, load_run)
        if not os.path.isdir(candidate_dir):
            return None, None

    model_files = [
        f for f in os.listdir(candidate_dir) if f.startswith("model_") and f.endswith(".pt")
    ]
    if not model_files:
        return None, None

    model_files.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))
    return os.path.join(candidate_dir, model_files[-1]), candidate_dir


def _get_log_root(cfg: DictConfig) -> str:
    """Get log root directory from algo_log_name config."""
    return str(Path(ROOT_DIR) / "logs" / cfg.algo.algo_log_name)


def play_appo(cfg: DictConfig, rl_cfg: dict) -> str | None:
    """Play mode for APPO."""
    import numpy as np
    from rsl_rl.utils import resolve_callable
    from tensordict import TensorDict

    from unilab.base import registry
    from unilab.utils.reward_utils import extract_reward_config
    from unilab.utils.rsl_rl_compat import convert_config_v3_to_v4, is_rsl_rl_v4, is_rsl_rl_v5

    env_cfg_override = extract_reward_config(cfg)

    device = cfg.training.device or (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device for play: {device}")

    env = registry.make(
        cfg.training.task_name,
        num_envs=cfg.training.play_env_num,
        sim_backend=cfg.training.sim_backend,
        env_cfg_override=env_cfg_override,
    )
    from unilab.utils.obs_utils import get_obs_dims

    obs_dim, privileged_dim = get_obs_dims(env.obs_groups_spec)
    action_dim = env.action_space.shape[0]

    rl_cfg_dict = dict(rl_cfg)
    if "obs_groups" not in rl_cfg_dict:
        rl_cfg_dict["obs_groups"] = {"actor": {"policy": obs_dim}}
    else:
        actor_group = rl_cfg_dict["obs_groups"].get(
            "actor", rl_cfg_dict["obs_groups"].get("policy", {})
        )
        if isinstance(actor_group, dict) and "policy" in actor_group:
            actor_group["policy"] = obs_dim

    if is_rsl_rl_v5():
        pass
    elif is_rsl_rl_v4():
        rl_cfg_dict = convert_config_v3_to_v4(rl_cfg_dict)

    from copy import deepcopy

    obs_example = torch.zeros((cfg.training.play_env_num, obs_dim), device=device)
    td_example = TensorDict({"policy": obs_example}, batch_size=cfg.training.play_env_num)

    actor_cfg = deepcopy(rl_cfg_dict["actor"])
    actor_cls = resolve_callable(actor_cfg.pop("class_name"))
    actor_cfg.pop("num_actions", None)
    actor = actor_cls(td_example, rl_cfg_dict["obs_groups"], "actor", action_dim, **actor_cfg)
    actor = actor.to(device)
    actor.eval()

    log_root = _get_log_root(cfg)
    base_log_dir = os.path.join(log_root, cfg.training.task_name)
    load_path, load_path_dir = resolve_appo_checkpoint_path(base_log_dir, cfg.algo.load_run)

    if not load_path or not os.path.exists(load_path):
        print(f"Could not find run to load. load_path={load_path}")
        return None

    print(f"Loading model: {load_path}")
    checkpoint = torch.load(load_path, map_location=device, weights_only=True)
    actor.load_state_dict(checkpoint["actor"])

    if cfg.training.sim_backend == "motrix":
        print("Starting interactive visualization (motrix native renderer)...")
        print("Close the render window to exit.")
        try:
            run_motrix_play_loop(
                env=env,
                actor=actor,
                device=device,
                play_env_num=cfg.training.play_env_num,
            )
        except Exception as e:
            if "RenderClosedError" in str(type(e).__name__):
                print("Render window closed.")
            else:
                raise
        return None

    import mediapy as media

    from unilab.utils import render_many

    output_video = os.path.join(load_path_dir, "play_video.mp4")
    print(f"Rendering video to {output_video}...")

    if env.state is None:
        env.init_state()
    env_indices = np.arange(cfg.training.play_env_num, dtype=np.int32)
    obs_out, _ = env.reset(env_indices)
    obs_np = np.asarray(obs_out["obs"], dtype=np.float32)

    state_list = []
    num_steps = int(getattr(cfg.training, "play_steps", 1000))

    print("Collecting physics states...")
    with torch.inference_mode():
        for _ in range(num_steps):
            obs_torch = torch.from_numpy(obs_np).to(device)
            td = TensorDict({"policy": obs_torch}, batch_size=cfg.training.play_env_num)
            actions_torch = actor(td)
            actions_np = actions_torch.cpu().numpy().astype(np.float32)
            state = env.step(actions_np)
            obs_np = np.asarray(state.obs["obs"], dtype=np.float32)
            state_list.append(np.asarray(env._backend.get_physics_state(), dtype=np.float32).copy())

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
    return output_video


@hydra.main(version_base="1.3", config_path="../conf/appo", config_name="config")
def main(cfg: DictConfig) -> None:
    ensure_registries()

    from unilab.utils.reward_utils import extract_reward_config

    env_cfg_override = extract_reward_config(cfg)

    # Convert algo config to plain dict for APPORunner / RSL-RL internals
    rl_cfg = OmegaConf.to_container(cfg.algo, resolve=True)

    if cfg.training.log_dir is None:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_root = _get_log_root(cfg)
        log_dir = os.path.join(
            log_root,
            cfg.training.task_name,
            f"{timestamp}_{cfg.training.sim_backend}",
        )
    else:
        log_dir = cfg.training.log_dir

    collector_device = cfg.training.collector_device
    if collector_device == "gpu":
        collector_device = "mps" if torch.backends.mps.is_available() else "cuda"

    learner_device = cfg.training.device or (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    tracker = None
    if not cfg.training.play_only:
        tracker = ExperimentTracker(
            root_dir=ROOT_DIR,
            log_dir=log_dir,
            algo_name="appo",
            task_name=cfg.training.task_name,
            sim_backend=cfg.training.sim_backend,
            training_cfg=cfg.training,
            full_cfg=cfg,
            device=learner_device,
            collector_device=collector_device,
        )
        tracker.start()

    try:
        if not cfg.training.play_only:
            from unilab.algos.torch.appo.runner import APPORunner

            runner = APPORunner(
                **build_appo_runner_kwargs(
                    cfg,
                    env_cfg_override=env_cfg_override,
                    collector_device=collector_device,
                    rl_cfg=rl_cfg,
                )
            )

            try:
                runner.learn(
                    max_iterations=cfg.algo.max_iterations,
                    save_interval=cfg.algo.save_interval,
                    log_dir=log_dir,
                    logger_type=cfg.training.logger,
                )
                if tracker is not None:
                    tracker.update_summary(getattr(runner, "last_run_summary", None))
            finally:
                runner.close()

        if cfg.training.play_only or not cfg.training.no_play:
            play_video_path = play_appo(cfg, rl_cfg)
            if tracker is not None:
                tracker.log_video(play_video_path)
    finally:
        if tracker is not None:
            tracker.finish()


if __name__ == "__main__":
    main()
