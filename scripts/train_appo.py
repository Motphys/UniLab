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


def ensure_registries():
    try:
        import unilab.envs.locomotion

        package = unilab.envs.locomotion
        if hasattr(package, "__path__"):
            for _, name, ispkg in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
                try:
                    importlib.import_module(name)
                except Exception:
                    pass
    except ImportError:
        pass

    try:
        import unilab.envs.locomotion.walking

        package = unilab.envs.locomotion.walking
        if hasattr(package, "__path__"):
            for _, name, ispkg in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
                try:
                    importlib.import_module(name)
                except Exception:
                    pass
    except ImportError:
        pass


def play_appo(cfg: DictConfig, rl_cfg: dict):
    """Play mode for APPO."""
    import mediapy as media
    import numpy as np
    from rsl_rl.utils import resolve_callable
    from tensordict import TensorDict

    from unilab.base import registry
    from unilab.utils import render_many
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

    base_log_dir = os.path.join(ROOT_DIR, "logs", "appo", cfg.training.task_name)
    load_path = None
    load_path_dir = None
    load_run = cfg.training.load_run
    if load_run == "-1":
        if os.path.exists(base_log_dir):
            all_runs = sorted(
                [
                    d
                    for d in os.listdir(base_log_dir)
                    if os.path.isdir(os.path.join(base_log_dir, d))
                ]
            )
            if all_runs:
                latest_run_dir = os.path.join(base_log_dir, all_runs[-1])
                model_files = [
                    f
                    for f in os.listdir(latest_run_dir)
                    if f.startswith("model_") and f.endswith(".pt")
                ]
                if model_files:
                    model_files.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))
                    load_path = os.path.join(latest_run_dir, model_files[-1])
                    load_path_dir = latest_run_dir
    else:
        if os.path.exists(load_run):
            load_path = load_run
            load_path_dir = os.path.dirname(load_path)
        else:
            potential_dir = os.path.join(base_log_dir, load_run)
            if os.path.isdir(potential_dir):
                model_files = [
                    f
                    for f in os.listdir(potential_dir)
                    if f.startswith("model_") and f.endswith(".pt")
                ]
                if model_files:
                    model_files.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))
                    load_path = os.path.join(potential_dir, model_files[-1])
                    load_path_dir = potential_dir

    if not load_path or not os.path.exists(load_path):
        print(f"Could not find run to load. load_path={load_path}")
        return

    print(f"Loading model: {load_path}")
    checkpoint = torch.load(load_path, map_location=device, weights_only=True)
    actor.load_state_dict(checkpoint["actor"])

    output_video = os.path.join(load_path_dir, "play_video.mp4")
    print(f"Rendering video to {output_video}...")

    if env.state is None:
        env.init_state()
    env_indices = np.arange(cfg.training.play_env_num, dtype=np.int32)
    from unilab.utils.obs_utils import flatten_obs_dict

    obs_out, _ = env.reset(env_indices)
    obs_np = np.asarray(obs_out["obs"], dtype=np.float32)

    state_list = []
    num_steps = 150

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
        state_list, env.cfg.model_file, width=1280, height=720, camera_id=-1
    )

    print(f"Saving video to {output_video} with mediapy...")
    media.write_video(str(output_video), frames, fps=int(1.0 / env.cfg.ctrl_dt))
    print("Done.")


@hydra.main(version_base="1.3", config_path="../conf/appo", config_name="config")
def main(cfg: DictConfig) -> None:
    ensure_registries()

    from unilab.utils.reward_utils import extract_reward_config

    env_cfg_override = extract_reward_config(cfg)

    # Convert algo config to plain dict for APPORunner / RSL-RL internals
    rl_cfg = OmegaConf.to_container(cfg.algo, resolve=True)

    if cfg.training.log_dir is None:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_dir = os.path.join(
            ROOT_DIR,
            "logs",
            "appo",
            cfg.training.task_name,
            f"{timestamp}_{cfg.training.sim_backend}",
        )
    else:
        log_dir = cfg.training.log_dir

    if not cfg.training.play_only:
        from unilab.algos.torch.appo.runner import APPORunner

        collector_device = cfg.training.collector_device
        if collector_device == "gpu":
            collector_device = "mps" if torch.backends.mps.is_available() else "cuda"

        runner_kwargs = {}
        if cfg.training.replay_queue_size is not None:
            runner_kwargs["replay_queue_size"] = cfg.training.replay_queue_size

        runner = APPORunner(
            env_name=cfg.training.task_name,
            env_cfg_overrides=env_cfg_override,
            rl_cfg=rl_cfg,
            device=cfg.training.device,
            collector_device=collector_device,
            num_envs=cfg.algo.num_envs,
            steps_per_env=cfg.algo.steps_per_env,
            **runner_kwargs,
        )

        try:
            runner.learn(
                max_iterations=cfg.algo.max_iterations,
                save_interval=cfg.algo.save_interval,
                log_dir=log_dir,
                logger_type=cfg.training.logger,
            )
        finally:
            runner.close()

    if cfg.training.play_only or not cfg.training.no_play:
        play_appo(cfg, rl_cfg)


if __name__ == "__main__":
    main()
