"""Train APPO agent — native multiprocessing."""

import argparse
import sys
import os
import datetime
from pathlib import Path
import torch
import pkgutil
import importlib

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


def play_appo(args, rl_cfg):
    """Play mode for APPO."""
    import torch
    import numpy as np
    import mediapy as media
    from tensordict import TensorDict
    from unilab.base import registry
    from unilab.utils import render_many
    from rsl_rl.utils import resolve_callable
    from unilab.utils.rsl_rl_compat import is_rsl_rl_v4, convert_config_v3_to_v4

    # Normalize to plain dict so ConfigDict doesn't cause isinstance issues
    if hasattr(rl_cfg, "to_dict"):
        rl_cfg = rl_cfg.to_dict()

    device = args.device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device for play: {device}")

    env = registry.make(args.task, num_envs=args.play_env_num, sim_backend="mujoco")
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    rl_cfg_dict = dict(rl_cfg)
    if "obs_groups" not in rl_cfg_dict:
        rl_cfg_dict["obs_groups"] = {"actor": {"policy": obs_dim}}
    else:
        actor_group = rl_cfg_dict["obs_groups"].get("actor", rl_cfg_dict["obs_groups"].get("policy", {}))
        if isinstance(actor_group, dict) and "policy" in actor_group:
            actor_group["policy"] = obs_dim

    if is_rsl_rl_v4():
        rl_cfg_dict = convert_config_v3_to_v4(rl_cfg_dict)

    obs_example = torch.zeros((args.play_env_num, obs_dim), device=device)
    td_example = TensorDict({"policy": obs_example}, batch_size=args.play_env_num)

    actor_cfg = rl_cfg_dict["actor"].copy()
    actor_cls = resolve_callable(actor_cfg.pop("class_name"))
    actor = actor_cls(td_example, rl_cfg_dict["obs_groups"], "actor", action_dim, **actor_cfg)
    actor = actor.to(device)
    actor.eval()

    base_log_dir = os.path.join(ROOT_DIR, "logs", "appo", args.task)
    load_path = None
    load_path_dir = None
    if args.load_run == "-1":
        if os.path.exists(base_log_dir):
            all_runs = sorted([d for d in os.listdir(base_log_dir) if os.path.isdir(os.path.join(base_log_dir, d))])
            if all_runs:
                latest_run_dir = os.path.join(base_log_dir, all_runs[-1])
                model_files = [f for f in os.listdir(latest_run_dir) if f.startswith("model_") and f.endswith(".pt")]
                if model_files:
                    model_files.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))
                    load_path = os.path.join(latest_run_dir, model_files[-1])
                    load_path_dir = latest_run_dir
    else:
        if os.path.exists(args.load_run):
            load_path = args.load_run
            load_path_dir = os.path.dirname(load_path)
        else:
            potential_dir = os.path.join(base_log_dir, args.load_run)
            if os.path.isdir(potential_dir):
                model_files = [f for f in os.listdir(potential_dir) if f.startswith("model_") and f.endswith(".pt")]
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
    env_indices = np.arange(args.play_env_num, dtype=np.int32)
    _, obs_out, _ = env.reset(env_indices)
    obs_np = np.asarray(obs_out, dtype=np.float32)

    state_list = []
    num_steps = 150

    print("Collecting physics states...")
    with torch.inference_mode():
        for _ in range(num_steps):
            obs_torch = torch.from_numpy(obs_np).to(device)
            td = TensorDict({"policy": obs_torch}, batch_size=args.play_env_num)
            actions_torch = actor(td)
            actions_np = actions_torch.cpu().numpy().astype(np.float32)
            state = env.step(actions_np)
            if hasattr(state, "obs"):
                next_obs_raw = state.obs
            else:
                next_obs_raw = state[0]
            obs_np = np.asarray(next_obs_raw, dtype=np.float32)
            state_list.append(np.asarray(env.state.physics_state, dtype=np.float32).copy())

    print("Rendering frames...")
    frames = render_many.render_states_get_frames(
        state_list,
        env.cfg.model_file,
        width=1280,
        height=720,
        camera_id=-1
    )

    print(f"Saving video to {output_video} with mediapy...")
    media.write_video(str(output_video), frames, fps=int(1.0/env.cfg.ctrl_dt))
    print("Done.")


def main():
    parser = argparse.ArgumentParser(description="Train APPO (native multiprocessing)")
    parser.add_argument("--task", type=str, default="Go2JoystickFlatTerrain")
    parser.add_argument("--max_iterations", type=int, default=1500)
    parser.add_argument("--save_interval", type=int, default=50)
    parser.add_argument("--total_envs", type=int, default=1024)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--collector_device", type=str, default=None)
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument("--steps_per_env", type=int, default=24)
    parser.add_argument("--play_only", action="store_true", help="Skip training, only play")
    parser.add_argument("--no_play", action="store_true", help="Skip play after training")
    parser.add_argument("--load_run", type=str, default="-1", help="Run ID to load or path")
    parser.add_argument("--play_env_num", type=int, default=16, help="Number of play envs")
    parser.add_argument("--logger", type=str, default="tensorboard", choices=["tensorboard", "wandb", "none", "no_print"])

    args = parser.parse_args()

    ensure_registries()

    from unilab.config.locomotion_params import appo_config
    rl_cfg = appo_config(args.task)

    if args.log_dir is None:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        args.log_dir = os.path.join(ROOT_DIR, "logs", "appo", args.task, f"{timestamp}_mujoco")

    if not args.play_only:
        from unilab.algos.torch.appo.runner import APPORunner

        collector_device = args.collector_device
        if collector_device == "gpu":
            import torch
            collector_device = "mps" if torch.backends.mps.is_available() else "cuda"

        runner = APPORunner(
            env_name=args.task,
            env_cfg_overrides={},
            rl_cfg=rl_cfg,
            device=args.device,
            collector_device=collector_device,
            num_envs=args.total_envs,
            steps_per_env=args.steps_per_env,
        )

        try:
            runner.learn(
                max_iterations=args.max_iterations,
                save_interval=args.save_interval,
                log_dir=args.log_dir,
                logger_type=args.logger,
            )
        finally:
            runner.close()

    if args.play_only or not args.no_play:
        play_appo(args, rl_cfg)


if __name__ == "__main__":
    main()
