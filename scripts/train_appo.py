import argparse
import sys
import os
import datetime
from pathlib import Path
import torch
import numpy as np
import pkgutil
import importlib

# Add project root to sys.path
ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR))


# Ensure all environment modules are imported so they are registered
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


ensure_registries()

from unilab.algos.appo.runner import APPORunner
from unilab.config import locomotion_params
from unilab.envs import registry
from unilab.utils import render_many
from unilab.utils.mlx_torch_utils import mlx_to_torch, to_numpy


def get_latest_run(log_dir):
    """Find the latest run in the log directory that contains a model."""
    if not os.path.exists(log_dir):
        return None
    runs = sorted([d for d in os.listdir(log_dir) if os.path.isdir(os.path.join(log_dir, d)) and d != "git"])
    for run_id in reversed(runs):
        run_path = os.path.join(log_dir, run_id)
        if any(f.endswith(".pt") for f in os.listdir(run_path)):
            return run_path
    if runs:
        return os.path.join(log_dir, runs[-1])
    return None


def play(args, rl_cfg):
    """Play mode: load a trained APPO checkpoint and render video."""
    import mediapy as media
    from tensordict import TensorDict
    from rsl_rl.models import MLPModel
    from rsl_rl.utils import resolve_callable
    from unilab.utils.rsl_rl_compat import convert_config_v3_to_v4, is_rsl_rl_v4

    if "policy" in rl_cfg:
        rl_cfg["policy"]["noise_std_type"] = "log"
    elif "actor" in rl_cfg:
        rl_cfg["actor"]["noise_std_type"] = "log"

    if is_rsl_rl_v4():
        rl_cfg = convert_config_v3_to_v4(rl_cfg)

    device = args.device

    # --- Locate checkpoint ---
    base_log_dir = ROOT_DIR / "logs" / "appo_train" / args.task
    load_path = None

    if args.load_run == "-1":
        load_path = get_latest_run(str(base_log_dir))
    else:
        if os.path.exists(args.load_run):
            load_path = args.load_run
        else:
            load_path = str(base_log_dir / args.load_run)

    if not load_path or not os.path.exists(load_path):
        print(f"Could not find run to load at {load_path}")
        sys.exit(1)

    if os.path.isdir(load_path):
        model_files = [f for f in os.listdir(load_path) if f.startswith("model_") and f.endswith(".pt")]
        if model_files:
            model_files.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))
            load_path_dir = load_path
            load_path = os.path.join(load_path, model_files[-1])
            print(f"Loading latest model: {load_path}")
        else:
            print(f"No model files found in {load_path}")
            sys.exit(1)
    else:
        load_path_dir = os.path.dirname(load_path)

    # --- Create environment ---
    env = registry.make(args.task, num_envs=args.play_env_num, sim_backend="mujoco")
    obs_dim = env.observation_space.shape[0]
    num_actions = env.action_space.shape[0]

    # --- Build actor model and load weights ---
    obs_example = torch.zeros((args.play_env_num, obs_dim), device=device)
    td_example = TensorDict({"policy": obs_example}, batch_size=args.play_env_num)

    actor_cfg = rl_cfg["actor"].copy()
    actor_cls = resolve_callable(actor_cfg.pop("class_name"))
    actor = actor_cls(td_example, rl_cfg["obs_groups"], "actor", num_actions, **actor_cfg)
    actor = actor.to(device)
    actor.eval()

    checkpoint = torch.load(load_path, map_location=device, weights_only=True)
    actor.load_state_dict(checkpoint["actor_state_dict"])
    print(f"Loaded checkpoint iteration {checkpoint.get('iteration', '?')}")

    # --- Rollout ---
    output_video = Path(load_path_dir) / "play_video.mp4"
    print(f"Rendering video to {output_video}...")

    # Reset (MLX backend may return MLX arrays; convert to torch)
    env.init_state()
    try:
        import mlx.core as mx
        env_indices = mx.arange(args.play_env_num, dtype=mx.int32)
    except ImportError:
        env_indices = np.arange(args.play_env_num)
    _, obs_out, _ = env.reset(env_indices)
    obs = mlx_to_torch(obs_out, device)

    state_list = []
    num_steps = 150

    print("Collecting physics states...")
    with torch.inference_mode():
        for _ in range(num_steps):
            obs_td = TensorDict({"policy": obs}, batch_size=args.play_env_num, device=device)
            actions = actor(obs_td)
            actions_np = actions.detach().cpu().numpy()

            state = env.step(actions_np)
            obs = mlx_to_torch(state.obs, device)
            state_list.append(to_numpy(state.physics_state).copy())

    print("Rendering frames...")
    frames = render_many.render_states_get_frames(
        state_list,
        env.cfg.model_file,
        width=1280,
        height=720,
        camera_id=-1,
    )

    print(f"Saving video to {output_video} with mediapy...")
    media.write_video(str(output_video), frames, fps=int(1.0 / env.cfg.ctrl_dt))
    print("Done!")
    env.close()


def train(args, rl_cfg):
    """Training mode."""
    env_cfg_overrides = {}
    if "policy" in rl_cfg:
        rl_cfg["policy"]["noise_std_type"] = "log"
    elif "actor" in rl_cfg:
        rl_cfg["actor"]["noise_std_type"] = "log"
    else:
        print("WARNING: Could not find 'policy' or 'actor' in config to set noise_std_type.")

    num_envs_per_worker = max(1, args.total_envs // args.num_workers)

    # Setup logging directory
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = str(ROOT_DIR / "logs" / "appo_train" / args.task / timestamp)

    print(
        f"Starting APPO Runner with {args.num_workers} workers, "
        f"total_envs={args.total_envs} ({num_envs_per_worker}/worker) "
        f"on learner_device={args.device}..."
    )
    print(f"Log dir: {log_dir}")

    runner = APPORunner(
        env_name=args.task,
        env_cfg_overrides=env_cfg_overrides,
        rl_cfg=rl_cfg,
        device=args.device,
        num_workers=args.num_workers,
        steps_per_env=args.steps_per_env,
        num_envs_per_worker=num_envs_per_worker,
    )

    try:
        runner.learn(
            max_iterations=args.max_iterations,
            save_interval=args.save_interval,
            log_dir=log_dir,
        )
    except KeyboardInterrupt:
        print("Interrupted by user.")
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Closing runner...")
        runner.close()


def main():
    parser = argparse.ArgumentParser(description="Train or Play APPO agent (Ray-based)")
    parser.add_argument("--task", type=str, default="Go2JoystickFlatTerrain", help="Task name")
    parser.add_argument("--play_only", action="store_true", help="Play mode only")
    parser.add_argument("--load_run", type=str, default="-1", help="Run ID to load or path")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of Ray rollout workers")

    # Auto-detect device
    default_device = "cpu"
    if torch.cuda.is_available():
        default_device = "cuda:0"
    elif torch.backends.mps.is_available():
        default_device = "mps"

    parser.add_argument("--device", type=str, default=default_device, help="Device (e.g. cuda:0, mps, cpu)")
    parser.add_argument("--steps_per_env", type=int, default=24, help="Steps per environment per iteration")
    parser.add_argument("--total_envs", type=int, default=1024, help="Total number of environments")
    parser.add_argument("--play_env_num", type=int, default=16, help="Number of play envs")
    parser.add_argument("--max_iterations", type=int, default=1500, help="Total iterations")
    parser.add_argument("--save_interval", type=int, default=50, help="Save checkpoint every N iterations")

    args = parser.parse_args()

    # Get config
    print(f"Loading config for {args.task}...")
    rl_cfg = locomotion_params.rsl_rl_config(args.task)
    rl_cfg = rl_cfg.to_dict()

    if args.play_only:
        play(args, rl_cfg)
    else:
        train(args, rl_cfg)


if __name__ == "__main__":
    main()
