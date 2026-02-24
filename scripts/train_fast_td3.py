"""Train FastTD3 agent — native multiprocessing, no Ray."""

import argparse
import sys
import os
import datetime
from pathlib import Path
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


def main():
    parser = argparse.ArgumentParser(description="Train FastTD3 (no Ray)")
    parser.add_argument("--task", type=str, default="Go2JoystickFlatTerrain")
    parser.add_argument("--max_iterations", type=int, default=None)
    parser.add_argument("--num_envs", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--collector_device", type=str, default=None)
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument("--play_only", action="store_true", help="Play mode only")
    parser.add_argument("--load_run", type=str, default="-1", help="Run ID to load or path")
    parser.add_argument("--play_env_num", type=int, default=16, help="Number of play envs")
    args = parser.parse_args()

    ensure_registries()

    from unilab.config.locomotion_params import fast_td3_config
    cfg = fast_td3_config(args.task)

    if args.max_iterations is not None:
        cfg.max_iterations = args.max_iterations
    if args.num_envs is not None:
        cfg.num_envs = args.num_envs

    if args.log_dir is None:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        args.log_dir = os.path.join(ROOT_DIR, "logs", "fast_td3", args.task, timestamp)

    if not args.play_only:
        from unilab.algos.torch.fast_td3.runner import FastTD3Runner

        runner = FastTD3Runner(
            env_name=args.task,
            device=args.device,
            collector_device=args.collector_device,
            num_envs=cfg.num_envs,
            steps_per_env=cfg.num_steps_per_env,
            replay_buffer_n=cfg.replay_buffer_n,
            batch_size=cfg.batch_size,
            warmup_steps=cfg.warmup_steps,
            updates_per_step=cfg.updates_per_step,
            policy_delay=cfg.policy_delay,
            gamma=cfg.gamma,
            tau=cfg.tau,
            actor_lr=cfg.actor_lr,
            critic_lr=cfg.critic_lr,
            actor_hidden_dim=cfg.actor_hidden_dim,
            critic_hidden_dim=cfg.critic_hidden_dim,
            num_atoms=cfg.num_atoms,
            exploration_noise=cfg.exploration_noise,
            use_layer_norm=cfg.use_layer_norm,
        )

        try:
            runner.learn(
                max_iterations=cfg.max_iterations,
                save_interval=cfg.save_interval,
                log_dir=args.log_dir,
            )
        finally:
            runner.close()
            
    else:
        # Play mode
        import torch
        import numpy as np
        import mediapy as media
        from unilab.envs import registry
        from unilab.utils.run_utils import get_latest_run
        from unilab.utils import render_many
        from unilab.algos.torch.common.worker import _build_actor

        device = args.device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Using device for play: {device}")

        # Create env
        env = registry.make(args.task, num_envs=args.play_env_num, sim_backend="mujoco")
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]

        # Build actor
        actor = _build_actor(
            algo_type="td3",
            obs_dim=obs_dim,
            action_dim=action_dim,
            actor_hidden_dim=cfg.actor_hidden_dim,
            use_layer_norm=cfg.use_layer_norm,
            device=device,
        )
        actor.eval()

        # Load weights
        base_log_dir = os.path.join(ROOT_DIR, "logs", "fast_td3", args.task)
        if args.task.endswith("Play") and (not os.path.exists(base_log_dir) or not os.listdir(base_log_dir)):
            base_log_dir = os.path.join(ROOT_DIR, "logs", "fast_td3", args.task[:-4])

        load_path = None
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
            sys.exit(1)

        print(f"Loading model: {load_path}")
        checkpoint = torch.load(load_path, map_location=device, weights_only=True)
        # FastTD3 Learner get_state_dict() returns {"actor": ..., "qnet": ...}
        actor.load_state_dict(checkpoint["actor"])

        output_video = os.path.join(load_path_dir, "play_video.mp4")
        print(f"Rendering video to {output_video}...")

        # Reset Env
        if env.state is None:
            env.init_state()
        try:
            import mlx.core as mx
            env_indices = mx.arange(args.play_env_num, dtype=mx.int32)
        except ImportError:
            env_indices = np.arange(args.play_env_num)
            
        _, obs_out, _ = env.reset(env_indices)
        from unilab.algos.torch.common.worker import _mx_to_np
        obs_np = _mx_to_np(obs_out)

        state_list = []
        num_steps = 150

        from unilab.utils.mlx_torch_utils import to_numpy
        
        print("Collecting physics states...")
        with torch.inference_mode():
            for _ in range(num_steps):
                obs_torch = torch.from_numpy(obs_np).to(device)
                
                # TD3 inference (deterministic)
                actions_torch = actor(obs_torch)
                actions_np = actions_torch.cpu().numpy()
                
                state = env.step(actions_np)
                
                if hasattr(state, "obs"):
                    next_obs_raw = state.obs
                else:
                    next_obs_raw = state[0]
                    
                obs_np = _mx_to_np(next_obs_raw)
                    
                state_list.append(to_numpy(env.state.physics_state).copy())

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


if __name__ == "__main__":
    main()
