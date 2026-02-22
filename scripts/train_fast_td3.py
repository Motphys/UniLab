"""Train FastTD3 agent with Ray-based async rollout workers."""

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


ensure_registries()

from unilab.algos.torch.fast_td3.runner import FastTD3Runner  # noqa: E402
from unilab.config import locomotion_params  # noqa: E402
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
    """Play mode: load a trained FastTD3 checkpoint and render video."""
    import mediapy as media
    from tensordict import TensorDict
    from rsl_rl.models import MLPModel
    from rsl_rl.utils import resolve_callable
    from unilab.utils.rsl_rl_compat import convert_config_v3_to_v4, is_rsl_rl_v4
    from unilab.utils import render_many
    from unilab.envs import registry

    if is_rsl_rl_v4():
        rl_cfg = convert_config_v3_to_v4(rl_cfg)

    device = args.device

    # --- Locate checkpoint ---
    base_log_dir = ROOT_DIR / "logs" / "fast_td3_train" / args.task
    load_path = None

    if not args.checkpoint:
        load_path = get_latest_run(str(base_log_dir))
    else:
        if os.path.exists(args.checkpoint):
            load_path = args.checkpoint
        else:
            load_path = str(base_log_dir / args.checkpoint)

    if not load_path or not os.path.exists(load_path):
        print(f"Could not find run to load at {load_path}")
        # sys.exit(1) # Don't exit, just warn? No, we need model.
        # But if user wants random policy?
        print("Playing with random policy (no checkpoint found).")
        load_path = None

    if load_path and os.path.isdir(load_path):
        model_files = [f for f in os.listdir(load_path) if f.startswith("model_") and f.endswith(".pt")]
        if model_files:
            import re
            # Extract number from model_X.pt
            def extract_iter(x):
                m = re.search(r"model_(\d+).pt", x)
                return int(m.group(1)) if m else -1
            
            model_files.sort(key=extract_iter)
            load_path_dir = load_path
            load_path = os.path.join(load_path, model_files[-1])
            print(f"Loading latest model: {load_path}")
        else:
            print(f"No model files found in {load_path}")
            load_path = None
            load_path_dir = args.checkpoint if args.checkpoint else str(base_log_dir)
    elif load_path:
        load_path_dir = os.path.dirname(load_path)
    else:
        load_path_dir = str(base_log_dir)

    # --- Create environment ---
    # Play locally
    env = registry.make(args.task, num_envs=args.play_env_num, sim_backend="mujoco")
    obs_dim = env.observation_space.shape[0]
    num_actions = env.action_space.shape[0]

    # --- Build actor model and load weights ---
    obs_example = torch.zeros((args.play_env_num, obs_dim), device=device)
    td_example = TensorDict({"policy": obs_example}, batch_size=args.play_env_num)

    actor_cfg = rl_cfg["actor"].copy()
    actor_cls = resolve_callable(actor_cfg.pop("class_name"))
    # Make sure we match signature
    actor = actor_cls(td_example, rl_cfg["obs_groups"], "actor", num_actions, **actor_cfg)
    actor = actor.to(device)
    actor.eval()

    if load_path:
        checkpoint = torch.load(load_path, map_location=device)
        # FastTD3 saves 'actor' key in state_dict (not actor_state_dict, based on learner.py)
        # Check keys
        if "actor" in checkpoint:
             actor.load_state_dict(checkpoint["actor"])
        elif "actor_state_dict" in checkpoint:
             actor.load_state_dict(checkpoint["actor_state_dict"])
        else:
             print("Warning: Could not find 'actor' or 'actor_state_dict' in checkpoint.")

        print(f"Loaded checkpoint from {load_path}")

    # --- Rollout ---
    output_video = Path(load_path_dir) / "play_video.mp4"
    if not os.path.exists(load_path_dir):
        os.makedirs(load_path_dir, exist_ok=True)
        
    print(f"Rendering video to {output_video}...")

    # Reset
    # unilab envs reset takes 'env_indices' or 'seed' usually?
    # registry.make returns MjNpEnv or similar. reset(seed=None, options=None)?
    # MjNpEnv.reset(self, seed=None, options=None) -> obs, info
    # BUT wait, the previous error said: Go2WalkTaskMj.reset() missing argument: 'env_indices'
    # This implies Go2WalkTaskMj requiresenv_indices.
    # Let's import numpy
    import numpy as np
    
    try:
        import mlx.core as mx
        env_indices = mx.arange(args.play_env_num, dtype=mx.int32)
    except ImportError:
        env_indices = np.arange(args.play_env_num)
    try:
        _, obs_out, _ = env.reset(env_indices)
    except TypeError:
        obs_out, _ = env.reset()

    obs = mlx_to_torch(obs_out, device)

    state_list = []
    num_steps = 200 # ~4s

    print("Collecting physics states...")
    with torch.inference_mode():
        for _ in range(num_steps):
            obs_td = TensorDict({"policy": obs}, batch_size=args.play_env_num, device=device)
            # deterministic play: stochastic=False?
            # MLPModel(obs, stochastic_output=False)
            actions = actor(obs_td) 
            # Apply tanh if FastTD3 (it does tanh in learner update, but actor output itself might be raw logits? 
            # No, MLPModel usually ends with activation if configured? 
            # In learner.py we manually tanh. So here we must also manually tanh!
            actions = torch.tanh(actions)
            
            actions_np = actions.detach().cpu().numpy()

            # Step
            # unilab env step returns: State(obs, reward, terminated, truncated, info, physics_state)
            # or (obs, reward, terminated, truncated, info)?
            # The previous error "Go2WalkTaskMj.reset() missing..." suggests it's a specific env class.
            # RSL-RL env wrapper expects step -> (obs, rew, done, info).
            # But registry.make returns the raw env?
            # In train_appo.py: state = env.step(actions_np); obs=state.obs...
            # This implies it returns an object with attributes.
            state = env.step(actions_np)

            if hasattr(state, "obs"):
                obs = mlx_to_torch(state.obs, device)
            else:
                obs = mlx_to_torch(state[0], device)

            if hasattr(env, "state") and hasattr(env.state, "physics_state"):
                state_list.append(to_numpy(env.state.physics_state).copy())
            elif hasattr(state, "physics_state"):
                state_list.append(to_numpy(state.physics_state).copy())

    print("Rendering frames...")
    # Fix num_processes=1 for safety
    frames = render_many.render_states_get_frames(
        state_list,
        env.cfg.model_file,
        width=1280,
        height=720,
        camera_id=-1,
        num_processes=1
    )

    print(f"Saving video to {output_video} with mediapy...")
    media.write_video(str(output_video), frames, fps=int(1.0 / env.cfg.ctrl_dt))
    print("Done!")
    env.close()


def main():
    parser = argparse.ArgumentParser(description="Train FastTD3 agent (Ray-based async)")
    parser.add_argument("--task", type=str, default="Go2JoystickFlatTerrain", help="Task name")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of Ray rollout workers")
    
    default_device = "cpu"
    if torch.cuda.is_available():
        default_device = "cuda:0"
    elif torch.backends.mps.is_available():
        default_device = "mps"

    parser.add_argument("--device", type=str, default=default_device, help="Learner device")
    parser.add_argument("--steps_per_env", type=int, default=24, help="Steps per env per iteration")
    parser.add_argument("--total_envs", type=int, default=4096, help="Total environments across workers")
    parser.add_argument("--play_env_num", type=int, default=16, help="Number of play envs")
    parser.add_argument("--max_iterations", type=int, default=1500, help="Training iterations")
    parser.add_argument("--save_interval", type=int, default=50, help="Checkpoint interval")
    parser.add_argument("--replay_buffer_n", type=int, default=1000, help="Replay buffer depth per env (total = N * total_envs)")
    parser.add_argument("--batch_size", type=int, default=4096, help="Mini-batch size for learning")
    parser.add_argument("--warmup_steps", type=int, default=5000, help="Random transitions before learning")
    parser.add_argument("--updates_per_step", type=int, default=1, help="Gradient steps per env step")
    parser.add_argument("--exploration_noise", type=float, default=0.5, help="Gaussian exploration noise std")
    parser.add_argument("--play_only", action="store_true", help="Play with trained policy (no training)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint for play mode")

    args = parser.parse_args()

    print(f"Loading config for {args.task}...")
    rl_cfg = locomotion_params.fast_td3_config(args.task)
    rl_cfg = rl_cfg.to_dict()

    if args.play_only:
        play(args, rl_cfg)
        return

    num_envs_per_worker = max(1, args.total_envs // args.num_workers)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = str(ROOT_DIR / "logs" / "fast_td3_train" / args.task / timestamp)

    print(
        f"Starting FastTD3 with {args.num_workers} workers, "
        f"total_envs={args.total_envs} ({num_envs_per_worker}/worker) "
        f"on device={args.device}"
    )
    print(f"Log dir: {log_dir}")

    runner = FastTD3Runner(
        env_name=args.task,
        env_cfg_overrides={},
        rl_cfg=rl_cfg,
        device=args.device,
        num_workers=args.num_workers,
        steps_per_env=args.steps_per_env,
        num_envs_per_worker=num_envs_per_worker,
        replay_buffer_n=args.replay_buffer_n,
        batch_size=args.batch_size,
        warmup_steps=args.warmup_steps,
        updates_per_step=args.updates_per_step,
        exploration_noise=args.exploration_noise,
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


if __name__ == "__main__":
    main()
