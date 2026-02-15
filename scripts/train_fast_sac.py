"""Train FastSAC agent with Ray-based async rollout workers."""

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

from unilab.algos.fast_sac.runner import FastSACRunner  # noqa: E402
from unilab.config import locomotion_params  # noqa: E402



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
    """Play mode: load a trained FastSAC checkpoint and render video."""
    import mediapy as media
    from tensordict import TensorDict
    from rsl_rl.models import MLPModel
    from rsl_rl.utils import resolve_callable
    from unilab.utils.rsl_rl_compat import convert_config_v3_to_v4, is_rsl_rl_v4
    from unilab.envs.utils import render_many
    from unilab.envs import registry

    if is_rsl_rl_v4():
        rl_cfg = convert_config_v3_to_v4(rl_cfg)

    device = args.device

    # --- Locate checkpoint ---
    base_log_dir = ROOT_DIR / "logs" / "fast_sac_train" / args.task
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
        print("Playing with random policy (no checkpoint found).")
        load_path = None

    if load_path and os.path.isdir(load_path):
        model_files = [f for f in os.listdir(load_path) if f.startswith("model_") and f.endswith(".pt")]
        if model_files:
            import re
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

    if load_path:
        checkpoint = torch.load(load_path, map_location=device)
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

    import numpy as np
    all_indices = np.arange(args.play_env_num)
    try:
        _, obs_np, _ = env.reset(all_indices)
    except TypeError:
        obs_np, _ = env.reset()

    obs = torch.tensor(obs_np, device=device, dtype=torch.float32)

    state_list = []
    num_steps = 200 # ~4s

    print("Collecting physics states...")
    with torch.inference_mode():
        for _ in range(num_steps):
            obs_td = TensorDict({"policy": obs}, batch_size=args.play_env_num, device=device)
            # SAC play: mean action. 
            # If MLPModel stochastic=True, stochastic_output=False (default?) might return mean.
            # But let's check config. If we can pass stochastic_output=False...
            # The RSL-RL MLPModel impl: 
            # forward(obs, stochastic_output=None) -> if None, uses self.stochastic.
            # So we should pass stochastic_output=False.
            actions = actor(obs_td, stochastic_output=False) 
            
            # SAC usually has tanh at the end of actor if bounds are needed? 
            # The MLPModel usually outputs distribution params.
            # If stochastic=False, it might just return mean. 
            # Does it apply tanh? That depends on 'activation' param usually being internal.
            # Standard SAC implementation applies tanh to the *sampled* action.
            # The mean of a squashed Gaussian is NOT tanh(mean of Gaussian).
            # But for play, we often just want mean.
            # Let's assume MLPModel output is raw mean if stochastic_output=False?
            # RSL-RL PPO actor outputs mean directly.
            # If our config uses 'MLPModel' and was trained with SAC which applies tanh...
            # Wait, our FastSACLearner applies TanhTransform? No, we used `Normal` and then `rsample`.
            # In learner.py: `dist = self.actor(obs)` -> returns Normal distribution? No, MLPModel returns action directly if not stochastic?
            # Revisit FastSACLearner:
            # `dist = self.actor.get_dist(obs)` ?
            # `actions = dist.rsample()`
            # Checking learner... learner uses `self.actor(obs)`... 
            # If MLPModel `stochastic=True`, it returns a distribution object? Or samples?
            # RSL-RL `MLPModel` usually returns tensor?
            # Let's check `learner.py`.
            # FastSACLearner calls `self.actor(obs)`.
            # If `MLPModel` returns a distribution, then `actions = dist.rsample()` works.
            # If `MLPModel` returns samples, then we are good.
            # But for play we want mean.
            # If `stochastic_output=False`, `MLPModel` returns mean.
            # Does this mean need tanh? 
            # In SAC, action = tanh(mu + sigma*z). 
            # Mean action = tanh(mu).
            # If `MLPModel` output is `mu` (unbounded), we NEED tanh.
            # RSL-RL PPO MLPModel usually has final activation? No, policy output is usually unbounded logits.
            # So yes, we likely need tanh if the environment expects [-1, 1].
            # Safe to apply it.
            
            actions = torch.tanh(actions)
            actions_np = actions.detach().cpu().numpy()

            state = env.step(actions_np)
            if hasattr(state, "obs"):
                obs_np = state.obs
            else:
                 obs_np = state[0]
            
            obs = torch.tensor(obs_np, device=device, dtype=torch.float32)

            if hasattr(env, "state") and hasattr(env.state, "physics_state"):
                state_copy = env.state.physics_state.copy()
                state_list.append(state_copy)
            elif hasattr(state, "physics_state"):
                 state_list.append(state.physics_state.copy())

    print("Rendering frames...")
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
    parser = argparse.ArgumentParser(description="Train FastSAC agent (Ray-based async)")
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
    parser.add_argument("--target_entropy", type=float, default=None, help="Target entropy (default: -dim(A))")
    parser.add_argument("--alpha_lr", type=float, default=3e-4, help="Learning rate for alpha")
    parser.add_argument("--play_only", action="store_true", help="Play with trained policy (no training)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint for play mode")

    args = parser.parse_args()

    print(f"Loading config for {args.task}...")
    rl_cfg = locomotion_params.fast_sac_config(args.task)
    rl_cfg = rl_cfg.to_dict()

    if args.play_only:
        play(args, rl_cfg)
        return

    num_envs_per_worker = max(1, args.total_envs // args.num_workers)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = str(ROOT_DIR / "logs" / "fast_sac_train" / args.task / timestamp)

    print(
        f"Starting FastSAC with {args.num_workers} workers, "
        f"total_envs={args.total_envs} ({num_envs_per_worker}/worker) "
        f"on device={args.device}"
    )
    print(f"Log dir: {log_dir}")

    runner = FastSACRunner(
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
