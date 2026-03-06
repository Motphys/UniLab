
import os
import sys
import argparse
import numpy as np
import datetime
from pathlib import Path
import pkgutil
import importlib
import torch
from tensordict import TensorDict
import mediapy as media

# Add workspace root to python path dynamically
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
                except Exception as e:
                    # Ignore errors during discovery
                    pass
    except ImportError:
        pass

ensure_registries()

from unilab.envs import registry
from unilab.config import locomotion_params
from unilab.utils import render_many
from unilab.utils.torch_utils import to_torch, to_numpy

# Try importing rsl_rl
try:
    from rsl_rl.runners import OnPolicyRunner
except ImportError:
    print("Could not import rsl_rl. Please ensure it is installed.")
    sys.exit(1)

from unilab.utils.rsl_rl_compat import is_rsl_rl_v4, convert_config_v3_to_v4
from unilab.utils.run_utils import get_latest_run

class RslRlVecEnvWrapper:
    """Wrapper to adapt MjNpEnv to RSL-RL OnPolicyRunner interface."""
    def __init__(self, env, device='cuda'):
        self.env = env
        # Expose cfg to RSL-RL runner if needed (some versions check env.cfg)
        self.cfg = env.cfg
        self.device = device
        self.num_envs = env.num_envs
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.num_obs = env.observation_space.shape[0]
        self.num_privileged_obs = self.num_obs
        self.num_actions = env.action_space.shape[0]
        
        self.episode_returns = torch.zeros(self.num_envs, device=self.device)
        self.episode_lengths = torch.zeros(self.num_envs, device=self.device)
        
        # Compatibility attribute names for rsl-rl
        self.episode_length_buf = self.episode_lengths 
        self.max_episode_length = np.ceil(env.cfg.max_episode_seconds / env.cfg.ctrl_dt)

        # RSL-RL runner calls get_observations() in __init__, so we need to ensure env is reset
        self.reset()
        
    def step(self, actions):
        # Convert actions to numpy (CPU)
        if isinstance(actions, torch.Tensor):
            actions_np = actions.detach().cpu().numpy()
        else:
            actions_np = actions
            
        # Step the environment
        state = self.env.step(actions_np)
        
        # Convert output to torch tensors on target device
        obs = to_torch(state.obs, self.device)
        rewards = to_torch(state.reward, self.device)
        dones = to_torch(state.done, self.device).bool()
        
        # Update logging info
        self.episode_returns += rewards
        self.episode_lengths += 1
        
        infos = {}
        # Check for dones
        done_indices = torch.nonzero(dones).flatten()
        if len(done_indices) > 0:
            # Handle limits and timeouts (RSL-RL expects 'time_outs' in extras/infos)
            if hasattr(state, "truncated"):
                infos["time_outs"] = to_torch(state.truncated, self.device).bool()
            
            # Reset buffers for done envs
            self.episode_returns[done_indices] = 0
            self.episode_lengths[done_indices] = 0

        # Pass per-step logs if available (gs_playground style)
        # prioritizing 'log' over 'episode' allows per-step metric logging
        if hasattr(state, "info") and "log" in state.info:
            infos["log"] = state.info["log"]
        
        obs_dict = TensorDict(
            {"policy": obs}, 
            batch_size=self.num_envs, 
            device=self.device
        )
        
        return obs_dict, rewards, dones, infos

    def reset(self):
        # Reset all environments (MLX backend expects mx.array for env_indices)
        if self.env.state is None:
            self.env.init_state()
        try:
            import mlx.core as mx
            env_indices = mx.arange(self.num_envs, dtype=mx.int32)
        except ImportError:
            env_indices = np.arange(self.num_envs)
        _, obs_out, _ = self.env.reset(env_indices)
        obs = to_torch(obs_out, self.device)

        self.episode_returns[:] = 0
        self.episode_lengths[:] = 0

        return TensorDict(
            {"policy": obs},
            batch_size=self.num_envs,
            device=self.device
        ), {}

    def get_observations(self):
        obs = to_torch(self.env.state.obs, self.device)
        return TensorDict(
            {"policy": obs},
            batch_size=self.num_envs,
            device=self.device
        )

    def get_privileged_observations(self):
        obs = to_torch(self.env.state.obs, self.device)
        return obs




def main():
    parser = argparse.ArgumentParser(description="Train or Play RSL-RL agent")
    parser.add_argument("--task", type=str, required=True, help="Task name")
    parser.add_argument("--play_only", action="store_true", help="Play mode only")
    parser.add_argument("--load_run", type=str, default="-1", help="Run ID to load or path")
    parser.add_argument("--env_num", type=int, default=None, help="Number of training envs (task default if unset)")
    parser.add_argument("--play_env_num", type=int, default=16, help="Number of play envs")
    parser.add_argument("--num_timesteps", type=int, default=None, help="Overwritten total timesteps")
    parser.add_argument("--logger", type=str, default="tensorboard", choices=["tensorboard", "wandb", "none", "no_print"])
    
    args = parser.parse_args()
    if args.env_num is None:
        args.env_num = locomotion_params.get_default_env_num(args.task)
    
    # Load config
    cfg = locomotion_params.rsl_rl_config(args.task)
    
    # Override Max Iterations if timesteps provided
    if args.num_timesteps:
        n_steps_per_iter = cfg.num_steps_per_env * args.env_num
        max_iters = int(args.num_timesteps / n_steps_per_iter)
        cfg.max_iterations = max(1, max_iters)
        print(f"Overriding max_iterations to {max_iters} based on num_timesteps {args.num_timesteps}")

    if not args.play_only:
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        log_dir = str(ROOT_DIR / "logs" / "rsl_rl_train" / args.task / timestamp)
    else:
        log_dir = None

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    # TRAIN MODE
    if not args.play_only:
        # Create environment
        env = registry.make(args.task, num_envs=args.env_num, sim_backend="mujoco")
        wrapped_env = RslRlVecEnvWrapper(env, device=device)
        
        # Convert ConfigDict to regular dict for RSL-RL
        train_cfg = cfg.to_dict()
        
        if "runner" not in train_cfg:
            train_cfg["runner"] = {}
        if args.logger in ["tensorboard", "wandb"]:
            train_cfg["runner"]["logger"] = args.logger
        else:
            train_cfg["runner"]["logger"] = "none"

        if is_rsl_rl_v4():
            train_cfg = convert_config_v3_to_v4(train_cfg)
        
        # Runner
        runner = OnPolicyRunner(wrapped_env, train_cfg, log_dir=log_dir, device=device)
        
        # Load capability for training (resume)
        resume_path = None
        if args.load_run != "-1":
            # If exact path
            if os.path.exists(args.load_run):
                resume_path = args.load_run
            else:
                # Check in logs/rsl_rl_train/task_name
                # The structure is log_root/task_name/timestamp
                # If load_run is just a timestamp?
                base_log_dir = ROOT_DIR / "logs" / "rsl_rl_train" / args.task
                run_path = base_log_dir / args.load_run
                if run_path.exists():
                    resume_path = str(run_path)
        
        if resume_path:
            print(f"Resuming from {resume_path}")
            runner.load(resume_path)
             
        runner.learn(num_learning_iterations=cfg.max_iterations, init_at_random_ep_len=True)
        
    # PLAY MODE
    else:
        # Create environment (play num)
        env = registry.make(args.task, num_envs=args.play_env_num, sim_backend="mujoco")
        
        # We need a dummy wrapper just to be compatible with runner for loading policy
        wrapped_env = RslRlVecEnvWrapper(env, device=device)
        train_cfg = cfg.to_dict()
        if is_rsl_rl_v4():
            train_cfg = convert_config_v3_to_v4(train_cfg)
        
        # Need to find the model to load
        base_log_dir = ROOT_DIR / "logs" / "rsl_rl_train" / args.task
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
            
        # If load_path is a directory, find the latest model file
        if os.path.isdir(load_path):
            model_files = [f for f in os.listdir(load_path) if f.startswith("model_") and f.endswith(".pt")]
            if len(model_files) > 0:
                # Sort by iteration number
                model_files.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))
                load_path_dir = load_path # keep dir for video output
                load_path = os.path.join(load_path, model_files[-1])
                print(f"Loading latest model: {load_path}")
            else:
                print(f"No model files found in {load_path}")
                sys.exit(1)
        else:
            load_path_dir = os.path.dirname(load_path)
             
        # Initialize runner just to load policy
        # NOTE: For play, we don't care about log_dir so much, but runner needs it
        runner = OnPolicyRunner(wrapped_env, train_cfg, log_dir=log_dir, device=device)
        runner.load(load_path)
        policy = runner.get_inference_policy(device=device)
        
        output_video = Path(load_path_dir) / "play_video.mp4"
        
        print(f"Rendering video to {output_video}...")
        # Reset Environment
        obs, _ = wrapped_env.reset()
        
        state_list = []
        num_steps = 150
        
        # Collect states (physics_state may be MLX -> numpy for render_many)
        print("Collecting physics states...")
        with torch.inference_mode():
            for _ in range(num_steps):
                actions = policy(obs)
                obs, _, _, _ = wrapped_env.step(actions)
                state_list.append(to_numpy(env.state.physics_state).copy())
        
        print("Rendering frames...")
        # Use render_many to get frames, then use mediapy to save
        frames = render_many.render_states_get_frames(
            state_list, 
            env.cfg.model_file, 
            width=1280,
            height=720,
            camera_id=-1 # or specific camera
        )

        print(f"Saving video to {output_video} with mediapy...")
        media.write_video(str(output_video), frames, fps=int(1.0/env.cfg.ctrl_dt))

if __name__ == "__main__":
    main()
