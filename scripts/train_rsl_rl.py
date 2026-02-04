
import os
import sys
import argparse
import numpy as np
from pathlib import Path
import pkgutil
import importlib
import torch
from tensordict import TensorDict

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
from unilab.envs.utils import render_many

# Try importing rsl_rl
try:
    from rsl_rl.runners import OnPolicyRunner
except ImportError:
    print("Could not import rsl_rl. Please ensure it is installed.")
    sys.exit(1)

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
        
        # Convert output to torch (GPU)
        obs = torch.tensor(state.obs, device=self.device, dtype=torch.float32)
        rewards = torch.tensor(state.reward, device=self.device, dtype=torch.float32)
        dones = torch.tensor(state.done, device=self.device, dtype=torch.bool)
        
        # Update logging info
        self.episode_returns += rewards
        self.episode_lengths += 1
        
        infos = {}
        # Check for dones
        done_indices = torch.nonzero(dones).flatten()
        if len(done_indices) > 0:
            infos["episode"] = {
                "r": self.episode_returns[done_indices].clone(),
                "l": self.episode_lengths[done_indices].clone()
            }
            # Reset buffers for done envs
            self.episode_returns[done_indices] = 0
            self.episode_lengths[done_indices] = 0
            
            # Handle limits and timeouts (RSL-RL expects 'time_outs' in extras/infos)
            if hasattr(state, "truncated"):
                infos["time_outs"] = torch.tensor(state.truncated, device=self.device, dtype=torch.bool)
        
        obs_dict = TensorDict(
            {"policy": obs}, 
            batch_size=self.num_envs, 
            device=self.device
        )
        
        return obs_dict, rewards, dones, infos

    def reset(self):
        # Reset all environments
        if self.env.state is None:
            self.env.init_state()

        _, obs_np, _ = self.env.reset(np.arange(self.num_envs))
        obs = torch.tensor(obs_np, device=self.device, dtype=torch.float32)
        
        self.episode_returns[:] = 0
        self.episode_lengths[:] = 0
        
        return TensorDict(
            {"policy": obs}, 
            batch_size=self.num_envs, 
            device=self.device
        ), {}

    def get_observations(self):
        obs = torch.tensor(self.env.state.obs, device=self.device, dtype=torch.float32)
        return TensorDict(
            {"policy": obs}, 
            batch_size=self.num_envs, 
            device=self.device
        )

    def get_privileged_observations(self):
        obs = torch.tensor(self.env.state.obs, device=self.device, dtype=torch.float32)
        return obs


def get_latest_run(log_dir):
    """Find the latest run in the log directory."""
    if not os.path.exists(log_dir):
        return None
    runs = sorted([d for d in os.listdir(log_dir) if os.path.isdir(os.path.join(log_dir, d)) and d != "git"])
    if len(runs) > 0:
        return os.path.join(log_dir, runs[-1])
    return None


def main():
    parser = argparse.ArgumentParser(description="Train or Play RSL-RL agent")
    parser.add_argument("--task", type=str, required=True, help="Task name")
    parser.add_argument("--play_only", action="store_true", help="Play mode only")
    parser.add_argument("--load_run", type=str, default="-1", help="Run ID to load or path")
    parser.add_argument("--env_num", type=int, default=1024, help="Number of training envs")
    parser.add_argument("--play_env_num", type=int, default=16, help="Number of play envs")
    parser.add_argument("--num_timesteps", type=int, default=None, help="Overwritten total timesteps")
    
    args = parser.parse_args()
    
    # Load config
    cfg = locomotion_params.rsl_rl_config(args.task)
    
    # Override Max Iterations if timesteps provided
    if args.num_timesteps:
        n_steps_per_iter = cfg.num_steps_per_env * args.env_num
        max_iters = int(args.num_timesteps / n_steps_per_iter)
        cfg.max_iterations = max(1, max_iters)
        print(f"Overriding max_iterations to {max_iters} based on num_timesteps {args.num_timesteps}")

    log_root = ROOT_DIR / "logs" / "rsl_rl_train"
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
        
        # Runner
        runner = OnPolicyRunner(wrapped_env, train_cfg, log_dir=str(log_root), device=device)
        
        # Load capability for training (resume)
        resume_path = None
        if args.load_run != "-1":
            # If exact path
            if os.path.exists(args.load_run):
                resume_path = args.load_run
            else:
                # Check in log_root/task/run
                # The structure is usually log_root/task_name/run_name
                # If load_run is just a name or number?
                task_log_dir = log_root / args.task
                run_path = task_log_dir / args.load_run
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
        
        # Need to find the model to load
        task_log_dir = log_root / args.task
        load_path = None
        
        if args.load_run == "-1":
            load_path = get_latest_run(str(task_log_dir))
        else:
            if os.path.exists(args.load_run):
                load_path = args.load_run
            else:
                load_path = str(task_log_dir / args.load_run)
                 
        if not load_path or not os.path.exists(load_path):
            print(f"Could not find run to load at {load_path}")
            sys.exit(1)
             
        # Initialize runner just to load policy
        runner = OnPolicyRunner(wrapped_env, train_cfg, log_dir=str(log_root), device=device)
        runner.load(load_path)
        policy = runner.get_inference_policy(device=device)
        
        output_video = Path(load_path) / "play_video.mp4"
        
        print(f"Rendering video to {output_video}...")
        # Reset Environment
        obs, _ = wrapped_env.reset()
        
        state_list = []
        num_steps = 300
        
        # Collect states
        print("Collecting physics states...")
        with torch.inference_mode():
            for _ in range(num_steps):
                actions = policy(obs)
                obs, _, _, _ = wrapped_env.step(actions)
                # Copy physics state
                state_copy = env.state.physics_state.copy()
                state_list.append(state_copy)
        
        render_many.render_states_to_video(
            state_list, 
            env.cfg.model_file, 
            str(output_video), 
            fps=int(1.0/env.cfg.ctrl_dt),
            width=1280,
            height=720
        )

if __name__ == "__main__":
    main()
