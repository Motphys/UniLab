
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
    for pkg_name in ("unilab.envs.locomotion", "unilab.envs.manipulation"):
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

ensure_registries()

from unilab.base import registry
from unilab.config import locomotion_params, manipulation_params
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
    """Wrapper to adapt NpEnv to RSL-RL OnPolicyRunner interface."""
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
        # Reset all environments
        if self.env.state is None:
            self.env.init_state()
        env_indices = np.arange(self.num_envs, dtype=np.int32)
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

def RslRlAacVecEnvWrapper(RslRlVecEnvWrapper): #Asymmetric Actor-Critic
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


def play_rsl_rl(args, cfg, device):
    """Play mode for RSL-RL."""
    import torch
    from unilab.base import registry
    from unilab.utils.torch_utils import to_numpy

    env = registry.make(args.task, num_envs=args.play_env_num, sim_backend=args.sim_backend)
    wrapped_env = RslRlVecEnvWrapper(env, device=device)
    train_cfg = cfg.to_dict()
    if is_rsl_rl_v4():
        train_cfg = convert_config_v3_to_v4(train_cfg)

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
        return

    if os.path.isdir(load_path):
        model_files = [f for f in os.listdir(load_path) if f.startswith("model_") and f.endswith(".pt")]
        if len(model_files) > 0:
            model_files.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))
            load_path_dir = load_path
            load_path = os.path.join(load_path, model_files[-1])
            print(f"Loading latest model: {load_path}")
        else:
            print(f"No model files found in {load_path}")
            return
    else:
        load_path_dir = os.path.dirname(load_path)

    log_dir = str(ROOT_DIR / "logs" / "rsl_rl_train" / args.task / "play_temp")
    runner = OnPolicyRunner(wrapped_env, train_cfg, log_dir=log_dir, device=device)
    runner.load(load_path)
    policy = runner.get_inference_policy(device=device)

    # Use native rendering for motrix backend
    if args.sim_backend == "motrix":
        print("Starting interactive visualization (motrix native renderer)...")
        print("Close the render window to exit.")
        env._backend.init_renderer()
        obs, _ = wrapped_env.reset()

        import time
        last_render_time = time.perf_counter()
        render_dt = 1.0 / 60.0  # 60 FPS

        with torch.inference_mode():
            try:
                while True:
                    actions = policy(obs)
                    obs, _, _, _ = wrapped_env.step(actions)

                    # Time sync for rendering
                    current_time = time.perf_counter()
                    elapsed = current_time - last_render_time
                    if elapsed < render_dt:
                        time.sleep(render_dt - elapsed)
                    last_render_time = time.perf_counter()

                    env._backend.render()
            except Exception as e:
                if "RenderClosedError" in str(type(e).__name__):
                    print("Render window closed.")
                else:
                    raise
    else:
        # MuJoCo backend: render to video
        import mediapy as media
        from unilab.utils import render_many

        output_video = Path(load_path_dir) / "play_video.mp4"
        print(f"Rendering video to {output_video}...")

        obs, _ = wrapped_env.reset()
        state_list = []

        print("Collecting physics states...")
        with torch.inference_mode():
            for _ in range(args.play_steps):
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
            cam_distance=args.cam_distance,
            cam_elevation=args.cam_elevation,
            cam_azimuth=args.cam_azimuth,
        )

        print(f"Saving video to {output_video} with mediapy...")
        media.write_video(str(output_video), frames, fps=int(1.0/env.cfg.ctrl_dt))
        print("Done.")


def main():
    parser = argparse.ArgumentParser(description="Train or Play RSL-RL agent")
    parser.add_argument("--task", type=str, required=True, help="Task name")
    parser.add_argument("--play_only", action="store_true", help="Skip training, only play")
    parser.add_argument("--no_play", action="store_true", help="Skip play after training")
    parser.add_argument("--load_run", type=str, default="-1", help="Run ID to load or path")
    parser.add_argument("--env_num", type=int, default=4096, help="Number of training envs (task default if unset)")
    parser.add_argument("--play_env_num", type=int, default=16, help="Number of play envs")
    parser.add_argument("--play_steps", type=int, default=200, help="Number of steps for play video")
    parser.add_argument("--num_timesteps", type=int, default=None, help="Overwritten total timesteps")
    parser.add_argument("--logger", type=str, default="tensorboard", choices=["tensorboard", "wandb", "none", "no_print"])
    parser.add_argument("--sim_backend", type=str, default="mujoco", choices=["mujoco", "motrix"], help="Simulation backend")
    # Video rendering
    parser.add_argument("--cam_distance", type=float, default=6.0, help="Camera distance for play video")
    parser.add_argument("--cam_elevation", type=float, default=-20.0, help="Camera elevation angle (degrees) for play video")
    parser.add_argument("--cam_azimuth", type=float, default=90.0, help="Camera azimuth angle (degrees) for play video")

    args = parser.parse_args()

    # Determine which params module to use based on task registration
    params = manipulation_params if args.task in manipulation_params.DEFAULT_ENV_NUM_BY_TASK else locomotion_params

    # Load config
    cfg = params.ppo_config(args.task) if hasattr(params, 'ppo_config') else params.rsl_rl_config(args.task)

    if args.env_num is None:
        args.env_num = cfg.num_envs
    
    # Override Max Iterations if timesteps provided
    if args.num_timesteps:
        n_steps_per_iter = cfg.num_steps_per_env * args.env_num
        max_iters = int(args.num_timesteps / n_steps_per_iter)
        cfg.max_iterations = max(1, max_iters)
        print(f"Overriding max_iterations to {max_iters} based on num_timesteps {args.num_timesteps}")

    if not args.play_only:
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        log_dir = str(ROOT_DIR / "logs" / "rsl_rl_train" / args.task / f"{timestamp}_{args.sim_backend}")
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
        env = registry.make(args.task, num_envs=args.env_num, sim_backend=args.sim_backend)
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

    if args.play_only or not args.no_play:
        play_rsl_rl(args, cfg, device)


if __name__ == "__main__":
    main()
