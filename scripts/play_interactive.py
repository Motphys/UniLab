"""Interactive play script: opens a live MuJoCo viewer for a trained RSL-RL policy.

Usage:
    # Load the latest checkpoint for a task
    python scripts/play_interactive.py --task Go2JoystickFlatTerrain

    # Load a specific run
    python scripts/play_interactive.py --task Go2JoystickFlatTerrain --load_run 2024-02-04_12-00-00

Camera controls (MuJoCo viewer):
    Mouse drag     - rotate
    Scroll         - zoom
    Right-drag     - pan
"""

import os
import sys
import time
import argparse
import pkgutil
import importlib
from pathlib import Path

import numpy as np
import torch
import mujoco
import mujoco.viewer

ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR))


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
from unilab.config import locomotion_params
from unilab.utils.rsl_rl_compat import is_rsl_rl_v4, convert_config_v3_to_v4
from unilab.utils.run_utils import get_latest_run
from unilab.utils.torch_utils import to_torch

try:
    from rsl_rl.runners import OnPolicyRunner
except ImportError:
    print("Could not import rsl_rl. Please ensure it is installed.")
    sys.exit(1)

from tensordict import TensorDict


# ---------------------------------------------------------------------------
# Minimal env wrapper (mirrors RslRlVecEnvWrapper in train_rsl_rl.py)
# ---------------------------------------------------------------------------

class RslRlVecEnvWrapper:
    def __init__(self, env, device="cuda"):
        self.env = env
        self.cfg = env.cfg
        self.device = device
        self.num_envs = env.num_envs
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.num_obs = env.observation_space.shape[0]
        self.num_privileged_obs = self.num_obs
        self.num_actions = env.action_space.shape[0]
        self.episode_returns = torch.zeros(self.num_envs, device=device)
        self.episode_lengths = torch.zeros(self.num_envs, device=device)
        self.episode_length_buf = self.episode_lengths
        self.max_episode_length = int(env.cfg.max_episode_seconds / env.cfg.ctrl_dt)
        self.reset()

    def step(self, actions):
        actions_np = actions.detach().cpu().numpy() if isinstance(actions, torch.Tensor) else actions
        state = self.env.step(actions_np)
        obs = to_torch(state.obs, self.device)
        rewards = to_torch(state.reward, self.device)
        dones = to_torch(state.done, self.device).bool()
        self.episode_returns += rewards
        self.episode_lengths += 1
        infos = {}
        done_idx = torch.nonzero(dones).flatten()
        if len(done_idx) > 0:
            if hasattr(state, "truncated"):
                infos["time_outs"] = to_torch(state.truncated, self.device).bool()
            self.episode_returns[done_idx] = 0
            self.episode_lengths[done_idx] = 0
        if hasattr(state, "info") and "log" in state.info:
            infos["log"] = state.info["log"]
        obs_dict = TensorDict({"policy": obs}, batch_size=self.num_envs, device=self.device)
        return obs_dict, rewards, dones, infos

    def reset(self):
        if self.env.state is None:
            self.env.init_state()
        env_indices = np.arange(self.num_envs, dtype=np.int32)
        _, obs_out, _ = self.env.reset(env_indices)
        obs = to_torch(obs_out, self.device)
        self.episode_returns[:] = 0
        self.episode_lengths[:] = 0
        return TensorDict({"policy": obs}, batch_size=self.num_envs, device=self.device), {}

    def get_observations(self):
        obs = to_torch(self.env.state.obs, self.device)
        return TensorDict({"policy": obs}, batch_size=self.num_envs, device=self.device)

    def get_privileged_observations(self):
        return to_torch(self.env.state.obs, self.device)


# ---------------------------------------------------------------------------
# Checkpoint resolution helpers
# ---------------------------------------------------------------------------

def resolve_checkpoint(task: str, load_run: str) -> str | None:
    base = ROOT_DIR / "logs" / "rsl_rl_train" / task
    if load_run == "-1":
        path = get_latest_run(str(base))
    elif os.path.exists(load_run):
        path = load_run
    else:
        path = str(base / load_run)

    if not path or not os.path.exists(path):
        print(f"[play_interactive] Run not found: {path}")
        return None

    if os.path.isdir(path):
        model_files = sorted(
            [f for f in os.listdir(path) if f.startswith("model_") and f.endswith(".pt")],
            key=lambda f: int(f.split("_")[1].split(".")[0]),
        )
        if not model_files:
            print(f"[play_interactive] No model_*.pt files in {path}")
            return None
        path = os.path.join(path, model_files[-1])

    print(f"[play_interactive] Loading checkpoint: {path}")
    return path


# ---------------------------------------------------------------------------
# Interactive play
# ---------------------------------------------------------------------------

def play_interactive(args):
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"[play_interactive] Device: {device}")

    # Always use a single env for interactive view
    env = registry.make(args.task, num_envs=1, sim_backend="mujoco")
    wrapped_env = RslRlVecEnvWrapper(env, device=device)

    cfg = locomotion_params.rsl_rl_config(args.task)
    train_cfg = cfg.to_dict()
    if is_rsl_rl_v4():
        train_cfg = convert_config_v3_to_v4(train_cfg)

    policy = None
    if args.action_mode == "policy":
        ckpt = resolve_checkpoint(args.task, args.load_run)
        if ckpt is None:
            print("[play_interactive] WARNING: no checkpoint found — falling back to zero actions.")
        else:
            log_dir = str(ROOT_DIR / "logs" / "rsl_rl_train" / args.task / "play_temp")
            runner = OnPolicyRunner(wrapped_env, train_cfg, log_dir=log_dir, device=device)
            runner.load(ckpt)
            policy = runner.get_inference_policy(device=device)

    print(f"[play_interactive] Action mode: {args.action_mode}")

    # Dedicated MjData for the viewer (never touches the rollout workers)
    mj_model = env._model
    viz_data = mujoco.MjData(mj_model)
    state_spec = mujoco.mjtState.mjSTATE_FULLPHYSICS
    ctrl_dt = env.cfg.ctrl_dt

    obs, _ = wrapped_env.reset()

    print("[play_interactive] Opening viewer — close the window or press Esc to quit.")

    # Get action bounds for random mode
    action_low = env.action_space.low
    action_high = env.action_space.high

    with mujoco.viewer.launch_passive(mj_model, viz_data) as viewer:
        with torch.inference_mode():
            while viewer.is_running():
                t0 = time.perf_counter()

                if args.action_mode == "policy" and policy is not None:
                    actions = policy(obs)
                elif args.action_mode == "random":
                    actions = torch.from_numpy(
                        np.random.uniform(action_low, action_high, size=(1, env.action_space.shape[0]))
                    ).to(device).float()
                else:  # zero
                    actions = torch.zeros(1, env.action_space.shape[0], device=device)

                obs, _, _, _ = wrapped_env.step(actions)

                # Push env state[0] into viz_data and refresh scene
                phys = env.state.physics_state[0].astype(np.float64)
                mujoco.mj_setState(mj_model, viz_data, phys, state_spec)
                mujoco.mj_forward(mj_model, viz_data)
                viewer.sync()

                # Real-time pacing
                elapsed = time.perf_counter() - t0
                if ctrl_dt - elapsed > 0:
                    time.sleep(ctrl_dt - elapsed)

    print("[play_interactive] Done.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Interactive MuJoCo viewer for a trained RSL-RL policy")
    parser.add_argument("--task", type=str, required=True,
                        help="Task name, e.g. Go2JoystickFlatTerrain")
    parser.add_argument("--load_run", type=str, default="-1",
                        help="Run timestamp or path to load (-1 = latest)")
    parser.add_argument("--action_mode", type=str, default="policy", choices=["policy", "zero", "random"],
                        help="Action mode: policy (load ckpt), zero, or random")
    args = parser.parse_args()
    play_interactive(args)


if __name__ == "__main__":
    main()
