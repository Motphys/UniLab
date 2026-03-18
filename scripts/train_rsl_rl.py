import datetime
import importlib
import os
import pkgutil
import sys
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict

ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR))


from unilab.utils.obs_utils import flatten_obs_dict
from unilab.utils.torch_utils import to_numpy, to_torch

try:
    from rsl_rl.runners import OnPolicyRunner
except ImportError:
    print("Could not import rsl_rl. Please ensure it is installed.")
    sys.exit(1)

from unilab.utils.rsl_rl_compat import (
    convert_config_v3_to_v4,
    convert_config_v5,
    is_rsl_rl_v4,
    is_rsl_rl_v5,
)
from unilab.utils.run_utils import get_latest_run


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


_GO1_LEGACY_COMMAND_VEL_LIMIT = [[0.5, 0.0, 0.0], [0.5, 0.0, 0.0]]
_GO1_LEGACY_REWARD_CONFIG = {
    "scales": {
        "tracking_lin_vel": 1.0,
        "tracking_ang_vel": 0.2,
        "lin_vel_z": -5.0,
        "ang_vel_xy": -0.1,
        "base_height": -100.0,
        "action_rate": -0.005,
        "similar_to_default": -0.1,
    },
    "tracking_sigma": 0.25,
    "base_height_target": 0.3,
}

_G1_MOTRIX_LEGACY_PPO_REWARD_CONFIG = {
    "scales": {
        "tracking_lin_vel": 3.0,
        "tracking_ang_vel": 0.25,
        "forward_progress": 1.5,
        "under_speed": -1.0,
        "upper_body_pose": -0.05,
        "feet_phase": 0.5,
        "lin_vel_z": -1.0,
        "ang_vel_xy": -0.2,
        "base_height": -120.0,
        "orientation": -2.5,
        "action_rate": -0.005,
        "pose": -0.05,
    },
    "tracking_sigma": 0.25,
    "gait_frequency": 1.5,
    "feet_phase_swing_height": 0.09,
    "feet_phase_tracking_sigma": 0.008,
    "base_height_target": 0.754,
    "min_base_height": 0.50,
    "max_tilt_deg": 35.0,
}


def get_go1_motrix_legacy_ppo_profile(task_name: str, sim_backend: str) -> dict | None:
    if task_name != "Go1JoystickFlatTerrain" or sim_backend != "motrix":
        return None
    return {
        "algo_overrides": {
            "max_iterations": 151,
            "empirical_normalization": True,
            "policy": {"init_noise_std": 0.5},
            "algorithm": {"learning_rate": 3.0e-4, "entropy_coef": 1.0e-3},
        },
        "env_cfg_override": {
            "legacy_motrix_profile": {
                "enabled": True,
                "command_vel_limit": _GO1_LEGACY_COMMAND_VEL_LIMIT,
            },
            "reward_config": _GO1_LEGACY_REWARD_CONFIG,
        },
    }


def get_g1_motrix_legacy_ppo_profile(task_name: str, sim_backend: str) -> dict | None:
    if task_name != "G1JoystickFlatTerrain" or sim_backend != "motrix":
        return None
    return {
        "algo_overrides": {
            "max_iterations": 151,
            "empirical_normalization": True,
            "policy": {"init_noise_std": 0.5},
            "algorithm": {"learning_rate": 3.0e-4, "entropy_coef": 5.0e-3},
        },
        "env_cfg_override": {
            "reward_config": _G1_MOTRIX_LEGACY_PPO_REWARD_CONFIG,
            "backend_overrides": {
                "enabled": True,
                "control_action_scale": 0.5,
                "command_vel_limit": [[0.45, 0.0, 0.0], [0.55, 0.0, 0.0]],
                "gait_phase_init_mode": "independent",
                "reset_base_qvel_limit": 0.05,
                "reward_scale_overrides": {
                    "under_speed": -1.0,
                    "upper_body_pose": -0.05,
                    "tracking_ang_vel": 0.25,
                },
            },
        },
    }


def build_go1_motrix_legacy_ppo_env_cfg_override(cfg: DictConfig) -> dict:
    env_cfg_override = {}
    if hasattr(cfg, "reward") and cfg.reward:
        env_cfg_override["reward_config"] = OmegaConf.to_container(cfg.reward, resolve=True)

    profile = get_go1_motrix_legacy_ppo_profile(cfg.training.task_name, cfg.training.sim_backend)
    if profile is not None:
        cfg.algo.max_iterations = profile["algo_overrides"]["max_iterations"]
        cfg.algo.empirical_normalization = profile["algo_overrides"]["empirical_normalization"]
        cfg.algo.policy.init_noise_std = profile["algo_overrides"]["policy"]["init_noise_std"]
        cfg.algo.algorithm.learning_rate = profile["algo_overrides"]["algorithm"][
            "learning_rate"
        ]
        cfg.algo.algorithm.entropy_coef = profile["algo_overrides"]["algorithm"]["entropy_coef"]
        env_cfg_override.update(profile["env_cfg_override"])

    return env_cfg_override


def build_g1_motrix_legacy_ppo_env_cfg_override(cfg: DictConfig) -> dict:
    env_cfg_override = {}
    profile = get_g1_motrix_legacy_ppo_profile(cfg.training.task_name, cfg.training.sim_backend)
    if profile is not None:
        cfg.algo.max_iterations = profile["algo_overrides"]["max_iterations"]
        cfg.algo.empirical_normalization = profile["algo_overrides"]["empirical_normalization"]
        cfg.algo.policy.init_noise_std = profile["algo_overrides"]["policy"]["init_noise_std"]
        cfg.algo.algorithm.learning_rate = profile["algo_overrides"]["algorithm"][
            "learning_rate"
        ]
        cfg.algo.algorithm.entropy_coef = profile["algo_overrides"]["algorithm"]["entropy_coef"]
        cfg.algo.obs_groups["actor"] = ["policy"]
        env_cfg_override.update(profile["env_cfg_override"])
    return env_cfg_override


class RslRlVecEnvWrapper:
    """Wrapper to adapt NpEnv to RSL-RL OnPolicyRunner interface."""

    def __init__(self, env, device="cuda"):
        self.env = env
        self.cfg = env.cfg
        self.device = device
        self.num_envs = env.num_envs
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.num_obs = sum(env.obs_groups_spec.values())
        self.num_privileged_obs = self.num_obs
        self.num_actions = env.action_space.shape[0]

        self.episode_returns = torch.zeros(self.num_envs, device=self.device)
        self.episode_lengths = torch.zeros(self.num_envs, device=self.device)

        self.episode_length_buf = self.episode_lengths
        self.max_episode_length = np.ceil(env.cfg.max_episode_seconds / env.cfg.ctrl_dt)

        self.reset()

    def _obs_to_tensordict(self, obs: dict[str, np.ndarray]) -> TensorDict:
        actor = to_torch(obs["obs"], self.device)
        td = {"actor": actor}
        if "privileged" in obs:
            td["privileged"] = to_torch(obs["privileged"], self.device)
            if hasattr(self.env, "get_policy_obs"):
                policy_obs = self.env.get_policy_obs(obs)
            else:
                policy_obs = flatten_obs_dict(obs)
            td["policy"] = to_torch(policy_obs, self.device)
        else:
            td["policy"] = actor
        return TensorDict(td, batch_size=self.num_envs, device=self.device)

    def step(self, actions):
        if isinstance(actions, torch.Tensor):
            actions_np = actions.detach().cpu().numpy()
        else:
            actions_np = actions

        state = self.env.step(actions_np)

        # Convert output to torch tensors on target device
        rewards = to_torch(state.reward, self.device)
        dones = to_torch(state.done, self.device).bool()

        self.episode_returns += rewards
        self.episode_lengths += 1

        infos = {}
        done_indices = torch.nonzero(dones).flatten()
        if len(done_indices) > 0:
            if hasattr(state, "truncated"):
                infos["time_outs"] = to_torch(state.truncated, self.device).bool()
            self.episode_returns[done_indices] = 0
            self.episode_lengths[done_indices] = 0

        if hasattr(state, "info") and "log" in state.info:
            infos["log"] = state.info["log"]

        obs_dict = self._obs_to_tensordict(state.obs)
        return obs_dict, rewards, dones, infos

    def reset(self):
        if self.env.state is None:
            self.env.init_state()
        env_indices = np.arange(self.num_envs, dtype=np.int32)
        obs_out, _ = self.env.reset(env_indices)

        self.episode_returns[:] = 0
        self.episode_lengths[:] = 0

        return self._obs_to_tensordict(obs_out), {}

    def get_observations(self):
        return self._obs_to_tensordict(self.env.state.obs)

    def get_privileged_observations(self):
        obs = to_torch(flatten_obs_dict(self.env.state.obs), self.device)
        return obs


def play_rsl_rl(cfg: DictConfig, device: str):
    """Play mode for RSL-RL."""

    from unilab.base import registry

    env_cfg_override = build_go1_motrix_legacy_ppo_env_cfg_override(cfg)
    env_cfg_override.update(build_g1_motrix_legacy_ppo_env_cfg_override(cfg))

    env = registry.make(
        cfg.training.task_name,
        num_envs=cfg.training.play_env_num,
        sim_backend=cfg.training.sim_backend,
        env_cfg_override=env_cfg_override,
    )
    wrapped_env = RslRlVecEnvWrapper(env, device=device)
    train_cfg = OmegaConf.to_container(cfg.algo, resolve=True)
    if "runner" not in train_cfg:
        train_cfg["runner"] = {}
    train_cfg["runner"]["logger"] = "none"
    if is_rsl_rl_v5():
        train_cfg = convert_config_v5(train_cfg)

    base_log_dir = ROOT_DIR / "logs" / "rsl_rl_train" / cfg.training.task_name
    load_path = None

    load_run = cfg.training.load_run
    if load_run == "-1":
        load_path = get_latest_run(str(base_log_dir))
    else:
        if os.path.exists(load_run):
            load_path = load_run
        else:
            load_path = str(base_log_dir / load_run)

    if not load_path or not os.path.exists(load_path):
        print(f"Could not find run to load at {load_path}")
        return

    if os.path.isdir(load_path):
        model_files = [
            f for f in os.listdir(load_path) if f.startswith("model_") and f.endswith(".pt")
        ]
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

    _ckpt_keys = set(torch.load(load_path, map_location="cpu", weights_only=True).keys())
    if "actor_state_dict" not in _ckpt_keys:
        print(
            f"Checkpoint at {load_path} is not an rsl-rl checkpoint "
            f"(found keys: {_ckpt_keys}). Aborting play."
        )
        return

    runner = OnPolicyRunner(wrapped_env, train_cfg, log_dir=None, device=device)
    runner.load(load_path)
    policy = runner.get_inference_policy(device=device)

    if cfg.training.sim_backend == "motrix":
        print("Starting interactive visualization (motrix native renderer)...")
        print("Close the render window to exit.")
        env._backend.init_renderer()
        obs, _ = wrapped_env.reset()

        import time

        last_render_time = time.perf_counter()
        render_dt = 1.0 / 60.0

        with torch.inference_mode():
            try:
                while True:
                    actions = policy(obs)
                    obs, _, _, _ = wrapped_env.step(actions)

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
        import mediapy as media

        from unilab.utils import render_many

        output_video = Path(load_path_dir) / "play_video.mp4"
        print(f"Rendering video to {output_video}...")

        obs, _ = wrapped_env.reset()
        state_list = []

        print("Collecting physics states...")
        with torch.inference_mode():
            for _ in range(cfg.training.play_steps):
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
            cam_distance=cfg.training.cam_distance,
            cam_elevation=cfg.training.cam_elevation,
            cam_azimuth=cfg.training.cam_azimuth,
        )

        print(f"Saving video to {output_video} with mediapy...")
        media.write_video(str(output_video), frames, fps=int(1.0 / env.cfg.ctrl_dt))
        print("Done.")


@hydra.main(version_base="1.3", config_path="../conf/ppo", config_name="config")
def main(cfg: DictConfig) -> None:
    ensure_registries()

    from omegaconf import OmegaConf

    from unilab.base import registry

    env_cfg_override = build_go1_motrix_legacy_ppo_env_cfg_override(cfg)
    env_cfg_override.update(build_g1_motrix_legacy_ppo_env_cfg_override(cfg))

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    # Compute effective max_iterations (supports num_timesteps override)
    max_iterations = cfg.algo.max_iterations
    if cfg.training.num_timesteps:
        n_steps_per_iter = cfg.algo.num_steps_per_env * cfg.algo.num_envs
        max_iterations = max(1, int(cfg.training.num_timesteps / n_steps_per_iter))
        print(
            f"Overriding max_iterations to {max_iterations} based on "
            f"num_timesteps {cfg.training.num_timesteps}"
        )

    if not cfg.training.play_only:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_dir = str(
            ROOT_DIR
            / "logs"
            / "rsl_rl_train"
            / cfg.training.task_name
            / f"{timestamp}_{cfg.training.sim_backend}"
        )
    else:
        log_dir = None

    if not cfg.training.play_only:
        env = registry.make(
            cfg.training.task_name,
            num_envs=cfg.algo.num_envs,
            sim_backend=cfg.training.sim_backend,
            env_cfg_override=env_cfg_override,
        )
        wrapped_env = RslRlVecEnvWrapper(env, device=device)

        train_cfg = OmegaConf.to_container(cfg.algo, resolve=True)
        if "runner" not in train_cfg:
            train_cfg["runner"] = {}
        if cfg.training.logger in ["tensorboard", "wandb"]:
            train_cfg["runner"]["logger"] = cfg.training.logger
        else:
            train_cfg["runner"]["logger"] = "none"

        if is_rsl_rl_v5():
            train_cfg = convert_config_v5(train_cfg)

        runner = OnPolicyRunner(wrapped_env, train_cfg, log_dir=log_dir, device=device)

        if cfg.training.load_run != "-1":
            if os.path.exists(cfg.training.load_run):
                resume_path = cfg.training.load_run
            else:
                base_log_dir = ROOT_DIR / "logs" / "rsl_rl_train" / cfg.training.task_name
                run_path = base_log_dir / cfg.training.load_run
                resume_path = str(run_path) if run_path.exists() else None

            if resume_path:
                print(f"Resuming from {resume_path}")
                runner.load(resume_path)

        runner.learn(num_learning_iterations=max_iterations, init_at_random_ep_len=True)
        env.close()

    if cfg.training.play_only or not cfg.training.no_play:
        play_rsl_rl(cfg, device)


if __name__ == "__main__":
    main()
