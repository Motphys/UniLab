"""Integration tests for RSL-RL PPO training.

Requires MuJoCo and rsl_rl to be installed. Run with:
    uv run pytest -m veryslow -v -k rsl_rl
"""

from __future__ import annotations

import sys
import tempfile

import pytest

pytest.importorskip("mujoco")
rsl_rl = pytest.importorskip("rsl_rl")

import numpy as np
import torch
from tensordict import TensorDict

from unilab.base import registry
from unilab.config.structured_configs import PPOConfig
from unilab.utils.algo_utils import ensure_registries
from unilab.utils.obs_utils import flatten_obs_dict
from unilab.utils.rsl_rl_compat import convert_config_v5, is_rsl_rl_v5
from unilab.utils.torch_utils import to_torch

ensure_registries()


# ---------------------------------------------------------------------------
# Minimal wrapper (same as scripts/train_rsl_rl.py)
# ---------------------------------------------------------------------------


class _RslRlVecEnvWrapper:
    """Lightweight RSL-RL wrapper for testing."""

    def __init__(self, env, device="cpu"):
        self.env = env
        self.cfg = env.cfg
        self.device = device
        self.num_envs = env.num_envs
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.num_obs = sum(env.obs_groups_spec.values())
        self.num_privileged_obs = self.num_obs
        self.num_actions = env.action_space.shape[0]

        self.episode_returns = torch.zeros(self.num_envs, device=device)
        self.episode_lengths = torch.zeros(self.num_envs, device=device)
        self.episode_length_buf = self.episode_lengths
        self.max_episode_length = np.ceil(env.cfg.max_episode_seconds / env.cfg.ctrl_dt)
        self.reset()

    def _obs_to_tensordict(self, obs: dict[str, np.ndarray]) -> TensorDict:
        actor = to_torch(obs["obs"], self.device)
        td = {"actor": actor}
        if "privileged" in obs:
            td["privileged"] = to_torch(obs["privileged"], self.device)
            td["policy"] = to_torch(flatten_obs_dict(obs), self.device)
        else:
            td["policy"] = actor
        return TensorDict(td, batch_size=self.num_envs, device=self.device)

    def step(self, actions):
        actions_np = (
            actions.detach().cpu().numpy() if isinstance(actions, torch.Tensor) else actions
        )
        state = self.env.step(actions_np)
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
        return self._obs_to_tensordict(state.obs), rewards, dones, infos

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
        return to_torch(flatten_obs_dict(self.env.state.obs), self.device)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.veryslow
@pytest.mark.parametrize(
    "env_name",
    [
        "Go2JoystickFlatTerrain",
        "G1JoystickFlatTerrain",
    ],
)
def test_rsl_rl_ppo_one_iteration(
    env_name: str, default_go2_reward_config, default_g1_reward_config
):
    """RSL-RL PPO can complete 1 training iteration on a real env."""
    from rsl_rl.runners import OnPolicyRunner

    reward_cfg = default_go2_reward_config if "Go2" in env_name else default_g1_reward_config
    env = registry.make(
        env_name, num_envs=256, sim_backend="mujoco", env_cfg_override={"reward_config": reward_cfg}
    )
    wrapped = _RslRlVecEnvWrapper(env, device="cpu")

    cfg = PPOConfig()
    train_cfg = cfg.to_dict()
    train_cfg["runner"] = {"logger": "none"}
    # Small network + short loop; large num_envs to saturate CPU
    train_cfg["num_steps_per_env"] = 8
    train_cfg["policy"] = {
        "class_name": "ActorCritic",
        "actor_hidden_dims": [64, 64],
        "critic_hidden_dims": [64, 64],
        "activation": "elu",
        "init_noise_std": 1.0,
    }
    train_cfg["algorithm"]["num_learning_epochs"] = 1
    train_cfg["algorithm"]["num_mini_batches"] = 2
    if is_rsl_rl_v5():
        train_cfg = convert_config_v5(train_cfg)

    with tempfile.TemporaryDirectory() as tmpdir:
        runner = OnPolicyRunner(wrapped, train_cfg, log_dir=tmpdir, device="cpu")
        runner.learn(num_learning_iterations=1, init_at_random_ep_len=True)

    env.close()
