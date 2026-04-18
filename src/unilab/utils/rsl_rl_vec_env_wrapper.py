"""Shared RSL-RL vectorized environment wrapper.

This module provides a unified RslRlVecEnvWrapper that aligns with the current
env contract (obs, info) reset format and is used by both training and play scripts.
"""

import numpy as np
import torch
from tensordict import TensorDict

from unilab.utils.obs_utils import flatten_policy_obs_dict
from unilab.utils.torch_utils import to_torch


class RslRlVecEnvWrapper:
    """Wrapper to adapt NpEnv to RSL-RL OnPolicyRunner interface.

    This wrapper aligns with the current env contract:
    - reset() returns (obs_dict, info_dict)
    - step() returns state with .obs, .reward, .done, .truncated, .info attributes

    Args:
        env: The environment to wrap (must follow NpEnv contract).
        device: Device to place tensors on ("cuda", "mps", or "cpu").
        policy_obs_mode: Observation mode for policy ("flat" or "actor").
            "flat" uses flattened obs dict, "actor" uses only the "obs" key.
            Default is "flat" for backward compatibility with training scripts.
    """

    def __init__(self, env, device: str = "cuda", policy_obs_mode: str = "flat"):
        self.env = env
        self.cfg = env.cfg
        self.device = device
        self.policy_obs_mode = policy_obs_mode
        self.num_envs = env.num_envs
        self.observation_space = env.observation_space
        self.action_space = env.action_space

        # Compute observation dimensions
        self._actor_obs_dim = int(env.obs_groups_spec.get("obs", sum(env.obs_groups_spec.values())))
        self._critic_obs_dim = int(env.obs_groups_spec.get("critic", self._actor_obs_dim))
        self._flat_obs_dim = self._actor_obs_dim
        self.num_obs = self._flat_obs_dim if policy_obs_mode == "flat" else self._actor_obs_dim
        # Legacy RSL-RL field name; semantically this is the critic-path observation dim.
        self.num_privileged_obs = self._critic_obs_dim
        self.num_actions = env.action_space.shape[0]

        # Episode tracking
        self.episode_returns = torch.zeros(self.num_envs, device=device)
        self.episode_lengths = torch.zeros(self.num_envs, device=device)
        self.episode_length_buf = self.episode_lengths
        self.max_episode_length = int(env.cfg.max_episode_seconds / env.cfg.ctrl_dt)

        # Initialize
        self.reset()

    def _obs_to_tensordict(self, obs: dict[str, np.ndarray]) -> TensorDict:
        """Convert observation dict to TensorDict for RSL-RL.

        Args:
            obs: Observation dictionary with mandatory "obs" and optional "critic".

        Returns:
            TensorDict with "policy" and "actor" keys, plus optional "critic".
        """
        actor = to_torch(obs["obs"], self.device)

        if self.policy_obs_mode == "actor":
            policy = actor
        else:
            policy = to_torch(flatten_policy_obs_dict(obs), self.device)

        td: dict[str, torch.Tensor] = {"policy": policy, "actor": actor}

        if "critic" in obs:
            td["critic"] = to_torch(obs["critic"], self.device)

        return TensorDict(td, batch_size=self.num_envs, device=self.device)

    def step(self, actions):
        """Execute one step in the environment.

        Args:
            actions: Actions to execute (torch.Tensor or numpy array).

        Returns:
            Tuple of (obs_tensordict, rewards, dones, infos).
        """
        # Convert actions to numpy
        if isinstance(actions, torch.Tensor):
            actions_np = actions.detach().cpu().numpy()
        else:
            actions_np = actions

        # Step the environment
        state = self.env.step(actions_np)

        # Convert outputs to torch tensors
        rewards = to_torch(state.reward, self.device)
        dones = to_torch(state.done, self.device).bool()

        # Update episode statistics
        self.episode_returns += rewards
        self.episode_lengths += 1

        # Build info dict
        infos = {}
        done_indices = torch.nonzero(dones).flatten()
        if len(done_indices) > 0:
            if hasattr(state, "truncated"):
                infos["time_outs"] = to_torch(state.truncated, self.device).bool()
                final_observation = getattr(state, "final_observation", None)
                if (
                    final_observation is None
                    and hasattr(state, "info")
                    and isinstance(state.info, dict)
                ):
                    final_observation = state.info.get("final_observation")
                if torch.any(infos["time_outs"]) and isinstance(final_observation, dict):
                    infos["time_out_bootstrap_obs"] = self._obs_to_tensordict(final_observation)
            self.episode_returns[done_indices] = 0
            self.episode_lengths[done_indices] = 0

        if hasattr(state, "info") and "log" in state.info:
            infos["log"] = state.info["log"]

        obs_dict = self._obs_to_tensordict(state.obs)
        return obs_dict, rewards, dones, infos

    def reset(self):
        """Reset the environment.

        Returns:
            Tuple of (obs_tensordict, empty_info_dict).
        """
        # Ensure state is initialized
        if self.env.state is None:
            self.env.init_state()

        # Reset all environments
        env_indices = np.arange(self.num_envs, dtype=np.int32)
        obs_out, _ = self.env.reset(env_indices)

        # Reset episode statistics
        self.episode_returns[:] = 0
        self.episode_lengths[:] = 0

        return self._obs_to_tensordict(obs_out), {}

    def get_observations(self):
        """Get current observations without stepping.

        Returns:
            TensorDict with current observations.
        """
        return self._obs_to_tensordict(self.env.state.obs)

    def get_privileged_observations(self):
        """Get current critic observations via the legacy RSL-RL hook name.

        Returns:
            Torch tensor with critic-path observations (or actor obs if unavailable).
        """
        obs = self.env.state.obs
        critic_base = obs.get("critic", obs["obs"])
        return to_torch(critic_base, self.device)
