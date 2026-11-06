"""Adapter from the UniLab TorchEnv contract to RSL-RL's VecEnv."""

from __future__ import annotations

import math
from typing import Any

import torch
from tensordict import TensorDict


def _to_policy_tensor(x: Any, device: str | torch.device) -> torch.Tensor:
    """Move a TorchEnv observation tensor onto the policy device."""
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"TorchEnv observation must be a torch.Tensor, got {type(x).__name__}")
    tensor = x.to(device)
    # UniLab policies use float32. Some backends publish float64 observations;
    # preserve integer/bool tensors.
    if tensor.is_floating_point() and tensor.dtype != torch.float32:
        tensor = tensor.float()
    return tensor


def get_policy_obs_dims(obs_groups_spec: dict[str, int]) -> tuple[int, int]:
    """Return ``(actor_obs_dim, flat_policy_obs_dim)`` for RSL-RL policies."""
    actor_obs_dim = int(obs_groups_spec.get("obs", 0))
    flat_policy_obs_dim = int(
        sum(dim for group_name, dim in obs_groups_spec.items() if group_name != "critic")
    )
    return actor_obs_dim, flat_policy_obs_dim or actor_obs_dim


class RslRlVecEnvAdapter:
    """Adapter from the UniLab TorchEnv contract to RSL-RL's VecEnv."""

    def __init__(
        self,
        env: Any,
        device: str = "cpu",
        policy_obs_mode: str = "flat",
    ) -> None:
        if policy_obs_mode == "auto":
            policy_obs_mode = "flat"
        if policy_obs_mode not in {"actor", "flat"}:
            raise ValueError(
                f"Unsupported policy_obs_mode={policy_obs_mode!r}; expected 'actor' or 'flat'."
            )

        self.env = env
        self.cfg = env.cfg
        self.device = device
        self.policy_obs_mode = policy_obs_mode
        self.num_envs = env.num_envs
        self.observation_space = env.observation_space
        self.action_space = env.action_space

        self._actor_obs_dim, self._flat_obs_dim = get_policy_obs_dims(env.obs_groups_spec)
        self.num_obs = (
            self._actor_obs_dim if self.policy_obs_mode == "actor" else self._flat_obs_dim
        )
        self.num_privileged_obs = int(env.obs_groups_spec.get("critic", self.num_obs))
        action_shape = env.action_space.shape
        if action_shape is None:
            raise ValueError("env.action_space.shape must be defined")
        self.num_actions = int(action_shape[0])

        self.episode_returns = torch.zeros(self.num_envs, device=device)
        self.episode_lengths = torch.zeros(self.num_envs, device=device)
        self.max_episode_length = math.ceil(env.cfg.max_episode_seconds / env.cfg.ctrl_dt)
        self.reset()

    @property
    def episode_length_buf(self) -> torch.Tensor:
        """RSL-RL VecEnv contract view of the adapter's episode bookkeeping."""
        return self.episode_lengths

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor) -> None:
        # RSL-RL's init_at_random_ep_len assigns a fresh tensor here at learn()
        # start. Propagate the staggered counters into the wrapped env so the
        # env's own timeout accounting (ManagerBasedRlEnv.episode_length_buf,
        # mirrored from state.info["steps"]) staggers too — otherwise only the
        # adapter's local bookkeeping would randomize. Cold path: the runner
        # calls this at most once per learn().
        value = value.to(device=self.device).clone()
        self.episode_lengths = value
        set_env_episode_lengths = getattr(self.env, "set_episode_length_buf", None)
        if callable(set_env_episode_lengths):
            set_env_episode_lengths(
                value.to(device=self.env.device, dtype=torch.int64).contiguous()
            )

    def _policy_obs(self, obs: dict[str, Any]) -> torch.Tensor:
        if self.policy_obs_mode == "actor":
            return _to_policy_tensor(obs["obs"], self.device)

        policy_groups = [
            _to_policy_tensor(value, self.device)
            for group_name, value in obs.items()
            if group_name != "critic"
        ]
        if not policy_groups:
            raise KeyError("Observation dict must contain at least one non-critic group")
        if len(policy_groups) == 1:
            return policy_groups[0]
        return torch.cat(policy_groups, dim=-1)

    def _obs_to_tensordict(self, obs: dict[str, Any]) -> TensorDict:
        td_dict: dict[str, torch.Tensor] = {
            "actor": _to_policy_tensor(obs["obs"], self.device),
            "policy": self._policy_obs(obs),
        }
        if "critic" in obs:
            td_dict["critic"] = _to_policy_tensor(obs["critic"], self.device)
        return TensorDict(td_dict, batch_size=self.num_envs, device=self.device)

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        env_device = self.env.device
        if not isinstance(actions, torch.Tensor):
            raise TypeError(
                f"RslRlVecEnvAdapter action must be a torch.Tensor, got {type(actions).__name__}"
            )
        action_tensor = actions.to(device=env_device, dtype=torch.float32).contiguous()
        state = self.env.step(action_tensor)
        rewards = _to_policy_tensor(state.reward, self.device)
        dones = _to_policy_tensor(state.terminated | state.truncated, self.device).bool()

        self.episode_returns += rewards
        self.episode_lengths += 1

        infos: dict[str, torch.Tensor | dict[str, Any]] = {}
        if bool(dones.any()):
            infos["time_outs"] = _to_policy_tensor(state.truncated, self.device).bool()
            self.episode_returns.masked_fill_(dones, 0.0)
            self.episode_lengths.masked_fill_(dones, 0)

        if "log" in state.info:
            infos["log"] = state.info["log"]

        return self._obs_to_tensordict(state.obs), rewards, dones, infos

    def reset(self) -> tuple[TensorDict, dict[str, Any]]:
        if self.env.state is None:
            self.env.init_state()

        obs_out, info = self.env.reset()
        self.episode_returns[:] = 0
        self.episode_lengths[:] = 0
        return self._obs_to_tensordict(obs_out), info

    def get_observations(self) -> TensorDict:
        assert self.env.state is not None
        return self._obs_to_tensordict(self.env.state.obs)

    def get_privileged_observations(self) -> torch.Tensor:
        assert self.env.state is not None
        obs = self.env.state.obs
        return _to_policy_tensor(obs.get("critic", obs["obs"]), self.device)

    def export_training_state(self) -> dict[str, Any] | None:
        """Export the wrapped environment's plain versioned training state."""
        export = getattr(self.env, "export_training_state", None)
        if not callable(export):
            return None
        state = export()
        if not isinstance(state, dict):
            raise TypeError("TorchEnv training state export must return a dictionary")
        return state

    def import_training_state(self, state: dict[str, Any]) -> None:
        """Restore the wrapped environment's plain versioned training state."""
        importer = getattr(self.env, "import_training_state", None)
        if not callable(importer):
            raise TypeError(
                "TorchEnv does not implement import_training_state; cannot restore "
                "checkpoint environment state"
            )
        importer(state)

    def close(self) -> None:
        self.env.close()
