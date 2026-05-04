"""HORA-owned RSL-RL wrapper helpers for teacher-policy runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from tensordict import TensorDict

from unilab.base.final_observation import resolve_terminal_observation_contract
from unilab.training.rsl_rl import RslRlVecEnvWrapper
from unilab.utils.tensor import to_numpy, to_torch

from .observations import build_hora_obs_tensordict
from .runtime import is_hora_ppo_runtime


@dataclass(frozen=True)
class HoraRslRlPPORuntime:
    """Resolved HORA PPO runtime consumed by the generic RSL-RL script."""

    wrapper_cls: type[RslRlVecEnvWrapper]


def resolve_hora_ppo_runtime(
    rl_cfg: dict[str, Any],
) -> HoraRslRlPPORuntime | None:
    """Resolve HORA PPO entrypoints from an explicit runtime marker."""
    if not is_hora_ppo_runtime(rl_cfg):
        return None
    return HoraRslRlPPORuntime(wrapper_cls=HoraRslRlVecEnvWrapper)


def resolve_hora_ppo_wrapper_cls(
    rl_cfg: dict[str, Any],
) -> type[RslRlVecEnvWrapper] | None:
    """Return the HORA-specific PPO wrapper class when the config selects it.

    Args:
        rl_cfg: Resolved algorithm config dictionary from Hydra composition.

    Returns:
        ``HoraRslRlVecEnvWrapper`` when the owner config selects HORA PPO, otherwise
        ``None``.
    """
    runtime = resolve_hora_ppo_runtime(rl_cfg)
    if runtime is None:
        return None
    return runtime.wrapper_cls


class HoraRslRlVecEnvWrapper(RslRlVecEnvWrapper):
    """RSL-RL adapter that preserves HORA teacher-policy observation payloads."""

    def _obs_to_tensordict(
        self,
        obs: dict[str, Any],
        info: dict[str, Any] | None = None,
    ) -> TensorDict:
        """Convert env outputs to a HORA-aware TensorDict.

        Args:
            obs: Environment observation dict following the UniLab env contract.
            info: Optional env info dict containing HORA privileged payloads.

        Returns:
            TensorDict preserving generic observation keys plus HORA privileged inputs.
        """
        policy_obs = to_numpy(self._policy_obs(obs))
        return build_hora_obs_tensordict(
            obs,
            info=info,
            device=self.device,
            batch_size=self.num_envs,
            policy_obs=policy_obs,
        )

    def step(
        self, actions: torch.Tensor | np.ndarray
    ) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        """Step the wrapped env while keeping HORA bootstrap payloads intact.

        Args:
            actions: Torch or numpy action batch with shape ``(num_envs, action_dim)``.

        Returns:
            Tuple ``(obs_td, rewards, dones, infos)`` matching the RSL-RL VecEnv
            contract while preserving HORA privileged observations.
        """
        actions_np = to_numpy(actions)
        state = self.env.step(actions_np)
        rewards = to_torch(state.reward, self.device)
        dones = self._resolve_done(state)

        self.episode_returns += rewards
        self.episode_lengths += 1

        infos: dict[str, torch.Tensor | TensorDict | dict[str, Any]] = {}
        done_idx = torch.nonzero(dones).flatten()
        if len(done_idx) > 0:
            infos["time_outs"] = to_torch(state.truncated, self.device).bool()

            final_observation = self._resolve_final_observation(state)
            terminal_contract = resolve_terminal_observation_contract(
                next_obs_batch_size=self.num_envs,
                final_observation=final_observation,
                done=to_numpy(dones),
                info=state.info,
                truncated=to_numpy(infos["time_outs"]) if "time_outs" in infos else None,
            )
            if np.any(terminal_contract.timeout_terminal_mask) and final_observation is not None:
                infos["time_out_bootstrap_obs"] = self._obs_to_tensordict(final_observation)

            self.episode_returns[done_idx] = 0
            self.episode_lengths[done_idx] = 0

        if "log" in state.info:
            infos["log"] = state.info["log"]

        return (
            self._obs_to_tensordict(state.obs, state.info),
            rewards,
            dones,
            infos,
        )

    def reset(self) -> tuple[TensorDict, dict[str, Any]]:
        """Reset the wrapped env and preserve HORA privileged reset payloads.

        Args:
            None.

        Returns:
            Tuple ``(obs_td, info)`` where ``obs_td`` retains HORA privileged inputs.
        """
        if self.env.state is None:
            self.env.init_state()

        env_indices = np.arange(self.num_envs, dtype=np.int32)
        obs_out, info = self.env.reset(env_indices)
        self.episode_returns[:] = 0
        self.episode_lengths[:] = 0
        return self._obs_to_tensordict(obs_out, info), info

    def get_observations(self) -> TensorDict:
        """Return the current HORA-aware observation TensorDict.

        Args:
            None.

        Returns:
            TensorDict containing the current observation batch with HORA extras.
        """
        assert self.env.state is not None
        return self._obs_to_tensordict(self.env.state.obs, self.env.state.info)
