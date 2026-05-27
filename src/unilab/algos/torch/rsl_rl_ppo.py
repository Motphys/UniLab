from __future__ import annotations

from typing import Any

import torch
from rsl_rl.algorithms import PPO
from tensordict import TensorDict


class FinalObservationAwarePPO(PPO):
    """PPO variant that bootstraps time limits from env final_observation."""

    def __init__(
        self,
        *args: Any,
        enable_compile: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.enable_compile = (
            bool(enable_compile)
            and torch.device(self.device).type == "cuda"
            and hasattr(torch, "compile")
        )
        if self.enable_compile:
            self._compile_training_methods()

    def _compile_training_methods(self) -> None:
        compile_fn = getattr(torch, "compile", None)
        if compile_fn is None or torch.device(self.device).type != "cuda":
            return

        compile_kwargs = {"options": {"triton.cudagraphs": False}}
        if hasattr(self.actor, "mlp"):
            self.actor.mlp.forward = compile_fn(self.actor.mlp.forward, **compile_kwargs)
        if hasattr(self.critic, "mlp"):
            self.critic.mlp.forward = compile_fn(self.critic.mlp.forward, **compile_kwargs)
        if self.rnd:
            self.rnd.predictor.forward = compile_fn(self.rnd.predictor.forward, **compile_kwargs)
            self.rnd.target.forward = compile_fn(self.rnd.target.forward, **compile_kwargs)

    def process_env_step(
        self,
        obs: TensorDict,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        extras: dict[str, torch.Tensor | TensorDict],
    ) -> None:
        self.actor.update_normalization(obs)
        self.critic.update_normalization(obs)
        if self.rnd:
            self.rnd.update_normalization(obs)

        self.transition.rewards = rewards.clone()
        self.transition.dones = dones

        if self.rnd:
            self.intrinsic_rewards = self.rnd.get_intrinsic_reward(obs)
            self.transition.rewards += self.intrinsic_rewards

        timeouts = extras.get("time_outs")
        timeout_bootstrap_obs = extras.get("time_out_bootstrap_obs")
        if isinstance(timeouts, torch.Tensor):
            timeout_mask = timeouts.to(self.device).float()
            if timeout_bootstrap_obs is not None and torch.count_nonzero(timeout_mask) > 0:
                bootstrap_obs = timeout_bootstrap_obs.to(self.device)
                bootstrap_values = self.critic(bootstrap_obs).detach()
                self.transition.rewards += self.gamma * torch.squeeze(
                    bootstrap_values * timeout_mask.unsqueeze(1), 1
                )
            else:
                transition_values = self.transition.values
                assert transition_values is not None
                self.transition.rewards += self.gamma * torch.squeeze(
                    transition_values * timeout_mask.unsqueeze(1), 1
                )

        self.storage.add_transition(self.transition)
        self.transition.clear()
        self.actor.reset(dones)
        self.critic.reset(dones)
