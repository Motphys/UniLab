# SPDX-License-Identifier: BSD-3-Clause
"""PPO update with a concurrently trained supervised state estimator."""

from __future__ import annotations

import contextlib
from typing import Any

import torch
from tensordict import TensorDict
from torch import nn, optim

from .actor_critic import CSEActorCritic
from .storage import CSERolloutStorage


class CSEPPO:
    actor_critic: CSEActorCritic

    def __init__(
        self,
        actor_critic: CSEActorCritic,
        num_learning_epochs: int = 1,
        num_mini_batches: int = 1,
        clip_param: float = 0.2,
        gamma: float = 0.998,
        lam: float = 0.95,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.0,
        learning_rate: float = 1e-3,
        max_grad_norm: float = 1.0,
        use_clipped_value_loss: bool = True,
        schedule: str = "fixed",
        desired_kl: float | None = 0.01,
        min_learning_rate: float = 1e-5,
        max_learning_rate: float = 1e-2,
        min_policy_std: float = 1e-2,
        max_policy_std: float | None = None,
        use_amp: bool = False,
        amp_dtype: str = "bfloat16",
        device: str = "cpu",
        **kwargs: Any,
    ) -> None:
        del kwargs
        self.device = device
        self.actor_critic = actor_critic.to(device)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=float(learning_rate))
        self.storage: CSERolloutStorage | None = None
        self.transition = CSERolloutStorage.Transition()
        self.num_learning_epochs = int(num_learning_epochs)
        self.num_mini_batches = int(num_mini_batches)
        self.clip_param = float(clip_param)
        self.gamma = float(gamma)
        self.lam = float(lam)
        self.value_loss_coef = float(value_loss_coef)
        self.entropy_coef = float(entropy_coef)
        self.max_grad_norm = float(max_grad_norm)
        self.use_clipped_value_loss = bool(use_clipped_value_loss)
        self.learning_rate = float(learning_rate)
        self.schedule = schedule
        self.desired_kl = desired_kl
        self.min_learning_rate = float(min_learning_rate)
        self.max_learning_rate = float(max_learning_rate)
        self.min_policy_std = float(min_policy_std)
        self.max_policy_std = None if max_policy_std is None else float(max_policy_std)
        self._amp_enabled = bool(use_amp) and "cuda" in str(device)
        self._amp_dtype = torch.bfloat16 if amp_dtype == "bfloat16" else torch.float16

    def init_storage(
        self,
        num_envs: int,
        num_transitions_per_env: int,
        actor_obs_shape,
        critic_obs_shape,
        action_shape,
    ) -> None:
        self.storage = CSERolloutStorage(
            num_envs,
            num_transitions_per_env,
            actor_obs_shape,
            critic_obs_shape,
            action_shape,
            self.device,
        )

    def test_mode(self) -> None:
        self.actor_critic.eval()

    def train_mode(self) -> None:
        self.actor_critic.train()

    def act(self, obs: torch.Tensor, critic_obs: torch.Tensor) -> torch.Tensor:
        self.transition.actions = self.actor_critic.act(obs).detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(
            self.transition.actions
        ).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        return self.transition.actions

    def process_env_step(
        self,
        next_obs: TensorDict | torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        extras: dict[str, Any],
    ) -> None:
        del next_obs
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        timeouts = extras.get("time_outs")
        bootstrap = extras.get("time_out_bootstrap_obs")
        if isinstance(timeouts, torch.Tensor):
            mask = timeouts.to(self.device).bool().view(-1).float()
            if bootstrap is not None and torch.count_nonzero(mask) > 0:
                values = self.actor_critic.evaluate(_critic_obs(bootstrap.to(self.device))).detach()
            else:
                values = self.transition.values
                assert values is not None
            correction = self.gamma * torch.squeeze(values * mask.unsqueeze(1), 1)
            rewards = self.transition.rewards
            assert rewards is not None
            if rewards.ndim == 2 and rewards.shape[-1] == 1:
                correction = correction.unsqueeze(1)
            self.transition.rewards = rewards + correction
        assert self.storage is not None
        self.storage.add_transition(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)

    def compute_returns(self, last_critic_obs: torch.Tensor) -> None:
        assert self.storage is not None
        self.storage.compute_returns(
            self.actor_critic.evaluate(last_critic_obs).detach(), self.gamma, self.lam
        )

    def _amp_ctx(self):
        return (
            torch.autocast(device_type="cuda", dtype=self._amp_dtype)
            if self._amp_enabled
            else contextlib.nullcontext()
        )

    def _adapt_learning_rate(self, kl_mean: float) -> None:
        if self.desired_kl is None or self.schedule != "adaptive":
            return
        if kl_mean > self.desired_kl * 2:
            self.learning_rate = max(self.min_learning_rate, self.learning_rate / 1.5)
        elif 0 < kl_mean < self.desired_kl / 2:
            self.learning_rate = min(self.max_learning_rate, self.learning_rate * 1.5)
        for group in self.optimizer.param_groups:
            group["lr"] = self.learning_rate

    def update(self) -> tuple[float, float, float]:
        assert self.storage is not None
        value_total = policy_total = estimator_total = 0.0
        for batch in self.storage.mini_batch_generator(
            self.num_mini_batches, self.num_learning_epochs
        ):
            (
                obs,
                critic_obs,
                actions,
                old_values,
                advantages,
                returns,
                old_log_prob,
                old_mu,
                old_sigma,
            ) = batch
            with self._amp_ctx():
                self.actor_critic.act(obs)
                log_prob = self.actor_critic.get_actions_log_prob(actions)
                values = self.actor_critic.evaluate(critic_obs)
            log_prob, values = log_prob.float(), values.float()
            mu, sigma, entropy = (
                self.actor_critic.action_mean.float(),
                self.actor_critic.action_std,
                self.actor_critic.entropy,
            )
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma / old_sigma + 1e-5)
                        + (old_sigma.square() + (old_mu - mu).square()) / (2 * sigma.square())
                        - 0.5,
                        dim=-1,
                    )
                self._adapt_learning_rate(float(kl.mean()))
            ratio = torch.exp(log_prob - old_log_prob.squeeze())
            surrogate = torch.max(
                -advantages.squeeze() * ratio,
                -advantages.squeeze() * ratio.clamp(1 - self.clip_param, 1 + self.clip_param),
            ).mean()
            if self.use_clipped_value_loss:
                value_clipped = old_values + (values - old_values).clamp(
                    -self.clip_param, self.clip_param
                )
                value_loss = torch.max(
                    (values - returns).square(), (value_clipped - returns).square()
                ).mean()
            else:
                value_loss = (returns - values).square().mean()
            loss = (
                surrogate + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()
            )
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()
            self.actor_critic.std.data.clamp_(min=self.min_policy_std, max=self.max_policy_std)
            estimator_loss = self.actor_critic.estimator.update(
                obs, critic_obs, autocast_enabled=self._amp_enabled, autocast_dtype=self._amp_dtype
            )
            value_total += float(value_loss.item())
            policy_total += float(surrogate.item())
            estimator_total += estimator_loss
        updates = self.num_learning_epochs * self.num_mini_batches
        self.storage.clear()
        return value_total / updates, policy_total / updates, estimator_total / updates


def _critic_obs(obs: TensorDict | torch.Tensor) -> torch.Tensor:
    if isinstance(obs, TensorDict):
        for key in ("critic", "policy", "actor"):
            if key in obs.keys():
                return obs[key]
        raise KeyError("CSE-PPO TensorDict obs must contain critic, policy, or actor")
    return obs
