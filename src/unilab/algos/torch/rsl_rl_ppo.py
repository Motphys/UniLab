from __future__ import annotations

import math
from typing import Any, cast

import torch
from rsl_rl.algorithms import PPO
from tensordict import TensorDict

_LOG_2_PI = math.log(2.0 * math.pi)
_NORMAL_ENTROPY_OFFSET = 0.5 * (1.0 + _LOG_2_PI)


class FinalObservationAwarePPO(PPO):
    """PPO variant that bootstraps time limits from env final_observation."""

    learning_rate: float

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
        self._minibatch_loss_fn = self._minibatch_loss_tensors
        if self.enable_compile:
            self._compile_training_methods()

    def _compile_training_methods(self) -> None:
        compile_fn = getattr(torch, "compile", None)
        if compile_fn is None or torch.device(self.device).type != "cuda":
            return

        self._minibatch_loss_fn = compile_fn(
            self._minibatch_loss_tensors,
            mode="reduce-overhead",
            fullgraph=False,
        )

    @staticmethod
    def _model_obs_tensor(model: Any, obs: TensorDict) -> torch.Tensor:
        obs_groups = getattr(model, "obs_groups", None)
        if not obs_groups:
            raise RuntimeError("PPO compiled update requires model.obs_groups")
        tensors = [obs[group] for group in obs_groups]
        if len(tensors) == 1:
            return tensors[0]
        return torch.cat(tensors, dim=-1)

    def _supports_compiled_update_path(self) -> bool:
        if not self.enable_compile:
            return False
        if self.rnd or self.symmetry or self.is_multi_gpu:
            return False
        if self.actor.is_recurrent or self.critic.is_recurrent:
            return False
        distribution: Any = getattr(self.actor, "distribution", None)
        if distribution is None or not hasattr(distribution, "std_type"):
            return False
        if distribution.std_type == "scalar":
            return hasattr(distribution, "std_param")
        if distribution.std_type == "log":
            return hasattr(distribution, "log_std_param")
        return False

    def _actor_mean_std(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        distribution: Any = self.actor.distribution
        if distribution is None:
            raise RuntimeError("PPO actor must expose a stochastic distribution")

        mean = self.actor.mlp(self.actor.obs_normalizer(obs))
        if distribution.std_type == "scalar":
            std = distribution.std_param.expand_as(mean)
        else:
            std = torch.exp(distribution.log_std_param).expand_as(mean)
        return mean, std

    def _critic_value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic.mlp(self.critic.obs_normalizer(obs)).squeeze(-1)

    @staticmethod
    def _gaussian_log_prob(
        actions: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
    ) -> torch.Tensor:
        normalized = (actions - mean) / std
        return (-0.5 * (normalized.pow(2) + 2.0 * torch.log(std) + _LOG_2_PI)).sum(dim=-1)

    @staticmethod
    def _gaussian_entropy(std: torch.Tensor) -> torch.Tensor:
        return (torch.log(std) + _NORMAL_ENTROPY_OFFSET).sum(dim=-1)

    def _minibatch_loss_tensors(
        self,
        actor_obs: torch.Tensor,
        critic_obs: torch.Tensor,
        actions: torch.Tensor,
        target_values: torch.Tensor,
        advantages: torch.Tensor,
        old_actions_log_prob: torch.Tensor,
        old_values: torch.Tensor,
        old_mu: torch.Tensor,
        old_sigma: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, sigma = self._actor_mean_std(actor_obs)
        actions_log_prob = self._gaussian_log_prob(actions, mu, sigma)
        values = self._critic_value(critic_obs)
        entropy = self._gaussian_entropy(sigma).mean()

        old_actions_log_prob = old_actions_log_prob.squeeze(-1)
        old_values = old_values.squeeze(-1)
        target_values = target_values.squeeze(-1)
        advantages = advantages.squeeze(-1)

        ratio = torch.exp(actions_log_prob - old_actions_log_prob)
        surrogate = -advantages * ratio
        surrogate_clipped = -advantages * torch.clamp(
            ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
        )
        surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

        if self.use_clipped_value_loss:
            value_clipped = old_values + (values - old_values).clamp(
                -self.clip_param, self.clip_param
            )
            value_losses = (values - target_values).pow(2)
            value_losses_clipped = (value_clipped - target_values).pow(2)
            value_loss = torch.max(value_losses, value_losses_clipped).mean()
        else:
            value_loss = (target_values - values).pow(2).mean()

        kl = torch.sum(
            torch.log(sigma / old_sigma + 1e-5)
            + (old_sigma.pow(2) + (old_mu - mu).pow(2)) / (2.0 * sigma.pow(2))
            - 0.5,
            dim=-1,
        )
        kl_mean = torch.mean(kl)

        loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy
        return loss, surrogate_loss, value_loss, entropy, kl_mean

    def update(self) -> dict[str, float]:
        if not self._supports_compiled_update_path():
            return cast(dict[str, float], super().update())

        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0

        generator = self.storage.mini_batch_generator(
            self.num_mini_batches, self.num_learning_epochs
        )

        for batch in generator:
            actions = cast(torch.Tensor, batch.actions)
            values = cast(torch.Tensor, batch.values)
            advantages = cast(torch.Tensor, batch.advantages)
            returns = cast(torch.Tensor, batch.returns)
            old_actions_log_prob = cast(torch.Tensor, batch.old_actions_log_prob)
            old_distribution_params = batch.old_distribution_params
            if old_distribution_params is None:
                raise RuntimeError("PPO compiled update requires old distribution parameters")

            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            actor_obs = self._model_obs_tensor(self.actor, batch.observations)
            critic_obs = self._model_obs_tensor(self.critic, batch.observations)
            old_mu, old_sigma = old_distribution_params

            loss, surrogate_loss, value_loss, entropy, kl_mean = self._minibatch_loss_fn(
                actor_obs,
                critic_obs,
                actions,
                returns,
                advantages,
                old_actions_log_prob,
                values,
                old_mu,
                old_sigma,
            )

            if self.desired_kl is not None and self.schedule == "adaptive":
                kl_value = float(kl_mean.detach())
                learning_rate = float(self.learning_rate)
                if kl_value > self.desired_kl * 2.0:
                    learning_rate = max(1e-5, learning_rate / 1.5)
                elif kl_value < self.desired_kl / 2.0 and kl_value > 0.0:
                    learning_rate = min(1e-2, learning_rate * 1.5)

                self.learning_rate = learning_rate
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = learning_rate

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        self.storage.clear()
        return {
            "value": mean_value_loss / num_updates,
            "surrogate": mean_surrogate_loss / num_updates,
            "entropy": mean_entropy / num_updates,
        }

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
