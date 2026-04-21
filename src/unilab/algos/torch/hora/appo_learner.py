"""HORA-owned APPO learner with grouped actor and privileged observations."""

from __future__ import annotations

from itertools import chain

import torch
import torch.nn as nn
from tensordict import TensorDict

from unilab.algos.torch.appo.learner import APPOLearner, vtrace_advantages


def _build_hora_obs_td(
    actor_obs: torch.Tensor,
    *,
    device: str,
    priv_info: torch.Tensor | None = None,
) -> TensorDict:
    if priv_info is not None:
        return TensorDict(
            {"actor": actor_obs, "priv_info": priv_info},
            batch_size=actor_obs.shape[0],
            device=device,
        )
    return TensorDict({"policy": actor_obs}, batch_size=actor_obs.shape[0], device=device)


class HoraAPPOLearner(APPOLearner):
    """APPO learner variant for HORA grouped observations."""

    def process_batch(self, batch_dict):
        """Compute V-trace targets for grouped HORA rollouts."""
        obs = batch_dict["observations"]
        critic_base = batch_dict.get("critic", None)
        priv_info = batch_dict.get("priv_info", None)
        rewards = batch_dict["rewards"]
        dones = batch_dict["dones"].float()
        last_obs = batch_dict["last_obs"]
        last_priv_info = batch_dict.get("last_priv_info", None)
        behavior_log_probs = batch_dict["actions_log_prob"]
        actions = batch_dict["actions"]

        T, N = obs.shape[:2]
        obs_flat = obs.flatten(0, 1)
        priv_info_flat = priv_info.flatten(0, 1) if priv_info is not None else None

        obs_td = _build_hora_obs_td(obs_flat, device=self.device, priv_info=priv_info_flat)
        last_obs_td = _build_hora_obs_td(last_obs, device=self.device, priv_info=last_priv_info)

        if critic_base is None:
            critic_base = obs
        critic_obs = critic_base
        critic_obs_flat = critic_obs.flatten(0, 1)
        critic_obs_td = _build_hora_obs_td(obs_flat, device=self.device, priv_info=priv_info_flat)
        critic_last_obs_td = _build_hora_obs_td(
            last_obs,
            device=self.device,
            priv_info=last_priv_info,
        )

        if hasattr(self.actor, "update_normalization"):
            self.actor.update_normalization(obs_td)
            self.actor.update_normalization(last_obs_td)
        if hasattr(self.critic, "update_normalization"):
            self.critic.update_normalization(critic_obs_td)
            self.critic.update_normalization(critic_last_obs_td)

        batch_dict["_critic_obs_flat"] = critic_obs_flat
        batch_dict["_critic_obs_td"] = critic_obs_td

        with torch.inference_mode():
            values_flat = self.critic(critic_obs_td)
            last_values = self.critic(critic_last_obs_td).squeeze(-1)
        values = values_flat.view(T, N, -1).squeeze(-1)

        actions_flat = actions.flatten(0, 1)
        with torch.inference_mode():
            self.target_actor(obs_td, stochastic_output=True)
            target_log_probs_flat = self.target_actor.get_output_log_prob(actions_flat)
            batch_dict["_old_mu"] = self.target_actor.output_mean.clone()
            batch_dict["_old_sigma"] = self.target_actor.output_std.clone()
        target_log_probs = target_log_probs_flat.view(T, N)

        vs, advantages = vtrace_advantages(
            behavior_log_probs=behavior_log_probs,
            target_log_probs=target_log_probs,
            rewards=rewards,
            values=values,
            bootstrap_values=last_values,
            dones=dones,
            gamma=self.gamma,
            clip_rho=self.vtrace_clip_rho,
            clip_c=self.vtrace_clip_c,
        )

        batch_dict["values"] = values
        batch_dict["advantages"] = advantages
        batch_dict["returns"] = vs
        batch_dict["target_log_probs"] = target_log_probs
        batch_dict["_obs_td"] = obs_td

        return batch_dict

    def update(self, batch_dict):
        """Perform APPO update for grouped HORA observations."""
        obs_flat = batch_dict["observations"].flatten(0, 1)
        priv_info = batch_dict.get("priv_info")
        priv_info_flat = priv_info.flatten(0, 1) if priv_info is not None else None
        actions_flat = batch_dict["actions"].flatten(0, 1)
        returns_flat = batch_dict["returns"].flatten(0, 1)
        advantages_flat = batch_dict["advantages"].flatten(0, 1)
        behavior_log_probs_flat = batch_dict["actions_log_prob"].flatten(0, 1)
        old_values_flat = batch_dict["values"].flatten(0, 1)
        target_log_probs_flat = batch_dict["target_log_probs"].flatten(0, 1)
        advantages_flat = (advantages_flat - advantages_flat.mean()) / (
            advantages_flat.std() + 1e-8
        )

        obs_td = batch_dict.get("_obs_td")
        if obs_td is None:
            obs_td = _build_hora_obs_td(obs_flat, device=self.device, priv_info=priv_info_flat)

        critic_obs_td = batch_dict.get("_critic_obs_td")
        if critic_obs_td is None:
            critic_obs_td = _build_hora_obs_td(
                obs_flat,
                device=self.device,
                priv_info=priv_info_flat,
            )

        with torch.inference_mode():
            old_mu_flat = batch_dict["_old_mu"]
            old_sigma_flat = batch_dict["_old_sigma"]

        batch_size = obs_flat.shape[0]
        mini_batch_size = batch_size // self.num_mini_batches

        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_kl = 0.0
        num_updates = 0

        for _epoch in range(self.num_learning_epochs):
            indices = torch.randperm(batch_size, device=self.device)

            for i in range(self.num_mini_batches):
                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                batch_idx = indices[start:end]

                obs_mini_td = obs_td[batch_idx]
                critic_obs_mini_td = critic_obs_td[batch_idx]
                actions_mini = actions_flat[batch_idx]
                target_values_mini = returns_flat[batch_idx]
                advantages_mini = advantages_flat[batch_idx]
                behavior_logp_mini = behavior_log_probs_flat[batch_idx]
                old_values_mini = old_values_flat[batch_idx]
                target_logp_mini = target_log_probs_flat[batch_idx]
                old_mu_mini = old_mu_flat[batch_idx]
                old_sigma_mini = old_sigma_flat[batch_idx]

                _ = self.actor(obs_mini_td, stochastic_output=True)
                current_log_prob = self.actor.get_output_log_prob(actions_mini)
                value = self.critic(critic_obs_mini_td).squeeze(-1)
                entropy = self.actor.output_entropy.mean()

                mu = self.actor.output_mean
                sigma = self.actor.output_std

                with torch.no_grad():
                    clipped_rho = torch.clamp(
                        torch.exp(behavior_logp_mini - target_logp_mini), max=1.0
                    )
                ratio = clipped_rho * torch.exp(current_log_prob - behavior_logp_mini)

                surrogate = -advantages_mini * ratio
                surrogate_clipped = -advantages_mini * torch.clamp(
                    ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
                )
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

                if self.desired_kl is not None and self.schedule == "adaptive":
                    with torch.inference_mode():
                        kl = torch.sum(
                            torch.log(sigma / old_sigma_mini + 1e-5)
                            + (old_sigma_mini.pow(2) + (old_mu_mini - mu).pow(2))
                            / (2.0 * sigma.pow(2))
                            - 0.5,
                            dim=-1,
                        )
                        kl_mean = torch.mean(kl)

                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                        for param_group in self.optimizer.param_groups:
                            param_group["lr"] = self.learning_rate

                        mean_kl += kl_mean.item()

                if self.use_clipped_value_loss:
                    value_clipped = old_values_mini + (value - old_values_mini).clamp(
                        -self.clip_param, self.clip_param
                    )
                    value_losses = (value - target_values_mini).pow(2)
                    value_losses_clipped = (value_clipped - target_values_mini).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (value - target_values_mini).pow(2).mean()

                loss = (
                    surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy
                )

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    chain(self.actor.parameters(), self.critic.parameters()), self.max_grad_norm
                )
                self.optimizer.step()

                mean_value_loss += value_loss.item()
                mean_surrogate_loss += surrogate_loss.item()
                mean_entropy += entropy.item()
                num_updates += 1

        self._update_counter += 1
        if self._update_counter % self.target_update_freq == 0:
            self.update_target_network()

        num_updates = max(num_updates, 1)
        return {
            "surrogate_loss": mean_surrogate_loss / num_updates,
            "value_loss": mean_value_loss / num_updates,
            "entropy": mean_entropy / num_updates,
            "kl": mean_kl / num_updates if self.schedule == "adaptive" else 0.0,
        }
