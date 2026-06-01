"""HORA-owned APPO learner with grouped actor and privileged observations."""

from __future__ import annotations

from typing import cast

import torch
from tensordict import TensorDict

from unilab.algos.torch.appo.learner import (
    APPOLearner,
    _distribution_std,
    _sample_tensor_for_metric,
    vtrace_advantages,
)
from unilab.algos.torch.hora.models import HoraActorModel, HoraCriticModel


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


def _derive_priv_info_from_critic(
    actor_obs: torch.Tensor,
    critic_obs: torch.Tensor | None,
    *,
    context: str,
) -> torch.Tensor:
    if critic_obs is None:
        raise ValueError(f"HORA APPO {context} requires critic observations.")
    actor_dim = int(actor_obs.shape[-1])
    critic_dim = int(critic_obs.shape[-1])
    if critic_dim <= actor_dim:
        raise ValueError(
            f"HORA APPO {context} requires critic observations to include privileged tail "
            f"features; got actor_dim={actor_dim}, critic_dim={critic_dim}."
        )
    return critic_obs[..., actor_dim:]


class HoraAPPOLearner(APPOLearner):
    """APPO learner variant for HORA grouped observations."""

    def _minibatch_policy_value(
        self,
        obs_mini: torch.Tensor,
        critic_obs_mini: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        priv_info = _derive_priv_info_from_critic(
            obs_mini,
            critic_obs_mini,
            context="minibatch update",
        )
        actor = cast(HoraActorModel, self.actor)
        critic = cast(HoraCriticModel, self.critic)
        shared_actor = actor.shared
        mean = shared_actor.policy_mean_from_tensors(
            obs_mini,
            priv_info,
            prefer_student=actor.prefer_student,
        )
        std = _distribution_std(shared_actor.distribution, mean)
        value = critic.shared.value_from_tensors(
            obs_mini,
            priv_info,
            prefer_student=False,
        ).squeeze(-1)
        return mean, std, value

    def process_batch(self, batch_dict):
        """Compute V-trace targets for grouped HORA rollouts."""
        obs = batch_dict["observations"]
        critic_base = batch_dict.get("critic", None)
        rewards = batch_dict["rewards"]
        dones = batch_dict["dones"].float()
        last_obs = batch_dict["last_obs"]
        last_critic = batch_dict.get("last_critic", None)
        behavior_log_probs = batch_dict["actions_log_prob"]
        actions = batch_dict["actions"]

        T, N = obs.shape[:2]
        priv_info = _derive_priv_info_from_critic(
            obs,
            critic_base,
            context="rollout batch",
        )
        last_priv_info = _derive_priv_info_from_critic(
            last_obs,
            last_critic,
            context="bootstrap batch",
        )
        obs_flat = obs.flatten(0, 1)
        priv_info_flat = priv_info.flatten(0, 1)

        obs_td = _build_hora_obs_td(obs_flat, device=self.device, priv_info=priv_info_flat)
        last_obs_td = _build_hora_obs_td(last_obs, device=self.device, priv_info=last_priv_info)

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
            self.sync_target_actor_buffers()
        if hasattr(self.critic, "update_normalization"):
            self.critic.update_normalization(critic_obs_td)
            self.critic.update_normalization(critic_last_obs_td)

        batch_dict["_critic_obs_flat"] = critic_obs_flat

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
        with torch.inference_mode():
            rhos = torch.exp(target_log_probs - behavior_log_probs)
            rho_sample = _sample_tensor_for_metric(rhos)
            batch_dict["_appo_process_metrics"] = {
                "vtrace/rho_clip_fraction": float(
                    (rhos > float(self.vtrace_clip_rho)).float().mean().item()
                ),
                "vtrace/rho_raw_p99": float(torch.quantile(rho_sample, 0.99).item()),
            }

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

        return batch_dict
