"""HORA-owned FastSAC learner."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.optim as optim

from unilab.algos.torch.fast_sac.learner import FastSACLearner
from unilab.algos.torch.hora.sac_models import HoraSACActor


def derive_priv_info_from_critic_obs(
    actor_obs: torch.Tensor,
    critic_obs: torch.Tensor,
    *,
    context: str,
) -> torch.Tensor:
    """Return the privileged tail encoded in the critic observation contract."""
    actor_dim = int(actor_obs.shape[-1])
    critic_dim = int(critic_obs.shape[-1])
    if critic_dim <= actor_dim:
        raise ValueError(
            f"HORA-SAC {context} requires critic observations to include privileged tail "
            f"features; got actor_dim={actor_dim}, critic_dim={critic_dim}."
        )
    return critic_obs[..., actor_dim:]


class HoraSACLearner(FastSACLearner):
    """FastSAC learner variant whose actor consumes HORA privileged info."""

    def __init__(
        self,
        *,
        obs_dim: int,
        critic_obs_dim: int,
        priv_info_dim: int,
        action_dim: int,
        device: str = "cpu",
        actor_hidden_dim: int = 512,
        priv_info_embed_dim: int = 9,
        priv_mlp_hidden_dims: Sequence[int] = (256, 128, 9),
        log_std_max: float = 0.0,
        log_std_min: float = -5.0,
        use_tanh: bool = True,
        use_layer_norm: bool = True,
        actor_lr: float = 3e-4,
        weight_decay: float = 0.001,
        use_symmetry: bool = False,
        symmetry_augmentation: Any | None = None,
        **kwargs: Any,
    ) -> None:
        if use_symmetry or symmetry_augmentation is not None:
            raise ValueError("HORA-SAC does not support symmetry augmentation.")
        if int(priv_info_dim) <= 0:
            raise ValueError(f"HORA-SAC requires positive priv_info_dim, got {priv_info_dim}.")

        super().__init__(
            obs_dim=obs_dim,
            critic_obs_dim=critic_obs_dim,
            action_dim=action_dim,
            device=device,
            actor_hidden_dim=actor_hidden_dim,
            log_std_max=log_std_max,
            log_std_min=log_std_min,
            use_tanh=use_tanh,
            use_layer_norm=use_layer_norm,
            actor_lr=actor_lr,
            weight_decay=weight_decay,
            use_symmetry=False,
            symmetry_augmentation=None,
            **kwargs,
        )
        self.priv_info_dim = int(priv_info_dim)
        self.actor = HoraSACActor(
            obs_dim=obs_dim,
            priv_info_dim=self.priv_info_dim,
            action_dim=action_dim,
            hidden_dim=actor_hidden_dim,
            priv_info_embed_dim=priv_info_embed_dim,
            priv_mlp_hidden_dims=tuple(priv_mlp_hidden_dims),
            log_std_max=log_std_max,
            log_std_min=log_std_min,
            use_tanh=use_tanh,
            use_layer_norm=use_layer_norm,
            device=device,
        )
        _fused = isinstance(device, str) and device.startswith("cuda")
        self.actor_optimizer = optim.AdamW(
            self.actor.parameters(),
            lr=actor_lr,
            weight_decay=weight_decay,
            fused=_fused,
            betas=(0.9, 0.95),
        )

    def _get_actions_and_log_probs_for_critic(
        self,
        actor_obs: torch.Tensor,
        critic_obs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        priv_info = derive_priv_info_from_critic_obs(
            actor_obs,
            critic_obs,
            context="critic update",
        )
        return self.actor.get_actions_and_log_probs(actor_obs, priv_info)

    def _get_actions_and_log_probs_for_actor(
        self,
        actor_obs: torch.Tensor,
        critic_obs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        priv_info = derive_priv_info_from_critic_obs(
            actor_obs,
            critic_obs,
            context="actor update",
        )
        return self.actor.get_actions_and_log_probs(actor_obs, priv_info)
