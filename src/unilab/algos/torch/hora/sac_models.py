"""HORA SAC actor models."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Tuple, cast

import torch
import torch.nn as nn


def _build_mlp(
    input_dim: int,
    hidden_dims: Sequence[int],
    *,
    activation: type[nn.Module],
    use_layer_norm: bool,
    device: str | torch.device,
) -> tuple[nn.Sequential, int]:
    layers: list[nn.Module] = []
    current_dim = int(input_dim)
    for hidden_dim in hidden_dims:
        next_dim = int(hidden_dim)
        layers.append(nn.Linear(current_dim, next_dim, device=device))
        if use_layer_norm:
            layers.append(nn.LayerNorm(next_dim, device=device))
        layers.append(activation())
        current_dim = next_dim
    return nn.Sequential(*layers), current_dim


class HoraSACActor(nn.Module):
    """Privileged HORA teacher actor for SAC.

    The named modules below are intentionally stable for later distillation:
    ``priv_encoder``, ``actor_trunk``, ``action_mean_head``, and
    ``action_logstd_head``.
    """

    action_scale: torch.Tensor
    action_bias: torch.Tensor

    def __init__(
        self,
        obs_dim: int,
        priv_info_dim: int,
        action_dim: int,
        *,
        hidden_dim: int = 512,
        priv_info_embed_dim: int = 9,
        priv_mlp_hidden_dims: Sequence[int] = (256, 128, 9),
        log_std_max: float = 0.0,
        log_std_min: float = -5.0,
        use_tanh: bool = True,
        use_layer_norm: bool = True,
        device: str | torch.device = "cpu",
        action_scale: torch.Tensor | None = None,
        action_bias: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.priv_info_dim = int(priv_info_dim)
        self.action_dim = int(action_dim)
        self.priv_info_embed_dim = int(priv_info_embed_dim)
        self.log_std_max = float(log_std_max)
        self.log_std_min = float(log_std_min)
        self.use_tanh = bool(use_tanh)

        self.priv_encoder, priv_out_dim = _build_mlp(
            self.priv_info_dim,
            tuple(priv_mlp_hidden_dims),
            activation=nn.SiLU,
            use_layer_norm=use_layer_norm,
            device=device,
        )
        if priv_out_dim != self.priv_info_embed_dim:
            self.priv_projection = nn.Linear(
                priv_out_dim,
                self.priv_info_embed_dim,
                device=device,
            )
        else:
            self.priv_projection = nn.Identity()

        trunk_dims = (hidden_dim, hidden_dim // 2, hidden_dim // 4)
        self.actor_trunk, trunk_out_dim = _build_mlp(
            self.obs_dim + self.priv_info_embed_dim,
            trunk_dims,
            activation=nn.SiLU,
            use_layer_norm=use_layer_norm,
            device=device,
        )
        self.action_mean_head = nn.Linear(trunk_out_dim, self.action_dim, device=device)
        self.action_logstd_head = nn.Linear(trunk_out_dim, self.action_dim, device=device)

        nn.init.constant_(self.action_mean_head.weight, 0.0)
        nn.init.constant_(self.action_mean_head.bias, 0.0)
        nn.init.constant_(self.action_logstd_head.weight, 0.0)
        nn.init.constant_(self.action_logstd_head.bias, 0.0)

        if action_scale is not None:
            self.register_buffer("action_scale", action_scale.to(device))
        else:
            self.register_buffer("action_scale", torch.ones(self.action_dim, device=device))
        if action_bias is not None:
            self.register_buffer("action_bias", action_bias.to(device))
        else:
            self.register_buffer("action_bias", torch.zeros(self.action_dim, device=device))

    def encode_privileged_info(self, priv_info: torch.Tensor) -> torch.Tensor:
        encoded = self.priv_encoder(priv_info)
        return torch.tanh(self.priv_projection(encoded))

    def _distribution_params(
        self,
        obs: torch.Tensor,
        priv_info: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode_privileged_info(priv_info)
        latent = self.actor_trunk(torch.cat([obs, z], dim=-1))
        mean = self.action_mean_head(latent)
        log_std = self.action_logstd_head(latent)
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1)
        mean = torch.clamp(mean, -10.0, 10.0)
        mean = torch.nan_to_num(mean, nan=0.0)
        log_std = torch.nan_to_num(log_std, nan=self.log_std_min)
        return mean, log_std

    def forward(
        self,
        obs: torch.Tensor,
        priv_info: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std = self._distribution_params(obs, priv_info)
        if self.use_tanh:
            action = torch.tanh(mean) * self.action_scale + self.action_bias
        else:
            action = mean
        return action, mean, log_std

    def as_export_module(self) -> nn.Module:
        """Return a wrapper with explicit actor/priv inputs for ONNX export."""
        actor = self

        class _Wrapper(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.base = actor

            def forward(self, obs: torch.Tensor, priv_info: torch.Tensor) -> torch.Tensor:
                action, _, _ = self.base(obs, priv_info)
                return cast(torch.Tensor, action)

        return _Wrapper()

    def get_actions_and_log_probs(
        self,
        obs: torch.Tensor,
        priv_info: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _, mean, log_std = self(obs, priv_info)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        raw_action = dist.rsample()

        if self.use_tanh:
            tanh_action = torch.tanh(raw_action)
            action = tanh_action * self.action_scale + self.action_bias
            log_prob = dist.log_prob(raw_action)
            log_prob -= torch.log(1 - tanh_action.pow(2) + 1e-6)
            log_prob -= torch.log(self.action_scale + 1e-6)
        else:
            action = raw_action
            log_prob = dist.log_prob(raw_action)

        return action, log_prob.sum(1), log_std

    @torch.no_grad()
    def explore(
        self,
        obs: torch.Tensor,
        priv_info: torch.Tensor,
        deterministic: bool = False,
    ) -> torch.Tensor:
        _, mean, log_std = self.forward(obs, priv_info)
        if deterministic:
            if self.use_tanh:
                return torch.tanh(mean) * self.action_scale + self.action_bias
            return mean

        std = log_std.exp()
        raw_action = torch.distributions.Normal(mean, std).rsample()
        if self.use_tanh:
            return torch.tanh(raw_action) * self.action_scale + self.action_bias
        return raw_action
