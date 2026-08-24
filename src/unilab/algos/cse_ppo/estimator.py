# SPDX-License-Identifier: BSD-3-Clause
"""Concurrent state estimator used by the CSE-PPO actor."""

from __future__ import annotations

import contextlib
from collections.abc import Sequence
from typing import cast

import torch
from torch import nn, optim
from torch.nn import functional as F


def get_activation(name: str) -> nn.Module:
    activations = {
        "elu": nn.ELU,
        "selu": nn.SELU,
        "relu": nn.ReLU,
        "crelu": nn.ReLU,
        "silu": nn.SiLU,
        "lrelu": nn.LeakyReLU,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
    }
    try:
        return activations[name]()
    except KeyError as exc:
        raise ValueError(f"Unsupported activation: {name}") from exc


def _mlp(
    input_dim: int, output_dim: int, hidden_dims: Sequence[int], activation: str
) -> nn.Sequential:
    layers: list[nn.Module] = []
    last = int(input_dim)
    for dim in hidden_dims:
        layers.extend((nn.Linear(last, int(dim)), get_activation(activation)))
        last = int(dim)
    layers.append(nn.Linear(last, int(output_dim)))
    return nn.Sequential(*layers)


class CSEEstimator(nn.Module):
    """Supervised encoder/decoder for the privileged current-state target."""

    def __init__(
        self,
        temporal_steps: int,
        num_one_step_obs: int,
        num_pred: int = 12,
        enc_hidden_dims: Sequence[int] = (256, 128),
        latent_dim: int = 19,
        dec_hidden_dims: Sequence[int] = (64,),
        activation: str = "elu",
        learning_rate: float = 1e-5,
        max_grad_norm: float = 10.0,
        target_weights: Sequence[float] | None = None,
        target_start: int = 0,
        target_group_sizes: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        if temporal_steps <= 0:
            raise ValueError("temporal_steps must be positive")
        if num_one_step_obs <= 0:
            raise ValueError("num_one_step_obs must be positive")
        if num_pred <= 0:
            raise ValueError("num_pred must be positive")
        self.temporal_steps = int(temporal_steps)
        self.num_one_step_obs = int(num_one_step_obs)
        self.num_pred = int(num_pred)
        self.num_latent = int(latent_dim)
        self.target_start = int(target_start)
        self.max_grad_norm = float(max_grad_norm)
        self.target_group_sizes = (
            tuple(int(size) for size in target_group_sizes)
            if target_group_sizes is not None
            else None
        )
        if self.target_group_sizes is not None and sum(self.target_group_sizes) != self.num_pred:
            raise ValueError(
                f"target_group_sizes {self.target_group_sizes} must sum to num_pred {self.num_pred}"
            )
        weight_count = (
            len(self.target_group_sizes) if self.target_group_sizes is not None else self.num_pred
        )
        weights = (
            torch.ones(weight_count)
            if target_weights is None
            else torch.as_tensor(list(target_weights), dtype=torch.float32)
        )
        if weights.numel() != weight_count:
            kind = "groups" if self.target_group_sizes is not None else "num_pred"
            raise ValueError(f"target_weights length {weights.numel()} != {weight_count} ({kind})")
        self.register_buffer("target_weights", weights)
        self.encoder = _mlp(
            self.temporal_steps * self.num_one_step_obs,
            self.num_latent,
            enc_hidden_dims,
            activation,
        )
        self.decoder = _mlp(self.num_latent, self.num_pred, dec_hidden_dims, activation)
        self.learning_rate = float(learning_rate)
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

    def encode(self, obs_history: torch.Tensor) -> torch.Tensor:
        return self.encoder(obs_history)

    def get_latent(self, obs_history: torch.Tensor) -> torch.Tensor:
        return self.encoder(obs_history.detach()).detach()

    def forward(self, obs_history: torch.Tensor) -> torch.Tensor:
        return self.get_latent(obs_history)

    def predict(self, obs_history: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(obs_history.detach())).detach()

    def _regression_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        weights = cast(torch.Tensor, self.target_weights).to(pred.device)
        if self.target_group_sizes is None:
            return (weights * F.mse_loss(pred, target, reduction="none")).mean()
        loss = pred.new_zeros(())
        offset = 0
        for size, weight in zip(self.target_group_sizes, weights):
            part = slice(offset, offset + size)
            loss = loss + F.mse_loss(pred[:, part] * weight, target[:, part] * weight)
            offset += size
        return loss

    def update(
        self,
        obs_history: torch.Tensor,
        critic_obs: torch.Tensor,
        lr: float | None = None,
        autocast_enabled: bool = False,
        autocast_dtype: torch.dtype | None = None,
    ) -> float:
        if lr is not None:
            self.learning_rate = float(lr)
            for group in self.optimizer.param_groups:
                group["lr"] = self.learning_rate
        end = self.target_start + self.num_pred
        if critic_obs.shape[-1] < end:
            raise ValueError(
                "critic_obs is too small for the CSE estimator target slice: "
                f"shape={tuple(critic_obs.shape)}, target=[{self.target_start}:{end}]"
            )
        target = critic_obs[:, self.target_start : end].detach()
        amp_ctx = (
            torch.autocast(device_type="cuda", dtype=autocast_dtype or torch.bfloat16)
            if autocast_enabled
            else contextlib.nullcontext()
        )
        with amp_ctx:
            loss = self._regression_loss(self.decoder(self.encoder(obs_history)), target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        self.optimizer.step()
        return float(loss.item())
