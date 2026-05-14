# SPDX-License-Identifier: BSD-3-Clause
#
# Adapted from the HIMLoco RSL-RL HIM estimator for UniLab.

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def get_activation(name: str) -> nn.Module:
    if name == "elu":
        return nn.ELU()
    if name == "selu":
        return nn.SELU()
    if name == "relu":
        return nn.ReLU()
    if name == "crelu":
        return nn.ReLU()
    if name == "silu":
        return nn.SiLU()
    if name == "lrelu":
        return nn.LeakyReLU()
    if name == "tanh":
        return nn.Tanh()
    if name == "sigmoid":
        return nn.Sigmoid()
    raise ValueError(f"Unsupported activation: {name}")


class HIMEstimator(nn.Module):
    def __init__(
        self,
        temporal_steps: int,
        num_one_step_obs: int,
        enc_hidden_dims: list[int] | tuple[int, ...] = (128, 64, 16),
        tar_hidden_dims: list[int] | tuple[int, ...] = (128, 64),
        activation: str = "elu",
        learning_rate: float = 1e-3,
        max_grad_norm: float = 10.0,
        num_prototype: int = 32,
        temperature: float = 3.0,
        velocity_target_start: int | None = None,
        target_obs_start: int = 3,
    ) -> None:
        super().__init__()
        if temporal_steps <= 0:
            raise ValueError("temporal_steps must be positive")
        if num_one_step_obs <= 0:
            raise ValueError("num_one_step_obs must be positive")
        if len(enc_hidden_dims) == 0:
            raise ValueError("enc_hidden_dims must not be empty")

        self.temporal_steps = int(temporal_steps)
        self.num_one_step_obs = int(num_one_step_obs)
        self.num_latent = int(enc_hidden_dims[-1])
        self.max_grad_norm = float(max_grad_norm)
        self.temperature = float(temperature)
        self.velocity_target_start = (
            int(num_one_step_obs) if velocity_target_start is None else int(velocity_target_start)
        )
        self.target_obs_start = int(target_obs_start)

        enc_input_dim = self.temporal_steps * self.num_one_step_obs
        enc_layers: list[nn.Module] = []
        for hidden_dim in enc_hidden_dims[:-1]:
            enc_layers += [nn.Linear(enc_input_dim, int(hidden_dim)), get_activation(activation)]
            enc_input_dim = int(hidden_dim)
        enc_layers += [nn.Linear(enc_input_dim, self.num_latent + 3)]
        self.encoder = nn.Sequential(*enc_layers)

        tar_input_dim = self.num_one_step_obs
        tar_layers: list[nn.Module] = []
        for hidden_dim in tar_hidden_dims:
            tar_layers += [nn.Linear(tar_input_dim, int(hidden_dim)), get_activation(activation)]
            tar_input_dim = int(hidden_dim)
        tar_layers += [nn.Linear(tar_input_dim, self.num_latent)]
        self.target = nn.Sequential(*tar_layers)

        self.proto = nn.Embedding(int(num_prototype), self.num_latent)
        self.learning_rate = float(learning_rate)
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

    def get_latent(self, obs_history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        vel, z = self.encode(obs_history)
        return vel.detach(), z.detach()

    def forward(self, obs_history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        vel, z = self.encode(obs_history.detach())
        return vel.detach(), z.detach()

    def encode(self, obs_history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        parts = self.encoder(obs_history.detach())
        vel, z = parts[..., :3], parts[..., 3:]
        z = F.normalize(z, dim=-1, p=2)
        return vel, z

    def update(
        self,
        obs_history: torch.Tensor,
        next_critic_obs: torch.Tensor,
        lr: float | None = None,
    ) -> tuple[float, float]:
        if lr is not None:
            self.learning_rate = float(lr)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.learning_rate

        vel_start = self.velocity_target_start
        vel_end = vel_start + 3
        target_start = self.target_obs_start
        target_end = target_start + self.num_one_step_obs
        if next_critic_obs.shape[-1] < max(vel_end, target_end):
            raise ValueError(
                "next_critic_obs is too small for HIM estimator slices: "
                f"shape={tuple(next_critic_obs.shape)}, velocity=[{vel_start}:{vel_end}], "
                f"target=[{target_start}:{target_end}]"
            )

        vel = next_critic_obs[:, vel_start:vel_end].detach()
        next_obs = next_critic_obs[:, target_start:target_end].detach()

        parts = self.encoder(obs_history)
        pred_vel, z_s = parts[..., :3], parts[..., 3:]
        z_t = self.target(next_obs)

        z_s = F.normalize(z_s, dim=-1, p=2)
        z_t = F.normalize(z_t, dim=-1, p=2)

        with torch.no_grad():
            self.proto.weight.copy_(F.normalize(self.proto.weight.data.clone(), dim=-1, p=2))

        score_s = z_s @ self.proto.weight.T
        score_t = z_t @ self.proto.weight.T

        with torch.no_grad():
            q_s = sinkhorn(score_s)
            q_t = sinkhorn(score_t)

        log_p_s = F.log_softmax(score_s / self.temperature, dim=-1)
        log_p_t = F.log_softmax(score_t / self.temperature, dim=-1)

        swap_loss = -0.5 * (q_s * log_p_t + q_t * log_p_s).mean()
        estimation_loss = F.mse_loss(pred_vel, vel)
        loss = estimation_loss + swap_loss

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return float(estimation_loss.item()), float(swap_loss.item())


@torch.no_grad()
def sinkhorn(out: torch.Tensor, eps: float = 0.05, iters: int = 3) -> torch.Tensor:
    q = torch.exp(out / eps).T
    k, b = q.shape
    q /= q.sum()

    for _ in range(iters):
        q /= torch.sum(q, dim=1, keepdim=True)
        q /= k
        q /= torch.sum(q, dim=0, keepdim=True)
        q /= b
    return (q * b).T
