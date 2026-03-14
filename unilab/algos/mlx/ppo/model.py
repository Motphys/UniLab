"""Actor-Critic model for MLX PPO."""

from __future__ import annotations

import math
from typing import Sequence, Tuple

import mlx.core as mx
import mlx.nn as nn

from unilab.algos.mlx.common import EmpiricalNormalization, MLP, diag_gaussian_log_prob


class MLPActorCritic(nn.Module):
    """Shared utility class containing actor and critic MLPs."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        actor_hidden_dims: Sequence[int],
        critic_hidden_dims: Sequence[int],
        activation: str = "tanh",
        init_log_std: float = 0.0,
        min_log_std: float = -5.0,
        max_log_std: float = 2.0,
        obs_normalization: bool = False,
        noise_std_type: str = "log",
        state_dependent_std: bool = False,
        dtype=mx.float32,
    ) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.dtype = dtype
        self.noise_std_type = noise_std_type
        self.state_dependent_std = bool(state_dependent_std)
        self.obs_normalization = bool(obs_normalization)
        self.obs_normalizer = (
            EmpiricalNormalization(obs_dim, dtype=dtype) if self.obs_normalization else None
        )
        actor_output_dim = action_dim * 2 if self.state_dependent_std else action_dim
        self.actor = MLP(obs_dim, actor_output_dim, actor_hidden_dims, activation=activation)
        self.critic = MLP(obs_dim, 1, critic_hidden_dims, activation=activation)
        self.actor.init_orthogonal(hidden_gain=math.sqrt(2.0), output_gain=0.01)
        self.critic.init_orthogonal(hidden_gain=math.sqrt(2.0), output_gain=1.0)
        if self.state_dependent_std:
            # Keep std head conservative at init like rsl-rl.
            self.actor.layers[-1].weight[self.action_dim :] = 0.0
            if self.noise_std_type == "scalar":
                self.actor.layers[-1].bias[self.action_dim :] = float(
                    mx.exp(mx.array(init_log_std)).item()
                )
            elif self.noise_std_type == "log":
                self.actor.layers[-1].bias[self.action_dim :] = float(init_log_std)
            else:
                raise ValueError(f"Unknown noise_std_type: {self.noise_std_type}")
        else:
            if self.noise_std_type == "scalar":
                self.std = mx.full(
                    (action_dim,), float(mx.exp(mx.array(init_log_std)).item()), dtype=self.dtype
                )
            elif self.noise_std_type == "log":
                self.log_std = mx.full((action_dim,), float(init_log_std), dtype=self.dtype)
            else:
                raise ValueError(f"Unknown noise_std_type: {self.noise_std_type}")
        self.min_log_std = float(min_log_std)
        self.max_log_std = float(max_log_std)

    def clipped_log_std(self) -> mx.array:
        """Clamp log-std to avoid numerical explosion."""
        if self.noise_std_type == "log":
            return mx.clip(self.log_std, self.min_log_std, self.max_log_std)
        std = mx.maximum(self.std, 1e-4)
        log_std = mx.log(std)
        return mx.clip(log_std, self.min_log_std, self.max_log_std)

    def policy(self, obs: mx.array) -> mx.array:
        mean, _, _ = self.distribution_params(obs)
        return mean

    def distribution_params(self, obs: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        if self.obs_normalizer is not None:
            obs = self.obs_normalizer(obs)
        if self.state_dependent_std:
            out = self.actor(obs)
            mean = out[:, : self.action_dim]
            std_head = out[:, self.action_dim :]
            if self.noise_std_type == "scalar":
                std = mx.maximum(nn.softplus(std_head), 1e-4)
                log_std = mx.log(std)
            elif self.noise_std_type == "log":
                log_std = mx.clip(std_head, self.min_log_std, self.max_log_std)
                std = mx.maximum(mx.exp(log_std), 1e-4)
            else:
                raise ValueError(f"Unknown noise_std_type: {self.noise_std_type}")
        else:
            mean = self.actor(obs)
            if self.noise_std_type == "scalar":
                std_base = mx.maximum(self.std, 1e-4)
                std = mx.broadcast_to(std_base, mean.shape)
                log_std = mx.log(std)
            elif self.noise_std_type == "log":
                log_std_base = mx.clip(self.log_std, self.min_log_std, self.max_log_std)
                log_std = mx.broadcast_to(log_std_base, mean.shape)
                std = mx.maximum(mx.exp(log_std), 1e-4)
            else:
                raise ValueError(f"Unknown noise_std_type: {self.noise_std_type}")
        return mean, std, log_std

    def value(self, obs: mx.array) -> mx.array:
        if self.obs_normalizer is not None:
            obs = self.obs_normalizer(obs)
        return mx.squeeze(self.critic(obs), axis=-1)

    def update_normalization(self, obs: mx.array) -> None:
        if self.obs_normalizer is not None:
            self.obs_normalizer.update(obs)

    def act(self, obs: mx.array) -> Tuple[mx.array, mx.array, mx.array, mx.array]:
        """Sample actions and return MLX tensors."""
        mean, std, log_std = self.distribution_params(obs)
        noise = mx.random.normal(mean.shape)
        actions = mean + noise * std
        log_probs = diag_gaussian_log_prob(actions, mean, log_std)
        values = self.value(obs)
        return actions, log_probs, values, mean, std

    def current_action_std(self, action_shape: tuple[int, ...]) -> mx.array:
        """Return broadcasted std tensor for current policy."""
        if self.noise_std_type == "scalar":
            std = mx.maximum(self.std, 1e-4)
            return mx.broadcast_to(std, action_shape)
        log_std = self.clipped_log_std()
        return mx.broadcast_to(mx.maximum(mx.exp(log_std), 1e-4), action_shape)
