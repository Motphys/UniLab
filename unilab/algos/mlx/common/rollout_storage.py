"""Rollout buffer for on-policy algorithms."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Generator, Union

import mlx.core as mx


@dataclass
class RolloutBuffer:
    """On-policy rollout storage for vectorized environments."""

    num_steps: int
    num_envs: int
    obs_dim: int
    action_dim: int
    gamma: float
    lam: float
    dtype: mx.Dtype = mx.float32

    def __post_init__(self) -> None:
        self.observations: Union[list[mx.array], mx.array] = []
        self.actions: Union[list[mx.array], mx.array] = []
        self.log_probs: Union[list[mx.array], mx.array] = []
        self.mu: Union[list[mx.array], mx.array] = []
        self.sigma: Union[list[mx.array], mx.array] = []
        self.rewards: list[mx.array] = []
        self.dones: list[mx.array] = []
        self.values: Union[list[mx.array], mx.array] = []
        self.advantages = mx.zeros((self.num_steps, self.num_envs), dtype=self.dtype)
        self.returns = mx.zeros((self.num_steps, self.num_envs), dtype=self.dtype)
        self.step = 0

    def add(
        self,
        obs: mx.array,
        actions: mx.array,
        log_probs: mx.array,
        action_mean: mx.array,
        action_std: mx.array,
        rewards: mx.array,
        dones: mx.array,
        values: mx.array,
    ) -> None:
        if self.step >= self.num_steps:
            raise OverflowError("Rollout buffer overflow.")
        self.observations.append(obs)  # type: ignore[union-attr]
        self.actions.append(actions)  # type: ignore[union-attr]
        self.log_probs.append(log_probs)  # type: ignore[union-attr]
        self.mu.append(action_mean)  # type: ignore[union-attr]
        self.sigma.append(action_std)  # type: ignore[union-attr]
        self.rewards.append(rewards)
        self.dones.append(dones)
        self.values.append(values)  # type: ignore[union-attr]
        self.step += 1

    def compute_returns_and_advantages(self, last_values: mx.array) -> None:
        rewards = self.rewards
        dones = self.dones
        values = self.values
        gae = mx.zeros((self.num_envs,), dtype=self.dtype)
        advantages: list[mx.array] = [
            mx.zeros((self.num_envs,), dtype=self.dtype) for _ in range(self.num_steps)
        ]
        returns: list[mx.array] = [
            mx.zeros((self.num_envs,), dtype=self.dtype) for _ in range(self.num_steps)
        ]
        for t in reversed(range(self.num_steps)):
            if t == self.num_steps - 1:
                next_values = last_values
            else:
                next_values = values[t + 1]
            next_non_terminal = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_values * next_non_terminal - values[t]
            gae = delta + self.gamma * self.lam * next_non_terminal * gae
            advantages[t] = gae
            returns[t] = gae + values[t]

        self.observations = mx.stack(self.observations, axis=0)  # type: ignore[arg-type]
        self.actions = mx.stack(self.actions, axis=0)  # type: ignore[arg-type]
        self.log_probs = mx.stack(self.log_probs, axis=0)  # type: ignore[arg-type]
        self.mu = mx.stack(self.mu, axis=0)  # type: ignore[arg-type]
        self.sigma = mx.stack(self.sigma, axis=0)  # type: ignore[arg-type]
        self.values = mx.stack(self.values, axis=0)  # type: ignore[arg-type]
        self.advantages = mx.stack(advantages, axis=0)
        self.returns = mx.stack(returns, axis=0)
        adv = mx.reshape(self.advantages, (-1,))
        self.advantages = (self.advantages - mx.mean(adv)) / (mx.std(adv) + 1e-8)

    def mini_batch_generator(
        self, num_mini_batches: int, num_epochs: int
    ) -> Generator[Dict[str, mx.array], None, None]:
        batch_size = self.num_steps * self.num_envs
        mini_batch_size = batch_size // num_mini_batches

        obs = mx.reshape(self.observations, (batch_size, self.obs_dim))  # type: ignore[arg-type]
        actions = mx.reshape(self.actions, (batch_size, self.action_dim))  # type: ignore[arg-type]
        log_probs = mx.reshape(self.log_probs, (batch_size,))  # type: ignore[arg-type]
        mu = mx.reshape(self.mu, (batch_size, self.action_dim))  # type: ignore[arg-type]
        sigma = mx.reshape(self.sigma, (batch_size, self.action_dim))  # type: ignore[arg-type]
        returns = mx.reshape(self.returns, (batch_size,))
        advantages = mx.reshape(self.advantages, (batch_size,))
        values = mx.reshape(self.values, (batch_size,))  # type: ignore[arg-type]

        for _ in range(num_epochs):
            shuffled = mx.random.permutation(batch_size)
            # Build all mini-batch indices once per epoch to reduce Python overhead.
            batch_indices = mx.reshape(
                shuffled.astype(mx.int32),
                (num_mini_batches, mini_batch_size),
            )
            for i in range(num_mini_batches):
                idx = batch_indices[i]
                yield {
                    "obs": obs[idx],
                    "actions": actions[idx],
                    "old_log_probs": log_probs[idx],
                    "old_mu": mu[idx],
                    "old_sigma": sigma[idx],
                    "returns": returns[idx],
                    "advantages": advantages[idx],
                    "old_values": values[idx],
                }

    def clear(self) -> None:
        self.observations = []
        self.actions = []
        self.log_probs = []
        self.mu = []
        self.sigma = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.advantages = mx.zeros((self.num_steps, self.num_envs), dtype=self.dtype)
        self.returns = mx.zeros((self.num_steps, self.num_envs), dtype=self.dtype)
        self.step = 0
