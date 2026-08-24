# SPDX-License-Identifier: BSD-3-Clause
"""Rollout storage for CSE-PPO."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import torch


class CSERolloutStorage:
    class Transition:
        observations: torch.Tensor | None
        critic_observations: torch.Tensor | None
        actions: torch.Tensor | None
        rewards: torch.Tensor | None
        dones: torch.Tensor | None
        values: torch.Tensor | None
        actions_log_prob: torch.Tensor | None
        action_mean: torch.Tensor | None
        action_sigma: torch.Tensor | None

        def __init__(self) -> None:
            self.observations = self.critic_observations = None
            self.actions = self.rewards = self.dones = self.values = None
            self.actions_log_prob = self.action_mean = self.action_sigma = None

        def clear(self) -> None:
            self.observations = None
            self.critic_observations = None
            self.actions = None
            self.rewards = None
            self.dones = None
            self.values = None
            self.actions_log_prob = None
            self.action_mean = None
            self.action_sigma = None

    def __init__(
        self,
        num_envs: int,
        num_transitions_per_env: int,
        obs_shape: Sequence[int],
        privileged_obs_shape: Sequence[int | None],
        actions_shape: Sequence[int],
        device: str = "cpu",
    ) -> None:
        self.device = device
        self.num_transitions_per_env = int(num_transitions_per_env)
        self.num_envs = int(num_envs)
        self.obs_shape = tuple(obs_shape)
        self.privileged_obs_shape = tuple(privileged_obs_shape)
        self.actions_shape = tuple(actions_shape)
        self.step = 0
        self.observations = torch.zeros(
            self.num_transitions_per_env, self.num_envs, *self.obs_shape, device=device
        )
        self.privileged_observations = None
        if self.privileged_obs_shape and self.privileged_obs_shape[0] is not None:
            if any(dim is None for dim in self.privileged_obs_shape):
                raise ValueError("privileged_obs_shape cannot contain None values")
            privileged_shape = cast(tuple[int, ...], self.privileged_obs_shape)
            self.privileged_observations = torch.zeros(
                self.num_transitions_per_env,
                self.num_envs,
                *privileged_shape,
                device=device,
            )
        self.rewards = torch.zeros(self.num_transitions_per_env, self.num_envs, 1, device=device)
        self.actions = torch.zeros(
            self.num_transitions_per_env, self.num_envs, *self.actions_shape, device=device
        )
        self.dones = torch.zeros(
            self.num_transitions_per_env, self.num_envs, 1, dtype=torch.bool, device=device
        )
        self.actions_log_prob = torch.zeros_like(self.rewards)
        self.values = torch.zeros_like(self.rewards)
        self.returns = torch.zeros_like(self.rewards)
        self.advantages = torch.zeros_like(self.rewards)
        self.mu = torch.zeros_like(self.actions)
        self.sigma = torch.zeros_like(self.actions)

    def add_transition(self, transition: Transition) -> None:
        if self.step >= self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow")
        required = (
            "observations",
            "actions",
            "rewards",
            "dones",
            "values",
            "actions_log_prob",
            "action_mean",
            "action_sigma",
        )
        if any(getattr(transition, name) is None for name in required):
            raise ValueError("incomplete CSE-PPO transition")
        observations = transition.observations
        actions = transition.actions
        rewards = transition.rewards
        dones = transition.dones
        values = transition.values
        actions_log_prob = transition.actions_log_prob
        action_mean = transition.action_mean
        action_sigma = transition.action_sigma
        assert (
            observations is not None
            and actions is not None
            and rewards is not None
            and dones is not None
            and values is not None
            and actions_log_prob is not None
            and action_mean is not None
            and action_sigma is not None
        )
        self.observations[self.step].copy_(observations)
        if self.privileged_observations is not None:
            critic_observations = transition.critic_observations
            if critic_observations is None:
                raise ValueError("transition.critic_observations is required")
            self.privileged_observations[self.step].copy_(critic_observations)
        self.actions[self.step].copy_(actions)
        self.rewards[self.step].copy_(rewards.view(-1, 1))
        self.dones[self.step].copy_(dones.view(-1, 1).bool())
        self.values[self.step].copy_(values)
        self.actions_log_prob[self.step].copy_(actions_log_prob.view(-1, 1))
        self.mu[self.step].copy_(action_mean)
        self.sigma[self.step].copy_(action_sigma)
        self.step += 1

    def clear(self) -> None:
        self.step = 0

    def compute_returns(self, last_values: torch.Tensor, gamma: float, lam: float) -> None:
        advantage = torch.zeros_like(last_values)
        for step in reversed(range(self.num_transitions_per_env)):
            next_values = (
                last_values if step == self.num_transitions_per_env - 1 else self.values[step + 1]
            )
            not_terminal = 1.0 - self.dones[step].float()
            delta = self.rewards[step] + not_terminal * gamma * next_values - self.values[step]
            advantage = delta + not_terminal * gamma * lam * advantage
            self.returns[step] = advantage + self.values[step]
        self.advantages = self.returns - self.values
        self.advantages = (self.advantages - self.advantages.mean()) / (
            self.advantages.std() + 1e-8
        )

    def mini_batch_generator(self, num_mini_batches: int, num_epochs: int = 8):
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // int(num_mini_batches)
        if mini_batch_size <= 0:
            raise ValueError("num_mini_batches is too large for the rollout batch")
        indices = torch.randperm(int(num_mini_batches) * mini_batch_size, device=self.device)
        arrays = [
            self.observations.flatten(0, 1),
            self.privileged_observations.flatten(0, 1)
            if self.privileged_observations is not None
            else self.observations.flatten(0, 1),
            self.actions.flatten(0, 1),
            self.values.flatten(0, 1),
            self.advantages.flatten(0, 1),
            self.returns.flatten(0, 1),
            self.actions_log_prob.flatten(0, 1),
            self.mu.flatten(0, 1),
            self.sigma.flatten(0, 1),
        ]
        for _ in range(int(num_epochs)):
            for i in range(int(num_mini_batches)):
                idx = indices[i * mini_batch_size : (i + 1) * mini_batch_size]
                yield tuple(array[idx] for array in arrays)
