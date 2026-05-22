# SPDX-License-Identifier: BSD-3-Clause
#
# Adapted from the HIMLoco RSL-RL HIM rollout storage for UniLab.

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import torch


class HIMRolloutStorage:
    class Transition:
        def __init__(self) -> None:
            self.observations: torch.Tensor | None = None
            self.critic_observations: torch.Tensor | None = None
            self.next_critic_observations: torch.Tensor | None = None
            self.actions: torch.Tensor | None = None
            self.rewards: torch.Tensor | None = None
            self.dones: torch.Tensor | None = None
            self.values: torch.Tensor | None = None
            self.actions_log_prob: torch.Tensor | None = None
            self.action_mean: torch.Tensor | None = None
            self.action_sigma: torch.Tensor | None = None

        def clear(self) -> None:
            self.observations = None
            self.critic_observations = None
            self.next_critic_observations = None
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
        self.obs_shape = tuple(obs_shape)
        self.privileged_obs_shape = tuple(privileged_obs_shape)
        self.actions_shape = tuple(actions_shape)
        self.num_transitions_per_env = int(num_transitions_per_env)
        self.num_envs = int(num_envs)
        self.step = 0

        self.observations = torch.zeros(
            self.num_transitions_per_env,
            self.num_envs,
            *self.obs_shape,
            device=self.device,
        )
        if self.privileged_obs_shape and self.privileged_obs_shape[0] is not None:
            if any(dim is None for dim in self.privileged_obs_shape):
                raise ValueError("privileged_obs_shape cannot contain None values")
            privileged_obs_shape = cast(tuple[int, ...], self.privileged_obs_shape)
            self.privileged_observations = torch.zeros(
                self.num_transitions_per_env,
                self.num_envs,
                *privileged_obs_shape,
                device=self.device,
            )
            self.next_privileged_observations = torch.zeros_like(self.privileged_observations)
        else:
            self.privileged_observations = None
            self.next_privileged_observations = None

        self.rewards = torch.zeros(
            self.num_transitions_per_env, self.num_envs, 1, device=self.device
        )
        self.actions = torch.zeros(
            self.num_transitions_per_env,
            self.num_envs,
            *self.actions_shape,
            device=self.device,
        )
        self.dones = torch.zeros(
            self.num_transitions_per_env, self.num_envs, 1, device=self.device
        ).bool()
        self.actions_log_prob = torch.zeros_like(self.rewards)
        self.values = torch.zeros_like(self.rewards)
        self.returns = torch.zeros_like(self.rewards)
        self.advantages = torch.zeros_like(self.rewards)
        self.mu = torch.zeros_like(self.actions)
        self.sigma = torch.zeros_like(self.actions)

    def add_transition(self, transition: Transition) -> None:
        if self.step >= self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow")
        if transition.observations is None:
            raise ValueError("transition.observations is required")
        if transition.actions is None:
            raise ValueError("transition.actions is required")
        if transition.rewards is None:
            raise ValueError("transition.rewards is required")
        if transition.dones is None:
            raise ValueError("transition.dones is required")
        if transition.values is None:
            raise ValueError("transition.values is required")
        if transition.actions_log_prob is None:
            raise ValueError("transition.actions_log_prob is required")
        if transition.action_mean is None or transition.action_sigma is None:
            raise ValueError("transition action distribution stats are required")

        self.observations[self.step].copy_(transition.observations)
        if self.privileged_observations is not None:
            if transition.critic_observations is None:
                raise ValueError("transition.critic_observations is required")
            if transition.next_critic_observations is None:
                raise ValueError("transition.next_critic_observations is required")
            assert self.next_privileged_observations is not None
            self.privileged_observations[self.step].copy_(transition.critic_observations)
            self.next_privileged_observations[self.step].copy_(transition.next_critic_observations)
        self.actions[self.step].copy_(transition.actions)
        self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
        self.dones[self.step].copy_(transition.dones.view(-1, 1).bool())
        self.values[self.step].copy_(transition.values)
        self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))
        self.mu[self.step].copy_(transition.action_mean)
        self.sigma[self.step].copy_(transition.action_sigma)
        self.step += 1

    def add_transitions(self, transition: Transition) -> None:
        self.add_transition(transition)

    def clear(self) -> None:
        self.step = 0

    def compute_returns(self, last_values: torch.Tensor, gamma: float, lam: float) -> None:
        advantage = torch.zeros_like(last_values)
        for step in reversed(range(self.num_transitions_per_env)):
            if step == self.num_transitions_per_env - 1:
                next_values = last_values
            else:
                next_values = self.values[step + 1]
            next_is_not_terminal = 1.0 - self.dones[step].float()
            delta = (
                self.rewards[step] + next_is_not_terminal * gamma * next_values - self.values[step]
            )
            advantage = delta + next_is_not_terminal * gamma * lam * advantage
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
        indices = torch.randperm(
            int(num_mini_batches) * mini_batch_size,
            requires_grad=False,
            device=self.device,
        )

        observations = self.observations.flatten(0, 1)
        if self.privileged_observations is not None:
            assert self.next_privileged_observations is not None
            critic_observations = self.privileged_observations.flatten(0, 1)
            next_critic_observations = self.next_privileged_observations.flatten(0, 1)
        else:
            critic_observations = observations
            next_critic_observations = observations

        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        old_mu = self.mu.flatten(0, 1)
        old_sigma = self.sigma.flatten(0, 1)

        for _ in range(int(num_epochs)):
            for i in range(int(num_mini_batches)):
                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                batch_idx = indices[start:end]
                yield (
                    observations[batch_idx],
                    critic_observations[batch_idx],
                    actions[batch_idx],
                    next_critic_observations[batch_idx],
                    values[batch_idx],
                    advantages[batch_idx],
                    returns[batch_idx],
                    old_actions_log_prob[batch_idx],
                    old_mu[batch_idx],
                    old_sigma[batch_idx],
                )
