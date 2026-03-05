"""Replay Buffer for off-policy RL algorithms (TD3, SAC).

Simple ring-buffer implementation that stores transitions as flat tensors
and supports batch insertion from multiple collector workers + uniform random sampling.
"""

import torch


class ReplayBuffer:
    """Fixed-size ring buffer for (obs, action, reward, next_obs, done) tuples."""

    def __init__(self, capacity: int, obs_dim: int, action_dim: int, device: str = "cpu"):
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0

        # Pre-allocate storage on target device
        self.obs = torch.zeros((capacity, obs_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros((capacity, action_dim), dtype=torch.float32, device=device)
        self.rewards = torch.zeros((capacity,), dtype=torch.float32, device=device)
        self.next_obs = torch.zeros((capacity, obs_dim), dtype=torch.float32, device=device)
        self.dones = torch.zeros((capacity,), dtype=torch.float32, device=device)

    def add(self, obs, action, reward, next_obs, done):
        """Add a single transition."""
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_obs[self.ptr] = next_obs
        self.dones[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def add_batch(self, obs, actions, rewards, next_obs, dones):
        """Add a batch of transitions.

        Args:
            obs:      [B, obs_dim]
            actions:  [B, action_dim]
            rewards:  [B]
            next_obs: [B, obs_dim]
            dones:    [B]
        """
        batch_size = obs.shape[0]

        if batch_size == 0:
            return

        # Move to buffer device
        obs = obs.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_obs = next_obs.to(self.device)
        dones = dones.to(self.device)

        # Handle wrap-around
        if self.ptr + batch_size <= self.capacity:
            self.obs[self.ptr:self.ptr + batch_size] = obs
            self.actions[self.ptr:self.ptr + batch_size] = actions
            self.rewards[self.ptr:self.ptr + batch_size] = rewards
            self.next_obs[self.ptr:self.ptr + batch_size] = next_obs
            self.dones[self.ptr:self.ptr + batch_size] = dones
        else:
            # Split across boundary
            first = self.capacity - self.ptr
            self.obs[self.ptr:] = obs[:first]
            self.actions[self.ptr:] = actions[:first]
            self.rewards[self.ptr:] = rewards[:first]
            self.next_obs[self.ptr:] = next_obs[:first]
            self.dones[self.ptr:] = dones[:first]

            remainder = batch_size - first
            self.obs[:remainder] = obs[first:]
            self.actions[:remainder] = actions[first:]
            self.rewards[:remainder] = rewards[first:]
            self.next_obs[:remainder] = next_obs[first:]
            self.dones[:remainder] = dones[first:]

        self.ptr = (self.ptr + batch_size) % self.capacity
        self.size = min(self.size + batch_size, self.capacity)

    def sample(self, batch_size: int):
        """Uniformly sample a batch of transitions.

        Returns:
            tuple of (obs, actions, rewards, next_obs, dones), all on self.device
        """
        indices = torch.randint(0, self.size, (batch_size,), device=self.device)
        return (
            self.obs[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_obs[indices],
            self.dones[indices],
        )

    def __len__(self):
        return self.size
