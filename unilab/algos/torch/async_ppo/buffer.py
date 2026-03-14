"""On-policy replay buffer for async PPO."""

import torch


class OnPolicyReplayBuffer:
    """Stores recent rollouts with device-adaptive layout and zero-copy optimization."""

    def __init__(
        self,
        capacity_rollouts: int,
        num_envs: int,
        num_steps: int,
        obs_dim: int,
        action_dim: int,
        device: str,
    ):
        self.capacity = capacity_rollouts
        self.num_envs = num_envs
        self.num_steps = num_steps
        self.device = device
        self._obs_dim = obs_dim
        self._action_dim = action_dim

        self.ptr = torch.zeros(1, dtype=torch.int64).share_memory_()
        self.count = torch.zeros(1, dtype=torch.int64).share_memory_()

        if device == "cuda":
            # Use pinned memory for faster H2D transfer
            self.obs = torch.zeros(
                capacity_rollouts, num_steps, num_envs, obs_dim, pin_memory=True
            ).share_memory_()
            self.actions = torch.zeros(
                capacity_rollouts, num_steps, num_envs, action_dim, pin_memory=True
            ).share_memory_()
            self.rewards = torch.zeros(
                capacity_rollouts, num_steps, num_envs, pin_memory=True
            ).share_memory_()
            self.dones = torch.zeros(
                capacity_rollouts, num_steps, num_envs, pin_memory=True
            ).share_memory_()
            self.log_probs = torch.zeros(
                capacity_rollouts, num_steps, num_envs, pin_memory=True
            ).share_memory_()
            self.values = torch.zeros(
                capacity_rollouts, num_steps, num_envs, pin_memory=True
            ).share_memory_()
            self.last_obs = torch.zeros(
                capacity_rollouts, num_envs, obs_dim, pin_memory=True
            ).share_memory_()

            # GPU cache
            self.obs_gpu = torch.empty_like(self.obs, device="cuda")
            self.actions_gpu = torch.empty_like(self.actions, device="cuda")
            self.rewards_gpu = torch.empty_like(self.rewards, device="cuda")
            self.dones_gpu = torch.empty_like(self.dones, device="cuda")
            self.log_probs_gpu = torch.empty_like(self.log_probs, device="cuda")
            self.values_gpu = torch.empty_like(self.values, device="cuda")
            self.last_obs_gpu = torch.empty_like(self.last_obs, device="cuda")
            self._synced_ptr = 0
            self._cuda_stream = torch.cuda.Stream()
        else:
            # Packed layout for MPS/CPU
            total_dim = num_steps * num_envs * (obs_dim + action_dim + 4) + num_envs * obs_dim
            self._storage = torch.zeros(capacity_rollouts, total_dim).share_memory_()

    def add_rollout(self, rollout: dict) -> None:
        """Add complete rollout."""
        idx = int(self.ptr[0]) % self.capacity

        if self.device == "cuda":
            self.obs[idx] = rollout["observations"]
            self.actions[idx] = rollout["actions"]
            self.rewards[idx] = rollout["rewards"]
            self.dones[idx] = rollout["dones"]
            self.log_probs[idx] = rollout["log_probs"]
            self.values[idx] = rollout["values"]
            self.last_obs[idx] = rollout["last_obs"]
        else:
            # Pack into single tensor
            flat = torch.cat(
                [
                    rollout["observations"].flatten(),
                    rollout["actions"].flatten(),
                    rollout["rewards"].flatten(),
                    rollout["dones"].flatten(),
                    rollout["log_probs"].flatten(),
                    rollout["values"].flatten(),
                    rollout["last_obs"].flatten(),
                ]
            )
            self._storage[idx] = flat

        self.ptr[0] += 1
        self.count[0] = min(int(self.count[0]) + 1, self.capacity)
        if int(self.ptr[0]) % 10 == 0:
            print(f"[Buffer] Added rollout {int(self.ptr[0])}, count={int(self.count[0])}")

    def get_latest(self) -> dict:
        """Get most recent rollout with async H2D transfer."""
        idx = (int(self.ptr[0]) - 1) % self.capacity

        if self.device == "cuda":
            # Async H2D transfer with dedicated stream
            if int(self.ptr[0]) > self._synced_ptr:
                with torch.cuda.stream(self._cuda_stream):
                    self.obs_gpu[idx].copy_(self.obs[idx], non_blocking=True)
                    self.actions_gpu[idx].copy_(self.actions[idx], non_blocking=True)
                    self.rewards_gpu[idx].copy_(self.rewards[idx], non_blocking=True)
                    self.dones_gpu[idx].copy_(self.dones[idx], non_blocking=True)
                    self.log_probs_gpu[idx].copy_(self.log_probs[idx], non_blocking=True)
                    self.values_gpu[idx].copy_(self.values[idx], non_blocking=True)
                    self.last_obs_gpu[idx].copy_(self.last_obs[idx], non_blocking=True)
                self._synced_ptr = int(self.ptr[0])

            # Sync stream before returning
            self._cuda_stream.synchronize()

            return {
                "observations": self.obs_gpu[idx],
                "actions": self.actions_gpu[idx],
                "rewards": self.rewards_gpu[idx],
                "dones": self.dones_gpu[idx],
                "log_probs": self.log_probs_gpu[idx],
                "values": self.values_gpu[idx],
                "last_obs": self.last_obs_gpu[idx],
            }
        else:
            # Unpack from storage
            flat = self._storage[idx].to(self.device)
            c = 0
            obs_size = self.num_steps * self.num_envs * self._obs_dim
            act_size = self.num_steps * self.num_envs * self._action_dim
            scalar_size = self.num_steps * self.num_envs
            last_obs_size = self.num_envs * self._obs_dim

            obs = flat[c : c + obs_size].view(self.num_steps, self.num_envs, self._obs_dim)
            c += obs_size
            actions = flat[c : c + act_size].view(self.num_steps, self.num_envs, self._action_dim)
            c += act_size
            rewards = flat[c : c + scalar_size].view(self.num_steps, self.num_envs)
            c += scalar_size
            dones = flat[c : c + scalar_size].view(self.num_steps, self.num_envs)
            c += scalar_size
            log_probs = flat[c : c + scalar_size].view(self.num_steps, self.num_envs)
            c += scalar_size
            values = flat[c : c + scalar_size].view(self.num_steps, self.num_envs)
            c += scalar_size
            last_obs = flat[c : c + last_obs_size].view(self.num_envs, self._obs_dim)

            return {
                "observations": obs,
                "actions": actions,
                "rewards": rewards,
                "dones": dones,
                "log_probs": log_probs,
                "values": values,
                "last_obs": last_obs,
            }

    def is_ready(self) -> bool:
        """Check if data available."""
        return int(self.count[0]) > 0
