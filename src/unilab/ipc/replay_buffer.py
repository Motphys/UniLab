"""Minimal zero-copy replay buffer for off-policy RL."""

from typing import Dict

import torch

from unilab.algos.torch.common.shared_buffer import SharedBufferBase


class ReplayBuffer(SharedBufferBase):
    """Shared tensor replay buffer with device-adaptive sampling.

    Design:
    - CUDA: 6 separate contiguous host tensors for efficient strided H2D sync;
            lazy sync to GPU cache on sample() call
    - MPS/CPU: single packed host tensor; 1 H2D Metal command per sample()
    """

    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        action_dim: int,
        device: str,
        privileged_dim: int = 0,
        defer_gpu: bool = False,
    ):
        super().__init__(capacity, device, defer_gpu=defer_gpu)
        self._obs_dim = obs_dim
        self._action_dim = action_dim
        self._privileged_dim = privileged_dim

        self.size = torch.zeros(1, dtype=torch.int64).share_memory_()

        if device == "cuda":
            # 6 separate contiguous shared tensors — keeps H2D copy_ on contiguous memory
            self.obs = torch.zeros(capacity, obs_dim).share_memory_()
            self.next_obs = torch.zeros(capacity, obs_dim).share_memory_()
            self.actions = torch.zeros(capacity, action_dim).share_memory_()
            self.rewards = torch.zeros(capacity).share_memory_()
            self.dones = torch.zeros(capacity).share_memory_()
            self.truncated = torch.zeros(capacity).share_memory_()

            if privileged_dim > 0:
                self.privileged_obs = torch.zeros(capacity, privileged_dim).share_memory_()
                self.next_privileged_obs = torch.zeros(capacity, privileged_dim).share_memory_()

            if not defer_gpu:
                # GPU cache for zero-copy sampling
                self.obs_gpu = torch.empty(capacity, obs_dim, device="cuda")
                self.next_obs_gpu = torch.empty(capacity, obs_dim, device="cuda")
                self.actions_gpu = torch.empty(capacity, action_dim, device="cuda")
                self.rewards_gpu = torch.empty(capacity, device="cuda")
                self.dones_gpu = torch.empty(capacity, device="cuda")
                self.truncated_gpu = torch.empty(capacity, device="cuda")

                if privileged_dim > 0:
                    self.privileged_obs_gpu = torch.empty(capacity, privileged_dim, device="cuda")
                    self.next_privileged_obs_gpu = torch.empty(
                        capacity, privileged_dim, device="cuda"
                    )
        else:
            # Single packed host tensor; layout: [obs | next_obs | actions | rew | done | trunc | priv | next_priv]
            total_dim = 2 * obs_dim + action_dim + 3 + 2 * privileged_dim
            self._storage = torch.zeros(capacity, total_dim).share_memory_()

            c = 0
            self._obs_sl = slice(c, c + obs_dim)
            c += obs_dim
            self._nobs_sl = slice(c, c + obs_dim)
            c += obs_dim
            self._act_sl = slice(c, c + action_dim)
            c += action_dim
            self._rew_col = c
            c += 1
            self._done_col = c
            c += 1
            self._trunc_col = c
            c += 1

            if privileged_dim > 0:
                self._priv_sl = slice(c, c + privileged_dim)
                c += privileged_dim
                self._npriv_sl = slice(c, c + privileged_dim)

    def __getstate__(self) -> dict:
        """Custom pickle support.

        The collector subprocess only calls add(), which writes to the CPU
        shared-memory tensors (self.obs, self.actions, …).  It never calls
        sample(), so it doesn't need the GPU cache or the CUDA stream.
        Neither torch.cuda.Stream nor CUDA tensors are picklable, so we strip
        them here.  The original object in the learner process is unaffected.
        """
        state = self.__dict__.copy()
        state["_cuda_stream"] = None
        for key in (
            "obs_gpu",
            "next_obs_gpu",
            "actions_gpu",
            "rewards_gpu",
            "dones_gpu",
            "truncated_gpu",
            "privileged_obs_gpu",
            "next_privileged_obs_gpu",
        ):
            state.pop(key, None)
        return state

    def init_local_gpu_cache(self, device: str) -> None:
        """Per-process GPU cache initialisation for multi-GPU training.

        Each Learner process calls this once after being spawned, binding
        its own GPU cache to ``device``.  The shared host tensors (obs,
        actions, …) already exist; only the GPU-side copies are created here.
        """
        assert self.device == "cuda", "init_local_gpu_cache is only for CUDA buffers"
        obs_dim = self.obs.shape[1]
        act_dim = self.actions.shape[1]
        self.obs_gpu = torch.zeros(self.capacity, obs_dim, device=device)
        self.next_obs_gpu = torch.zeros(self.capacity, obs_dim, device=device)
        self.actions_gpu = torch.zeros(self.capacity, act_dim, device=device)
        self.rewards_gpu = torch.zeros(self.capacity, device=device)
        self.dones_gpu = torch.zeros(self.capacity, device=device)
        self.truncated_gpu = torch.zeros(self.capacity, device=device)
        if self._privileged_dim > 0:
            self.privileged_obs_gpu = torch.zeros(
                self.capacity, self._privileged_dim, device=device
            )
            self.next_privileged_obs_gpu = torch.zeros(
                self.capacity, self._privileged_dim, device=device
            )
        self._gpu_synced_ptr = 0
        self._cuda_stream = torch.cuda.Stream(device=device)

    def add(
        self,
        obs,
        actions,
        rewards,
        next_obs,
        dones,
        truncated,
        privileged=None,
        next_privileged=None,
    ):
        """Add batch (called by collector)."""
        n = obs.shape[0]
        idx = int(self.ptr[0]) % self.capacity

        if self.device == "cuda":
            if idx + n <= self.capacity:
                self.obs[idx : idx + n] = obs
                self.next_obs[idx : idx + n] = next_obs
                self.actions[idx : idx + n] = actions
                self.rewards[idx : idx + n] = rewards
                self.dones[idx : idx + n] = dones
                self.truncated[idx : idx + n] = truncated
                if self._privileged_dim > 0 and privileged is not None:
                    assert next_privileged is not None
                    self.privileged_obs[idx : idx + n] = privileged
                    self.next_privileged_obs[idx : idx + n] = next_privileged
            else:
                split = self.capacity - idx
                self.obs[idx:] = obs[:split]
                self.obs[: n - split] = obs[split:]
                self.next_obs[idx:] = next_obs[:split]
                self.next_obs[: n - split] = next_obs[split:]
                self.actions[idx:] = actions[:split]
                self.actions[: n - split] = actions[split:]
                self.rewards[idx:] = rewards[:split]
                self.rewards[: n - split] = rewards[split:]
                self.dones[idx:] = dones[:split]
                self.dones[: n - split] = dones[split:]
                self.truncated[idx:] = truncated[:split]
                self.truncated[: n - split] = truncated[split:]
                if self._privileged_dim > 0 and privileged is not None:
                    assert next_privileged is not None
                    self.privileged_obs[idx:] = privileged[:split]
                    self.privileged_obs[: n - split] = privileged[split:]
                    self.next_privileged_obs[idx:] = next_privileged[:split]
                    self.next_privileged_obs[: n - split] = next_privileged[split:]
        else:
            parts = [
                obs,
                next_obs,
                actions,
                rewards.unsqueeze(1),
                dones.unsqueeze(1),
                truncated.unsqueeze(1),
            ]
            if self._privileged_dim > 0 and privileged is not None:
                assert next_privileged is not None
                parts.extend([privileged, next_privileged])
            row = torch.cat(parts, dim=1)

            if idx + n <= self.capacity:
                self._storage[idx : idx + n] = row
            else:
                split = self.capacity - idx
                self._storage[idx:] = row[:split]
                self._storage[: n - split] = row[split:]

        self.ptr[0] += n
        self.size[0] = min(int(self.size[0]) + n, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample batch (called by learner)."""
        size = int(self.size[0])
        indices = torch.randint(0, size, (batch_size,))

        if self.device == "cuda":
            # Lazy sync: only sync new data since last sample
            ptr = int(self.ptr[0])
            if ptr > self._gpu_synced_ptr:
                delta = ptr - self._gpu_synced_ptr
                idx = self._gpu_synced_ptr % self.capacity

                if idx + delta <= self.capacity:
                    self.obs_gpu[idx : idx + delta].copy_(
                        self.obs[idx : idx + delta], non_blocking=True
                    )
                    self.next_obs_gpu[idx : idx + delta].copy_(
                        self.next_obs[idx : idx + delta], non_blocking=True
                    )
                    self.actions_gpu[idx : idx + delta].copy_(
                        self.actions[idx : idx + delta], non_blocking=True
                    )
                    self.rewards_gpu[idx : idx + delta].copy_(
                        self.rewards[idx : idx + delta], non_blocking=True
                    )
                    self.dones_gpu[idx : idx + delta].copy_(
                        self.dones[idx : idx + delta], non_blocking=True
                    )
                    self.truncated_gpu[idx : idx + delta].copy_(
                        self.truncated[idx : idx + delta], non_blocking=True
                    )
                    if self._privileged_dim > 0:
                        self.privileged_obs_gpu[idx : idx + delta].copy_(
                            self.privileged_obs[idx : idx + delta], non_blocking=True
                        )
                        self.next_privileged_obs_gpu[idx : idx + delta].copy_(
                            self.next_privileged_obs[idx : idx + delta], non_blocking=True
                        )
                else:
                    split = self.capacity - idx
                    self.obs_gpu[idx:].copy_(self.obs[idx:], non_blocking=True)
                    self.obs_gpu[: delta - split].copy_(
                        self.obs[: delta - split], non_blocking=True
                    )
                    self.next_obs_gpu[idx:].copy_(self.next_obs[idx:], non_blocking=True)
                    self.next_obs_gpu[: delta - split].copy_(
                        self.next_obs[: delta - split], non_blocking=True
                    )
                    self.actions_gpu[idx:].copy_(self.actions[idx:], non_blocking=True)
                    self.actions_gpu[: delta - split].copy_(
                        self.actions[: delta - split], non_blocking=True
                    )
                    self.rewards_gpu[idx:].copy_(self.rewards[idx:], non_blocking=True)
                    self.rewards_gpu[: delta - split].copy_(
                        self.rewards[: delta - split], non_blocking=True
                    )
                    self.dones_gpu[idx:].copy_(self.dones[idx:], non_blocking=True)
                    self.dones_gpu[: delta - split].copy_(
                        self.dones[: delta - split], non_blocking=True
                    )
                    self.truncated_gpu[idx:].copy_(self.truncated[idx:], non_blocking=True)
                    self.truncated_gpu[: delta - split].copy_(
                        self.truncated[: delta - split], non_blocking=True
                    )
                    if self._privileged_dim > 0:
                        self.privileged_obs_gpu[idx:].copy_(
                            self.privileged_obs[idx:], non_blocking=True
                        )
                        self.privileged_obs_gpu[: delta - split].copy_(
                            self.privileged_obs[: delta - split], non_blocking=True
                        )
                        self.next_privileged_obs_gpu[idx:].copy_(
                            self.next_privileged_obs[idx:], non_blocking=True
                        )
                        self.next_privileged_obs_gpu[: delta - split].copy_(
                            self.next_privileged_obs[: delta - split], non_blocking=True
                        )

                self._gpu_synced_ptr = ptr

            indices = indices.to(self.obs_gpu.device)
            batch = {
                "obs": self.obs_gpu[indices],
                "actions": self.actions_gpu[indices],
                "rewards": self.rewards_gpu[indices],
                "next_obs": self.next_obs_gpu[indices],
                "dones": self.dones_gpu[indices],
                "truncated": self.truncated_gpu[indices],
            }
            if self._privileged_dim > 0:
                batch["privileged"] = self.privileged_obs_gpu[indices]
                batch["next_privileged"] = self.next_privileged_obs_gpu[indices]
            return batch
        else:
            chunk = self._storage[indices].to(self.device)
            batch = {
                "obs": chunk[:, self._obs_sl],
                "next_obs": chunk[:, self._nobs_sl],
                "actions": chunk[:, self._act_sl],
                "rewards": chunk[:, self._rew_col],
                "dones": chunk[:, self._done_col],
                "truncated": chunk[:, self._trunc_col],
            }
            if self._privileged_dim > 0:
                batch["privileged"] = chunk[:, self._priv_sl]
                batch["next_privileged"] = chunk[:, self._npriv_sl]
            return batch
