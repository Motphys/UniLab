"""Packed shared-memory replay buffer for off-policy RL."""

import time
from typing import Any, Dict

import torch

from unilab.ipc.shared_buffer import SharedBufferBase


class ReplayBuffer(SharedBufferBase):
    """Shared replay buffer backed by authoritative packed CPU storage.

    Device transfer is owned by replay pipeline transfer backends.  The
    fallback sample() path copies a sampled packed batch to ``self.device`` and
    keeps no per-device replay cache.
    """

    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        action_dim: int,
        device: str,
        defer_gpu: bool = False,
        critic_dim: int = 0,
        packed_cpu_storage: bool = False,
    ):
        super().__init__(capacity, device, defer_gpu=defer_gpu)
        del packed_cpu_storage
        self._obs_dim = obs_dim
        self._action_dim = action_dim
        self._critic_dim = critic_dim
        self.last_incremental_h2d_time_s = 0.0
        self._packed_cpu_storage = True
        self.trace_recorder: Any | None = None
        self.trace_thread_time = False
        self.trace_cuda_events = True

        self.size = torch.zeros(1, dtype=torch.int64).share_memory_()
        self._init_packed_storage(capacity, obs_dim, action_dim, critic_dim)

    def _init_packed_storage(
        self, capacity: int, obs_dim: int, action_dim: int, critic_dim: int
    ) -> None:
        total_dim = 2 * obs_dim + action_dim + 3 + 2 * critic_dim
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

        if critic_dim > 0:
            self._critic_sl = slice(c, c + critic_dim)
            c += critic_dim
            self._ncritic_sl = slice(c, c + critic_dim)
            c += critic_dim

    def __getstate__(self) -> dict:
        """Custom pickle support.

        The collector subprocess only calls add(), which writes to the CPU
        shared-memory tensor.  The original object in the learner process is
        unaffected.
        """
        state = self.__dict__.copy()
        state["trace_recorder"] = None
        return state

    def add(
        self,
        obs,
        actions,
        rewards,
        next_obs,
        dones,
        truncated,
        terminal_mask=None,
        terminal_next_obs=None,
        critic=None,
        next_critic=None,
        terminal_next_critic=None,
    ):
        """Add batch (called by collector).

        `dones` follows the UniLab env lifecycle contract:
        done = terminated | truncated. Learners must pair it with
        `truncated` when computing bootstrap masks.
        """
        _trace_ns = time.perf_counter_ns() if self.trace_recorder is not None else 0
        n = obs.shape[0]
        idx = int(self.ptr[0]) % self.capacity
        has_critic = self._critic_dim > 0 and critic is not None
        if self._critic_dim > 0 and (critic is None or next_critic is None):
            raise ValueError("ReplayBuffer with critic_dim > 0 requires critic and next_critic")

        parts = [
            obs,
            next_obs,
            actions,
            rewards.unsqueeze(1),
            dones.unsqueeze(1),
            truncated.unsqueeze(1),
        ]
        if has_critic:
            assert next_critic is not None
            parts.extend([critic, next_critic])
        row = torch.cat(parts, dim=1)

        if idx + n <= self.capacity:
            self._storage[idx : idx + n] = row
            self._patch_terminal_next_observations(
                self._storage[idx : idx + n, self._nobs_sl],
                terminal_mask,
                terminal_next_obs,
                self._storage[idx : idx + n, self._ncritic_sl] if has_critic else None,
                terminal_next_critic,
            )
        else:
            split = self.capacity - idx
            self._storage[idx:] = row[:split]
            self._storage[: n - split] = row[split:]
            self._patch_terminal_next_observations(
                self._storage[idx:, self._nobs_sl],
                terminal_mask[:split] if terminal_mask is not None else None,
                terminal_next_obs[:split] if terminal_next_obs is not None else None,
                self._storage[idx:, self._ncritic_sl] if has_critic else None,
                terminal_next_critic[:split] if terminal_next_critic is not None else None,
            )
            self._patch_terminal_next_observations(
                self._storage[: n - split, self._nobs_sl],
                terminal_mask[split:] if terminal_mask is not None else None,
                terminal_next_obs[split:] if terminal_next_obs is not None else None,
                self._storage[: n - split, self._ncritic_sl] if has_critic else None,
                terminal_next_critic[split:] if terminal_next_critic is not None else None,
            )

        self.ptr[0] += n
        self.size[0] = min(int(self.size[0]) + n, self.capacity)
        if self.trace_recorder is not None:
            self.trace_recorder.add_slice(
                "replay/add",
                category="replay",
                start_ns=_trace_ns,
                end_ns=time.perf_counter_ns(),
                args={"batch_size": int(n), "device": self.device},
            )

    @staticmethod
    def _patch_terminal_next_observations(
        target_next_obs,
        terminal_mask,
        terminal_next_obs,
        target_next_critic=None,
        terminal_next_critic=None,
    ) -> None:
        if terminal_mask is None or terminal_next_obs is None:
            return
        if terminal_mask.ndim != 1 or terminal_mask.shape[0] != target_next_obs.shape[0]:
            return
        if not torch.any(terminal_mask):
            return

        target_next_obs[terminal_mask] = terminal_next_obs[terminal_mask]

        if target_next_critic is not None and terminal_next_critic is not None:
            target_next_critic[terminal_mask] = terminal_next_critic[terminal_mask]

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample batch (called by learner)."""
        self.last_incremental_h2d_time_s = 0.0
        _trace_ns = time.perf_counter_ns() if self.trace_recorder is not None else 0
        size = int(self.size[0])
        _indices_ns = time.perf_counter_ns() if self.trace_recorder is not None else 0
        indices = torch.randint(0, size, (batch_size,))
        if self.trace_recorder is not None:
            self.trace_recorder.add_slice(
                "replay/sample_indices",
                category="replay",
                start_ns=_indices_ns,
                end_ns=time.perf_counter_ns(),
                args={"batch_size": int(batch_size), "size": int(size)},
            )

        chunk = self._storage[indices].to(self.device)
        batch = {
            "obs": chunk[:, self._obs_sl],
            "next_obs": chunk[:, self._nobs_sl],
            "actions": chunk[:, self._act_sl],
            "rewards": chunk[:, self._rew_col],
            "dones": chunk[:, self._done_col],
            "truncated": chunk[:, self._trunc_col],
        }
        if self._critic_dim > 0:
            batch["critic"] = chunk[:, self._critic_sl]
            batch["next_critic"] = chunk[:, self._ncritic_sl]
        if self.trace_recorder is not None:
            self.trace_recorder.add_slice(
                "replay/sample",
                category="replay",
                start_ns=_trace_ns,
                end_ns=time.perf_counter_ns(),
                args={"batch_size": int(batch_size), "device": self.device},
            )
        return batch
