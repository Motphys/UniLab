#!/usr/bin/env python3
"""Benchmark: split (6 separate tensors) vs packed (single concatenated tensor) buffer layout.

Optimization point 2: the current replay buffer stores 6 fields as separate shared
tensors and calls .to("mps") six times per sample() call (6 Metal command submissions).
A packed layout stores all fields in a single shared tensor and submits only 1
Metal command, then slices columns on the GPU.

Usage:
    python benchmark/benchmark_ipc_buffer_layout.py
"""

import time
import sys

import numpy as np
import torch


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def sync_device(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


# ---------------------------------------------------------------------------
# Split layout  (current architecture)
# ---------------------------------------------------------------------------

class SplitBuffer:
    """6 separate shared tensors – mimics current SharedReplayBuffer."""

    def __init__(self, capacity: int, obs_dim: int, action_dim: int):
        self.capacity   = capacity
        self.obs_dim    = obs_dim
        self.action_dim = action_dim
        self._ptr       = 0

        self.obs       = torch.zeros(capacity, obs_dim).share_memory_()
        self.next_obs  = torch.zeros(capacity, obs_dim).share_memory_()
        self.actions   = torch.zeros(capacity, action_dim).share_memory_()
        self.rewards   = torch.zeros(capacity).share_memory_()
        self.dones     = torch.zeros(capacity).share_memory_()
        self.truncated = torch.zeros(capacity).share_memory_()

    def add(self, obs, next_obs, actions, rewards, dones, truncated):
        n   = obs.shape[0]
        idx = self._ptr % self.capacity
        end = min(idx + n, self.capacity)
        k   = end - idx

        self.obs[idx:end]       = obs[:k]
        self.next_obs[idx:end]  = next_obs[:k]
        self.actions[idx:end]   = actions[:k]
        self.rewards[idx:end]   = rewards[:k]
        self.dones[idx:end]     = dones[:k]
        self.truncated[idx:end] = truncated[:k]
        self._ptr += k

    def sample(self, batch_size: int, device: str):
        size    = min(self._ptr, self.capacity)
        indices = torch.randint(0, size, (batch_size,))
        return {
            "obs":      self.obs[indices].to(device),
            "next_obs": self.next_obs[indices].to(device),
            "actions":  self.actions[indices].to(device),
            "rewards":  self.rewards[indices].to(device),
            "dones":    self.dones[indices].to(device),
            "truncated": self.truncated[indices].to(device),
        }

    def sample_cpu_index(self, batch_size: int):
        """CPU index only – no H2D."""
        size    = min(self._ptr, self.capacity)
        indices = torch.randint(0, size, (batch_size,))
        return [
            self.obs[indices],
            self.next_obs[indices],
            self.actions[indices],
            self.rewards[indices],
            self.dones[indices],
            self.truncated[indices],
        ]

    def h2d_only(self, cpu_chunks, device: str):
        """H2D from pre-indexed CPU tensors."""
        return [c.to(device) for c in cpu_chunks]


# ---------------------------------------------------------------------------
# Packed layout  (proposed architecture)
# ---------------------------------------------------------------------------

class PackedBuffer:
    """Single packed shared tensor – 1 Metal command per sample()."""

    def __init__(self, capacity: int, obs_dim: int, action_dim: int):
        self.capacity   = capacity
        self.obs_dim    = obs_dim
        self.action_dim = action_dim
        self._ptr       = 0

        # column layout: [obs | next_obs | actions | rewards | dones | truncated]
        total_dim = obs_dim + obs_dim + action_dim + 3
        self._storage = torch.zeros(capacity, total_dim).share_memory_()

        # zero-copy views
        c = 0
        self._obs_sl       = slice(c, c + obs_dim);    c += obs_dim
        self._nobs_sl      = slice(c, c + obs_dim);    c += obs_dim
        self._act_sl       = slice(c, c + action_dim); c += action_dim
        self._rew_col      = c;  c += 1
        self._don_col      = c;  c += 1
        self._trun_col     = c

    def add(self, obs, next_obs, actions, rewards, dones, truncated):
        n   = obs.shape[0]
        idx = self._ptr % self.capacity
        end = min(idx + n, self.capacity)
        k   = end - idx

        row = torch.cat([
            obs[:k],
            next_obs[:k],
            actions[:k],
            rewards[:k].unsqueeze(1),
            dones[:k].unsqueeze(1),
            truncated[:k].unsqueeze(1),
        ], dim=1)
        self._storage[idx:end] = row
        self._ptr += k

    def sample(self, batch_size: int, device: str):
        size    = min(self._ptr, self.capacity)
        indices = torch.randint(0, size, (batch_size,))
        chunk   = self._storage[indices].to(device)   # single H2D
        return {
            "obs":      chunk[:, self._obs_sl],
            "next_obs": chunk[:, self._nobs_sl],
            "actions":  chunk[:, self._act_sl],
            "rewards":  chunk[:, self._rew_col],
            "dones":    chunk[:, self._don_col],
            "truncated": chunk[:, self._trun_col],
        }

    def sample_cpu_index(self, batch_size: int):
        size    = min(self._ptr, self.capacity)
        indices = torch.randint(0, size, (batch_size,))
        return [self._storage[indices]]

    def h2d_only(self, cpu_chunks, device: str):
        return [cpu_chunks[0].to(device)]


# ---------------------------------------------------------------------------
# timing helpers
# ---------------------------------------------------------------------------

def _fill_buffer(buf, capacity: int, obs_dim: int, action_dim: int,
                 chunk: int = 4096):
    """Pre-populate buffer to at least `capacity` entries."""
    filled = 0
    while filled < capacity:
        n = min(chunk, capacity - filled)
        buf.add(
            torch.randn(n, obs_dim),
            torch.randn(n, obs_dim),
            torch.randn(n, action_dim),
            torch.randn(n),
            torch.zeros(n),
            torch.zeros(n),
        )
        filled += n


def bench_add(buf, obs_dim: int, action_dim: int, n: int,
              warmup: int, measured: int):
    obs      = torch.randn(n, obs_dim)
    next_obs = torch.randn(n, obs_dim)
    actions  = torch.randn(n, action_dim)
    rewards  = torch.randn(n)
    dones    = torch.zeros(n)
    truncated = torch.zeros(n)

    times = []
    for i in range(warmup + measured):
        t0 = time.perf_counter()
        buf.add(obs, next_obs, actions, rewards, dones, truncated)
        t1 = time.perf_counter()
        if i >= warmup:
            times.append((t1 - t0) * 1e3)
    a = np.array(times)
    return a.mean(), a.std()


def bench_sample(buf, batch_size: int, device: str, warmup: int, measured: int):
    """Full sample() including CPU index + H2D."""
    times_total, times_cpu, times_h2d = [], [], []

    for i in range(warmup + measured):
        # Full
        sync_device(device)
        t0 = time.perf_counter()
        buf.sample(batch_size, device)
        sync_device(device)
        t1 = time.perf_counter()

        # CPU index only
        tc0 = time.perf_counter()
        cpu_chunks = buf.sample_cpu_index(batch_size)
        tc1 = time.perf_counter()

        # H2D only
        sync_device(device)
        th0 = time.perf_counter()
        buf.h2d_only(cpu_chunks, device)
        sync_device(device)
        th1 = time.perf_counter()

        if i >= warmup:
            times_total.append((t1 - t0) * 1e3)
            times_cpu.append((tc1 - tc0) * 1e3)
            times_h2d.append((th1 - th0) * 1e3)

    def s(lst):
        a = np.array(lst)
        return a.mean(), a.std()

    return s(times_total), s(times_cpu), s(times_h2d)


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def main():
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
        print("WARNING: No GPU detected, running on CPU (results informational only)\n")

    print(f"Device:  {device}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Platform: {sys.platform}\n")

    capacity   = 4096 * 1024   # 4,194,304
    obs_dim    = 45
    action_dim = 12
    add_n      = 4096          # rows written per add() call (collector step)
    batch_size = 8192 * 8      # 65,536
    warmup     = 5
    measured   = 20

    print("=== Buffer Layout Benchmark (MPS) ===\n")
    print(f"capacity={capacity:,}  obs_dim={obs_dim}  action_dim={action_dim}")
    print(f"add_n={add_n:,}  batch_size={batch_size:,}\n")

    # --- allocate ---
    print("Allocating Split buffer …")
    split = SplitBuffer(capacity, obs_dim, action_dim)
    print("Allocating Packed buffer …")
    packed = PackedBuffer(capacity, obs_dim, action_dim)

    # pre-fill so sample() can draw full batches
    print("Pre-filling Split buffer …")
    _fill_buffer(split, capacity, obs_dim, action_dim)
    print("Pre-filling Packed buffer …")
    _fill_buffer(packed, capacity, obs_dim, action_dim)
    print("Done.\n")

    # --- add() ---
    print("Benchmarking add() …")
    add_s_m, add_s_s = bench_add(split,  obs_dim, action_dim, add_n, warmup, measured)
    add_p_m, add_p_s = bench_add(packed, obs_dim, action_dim, add_n, warmup, measured)

    # --- sample() ---
    print("Benchmarking sample() …")
    (tot_s, tot_s_s), (cpu_s, cpu_s_s), (h2d_s, h2d_s_s) = bench_sample(
        split, batch_size, device, warmup, measured
    )
    (tot_p, tot_p_s), (cpu_p, cpu_p_s), (h2d_p, h2d_p_s) = bench_sample(
        packed, batch_size, device, warmup, measured
    )

    speedup = tot_s / tot_p if tot_p > 0 else float("nan")

    print()
    print("=" * 65)
    print(f"add() per call (n={add_n:,}):")
    print(f"  Split:   {add_s_m:.3f} ± {add_s_s:.3f} ms")
    print(f"  Packed:  {add_p_m:.3f} ± {add_p_s:.3f} ms")
    print()
    print(f"sample() total (batch={batch_size:,}):")
    print(
        f"  Split:   {tot_s:.3f} ± {tot_s_s:.3f} ms"
        f"   (CPU index: {cpu_s:.3f} ms, H2D×6: {h2d_s:.3f} ms)"
    )
    print(
        f"  Packed:  {tot_p:.3f} ± {tot_p_s:.3f} ms"
        f"   (CPU index: {cpu_p:.3f} ms, H2D×1: {h2d_p:.3f} ms)"
    )
    print(f"  Speedup: {speedup:.2f}x")
    print("=" * 65)


if __name__ == "__main__":
    main()
