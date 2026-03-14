#!/usr/bin/env python3
"""GPU Buffer Benchmark - Host vs GPU replay buffer performance.

Cross-platform: CUDA, MPS (macOS), CPU fallback.
Usage: python scripts/benchmark_gpu_buffer.py
"""

import torch
import torch.nn as nn
import numpy as np
import time
import sys


def sync_device(device):
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


class SimpleActor(nn.Module):
    def __init__(self, obs_dim, hidden_dim, action_dim, device):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, action_dim),
        ).to(device)

    def forward(self, x):
        return self.net(x)


class SimpleCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim, device):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        ).to(device)

    def forward(self, obs, act):
        return self.net(torch.cat([obs, act], dim=-1))


def benchmark_host_buffer(capacity, obs_dim, action_dim, batch_size, num_updates, device):
    """Host buffer: sample from numpy, transfer to device (current architecture)."""
    obs_buf = np.random.randn(capacity, obs_dim).astype(np.float32)
    actions_buf = np.random.randn(capacity, action_dim).astype(np.float32)
    next_obs_buf = np.random.randn(capacity, obs_dim).astype(np.float32)

    actor = SimpleActor(obs_dim, 512, action_dim, device)
    critic = SimpleCritic(obs_dim, action_dim, 768, device)
    optimizer = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=3e-4)

    times = []
    for i in range(num_updates):
        sync_device(device)
        start = time.perf_counter()

        # Sample (simulate SharedReplayBuffer.sample_torch)
        indices = np.random.randint(0, capacity, size=batch_size)
        obs = torch.from_numpy(obs_buf[indices].copy()).to(device, non_blocking=True)
        actions = torch.from_numpy(actions_buf[indices].copy()).to(device, non_blocking=True)
        next_obs = torch.from_numpy(next_obs_buf[indices].copy()).to(device, non_blocking=True)
        sync_device(device)

        # Training step
        pred_actions = actor(obs)
        q_values = critic(obs, actions)
        loss = (pred_actions - actions).pow(2).mean() + q_values.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        sync_device(device)
        elapsed = time.perf_counter() - start
        if i >= 2:  # Skip warmup
            times.append(elapsed * 1000)

    return np.mean(times), np.std(times)


def benchmark_gpu_buffer(capacity, obs_dim, action_dim, batch_size, num_updates, device):
    """GPU buffer: sample directly on device (proposed architecture)."""
    obs_buf = torch.randn(capacity, obs_dim, device=device)
    actions_buf = torch.randn(capacity, action_dim, device=device)
    next_obs_buf = torch.randn(capacity, obs_dim, device=device)

    actor = SimpleActor(obs_dim, 512, action_dim, device)
    critic = SimpleCritic(obs_dim, action_dim, 768, device)
    optimizer = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=3e-4)

    times = []
    for i in range(num_updates):
        sync_device(device)
        start = time.perf_counter()

        # Sample
        indices = torch.randint(0, capacity, (batch_size,), device=device)
        obs = obs_buf[indices]
        actions = actions_buf[indices]
        next_obs = next_obs_buf[indices]

        # Training step
        pred_actions = actor(obs)
        q_values = critic(obs, actions)
        loss = (pred_actions - actions).pow(2).mean() + q_values.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        sync_device(device)
        elapsed = time.perf_counter() - start
        if i >= 2:
            times.append(elapsed * 1000)

    return np.mean(times), np.std(times)


if __name__ == "__main__":
    # Device detection
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
        print("WARNING: No GPU detected, running on CPU (results not meaningful)\n")

    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Platform: {sys.platform}\n")

    # Go1 config
    capacity = 4096 * 1024
    obs_dim = 48
    action_dim = 12
    batch_size = 8192
    num_updates = 12  # 2 warmup + 10 measured

    print("=== Configuration ===")
    print(f"Capacity: {capacity:,}")
    print(f"Obs dim: {obs_dim}, Action dim: {action_dim}")
    print(f"Batch size: {batch_size}")
    print(f"Updates: {num_updates} (2 warmup + 10 measured)\n")

    print("=== Running Benchmarks ===")
    print("(This may take 1-2 minutes...)\n")

    print("[1/2] Host Buffer (current)...")
    host_mean, host_std = benchmark_host_buffer(
        capacity, obs_dim, action_dim, batch_size, num_updates, device
    )

    print("[2/2] GPU Buffer (proposed)...")
    gpu_mean, gpu_std = benchmark_gpu_buffer(
        capacity, obs_dim, action_dim, batch_size, num_updates, device
    )

    # Results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Host Buffer: {host_mean:.2f} ± {host_std:.2f} ms per update")
    print(f"GPU Buffer:  {gpu_mean:.2f} ± {gpu_std:.2f} ms per update")
    print(f"\nSpeedup: {host_mean / gpu_mean:.2f}x")
    print("=" * 60)

    # Extrapolate
    updates_per_step = 8
    print(f"\nPer env step (updates_per_step={updates_per_step}):")
    print(f"  Host: {host_mean * updates_per_step:.1f} ms")
    print(f"  GPU:  {gpu_mean * updates_per_step:.1f} ms")
    print(f"  Speedup: {(host_mean * updates_per_step) / (gpu_mean * updates_per_step):.2f}x")
