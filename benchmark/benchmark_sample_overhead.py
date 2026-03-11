#!/usr/bin/env python3
"""Realistic sampling benchmark - measures actual sample overhead."""

import torch
import numpy as np
import time
import sys
import multiprocessing as mp
from multiprocessing import shared_memory

def sync_device(device):
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()

def benchmark_host_sample_realistic(capacity, obs_dim, action_dim, batch_size, num_samples, device):
    """Simulate real SharedReplayBuffer.sample_torch() with lock."""
    # Create numpy buffers (host memory)
    obs = np.random.randn(capacity, obs_dim).astype(np.float32)
    actions = np.random.randn(capacity, action_dim).astype(np.float32)
    rewards = np.random.randn(capacity).astype(np.float32)
    next_obs = np.random.randn(capacity, obs_dim).astype(np.float32)
    dones = np.zeros(capacity, dtype=np.float32)
    truncated = np.zeros(capacity, dtype=np.float32)

    lock = mp.get_context("spawn").Lock()

    times = []
    for i in range(num_samples):
        sync_device(device)
        start = time.perf_counter()

        # Simulate SharedReplayBuffer.sample_torch() logic
        with lock:
            indices = np.random.randint(0, capacity, size=batch_size)
            obs_copy = obs[indices].copy()
            actions_copy = actions[indices].copy()
            rewards_copy = rewards[indices].copy()
            next_obs_copy = next_obs[indices].copy()
            dones_copy = dones[indices].copy()
            truncated_copy = truncated[indices].copy()

        # Transfer to device (non-blocking)
        obs_t = torch.from_numpy(obs_copy).to(device, non_blocking=True)
        actions_t = torch.from_numpy(actions_copy).to(device, non_blocking=True)
        rewards_t = torch.from_numpy(rewards_copy).to(device, non_blocking=True)
        next_obs_t = torch.from_numpy(next_obs_copy).to(device, non_blocking=True)
        dones_t = torch.from_numpy(dones_copy).to(device, non_blocking=True)
        truncated_t = torch.from_numpy(truncated_copy).to(device, non_blocking=True)

        sync_device(device)
        elapsed = time.perf_counter() - start
        if i >= 2:
            times.append(elapsed * 1000)

    return np.mean(times), np.std(times)

def benchmark_gpu_sample(capacity, obs_dim, action_dim, batch_size, num_samples, device):
    """GPU buffer: direct indexing on device."""
    obs = torch.randn(capacity, obs_dim, device=device)
    actions = torch.randn(capacity, action_dim, device=device)
    rewards = torch.randn(capacity, device=device)
    next_obs = torch.randn(capacity, obs_dim, device=device)
    dones = torch.zeros(capacity, device=device)
    truncated = torch.zeros(capacity, device=device)

    times = []
    for i in range(num_samples):
        sync_device(device)
        start = time.perf_counter()

        indices = torch.randint(0, capacity, (batch_size,), device=device)
        obs_t = obs[indices]
        actions_t = actions[indices]
        rewards_t = rewards[indices]
        next_obs_t = next_obs[indices]
        dones_t = dones[indices]
        truncated_t = truncated[indices]

        sync_device(device)
        elapsed = time.perf_counter() - start
        if i >= 2:
            times.append(elapsed * 1000)

    return np.mean(times), np.std(times)

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}\n")

    capacity = 4096 * 1024
    obs_dim = 48
    action_dim = 12
    batch_size = 8192
    num_samples = 50

    print("=== Pure Sampling Overhead (Realistic) ===")
    print(f"Capacity: {capacity:,}, Batch: {batch_size}")
    print(f"Samples: {num_samples} (2 warmup + 48 measured)\n")

    print("[1/2] Host Buffer (with lock + copy)...")
    host_mean, host_std = benchmark_host_sample_realistic(
        capacity, obs_dim, action_dim, batch_size, num_samples, device
    )

    print("[2/2] GPU Buffer...")
    gpu_mean, gpu_std = benchmark_gpu_sample(
        capacity, obs_dim, action_dim, batch_size, num_samples, device
    )

    print("\n" + "="*60)
    print("SAMPLING OVERHEAD")
    print("="*60)
    print(f"Host: {host_mean:.3f} ± {host_std:.3f} ms")
    print(f"GPU:  {gpu_mean:.3f} ± {gpu_std:.3f} ms")
    print(f"Overhead: {host_mean - gpu_mean:.3f} ms ({(host_mean/gpu_mean):.2f}x)")
    print("="*60)

    updates_per_step = 8
    print(f"\nPer env step ({updates_per_step} updates):")
    print(f"  Host overhead: {(host_mean - gpu_mean) * updates_per_step:.2f} ms")
