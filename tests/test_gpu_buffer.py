#!/usr/bin/env python3
"""Quick test for GPUReplayBuffer functionality."""

import torch
import numpy as np
from unilab.ipc import SharedReplayBuffer, GPUReplayBuffer

def test_gpu_buffer():
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Testing on device: {device}")

    capacity = 1000
    obs_dim = 48
    action_dim = 12
    batch_size = 256

    # Create host buffer
    host_buffer = SharedReplayBuffer(capacity, obs_dim, action_dim, create=True)

    # Fill with test data
    for i in range(3):
        batch = 100
        host_buffer.add_batch(
            obs=np.random.randn(batch, obs_dim).astype(np.float32),
            actions=np.random.randn(batch, action_dim).astype(np.float32),
            rewards=np.random.randn(batch).astype(np.float32),
            next_obs=np.random.randn(batch, obs_dim).astype(np.float32),
            dones=np.zeros(batch, dtype=np.float32),
            truncated=np.zeros(batch, dtype=np.float32),
        )

    print(f"Host buffer size: {host_buffer.size}")

    # Create GPU buffer
    gpu_buffer = GPUReplayBuffer(capacity, obs_dim, action_dim, device)

    # Sync from host
    synced = gpu_buffer.sync_from_host(host_buffer)
    print(f"Synced {synced} samples to GPU")
    print(f"GPU buffer size: {gpu_buffer.size}")

    # Sample
    batch = gpu_buffer.sample(batch_size)
    print(f"Sampled batch keys: {batch.keys()}")
    print(f"Obs shape: {batch['obs'].shape}, device: {batch['obs'].device}")

    # Verify data integrity
    assert batch['obs'].shape == (batch_size, obs_dim)
    assert batch['actions'].shape == (batch_size, action_dim)
    assert str(batch['obs'].device).startswith(device.split(':')[0])

    print("✅ All tests passed!")

    host_buffer.cleanup()

if __name__ == "__main__":
    test_gpu_buffer()
