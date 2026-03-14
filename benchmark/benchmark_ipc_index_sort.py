#!/usr/bin/env python3
"""Benchmark: sorted vs unsorted indices for shared-memory scatter read + H2D copy.

Optimization point 1: replay buffer sample() uses random (unsorted) indices to
scatter-read from a CPU shared tensor.  Sorting the indices first can improve
cache utilization on the CPU read, potentially reducing latency before the H2D
transfer begins.

Usage:
    python benchmark/benchmark_ipc_index_sort.py
"""

import sys
import time

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


def make_shared_buffers(capacity: int, obs_dim: int, action_dim: int):
    """Create 6 shared-memory tensors that mimic a real SharedReplayBuffer."""
    obs = torch.zeros(capacity, obs_dim).share_memory_()
    next_obs = torch.zeros(capacity, obs_dim).share_memory_()
    actions = torch.zeros(capacity, action_dim).share_memory_()
    rewards = torch.zeros(capacity).share_memory_()
    dones = torch.zeros(capacity).share_memory_()
    truncated = torch.zeros(capacity).share_memory_()

    # Fill with realistic random data
    obs.uniform_(-1, 1)
    next_obs.uniform_(-1, 1)
    actions.uniform_(-1, 1)
    rewards.uniform_(-1, 1)

    return obs, next_obs, actions, rewards, dones, truncated


# ---------------------------------------------------------------------------
# measurement helpers
# ---------------------------------------------------------------------------


def _measure_cpu_index(buffers, indices):
    """CPU scatter read only – no H2D."""
    obs, next_obs, actions, rewards, dones, truncated = buffers
    _ = (
        obs[indices],
        next_obs[indices],
        actions[indices],
        rewards[indices],
        dones[indices],
        truncated[indices],
    )


def _measure_h2d(cpu_chunks, device):
    """H2D transfer only – tensors already indexed."""
    for t in cpu_chunks:
        _ = t.to(device)


def _measure_full(buffers, indices, device):
    """Full flow: CPU scatter read + H2D."""
    obs, next_obs, actions, rewards, dones, truncated = buffers
    _ = {
        "obs": obs[indices].to(device),
        "next_obs": next_obs[indices].to(device),
        "actions": actions[indices].to(device),
        "rewards": rewards[indices].to(device),
        "dones": dones[indices].to(device),
        "truncated": truncated[indices].to(device),
    }


# ---------------------------------------------------------------------------
# benchmarks
# ---------------------------------------------------------------------------


def run_benchmark(
    buffers,
    capacity: int,
    batch_size: int,
    device: str,
    warmup: int = 5,
    measured: int = 20,
    sorted_idx: bool = False,
):
    """
    Returns (cpu_index_ms, h2d_ms, total_ms) as (mean, std) tuples.
    """
    cpu_times, h2d_times, total_times = [], [], []

    for i in range(warmup + measured):
        indices_raw = torch.randint(0, capacity, (batch_size,))
        indices = torch.sort(indices_raw)[0] if sorted_idx else indices_raw

        # --- CPU index only ---
        sync_device(device)
        t0 = time.perf_counter()
        cpu_chunks = [
            buffers[0][indices],
            buffers[1][indices],
            buffers[2][indices],
            buffers[3][indices],
            buffers[4][indices],
            buffers[5][indices],
        ]
        t1 = time.perf_counter()

        # --- H2D only (from already-indexed CPU tensors) ---
        sync_device(device)
        t2 = time.perf_counter()
        _ = [c.to(device) for c in cpu_chunks]
        sync_device(device)
        t3 = time.perf_counter()

        # --- Full flow (fresh indices) ---
        indices2_raw = torch.randint(0, capacity, (batch_size,))
        indices2 = torch.sort(indices2_raw)[0] if sorted_idx else indices2_raw
        sync_device(device)
        t4 = time.perf_counter()
        _measure_full(buffers, indices2, device)
        sync_device(device)
        t5 = time.perf_counter()

        if i >= warmup:
            cpu_times.append((t1 - t0) * 1e3)
            h2d_times.append((t3 - t2) * 1e3)
            total_times.append((t5 - t4) * 1e3)

    def s(lst):
        a = np.array(lst)
        return a.mean(), a.std()

    return s(cpu_times), s(h2d_times), s(total_times)


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def main():
    # Device detection
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

    # --- config (aligned with real training) ---
    capacity = 4096 * 1024  # 4,194,304
    obs_dim = 45
    action_dim = 12
    batch_size = 8192 * 8  # 65,536  (batch_size × updates_per_step)
    warmup = 5
    measured = 20

    print("=== Index Sort Benchmark (MPS) ===")
    print(f"capacity={capacity:,}  batch={batch_size:,}  fields=6\n")
    print("Allocating shared tensors …")
    buffers = make_shared_buffers(capacity, obs_dim, action_dim)
    print("Done.\n")

    print("Running unsorted …")
    (cpu_u, cpu_u_s), (h2d_u, h2d_u_s), (tot_u, tot_u_s) = run_benchmark(
        buffers, capacity, batch_size, device, warmup=warmup, measured=measured, sorted_idx=False
    )

    print("Running sorted …")
    (cpu_s, cpu_s_s), (h2d_s, h2d_s_s), (tot_s, tot_s_s) = run_benchmark(
        buffers, capacity, batch_size, device, warmup=warmup, measured=measured, sorted_idx=True
    )

    # Sort overhead: sorting itself costs some time.
    # Measure it separately (included in the Sorted "CPU index" column).
    sort_times = []
    for _ in range(measured):
        idx = torch.randint(0, capacity, (batch_size,))
        t0 = time.perf_counter()
        torch.sort(idx)
        t1 = time.perf_counter()
        sort_times.append((t1 - t0) * 1e3)
    sort_overhead = float(np.mean(sort_times))

    # Net gain = total unsorted − total sorted (positive means sorted is faster)
    net_gain = tot_u - tot_s

    # --- pretty print ---
    w = 14

    def speedup(a, b):
        return a / b if b > 0 else float("nan")

    print()
    print("=" * 68)
    header = f"{'':20s}{'CPU index':>{w}}  {'H2D copy':>{w}}  {'Total':>{w}}"
    print(header)
    print("-" * 68)

    def row(label, cpu_m, cpu_s, h2d_m, h2d_s, tot_m, tot_s_v):
        return (
            f"{label:20s}"
            f"{cpu_m:>{w - 5}.2f} ms ±{cpu_s:.2f}  "
            f"{h2d_m:>{w - 5}.2f} ms ±{h2d_s:.2f}  "
            f"{tot_m:>{w - 5}.2f} ms ±{tot_s_v:.2f}"
        )

    print(row("Unsorted:", cpu_u, cpu_u_s, h2d_u, h2d_u_s, tot_u, tot_u_s))
    print(row("Sorted:  ", cpu_s, cpu_s_s, h2d_s, h2d_s_s, tot_s, tot_s_s))

    sp_cpu = speedup(cpu_u, cpu_s)
    sp_h2d = speedup(h2d_u, h2d_s)
    sp_tot = speedup(tot_u, tot_s)
    print(
        f"{'Speedup:':20s}"
        f"{sp_cpu:>{w - 5}.2f}x        "
        f"{sp_h2d:>{w - 5}.2f}x        "
        f"{sp_tot:>{w - 5}.2f}x"
    )

    print("-" * 68)
    print(f"Sort overhead:      {sort_overhead:.2f} ms   (included in Sorted CPU index)")
    sign = "+" if net_gain >= 0 else "-"
    print(
        f"Net gain:           {sign}{abs(net_gain):.2f} ms per iteration "
        f"({'sorted faster' if net_gain >= 0 else 'unsorted faster'})"
    )
    print("=" * 68)


if __name__ == "__main__":
    main()
