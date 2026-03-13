#!/usr/bin/env python3
"""Benchmark: write_weights() with vs without a cpu_buffer intermediate copy.

Optimization point 3: the current weight-sync path copies parameters from MPS to a
CPU tensor first (MPS → cpu_buffer → shm numpy array).  The proposed "direct" path
skips the intermediate CPU tensor and writes straight from .cpu().numpy() into shm
(MPS → shm numpy array).

The model architecture matches the real SACActor used in training:
    obs=45 → Linear(512) → LayerNorm → SiLU
           → Linear(256) → LayerNorm → SiLU
           → Linear(128) → LayerNorm → SiLU
           → Linear(action_dim)

Usage:
    python benchmark/benchmark_ipc_weight_sync.py
"""

import time
import sys
import threading
from multiprocessing.shared_memory import SharedMemory

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def sync_device(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


# ---------------------------------------------------------------------------
# Actor model (SACActor-equivalent)
# ---------------------------------------------------------------------------

class SACActor(nn.Module):
    """Matches the real SACActor architecture used in training."""

    def __init__(self, obs_dim: int, action_dim: int,
                 hidden: tuple = (512, 256, 128)):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.LayerNorm(h), nn.SiLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------------------
# Weight-sync implementations
# ---------------------------------------------------------------------------

def write_weights_current(model: nn.Module, cpu_buffer: np.ndarray,
                           shm_array: np.ndarray, lock: threading.Lock,
                           version_arr: np.ndarray, device: str) -> None:
    """Current implementation: MPS → CPU tensor → shm numpy."""
    sync_device(device)
    with lock:
        offset = 0
        cpu_t = torch.from_numpy(cpu_buffer)   # zero-copy view of shm buffer
        for param in model.parameters():
            flat = param.detach().cpu().flatten()  # MPS → CPU tensor
            n = flat.numel()
            cpu_t[offset: offset + n].copy_(flat) # CPU tensor → shm (via cpu_buffer)
            offset += n
        version_arr[0] += 1


def write_weights_direct(model: nn.Module, shm_array: np.ndarray,
                          lock: threading.Lock, version_arr: np.ndarray,
                          device: str) -> None:
    """Direct implementation: MPS → shm numpy (no intermediate CPU tensor)."""
    sync_device(device)
    with lock:
        offset = 0
        for param in model.parameters():
            arr = param.detach().cpu().numpy().ravel()  # MPS → numpy (shm-backed)
            n = arr.size
            shm_array[offset: offset + n] = arr
            offset += n
        version_arr[0] += 1


# ---------------------------------------------------------------------------
# benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(model: nn.Module, total_params: int, device: str,
                  warmup: int, measured: int):
    """Returns (current_times_ms, direct_times_ms)."""

    # Allocate real shared memory (no ordinary numpy array)
    shm = SharedMemory(create=True, size=total_params * 4)   # float32
    shm_array  = np.ndarray((total_params,), dtype=np.float32, buffer=shm.buf)
    cpu_buffer = np.zeros(total_params, dtype=np.float32)    # intermediate buffer
    version_arr = np.zeros(1, dtype=np.int64)
    lock = threading.Lock()

    cur_times  = []
    dir_times  = []

    for i in range(warmup + measured):
        # --- current ---
        sync_device(device)
        t0 = time.perf_counter()
        write_weights_current(model, cpu_buffer, shm_array, lock, version_arr, device)
        t1 = time.perf_counter()

        # --- direct ---
        sync_device(device)
        t2 = time.perf_counter()
        write_weights_direct(model, shm_array, lock, version_arr, device)
        t3 = time.perf_counter()

        if i >= warmup:
            cur_times.append((t1 - t0) * 1e3)
            dir_times.append((t3 - t2) * 1e3)

    shm.close()
    shm.unlink()

    return np.array(cur_times), np.array(dir_times)


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

    obs_dim    = 45
    action_dim = 12
    hidden     = (512, 256, 128)
    warmup     = 5
    measured   = 20

    model = SACActor(obs_dim, action_dim, hidden).to(device)
    model.eval()

    total_params = count_params(model)
    size_mb = total_params * 4 / 1024 / 1024

    print("=== Weight Sync Benchmark (MPS) ===")
    print(f"Model: SACActor  obs={obs_dim}  hidden={hidden}  action={action_dim}")
    print(f"Total params: {total_params:,}  (~{size_mb:.2f} MB)\n")

    print(f"Running {warmup} warmup + {measured} measured iterations …\n")
    cur_arr, dir_arr = run_benchmark(
        model, total_params, device, warmup=warmup, measured=measured
    )

    speedup = cur_arr.mean() / dir_arr.mean() if dir_arr.mean() > 0 else float("nan")
    saving  = cur_arr.mean() - dir_arr.mean()

    w = 10
    print(f"{'':14s}{'Mean':>{w}}  {'Std':>{w}}  {'Min':>{w}}  {'Max':>{w}}")
    print("-" * 55)

    def row(label, arr):
        return (
            f"{label:14s}"
            f"{arr.mean():>{w}.3f} ms"
            f"  ±{arr.std():>{w-2}.3f}"
            f"  {arr.min():>{w}.3f}"
            f"  {arr.max():>{w}.3f}"
        )

    print(row("Current:", cur_arr))
    print(row("Direct: ", dir_arr))
    print("-" * 55)
    sign = "save" if saving >= 0 else "cost"
    print(
        f"Speedup:       {speedup:.2f}x  "
        f"({sign} {abs(saving):.3f} ms per iteration)"
    )
    print("=" * 55)


if __name__ == "__main__":
    main()
