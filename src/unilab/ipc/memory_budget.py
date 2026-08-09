"""Memory budget estimation for async RL training buffers.

Pure functions that estimate memory usage and warn if the system
is likely to OOM before allocating large shared buffers.
"""

from __future__ import annotations

import os
import shutil
import sys


def estimate_offpolicy_bytes(
    num_envs: int,
    replay_buffer_n: int,
    obs_dim: int,
    action_dim: int,
    critic_dim: int,
    ingress_depth: int = 2,
) -> dict[str, int | str]:
    """Estimate host shared memory for bounded off-policy replay ingress."""
    row_width = 2 * obs_dim + action_dim + 3 + 2 * critic_dim
    capacity = replay_buffer_n * num_envs
    ingress_bytes = int(ingress_depth) * num_envs * row_width * 4
    return {
        "replay_buffer": 0,
        "bounded_ingress_slots": ingress_bytes,
        "total": ingress_bytes,
        "breakdown": (
            "Replay: 0 MB shared host memory "
            f"({capacity} rows remain authoritative on the learner device)\n"
            f"  Bounded ingress: {ingress_bytes / 1024**2:.0f} MB "
            f"({int(ingress_depth)} slots × {num_envs} rows × {row_width} cols × 4B)\n"
            "  Excludes MuJoCo BatchEnvPool/native allocations, CUDA pinned/shared "
            "registration, and driver memory."
        ),
    }


def estimate_appo_bytes(
    num_envs: int,
    steps_per_env: int,
    obs_dim: int,
    action_dim: int,
    critic_dim: int,
    num_slots: int = 4,
) -> dict[str, int | str]:
    """Estimate memory for APPO rollout ring buffer."""
    per_step = obs_dim + action_dim + 1 + 1 + 1 + 1 + critic_dim
    per_slot = num_envs * steps_per_env * per_step * 4
    last_obs_per_slot = num_envs * (obs_dim + critic_dim) * 4
    total_per_slot = per_slot + last_obs_per_slot
    total = total_per_slot * num_slots

    return {
        "ring_buffer": total,
        "total": total,
        "breakdown": (
            f"Ring buffer: {total / 1024**2:.0f} MB "
            f"({num_slots} slots × {num_envs} envs × {steps_per_env} steps × "
            f"{per_step} cols × 4B)"
        ),
    }


def get_available_memory_bytes() -> int | None:
    """Best-effort available memory detection."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError):
        pass

    try:
        import psutil

        return int(psutil.virtual_memory().available)
    except ImportError:
        pass

    return None


def get_shared_memory_available_bytes(path: str = "/dev/shm") -> int | None:
    """Best-effort available shared-memory space detection."""
    try:
        return int(shutil.disk_usage(path).free)
    except (OSError, ValueError):
        return None


def raise_if_shared_memory_over_budget(
    estimated: dict[str, int | str],
    label: str,
    threshold: float = 0.8,
    path: str = "/dev/shm",
) -> None:
    """Fail before allocating shared buffers that exceed shared-memory capacity."""
    available = get_shared_memory_available_bytes(path)
    if available is None:
        return

    total = int(estimated["total"])
    ratio = total / max(available, 1)
    if total <= available * threshold:
        return

    est_gb = total / 1024**3
    avail_gb = available / 1024**3
    raise MemoryError(
        f"{label}: estimated shared-memory allocation {est_gb:.1f} GB exceeds "
        f"{path} available {avail_gb:.1f} GB ({ratio:.0%} usage). "
        "Reduce algo.num_envs or algo.replay_buffer_n, or increase /dev/shm."
    )


def warn_if_over_budget(
    estimated: dict[str, int | str],
    label: str,
    threshold: float = 0.8,
) -> None:
    """Print a warning if estimated memory exceeds threshold of available."""
    if os.environ.get("UNILAB_SKIP_MEMORY_CHECK"):
        return

    available = get_available_memory_bytes()
    if available is None:
        return

    total = int(estimated["total"])
    ratio = total / available

    if ratio > threshold:
        est_gb = total / 1024**3
        avail_gb = available / 1024**3
        breakdown = estimated.get("breakdown", "")
        print(
            f"\n[Memory Warning] {label}: estimated {est_gb:.1f} GB, "
            f"available {avail_gb:.1f} GB ({ratio:.0%} usage).\n"
            f"  {breakdown}\n"
            f"  Consider reducing algo.num_envs or algo.replay_buffer_n.\n"
            f"  Native backend and driver memory may push actual usage higher.\n"
            f"  Suppress: export UNILAB_SKIP_MEMORY_CHECK=1\n",
            file=sys.stderr,
            flush=True,
        )
