"""RSL-RL distributed-training helpers.

The small pure helpers here (torchrun environment readers, ``training.devices``
normalization, launch-time device validation) are owned by UniLab so that
single-process PPO training and playback run without uni_rl installed. The
multi-GPU launcher and CPU partitioner stay owned by ``uni_rl.ipc.dp_launcher``
and are imported lazily only when a multi-rank topology is requested.
``UNILAB_DP_LOG_DIR`` is a shared environment-variable contract with that
launcher: it sets the variable for spawned workers, workers read it here.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Sequence, cast

import torch
from omegaconf import open_dict

UNILAB_DP_LOG_DIR = "UNILAB_DP_LOG_DIR"


def current_torch_distributed_rank() -> int:
    """Global torch-distributed rank of this process (0 outside torchrun)."""
    return int(os.environ.get("RANK", "0"))


def current_torch_distributed_local_rank() -> int:
    """Node-local torch-distributed rank of this process (0 outside torchrun)."""
    return int(os.environ.get("LOCAL_RANK", "0"))


def current_torch_distributed_world_size() -> int:
    """Torch-distributed world size of this process (1 outside torchrun)."""
    return int(os.environ.get("WORLD_SIZE", "1"))


def torchrun_ranks_are_colocated(world_size: int | None = None) -> bool:
    """Whether the torchrun ranks of this run share one host.

    The integrated data-parallel launcher puts every rank on one host, where
    partitioning the CPU affinity per rank avoids cross-rank contention. An
    external multi-node torchrun instead reports ``LOCAL_WORLD_SIZE <
    WORLD_SIZE``; each node then owns a full machine, so every rank must keep
    the host's complete CPU budget rather than a world-size slice of it.
    """
    resolved_world_size = int(
        world_size if world_size is not None else os.environ.get("WORLD_SIZE", "1")
    )
    if resolved_world_size < 1:
        raise ValueError(f"world_size must be positive, got {resolved_world_size}")
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", str(resolved_world_size)))
    if local_world_size < 1:
        raise ValueError(f"LOCAL_WORLD_SIZE must be positive, got {local_world_size}")
    return local_world_size == resolved_world_size


def resolve_dp_topology(devices_cfg: Any) -> tuple[int, ...] | None:
    """Normalize ``training.devices`` into an ordered CUDA-index tuple.

    Returns None for the single-card default (null / empty list). The user
    given order is preserved: rank i maps to ``cuda:{devices[i]}``.
    """
    if devices_cfg is None:
        return None
    devices = list(devices_cfg)
    if len(devices) == 0:
        return None
    normalized: list[int] = []
    for entry in devices:
        if isinstance(entry, bool) or not isinstance(entry, int):
            raise ValueError(
                f"training.devices entries must be integer CUDA indices, got {entry!r}"
            )
        if entry < 0:
            raise ValueError(f"training.devices entries must be non-negative, got {entry}")
        normalized.append(int(entry))
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"training.devices must not contain duplicates, got {normalized}")
    return tuple(normalized)


def validate_dp_launchable(devices: tuple[int, ...]) -> None:
    """Fail fast at launch time when the host lacks any requested CUDA device."""
    device_count = torch.cuda.device_count()
    missing = [index for index in devices if index >= device_count]
    if missing:
        raise ValueError(
            f"training.devices={list(devices)} requires CUDA device index(es) {missing}, "
            f"but torch.cuda.device_count()={device_count}"
        )


def _require_uni_rl_dp_launcher():
    """Import uni_rl's data-parallel launcher or raise an actionable error."""
    try:
        from uni_rl.ipc import dp_launcher
    except ModuleNotFoundError as exc:
        if exc.name is not None and exc.name.split(".")[0] != "uni_rl":
            raise
        raise ModuleNotFoundError(
            "Multi-GPU PPO (training.devices with more than one entry, or an "
            "external torchrun launch) requires the optional unilab-rl package; "
            "install it with: pip install unilab[uni_rl]"
        ) from exc
    return dp_launcher


def resolve_collector_cpu_ids(
    world_size: int,
    rank: int,
    cpu_count: int | None = None,
    explicit: Any = None,
) -> list[int] | None:
    """Resolve the CPU ids exclusively owned by this rank's collector.

    Single-rank runs return None without touching uni_rl; multi-rank CPU
    partitioning is owned by ``uni_rl.ipc.dp_launcher`` and requires the
    optional unilab-rl package.
    """
    if int(world_size) <= 1:
        return None
    result = _require_uni_rl_dp_launcher().resolve_collector_cpu_ids(
        world_size,
        rank,
        cpu_count,
        explicit=explicit,
    )
    return cast("list[int] | None", result)


def launch_torchrun_workers(
    devices: tuple[int, ...],
    *,
    script_path: str | os.PathLike[str],
    argv: Sequence[str],
    log_dir: str,
) -> None:
    """Launch one local torchrun worker per configured CUDA device.

    Worker supervision is owned by ``uni_rl.ipc.dp_launcher`` and requires the
    optional unilab-rl package.
    """
    _require_uni_rl_dp_launcher().launch_torchrun_workers(
        devices,
        script_path=Path(script_path),
        argv=argv,
        log_dir=log_dir,
    )


def apply_rsl_rl_rank_seed(cfg: Any, rank: int) -> int:
    """Apply RSL-RL's ``base seed + global rank`` data-parallel contract."""
    if rank < 0:
        raise ValueError(f"rank must be non-negative, got {rank}")
    base_seed = int(cfg.algo.seed)
    with open_dict(cfg):
        cfg.algo.seed = base_seed + int(rank)
    return int(cfg.algo.seed)


def resolve_rsl_rl_device(
    *,
    configured_device: str | None,
    devices: tuple[int, ...] | None,
    world_size: int,
    local_rank: int,
    default_device: str,
) -> str:
    """Resolve the exact device string expected by RSL-RL's runner.

    Integrated multi-GPU workers see the selected physical devices through a
    remapped ``CUDA_VISIBLE_DEVICES`` list, so RSL-RL must receive
    ``cuda:LOCAL_RANK`` rather than the original host-global device index.
    """
    if configured_device is not None and devices is not None:
        raise ValueError("Set either training.device or training.devices, not both")
    if world_size < 1:
        raise ValueError(f"world_size must be positive, got {world_size}")
    if local_rank < 0 or local_rank >= world_size:
        raise ValueError(f"local_rank={local_rank} is out of range for world_size={world_size}")
    if world_size > 1:
        if configured_device is not None:
            raise ValueError(
                "training.device cannot select one device in a distributed run; "
                "use training.devices"
            )
        if devices is not None and len(devices) != world_size:
            raise ValueError(
                f"training.devices has {len(devices)} entries but WORLD_SIZE={world_size}"
            )
        return f"cuda:{local_rank}"
    if devices is not None:
        return f"cuda:{devices[0]}"
    return configured_device or default_device


def ppo_samples_per_iteration(*, num_envs: int, num_steps_per_env: int, world_size: int) -> int:
    """Return the global fresh rollout sample count for one PPO iteration."""
    return int(num_envs) * int(num_steps_per_env) * int(world_size)


def finish_rsl_rl_distributed(*, training_succeeded: bool) -> None:
    """Synchronize successful ranks and release RSL-RL's process group."""
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return
    try:
        if training_succeeded:
            torch.distributed.barrier()
    finally:
        torch.distributed.destroy_process_group()


@contextmanager
def rsl_rl_single_process_topology() -> Iterator[None]:
    """Temporarily hide torchrun's worker topology from rank-0-only work.

    Destroying a process group does not clear ``WORLD_SIZE`` / ``RANK`` /
    ``LOCAL_RANK``. RSL-RL would therefore initialize a second distributed
    group when rank 0 constructs a fresh runner for post-training playback,
    even though every other rank has already exited. Present the playback
    scope as a single-process runtime, then restore launcher-owned variables.
    """
    single_process_topology = {
        "WORLD_SIZE": "1",
        "RANK": "0",
        "LOCAL_RANK": "0",
    }
    previous = {name: os.environ.get(name) for name in single_process_topology}
    os.environ.update(single_process_topology)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
