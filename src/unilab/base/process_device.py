"""Training-worker device routing for UniSim backends.

The learner device and the device consumed by a simulator are related, but
they do not always use the same index namespace.  In particular, the
off-policy launcher keeps the host-visible CUDA namespace while the PPO
``torchrun`` launcher remaps ``CUDA_VISIBLE_DEVICES`` and therefore exposes a
rank-local index to child processes.  The helpers in this module keep that
translation on the cold path, next to the backend process binding contract.
"""

from __future__ import annotations

import os
import warnings
from collections.abc import Mapping, Sequence
from typing import Any, cast

# These backends consume an explicit integer device id while materializing
# their simulator.  MuJoCo/Motrix/Drake either run on the host or own their
# device selection internally and must not receive a synthetic override.
BACKEND_ENV_DEVICE_FIELDS: dict[str, str] = {
    "isaacgym": "isaacgym_device_id",
    "isaacsim": "isaacsim_device_id",
    "genesis": "genesis_device_id",
}


# Set once ``bind_genesis_process_device`` has pinned CUDA_VISIBLE_DEVICES for
# this process.  Genesis/Quadrants binds its CUDA runtime to the first visible
# device regardless of torch's current device (verified on genesis_world
# 1.3.3 / Quadrants 1.3.0, issue #1508), so a non-zero request is honored by
# shrinking visibility to the target GPU and using the in-process index 0.
# The flag flips the resolution helpers into the pinned namespace.
_genesis_device_pinned = False


def _normalize_backend(backend_type: str) -> str:
    if not isinstance(backend_type, str) or not backend_type.strip():
        raise ValueError(f"backend_type must be a non-empty string, got {backend_type!r}")
    return backend_type.strip().lower()


def _normalize_device_indices(devices: Sequence[int] | None) -> tuple[int, ...] | None:
    if devices is None:
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


def _cuda_device_index(device: str | None) -> int | None:
    """Extract an integer CUDA index from a device string.

    ``cuda`` without an explicit suffix is resolved through the current CUDA
    device when possible.  This is deliberately a cold-path helper; it is not
    used from environment ``step``/``reset`` loops.
    """

    if device is None:
        return None
    value = str(device).strip().lower()
    if value == "cuda":
        try:
            import torch

            if torch.cuda.is_available():
                return int(torch.cuda.current_device())
        except Exception:
            # Device discovery is only a fallback for an unindexed alias.  A
            # configured topology below remains authoritative if available.
            pass
        return 0
    if not value.startswith("cuda:"):
        return None
    index_text = value.split(":", 1)[1].strip()
    if not index_text:
        raise ValueError(f"CUDA device alias {device!r} has an empty index")
    try:
        index = int(index_text)
    except ValueError as exc:
        raise ValueError(f"CUDA device alias {device!r} has a non-integer index") from exc
    if index < 0:
        raise ValueError(f"CUDA device alias {device!r} has a negative index")
    return index


def resolve_backend_env_device_id(
    backend_type: str,
    *,
    devices: Sequence[int] | None = None,
    rank: int = 0,
    local_rank: int | None = None,
    world_size: int = 1,
    learner_device: str | None = None,
) -> int | None:
    """Resolve the integer simulator device id for a training rank.

    Args:
        backend_type: Selected UniSim backend.
        devices: ``training.devices`` in the host-visible namespace.  This is
            used by the off-policy launcher and by single-process PPO.
        rank: Off-policy data-parallel rank (rank zero by default).
        local_rank: ``LOCAL_RANK`` from torchrun.  In a distributed PPO worker
            this is the logical index inside the launcher's remapped
            ``CUDA_VISIBLE_DEVICES`` list.
        world_size: Torchrun world size.  Values greater than one select the
            ``local_rank`` namespace; values of one select ``devices[rank]``.
        learner_device: Explicit learner device fallback when no topology was
            configured (for example APPO or a single-device play command).

    Returns ``None`` for backends without an explicit simulator device field.
    For a torchrun worker the returned id is intentionally *local* (rather
    than the host index in ``devices``), because the worker subprocess inherits
    the remapped ``CUDA_VISIBLE_DEVICES`` environment.
    """

    backend = _normalize_backend(backend_type)
    field = BACKEND_ENV_DEVICE_FIELDS.get(backend)
    if field is None:
        return None

    if backend == "genesis" and _genesis_device_pinned:
        # Post-pin the process sees exactly one CUDA device; every rank-local
        # consumer (learner probe, spawn collector, playback) must use the
        # in-process index 0 regardless of the original host/rank topology.
        return 0

    normalized_devices = _normalize_device_indices(devices)
    world_size = int(world_size)
    if world_size < 1:
        raise ValueError(f"world_size must be positive, got {world_size}")

    if world_size > 1:
        resolved_local_rank = int(rank if local_rank is None else local_rank)
        if resolved_local_rank < 0 or resolved_local_rank >= world_size:
            raise ValueError(
                f"local_rank={resolved_local_rank} is out of range for world_size={world_size}"
            )
        if normalized_devices is not None and len(normalized_devices) != world_size:
            raise ValueError(
                f"training.devices has {len(normalized_devices)} entries but "
                f"WORLD_SIZE={world_size}"
            )
        # torchrun launch_torchrun_workers remaps CVD to the selected physical
        # devices.  Isaac workers inherit that environment, so LOCAL_RANK is
        # the correct payload index.
        return resolved_local_rank

    if normalized_devices:
        resolved_rank = int(rank)
        if resolved_rank < 0 or resolved_rank >= len(normalized_devices):
            raise ValueError(
                f"rank={resolved_rank} is out of range for training.devices="
                f"{list(normalized_devices)}"
            )
        return normalized_devices[resolved_rank]

    return _cuda_device_index(learner_device)


def apply_backend_env_device_override(
    env_cfg_override: Mapping[str, Any] | None,
    backend_type: str,
    *,
    devices: Sequence[int] | None = None,
    rank: int = 0,
    local_rank: int | None = None,
    world_size: int = 1,
    learner_device: str | None = None,
) -> dict[str, Any]:
    """Return an env override carrying the rank-selected simulator device.

    The input mapping is never mutated.  If no topology/device can be
    resolved, the owner-configured value is preserved.  This lets one helper
    serve training, playback, and custom entrypoints while retaining the
    historical default (device zero) for single-process calls.
    """

    result = dict(env_cfg_override) if env_cfg_override is not None else {}
    backend = _normalize_backend(backend_type)
    field = BACKEND_ENV_DEVICE_FIELDS.get(backend)
    if field is None:
        return result
    device_id = resolve_backend_env_device_id(
        backend,
        devices=devices,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        learner_device=learner_device,
    )
    if device_id is not None:
        result[field] = int(device_id)
    return result


def warn_if_backend_device_collision(
    backend_type: str,
    *,
    devices: Sequence[int] | None,
    rank: int,
    device_id: int | None,
    source: str = "environment",
) -> None:
    """Warn when a multi-rank simulator still resolves to device zero.

    This is a transition guard for older adapters/configuration paths.  Rank
    zero legitimately owns device zero; only a non-zero rank resolving to zero
    is a collision.  The warning is intentionally emitted at construction
    time, never from a hot simulation path.
    """

    backend = _normalize_backend(backend_type)
    if backend not in BACKEND_ENV_DEVICE_FIELDS:
        return
    if backend == "genesis" and _genesis_device_pinned:
        # A successful pin places every rank on its own physical GPU; the
        # in-process index 0 that follows is not a collision.
        return
    normalized_devices = _normalize_device_indices(devices)
    if (
        normalized_devices is None
        or len(normalized_devices) <= 1
        or int(rank) <= 0
        or device_id != 0
    ):
        return
    warnings.warn(
        f"{backend} rank {int(rank)} resolved its {source} device to 0 while "
        f"training.devices={list(normalized_devices)} requests multiple devices; "
        "all simulator workers may be sharing GPU 0",
        RuntimeWarning,
        stacklevel=2,
    )


def resolve_backend_process_device(backend_type: str, learner_device: str | None) -> str | None:
    backend = _normalize_backend(backend_type)
    if backend not in {"mjwarp", "newton", "genesis"}:
        return None
    if learner_device is None:
        raise ValueError(f"{backend} requires an explicit CUDA process device")
    resolved = str(learner_device).strip()
    if resolved.split(":", 1)[0].lower() != "cuda":
        raise ValueError(
            f"{backend} requires a CUDA process device shared with its learner; got {resolved!r}"
        )
    return resolved


def configure_backend_process_device(backend_type: str, learner_device: str | None) -> str | None:
    resolved = resolve_backend_process_device(backend_type, learner_device)
    if resolved is None:
        return None
    if _normalize_backend(backend_type) == "genesis":
        return bind_genesis_process_device(resolved)
    return bind_backend_process_device_for_backend(backend_type, resolved)


def bind_backend_process_device_for_backend(backend_type: str, resolved: str) -> str | None:
    """Bind one backend's process-global accelerator device.

    This top-level callable is intentionally backend-aware and lazy.  It can
    be wrapped with :func:`functools.partial` and injected into uni_rl's
    spawn-based collectors while remaining pickleable by module reference.
    """
    backend = _normalize_backend(backend_type)
    if backend == "newton":
        from unisim.backend.newton.runtime import bind_newton_process_device

        return cast(str | None, bind_newton_process_device(resolved))
    if backend == "mjwarp":
        from unisim.backend.mjwarp.runtime import bind_mjwarp_process_device

        return cast(str | None, bind_mjwarp_process_device(resolved))
    return None


def bind_backend_process_device(resolved: str) -> str | None:
    """Bind a resolved backend process device in the current process.

    Top-level on purpose: uni_rl's off-policy collectors receive this as the
    injected ``backend_device_binder`` and pickle it by reference into
    spawn-based subprocesses. The mjwarp import stays lazy so the binder is
    importable without the ``mjwarp`` extra installed.
    """
    return bind_backend_process_device_for_backend("mjwarp", resolved)


def _pin_cuda_visible_devices(index: int) -> None:
    """Shrink ``CUDA_VISIBLE_DEVICES`` to the single entry at ``index``.

    The index addresses the *current* visibility namespace: with no variable
    set it is the host index, otherwise it indexes into the existing entries
    (which may be physical indices or UUIDs).  This only works before the
    first CUDA context exists, so an already-initialized torch runtime fails
    closed with an actionable error instead of crashing inside the engine.
    """

    import torch

    if torch.cuda.is_initialized():
        raise RuntimeError(
            "genesis device routing must pin CUDA_VISIBLE_DEVICES before any CUDA "
            "context is created in this process, but torch CUDA is already "
            "initialized; move configure_backend_process_device earlier in the "
            "entrypoint (before seeding/learner construction)"
        )
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    entries = [entry.strip() for entry in raw.split(",") if entry.strip()] if raw else None
    if entries is None:
        count = int(torch.cuda.device_count())
        if index >= count:
            raise ValueError(
                f"genesis device index {index} is out of range; torch.cuda.device_count()={count}"
            )
        target = str(index)
    else:
        if index >= len(entries):
            raise ValueError(
                f"genesis device index {index} is out of range for "
                f"CUDA_VISIBLE_DEVICES={raw!r} ({len(entries)} entr(ies))"
            )
        target = entries[index]
    os.environ["CUDA_VISIBLE_DEVICES"] = target
    # ``device_count`` is lru-cached; drop any pre-pin host-wide count so
    # later callers observe the pinned single-device namespace.
    cache_clear = getattr(torch.cuda.device_count, "cache_clear", None)
    if callable(cache_clear):
        cache_clear()


def pin_genesis_device_before_cuda_init(
    backend_type: str,
    *,
    devices: Sequence[int] | None = None,
    rank: int = 0,
    local_rank: int | None = None,
    world_size: int = 1,
    learner_device: str | None = None,
) -> str | None:
    """Pin Genesis to its rank device before the first torch CUDA call.

    ``torch.cuda.is_available()`` already latches ``CUDA_VISIBLE_DEVICES`` in
    the CUDA runtime, so the pin must run ahead of *any* torch CUDA query —
    entrypoints should call this before registry/bootstrap/device detection.
    Pure config topology (``training.devices`` / ``LOCAL_RANK`` / an explicit
    ``cuda:N`` learner device) resolves without touching torch.  Returns the
    in-process device the caller must use when a pin happened, else ``None``.
    """

    if _normalize_backend(backend_type) != "genesis":
        return None
    device_id = resolve_backend_env_device_id(
        backend_type,
        devices=devices,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        learner_device=learner_device,
    )
    if not device_id:
        return None
    return bind_genesis_process_device(f"cuda:{device_id}")


def bind_genesis_process_device(resolved: str) -> str:
    """Select the CUDA device used by an in-process Genesis worker.

    Genesis initializes a process-wide session whose Quadrants CUDA runtime
    always binds the first entry of ``CUDA_VISIBLE_DEVICES``; torch's current
    device alone is *not* honored (issue #1508).  A non-zero request is
    therefore honored by pinning visibility to the target GPU before any CUDA
    context exists, after which the in-process device is ``cuda:0``.  The
    returned string is the device the rest of this process must actually use —
    callers that computed a pre-pin device (learner, probes, collectors) have
    to adopt the returned value.  Binding must happen before constructing the
    backend (and before ``gs.init``), including in spawn-based collector
    processes, and remains in effect for the lifetime of the process.
    """

    global _genesis_device_pinned
    device = str(resolved).strip()
    index = _cuda_device_index(device)
    if index is None:
        raise ValueError(f"genesis requires a CUDA process device; got {resolved!r}")
    import torch

    # Pin *before* any torch CUDA query: even ``torch.cuda.is_available()``
    # latches CUDA_VISIBLE_DEVICES in the runtime, after which rewriting it
    # would silently keep the process on the first previously visible GPU.
    if index > 0:
        if not _genesis_device_pinned:
            _pin_cuda_visible_devices(index)
            _genesis_device_pinned = True
        # Already pinned (or just pinned): the only valid in-process device is
        # index 0.  A stale pre-pin index from the same rank maps onto it.
        index = 0
    if not torch.cuda.is_available():
        raise ValueError(
            f"genesis requires CUDA device {device!r}, but CUDA is unavailable in this process"
        )
    torch.cuda.set_device(index)
    return f"cuda:{index}"


def _reset_genesis_device_pin_for_tests() -> None:
    """Clear the process pin latch; test-only seam (the CVD rewrite itself is
    reverted via ``monkeypatch.setitem``/``delitem`` on ``os.environ``)."""

    global _genesis_device_pinned
    _genesis_device_pinned = False


__all__ = [
    "BACKEND_ENV_DEVICE_FIELDS",
    "apply_backend_env_device_override",
    "bind_backend_process_device",
    "bind_backend_process_device_for_backend",
    "bind_genesis_process_device",
    "configure_backend_process_device",
    "pin_genesis_device_before_cuda_init",
    "resolve_backend_env_device_id",
    "resolve_backend_process_device",
    "warn_if_backend_device_collision",
]
