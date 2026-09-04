"""Training-worker device routing for UniSim backends."""

from __future__ import annotations

from typing import cast


def resolve_backend_process_device(backend_type: str, learner_device: str | None) -> str | None:
    if backend_type != "mjwarp":
        return None
    if learner_device is None:
        raise ValueError("mjwarp requires an explicit CUDA process device")
    resolved = str(learner_device).strip()
    if resolved.split(":", 1)[0].lower() != "cuda":
        raise ValueError(
            f"mjwarp requires a CUDA process device shared with its learner; got {resolved!r}"
        )
    return resolved


def configure_backend_process_device(backend_type: str, learner_device: str | None) -> str | None:
    resolved = resolve_backend_process_device(backend_type, learner_device)
    if resolved is None:
        return None
    return bind_backend_process_device(resolved)


def bind_backend_process_device(resolved: str) -> str | None:
    """Bind a resolved backend process device in the current process.

    Top-level on purpose: uni_rl's off-policy collectors receive this as the
    injected ``backend_device_binder`` and pickle it by reference into
    spawn-based subprocesses. The mjwarp import stays lazy so the binder is
    importable without the ``mjwarp`` extra installed.
    """
    from unisim.backend.mjwarp.runtime import bind_mjwarp_process_device

    return cast(str | None, bind_mjwarp_process_device(resolved))


__all__ = [
    "bind_backend_process_device",
    "configure_backend_process_device",
    "resolve_backend_process_device",
]
