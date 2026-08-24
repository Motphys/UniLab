"""Cold-path process-device routing for simulation backends.

The collector process must select any process-global accelerator device before
materializing its environment.  Backend-specific selection stays behind this
adapter so algorithm workers do not import or probe optional backend runtimes.
"""

from __future__ import annotations


def resolve_backend_process_device(
    backend_type: str,
    learner_device: str | None,
) -> str | None:
    """Return the backend device that must follow one rank's learner device.

    Host MuJoCo and the other adapters do not need a runner-owned process
    binding.  ``mjwarp`` allocates through Warp's process-global default
    device, so its collector must receive the exact CUDA device already
    assigned to the rank's learner.
    """
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


def configure_backend_process_device(
    backend_type: str,
    learner_device: str | None,
) -> str | None:
    """Bind a backend runtime to its resolved device before env materialization."""
    resolved = resolve_backend_process_device(backend_type, learner_device)
    if resolved is None:
        return None

    from .mjwarp.runtime import bind_mjwarp_process_device

    return bind_mjwarp_process_device(resolved)
