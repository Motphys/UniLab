"""Process-global runtime setup owned by the ``mjwarp`` backend."""

from __future__ import annotations

from .dependencies import load_mjwarp_dependencies


def bind_mjwarp_process_device(device: str) -> str:
    """Make one CUDA device Warp's default/current device for this process."""
    dependencies = load_mjwarp_dependencies()
    dependencies.warp.set_device(device)
    selected = dependencies.warp.get_device()
    if not bool(selected.is_cuda):
        raise RuntimeError(
            "mjwarp backend requires an active CUDA Warp device; "
            f"resolved {selected!s} from {device!r}"
        )
    return str(selected)
