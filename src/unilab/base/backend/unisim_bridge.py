"""UniSim-core consumer boundary used during roadmap #1428 migration.

This module is intentionally tiny: UniLab owns scene/task configuration while
``unisim-core`` owns the backend contract and engine adapters.  Existing
``unilab.base.backend`` imports remain a compatibility surface until Child 12;
new consumers should import this bridge (or ``unisim`` directly) so the
remaining legacy implementation can be removed without another API rename.
"""

from __future__ import annotations

from typing import Any

import unisim

from unilab.base.scene import SceneCfg


def create_unisim_backend(
    backend_type: str,
    scene: SceneCfg,
    num_envs: int,
    sim_dt: float,
    **kwargs: Any,
) -> unisim.SimBackend:
    """Construct a package-owned backend from a UniLab scene descriptor.

    Scene composition and robot-asset fetching must happen before this call on
    the UniLab cold path.  The package receives only the materialized model
    path and backend-neutral numeric options; missing optional SDKs fail closed
    in :mod:`unisim` with an actionable diagnostic.
    """
    if scene is None:
        raise ValueError("SceneCfg must be provided")
    model_path = scene.model_file
    if not model_path:
        raise ValueError("SceneCfg.model_file must be provided for UniSim backends")
    options = dict(kwargs)
    if backend_type != "fake":
        options.setdefault("model_path", model_path)
    options.setdefault("num_envs", num_envs)
    if backend_type != "fake":
        options.setdefault("frame_skip", 1)
    return unisim.create_backend(backend_type, **options)


SimBackend = unisim.SimBackend
BackendCapability = unisim.BackendCapability
BackendError = unisim.BackendError
UnsupportedCapabilityError = unisim.UnsupportedCapabilityError

__all__ = [
    "BackendCapability",
    "BackendError",
    "SimBackend",
    "UnsupportedCapabilityError",
    "create_unisim_backend",
]
