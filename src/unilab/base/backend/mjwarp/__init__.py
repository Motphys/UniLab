"""Independent optional backend built on :mod:`mujoco_warp`.

This package deliberately owns its own runtime implementation.  It may reuse
shared *cold-path* scene materialization helpers, but it never subclasses or
reads the runtime-private state of :mod:`unilab.base.backend.mujoco`.
"""

from __future__ import annotations

from .materialization import (
    MJWARP_MODEL_INVALIDATIONS,
    MJWARP_MODEL_MATERIALIZATION_VERSION,
    MjwarpModelFieldReceipt,
    MjwarpModelFieldRole,
    MjwarpModelInvalidationOutcome,
    MjwarpModelInvalidationReceipt,
    MjwarpModelMaterializationContractError,
    MjwarpModelMaterializationReceipt,
    MjwarpModelMaterializationRequest,
)


def __getattr__(name: str):
    if name in {"MjwarpBackend", "MjwarpDeviceCapacityDiagnostics"}:
        from .backend import MjwarpBackend, MjwarpDeviceCapacityDiagnostics

        return {
            "MjwarpBackend": MjwarpBackend,
            "MjwarpDeviceCapacityDiagnostics": MjwarpDeviceCapacityDiagnostics,
        }[name]
    if name == "MJWARP_AVAILABLE":
        from .dependencies import mjwarp_dependencies_available

        return mjwarp_dependencies_available()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "MJWARP_AVAILABLE",
    "MJWARP_MODEL_INVALIDATIONS",
    "MJWARP_MODEL_MATERIALIZATION_VERSION",
    "MjwarpBackend",
    "MjwarpDeviceCapacityDiagnostics",
    "MjwarpModelFieldReceipt",
    "MjwarpModelFieldRole",
    "MjwarpModelInvalidationOutcome",
    "MjwarpModelInvalidationReceipt",
    "MjwarpModelMaterializationContractError",
    "MjwarpModelMaterializationReceipt",
    "MjwarpModelMaterializationRequest",
]
