from __future__ import annotations

from .pool import DrakeEnvPool, DrakePoolOutput

__all__ = [
    "DRAKE_AVAILABLE",
    "DRAKE_NATIVE_AVAILABLE",
    "DrakeBackend",
    "DrakeEnvPool",
    "DrakePoolOutput",
    "NativeDrakeBackend",
]


def __getattr__(name: str):
    if name in {"DRAKE_AVAILABLE", "DRAKE_NATIVE_AVAILABLE", "DrakeBackend", "NativeDrakeBackend"}:
        from .backend import (
            DRAKE_AVAILABLE,
            DRAKE_NATIVE_AVAILABLE,
            DrakeBackend,
            NativeDrakeBackend,
        )

        values = {
            "DRAKE_AVAILABLE": DRAKE_AVAILABLE,
            "DRAKE_NATIVE_AVAILABLE": DRAKE_NATIVE_AVAILABLE,
            "DrakeBackend": DrakeBackend,
            "NativeDrakeBackend": NativeDrakeBackend,
        }
        return values[name]
    raise AttributeError(name)
