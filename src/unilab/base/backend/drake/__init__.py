from __future__ import annotations

from .pool import DrakeEnvPool, DrakePoolOutput

__all__ = ["DRAKE_AVAILABLE", "DrakeBackend", "DrakeEnvPool", "DrakePoolOutput"]


def __getattr__(name: str):
    if name in {"DRAKE_AVAILABLE", "DrakeBackend"}:
        from .backend import DRAKE_AVAILABLE, DrakeBackend

        values = {
            "DRAKE_AVAILABLE": DRAKE_AVAILABLE,
            "DrakeBackend": DrakeBackend,
        }
        return values[name]
    raise AttributeError(name)
