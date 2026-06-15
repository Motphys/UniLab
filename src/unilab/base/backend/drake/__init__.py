from __future__ import annotations

__all__ = [
    "DRAKE_AVAILABLE",
    "DRAKE_BATCH_AVAILABLE",
    "DrakeBackend",
    "run_drake_playback",
]


def __getattr__(name: str):
    if name in {"DRAKE_AVAILABLE", "DRAKE_BATCH_AVAILABLE", "DrakeBackend"}:
        from .backend import (
            DRAKE_AVAILABLE,
            DRAKE_BATCH_AVAILABLE,
            DrakeBackend,
        )

        values = {
            "DRAKE_AVAILABLE": DRAKE_AVAILABLE,
            "DRAKE_BATCH_AVAILABLE": DRAKE_BATCH_AVAILABLE,
            "DrakeBackend": DrakeBackend,
        }
        return values[name]
    if name == "run_drake_playback":
        from .playback import run_drake_playback

        return run_drake_playback
    raise AttributeError(name)
