"""Motrix backend package."""

from __future__ import annotations


def __getattr__(name: str):
    if name in {"MOTRIX_AVAILABLE", "MotrixBackend"}:
        from .backend import MOTRIX_AVAILABLE, MotrixBackend

        if name == "MOTRIX_AVAILABLE":
            return MOTRIX_AVAILABLE
        return MotrixBackend
    if name in {
        "add_motrix_tracking_frame_sensors",
        "materialize_motrix_hfield_attached_scene",
        "materialize_motrix_scene",
    }:
        from . import scene

        return getattr(scene, name)
    if name == "run_motrix_playback":
        from .playback import run_motrix_playback

        return run_motrix_playback
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "MOTRIX_AVAILABLE",
    "MotrixBackend",
    "add_motrix_tracking_frame_sensors",
    "materialize_motrix_hfield_attached_scene",
    "materialize_motrix_scene",
    "run_motrix_playback",
]
