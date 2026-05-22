"""MuJoCo backend package."""

from __future__ import annotations


def __getattr__(name: str):
    if name == "MuJoCoBackend":
        from .backend import MuJoCoBackend

        return MuJoCoBackend
    if name in {
        "materialize_visual_playback_model",
        "resolve_render_play_model_files",
        "run_mujoco_playback",
    }:
        from . import playback

        return getattr(playback, name)
    if name in {
        "add_sensor",
        "create_discardvisual_xml",
        "get_named_body_ids",
        "inject_mujoco_tracking_sensors",
        "materialize_mujoco_hfield_attached_scene",
        "materialize_scene_fragments",
        "materialize_scene_visual_override",
        "processed_xml",
    }:
        from . import xml

        return getattr(xml, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "MuJoCoBackend",
    "add_sensor",
    "create_discardvisual_xml",
    "get_named_body_ids",
    "inject_mujoco_tracking_sensors",
    "materialize_visual_playback_model",
    "materialize_mujoco_hfield_attached_scene",
    "resolve_render_play_model_files",
    "materialize_scene_fragments",
    "materialize_scene_visual_override",
    "processed_xml",
    "run_mujoco_playback",
]
