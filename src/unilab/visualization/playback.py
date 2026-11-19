"""Playback rendering entrypoint for trained policies."""

from __future__ import annotations

from typing import Any

from unisim.backend.base import CameraCfg


def camera_cfg_from_training(training_cfg: Any) -> CameraCfg:
    """Assemble the typed playback camera config from a training owner config.

    Hydra ``training.cam_*`` fields keep their names; this helper is the single
    normalization point so unknown keys fail closed at the unisim boundary
    instead of being silently ignored.
    """
    return CameraCfg.from_kwargs(
        {
            "cam_distance": training_cfg.cam_distance,
            "cam_elevation": training_cfg.cam_elevation,
            "cam_azimuth": training_cfg.cam_azimuth,
            "cam_lookat": getattr(training_cfg, "cam_lookat", None),
            "cam_tracking": getattr(training_cfg, "cam_tracking", False),
            "cam_tracking_env_idx": getattr(training_cfg, "cam_tracking_env_idx", 0),
            "cam_tracking_extra_envs": getattr(training_cfg, "cam_tracking_extra_envs", 2),
            "cam_fov": getattr(training_cfg, "cam_fov", None),
        }
    )


__all__ = [
    "camera_cfg_from_training",
]
