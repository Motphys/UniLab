"""Playback rendering compatibility entrypoint."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, TypeVar, cast

import numpy as np
from unisim.backend.base import CameraCfg, DebugOverlayGetter
from unisim.backend.mujoco.playback import (
    materialize_visual_playback_model as _materialize_visual_playback_model,
)
from unisim.backend.mujoco.playback import (
    resolve_render_play_model_files as _resolve_render_play_model_files,
)

ObsT = TypeVar("ObsT")


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


def render_play_mode(
    env,
    *,
    sim_backend: str,
    initialize: Callable[[], ObsT],
    step: Callable[[ObsT], ObsT],
    num_steps: int | None,
    output_video: str | Path | None = None,
    render_spacing: float | None = None,
    render_offset_mode: str | None = None,
    headless: bool | None = None,
    record_video: bool | None = None,
    frame_state_getter: Callable[[], np.ndarray] | None = None,
    camera_kwargs: CameraCfg | dict[str, Any] | None = None,
    debug_overlay_getter: DebugOverlayGetter | None = None,
    on_frame: Callable[[int, np.ndarray], np.ndarray | None] | None = None,
) -> str | None:
    """Run playback through the env/backend playback contract.

    ``sim_backend`` is retained for older call sites; backend selection now
    belongs to ``env.run_playback()`` and concrete backend implementations.
    """
    del sim_backend
    return cast(
        str | None,
        env.run_playback(
            initialize=initialize,
            step=step,
            num_steps=num_steps,
            output_video=output_video,
            render_spacing=render_spacing,
            render_offset_mode=render_offset_mode,
            headless=headless,
            record_video=record_video,
            frame_state_getter=frame_state_getter,
            camera_kwargs=camera_kwargs,
            debug_overlay_getter=debug_overlay_getter,
            on_frame=on_frame,
        ),
    )


__all__ = [
    "camera_cfg_from_training",
    "render_play_mode",
    "_materialize_visual_playback_model",
    "_resolve_render_play_model_files",
]
