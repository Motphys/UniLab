"""Playback rendering compatibility entrypoint."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, TypeVar, cast

import numpy as np

from unilab.base.backend.mujoco.playback import (
    materialize_visual_playback_model as _materialize_visual_playback_model,
)
from unilab.base.backend.mujoco.playback import (
    resolve_render_play_model_files as _resolve_render_play_model_files,
)

ObsT = TypeVar("ObsT")


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
    camera_kwargs: dict[str, Any] | None = None,
    extra_data_getter: Callable[[], np.ndarray | None] | None = None,
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
            extra_data_getter=extra_data_getter,
        ),
    )


__all__ = [
    "render_play_mode",
    "_materialize_visual_playback_model",
    "_resolve_render_play_model_files",
]
