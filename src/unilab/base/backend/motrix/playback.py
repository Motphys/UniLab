"""Motrix-owned playback execution helpers."""

from __future__ import annotations

import time
from os import PathLike
from typing import Any, Callable, TypeVar

import numpy as np

from unilab.base.backend.playback_common import env_cfg_value

ObsT = TypeVar("ObsT")


def run_motrix_playback(
    *,
    backend: Any,
    env: Any,
    initialize: Callable[[], ObsT],
    step: Callable[[ObsT], ObsT],
    num_steps: int | None,
    output_video: str | PathLike[str] | None,
    render_spacing: float | None,
    render_offset_mode: str | None,
    headless: bool,
    record_video: bool,
    camera_kwargs: dict[str, Any] | None,
) -> str | None:
    if record_video and not headless:
        raise ValueError("Motrix video recording requires headless=true.")

    if headless or record_video:
        if num_steps is None:
            raise ValueError("Motrix captured playback requires a finite num_steps value.")
        if record_video and output_video is None:
            raise ValueError("Motrix video recording requires an output_video path.")

        effective_spacing = (
            float(render_spacing)
            if render_spacing is not None
            else float(env_cfg_value(env, "render_spacing", 1.0))
        )
        backend.init_renderer(
            spacing=effective_spacing,
            offset_mode=str(render_offset_mode) if render_offset_mode is not None else "grid",
            headless=headless,
            capture=True,
            width=1280,
            height=720,
            camera_kwargs=dict(camera_kwargs or {}),
        )

        obs = initialize()
        frames: list[np.ndarray] | None = [] if record_video else None
        for _ in range(num_steps):
            obs = step(obs)
            frame = np.asarray(backend.capture_video_frame(), dtype=np.uint8)
            if frames is not None:
                frames.append(frame.copy())

        if not record_video:
            return None

        assert output_video is not None
        assert frames is not None
        import mediapy as media

        ctrl_dt = float(env_cfg_value(env, "ctrl_dt", 1.0 / 60.0))
        media.write_video(str(output_video), frames, fps=int(1.0 / ctrl_dt))
        return str(output_video)

    effective_spacing = (
        float(render_spacing)
        if render_spacing is not None
        else float(env_cfg_value(env, "render_spacing", 1.0))
    )
    backend.init_renderer(
        spacing=effective_spacing,
        offset_mode=str(render_offset_mode) if render_offset_mode is not None else "grid",
    )
    obs = initialize()
    last_render_time = time.perf_counter()
    render_dt = 1.0 / 60.0
    steps_run = 0

    while num_steps is None or steps_run < num_steps:
        obs = step(obs)
        current_time = time.perf_counter()
        elapsed = current_time - last_render_time
        if elapsed < render_dt:
            time.sleep(render_dt - elapsed)
        last_render_time = time.perf_counter()
        backend.render()
        steps_run += 1
    return None
