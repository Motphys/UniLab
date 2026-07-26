"""Drake-owned playback execution helpers."""

from __future__ import annotations

from os import PathLike
from typing import Any, Callable, TypeVar

import numpy as np

from unilab.base.backend.mujoco.playback import run_mujoco_playback

ObsT = TypeVar("ObsT")


def run_drake_playback(
    *,
    env: Any,
    initialize: Callable[[], ObsT],
    step: Callable[[ObsT], ObsT],
    num_steps: int | None,
    output_video: str | PathLike[str] | None,
    render_spacing: float | None,
    render_offset_mode: str | None,
    headless: bool,
    record_video: bool,
    frame_state_getter: Callable[[], np.ndarray] | None,
    camera_kwargs: dict[str, Any] | None,
    extra_data_getter: Callable[[], np.ndarray | None] | None = None,
) -> str | None:
    """Run Drake physics playback and optionally render it with MuJoCo.

    Drake owns the rollout: ``step`` must advance the Drake backend and
    ``frame_state_getter`` must read Drake state snapshots. MuJoCo is used only
    as an offline visual renderer for the recorded state sequence.
    """
    del render_offset_mode
    if record_video:
        return run_mujoco_playback(
            env=env,
            initialize=initialize,
            step=step,
            num_steps=num_steps,
            output_video=output_video,
            render_spacing=render_spacing,
            headless=True,
            record_video=True,
            frame_state_getter=frame_state_getter,
            camera_kwargs=camera_kwargs,
            extra_data_getter=extra_data_getter,
        )
    if not headless:
        raise NotImplementedError("Drake playback does not support interactive rendering yet.")

    obs = initialize()
    steps_run = 0
    while num_steps is None or steps_run < int(num_steps):
        obs = step(obs)
        steps_run += 1
    return None


__all__ = ["run_drake_playback"]
