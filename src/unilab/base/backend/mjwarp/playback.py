"""Cold-path MuJoCo offline playback bridge for ``mjwarp``."""

from __future__ import annotations

from collections.abc import Callable
from os import PathLike
from pathlib import Path
from typing import Any, TypeVar

import numpy as np

from unilab.base.backend.mujoco.playback import run_mujoco_playback

ObsT = TypeVar("ObsT")


def validate_mjwarp_visual_model(
    *,
    mujoco: Any,
    physics_model: Any,
    model_file: str | PathLike[str],
) -> str:
    """Validate the detached MuJoCo visual twin used for offline playback."""
    path = Path(model_file)
    if not path.is_file():
        raise ValueError(
            f"mjwarp offline playback visual model does not exist or is not a file: {path}"
        )
    try:
        visual_model = mujoco.MjModel.from_xml_path(str(path))
    except Exception as exc:
        raise ValueError(
            f"mjwarp offline playback could not load visual model {path}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    physics_dims = (int(physics_model.nq), int(physics_model.nv))
    visual_dims = (int(visual_model.nq), int(visual_model.nv))
    if visual_dims != physics_dims:
        raise ValueError(
            "mjwarp offline playback visual model state dimensions are incompatible: "
            f"physics nq/nv={physics_dims}, visual nq/nv={visual_dims}."
        )

    joint_object = mujoco.mjtObj.mjOBJ_JOINT

    def _joint_layout(model: Any) -> tuple[tuple[str | None, int, int, int], ...]:
        return tuple(
            (
                mujoco.mj_id2name(model, joint_object, joint_id),
                int(model.jnt_type[joint_id]),
                int(model.jnt_qposadr[joint_id]),
                int(model.jnt_dofadr[joint_id]),
            )
            for joint_id in range(int(model.njnt))
        )

    physics_layout = _joint_layout(physics_model)
    visual_layout = _joint_layout(visual_model)
    if visual_layout != physics_layout:
        raise ValueError(
            "mjwarp offline playback visual model joint layout is incompatible; "
            "joint names, types, qpos addresses, and dof addresses must match physics."
        )
    return str(path)


def run_mjwarp_playback(
    *,
    backend: Any,
    env: Any,
    initialize: Callable[[], ObsT],
    step: Callable[[ObsT], ObsT],
    num_steps: int | None,
    output_video: str | PathLike[str] | None,
    render_spacing: float | None,
    headless: bool,
    record_video: bool,
    snapshot_shape: tuple[int, int],
    frame_state_getter: Callable[[], np.ndarray] | None,
    camera_kwargs: dict[str, Any] | None,
    extra_data_getter: Callable[[], np.ndarray | None] | None = None,
) -> str:
    """Render detached mjwarp host snapshots with the existing MuJoCo pipeline."""
    if not headless:
        raise NotImplementedError(
            "mjwarp offline playback does not support interactive rendering; "
            "use training.play_render_mode=record."
        )
    if not record_video:
        raise ValueError("mjwarp offline playback requires record_video=true.")
    if isinstance(num_steps, bool) or num_steps is None or int(num_steps) <= 0:
        raise ValueError("mjwarp record playback requires a positive finite num_steps value.")
    if output_video is None:
        raise ValueError("mjwarp record playback requires an output_video path.")

    # Both checks are playback-only cold-path work. Physics construction and
    # step/reset never import the renderer or parse the visual model.
    backend.get_playback_model()
    try:
        from unilab.visualization import render_many

        renderer_usable = bool(render_many.render_backend_usable())
    except Exception as exc:
        raise RuntimeError(
            "mjwarp offline playback could not initialize the MuJoCo renderer: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    if not renderer_usable:
        raise RuntimeError(
            "mjwarp offline playback requires a usable MuJoCo off-screen renderer; "
            "configure EGL, OSMesa, or GLFW before recording."
        )

    getter = frame_state_getter or env.get_physics_state_snapshot
    expected_shape = snapshot_shape

    def _validated_state_getter() -> np.ndarray:
        state = np.asarray(getter(), dtype=np.float32)
        if state.shape != expected_shape:
            raise ValueError(
                "mjwarp offline playback snapshot must use [time, qpos, qvel] layout "
                f"with shape {expected_shape}, got {state.shape}."
            )
        return state

    result = run_mujoco_playback(
        env=env,
        initialize=initialize,
        step=step,
        num_steps=int(num_steps),
        output_video=output_video,
        render_spacing=render_spacing,
        headless=True,
        record_video=True,
        frame_state_getter=_validated_state_getter,
        camera_kwargs=camera_kwargs,
        extra_data_getter=extra_data_getter,
    )
    if result is None:
        raise RuntimeError(
            "mjwarp offline playback produced no frames; the MuJoCo renderer or worker "
            "failed after preflight."
        )
    return result
