"""Optional interactive evaluation; imported only when --render is requested.

Keep policy inference, environment stepping, and metrics in the caller.  This
iterator only displays their state and paces playback using wall-clock waits.
"""

from __future__ import annotations

import copy
import math
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any


@contextmanager
def _passive_viewer(mujoco: Any, model: Any, data: Any):
    """Wait for MuJoCo's GUI thread before Python tears down native libraries."""
    before = set(threading.enumerate())
    viewer = mujoco.viewer.launch_passive(model, data)
    # In MuJoCo 3.11 close() requests exit but does not join its daemon GUI
    # thread. Capture that launch's new thread, without touching older workers.
    viewer_threads = [
        thread
        for thread in threading.enumerate()
        if thread not in before and "_launch_internal" in thread.name
    ]
    try:
        with viewer:
            yield viewer
    finally:
        for thread in viewer_threads:
            thread.join(timeout=5.0)
            if thread.is_alive():
                raise RuntimeError("MuJoCo viewer did not finish shutting down within 5 seconds")


def rendered_steps(env: Any, count: int, *, speed: float = 1.0) -> Iterator[int]:
    """Render environment zero through UniLab's physics-snapshot contract.

    The environment must already be reset.  Its owner remains responsible for
    calling close() in a finally block, including on renderer errors or Ctrl+C.
    Rendering never changes ctrl_dt, the actions, or the simulation substeps.
    """
    if not math.isfinite(speed) or speed <= 0:
        raise ValueError("Playback speed must be finite and positive")
    period = float(env.step_dt) / speed
    if not math.isfinite(period) or period <= 0:
        raise ValueError("Interactive playback requires a finite, positive step_dt")
    # MuJoCo's batch backend supports snapshots, NOT init_play_renderer().
    # Use the same playback contract as UniLab's play_interactive entrypoint.
    import glfw
    import mujoco
    import mujoco.viewer

    from unilab.visualization.playback_state import (
        PhysicsStateApplier,
        assert_physics_state_playback_supported,
    )

    assert_physics_state_playback_supported(env, entrypoint="evaluate_rma --render")
    if not glfw.init():
        raise RuntimeError(
            "Cannot open the MuJoCo viewer: no usable GLFW display. "
            "In WSL, check WSLg/DISPLAY and OpenGL support. "
            "Omit --render to continue normal headless evaluation."
        )
    glfw.terminate()
    # Neither viewer.sync() nor mouse perturbations may write into the live
    # physics model/data: both playback objects belong exclusively to the GUI.
    source_model = env.get_playback_model(env_index=0)
    if isinstance(source_model, str):
        model = (
            mujoco.MjModel.from_binary_path(source_model)
            if source_model.lower().endswith(".mjb")
            else mujoco.MjModel.from_xml_path(source_model)
        )
    else:
        model = copy.deepcopy(source_model)
    data = mujoco.MjData(model)
    applier = PhysicsStateApplier(env, model, env_index=0)
    applier.apply(env.get_physics_state_snapshot()[0], data)
    print(
        f"[render] Showing environment 0 at {speed:g}x speed; "
        "close the window or press Ctrl+C to stop.",
        file=sys.stderr,
    )
    with _passive_viewer(mujoco, model, data) as viewer:
        with viewer.lock():
            viewer.cam.distance = 2.5
            viewer.cam.elevation = -20.0
            base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
            if base_id >= 0:
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                viewer.cam.trackbodyid = base_id
        viewer.sync()
        for index in range(count):
            if not viewer.is_running():
                break
            started = time.perf_counter()
            yield index
            with viewer.lock():
                applier.apply(env.get_physics_state_snapshot()[0], data)
            viewer.sync()
            remaining = period - (time.perf_counter() - started)
            if remaining > 0:
                time.sleep(remaining)
