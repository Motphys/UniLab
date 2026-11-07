"""Interactive, checkpoint-free robot-arm demonstrations."""

# pyright: reportAttributeAccessIssue=false

from __future__ import annotations

import math
import time
from pathlib import Path

from unilab.assets import ASSETS_ROOT_PATH

ROBOT_ARM_DEMOS: dict[str, str] = {
    "a1-left-arm": "robots/onerobotics_a1/mjcf/a1_left.xml",
    "a1-right-arm": "robots/onerobotics_a1/mjcf/a1_right.xml",
    "a1-bimanual": "robots/onerobotics_a1/mjcf/a1_bimanual_stand.xml",
}


def robot_model_path(demo_name: str) -> Path:
    """Resolve the MJCF owned by a registered robot-arm demo."""
    try:
        relative_path = ROBOT_ARM_DEMOS[demo_name]
    except KeyError as exc:
        available = ", ".join(sorted(ROBOT_ARM_DEMOS))
        raise ValueError(f"Unknown robot-arm demo {demo_name!r}. Available: {available}") from exc
    path = ASSETS_ROOT_PATH / relative_path
    if not path.is_file():
        raise FileNotFoundError(f"Robot-arm demo model not found: {path}")
    return path


def _animated_joint_ids(model) -> list[int]:
    import mujoco

    free_joint = int(mujoco.mjtJoint.mjJNT_FREE)
    return [
        joint_id for joint_id in range(model.njnt) if int(model.jnt_type[joint_id]) != free_joint
    ]


def _joint_center_and_amplitude(model, joint_id: int) -> tuple[float, float]:
    limited = bool(model.jnt_limited[joint_id])
    if not limited:
        return 0.0, 0.35
    lower, upper = (float(value) for value in model.jnt_range[joint_id])
    center = min(max(0.0, lower), upper)
    return center, min(0.35, 0.2 * (upper - lower))


def run_robot_arm_demo(demo_name: str) -> int:
    """Open a MuJoCo viewer and animate the selected A1 arm model."""
    try:
        import mujoco
        import mujoco.viewer
    except ImportError as exc:
        raise SystemExit(
            "Robot-arm demos require the MuJoCo extra. Install it with `uv sync --extra mujoco`."
        ) from exc

    model = mujoco.MjModel.from_xml_path(str(robot_model_path(demo_name)))
    data = mujoco.MjData(model)
    animated_joints = _animated_joint_ids(model)
    trajectories = [_joint_center_and_amplitude(model, joint_id) for joint_id in animated_joints]

    # Keep the imported floating base at a useful viewing height. The demo is
    # deliberately kinematic, so it exercises loading and articulation without
    # requiring a controller or trained checkpoint.
    if model.nq >= 7:
        data.qpos[:7] = [0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0]
    start = time.monotonic()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = [0.0, 0.0, 0.35]
        viewer.cam.distance = 1.6 if demo_name == "a1-bimanual" else 1.1
        viewer.cam.azimuth = 135.0
        viewer.cam.elevation = -20.0
        try:
            while viewer.is_running():
                elapsed = time.monotonic() - start
                for index, (joint_id, (center, amplitude)) in enumerate(
                    zip(animated_joints, trajectories, strict=True)
                ):
                    qpos_address = int(model.jnt_qposadr[joint_id])
                    phase = 0.55 * elapsed + 0.45 * index
                    data.qpos[qpos_address] = center + amplitude * math.sin(phase)
                mujoco.mj_forward(model, data)
                viewer.sync()
                time.sleep(1.0 / 60.0)
        except KeyboardInterrupt:
            pass
    return 0
