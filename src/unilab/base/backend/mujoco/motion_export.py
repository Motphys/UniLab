"""MuJoCo-only forward-kinematics export for motion-tracking NPZ conversion.

Cold-path tooling shared by the ``scripts/motion/`` CSV-to-NPZ converters. It
injects ``track_*`` sensors into a model, replays a (root + named joints)
trajectory through ``mj_forward``, and reads back joint/body states with the
same ``track_*``-sensor-first semantics the training backend uses.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from unilab.base.backend.mujoco.xml import inject_mujoco_tracking_sensors

_SENSOR_DIMS = (3, 4, 3, 3)
_SENSOR_PREFIXES = (
    "track_pos_w_",
    "track_quat_w_",
    "track_linvel_w_",
    "track_angvel_w_",
)


def compute_tracking_fk(
    model_file: str,
    *,
    joint_names: Sequence[str],
    base_poss: np.ndarray,
    base_rots: np.ndarray,
    base_lin_vels: np.ndarray,
    base_ang_vels: np.ndarray,
    dof_poss: np.ndarray,
    dof_vels: np.ndarray,
    progress: bool = False,
    progress_desc: str | None = None,
) -> dict[str, np.ndarray]:
    """Replay a root+joint trajectory and return tracking FK arrays.

    Args:
        model_file: MuJoCo XML scene file. ``track_*`` sensors are injected on
            a temporary copy; the source file is not modified.
        joint_names: Actuated joint names in the desired output column order.
        base_poss: (N, 3) root positions, world frame.
        base_rots: (N, 4) root quaternions, wxyz.
        base_lin_vels: (N, 3) root linear velocities, world frame.
        base_ang_vels: (N, 3) root angular velocities, world frame.
        dof_poss: (N, len(joint_names)) joint positions.
        dof_vels: (N, len(joint_names)) joint velocities.
        progress: Show a tqdm progress bar over frames.
        progress_desc: Optional tqdm description.

    Returns:
        Dict with float32 arrays ``joint_pos`` / ``joint_vel``
        (N, len(joint_names)) and ``body_pos_w`` / ``body_quat_w`` /
        ``body_lin_vel_w`` / ``body_ang_vel_w`` (N, nbody, ...) in MuJoCo
        body-id layout, including the implicit world body 0.
    """
    import mujoco

    tmp_model_path, _, _ = inject_mujoco_tracking_sensors(model_file)
    try:
        model = mujoco.MjModel.from_xml_path(tmp_model_path)
    finally:
        Path(tmp_model_path).unlink(missing_ok=True)
    data = mujoco.MjData(model)

    joint_indices = []
    for name in joint_names:
        jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jnt_id < 0:
            raise ValueError(f"Joint '{name}' not found in model")
        joint_indices.append(jnt_id)

    num_frames = base_poss.shape[0]
    num_joints = len(joint_indices)
    num_bodies = model.nbody

    joint_pos = np.zeros((num_frames, num_joints), dtype=np.float32)
    joint_vel = np.zeros((num_frames, num_joints), dtype=np.float32)
    body_pos_w = np.zeros((num_frames, num_bodies, 3), dtype=np.float32)
    body_quat_w = np.zeros((num_frames, num_bodies, 4), dtype=np.float32)
    body_lin_vel_w = np.zeros((num_frames, num_bodies, 3), dtype=np.float32)
    body_ang_vel_w = np.zeros((num_frames, num_bodies, 3), dtype=np.float32)

    # Keep arrays in model body-id layout (nbody), but read named bodies from
    # the injected track_* sensors to align with backend.get_body_*_w semantics.
    sensor_adrs = np.full((num_bodies, 4), -1, dtype=np.int32)
    for body_id in range(num_bodies):
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        if not body_name:
            continue
        for k, prefix in enumerate(_SENSOR_PREFIXES):
            sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, f"{prefix}{body_name}")
            if sensor_id >= 0:
                sensor_adrs[body_id, k] = model.sensor_adr[sensor_id]

    frame_iter = range(num_frames)
    if progress:
        from tqdm import tqdm

        frame_iter = tqdm(frame_iter, desc=progress_desc, leave=progress_desc is None)

    for i in frame_iter:
        # Set root state
        data.qpos[0:3] = base_poss[i]
        data.qpos[3:7] = base_rots[i]
        data.qvel[0:3] = base_lin_vels[i]
        data.qvel[3:6] = base_ang_vels[i]

        # Set joint states
        for j, jnt_id in enumerate(joint_indices):
            qpos_adr = model.jnt_qposadr[jnt_id]
            qvel_adr = model.jnt_dofadr[jnt_id]
            data.qpos[qpos_adr] = dof_poss[i, j]
            data.qvel[qvel_adr] = dof_vels[i, j]

        # Run forward pass so kinematics and sensors are up-to-date.
        mujoco.mj_forward(model, data)

        # Extract joint states
        for j, jnt_id in enumerate(joint_indices):
            qpos_adr = model.jnt_qposadr[jnt_id]
            qvel_adr = model.jnt_dofadr[jnt_id]
            joint_pos[i, j] = data.qpos[qpos_adr]
            joint_vel[i, j] = data.qvel[qvel_adr]

        # Extract body states
        for body_id in range(num_bodies):
            pos_adr, quat_adr, lin_adr, ang_adr = sensor_adrs[body_id]

            if pos_adr >= 0:
                body_pos_w[i, body_id] = data.sensordata[pos_adr : pos_adr + _SENSOR_DIMS[0]]
            else:
                body_pos_w[i, body_id] = data.xpos[body_id]

            if quat_adr >= 0:
                body_quat_w[i, body_id] = data.sensordata[quat_adr : quat_adr + _SENSOR_DIMS[1]]
            else:
                body_quat_w[i, body_id] = data.xquat[body_id]

            if lin_adr >= 0:
                body_lin_vel_w[i, body_id] = data.sensordata[lin_adr : lin_adr + _SENSOR_DIMS[2]]

            if ang_adr >= 0:
                body_ang_vel_w[i, body_id] = data.sensordata[ang_adr : ang_adr + _SENSOR_DIMS[3]]

    return {
        "joint_pos": joint_pos,
        "joint_vel": joint_vel,
        "body_pos_w": body_pos_w,
        "body_quat_w": body_quat_w,
        "body_lin_vel_w": body_lin_vel_w,
        "body_ang_vel_w": body_ang_vel_w,
    }
