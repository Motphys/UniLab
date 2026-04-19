"""MuJoCo-only batched rendering helpers.

This module renders many MuJoCo states into image frames by constructing
MuJoCo model/data/renderer objects inside worker processes. It is not available
for Motrix-only workflows.
"""

import math
import os
import subprocess
import sys
import textwrap
from collections.abc import Sequence
from typing import Any

import imageio

_USER_MUJOCO_GL = os.environ.get("MUJOCO_GL")

_EGL_PROBE_SCRIPT = textwrap.dedent(
    '''
    import mujoco

    xml = """
    <mujoco>
      <worldbody>
        <geom type="box" size="0.1 0.1 0.1" rgba="0 1 0 1"/>
      </worldbody>
    </mujoco>
    """

    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=8, width=8)
    mujoco.mj_forward(model, data)
    renderer.update_scene(data)
    renderer.render()
    renderer.close()
    '''
)


def _egl_runtime_usable() -> bool:
    env = os.environ.copy()
    env["MUJOCO_GL"] = "egl"
    env.setdefault("MUJOCO_EGL_DEVICE_ID", "0")

    try:
        subprocess.run(
            [sys.executable, "-c", _EGL_PROBE_SCRIPT],
            env=env,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return False

    os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", env["MUJOCO_EGL_DEVICE_ID"])
    return True


def _resolve_gl_backend() -> str:
    """Pick a valid MUJOCO_GL backend for the current platform.

    Respects an explicit user setting unless it's provably invalid (e.g. egl
    on macOS).  Falls back to glfw when EGL is requested but not available.
    """
    current = os.environ.get("MUJOCO_GL", "")
    safe_values = {"glfw", "osmesa", "disabled"}

    if sys.platform == "darwin":
        # macOS has no EGL support; glfw is the only off-screen option
        return current if current in safe_values else "glfw"

    # Linux / other: honour explicit non-egl choices supplied before import.
    if current in safe_values and current == _USER_MUJOCO_GL:
        return current

    # Probe EGL by creating a tiny MuJoCo renderer in a clean subprocess.
    if _egl_runtime_usable():
        return "egl"

    return "glfw"


# Must be set *before* importing mujoco (it reads the var at import time)
os.environ["MUJOCO_GL"] = _resolve_gl_backend()

import mujoco  # noqa: E402
import numpy as np


def get_grid_offsets(num_envs, spacing=1.0):
    rows = int(math.ceil(math.sqrt(num_envs)))
    cols = int(math.ceil(num_envs / rows))
    offsets = np.zeros((num_envs, 2))
    for i in range(num_envs):
        r = i // cols
        c = i % cols
        offsets[i, 0] = r * spacing
        offsets[i, 1] = c * spacing
    return offsets


# Worker global context
_worker_ctx: dict[str, Any] = {}


def _close_worker():
    """Explicitly close the renderer in the worker context."""
    if "renderer" in _worker_ctx:
        _worker_ctx["renderer"].close()


def init_worker(model_path, shape):
    """Initialize MuJoCo-only rendering context for a worker process."""
    import atexit

    def _load_model(path_like):
        path = str(path_like)
        loader = (
            mujoco.MjModel.from_binary_path
            if path.endswith(".mjb")
            else mujoco.MjModel.from_xml_path
        )
        return loader(path)

    if isinstance(model_path, Sequence) and not isinstance(model_path, (str, bytes, os.PathLike)):
        models = [_load_model(path) for path in model_path]
    else:
        models = [_load_model(model_path)]

    for model in models:
        model.vis.global_.offwidth = 3840
        model.vis.global_.offheight = 2160

    _worker_ctx["models"] = models
    _worker_ctx["data_list"] = [mujoco.MjData(model) for model in models]
    _worker_ctx["renderer"] = mujoco.Renderer(models[0], height=shape[1], width=shape[0])
    atexit.register(_close_worker)


def render_frame_job(args):
    """
    Worker function to render a single frame.
    args: (state_batch, offsets, transparent, cam_distance, cam_elevation, cam_azimuth, cam_lookat)
    """
    state_batch, offsets, transparent, cam_distance, cam_elevation, cam_azimuth, cam_lookat = args

    models = _worker_ctx["models"]
    data_list = _worker_ctx["data_list"]
    renderer = _worker_ctx["renderer"]

    # Visual options
    vopt = mujoco.MjvOption()
    vopt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = transparent
    pert = mujoco.MjvPerturb()
    catmask_dynamic = mujoco.mjtCatBit.mjCAT_DYNAMIC
    catmask_static = mujoco.mjtCatBit.mjCAT_STATIC

    # Helper to set state
    def set_state(model, d, s, offset=None):
        d.time = s[0]
        d.qpos[:] = s[1 : 1 + model.nq]
        d.qvel[:] = s[1 + model.nq : 1 + model.nq + model.nv]

        apply_root_offset = False

        if offset is not None:
            # Check if Root (Body 1) has a free joint or slide joints allowing X/Y movement
            # Body 0 is world. Body 1 is usually the robot base.
            robot_moved = False

            # Heuristic: Check joint at qpos 0, 1.
            # If jnt_type[0] is free (0), fine.
            # If jnt_type[0] is slide (2) and axis is x/y...

            # Better check: Does the first body have a joint?
            first_body_jnt = model.body_jntadr[1] if model.nbody > 1 else -1
            if first_body_jnt >= 0:
                jnt_type = model.jnt_type[first_body_jnt]
                # mjJNT_FREE=0
                if jnt_type == 0:
                    d.qpos[0] += offset[0]
                    d.qpos[1] += offset[1]
                    robot_moved = True

            # If robot wasn't moved via qpos, we need to manually offset geometries later
            if not robot_moved:
                apply_root_offset = True

            # 2. Box offset
            box_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "box")
            if box_id >= 0:
                jnt_adr = model.body_jntadr[box_id]
                if jnt_adr >= 0:
                    qpos_adr = model.jnt_qposadr[jnt_adr]
                    d.qpos[qpos_adr] += offset[0]
                    d.qpos[qpos_adr + 1] += offset[1]

            # 3. Target offset (target_x, target_y)
            target_x = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "target_x")
            if target_x >= 0:
                d.qpos[model.jnt_qposadr[target_x]] += offset[0]

            target_y = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "target_y")
            if target_y >= 0:
                d.qpos[model.jnt_qposadr[target_y]] += offset[1]

        mujoco.mj_forward(model, d)

        # Post-process: Shift all geometries if robot root wasn't moved
        if apply_root_offset and offset is not None:
            # Shift all geoms?
            # We should shift Everything that is PART OF THE ROBOT.
            # Or just everything?
            # Box and Target were already shifted via qpos.
            # BUT qpos shift updates body_pos which updates geom_pos.
            # If we shift ALL geom_pos, we double shift Box and Target!

            # So we need to shift geoms that belong to bodies which are NOT Box or Target.
            # Or simpler: Shift everything, but subtract offset from Box/Target qpos first? No.

            # Let's iterate bodies.
            # Simple heuristic: Shift everything except Box and Target?
            box_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "box")
            target_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "mocap_target")

            # Also target might be just a body named "mocap_target"

            for i in range(model.ngeom):
                body_id = model.geom_bodyid[i]
                # If it is robot body.
                # We want to shift generally everything that wasn't shifted by Qpos.
                # Box and Target were shifted by Qpos.
                # Floor (Plane) should usually NOT be shifted (infinite).
                # Everything else (Robot Base, Robot Links, Decoration) should be shifted.

                is_box_or_target = (body_id == box_body_id) or (body_id == target_body_id)
                is_plane = model.geom_type[i] == mujoco.mjtGeom.mjGEOM_PLANE

                if not is_box_or_target and not is_plane:
                    d.geom_xpos[i, 0] += offset[0]
                    d.geom_xpos[i, 1] += offset[1]

            # Also update site positions if they are visualized
            for i in range(model.nsite):
                body_id = model.site_bodyid[i]

                is_box_or_target = (body_id == box_body_id) or (body_id == target_body_id)

                if not is_box_or_target:
                    d.site_xpos[i, 0] += offset[0]
                    d.site_xpos[i, 1] += offset[1]

    num_envs = state_batch.shape[0]

    # 1. Clear/Init Scene
    primary_model = models[0]
    primary_data = data_list[0]
    set_state(
        primary_model, primary_data, state_batch[0], offsets[0] if offsets is not None else None
    )

    # Init Camera
    cam = mujoco.MjvCamera()
    if offsets is not None:
        center_x = np.mean(offsets[:, 0])
        center_y = np.mean(offsets[:, 1])
        if cam_lookat is None:
            cam.lookat = [center_x, center_y, 0.0]
        else:
            cam.lookat = [float(cam_lookat[0]), float(cam_lookat[1]), float(cam_lookat[2])]
        cam.distance = cam_distance
        cam.elevation = cam_elevation
        cam.azimuth = cam_azimuth
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    else:
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE

    renderer.update_scene(primary_data, camera=cam, scene_option=vopt)

    # 2. Add other robots
    for i in range(1, num_envs):
        model = models[min(i, len(models) - 1)]
        data = data_list[min(i, len(data_list) - 1)]
        set_state(model, data, state_batch[i], offsets[i] if offsets is not None else None)
        mujoco.mjv_addGeoms(model, data, vopt, pert, catmask_dynamic, renderer.scene)

        # Avoid duplicating world floor/static group-0 geoms across envs, while still
        # adding fixed-base static visuals (e.g., Sharpa base link visual).
        geomgroup0 = int(vopt.geomgroup[0])
        vopt.geomgroup[0] = 0
        mujoco.mjv_addGeoms(model, data, vopt, pert, catmask_static, renderer.scene)
        vopt.geomgroup[0] = geomgroup0

    return renderer.render()


def render_states_get_frames(
    state_list,
    model_path,
    width=1280,
    height=720,
    num_processes=8,
    camera_id=-1,
    cam_distance=2.0,
    cam_elevation=-20,
    cam_azimuth=90,
    cam_lookat=None,
    render_spacing=1.0,
):
    """
    Render a list of physics states and return the list of frames.

    Args:
        state_list: List of numpy arrays, each shape (num_envs, state_dim).
        model_path: Path to the mujoco XML model file.
        width: Width of the video.
        height: Height of the video.
        num_processes: Number of parallel processes to use.
        camera_id: Camera ID to render from.
        cam_distance: Camera distance from lookat point.
        cam_elevation: Camera elevation angle in degrees.
        cam_azimuth: Camera azimuth angle in degrees.
        cam_lookat: Optional [x, y, z] lookat override for the free camera.
        render_spacing: Grid spacing used to offset each env in composed video frames.
    Returns:
        List of numpy arrays (H, W, 3) (RGB)
    """
    if not state_list:
        print("No states to render.")
        return []

    num_envs = state_list[0].shape[0]
    offsets = get_grid_offsets(num_envs, spacing=render_spacing)
    shape = (width, height)

    print(
        f"Rendering {len(state_list)} frames for {num_envs} envs with {num_processes} processes..."
    )

    # Prepare arguments for each frame
    tasks = [
        (s, offsets, False, cam_distance, cam_elevation, cam_azimuth, cam_lookat)
        for s in state_list
    ]

    frames = []

    if num_processes <= 1:
        # Serial execution
        # Initialize context manually
        init_worker(model_path, shape)
        try:
            for task in tasks:
                res = render_frame_job(task)
                frames.append(res)
        finally:
            _close_worker()
    else:
        # Use multiprocessing Pool
        # On macOS, use spawn to avoid forking OpenGL/MuJoCo contexts.
        import multiprocessing

        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(
            processes=num_processes, initializer=init_worker, initargs=(model_path, shape)
        ) as pool:
            results = pool.map(render_frame_job, tasks)
            frames.extend(results)

    return frames


def _get_nearest_env_indices(offsets, primary_idx, max_extra):
    """Return indices of the *max_extra* environments closest to *primary_idx*."""
    if len(offsets) <= 1 + max_extra:
        return [i for i in range(len(offsets)) if i != primary_idx]
    primary = offsets[primary_idx]
    dists = np.linalg.norm(offsets - primary, axis=1)
    dists[primary_idx] = np.inf  # exclude self
    return list(np.argsort(dists)[:max_extra])


def render_frame_tracking_job(args):
    """Render a single frame with camera tracking on the primary env's root body.

    The camera uses ``mjCAMERA_TRACKING`` so it follows the robot each frame.
    Only the primary env + nearest neighbours are rendered.
    """
    (
        state_batch,
        offsets,
        env_indices,
        primary_local_idx,
        cam_distance,
        cam_elevation,
        cam_azimuth,
    ) = args

    models = _worker_ctx["models"]
    data_list = _worker_ctx["data_list"]
    renderer = _worker_ctx["renderer"]

    vopt = mujoco.MjvOption()
    pert = mujoco.MjvPerturb()
    catmask_dynamic = mujoco.mjtCatBit.mjCAT_DYNAMIC
    catmask_static = mujoco.mjtCatBit.mjCAT_STATIC

    def set_state(model, d, s, offset=None):
        d.time = s[0]
        d.qpos[:] = s[1 : 1 + model.nq]
        d.qvel[:] = s[1 + model.nq : 1 + model.nq + model.nv]

        apply_root_offset = False

        if offset is not None:
            robot_moved = False
            first_body_jnt = model.body_jntadr[1] if model.nbody > 1 else -1
            if first_body_jnt >= 0:
                jnt_type = model.jnt_type[first_body_jnt]
                if jnt_type == 0:  # mjJNT_FREE
                    d.qpos[0] += offset[0]
                    d.qpos[1] += offset[1]
                    robot_moved = True

            if not robot_moved:
                apply_root_offset = True

            box_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "box")
            if box_id >= 0:
                jnt_adr = model.body_jntadr[box_id]
                if jnt_adr >= 0:
                    qpos_adr = model.jnt_qposadr[jnt_adr]
                    d.qpos[qpos_adr] += offset[0]
                    d.qpos[qpos_adr + 1] += offset[1]

            target_x = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "target_x")
            if target_x >= 0:
                d.qpos[model.jnt_qposadr[target_x]] += offset[0]

            target_y = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "target_y")
            if target_y >= 0:
                d.qpos[model.jnt_qposadr[target_y]] += offset[1]

        mujoco.mj_forward(model, d)

        if apply_root_offset and offset is not None:
            box_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "box")
            target_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "mocap_target")

            for i in range(model.ngeom):
                body_id = model.geom_bodyid[i]
                is_box_or_target = (body_id == box_body_id) or (body_id == target_body_id)
                is_plane = model.geom_type[i] == mujoco.mjtGeom.mjGEOM_PLANE

                if not is_box_or_target and not is_plane:
                    d.geom_xpos[i, 0] += offset[0]
                    d.geom_xpos[i, 1] += offset[1]

            for i in range(model.nsite):
                body_id = model.site_bodyid[i]
                is_box_or_target = (body_id == box_body_id) or (body_id == target_body_id)
                if not is_box_or_target:
                    d.site_xpos[i, 0] += offset[0]
                    d.site_xpos[i, 1] += offset[1]

    # Primary env first — camera tracks body 1 of this env
    primary_global = env_indices[primary_local_idx]
    primary_model = models[min(primary_global, len(models) - 1)]
    primary_data = data_list[min(primary_global, len(data_list) - 1)]
    set_state(
        primary_model,
        primary_data,
        state_batch[primary_global],
        offsets[primary_global] if offsets is not None else None,
    )

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = 1  # robot root body
    cam.distance = cam_distance
    cam.elevation = cam_elevation
    cam.azimuth = cam_azimuth

    renderer.update_scene(primary_data, camera=cam, scene_option=vopt)

    # Add neighbour envs as background context
    for local_i, global_i in enumerate(env_indices):
        if local_i == primary_local_idx:
            continue
        model = models[min(global_i, len(models) - 1)]
        data = data_list[min(global_i, len(data_list) - 1)]
        set_state(
            model, data, state_batch[global_i], offsets[global_i] if offsets is not None else None
        )
        mujoco.mjv_addGeoms(model, data, vopt, pert, catmask_dynamic, renderer.scene)

        geomgroup0 = int(vopt.geomgroup[0])
        vopt.geomgroup[0] = 0
        mujoco.mjv_addGeoms(model, data, vopt, pert, catmask_static, renderer.scene)
        vopt.geomgroup[0] = geomgroup0

    return renderer.render()


def render_states_get_frames_tracking(
    state_list,
    model_path,
    width=1280,
    height=720,
    tracking_env_idx=0,
    max_extra_envs=2,
    cam_distance=2.0,
    cam_elevation=-20,
    cam_azimuth=90,
    render_spacing=1.0,
):
    """Render with camera tracking on a single primary environment.

    Only the primary env and its nearest neighbours are shown. The camera
    follows the root body of the primary env each frame (``mjCAMERA_TRACKING``).

    Args:
        state_list: List of numpy arrays, each shape (num_envs, state_dim).
        model_path: Path to the mujoco XML model file.
        tracking_env_idx: Index of the primary environment to track.
        max_extra_envs: Number of nearest-neighbour envs to render alongside.
        cam_distance: Camera distance from the tracked body.
        cam_elevation: Camera elevation angle in degrees.
        cam_azimuth: Camera azimuth angle in degrees.
        render_spacing: Grid spacing for env layout.
    """
    if not state_list:
        print("No states to render.")
        return []

    num_envs = state_list[0].shape[0]
    offsets = get_grid_offsets(num_envs, spacing=render_spacing)
    shape = (width, height)

    tracking_env_idx = min(tracking_env_idx, num_envs - 1)
    neighbour_indices = _get_nearest_env_indices(offsets, tracking_env_idx, max_extra_envs)
    env_indices = [tracking_env_idx] + neighbour_indices
    primary_local_idx = 0  # primary is always first in env_indices

    total_shown = len(env_indices)
    print(
        f"Rendering {len(state_list)} frames (tracking env {tracking_env_idx} "
        f"+ {total_shown - 1} neighbours) ..."
    )

    tasks = [
        (s, offsets, env_indices, primary_local_idx, cam_distance, cam_elevation, cam_azimuth)
        for s in state_list
    ]

    # Camera tracking changes each frame so multiprocessing gives inconsistent
    # results when workers don't share state. Default to serial.
    frames = []
    init_worker(model_path, shape)
    try:
        for task in tasks:
            frames.append(render_frame_tracking_job(task))
    finally:
        _close_worker()

    return frames


def render_states_to_video(
    state_list,
    model_path,
    output_path,
    fps=30,
    width=1280,
    height=720,
    num_processes=8,
    cam_distance=2.0,
    cam_elevation=-20,
    cam_azimuth=90,
    cam_lookat=None,
    render_spacing=1.0,
):
    """
    Render a list of physics states to a video file using parallel processing.
    """
    frames = render_states_get_frames(
        state_list,
        model_path,
        width,
        height,
        num_processes,
        cam_distance=cam_distance,
        cam_elevation=cam_elevation,
        cam_azimuth=cam_azimuth,
        cam_lookat=cam_lookat,
        render_spacing=render_spacing,
    )

    print(f"Saving video to {output_path}...")
    imageio.mimsave(output_path, frames, fps=fps)
    print("Done!")
