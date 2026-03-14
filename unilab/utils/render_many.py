import os
import math
import numpy as np
import mujoco
import imageio
from multiprocessing import Pool

# Use EGL for off-screen rendering on headless servers (no X11/display required)
os.environ.setdefault("MUJOCO_GL", "egl")


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
_worker_ctx = {}


def _close_worker():
    """Explicitly close the renderer in the worker context."""
    if "renderer" in _worker_ctx:
        _worker_ctx["renderer"].close()


def init_worker(model_path, shape):
    """Initialize MuJoCo context for worker process."""
    import atexit

    # Ensure EGL is used in spawned worker processes (headless server support)
    os.environ.setdefault("MUJOCO_GL", "egl")
    _worker_ctx["model"] = mujoco.MjModel.from_xml_path(model_path)
    _worker_ctx["model"].vis.global_.offwidth = 3840
    _worker_ctx["model"].vis.global_.offheight = 2160

    _worker_ctx["data"] = mujoco.MjData(_worker_ctx["model"])
    _worker_ctx["renderer"] = mujoco.Renderer(_worker_ctx["model"], height=shape[1], width=shape[0])
    atexit.register(_close_worker)


def render_frame_job(args):
    """
    Worker function to render a single frame.
    args: (state_batch, offsets, transparent, cam_distance, cam_elevation, cam_azimuth)
    """
    state_batch, offsets, transparent, cam_distance, cam_elevation, cam_azimuth = args

    model = _worker_ctx["model"]
    data = _worker_ctx["data"]
    renderer = _worker_ctx["renderer"]

    # Visual options
    vopt = mujoco.MjvOption()
    vopt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = transparent
    pert = mujoco.MjvPerturb()
    catmask = mujoco.mjtCatBit.mjCAT_DYNAMIC

    # Helper to set state
    def set_state(d, s, offset=None):
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
    set_state(data, state_batch[0], offsets[0] if offsets is not None else None)

    # Init Camera
    cam = mujoco.MjvCamera()
    if offsets is not None:
        center_x = np.mean(offsets[:, 0])
        center_y = np.mean(offsets[:, 1])
        cam.lookat = [center_x, center_y, 0.0]
        cam.distance = cam_distance
        cam.elevation = cam_elevation
        cam.azimuth = cam_azimuth
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    else:
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE

    renderer.update_scene(data, camera=cam, scene_option=vopt)

    # 2. Add other robots
    for i in range(1, num_envs):
        set_state(data, state_batch[i], offsets[i] if offsets is not None else None)
        mujoco.mjv_addGeoms(model, data, vopt, pert, catmask, renderer.scene)

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
    Returns:
        List of numpy arrays (H, W, 3) (RGB)
    """
    if not state_list:
        print("No states to render.")
        return []

    num_envs = state_list[0].shape[0]
    offsets = get_grid_offsets(num_envs)
    shape = (width, height)

    print(
        f"Rendering {len(state_list)} frames for {num_envs} envs with {num_processes} processes..."
    )

    # Prepare arguments for each frame
    tasks = [(s, offsets, False, cam_distance, cam_elevation, cam_azimuth) for s in state_list]

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
    )

    print(f"Saving video to {output_path}...")
    imageio.mimsave(output_path, frames, fps=fps)
    print("Done!")
