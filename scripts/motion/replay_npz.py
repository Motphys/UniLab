"""MuJoCo-only NPZ motion replay in the MuJoCo viewer.

Loads a preprocessed NPZ motion file and plays it back in the MuJoCo passive
viewer, setting qpos/qvel each frame so you can visually inspect the motion.
This replay path depends on the MuJoCo viewer/runtime and is not available for
Motrix-only workflows.

Usage:
    uv run scripts/motion/replay_npz.py --npz_file path/to/motion.npz

    # Custom model file
    uv run scripts/motion/replay_npz.py --npz_file motion.npz --model_file path/to/scene.xml

    # Loop playback
    uv run scripts/motion/replay_npz.py --npz_file motion.npz --loop

    # Slow-motion (0.5x speed)
    uv run scripts/motion/replay_npz.py --npz_file motion.npz --speed 0.5

Controls:
    Space: pause/resume
"""

# pyright: reportAttributeAccessIssue=false, reportReturnType=false

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from unilab.assets import ASSETS_ROOT_PATH


def load_npz(npz_file: str) -> dict[str, np.ndarray]:
    """Load NPZ motion file and return arrays as a dict."""
    data = np.load(npz_file, allow_pickle=True)
    motion = {
        "fps": int(data["fps"][0]),
        "joint_pos": data["joint_pos"],
        "joint_vel": data["joint_vel"],
        "body_pos_w": data["body_pos_w"],
        "body_quat_w": data["body_quat_w"],
        "body_lin_vel_w": data["body_lin_vel_w"],
        "body_ang_vel_w": data["body_ang_vel_w"],
    }
    if "joint_names" in data:
        motion["joint_names"] = data["joint_names"]
    return motion


def default_model_path() -> str:
    """Return path to the default G1 flat scene XML."""
    return str(ASSETS_ROOT_PATH / "robots" / "g1" / "scene_flat.xml")


def model_scalar_joint_names(model: mujoco.MjModel) -> list[str]:
    """Return non-free scalar joints in MuJoCo qpos order."""
    joints: list[tuple[int, str]] = []
    for joint_id in range(model.njnt):
        if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
            continue

        qpos_adr = int(model.jnt_qposadr[joint_id])
        qvel_adr = int(model.jnt_dofadr[joint_id])
        next_qpos_adr = int(model.nq)
        next_qvel_adr = int(model.nv)
        for other_id in range(model.njnt):
            other_qpos_adr = int(model.jnt_qposadr[other_id])
            other_qvel_adr = int(model.jnt_dofadr[other_id])
            if other_qpos_adr > qpos_adr:
                next_qpos_adr = min(next_qpos_adr, other_qpos_adr)
            if other_qvel_adr > qvel_adr:
                next_qvel_adr = min(next_qvel_adr, other_qvel_adr)

        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if not name:
            raise ValueError(f"Joint id {joint_id} has no name")
        if next_qpos_adr - qpos_adr != 1 or next_qvel_adr - qvel_adr != 1:
            raise ValueError(f"Joint '{name}' is not a scalar joint")
        joints.append((qpos_adr, name))
    return [name for _, name in sorted(joints)]


def resolve_motion_joint_names(
    motion: dict[str, np.ndarray],
    model: mujoco.MjModel,
    num_joints: int,
) -> list[str]:
    """Resolve the joint-name order used by joint_pos/joint_vel columns."""
    if "joint_names" in motion:
        names = [str(name) for name in motion["joint_names"].tolist()]
        if len(names) != num_joints:
            raise ValueError(
                f"NPZ joint_names length ({len(names)}) does not match joint_pos width ({num_joints})"
            )
        print("Joint order: NPZ joint_names")
        return names

    names = model_scalar_joint_names(model)
    if len(names) != num_joints:
        raise ValueError(
            "NPZ has no joint_names and joint_pos width does not match model scalar joints "
            f"({num_joints} vs {len(names)})."
        )
    print("Joint order: model qpos order (NPZ has no joint_names)")
    return names


def replay(args):
    motion = load_npz(args.npz_file)
    fps = motion["fps"]
    joint_pos = motion["joint_pos"]
    joint_vel = motion["joint_vel"]
    body_pos_w = motion["body_pos_w"]
    body_quat_w = motion["body_quat_w"]
    num_frames = joint_pos.shape[0]
    dt = 1.0 / fps

    print(f"Motion: {num_frames} frames @ {fps} Hz ({num_frames / fps:.2f}s)")
    print(f"Joints: {joint_pos.shape[1]}, Bodies: {body_pos_w.shape[1]}")
    print(f"Playback speed: {args.speed}x")

    model_file = args.model_file or default_model_path()
    print(f"Model: {model_file}")

    model = mujoco.MjModel.from_xml_path(model_file)
    data = mujoco.MjData(model)
    num_joints = joint_pos.shape[1]
    joint_names = resolve_motion_joint_names(motion, model, num_joints)

    # Resolve joint qpos/qvel addresses
    joint_qpos_adr = []
    joint_qvel_adr = []
    for name in joint_names:
        jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jnt_id < 0:
            raise ValueError(f"Motion joint '{name}' is not found in model")
        joint_qpos_adr.append(model.jnt_qposadr[jnt_id])
        joint_qvel_adr.append(model.jnt_dofadr[jnt_id])

    def set_frame(frame_idx: int):
        """Set MuJoCo data to the given motion frame."""
        # Root body (body 0) position and orientation from body_pos_w / body_quat_w
        # body index 0 is "world" in MuJoCo, body index 1 is typically the floating base
        # Use pelvis (first tracked body) as root — its world-frame pose goes into qpos[0:7]
        # The NPZ stores ALL model bodies, so index 1 is usually the floating-base body.
        root_body_id = 1  # floating base in most MuJoCo humanoid models
        if body_pos_w.shape[1] > root_body_id:
            data.qpos[0:3] = body_pos_w[frame_idx, root_body_id]
            data.qpos[3:7] = body_quat_w[frame_idx, root_body_id]
        else:
            # Fallback: use joint data only
            pass

        # Set joint positions and velocities
        for j in range(num_joints):
            data.qpos[joint_qpos_adr[j]] = joint_pos[frame_idx, j]
            data.qvel[joint_qvel_adr[j]] = joint_vel[frame_idx, j]

        # Run forward kinematics (no dynamics) to update body positions for rendering
        mujoco.mj_forward(model, data)

    paused = False
    last_status = ""

    def print_frame_status(frame_idx: int) -> None:
        nonlocal last_status
        status = "paused" if paused else "playing"
        message = f"Frame: {frame_idx + 1}/{num_frames} ({status})"
        if message != last_status:
            print(f"\r{message}", end="", flush=True)
            last_status = message

    def on_key(keycode: int) -> None:
        nonlocal last_status, paused
        if keycode == ord(" "):
            paused = not paused
            print()
            print(f"{'Paused' if paused else 'Resumed'} playback.")
            last_status = ""

    print("Opening viewer — close window or press Esc to quit.")
    print("Controls: Space=pause/resume")
    if args.loop:
        print("Looping enabled.")

    with mujoco.viewer.launch_passive(model, data, key_callback=on_key) as viewer:
        frame = 0
        while viewer.is_running():
            t0 = time.perf_counter()

            set_frame(frame)
            viewer.sync()
            print_frame_status(frame)

            if paused:
                time.sleep(0.05)
                continue

            frame += 1
            if frame >= num_frames:
                if args.loop:
                    frame = 0
                else:
                    print("Playback finished.")
                    frame = num_frames - 1
                    paused = True
                    print()
                    last_status = ""

            # Real-time pacing adjusted by speed factor
            target_dt = dt / args.speed
            elapsed = time.perf_counter() - t0
            if target_dt - elapsed > 0:
                time.sleep(target_dt - elapsed)

    print()
    print("Done.")


def main():
    parser = argparse.ArgumentParser(description="Replay NPZ motion in MuJoCo viewer")
    parser.add_argument("--npz_file", type=str, required=True, help="Path to NPZ motion file")
    parser.add_argument("--model_file", type=str, default=None, help="MuJoCo XML model file")
    parser.add_argument("--loop", action="store_true", help="Loop playback")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier")
    args = parser.parse_args()

    if not Path(args.npz_file).exists():
        print(f"Error: NPZ file not found: {args.npz_file}")
        return

    replay(args)


if __name__ == "__main__":
    main()
