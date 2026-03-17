"""Replay NPZ motion data in MuJoCo viewer.

Loads a preprocessed NPZ motion file and plays it back in the MuJoCo passive
viewer, setting qpos/qvel each frame so you can visually inspect the motion.

Usage:
    uv run python scripts/motion/replay_npz.py --npz_file path/to/motion.npz

    # Custom model file
    uv run python scripts/motion/replay_npz.py --npz_file motion.npz --model_file path/to/scene.xml

    # Loop playback
    uv run python scripts/motion/replay_npz.py --npz_file motion.npz --loop

    # Slow-motion (0.5x speed)
    uv run python scripts/motion/replay_npz.py --npz_file motion.npz --speed 0.5
"""

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from unilab.assets import ASSETS_ROOT_PATH


def load_npz(npz_file: str) -> dict[str, np.ndarray]:
    """Load NPZ motion file and return arrays as a dict."""
    data = np.load(npz_file)
    return {
        "fps": int(data["fps"][0]),
        "joint_pos": data["joint_pos"],
        "joint_vel": data["joint_vel"],
        "body_pos_w": data["body_pos_w"],
        "body_quat_w": data["body_quat_w"],
        "body_lin_vel_w": data["body_lin_vel_w"],
        "body_ang_vel_w": data["body_ang_vel_w"],
    }


def default_model_path() -> str:
    """Return path to the default G1 flat scene XML."""
    return str(ASSETS_ROOT_PATH / "robots" / "g1" / "scene_flat.xml")


# G1 joint names matching csv_to_npz.py order
G1_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]


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

    # Resolve joint qpos/qvel addresses
    joint_qpos_adr = []
    joint_qvel_adr = []
    for name in G1_JOINT_NAMES:
        jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jnt_id < 0:
            print(f"Warning: joint '{name}' not found in model, skipping")
            joint_qpos_adr.append(None)
            joint_qvel_adr.append(None)
        else:
            joint_qpos_adr.append(model.jnt_qposadr[jnt_id])
            joint_qvel_adr.append(model.jnt_dofadr[jnt_id])

    num_joints = joint_pos.shape[1]

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
        for j in range(min(num_joints, len(G1_JOINT_NAMES))):
            if joint_qpos_adr[j] is not None:
                data.qpos[joint_qpos_adr[j]] = joint_pos[frame_idx, j]
            if joint_qvel_adr[j] is not None:
                data.qvel[joint_qvel_adr[j]] = joint_vel[frame_idx, j]

        # Run forward kinematics (no dynamics) to update body positions for rendering
        mujoco.mj_forward(model, data)

    print("Opening viewer — close window or press Esc to quit.")
    if args.loop:
        print("Looping enabled.")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        frame = 0
        while viewer.is_running():
            t0 = time.perf_counter()

            set_frame(frame)
            viewer.sync()

            frame += 1
            if frame >= num_frames:
                if args.loop:
                    frame = 0
                else:
                    print("Playback finished.")
                    # Keep viewer open at last frame
                    while viewer.is_running():
                        time.sleep(0.05)
                    break

            # Real-time pacing adjusted by speed factor
            target_dt = dt / args.speed
            elapsed = time.perf_counter() - t0
            if target_dt - elapsed > 0:
                time.sleep(target_dt - elapsed)

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
