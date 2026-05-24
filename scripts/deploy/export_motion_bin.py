#!/usr/bin/env python3
"""Export a tracked-motion NPZ to the flat binary format used by State_WBT.

Layout (little-endian, contiguous):
  header (16 bytes):
    int32 fps
    int32 num_frames
    int32 num_joints      (29 for G1-29DOF)
    int32 num_bodies      (14 tracked bodies)
  data (all float32, frames-major):
    joint_pos       [num_frames][num_joints]
    joint_vel       [num_frames][num_joints]
    body_pos_w      [num_frames][num_bodies][3]
    body_quat_w     [num_frames][num_bodies][4]   (wxyz)
    body_lin_vel_w  [num_frames][num_bodies][3]
    body_ang_vel_w  [num_frames][num_bodies][3]

NPZ source layout (per src/unilab/envs/motion_tracking/g1/motion_loader.py):
  - 'fps' (int)
  - 'joint_pos' (N, 29)
  - 'joint_vel' (N, 29)
  - 'body_pos_w', 'body_quat_w', 'body_lin_vel_w', 'body_ang_vel_w'  (N, 31, *)
  Body axis is in MuJoCo body-id order; we slice down to 14 tracked bodies
  using mj_name2id() so the deploy side does not need to repeat the lookup.
"""
from __future__ import annotations

import argparse
import struct
from pathlib import Path

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_NPZ = REPO_ROOT / "src/unilab/assets/motions/g1/dance1_subject2_part.npz"
DEFAULT_SCENE = REPO_ROOT / "src/unilab/assets/robots/g1/scene_flat.xml"
DEFAULT_OUT = REPO_ROOT / "logs/deploy/dance1.bin"

TRACKED_BODY_NAMES = (
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--motion", type=Path, default=DEFAULT_NPZ)
    ap.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    ap.add_argument("--output", "-o", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--start-frame", type=int, default=0,
                    help="First frame to include (inclusive). Default 0.")
    end_group = ap.add_mutually_exclusive_group()
    end_group.add_argument("--end-frame", type=int, default=None,
                           help="One past the last frame (exclusive). "
                                "Default = NPZ frame count.")
    end_group.add_argument("--duration", type=float, default=None,
                           help="Seconds to keep starting at --start-frame. "
                                "Resolved to frames via NPZ fps.")
    args = ap.parse_args()

    if not args.motion.exists():
        raise SystemExit(f"NPZ not found: {args.motion}")
    if not args.scene.exists():
        raise SystemExit(f"Scene not found: {args.scene}")

    model = mujoco.MjModel.from_xml_path(str(args.scene))
    body_ids = []
    for nm in TRACKED_BODY_NAMES:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, nm)
        if bid < 0:
            raise SystemExit(f"Tracked body '{nm}' missing from model")
        body_ids.append(int(bid))

    with np.load(args.motion) as data:
        fps = int(np.asarray(data["fps"]).reshape(-1)[0])
        joint_pos = data["joint_pos"].astype(np.float32)
        joint_vel = data["joint_vel"].astype(np.float32)
        body_pos_w = data["body_pos_w"].astype(np.float32)
        body_quat_w = data["body_quat_w"].astype(np.float32)
        body_lin_vel_w = data["body_lin_vel_w"].astype(np.float32)
        body_ang_vel_w = data["body_ang_vel_w"].astype(np.float32)

    total_frames, num_joints = joint_pos.shape
    num_bodies = len(TRACKED_BODY_NAMES)

    if joint_pos.shape != joint_vel.shape:
        raise SystemExit(f"joint_pos {joint_pos.shape} != joint_vel {joint_vel.shape}")
    if num_joints != 29:
        raise SystemExit(f"Expected 29 joints, got {num_joints}")
    if body_pos_w.shape[0] != total_frames:
        raise SystemExit(f"body_pos_w frames {body_pos_w.shape[0]} != {total_frames}")

    start = args.start_frame
    if args.duration is not None:
        if args.duration <= 0:
            raise SystemExit(f"--duration must be > 0, got {args.duration}")
        end = start + int(round(args.duration * fps))
    elif args.end_frame is not None:
        end = args.end_frame
    else:
        end = total_frames
    if not (0 <= start < end <= total_frames):
        raise SystemExit(
            f"Invalid frame range [{start}, {end}); valid is [0, {total_frames}]"
        )
    if (start, end) != (0, total_frames):
        joint_pos = joint_pos[start:end]
        joint_vel = joint_vel[start:end]
        body_pos_w = body_pos_w[start:end]
        body_quat_w = body_quat_w[start:end]
        body_lin_vel_w = body_lin_vel_w[start:end]
        body_ang_vel_w = body_ang_vel_w[start:end]
    num_frames = end - start
    if body_pos_w.shape[1] < max(body_ids) + 1:
        raise SystemExit(
            f"NPZ body axis len {body_pos_w.shape[1]} insufficient for max body_id {max(body_ids)}"
        )

    # Slice 31-body axis down to 14 tracked bodies.
    body_pos_w = body_pos_w[:, body_ids]
    body_quat_w = body_quat_w[:, body_ids]
    body_lin_vel_w = body_lin_vel_w[:, body_ids]
    body_ang_vel_w = body_ang_vel_w[:, body_ids]

    # Sanity check quaternion is wxyz (norm ≈ 1 and first element typically positive
    # for "uprightish" frames). This is a soft check.
    quat_norms = np.linalg.norm(body_quat_w[0], axis=1)
    if not np.allclose(quat_norms, 1.0, atol=1e-3):
        print(f"WARN: frame-0 quat norms not unit: {quat_norms}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as f:
        f.write(struct.pack("<iiii", fps, num_frames, num_joints, num_bodies))
        joint_pos.astype("<f4").tofile(f)
        joint_vel.astype("<f4").tofile(f)
        body_pos_w.astype("<f4").tofile(f)
        body_quat_w.astype("<f4").tofile(f)
        body_lin_vel_w.astype("<f4").tofile(f)
        body_ang_vel_w.astype("<f4").tofile(f)

    expected_bytes = (
        16
        + (num_frames * num_joints * 4) * 2  # joint_pos + joint_vel
        + (num_frames * num_bodies * 3 * 4) * 3  # pos + linvel + angvel
        + (num_frames * num_bodies * 4 * 4)  # quat
    )
    actual_bytes = args.output.stat().st_size
    if actual_bytes != expected_bytes:
        raise SystemExit(
            f"Wrote {actual_bytes} bytes but expected {expected_bytes}"
        )

    duration_s = num_frames / fps
    print(f"Wrote {args.output} ({actual_bytes} bytes)")
    print(f"  fps={fps}, frames={num_frames}, joints={num_joints}, "
          f"bodies={num_bodies}, duration={duration_s:.2f}s")


if __name__ == "__main__":
    main()
