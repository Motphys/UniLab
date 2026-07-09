#!/usr/bin/env python3
"""Convert 29-DoF G1 motion NPZ files to 23-DoF.

Removes the 6 RM (redundant manipulator) joints:
  - waist_roll_joint, waist_pitch_joint
  - left_wrist_pitch_joint, left_wrist_yaw_joint
  - right_wrist_pitch_joint, right_wrist_yaw_joint

And maps body data from 31-body 29-DoF XML order to 24-body 23-DoF XML order.

29-DoF joint indices to remove (0-indexed): [13, 14, 20, 21, 27, 28]
29-DoF body indices to keep (0-indexed):  [0..12, 15..20, 23..27]
  → maps to 24 bodies in 23-DoF XML order (pelvis + 23 links)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# 29-DoF joint indices to REMOVE (0-indexed)
# waist_roll=13, waist_pitch=14, left_wrist_pitch=20, left_wrist_yaw=21,
# right_wrist_pitch=27, right_wrist_yaw=28
JOINT_REMOVE_IDX = [13, 14, 20, 21, 27, 28]

# 29-DoF body indices to KEEP → 25 bodies in 23-DoF NPZ order
#
# The exported NPZ includes the MuJoCo world body (id=0) at index 0,
# so NPZ index = MuJoCo body ID for all child bodies.
# Keeping: world(0), pelvis→right_ankle_roll(1-13),
# torso_link→left_wrist_roll_link(16-21),
# right_shoulder_pitch_link→right_wrist_roll_link(24-28)
BODY_KEEP_IDX = [
    0,  # world body (MuJoCo id=0)
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,  # pelvis → right_ankle_roll_link
    16,
    17,
    18,
    19,
    20,
    21,  # torso_link → left_wrist_roll_link
    24,
    25,
    26,
    27,
    28,  # right_shoulder_pitch_link → right_wrist_roll_link
]
# Result: 25 bodies (world + 24 child bodies matching 23-DoF XML body order)
# MuJoCo body ID directly indexes into this array (NPZ index = MuJoCo body ID).


def convert_motion_npz(src_path: Path, dst_path: Path, is_box: bool = False) -> None:
    """Convert a 29-DoF motion NPZ to 23-DoF and save."""
    data = np.load(src_path, allow_pickle=True)

    # Build output dict
    out: dict[str, np.ndarray] = {}

    # Copy fps (no change)
    out["fps"] = data["fps"]

    # Joint data: remove 6 RM joint columns
    joint_keep = [i for i in range(data["joint_pos"].shape[1]) if i not in JOINT_REMOVE_IDX]
    assert len(joint_keep) == 23, f"Expected 23 kept joints, got {len(joint_keep)}"
    out["joint_pos"] = data["joint_pos"][:, joint_keep]
    out["joint_vel"] = data["joint_vel"][:, joint_keep]

    # Body data: keep only mapped body indices
    assert len(BODY_KEEP_IDX) == 25, f"Expected 25 kept bodies (world + 24 child), got {len(BODY_KEEP_IDX)}"
    for key in ("body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"):
        out[key] = data[key][:, BODY_KEEP_IDX]

    # Object data (box tracking): pass through unchanged
    object_keys = ("object_pos_w", "object_quat_w", "object_lin_vel_w", "object_ang_vel_w")
    for key in object_keys:
        if key in data:
            out[key] = data[key]

    # Save
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dst_path, **out)

    # Print summary
    src_shapes = {k: data[k].shape for k in data.keys()}
    dst_shapes = {k: out[k].shape for k in out.keys()}
    print(f"Converted: {src_path.name}")
    print(f"  Source shapes:  {src_shapes}")
    print(f"  Output shapes:  {dst_shapes}")
    print(f"  Saved to:       {dst_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert 29-DoF G1 motion NPZ files to 23-DoF")
    parser.add_argument(
        "input",
        nargs="*",
        help="Input NPZ files (default: predefined G1 motion files)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=None,
        help="Output directory (default: same as input dir + '_23dof' suffix)",
    )
    parser.add_argument(
        "--suffix",
        default="_23dof",
        help="Suffix for output files (default: _23dof)",
    )
    parser.add_argument(
        "--copy-box-object-data",
        action="store_true",
        help="Copy object pose data for box tracking files",
    )
    args = parser.parse_args()

    # Default: convert the known 29-DoF motion files
    motion_root = (
        Path(__file__).resolve().parent.parent / "src" / "unilab" / "assets" / "motions" / "g1"
    )

    default_inputs = [
        motion_root / "flip_360_001__A304.npz",
        motion_root / "flip_from_wall_104__A304.npz",
        motion_root / "sub3_largebox_003_boxconverted.npz",
    ]

    inputs = [Path(p) for p in args.input] if args.input else default_inputs

    for src_path in inputs:
        if not src_path.exists():
            print(f"WARNING: {src_path} not found, skipping")
            continue

        if args.output_dir:
            dst_dir = args.output_dir
        else:
            # Default: same directory
            dst_dir = src_path.parent

        stem = src_path.stem  # e.g. "flip_360_001__A304"
        dst_name = f"{stem}{args.suffix}.npz"
        dst_path = dst_dir / dst_name

        is_box = "box" in stem.lower() or "largebox" in stem.lower()
        convert_motion_npz(src_path, dst_path, is_box=is_box)


if __name__ == "__main__":
    main()
