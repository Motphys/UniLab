"""MuJoCo-only CSV-to-NPZ conversion with forward kinematics.

This script converts motion data from CSV format (Unitree convention) to NPZ format
with precomputed forward kinematics for all bodies. It depends on MuJoCo model
loading and sensor evaluation, so it is not available for Motrix-only workflows.

Input CSV format:
- Base position (3): x, y, z
- Base quaternion (4): x, y, z, w (will be converted to w, x, y, z internally)
- Joint angles (29): all joint positions

Output NPZ format:
- fps: Frame rate (integer)
- joint_pos: Joint positions (N_frames × N_joints)
- joint_vel: Joint velocities (N_frames × N_joints)
- body_pos_w: Body positions in world frame (N_frames × N_bodies × 3)
- body_quat_w: Body quaternions in world frame (N_frames × N_bodies × 4, wxyz)
- body_lin_vel_w: Body linear velocities (N_frames × N_bodies × 3)
- body_ang_vel_w: Body angular velocities (N_frames × N_bodies × 3)

Interpolation and velocity estimation reuse the library implementation in
``unilab.tasks.motion_tracking.common.motion_loader``; forward kinematics reuse
``unilab.base.backend.compute_tracking_fk``.
"""

import argparse
from pathlib import Path

import numpy as np

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base.backend import compute_tracking_fk
from unilab.tasks.motion_tracking.common.motion_loader import interpolate_motion
from unilab.utils.rotation import np_quat_ensure_continuity


def load_csv_motion(
    motion_file: str, line_range: tuple[int, int] | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a Unitree-convention CSV into base/dof trajectory arrays."""
    if line_range is None:
        motion = np.loadtxt(motion_file, delimiter=",", dtype=np.float32, skiprows=1)
    else:
        motion = np.loadtxt(
            motion_file,
            delimiter=",",
            skiprows=max(1, line_range[0] - 1),
            max_rows=line_range[1] - line_range[0] + 1,
            dtype=np.float32,
        )

    motion_base_poss_input = motion[:, :3]
    # Convert quaternion from xyzw to wxyz
    motion_base_rots_input = motion[:, 3:7][:, [3, 0, 1, 2]]
    motion_base_rots_input = np_quat_ensure_continuity(motion_base_rots_input)
    motion_dof_poss_input = motion[:, 7:]
    return motion_base_poss_input, motion_base_rots_input, motion_dof_poss_input


def main():
    parser = argparse.ArgumentParser(
        description="Convert CSV motion to NPZ with forward kinematics"
    )
    parser.add_argument("--input_file", type=str, required=True, help="Input CSV file")
    parser.add_argument("--output_file", type=str, required=True, help="Output NPZ file")
    parser.add_argument("--input_fps", type=float, default=30.0, help="Input frame rate")
    parser.add_argument("--output_fps", type=float, default=50.0, help="Output frame rate")
    parser.add_argument(
        "--model_file",
        type=str,
        default=None,
        help="MuJoCo model file (default: G1 flat scene)",
    )
    parser.add_argument(
        "--start_time",
        type=float,
        default=None,
        help="Start time in seconds (overrides line_range)",
    )
    parser.add_argument(
        "--end_time",
        type=float,
        default=None,
        help="End time in seconds (overrides line_range)",
    )
    parser.add_argument(
        "--line_range",
        type=int,
        nargs=2,
        default=None,
        help="Line range to process (start, end)",
    )

    args = parser.parse_args()

    # Convert time range to line range if specified
    if args.start_time is not None or args.end_time is not None:
        input_fps = int(args.input_fps)

        # Calculate start and end frames (0-indexed in calculations, convert to 1-indexed for line_range)
        start_frame = 1  # Default: first line (1-indexed)
        if args.start_time is not None:
            start_frame = max(1, int(args.start_time * input_fps) + 1)

        end_frame = int(1e9)  # Default: very large number (read until EOF)
        if args.end_time is not None:
            end_frame = max(start_frame, int(args.end_time * input_fps) + 1)

        args.line_range = (start_frame, end_frame)

        start_time_display = args.start_time if args.start_time is not None else 0.0
        end_time_display = args.end_time if args.end_time is not None else "end"
        print(f"Time range: {start_time_display:.3f}s - {end_time_display}s")
        print(f"Converted to line range: {start_frame} - {end_frame}")

    # Default model file
    if args.model_file is None:
        args.model_file = str(ASSETS_ROOT_PATH / "robots" / "g1" / "scene_flat.xml")

    model_path = Path(args.model_file).expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"MuJoCo model file not found: {model_path}")
    args.model_file = str(model_path)

    # G1 joint names (in order)
    joint_names = [
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

    input_fps = int(args.input_fps)
    output_fps = int(args.output_fps)

    # Load and interpolate motion
    base_poss_input, base_rots_input, dof_poss_input = load_csv_motion(
        args.input_file,
        (args.line_range[0], args.line_range[1]) if args.line_range else None,
    )
    motion = interpolate_motion(
        base_poss_input,
        base_rots_input,
        dof_poss_input,
        input_fps=input_fps,
        output_fps=output_fps,
    )
    print(
        f"Motion interpolated: {base_poss_input.shape[0]} frames @ {input_fps} Hz "
        f"→ {motion.output_frames} frames @ {output_fps} Hz"
    )

    # Run forward kinematics and save to NPZ
    print(f"\nProcessing {motion.output_frames} frames...")
    arrays = compute_tracking_fk(
        args.model_file,
        joint_names=joint_names,
        base_poss=motion.base_poss,
        base_rots=motion.base_rots,
        base_lin_vels=motion.base_lin_vels,
        base_ang_vels=motion.base_ang_vels,
        dof_poss=motion.dof_poss,
        dof_vels=motion.dof_vels,
        progress=True,
    )

    print(f"\nSaving to {args.output_file}...")
    np.savez(
        args.output_file,
        fps=np.array([output_fps], dtype=np.int32),
        **arrays,
    )
    print("Done!")


if __name__ == "__main__":
    main()
