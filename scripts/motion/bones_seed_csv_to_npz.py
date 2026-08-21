"""MuJoCo-only BONES-SEED CSV-to-NPZ conversion with forward kinematics.

This script targets local BONES-SEED G1 CSV clips. Those files share one
36-column layout:

- ``Frame``
- ``root_translateX/Y/Z``
- ``root_rotateX/Y/Z``
- 29 ``*_joint_dof`` columns using G1 MuJoCo joint names

The exported NPZ format matches the motion tracking loader. The export relies on
MuJoCo model loading and tracking sensors, so it is not available for
Motrix-only workflows.

The exported NPZ format contains:
- ``fps``
- ``joint_pos``
- ``joint_vel``
- ``body_pos_w``
- ``body_quat_w``
- ``body_lin_vel_w``
- ``body_ang_vel_w``

Interpolation and velocity estimation reuse the library implementation in
``unilab.tasks.motion_tracking.common.motion_loader``; forward kinematics reuse
``unilab.base.backend.compute_tracking_fk``.

Usage:
    uv run scripts/motion/bones_seed_csv_to_npz.py
    uv run scripts/motion/bones_seed_csv_to_npz.py --dry-run
    uv run scripts/motion/bones_seed_csv_to_npz.py --input path/to/flip_090_001__A304.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scripts.motion.bones_seed_csv import (
    ROOT_COLUMNS,
    euler_deg_to_quat_wxyz,
    load_header,
    parse_joint_names,
    resolve_input_files,
)

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base.backend import compute_tracking_fk
from unilab.tasks.motion_tracking.common.motion_loader import interpolate_motion
from unilab.utils.rotation import np_quat_ensure_continuity

DEFAULT_INPUT = "src/unilab/assets/motions/g1/flip"
DEFAULT_OUTPUT_DIR = "src/unilab/assets/motions/g1/flip_npz"


def default_model_path() -> str:
    return str(ASSETS_ROOT_PATH / "robots" / "g1" / "scene_flat.xml")


def resolve_model_path(model_file: str | None) -> str:
    path = Path(model_file or default_model_path()).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"MuJoCo model file not found: {path}")
    return str(path)


def resolve_output_targets(
    input_path: str, output_path: str | None, csv_files: list[Path]
) -> list[Path]:
    input_root = Path(input_path).expanduser().resolve()
    if input_root.is_file():
        if output_path is None:
            return [input_root.with_suffix(".npz")]

        output = Path(output_path).expanduser().resolve()
        if output.exists() and output.is_dir():
            return [output / f"{input_root.stem}.npz"]
        if output.suffix.lower() != ".npz":
            raise ValueError("When --input is a file, --output must be an .npz file or a directory")
        return [output]

    output_root = Path(output_path or DEFAULT_OUTPUT_DIR).expanduser().resolve()
    return [output_root / f"{csv_file.stem}.npz" for csv_file in csv_files]


def load_csv_motion(
    motion_file: Path,
    *,
    position_scale: float,
    euler_order: str,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    """Load a BONES-SEED CSV into (joint_names, base/dof trajectory arrays)."""
    header = load_header(motion_file)
    joint_names = parse_joint_names(header, motion_file)

    motion = np.loadtxt(motion_file, delimiter=",", dtype=np.float32, skiprows=1)
    motion = np.atleast_2d(motion)
    if motion.shape[1] != len(header):
        raise ValueError(f"{motion_file} has {motion.shape[1]} columns, expected {len(header)}")

    frames = motion[:, 0].astype(np.int32)
    if frames.shape[0] > 1:
        frame_diffs = np.diff(frames)
        if not np.all(frame_diffs == 1):
            raise ValueError(
                f"{motion_file} has non-contiguous Frame values: {np.unique(frame_diffs)}"
            )

    motion_base_poss_input = motion[:, 1:4] * position_scale
    motion_base_rots_input = euler_deg_to_quat_wxyz(motion[:, 4:7], euler_order)
    motion_base_rots_input = np_quat_ensure_continuity(motion_base_rots_input)
    motion_dof_poss_input = np.deg2rad(motion[:, len(ROOT_COLUMNS) :])
    return joint_names, motion_base_poss_input, motion_base_rots_input, motion_dof_poss_input


def print_plan(
    csv_files: list[Path],
    output_files: list[Path],
    input_fps: float,
    output_fps: float,
    position_scale: float,
    euler_order: str,
) -> None:
    print(f"[bones_seed_csv_to_npz] Found {len(csv_files)} CSV file(s)")
    print(
        f"[bones_seed_csv_to_npz] Conversion: input_fps={input_fps:g}, "
        f"output_fps={output_fps:g}, position_scale={position_scale:g}, euler_order={euler_order}"
    )
    print(f"[bones_seed_csv_to_npz] Output target example: {output_files[0]}")


def run_dry_run(csv_files: list[Path], output_files: list[Path]) -> None:
    frame_counts: list[int] = []
    for csv_file in csv_files:
        header = load_header(csv_file)
        parse_joint_names(header, csv_file)
        motion = np.loadtxt(csv_file, delimiter=",", dtype=np.float32, skiprows=1)
        motion = np.atleast_2d(motion)
        frame_counts.append(int(motion.shape[0]))

    print(
        f"[bones_seed_csv_to_npz] Dry run OK: {len(csv_files)} clip(s), "
        f"frame range min={min(frame_counts)}, max={max(frame_counts)}"
    )
    if len(csv_files) > 1:
        print(f"[bones_seed_csv_to_npz] Planned output directory: {output_files[0].parent}")


def convert(args: argparse.Namespace) -> None:
    csv_files = resolve_input_files(args.input)
    output_files = resolve_output_targets(args.input, args.output, csv_files)
    model_file = resolve_model_path(args.model_file)

    print_plan(
        csv_files=csv_files,
        output_files=output_files,
        input_fps=args.input_fps,
        output_fps=args.output_fps,
        position_scale=args.position_scale,
        euler_order=args.euler_order,
    )

    if args.dry_run:
        run_dry_run(csv_files, output_files)
        return

    input_fps = int(args.input_fps)
    output_fps = int(args.output_fps)
    for csv_file, output_file in zip(csv_files, output_files, strict=True):
        print(f"[bones_seed_csv_to_npz] Converting {csv_file} -> {output_file}")
        joint_names, base_poss_input, base_rots_input, dof_poss_input = load_csv_motion(
            csv_file,
            position_scale=args.position_scale,
            euler_order=args.euler_order,
        )
        motion = interpolate_motion(
            base_poss_input,
            base_rots_input,
            dof_poss_input,
            input_fps=input_fps,
            output_fps=output_fps,
        )
        arrays = compute_tracking_fk(
            model_file,
            joint_names=joint_names,
            base_poss=motion.base_poss,
            base_rots=motion.base_rots,
            base_lin_vels=motion.base_lin_vels,
            base_ang_vels=motion.base_ang_vels,
            dof_poss=motion.dof_poss,
            dof_vels=motion.dof_vels,
            progress=True,
            progress_desc=output_file.stem,
        )
        output_file.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            output_file,
            fps=np.array([output_fps], dtype=np.int32),
            **arrays,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert G1 flip CSV motions to NPZ with forward kinematics"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=DEFAULT_INPUT,
        help=f"CSV file or root directory searched recursively for CSV files (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Output .npz path for file input, or output directory for directory input "
            f"(default directory: {DEFAULT_OUTPUT_DIR})"
        ),
    )
    parser.add_argument(
        "--input_fps",
        type=float,
        default=120.0,
        help="Input frame rate assumed for the CSV clips",
    )
    parser.add_argument(
        "--output_fps",
        type=float,
        default=50.0,
        help="Output frame rate written into the NPZ files",
    )
    parser.add_argument(
        "--model_file",
        type=str,
        default=None,
        help="MuJoCo model file (default: G1 flat scene)",
    )
    parser.add_argument(
        "--position_scale",
        type=float,
        default=0.01,
        help="Scale applied to root_translateXYZ before export",
    )
    parser.add_argument(
        "--euler_order",
        type=str,
        default="xyz",
        help="ProtoMotions/Scipy Euler order for root rotation; lowercase=extrinsic",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print the conversion plan without generating NPZ files",
    )
    args = parser.parse_args()

    if args.input_fps <= 0:
        raise ValueError("--input_fps must be positive")
    if args.output_fps <= 0:
        raise ValueError("--output_fps must be positive")
    if args.position_scale <= 0:
        raise ValueError("--position_scale must be positive")
    if len(args.euler_order) != 3 or any(ch not in "xyzXYZ" for ch in args.euler_order):
        raise ValueError("--euler_order must be a 3-character sequence using only xyzXYZ")
    return args


def main() -> None:
    convert(parse_args())


if __name__ == "__main__":
    main()
