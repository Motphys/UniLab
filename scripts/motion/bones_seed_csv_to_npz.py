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

Usage:
    uv run scripts/motion/bones_seed_csv_to_npz.py
    uv run scripts/motion/bones_seed_csv_to_npz.py --dry-run
    uv run scripts/motion/bones_seed_csv_to_npz.py --input path/to/flip_090_001__A304.csv
"""

# pyright: reportAttributeAccessIssue=false

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
from tqdm import tqdm

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base.backend.mujoco.xml import inject_mujoco_tracking_sensors
from unilab.envs.common.rotation import np_quat_angular_velocity, np_quat_ensure_continuity

ROOT_COLUMNS = [
    "Frame",
    "root_translateX",
    "root_translateY",
    "root_translateZ",
    "root_rotateX",
    "root_rotateY",
    "root_rotateZ",
]
EXPECTED_JOINT_COUNT = 29
DEFAULT_INPUT = "src/unilab/assets/motions/g1/flip"
DEFAULT_OUTPUT_DIR = "src/unilab/assets/motions/g1/flip_npz"


def quat_slerp(q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation between two quaternions (wxyz format)."""
    dot = np.dot(q1, q2)
    if dot < 0:
        q2 = -q2
        dot = -dot

    if dot > 0.9995:
        result = q1 + t * (q2 - q1)
        return result / np.linalg.norm(result)

    theta = np.arccos(np.clip(dot, -1, 1))
    sin_theta = np.sin(theta)
    w1 = np.sin((1 - t) * theta) / sin_theta
    w2 = np.sin(t * theta) / sin_theta
    return w1 * q1 + w2 * q2


def natural_sort_key(value: str | Path) -> list[int | str]:
    text = value.as_posix() if isinstance(value, Path) else value
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


def default_model_path() -> str:
    return str(ASSETS_ROOT_PATH / "robots" / "g1" / "scene_flat.xml")


def resolve_model_path(model_file: str | None) -> str:
    path = Path(model_file or default_model_path()).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"MuJoCo model file not found: {path}")
    return str(path)


def resolve_input_files(input_path: str) -> list[Path]:
    path = Path(input_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Input path not found: {path}")

    if path.is_file():
        if path.suffix.lower() != ".csv":
            raise ValueError(f"Expected a CSV file, got: {path}")
        return [path]

    csv_files = sorted(
        [candidate for candidate in path.rglob("*.csv") if candidate.is_file()],
        key=lambda candidate: natural_sort_key(candidate.relative_to(path)),
    )
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in directory: {path}")
    return csv_files


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


def load_header(csv_file: Path) -> list[str]:
    first_line = csv_file.read_text(encoding="utf-8").splitlines()[0]
    return [part.strip() for part in first_line.split(",")]


def parse_joint_names(header: list[str], csv_file: Path) -> list[str]:
    root_columns = header[: len(ROOT_COLUMNS)]
    if root_columns != ROOT_COLUMNS:
        raise ValueError(
            f"Unexpected root columns in {csv_file}.\n"
            f"Expected: {ROOT_COLUMNS}\n"
            f"Actual:   {root_columns}"
        )

    joint_columns = header[len(ROOT_COLUMNS) :]
    if len(joint_columns) != EXPECTED_JOINT_COUNT:
        raise ValueError(
            f"{csv_file} has {len(joint_columns)} joint columns, expected {EXPECTED_JOINT_COUNT}"
        )
    if len(set(joint_columns)) != len(joint_columns):
        raise ValueError(f"{csv_file} has duplicate joint columns")
    if any(not name.endswith("_dof") for name in joint_columns):
        raise ValueError(f"{csv_file} has non-joint columns after the root columns")
    return [name.removesuffix("_dof") for name in joint_columns]


def _swap_case_for_mujoco_euler(order: str) -> str:
    """Translate ProtoMotions/Scipy Euler case semantics to MuJoCo semantics."""
    return "".join(ch.lower() if ch.isupper() else ch.upper() for ch in order)


def euler_deg_to_quat_wxyz(euler_deg: np.ndarray, order: str) -> np.ndarray:
    mujoco_order = _swap_case_for_mujoco_euler(order)
    euler_rad = np.deg2rad(euler_deg)
    quat_wxyz = np.zeros((euler_deg.shape[0], 4), dtype=np.float64)
    for idx in range(euler_deg.shape[0]):
        mujoco.mju_euler2Quat(quat_wxyz[idx], euler_rad[idx], mujoco_order)
    return quat_wxyz


@dataclass
class MotionLoader:
    motion_file: Path
    input_fps: int
    output_fps: int
    position_scale: float
    euler_order: str

    def __post_init__(self) -> None:
        self.input_dt = 1.0 / self.input_fps
        self.output_dt = 1.0 / self.output_fps
        self._load_motion()
        self._interpolate_motion()
        self._compute_velocities()

    def _load_motion(self) -> None:
        header = load_header(self.motion_file)
        self.joint_names = parse_joint_names(header, self.motion_file)

        motion = np.loadtxt(self.motion_file, delimiter=",", dtype=np.float32, skiprows=1)
        motion = np.atleast_2d(motion)
        if motion.shape[1] != len(header):
            raise ValueError(
                f"{self.motion_file} has {motion.shape[1]} columns, expected {len(header)}"
            )

        self.frames = motion[:, 0].astype(np.int32)
        if self.frames.shape[0] > 1:
            frame_diffs = np.diff(self.frames)
            if not np.all(frame_diffs == 1):
                raise ValueError(
                    f"{self.motion_file} has non-contiguous Frame values: {np.unique(frame_diffs)}"
                )

        self.motion_base_poss_input = motion[:, 1:4] * self.position_scale
        self.motion_base_rots_input = euler_deg_to_quat_wxyz(motion[:, 4:7], self.euler_order)
        self.motion_base_rots_input = np_quat_ensure_continuity(self.motion_base_rots_input)
        self.motion_dof_poss_input = np.deg2rad(motion[:, len(ROOT_COLUMNS) :])

        self.input_frames = motion.shape[0]
        self.duration = (self.input_frames - 1) * self.input_dt

    def _interpolate_motion(self) -> None:
        times = np.arange(0, self.duration, self.output_dt, dtype=np.float32)
        self.output_frames = times.shape[0]
        index_0, index_1, blend = self._compute_frame_blend(times)

        self.motion_base_poss = (
            self.motion_base_poss_input[index_0] * (1 - blend[:, None])
            + self.motion_base_poss_input[index_1] * blend[:, None]
        )

        self.motion_base_rots = np.zeros((self.output_frames, 4), dtype=np.float32)
        for i in range(self.output_frames):
            self.motion_base_rots[i] = quat_slerp(
                self.motion_base_rots_input[index_0[i]],
                self.motion_base_rots_input[index_1[i]],
                blend[i],
            )
        self.motion_base_rots = np_quat_ensure_continuity(self.motion_base_rots)

        self.motion_dof_poss = (
            self.motion_dof_poss_input[index_0] * (1 - blend[:, None])
            + self.motion_dof_poss_input[index_1] * blend[:, None]
        )

    def _compute_frame_blend(self, times: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        phase = times / self.duration
        index_0 = np.floor(phase * (self.input_frames - 1)).astype(np.int32)
        index_1 = np.minimum(index_0 + 1, self.input_frames - 1)
        blend = phase * (self.input_frames - 1) - index_0
        return index_0, index_1, blend

    def _compute_velocities(self) -> None:
        self.motion_base_lin_vels = np.gradient(self.motion_base_poss, self.output_dt, axis=0)
        self.motion_dof_vels = np.gradient(self.motion_dof_poss, self.output_dt, axis=0)
        self.motion_base_ang_vels = np_quat_angular_velocity(self.motion_base_rots, self.output_dt)


def run_simulation(
    motion_loader: MotionLoader,
    model_file: str,
    output_file: Path,
) -> None:
    tmp_model_path, _, _ = inject_mujoco_tracking_sensors(model_file)
    try:
        model = mujoco.MjModel.from_xml_path(tmp_model_path)
    finally:
        Path(tmp_model_path).unlink(missing_ok=True)
    data = mujoco.MjData(model)

    joint_indices = []
    for name in motion_loader.joint_names:
        jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jnt_id < 0:
            raise ValueError(f"Joint '{name}' not found in model")
        joint_indices.append(jnt_id)

    num_frames = motion_loader.output_frames
    num_joints = len(joint_indices)
    num_bodies = model.nbody

    joint_pos = np.zeros((num_frames, num_joints), dtype=np.float32)
    joint_vel = np.zeros((num_frames, num_joints), dtype=np.float32)
    body_pos_w = np.zeros((num_frames, num_bodies, 3), dtype=np.float32)
    body_quat_w = np.zeros((num_frames, num_bodies, 4), dtype=np.float32)
    body_lin_vel_w = np.zeros((num_frames, num_bodies, 3), dtype=np.float32)
    body_ang_vel_w = np.zeros((num_frames, num_bodies, 3), dtype=np.float32)

    sensor_adrs = np.full((num_bodies, 4), -1, dtype=np.int32)
    sensor_dims = np.array([3, 4, 3, 3], dtype=np.int32)
    sensor_prefixes = (
        "track_pos_w_",
        "track_quat_w_",
        "track_linvel_w_",
        "track_angvel_w_",
    )
    for body_id in range(num_bodies):
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        if not body_name:
            continue
        for k, prefix in enumerate(sensor_prefixes):
            sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, f"{prefix}{body_name}")
            if sensor_id >= 0:
                sensor_adrs[body_id, k] = model.sensor_adr[sensor_id]

    for i in tqdm(range(num_frames), desc=output_file.stem, leave=False):
        data.qpos[0:3] = motion_loader.motion_base_poss[i]
        data.qpos[3:7] = motion_loader.motion_base_rots[i]
        data.qvel[0:3] = motion_loader.motion_base_lin_vels[i]
        data.qvel[3:6] = motion_loader.motion_base_ang_vels[i]

        for j, jnt_id in enumerate(joint_indices):
            qpos_adr = model.jnt_qposadr[jnt_id]
            qvel_adr = model.jnt_dofadr[jnt_id]
            data.qpos[qpos_adr] = motion_loader.motion_dof_poss[i, j]
            data.qvel[qvel_adr] = motion_loader.motion_dof_vels[i, j]

        mujoco.mj_forward(model, data)

        for j, jnt_id in enumerate(joint_indices):
            qpos_adr = model.jnt_qposadr[jnt_id]
            qvel_adr = model.jnt_dofadr[jnt_id]
            joint_pos[i, j] = data.qpos[qpos_adr]
            joint_vel[i, j] = data.qvel[qvel_adr]

        for body_id in range(num_bodies):
            pos_adr, quat_adr, lin_adr, ang_adr = sensor_adrs[body_id]

            if pos_adr >= 0:
                body_pos_w[i, body_id] = data.sensordata[pos_adr : pos_adr + sensor_dims[0]]
            else:
                body_pos_w[i, body_id] = data.xpos[body_id]

            if quat_adr >= 0:
                body_quat_w[i, body_id] = data.sensordata[quat_adr : quat_adr + sensor_dims[1]]
            else:
                body_quat_w[i, body_id] = data.xquat[body_id]

            if lin_adr >= 0:
                body_lin_vel_w[i, body_id] = data.sensordata[lin_adr : lin_adr + sensor_dims[2]]

            if ang_adr >= 0:
                body_ang_vel_w[i, body_id] = data.sensordata[ang_adr : ang_adr + sensor_dims[3]]

    output_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_file,
        fps=np.array([motion_loader.output_fps], dtype=np.int32),
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        body_pos_w=body_pos_w,
        body_quat_w=body_quat_w,
        body_lin_vel_w=body_lin_vel_w,
        body_ang_vel_w=body_ang_vel_w,
    )


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

    for csv_file, output_file in zip(csv_files, output_files, strict=True):
        print(f"[bones_seed_csv_to_npz] Converting {csv_file} -> {output_file}")
        motion_loader = MotionLoader(
            motion_file=csv_file,
            input_fps=int(args.input_fps),
            output_fps=int(args.output_fps),
            position_scale=args.position_scale,
            euler_order=args.euler_order,
        )
        run_simulation(motion_loader, model_file, output_file)


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
