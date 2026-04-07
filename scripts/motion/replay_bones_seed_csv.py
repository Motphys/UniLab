"""Replay BONES-SEED G1 CSV motions in the MuJoCo viewer.

This script targets the CSV files under ``src/unilab/assets/motions/g1/flip``.
Those files share one 36-column layout:

- ``Frame``
- ``root_translateX/Y/Z``
- ``root_rotateX/Y/Z``
- 29 ``*_joint_dof`` columns using G1 MuJoCo joint names

Assumptions:
- root translation is stored in centimeters and converted to meters
- root Euler angles are stored in degrees
- joint DOFs are stored in degrees
- root Euler order defaults to ProtoMotions/Scipy-style extrinsic ``xyz``

Usage:
    uv run python scripts/motion/replay_bones_seed_csv.py
    uv run python scripts/motion/replay_bones_seed_csv.py --input src/unilab/assets/motions/g1/flip
    uv run python scripts/motion/replay_bones_seed_csv.py --input src/unilab/assets/motions/g1/flip/flip_090_001__A304.csv
    uv run python scripts/motion/replay_bones_seed_csv.py --dry-run
"""

from __future__ import annotations

import argparse
import re
import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from unilab.assets import ASSETS_ROOT_PATH

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


@dataclass
class CsvMotion:
    path: Path
    frames: np.ndarray
    root_pos_m: np.ndarray
    root_quat_wxyz: np.ndarray
    joint_pos_rad: np.ndarray
    joint_names: list[str]

    @property
    def num_frames(self) -> int:
        return int(self.frames.shape[0])


def default_model_path() -> str:
    return str(ASSETS_ROOT_PATH / "robots" / "g1" / "scene_flat.xml")


def natural_sort_key(value: str | Path) -> list[int | str]:
    text = value.as_posix() if isinstance(value, Path) else value
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


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


def load_motion(csv_file: Path, position_scale: float, euler_order: str) -> CsvMotion:
    header = load_header(csv_file)
    joint_names = parse_joint_names(header, csv_file)

    raw = np.loadtxt(csv_file, delimiter=",", skiprows=1, dtype=np.float64)
    raw = np.atleast_2d(raw)
    if raw.shape[1] != len(header):
        raise ValueError(f"{csv_file} has {raw.shape[1]} columns, expected {len(header)}")

    frames = raw[:, 0].astype(np.int32)
    if frames.shape[0] > 1:
        frame_diffs = np.diff(frames)
        if not np.all(frame_diffs == 1):
            raise ValueError(
                f"{csv_file} has non-contiguous Frame values: {np.unique(frame_diffs)}"
            )

    return CsvMotion(
        path=csv_file,
        frames=frames,
        root_pos_m=raw[:, 1:4] * position_scale,
        root_quat_wxyz=euler_deg_to_quat_wxyz(raw[:, 4:7], euler_order),
        joint_pos_rad=np.deg2rad(raw[:, len(ROOT_COLUMNS) :]),
        joint_names=joint_names,
    )


def resolve_joint_qpos_addresses(model: mujoco.MjModel, joint_names: list[str]) -> list[int]:
    joint_qpos_adr: list[int] = []
    for joint_name in joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise ValueError(f"Joint '{joint_name}' not found in model")
        joint_qpos_adr.append(int(model.jnt_qposadr[joint_id]))
    return joint_qpos_adr


def print_dataset_summary(
    csv_files: list[Path],
    joint_count: int,
    fps: float,
    speed: float,
    position_scale: float,
    euler_order: str,
    loop: bool,
) -> None:
    print(f"[replay_bones_seed_csv] Found {len(csv_files)} CSV file(s)")
    print(
        "[replay_bones_seed_csv] Format: "
        f"{len(ROOT_COLUMNS) + joint_count} columns = root pose + {joint_count} joint DOFs"
    )
    print(
        "[replay_bones_seed_csv] Units: "
        "root_translate*=cm -> m, root_rotate*=deg, *_joint_dof=deg"
    )
    print(
        "[replay_bones_seed_csv] Playback: "
        f"fps={fps:g}, speed={speed:g}, root_euler_order={euler_order}, "
        f"position_scale={position_scale:g}"
    )
    if loop:
        print("[replay_bones_seed_csv] Playlist looping enabled")
    print("[replay_bones_seed_csv] Controls: Space=pause/resume, [=previous, ]=next")


def run_dry_run(csv_files: list[Path], position_scale: float, euler_order: str, fps: float) -> None:
    frame_counts: list[int] = []
    for csv_file in csv_files:
        motion = load_motion(csv_file, position_scale, euler_order)
        frame_counts.append(motion.num_frames)

    total_frames = sum(frame_counts)
    total_duration = total_frames / fps
    print(
        "[replay_bones_seed_csv] Dry run OK: "
        f"{len(csv_files)} clip(s), {total_frames} total frames, {total_duration:.2f}s @ {fps:g} fps"
    )
    print(
        "[replay_bones_seed_csv] Frame range per clip: "
        f"min={min(frame_counts)}, max={max(frame_counts)}"
    )


def replay(args: argparse.Namespace) -> None:
    csv_files = resolve_input_files(args.input)
    first_motion = load_motion(csv_files[0], args.position_scale, args.euler_order)
    model_file = resolve_model_path(args.model_file)

    print_dataset_summary(
        csv_files=csv_files,
        joint_count=len(first_motion.joint_names),
        fps=args.fps,
        speed=args.speed,
        position_scale=args.position_scale,
        euler_order=args.euler_order,
        loop=args.loop,
    )

    if args.dry_run:
        run_dry_run(csv_files, args.position_scale, args.euler_order, args.fps)
        return

    model = mujoco.MjModel.from_xml_path(model_file)
    data = mujoco.MjData(model)
    joint_qpos_adr = resolve_joint_qpos_addresses(model, first_motion.joint_names)

    state = {
        "motion_index": 0,
        "frame_index": 0,
        "paused": False,
        "pending_switch": 0,
    }
    motion = first_motion

    def print_motion_summary() -> None:
        duration = motion.num_frames / args.fps
        print(
            f"[replay_bones_seed_csv] [{state['motion_index'] + 1}/{len(csv_files)}] "
            f"{motion.path} ({motion.num_frames} frames, {duration:.2f}s @ {args.fps:g} fps)"
        )

    def load_motion_at(index: int) -> None:
        nonlocal motion
        motion = load_motion(csv_files[index], args.position_scale, args.euler_order)
        if motion.joint_names != first_motion.joint_names:
            raise ValueError(f"{motion.path} uses a different joint column layout")
        state["motion_index"] = index
        state["frame_index"] = 0
        mujoco.mj_resetData(model, data)
        print_motion_summary()

    def set_frame(frame_index: int) -> None:
        data.qpos[:] = 0
        data.qvel[:] = 0
        data.qpos[0:3] = motion.root_pos_m[frame_index]
        data.qpos[3:7] = motion.root_quat_wxyz[frame_index]

        for joint_idx, qpos_adr in enumerate(joint_qpos_adr):
            data.qpos[qpos_adr] = motion.joint_pos_rad[frame_index, joint_idx]

        mujoco.mj_forward(model, data)

    def request_switch(delta: int) -> None:
        if len(csv_files) > 1:
            state["pending_switch"] = delta

    def on_key(keycode: int) -> None:
        if keycode == ord(" "):
            state["paused"] = not state["paused"]
            print(f"[replay_bones_seed_csv] {'paused' if state['paused'] else 'resumed'}")
        elif keycode == ord("["):
            request_switch(-1)
        elif keycode == ord("]"):
            request_switch(1)

    print_motion_summary()
    print("[replay_bones_seed_csv] Opening viewer, close the window or press Esc to quit.")

    with mujoco.viewer.launch_passive(model, data, key_callback=on_key) as viewer:
        while viewer.is_running():
            tick_start = time.perf_counter()

            if state["pending_switch"] != 0:
                next_index = state["motion_index"] + state["pending_switch"]
                state["pending_switch"] = 0
                if args.loop:
                    next_index %= len(csv_files)
                elif not 0 <= next_index < len(csv_files):
                    next_index = state["motion_index"]
                if next_index != state["motion_index"]:
                    load_motion_at(next_index)

            set_frame(state["frame_index"])
            viewer.sync()

            if state["paused"]:
                time.sleep(0.05)
                continue

            state["frame_index"] += 1
            if state["frame_index"] >= motion.num_frames:
                next_index = state["motion_index"] + 1
                if next_index < len(csv_files):
                    load_motion_at(next_index)
                elif args.loop:
                    load_motion_at(0)
                else:
                    state["frame_index"] = motion.num_frames - 1
                    state["paused"] = True
                    print("[replay_bones_seed_csv] Reached the end of the playlist, paused on the last frame.")

            target_dt = (1.0 / args.fps) / args.speed
            elapsed = time.perf_counter() - tick_start
            if target_dt > elapsed:
                time.sleep(target_dt - elapsed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay BONES-SEED G1 CSV motion in MuJoCo")
    parser.add_argument(
        "--input",
        type=str,
        default=DEFAULT_INPUT,
        help=f"CSV file or root directory searched recursively for CSV files (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--model_file",
        type=str,
        default=None,
        help="MuJoCo XML model file (default: G1 flat scene)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=120.0,
        help="Playback FPS. The CSV files do not embed FPS metadata.",
    )
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier")
    parser.add_argument(
        "--position_scale",
        type=float,
        default=0.01,
        help="Scale applied to root_translateXYZ before sending to MuJoCo",
    )
    parser.add_argument(
        "--euler_order",
        type=str,
        default="xyz",
        help="ProtoMotions/Scipy Euler order for root rotation; lowercase=extrinsic",
    )
    parser.add_argument("--loop", action="store_true", help="Loop the whole playlist")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate CSV parsing and print a summary without opening the viewer",
    )
    args = parser.parse_args()

    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    if args.speed <= 0:
        raise ValueError("--speed must be positive")
    if args.position_scale <= 0:
        raise ValueError("--position_scale must be positive")
    if len(args.euler_order) != 3 or any(ch not in "xyzXYZ" for ch in args.euler_order):
        raise ValueError("--euler_order must be a 3-character sequence using only xyzXYZ")
    return args


def main() -> None:
    replay(parse_args())


if __name__ == "__main__":
    main()
