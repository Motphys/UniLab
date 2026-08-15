"""MuJoCo-only BONES-SEED CSV input contract helpers."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np

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


def natural_sort_key(value: str | Path) -> list[int | str]:
    text = value.as_posix() if isinstance(value, Path) else value
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


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
    import mujoco as _mujoco

    mujoco: Any = _mujoco

    mujoco_order = _swap_case_for_mujoco_euler(order)
    euler_rad = np.deg2rad(euler_deg)
    quat_wxyz = np.zeros((euler_deg.shape[0], 4), dtype=np.float64)
    for idx in range(euler_deg.shape[0]):
        mujoco.mju_euler2Quat(quat_wxyz[idx], euler_rad[idx], mujoco_order)
    return quat_wxyz
