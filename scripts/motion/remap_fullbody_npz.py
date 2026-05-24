"""Remap a full-body MuJoCo NPZ to the standard training-model body layout.

Some motion capture pipelines export NPZ files from a detailed MuJoCo model
(e.g. 51 bodies with collision spheres, finger links, etc.) that includes the
root free-joint in joint_pos/joint_vel.  The UniLab training pipeline expects
the NPZ to match the training model layout (e.g. 31 bodies from scene_flat.xml)
with only the actuated joint DOF.

This script:
  1. Reads ``body_names`` from the source NPZ to build a name-based remapping.
  2. Derives the target body list from the training-model XML.
  3. Reindexes all body arrays (pos, quat, lin_vel, ang_vel) to the target layout.
  4. Strips the root free-joint prefix from joint_pos (7 cols) and joint_vel (6 cols).
  5. Writes a canonical 7-key NPZ in float32 / int32.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from unilab.base.backend.mujoco.xml import _get_named_bodies

from unilab.assets import ASSETS_ROOT_PATH

# MuJoCo root free-joint sizes
_ROOT_QPOS_DIM = 7  # 3 pos + 4 quat
_ROOT_QVEL_DIM = 6  # 3 lin vel + 3 ang vel

_BODY_ARRAY_KEYS = ("body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")


def _build_body_remap(source_names: list[str], target_names: list[str]) -> np.ndarray:
    """Return an index array such that ``source[:, remap]`` matches target layout."""
    src_lookup = {name: idx for idx, name in enumerate(source_names)}
    remap = np.empty(len(target_names), dtype=np.int32)
    for i, name in enumerate(target_names):
        if name not in src_lookup:
            raise ValueError(
                f"Target body '{name}' not found in source NPZ body_names. "
                f"Source has {len(source_names)} bodies."
            )
        remap[i] = src_lookup[name]
    return remap


def remap_npz(input_path: str, output_path: str, model_file: str, *, dry_run: bool = False) -> None:
    """Convert a full-body NPZ to the standard training layout."""
    data = np.load(input_path, allow_pickle=True)

    # --- source body names ---------------------------------------------------
    if "body_names" not in data:
        raise ValueError(
            f"Source NPZ '{input_path}' has no 'body_names' key. Cannot determine body remapping."
        )
    source_body_names: list[str] = data["body_names"].tolist()

    # --- target body list from training model --------------------------------
    _, named_bodies = _get_named_bodies(model_file)
    target_body_names = ["world"] + named_bodies  # prepend MuJoCo implicit body 0

    remap = _build_body_remap(source_body_names, target_body_names)

    # --- remap body arrays ----------------------------------------------------
    remapped: dict[str, np.ndarray] = {}
    for key in _BODY_ARRAY_KEYS:
        remapped[key] = data[key][:, remap].astype(np.float32)

    # --- strip root free-joint from joint arrays -----------------------------
    src_joint_pos = data["joint_pos"]
    src_joint_vel = data["joint_vel"]

    num_actuated_pos = src_joint_pos.shape[1] - _ROOT_QPOS_DIM
    num_actuated_vel = src_joint_vel.shape[1] - _ROOT_QVEL_DIM
    if num_actuated_pos != num_actuated_vel:
        raise ValueError(
            f"Actuated DOF mismatch after stripping root: "
            f"joint_pos gives {num_actuated_pos}, joint_vel gives {num_actuated_vel}"
        )

    joint_pos = src_joint_pos[:, _ROOT_QPOS_DIM:].astype(np.float32)
    joint_vel = src_joint_vel[:, _ROOT_QVEL_DIM:].astype(np.float32)

    fps = np.array([int(np.asarray(data["fps"]).reshape(-1)[0])], dtype=np.int32)

    # --- summary --------------------------------------------------------------
    print(f"Source : {input_path}")
    print(f"  bodies  : {len(source_body_names)} -> {len(target_body_names)}")
    print(f"  joint_pos: {src_joint_pos.shape} -> {joint_pos.shape}")
    print(f"  joint_vel: {src_joint_vel.shape} -> {joint_vel.shape}")
    print(f"  fps      : {fps[0]}")
    print(f"  frames   : {joint_pos.shape[0]}")

    if dry_run:
        print("[dry-run] Validation passed. No file written.")
        return

    np.savez(
        output_path,
        fps=fps,
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        **remapped,
    )
    print(f"Output : {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remap a full-body MuJoCo NPZ to the standard training layout."
    )
    parser.add_argument("--input", required=True, help="Source NPZ file path")
    parser.add_argument("--output", required=True, help="Output NPZ file path")
    parser.add_argument(
        "--model_file",
        default=None,
        help="Training-model XML (default: G1 scene_flat.xml)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate only; do not write output file",
    )
    args = parser.parse_args()

    if args.model_file is None:
        args.model_file = str(ASSETS_ROOT_PATH / "robots" / "g1" / "scene_flat.xml")

    model_path = Path(args.model_file).expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    remap_npz(args.input, args.output, str(model_path), dry_run=args.dry_run)


if __name__ == "__main__":
    main()
