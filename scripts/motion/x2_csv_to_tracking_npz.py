"""Convert an X2 root+joint CSV motion to UniLab motion-tracking NPZ format.

The source CSV is expected to contain one frame per row:

- root position xyz
- root quaternion xyzw
- 29 X2 joint positions, in MuJoCo joint order

The output layout matches the existing X2 ``*_g1format.npz`` assets consumed by
the shared humanoid motion-tracking loader.

Velocity estimation reuses the library implementation in
``unilab.tasks.motion_tracking.common.motion_loader``; forward kinematics reuse
``unisim.backend.mujoco.motion_export.compute_tracking_fk``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np
from unisim.backend.mujoco.motion_export import compute_tracking_fk

from unilab.assets import ASSETS_ROOT_PATH
from unilab.tasks.motion_tracking.common.motion_loader import (
    compute_motion_velocities,
    quat_slerp,
)
from unilab.utils.rotation import np_quat_ensure_continuity

ROOT_QPOS_DIM = 7
ROOT_QVEL_DIM = 6
DEFAULT_INPUT = ASSETS_ROOT_PATH / "motions" / "x2" / "csv" / "shangxiaoche_5-28_15.csv"
DEFAULT_OUTPUT = ASSETS_ROOT_PATH / "motions" / "x2" / "shangxiaoche_5-28_15_g1format.npz"
DEFAULT_MODEL = ASSETS_ROOT_PATH / "robots" / "x2" / "x2_simple_collision.xml"


def _load_csv_qpos(input_path: Path, model_nq: int) -> np.ndarray:
    try:
        raw = np.loadtxt(input_path, delimiter=",", dtype=np.float32)
    except ValueError:
        raw = np.loadtxt(input_path, delimiter=",", dtype=np.float32, skiprows=1)

    if raw.ndim == 1:
        raw = raw[None, :]
    if raw.ndim != 2:
        raise ValueError(f"Expected 2D CSV data, got shape {raw.shape}")
    if raw.shape[1] != model_nq:
        raise ValueError(f"Expected CSV width to match model nq={model_nq}, got {raw.shape[1]}")

    qpos = raw.astype(np.float32, copy=True)
    # Source CSV stores the root quaternion as xyzw; MuJoCo qpos uses wxyz.
    qpos[:, 3:7] = raw[:, 3:7][:, [3, 0, 1, 2]]
    qpos[:, 3:7] = np_quat_ensure_continuity(qpos[:, 3:7])
    quat_norm = np.linalg.norm(qpos[:, 3:7], axis=1, keepdims=True)
    if np.any(quat_norm <= 0.0):
        raise ValueError("Root quaternion contains zero-norm entries")
    qpos[:, 3:7] /= quat_norm
    return qpos


def _resample_qpos(qpos: np.ndarray, input_fps: int, output_fps: int) -> np.ndarray:
    if input_fps <= 0 or output_fps <= 0:
        raise ValueError(f"fps values must be positive, got {input_fps} -> {output_fps}")
    if input_fps == output_fps:
        return qpos.astype(np.float32, copy=True)
    if qpos.shape[0] <= 1:
        return qpos.astype(np.float32, copy=True)

    duration = (qpos.shape[0] - 1) / float(input_fps)
    output_times = np.arange(0.0, duration, 1.0 / float(output_fps), dtype=np.float32)
    if output_times.size == 0:
        return qpos[:1].astype(np.float32, copy=True)

    source_phase = output_times * float(input_fps)
    index_0 = np.floor(source_phase).astype(np.int32)
    index_1 = np.minimum(index_0 + 1, qpos.shape[0] - 1)
    blend = source_phase - index_0

    out = np.empty((output_times.shape[0], qpos.shape[1]), dtype=np.float32)
    out[:, :3] = qpos[index_0, :3] * (1.0 - blend[:, None]) + qpos[index_1, :3] * blend[:, None]
    for frame, t in enumerate(blend):
        # Interpolate in float64, then cast back to float32.
        out[frame, 3:7] = quat_slerp(
            qpos[index_0[frame], 3:7].astype(np.float64),
            qpos[index_1[frame], 3:7].astype(np.float64),
            float(t),
        ).astype(np.float32)
    out[:, 7:] = qpos[index_0, 7:] * (1.0 - blend[:, None]) + qpos[index_1, 7:] * blend[:, None]
    out[:, 3:7] = np_quat_ensure_continuity(out[:, 3:7])
    return out


def _qvel_from_qpos(qpos: np.ndarray, fps: int) -> np.ndarray:
    dt = 1.0 / float(fps)
    base_lin_vels, base_ang_vels, dof_vels = compute_motion_velocities(
        qpos[:, :3], qpos[:, 3:7], qpos[:, 7:], dt
    )
    qvel = np.empty((qpos.shape[0], qpos.shape[1] - 1), dtype=np.float32)
    qvel[:, :3] = base_lin_vels.astype(np.float32)
    qvel[:, 3:6] = base_ang_vels.astype(np.float32)
    qvel[:, 6:] = dof_vels.astype(np.float32)
    return qvel


def _target_joint_names(model: mujoco.MjModel) -> list[str]:
    names: list[str] = []
    for joint_id in range(model.njnt):
        if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if not name:
            raise ValueError(f"Joint id {joint_id} has no name")
        names.append(name)
    return names


def convert_csv(
    input_path: Path,
    output_path: Path,
    model_file: Path,
    input_fps: int,
    output_fps: int,
    dry_run: bool,
) -> None:
    model = mujoco.MjModel.from_xml_path(str(model_file))

    target_names = _target_joint_names(model)
    qpos_input = _load_csv_qpos(input_path, model.nq)
    qpos = _resample_qpos(qpos_input, input_fps, output_fps)
    qvel = _qvel_from_qpos(qpos, output_fps)

    if qpos.shape[1] - ROOT_QPOS_DIM != len(target_names):
        raise ValueError(
            f"CSV joint count {qpos.shape[1] - ROOT_QPOS_DIM} does not match model joints "
            f"{len(target_names)}"
        )

    print(f"Source : {input_path}")
    print(f"Model  : {model_file}")
    print(f"Output : {output_path}")
    print(f"frames : {qpos_input.shape[0]} -> {qpos.shape[0]}")
    print(f"fps    : {input_fps} -> {output_fps}")
    print(f"joints : {qpos.shape[1] - ROOT_QPOS_DIM}")
    print(f"bodies : {model.nbody} (MuJoCo body-id layout, including world)")

    if dry_run:
        print("[dry-run] Validation passed. No output written.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    arrays = compute_tracking_fk(
        str(model_file),
        joint_names=target_names,
        base_poss=qpos[:, :3],
        base_rots=qpos[:, 3:7],
        base_lin_vels=qvel[:, :3],
        base_ang_vels=qvel[:, 3:6],
        dof_poss=qpos[:, ROOT_QPOS_DIM:],
        dof_vels=qvel[:, ROOT_QVEL_DIM:],
    )
    np.savez(
        output_path,
        fps=np.array([output_fps], dtype=np.int32),
        **arrays,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert an X2 root+joint CSV to UniLab motion-tracking NPZ format."
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Source X2 CSV path")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output NPZ path")
    parser.add_argument(
        "--model-file",
        default=str(DEFAULT_MODEL),
        help="Target MuJoCo XML used to regenerate body_* arrays",
    )
    parser.add_argument("--input-fps", type=int, default=30, help="Source CSV FPS")
    parser.add_argument("--output-fps", type=int, default=50, help="Output NPZ FPS")
    parser.add_argument("--dry-run", action="store_true", help="Validate without writing output")
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    model_file = Path(args.model_file).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")
    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_file}")

    convert_csv(
        input_path=input_path,
        output_path=output_path,
        model_file=model_file,
        input_fps=args.input_fps,
        output_fps=args.output_fps,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
