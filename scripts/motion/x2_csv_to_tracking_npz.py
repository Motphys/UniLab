"""Convert an X2 root+joint CSV motion to UniLab motion-tracking NPZ format.

The source CSV is expected to contain one frame per row:

- root position xyz
- root quaternion xyzw
- 29 X2 joint positions, in MuJoCo joint order

The output layout matches the existing X2 ``*_g1format.npz`` assets consumed by
the shared humanoid motion-tracking loader.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base.backend.mujoco.xml import inject_mujoco_tracking_sensors
from unilab.envs.common.rotation import np_quat_angular_velocity, np_quat_ensure_continuity

ROOT_QPOS_DIM = 7
ROOT_QVEL_DIM = 6
DEFAULT_INPUT = ASSETS_ROOT_PATH / "motions" / "x2" / "csv" / "shangxiaoche_5-28_15.csv"
DEFAULT_OUTPUT = ASSETS_ROOT_PATH / "motions" / "x2" / "shangxiaoche_5-28_15_g1format.npz"
DEFAULT_MODEL = ASSETS_ROOT_PATH / "robots" / "x2" / "x2_simple_collision.xml"


def _quat_slerp(q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
    q1 = q1.astype(np.float64, copy=False)
    q2 = q2.astype(np.float64, copy=False)
    dot = float(np.dot(q1, q2))
    if dot < 0.0:
        q2 = -q2
        dot = -dot
    if dot > 0.9995:
        result = q1 + t * (q2 - q1)
        return (result / np.linalg.norm(result)).astype(np.float32)

    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta = np.sin(theta)
    w1 = np.sin((1.0 - t) * theta) / sin_theta
    w2 = np.sin(t * theta) / sin_theta
    return (w1 * q1 + w2 * q2).astype(np.float32)


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
        out[frame, 3:7] = _quat_slerp(
            qpos[index_0[frame], 3:7],
            qpos[index_1[frame], 3:7],
            float(t),
        )
    out[:, 7:] = qpos[index_0, 7:] * (1.0 - blend[:, None]) + qpos[index_1, 7:] * blend[:, None]
    out[:, 3:7] = np_quat_ensure_continuity(out[:, 3:7])
    return out


def _qvel_from_qpos(qpos: np.ndarray, fps: int) -> np.ndarray:
    dt = 1.0 / float(fps)
    qvel = np.empty((qpos.shape[0], qpos.shape[1] - 1), dtype=np.float32)
    qvel[:, :3] = np.gradient(qpos[:, :3], dt, axis=0).astype(np.float32)
    qvel[:, 3:6] = np_quat_angular_velocity(qpos[:, 3:7], dt).astype(np.float32)
    qvel[:, 6:] = np.gradient(qpos[:, 7:], dt, axis=0).astype(np.float32)
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


def _sensor_addresses(model: mujoco.MjModel) -> np.ndarray:
    sensor_adrs = np.full((model.nbody, 4), -1, dtype=np.int32)
    prefixes = (
        "track_pos_w_",
        "track_quat_w_",
        "track_linvel_w_",
        "track_angvel_w_",
    )
    for body_id in range(model.nbody):
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        if not body_name:
            continue
        for slot, prefix in enumerate(prefixes):
            sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, prefix + body_name)
            if sensor_id >= 0:
                sensor_adrs[body_id, slot] = int(model.sensor_adr[sensor_id])
    return sensor_adrs


def _export_body_arrays(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    qvel: np.ndarray,
    target_joint_names: list[str],
) -> dict[str, np.ndarray]:
    data = mujoco.MjData(model)
    frames = qpos.shape[0]
    body_pos_w = np.zeros((frames, model.nbody, 3), dtype=np.float32)
    body_quat_w = np.zeros((frames, model.nbody, 4), dtype=np.float32)
    body_lin_vel_w = np.zeros((frames, model.nbody, 3), dtype=np.float32)
    body_ang_vel_w = np.zeros((frames, model.nbody, 3), dtype=np.float32)
    sensor_adrs = _sensor_addresses(model)

    joint_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in target_joint_names
    ]
    if any(joint_id < 0 for joint_id in joint_ids):
        raise ValueError("Target joint list contains joints not found in model")

    for frame in range(frames):
        data.qpos[:] = qpos[frame]
        data.qvel[:] = qvel[frame]
        mujoco.mj_forward(model, data)

        for body_id in range(model.nbody):
            pos_adr, quat_adr, lin_adr, ang_adr = sensor_adrs[body_id]
            if pos_adr >= 0:
                body_pos_w[frame, body_id] = data.sensordata[pos_adr : pos_adr + 3]
            else:
                body_pos_w[frame, body_id] = data.xpos[body_id]

            if quat_adr >= 0:
                body_quat_w[frame, body_id] = data.sensordata[quat_adr : quat_adr + 4]
            else:
                body_quat_w[frame, body_id] = data.xquat[body_id]

            if lin_adr >= 0:
                body_lin_vel_w[frame, body_id] = data.sensordata[lin_adr : lin_adr + 3]
            if ang_adr >= 0:
                body_ang_vel_w[frame, body_id] = data.sensordata[ang_adr : ang_adr + 3]

    return {
        "body_pos_w": body_pos_w,
        "body_quat_w": body_quat_w,
        "body_lin_vel_w": body_lin_vel_w,
        "body_ang_vel_w": body_ang_vel_w,
    }


def convert_csv(
    input_path: Path,
    output_path: Path,
    model_file: Path,
    input_fps: int,
    output_fps: int,
    dry_run: bool,
) -> None:
    tmp_model_path, _, _ = inject_mujoco_tracking_sensors(str(model_file))
    try:
        model = mujoco.MjModel.from_xml_path(tmp_model_path)
    finally:
        Path(tmp_model_path).unlink(missing_ok=True)

    target_names = _target_joint_names(model)
    qpos_input = _load_csv_qpos(input_path, model.nq)
    qpos = _resample_qpos(qpos_input, input_fps, output_fps)
    qvel = _qvel_from_qpos(qpos, output_fps)
    joint_pos = qpos[:, ROOT_QPOS_DIM:].astype(np.float32)
    joint_vel = qvel[:, ROOT_QVEL_DIM:].astype(np.float32)

    if joint_pos.shape[1] != len(target_names):
        raise ValueError(
            f"CSV joint count {joint_pos.shape[1]} does not match model joints {len(target_names)}"
        )

    print(f"Source : {input_path}")
    print(f"Model  : {model_file}")
    print(f"Output : {output_path}")
    print(f"frames : {qpos_input.shape[0]} -> {qpos.shape[0]}")
    print(f"fps    : {input_fps} -> {output_fps}")
    print(f"joints : {joint_pos.shape[1]}")
    print(f"bodies : {model.nbody} (MuJoCo body-id layout, including world)")

    if dry_run:
        print("[dry-run] Validation passed. No output written.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    body_arrays = _export_body_arrays(model, qpos, qvel, target_names)
    np.savez(
        output_path,
        fps=np.array([output_fps], dtype=np.int32),
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        **body_arrays,
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
