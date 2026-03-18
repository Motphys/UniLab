"""Convert CSV motion files to NPZ format with forward kinematics.

This script converts motion data from CSV format (Unitree convention) to NPZ format
with precomputed forward kinematics for all bodies.

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
"""

import argparse
from pathlib import Path

import mujoco
import numpy as np
from tqdm import tqdm

from unilab.assets import ASSETS_ROOT_PATH


def quat_slerp(q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation between two quaternions (wxyz format)."""
    # Ensure shortest path
    dot = np.dot(q1, q2)
    if dot < 0:
        q2 = -q2
        dot = -dot

    # If quaternions are very close, use linear interpolation
    if dot > 0.9995:
        result = q1 + t * (q2 - q1)
        return result / np.linalg.norm(result)

    # Compute angle
    theta = np.arccos(np.clip(dot, -1, 1))
    sin_theta = np.sin(theta)

    # Compute interpolation weights
    w1 = np.sin((1 - t) * theta) / sin_theta
    w2 = np.sin(t * theta) / sin_theta

    return w1 * q1 + w2 * q2


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    """Compute quaternion conjugate (wxyz format)."""
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply two quaternions (wxyz format)."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def axis_angle_from_quat(q: np.ndarray) -> np.ndarray:
    """Convert quaternion to axis-angle representation (wxyz format)."""
    # Normalize
    q = q / np.linalg.norm(q)
    w, x, y, z = q

    # Compute angle
    angle = 2 * np.arccos(np.clip(w, -1, 1))

    # Compute axis
    sin_half = np.sin(angle / 2)
    if abs(sin_half) < 1e-6:
        # No rotation
        return np.zeros(3)

    axis = np.array([x, y, z]) / sin_half
    return axis * angle


class MotionLoader:
    """Load and interpolate motion from CSV file."""

    def __init__(
        self,
        motion_file: str,
        input_fps: int,
        output_fps: int,
        line_range: tuple[int, int] | None = None,
    ):
        self.motion_file = motion_file
        self.input_fps = input_fps
        self.output_fps = output_fps
        self.input_dt = 1.0 / self.input_fps
        self.output_dt = 1.0 / self.output_fps
        self.line_range = line_range
        self._load_motion()
        self._interpolate_motion()
        self._compute_velocities()

    def _load_motion(self):
        """Load motion from CSV file."""
        if self.line_range is None:
            motion = np.loadtxt(self.motion_file, delimiter=",", dtype=np.float32)
        else:
            motion = np.loadtxt(
                self.motion_file,
                delimiter=",",
                skiprows=self.line_range[0] - 1,
                max_rows=self.line_range[1] - self.line_range[0] + 1,
                dtype=np.float32,
            )

        self.motion_base_poss_input = motion[:, :3]
        # Convert quaternion from xyzw to wxyz
        self.motion_base_rots_input = motion[:, 3:7][:, [3, 0, 1, 2]]
        self.motion_dof_poss_input = motion[:, 7:]

        self.input_frames = motion.shape[0]
        self.duration = (self.input_frames - 1) * self.input_dt

    def _interpolate_motion(self):
        """Interpolate motion to output FPS."""
        times = np.arange(0, self.duration, self.output_dt, dtype=np.float32)
        self.output_frames = times.shape[0]
        index_0, index_1, blend = self._compute_frame_blend(times)

        # Linear interpolation for positions
        self.motion_base_poss = (
            self.motion_base_poss_input[index_0] * (1 - blend[:, None])
            + self.motion_base_poss_input[index_1] * blend[:, None]
        )

        # Spherical linear interpolation for quaternions
        self.motion_base_rots = np.zeros((self.output_frames, 4), dtype=np.float32)
        for i in range(self.output_frames):
            self.motion_base_rots[i] = quat_slerp(
                self.motion_base_rots_input[index_0[i]],
                self.motion_base_rots_input[index_1[i]],
                blend[i],
            )

        # Linear interpolation for joint positions
        self.motion_dof_poss = (
            self.motion_dof_poss_input[index_0] * (1 - blend[:, None])
            + self.motion_dof_poss_input[index_1] * blend[:, None]
        )

        print(
            f"Motion interpolated: {self.input_frames} frames @ {self.input_fps} Hz "
            f"→ {self.output_frames} frames @ {self.output_fps} Hz"
        )

    def _compute_frame_blend(self, times: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute frame indices and blend weights for interpolation."""
        phase = times / self.duration
        index_0 = np.floor(phase * (self.input_frames - 1)).astype(np.int32)
        index_1 = np.minimum(index_0 + 1, self.input_frames - 1)
        blend = phase * (self.input_frames - 1) - index_0
        return index_0, index_1, blend

    def _compute_velocities(self):
        """Compute velocities using numerical differentiation."""
        # Linear velocities
        self.motion_base_lin_vels = np.gradient(self.motion_base_poss, self.output_dt, axis=0)
        self.motion_dof_vels = np.gradient(self.motion_dof_poss, self.output_dt, axis=0)

        # Angular velocities from quaternion derivatives
        self.motion_base_ang_vels = self._so3_derivative(self.motion_base_rots, self.output_dt)

    def _so3_derivative(self, rotations: np.ndarray, dt: float) -> np.ndarray:
        """Compute angular velocity from quaternion sequence."""
        # Use central differences
        q_prev = rotations[:-2]
        q_next = rotations[2:]

        # Compute relative quaternion
        omega = np.zeros((rotations.shape[0], 3), dtype=np.float32)
        for i in range(len(q_prev)):
            q_rel = quat_mul(q_next[i], quat_conjugate(q_prev[i]))
            omega[i + 1] = axis_angle_from_quat(q_rel) / (2.0 * dt)

        # Repeat first and last
        omega[0] = omega[1]
        omega[-1] = omega[-2]

        return omega


def run_simulation(
    motion_loader: MotionLoader,
    model_file: str,
    joint_names: list[str],
    output_file: str,
):
    """Run MuJoCo simulation to compute forward kinematics."""
    # Load model
    model = mujoco.MjModel.from_xml_path(model_file)
    data = mujoco.MjData(model)

    # Get joint indices
    joint_indices = []
    for name in joint_names:
        jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jnt_id < 0:
            raise ValueError(f"Joint '{name}' not found in model")
        joint_indices.append(jnt_id)

    # Prepare output arrays
    num_frames = motion_loader.output_frames
    num_joints = len(joint_indices)
    num_bodies = model.nbody

    joint_pos = np.zeros((num_frames, num_joints), dtype=np.float32)
    joint_vel = np.zeros((num_frames, num_joints), dtype=np.float32)
    body_pos_w = np.zeros((num_frames, num_bodies, 3), dtype=np.float32)
    body_quat_w = np.zeros((num_frames, num_bodies, 4), dtype=np.float32)
    body_lin_vel_w = np.zeros((num_frames, num_bodies, 3), dtype=np.float32)
    body_ang_vel_w = np.zeros((num_frames, num_bodies, 3), dtype=np.float32)

    print(f"\nProcessing {num_frames} frames...")
    for i in tqdm(range(num_frames)):
        # Set root state
        data.qpos[0:3] = motion_loader.motion_base_poss[i]
        data.qpos[3:7] = motion_loader.motion_base_rots[i]
        data.qvel[0:3] = motion_loader.motion_base_lin_vels[i]
        data.qvel[3:6] = motion_loader.motion_base_ang_vels[i]

        # Set joint states
        for j, jnt_id in enumerate(joint_indices):
            qpos_adr = model.jnt_qposadr[jnt_id]
            qvel_adr = model.jnt_dofadr[jnt_id]
            data.qpos[qpos_adr] = motion_loader.motion_dof_poss[i, j]
            data.qvel[qvel_adr] = motion_loader.motion_dof_vels[i, j]

        # Run forward kinematics
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comVel(model, data)

        # Extract joint states
        for j, jnt_id in enumerate(joint_indices):
            qpos_adr = model.jnt_qposadr[jnt_id]
            qvel_adr = model.jnt_dofadr[jnt_id]
            joint_pos[i, j] = data.qpos[qpos_adr]
            joint_vel[i, j] = data.qvel[qvel_adr]

        # Extract body states
        for body_id in range(num_bodies):
            body_pos_w[i, body_id] = data.xpos[body_id]
            body_quat_w[i, body_id] = data.xquat[body_id]
            body_lin_vel_w[i, body_id] = data.cvel[body_id, 3:6]
            body_ang_vel_w[i, body_id] = data.cvel[body_id, 0:3]

    # Save to NPZ
    print(f"\nSaving to {output_file}...")
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
    print("Done!")


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

    # Load and interpolate motion
    motion_loader = MotionLoader(
        args.input_file,
        int(args.input_fps),
        int(args.output_fps),
        tuple(args.line_range) if args.line_range else None,
    )

    # Run simulation
    run_simulation(motion_loader, args.model_file, joint_names, args.output_file)


if __name__ == "__main__":
    main()
