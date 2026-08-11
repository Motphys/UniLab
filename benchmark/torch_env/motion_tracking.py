"""G1MotionTrackingSAC (SAC/mujoco) update_state / reset_done workload.

Faithful xp-port of the NumPy computation in the collector-timed sections of
`uv run train --algo sac --task g1_motion_tracking --sim mujoco`
(num_envs=2048, 29-dof, 14 tracked bodies):

- `MotionTrackingEnv.update_state`
  (src/unilab/envs/motion_tracking/common/tracking.py):
  motion gather, relative transforms (transforms.py), terminations
  (terminations.py), 9 active reward terms (rewards.py, incl. per-term logging
  every 4 steps), observation build (observations.py, actor 160 / critic 289
  with the SAC +3 linvel tail), adaptive motion-sampler bookkeeping
  (motion_loader.py).
- `MotionTrackingDomainRandomizationProvider.build_reset_plan` /
  `build_reset_observation` + `build_motion_reference_state` (reset.py).
- `NpEnv._reset_done_envs` scatter/gather.

Excluded (identical across variants, not NumPy/Torch env math):
`backend.step` physics, `backend.set_state`, sensor/body-state reads (replaced
by persistent arrays), and the adaptive-sampler entropy metrics (3 scalar
reductions per reset).

The real `build_motion_reference_state` samples pose/velocity randomization
with a per-element Python loop (num_reset x 6 draws, twice). The NumPy
workload reproduces that faithfully; pass ``vectorized_reset_rng=True`` for a
column-wise vectorized NumPy draw (used for cross-backend RNG replay during
validation, which is also the only mode Torch implements).
"""

from __future__ import annotations

import numpy as np

N_ENVS = 2048
N_ACTION = 29
N_BODY = 14
NQ = 36
NV = 35
OBS_DIM = 160
CRITIC_DIM = 289
N_FRAMES = 870
CLIP_END_FRAME = 869
N_BINS = 18
ANCHOR_IDX = 7
EE_INDICES = (3, 6, 10, 13)
UNDESIRED_INDICES = (0, 1, 2, 4, 5, 7, 8, 9, 11, 12)

ANCHOR_POS_Z_THRESHOLD = 0.5
ANCHOR_ORI_THRESHOLD = 0.8
EE_BODY_POS_Z_THRESHOLD = 0.5
UNDESIRED_CONTACT_Z_THRESHOLD = 0.05
CTRL_DT = 0.02
SAMPLER_UNIFORM_RATIO = 0.1
SAMPLER_ALPHA = 0.001
JOINT_POSITION_RANGE = (-0.1, 0.1)
POSE_RANGES = ((-0.05, 0.05), (-0.05, 0.05), (-0.01, 0.01), (-0.1, 0.1), (-0.1, 0.1), (-0.2, 0.2))
VEL_RANGES = ((-0.5, 0.5), (-0.5, 0.5), (-0.2, 0.2), (-0.52, 0.52), (-0.52, 0.52), (-0.78, 0.78))
NOISE_LEVEL = 1.0
NOISE_SCALES = {"linvel": 0.1, "gyro": 0.2, "joint_angle": 0.01, "joint_vel": 1.5}

# (name, scale, 1/std^2) in YAML dispatch order; scale==0 terms are skipped by
# the real dispatcher and therefore absent here.
REWARD_TERMS = (
    ("motion_global_root_pos", 1.0, 1.0 / 0.3**2),
    ("motion_global_root_ori", 0.5, 1.0 / 0.4**2),
    ("motion_body_pos", 2.0, 1.0 / 0.3**2),
    ("motion_body_ori", 1.0, 1.0 / 0.4**2),
    ("motion_body_lin_vel", 1.0, 1.0 / 1.0**2),
    ("motion_body_ang_vel", 1.0, 1.0 / 3.14**2),
)


def _quat_mul(b, q1, q2):
    """Hamilton product, w-first, broadcasting over leading dims."""
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    return b.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        axis=-1,
    )


def _quat_from_euler_xyz(b, roll, pitch, yaw):
    cr, sr = b.cos(roll * 0.5), b.sin(roll * 0.5)
    cp, sp = b.cos(pitch * 0.5), b.sin(pitch * 0.5)
    cy, sy = b.cos(yaw * 0.5), b.sin(yaw * 0.5)
    return b.stack(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        axis=-1,
    )


def _quat_apply(b, q, v):
    """Rotate vectors v (..., 3) by quaternions q (..., 4), w-first."""
    qw = q[..., 0:1]
    qv = q[..., 1:4]
    # t = 2 * cross(qv, v)
    tx = 2 * (qv[..., 1:2] * v[..., 2:3] - qv[..., 2:3] * v[..., 1:2])
    ty = 2 * (qv[..., 2:3] * v[..., 0:1] - qv[..., 0:1] * v[..., 2:3])
    tz = 2 * (qv[..., 0:1] * v[..., 1:2] - qv[..., 1:2] * v[..., 0:1])
    # v + qw * t + cross(qv, t)
    return b.concat(
        [
            v[..., 0:1] + qw * tx + qv[..., 1:2] * tz - qv[..., 2:3] * ty,
            v[..., 1:2] + qw * ty + qv[..., 2:3] * tx - qv[..., 0:1] * tz,
            v[..., 2:3] + qw * tz + qv[..., 0:1] * ty - qv[..., 1:2] * tx,
        ],
        axis=-1,
    )


def _quat_inv(b, q):
    return b.concat([q[..., 0:1], -q[..., 1:4]], axis=-1)


def _gravity_z_in_body(q):
    return 2 * (q[..., 1] * q[..., 1] + q[..., 2] * q[..., 2]) - 1


class MotionTrackingWorkload:
    def __init__(
        self, b, rng, seed: int = 0, vectorized_reset_rng: bool = False, num_envs: int = N_ENVS
    ):
        self.b = b
        self.rng = rng
        self.vectorized_reset_rng = vectorized_reset_rng
        self.n_envs = num_envs
        rs = np.random.RandomState(seed)

        def f32(*shape):
            return rs.standard_normal(shape).astype(np.float32)

        def rand_quat(*shape):
            q = f32(*shape)
            q /= np.linalg.norm(q, axis=-1, keepdims=True)
            return q.astype(np.float32)

        n = num_envs
        self.default_angles = b.convert(rs.uniform(-0.5, 0.5, N_ACTION).astype(np.float32))
        da = b.to_numpy(self.default_angles)
        self.joint_lower = b.convert(da - 1.0)
        self.joint_upper = b.convert(da + 1.0)

        # Motion store (one clip, 870 frames @ fps 50).
        self.m_joint_pos = b.convert(f32(N_FRAMES, N_ACTION) * 0.3)
        self.m_joint_vel = b.convert(f32(N_FRAMES, N_ACTION))
        body_pos = f32(N_FRAMES, N_BODY, 3) * 0.2
        body_pos[..., 2] = np.abs(body_pos[..., 2]) + 0.15
        self.m_body_pos = b.convert(body_pos)
        self.m_body_quat = b.convert(rand_quat(N_FRAMES, N_BODY, 4))
        self.m_body_lin_vel = b.convert(f32(N_FRAMES, N_BODY, 3) * 0.5)
        self.m_body_ang_vel = b.convert(f32(N_FRAMES, N_BODY, 3))

        # Robot state (backend reads replaced by persistent arrays).
        self.body_pos = b.convert(b.to_numpy(self.m_body_pos)[0:1] + f32(n, N_BODY, 3) * 0.02)
        self.body_quat = b.convert(rand_quat(n, N_BODY, 4))
        self.body_lin_vel = b.convert(f32(n, N_BODY, 3) * 0.5)
        self.body_ang_vel = b.convert(f32(n, N_BODY, 3))
        # Force a few anchor-z mismatches so the terminated branch stays live.
        forced = b.to_numpy(self.body_pos)
        forced[:8, ANCHOR_IDX, 2] += 1.0
        self.body_pos = b.convert(forced)
        self.linvel = b.convert(f32(n, 3) * 0.5)
        self.gyro = b.convert(f32(n, 3) * 0.5)
        self.dof_pos = b.convert(da[None, :] + f32(n, N_ACTION) * 0.1)
        self.dof_vel = b.convert(f32(n, N_ACTION))

        # info / sampler state.
        self.current_actions = b.convert(f32(n, N_ACTION) * 0.3)
        self.last_actions = b.convert(f32(n, N_ACTION) * 0.3)
        self.steps = b.index(rs.randint(0, 500, n))
        self.current_frames = b.index(rs.randint(0, 200, n))
        self.bin_failed_count = b.convert(np.full(N_BINS, 0.05, dtype=np.float32))
        self.clip_end_truncated = b.zeros((n,)) > 0.0  # bool array

        # Reset plan constants (keyframe "stand").
        init_qpos = np.zeros(NQ, dtype=np.float32)
        init_qpos[2] = 0.754
        init_qpos[3] = 1.0
        init_qpos[7:] = da
        self.init_qpos = b.convert(init_qpos)

        # Step outputs kept across calls for the reset scatter/gather path.
        self.obs = {
            "obs": b.zeros((n, OBS_DIM)),
            "critic": b.zeros((n, CRITIC_DIM)),
        }
        self.final_obs = {
            "obs": b.zeros((n, OBS_DIM)),
            "critic": b.zeros((n, CRITIC_DIM)),
        }
        self.compat_final_obs = {
            "obs": b.zeros((n, OBS_DIM)),
            "critic": b.zeros((n, CRITIC_DIM)),
        }

    # ------------------------------------------------------------- gathers --
    def _motion_at_frames(self, frames):
        b = self.b
        return (
            b.take(self.m_joint_pos, frames),
            b.take(self.m_joint_vel, frames),
            b.take(self.m_body_pos, frames),
            b.take(self.m_body_quat, frames),
            b.take(self.m_body_lin_vel, frames),
            b.take(self.m_body_ang_vel, frames),
        )

    # ----------------------------------------------------------- transforms --
    def _update_relative_transforms(self, m_body_pos, m_body_quat):
        b = self.b
        anchor_pos = m_body_pos[:, ANCHOR_IDX]
        anchor_quat = m_body_quat[:, ANCHOR_IDX]
        r_anchor_pos = self.body_pos[:, ANCHOR_IDX]
        r_anchor_quat = self.body_quat[:, ANCHOR_IDX]

        delta_pos_z = anchor_pos[:, 2]
        delta_pos = b.stack([r_anchor_pos[:, 0], r_anchor_pos[:, 1], delta_pos_z], axis=1)

        # yaw-only relative rotation: yaw_quat(r_quat x conj(anchor_quat)).
        rw, rx, ry, rz = (r_anchor_quat[:, i] for i in range(4))
        aw, ax, ay, az = (anchor_quat[:, i] for i in range(4))
        qw = rw * aw + rx * ax + ry * ay + rz * az
        qx = -rw * ax + rx * aw - ry * az + rz * ay
        qy = -rw * ay + rx * az + ry * aw - rz * ax
        qz = -rw * az - rx * ay + ry * ax + rz * aw
        half_yaw = 0.5 * b.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
        dw = b.cos(half_yaw)
        dz = b.sin(half_yaw)

        mw = m_body_quat[..., 0]
        mx = m_body_quat[..., 1]
        my = m_body_quat[..., 2]
        mz = m_body_quat[..., 3]
        dw_c = dw[:, None]
        dz_c = dz[:, None]
        body_quat_relative = b.stack(
            [
                dw_c * mw - dz_c * mz,
                dw_c * mx - dz_c * my,
                dw_c * my + dz_c * mx,
                dw_c * mz + dz_c * mw,
            ],
            axis=-1,
        )

        rel = m_body_pos - anchor_pos[:, None, :]
        yc = (2 * dw * dz)[:, None]
        yz2 = (2 * dz * dz)[:, None]
        vx, vy, vz = rel[..., 0], rel[..., 1], rel[..., 2]
        body_pos_relative = b.stack(
            [
                vx - yc * vy - yz2 * vx + delta_pos[:, 0:1],
                vy + yc * vx - yz2 * vy + delta_pos[:, 1:2],
                vz + delta_pos[:, 2:3],
            ],
            axis=-1,
        )
        return body_pos_relative, body_quat_relative

    # ---------------------------------------------------------- terminations --
    def _compute_terminations(self, m_body_pos, m_body_quat, body_pos_relative):
        b = self.b
        anchor_pos = m_body_pos[:, ANCHOR_IDX]
        anchor_quat = m_body_quat[:, ANCHOR_IDX]
        r_anchor_pos = self.body_pos[:, ANCHOR_IDX]
        r_anchor_quat = self.body_quat[:, ANCHOR_IDX]

        terminated = b.abs(anchor_pos[:, 2] - r_anchor_pos[:, 2]) > ANCHOR_POS_Z_THRESHOLD
        ori_err = b.abs(_gravity_z_in_body(anchor_quat) - _gravity_z_in_body(r_anchor_quat))
        terminated = b.logical_or(terminated, ori_err > ANCHOR_ORI_THRESHOLD)
        ee = list(EE_INDICES)
        ee_err = b.abs(body_pos_relative[:, ee, 2] - self.body_pos[:, ee, 2])
        terminated = b.logical_or(terminated, b.any(ee_err > EE_BODY_POS_Z_THRESHOLD, axis=1))
        return terminated

    # --------------------------------------------------------------- reward --
    def _compute_reward(
        self,
        should_log,
        m_body_pos,
        m_body_quat,
        m_body_lin_vel,
        m_body_ang_vel,
        body_pos_relative,
        body_quat_relative,
    ):
        b = self.b
        terms = {}

        anchor_err = b.sum(
            b.square(m_body_pos[:, ANCHOR_IDX] - self.body_pos[:, ANCHOR_IDX]), axis=1
        )
        terms["motion_global_root_pos"] = anchor_err

        aq = m_body_quat[:, ANCHOR_IDX]
        rq = self.body_quat[:, ANCHOR_IDX]
        d = b.clip(b.abs(b.sum(aq * rq, axis=1)), 0.0, 1.0)
        terms["motion_global_root_ori"] = b.square(2 * b.arccos(d))

        pos_err = b.square(body_pos_relative - self.body_pos)
        terms["motion_body_pos"] = b.sum(pos_err, axis=(1, 2)) / N_BODY

        d_all = b.clip(b.abs(b.sum(body_quat_relative * self.body_quat, axis=-1)), 0.0, 1.0)
        ori_err_all = b.square(2 * b.arccos(d_all))
        terms["motion_body_ori"] = b.sum(ori_err_all, axis=1) / N_BODY

        lin_err = b.square(m_body_lin_vel - self.body_lin_vel)
        terms["motion_body_lin_vel"] = b.sum(lin_err, axis=(1, 2)) / N_BODY
        ang_err = b.square(m_body_ang_vel - self.body_ang_vel)
        terms["motion_body_ang_vel"] = b.sum(ang_err, axis=(1, 2)) / N_BODY

        reward = b.zeros((self.body_pos.shape[0],))
        log = {}
        for name, scale, inv_var in REWARD_TERMS:
            weighted = b.exp(terms[name] * (-inv_var)) * scale
            reward = reward + weighted
            if should_log:
                log[f"reward/{name}"] = b.scalar(b.mean(weighted))

        ar = b.sum(b.square(self.current_actions - self.last_actions), axis=1)
        weighted = ar * -0.1
        reward = reward + weighted
        if should_log:
            log["reward/action_rate_l2"] = b.scalar(b.mean(weighted))

        viol = b.maximum(self.joint_lower - self.dof_pos, 0.0) + b.maximum(
            self.dof_pos - self.joint_upper, 0.0
        )
        weighted = b.sum(b.square(viol), axis=1) * -2.0
        reward = reward + weighted
        if should_log:
            log["reward/joint_limit"] = b.scalar(b.mean(weighted))

        undesired = list(UNDESIRED_INDICES)
        contacts = self.body_pos[:, undesired, 2] < UNDESIRED_CONTACT_Z_THRESHOLD
        weighted = b.float_(b.sum(contacts, axis=1)) * -0.1
        reward = reward + weighted
        if should_log:
            log["reward/undesired_contacts"] = b.scalar(b.mean(weighted))

        self._last_log = log
        return reward * CTRL_DT

    # ------------------------------------------------------------------ obs --
    def _obs_noise(self, data, scale):
        noise = self.b.float_(self.rng.uniform(-1.0, 1.0, tuple(data.shape)))
        return data + noise * NOISE_LEVEL * scale

    def _compute_obs(
        self, motion, linvel, gyro, dof_pos, dof_vel, body_pos, body_quat, last_actions
    ):
        b = self.b
        m_joint_pos, m_joint_vel, m_body_pos, m_body_quat = motion[:4]
        m = linvel.shape[0]

        anchor_pos = m_body_pos[:, ANCHOR_IDX]
        anchor_quat = m_body_quat[:, ANCHOR_IDX]
        r_anchor_pos = body_pos[:, ANCHOR_IDX]
        r_anchor_quat = body_quat[:, ANCHOR_IDX]

        # Motion anchor pose in robot anchor frame (np_write_relative_anchor_
        # transform_pos_rot6d, geometry.py).
        aw, ax, ay, az = (r_anchor_quat[:, i] for i in range(4))
        vx = anchor_pos[:, 0] - r_anchor_pos[:, 0]
        vy = anchor_pos[:, 1] - r_anchor_pos[:, 1]
        vz = anchor_pos[:, 2] - r_anchor_pos[:, 2]
        qx, qy, qz = -ax, -ay, -az
        tx = 2 * (qy * vz - qz * vy)
        ty = 2 * (qz * vx - qx * vz)
        tz = 2 * (qx * vy - qy * vx)
        pos_b = b.stack(
            [
                vx + aw * tx + qy * tz - qz * ty,
                vy + aw * ty + qz * tx - qx * tz,
                vz + aw * tz + qx * ty - qy * tx,
            ],
            axis=1,
        )
        bw, bx, by, bz = (anchor_quat[:, i] for i in range(4))
        rw = aw * bw + ax * bx + ay * by + az * bz
        rx = aw * bx - ax * bw - ay * bz + az * by
        ry = aw * by + ax * bz - ay * bw - az * bx
        rz = aw * bz - ax * by + ay * bx - az * bw
        ori6 = b.stack(
            [
                1 - 2 * (ry * ry + rz * rz),
                2 * (rx * ry - rw * rz),
                2 * (rx * ry + rw * rz),
                1 - 2 * (rx * rx + rz * rz),
                2 * (rx * rz - rw * ry),
                2 * (ry * rz + rw * rx),
            ],
            axis=1,
        )

        joint_pos_rel = dof_pos - self.default_angles
        command = b.concat([m_joint_pos, m_joint_vel], axis=1)

        noisy_linvel = self._obs_noise(linvel, NOISE_SCALES["linvel"])
        noisy_gyro = self._obs_noise(gyro, NOISE_SCALES["gyro"])
        noisy_jpr = self._obs_noise(joint_pos_rel, NOISE_SCALES["joint_angle"])
        noisy_dv = self._obs_noise(dof_vel, NOISE_SCALES["joint_vel"])

        actor = b.concat(
            [command, pos_b, ori6, noisy_linvel, noisy_gyro, noisy_jpr, noisy_dv, last_actions],
            axis=1,
        )

        # Critic: clean channels + privileged body transforms in anchor frame.
        # Matches observations.write_body_pos_in_anchor_frame exactly.
        aqw = r_anchor_quat[:, None, 0:1]
        aqv = r_anchor_quat[:, None, 1:4]
        rel = body_pos - r_anchor_pos[:, None, :]
        txx = 2 * (aqv[..., 2:3] * rel[..., 1:2] - aqv[..., 1:2] * rel[..., 2:3])
        tyy = 2 * (aqv[..., 0:1] * rel[..., 2:3] - aqv[..., 2:3] * rel[..., 0:1])
        tzz = 2 * (aqv[..., 1:2] * rel[..., 0:1] - aqv[..., 0:1] * rel[..., 1:2])
        body_pos_b = b.concat(
            [
                rel[..., 0:1] + aqw * txx + aqv[..., 2:3] * tyy - aqv[..., 1:2] * tzz,
                rel[..., 1:2] + aqw * tyy + aqv[..., 0:1] * tzz - aqv[..., 2:3] * txx,
                rel[..., 2:3] + aqw * tzz + aqv[..., 1:2] * txx - aqv[..., 0:1] * tyy,
            ],
            axis=-1,
        )
        rel_quat = _quat_mul(b, _quat_inv(b, r_anchor_quat[:, None, :]), body_quat)
        rw2 = rel_quat[..., 0]
        rx2 = rel_quat[..., 1]
        ry2 = rel_quat[..., 2]
        rz2 = rel_quat[..., 3]
        body_ori6_b = b.stack(
            [
                1 - 2 * (ry2 * ry2 + rz2 * rz2),
                2 * (rx2 * ry2 - rw2 * rz2),
                2 * (rx2 * ry2 + rw2 * rz2),
                1 - 2 * (rx2 * rx2 + rz2 * rz2),
                2 * (rx2 * rz2 - rw2 * ry2),
                2 * (ry2 * rz2 + rw2 * rx2),
            ],
            axis=-1,
        )
        critic = b.concat(
            [
                command,
                pos_b,
                ori6,
                linvel,
                gyro,
                joint_pos_rel,
                dof_vel,
                last_actions,
                body_pos_b.reshape(m, N_BODY * 3),
                body_ori6_b.reshape(m, N_BODY * 6),
                linvel,  # SAC critic tail (+3)
            ],
            axis=1,
        )
        return {"obs": actor, "critic": critic}

    # --------------------------------------------------------- update_state --
    def update_state(self, should_log: bool):
        b = self.b
        motion = self._motion_at_frames(self.current_frames)
        m_body_pos, m_body_quat, m_lin, m_ang = motion[2], motion[3], motion[4], motion[5]

        body_pos_relative, body_quat_relative = self._update_relative_transforms(
            m_body_pos, m_body_quat
        )
        terminated = self._compute_terminations(m_body_pos, m_body_quat, body_pos_relative)
        reward = self._compute_reward(
            should_log,
            m_body_pos,
            m_body_quat,
            m_lin,
            m_ang,
            body_pos_relative,
            body_quat_relative,
        )
        obs = self._compute_obs(
            motion,
            self.linvel,
            self.gyro,
            self.dof_pos,
            self.dof_vel,
            self.body_pos,
            self.body_quat,
            self.current_actions,
        )
        self.obs = obs

        # MotionSampler.update_failure_stats (adaptive mode).
        if b.any_scalar(terminated):
            bin_idx = b.clip((self.current_frames * N_BINS) // N_FRAMES, 0, N_BINS - 1)
            failed = bin_idx[terminated]
            counts = b.float_(b.bincount(failed, N_BINS))
            self.bin_failed_count = (
                SAMPLER_ALPHA * counts + (1.0 - SAMPLER_ALPHA) * self.bin_failed_count
            )

        # MotionSampler.step + truncate_on_clip_end=True branch.
        self.current_frames = self.current_frames + 1
        clip_done = self.current_frames > CLIP_END_FRAME
        if b.any_scalar(clip_done):
            self.clip_end_truncated[b.nonzero(clip_done)] = True
        return obs, reward, terminated

    # ----------------------------------------------------------- reset_done --
    def _sample_pose_vel(self, ranges, n):
        """Per-element Python loop (faithful) or column-wise vectorized draw."""
        b = self.b
        if self.vectorized_reset_rng or b.is_torch:
            cols = [b.float_(self.rng.uniform(lo, hi, (n,))) for lo, hi in ranges]
            return b.stack(cols, axis=1)
        samples = np.array(
            [[self.rng.uniform(lo, hi, ()) for lo, hi in ranges] for _ in range(n)],
            dtype=np.float32,
        )
        return b.convert(samples)

    def reset_done(self, env_ids):
        b = self.b
        n = int(env_ids.shape[0])

        # NpEnv._reset_done_envs: steps reset + terminal-obs double copy.
        self.steps[env_ids] = 0
        for key in ("obs", "critic"):
            self.final_obs[key][env_ids] = self.obs[key][env_ids]
            self.compat_final_obs[key][env_ids] = self.final_obs[key][env_ids]

        # MotionSampler.sample_frames (adaptive).
        p = self.bin_failed_count + SAMPLER_UNIFORM_RATIO / N_BINS
        p = p / b.sum(p)
        bins = b.float_(self.rng.choice(N_BINS, n, p))
        off = b.float_(self.rng.uniform(0.0, 1.0, (n,)))
        frames = b.long((bins + off) / N_BINS * CLIP_END_FRAME)
        self.current_frames[env_ids] = frames

        motion = self._motion_at_frames(frames)
        m_joint_pos, m_joint_vel, m_body_pos, m_body_quat = motion[:4]

        # build_motion_reference_state (reset.py).
        root_pos = b.copy(m_body_pos[:, 0])
        root_ori = b.copy(m_body_quat[:, 0])
        root_lin_vel = b.copy(motion[4][:, 0])
        root_ang_vel = b.copy(motion[5][:, 0])
        joint_pos = b.copy(m_joint_pos)
        joint_vel = b.copy(m_joint_vel)

        pose = self._sample_pose_vel(POSE_RANGES, n)
        root_pos = root_pos + pose[:, 0:3]
        root_ori = _quat_mul(
            b, _quat_from_euler_xyz(b, pose[:, 3], pose[:, 4], pose[:, 5]), root_ori
        )
        vel = self._sample_pose_vel(VEL_RANGES, n)
        root_lin_vel = root_lin_vel + vel[:, 0:3]
        root_ang_vel = root_ang_vel + vel[:, 3:6]
        joint_pos = joint_pos + b.float_(
            self.rng.uniform(JOINT_POSITION_RANGE[0], JOINT_POSITION_RANGE[1], (n, N_ACTION))
        )
        joint_pos = b.clip(joint_pos, self.joint_lower, self.joint_upper)

        qpos = b.tile_batch(self.init_qpos, n)
        qpos[:, 0:3] = root_pos
        qpos[:, 3:7] = root_ori
        qpos[:, 7:] = joint_pos
        qvel = b.zeros((n, NV))
        qvel[:, 0:3] = root_lin_vel
        qvel[:, 3:6] = _quat_apply(b, _quat_inv(b, root_ori), root_ang_vel)
        qvel[:, 6:] = joint_vel

        new_actions = b.zeros((n, N_ACTION))

        # build_reset_observation: row gather + same obs math at batch n.
        obs_r = self._compute_obs(
            motion,
            self.linvel[env_ids],
            self.gyro[env_ids],
            self.dof_pos[env_ids],
            self.dof_vel[env_ids],
            self.body_pos[env_ids],
            self.body_quat[env_ids],
            new_actions,
        )

        # NpEnv._reset_done_envs: obs/info scatter.
        for key in ("obs", "critic"):
            self.obs[key][env_ids] = obs_r[key]
        self.current_actions[env_ids] = new_actions
        self.last_actions[env_ids] = new_actions

        info_updates = {"current_actions": new_actions, "last_actions": new_actions}
        return qpos, qvel, obs_r, info_updates
