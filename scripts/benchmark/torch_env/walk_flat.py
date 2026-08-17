"""G1WalkFlat (SAC/mujoco) update_state / reset_done workload.

Faithful xp-port of the NumPy computation in the collector-timed sections of
`uv run train --algo sac --task g1_walk_flat --sim mujoco` (num_envs=2048):

- `G1WalkEnv.update_state` (src/unilab/envs/locomotion/g1/joystick.py):
  termination, `_compute_reward` (9 active terms under the SAC scales incl.
  per-term logging every 4 steps), `_compute_obs` (noise + concat, walk
  profile), and the done-triggered curriculum bookkeeping.
- `G1WalkDomainRandomizationProvider.build_reset_plan` /
  `build_reset_observation` (qpos/qvel sampling, commands, gait phase, kp/kd
  payload, obs rebuild at batch n_reset).
- `NpEnv._reset_done_envs` scatter/gather (terminal-obs double copy, obs/info
  scatter).

Excluded (identical across variants, not NumPy/Torch env math):
`backend.step` physics, `backend.set_state`, sensor reads (replaced by
persistent state arrays), and the flat-terrain spawn add of zeros.
"""

from __future__ import annotations

import math

import numpy as np

N_ENVS = 2048
N_ACTION = 29
NQ = 36
NV = 35
OBS_DIM = 98
CRITIC_DIM = 101

MAX_TILT_RAD = math.radians(65.0)
MIN_BASE_HEIGHT = 0.3
TRACKING_SIGMA = 0.25
FEET_PHASE_SIGMA = 0.04
SWING_HEIGHT = 0.09
CTRL_DT = 0.02
COMMAND_LOW = (-0.6, -0.4, -0.8)
COMMAND_HIGH = (1.0, 0.4, 0.8)
ZERO_SMALL_XY_THRESHOLD = 0.2
RESET_BASE_QVEL_LIMIT = 0.5
EPISODE_WINDOW = 500  # EpisodeLengthTracker: 1000 * 2048 / 4096
CURRICULUM_DEGREE = 0.001
CURRICULUM_MIN_SCALE = 0.5
CURRICULUM_MAX_SCALE = 1.0
CURRICULUM_LEVEL_DOWN = 150.0
CURRICULUM_LEVEL_UP = 750.0
NOISE_LEVEL = 1.0
NOISE_SCALE_JOINT_ANGLE = 0.01
NOISE_SCALE_JOINT_VEL = 0.1

# conf/offpolicy/task/sac/g1_walk_flat/mujoco.yaml pose_weights (29-dof)
POSE_WEIGHTS = [0.01, 1.0, 5.0, 0.01, 5.0, 5.0] * 2 + [50.0] * 17

# Reward scales from the SAC owner YAML; penalty terms are multiplied by the
# curriculum penalty scale (initial_scale=0.5).
REWARD_SCALES = (
    ("tracking_lin_vel", 2.0, False),
    ("tracking_ang_vel", 1.5, False),
    ("penalty_ang_vel_xy", -1.0, True),
    ("penalty_orientation", -10.0, True),
    ("penalty_action_rate", -4.0, True),
    ("pose", -0.5, True),
    ("penalty_feet_ori", -20.0, True),
    ("feet_phase", 5.0, False),
    ("alive", 10.0, False),
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


class WalkFlatWorkload:
    def __init__(self, b, rng, seed: int = 0, num_envs: int = N_ENVS):
        self.b = b
        self.rng = rng
        self.n_envs = num_envs
        rs = np.random.RandomState(seed)

        def f32(*shape):
            return rs.standard_normal(shape).astype(np.float32)

        n = num_envs
        default_angles_np = rs.uniform(-0.5, 0.5, N_ACTION).astype(np.float32)
        self.default_angles = b.convert(default_angles_np)
        self.pose_weights = b.convert(np.array(POSE_WEIGHTS, dtype=np.float32))
        self.base_kp = b.convert(rs.uniform(50.0, 150.0, N_ACTION).astype(np.float32))
        self.base_kd = b.convert(rs.uniform(1.0, 5.0, N_ACTION).astype(np.float32))

        # Backend-owned state (sensor reads replaced by persistent arrays).
        self.linvel = b.convert(f32(n, 3) * 0.5)
        self.gyro = b.convert(f32(n, 3) * 0.5)
        gravity = f32(n, 3) * 0.05
        gravity[:, 2] += 1.0
        gravity /= np.linalg.norm(gravity, axis=1, keepdims=True)
        self.gravity = b.convert(gravity.astype(np.float32))
        self.dof_pos = b.convert(default_angles_np[None, :] + f32(n, N_ACTION) * 0.1)
        self.dof_vel = b.convert(f32(n, N_ACTION))
        base_z = 0.754 + f32(n) * 0.01
        base_z[:8] = 0.1  # force a few terminations -> done branch stays live
        self.base_z = b.convert(base_z.astype(np.float32))
        self.left_foot_pos = b.convert(np.abs(f32(n, 3)) * 0.05)
        self.right_foot_pos = b.convert(np.abs(f32(n, 3)) * 0.05)
        lq = f32(n, 4)
        lq /= np.linalg.norm(lq, axis=1, keepdims=True)
        rq = f32(n, 4)
        rq /= np.linalg.norm(rq, axis=1, keepdims=True)
        self.left_foot_quat = b.convert(lq.astype(np.float32))
        self.right_foot_quat = b.convert(rq.astype(np.float32))

        # info state.
        self.commands = b.convert(rs.uniform(COMMAND_LOW, COMMAND_HIGH, (n, 3)).astype(np.float32))
        self.current_actions = b.convert(f32(n, N_ACTION) * 0.3)
        self.last_actions = b.convert(f32(n, N_ACTION) * 0.3)
        self.gait_phase = b.convert(rs.uniform(0.0, 2 * np.pi, (n, 2)).astype(np.float32))
        self.steps = b.index(rs.randint(0, 1000, n))

        # Reset plan constants (keyframe "stand").
        init_qpos = np.zeros(NQ, dtype=np.float32)
        init_qpos[2] = 0.754
        init_qpos[3] = 1.0
        init_qpos[7:] = b.to_numpy(self.default_angles)
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

        # Curriculum bookkeeping state (scalar floats, as in the real code).
        self._ep_avg = 0.0
        self._penalty_scale = CURRICULUM_MIN_SCALE

    # ------------------------------------------------------------------ obs --
    def _obs_noise(self, data, scale):
        # G1BaseEnv._obs_noise with level=1.0, seed=None: full-size uniform draw
        # (float64 -> astype float32 on numpy) even when scale == 0.
        noise = self.b.float_(self.rng.uniform(-1.0, 1.0, tuple(data.shape)))
        return data + noise * NOISE_LEVEL * scale

    def _compute_obs(
        self, commands, last_actions, gait_phase, linvel, gyro, gravity, dof_pos, dof_vel
    ):
        b = self.b
        diff = dof_pos - self.default_angles
        noisy_gyro = self._obs_noise(gyro, 0.0)
        noisy_gravity = self._obs_noise(gravity, 0.0)
        noisy_diff = self._obs_noise(diff, NOISE_SCALE_JOINT_ANGLE)
        noisy_dof_vel = self._obs_noise(dof_vel, NOISE_SCALE_JOINT_VEL)
        actor = b.concat(
            [
                noisy_gyro * 0.25,
                -noisy_gravity,
                noisy_diff,
                noisy_dof_vel * 0.05,
                last_actions,
                commands,
                gait_phase,
            ],
            axis=1,
        )
        critic_base = b.concat(
            [
                gyro * 0.25,
                -gravity,
                diff,
                dof_vel * 0.05,
                last_actions,
                commands,
                gait_phase,
            ],
            axis=1,
        )
        critic = b.concat([critic_base, linvel * 2.0], axis=1)
        return {"obs": actor, "critic": critic}

    # --------------------------------------------------------------- reward --
    def _feet_phase_height_target(self, phi):
        b = self.b
        phi_normalized = b.fmod(phi + np.pi, 2 * np.pi) - np.pi
        x = (phi_normalized + np.pi) / (2 * np.pi)

        def bezier(y_start, y_end, t):
            return y_start + (y_end - y_start) * (t**3 + 3 * (t**2 * (1 - t)))

        stance = bezier(0.0, SWING_HEIGHT, 2 * x)
        swing = bezier(SWING_HEIGHT, 0.0, 2 * x - 1)
        return b.where(x <= 0.5, stance, swing)

    def _compute_reward(self, should_log: bool):
        b = self.b
        ps = self._penalty_scale
        cmd, linvel, gyro, g = self.commands, self.linvel, self.gyro, self.gravity
        q = self.dof_pos
        act, prev = self.current_actions, self.last_actions

        terms = {}
        err = b.sum(b.square(cmd[:, :2] - linvel[:, :2]), axis=1)
        terms["tracking_lin_vel"] = b.exp(-err / TRACKING_SIGMA)
        terms["tracking_ang_vel"] = b.exp(-b.square(cmd[:, 2] - gyro[:, 2]) / TRACKING_SIGMA)
        terms["penalty_ang_vel_xy"] = b.sum(b.square(gyro[:, :2]), axis=1)
        terms["penalty_orientation"] = b.square(g[:, 0]) + b.square(g[:, 1])
        terms["penalty_action_rate"] = b.sum(b.square(act - prev), axis=1)
        terms["pose"] = b.sum(self.pose_weights * b.square(q - self.default_angles), axis=1)
        lq, rq = self.left_foot_quat, self.right_foot_quat
        terms["penalty_feet_ori"] = (
            b.square(lq[:, 1]) + b.square(lq[:, 2]) + b.square(rq[:, 1]) + b.square(rq[:, 2])
        )
        left_target = self._feet_phase_height_target(self.gait_phase[:, 0])
        right_target = self._feet_phase_height_target(self.gait_phase[:, 1])
        feet_err = b.square(self.left_foot_pos[:, 2] - left_target) + b.square(
            self.right_foot_pos[:, 2] - right_target
        )
        gate = b.float_(b.maximum(linvel[:, 0], 0.0) >= 0.0)
        terms["feet_phase"] = b.exp(-feet_err / FEET_PHASE_SIGMA) * gate
        terms["alive"] = b.ones((self.n_envs,))

        reward = b.zeros((self.n_envs,))
        log = {}
        for name, scale, is_penalty in REWARD_SCALES:
            weighted = terms[name] * (scale * ps if is_penalty else scale)
            reward = reward + weighted
            if should_log:
                log[f"reward/{name}"] = b.scalar(b.mean(weighted))
        self._last_log = log
        return reward * CTRL_DT

    # --------------------------------------------------------- update_state --
    def update_state(self, should_log: bool):
        b = self.b
        tilt = b.arccos(b.clip(self.gravity[:, 2], -1.0, 1.0))
        terminated = b.logical_or(tilt > MAX_TILT_RAD, self.base_z < MIN_BASE_HEIGHT)
        reward = self._compute_reward(should_log)
        obs = self._compute_obs(
            self.commands,
            self.current_actions,
            self.gait_phase,
            self.linvel,
            self.gyro,
            self.gravity,
            self.dof_pos,
            self.dof_vel,
        )
        self.obs = obs

        # Done-triggered curriculum bookkeeping (truncated is NpEnv-side and
        # excluded; done == terminated here).
        if b.any_scalar(terminated):
            done_idx = b.nonzero(terminated)
            ep_len = b.float_(self.steps[done_idx]) + 1.0
            avg = b.scalar(b.mean(ep_len))
            w = min(float(done_idx.shape[0]) / EPISODE_WINDOW, 1.0)
            self._ep_avg = self._ep_avg * (1.0 - w) + avg * w
            if self._ep_avg > CURRICULUM_LEVEL_UP:
                self._penalty_scale *= 1.0 + CURRICULUM_DEGREE
            elif self._ep_avg < CURRICULUM_LEVEL_DOWN:
                self._penalty_scale *= 1.0 - CURRICULUM_DEGREE
            self._penalty_scale = min(
                max(self._penalty_scale, CURRICULUM_MIN_SCALE), CURRICULUM_MAX_SCALE
            )
        return obs, reward, terminated

    # ----------------------------------------------------------- reset_done --
    def reset_done(self, env_ids):
        b = self.b
        n = int(env_ids.shape[0])

        # NpEnv._reset_done_envs: steps reset + terminal-obs double copy.
        self.steps[env_ids] = 0
        for key in ("obs", "critic"):
            self.final_obs[key][env_ids] = self.obs[key][env_ids]
            self.compat_final_obs[key][env_ids] = self.final_obs[key][env_ids]

        # build_reset_plan: qpos/qvel sampling.
        qpos = b.tile_batch(self.init_qpos, n)
        qpos[:, 0:2] += b.float_(self.rng.uniform(-0.5, 0.5, (n, 2)))
        yaw = b.float_(self.rng.uniform(-np.pi, np.pi, (n,)))
        zeros_n = b.zeros((n,))
        yaw_quat = b.stack([b.cos(yaw * 0.5), zeros_n, zeros_n, b.sin(yaw * 0.5)], axis=1)
        qpos[:, 3:7] = _quat_mul(b, qpos[:, 3:7], yaw_quat)
        qvel = b.zeros((n, NV))
        qvel[:, 0:6] = b.float_(
            self.rng.uniform(-RESET_BASE_QVEL_LIMIT, RESET_BASE_QVEL_LIMIT, (n, 6))
        )

        # info_updates.
        commands = b.float_(self.rng.uniform(COMMAND_LOW, COMMAND_HIGH, (n, 3)))
        moving = b.sqrt(b.sum(b.square(commands[:, :2]), axis=1)) > ZERO_SMALL_XY_THRESHOLD
        commands[:, :2] = commands[:, :2] * b.float_(moving)[:, None]
        phi = b.float_(self.rng.uniform(0.0, 2 * np.pi, (n,)))
        gait_phase = b.stack([phi, phi + np.pi], axis=1)
        new_actions = b.zeros((n, N_ACTION))
        # DR payload consumed by backend.set_state (computed env-side).
        kp = self.base_kp * b.float_(self.rng.uniform(0.9, 1.1, (n, 1)))
        kd = self.base_kd * b.float_(self.rng.uniform(0.9, 1.1, (n, 1)))

        # build_reset_observation: row gather + same obs math at batch n.
        obs_r = self._compute_obs(
            commands,
            new_actions,
            gait_phase,
            self.linvel[env_ids],
            self.gyro[env_ids],
            self.gravity[env_ids],
            self.dof_pos[env_ids],
            self.dof_vel[env_ids],
        )

        # NpEnv._reset_done_envs: obs/info scatter.
        for key in ("obs", "critic"):
            self.obs[key][env_ids] = obs_r[key]
        self.commands[env_ids] = commands
        self.current_actions[env_ids] = new_actions
        self.last_actions[env_ids] = new_actions
        self.gait_phase[env_ids] = gait_phase

        info_updates = {
            "commands": commands,
            "current_actions": new_actions,
            "last_actions": new_actions,
            "gait_phase": gait_phase,
        }
        return qpos, qvel, obs_r, info_updates, (kp, kd)
