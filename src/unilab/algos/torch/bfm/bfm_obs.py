from __future__ import annotations

import numpy as np


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:

    qvec = q[..., :3]
    w = q[..., 3:4]
    t = 2.0 * np.cross(qvec, v)
    return v + w * t + np.cross(qvec, t)


def quat_rotate_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:

    q_conj = q.copy()
    q_conj[..., :3] *= -1.0
    return quat_rotate(q_conj, v)


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:

    ax, ay, az, aw = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bx, by, bz, bw = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    x = aw * bx + ax * bw + ay * bz - az * by
    y = aw * by - ax * bz + ay * bw + az * bx
    z = aw * bz + ax * by - ay * bx + az * bw
    w = aw * bw - ax * bx - ay * by - az * bz
    return np.stack([x, y, z, w], axis=-1)


def calc_heading_quat_inv(q: np.ndarray) -> np.ndarray:

    ref_dir = np.zeros(q.shape[:-1] + (3,), dtype=q.dtype)
    ref_dir[..., 0] = 1.0
    rot_dir = quat_rotate(q, ref_dir)
    heading = np.arctan2(rot_dir[..., 1], rot_dir[..., 0])

    half = -0.5 * heading
    out = np.zeros(q.shape[:-1] + (4,), dtype=q.dtype)
    out[..., 2] = np.sin(half)
    out[..., 3] = np.cos(half)
    return out


def quat_to_tan_norm(q: np.ndarray) -> np.ndarray:

    ref_tan = np.zeros(q.shape[:-1] + (3,), dtype=q.dtype)
    ref_tan[..., 0] = 1.0
    tan = quat_rotate(q, ref_tan)
    ref_norm = np.zeros(q.shape[:-1] + (3,), dtype=q.dtype)
    ref_norm[..., 2] = 1.0
    norm = quat_rotate(q, ref_norm)
    return np.concatenate([tan, norm], axis=-1)


def add_extend_bodies(
    body_pos: np.ndarray,
    body_quat: np.ndarray,
    body_vel: np.ndarray,
    body_ang_vel: np.ndarray,
    extends: list[tuple[int, np.ndarray]],
):

    if not extends:
        return body_pos, body_quat, body_vel, body_ang_vel
    epos, equat, evel, eang = [], [], [], []
    for parent_idx, offset_local in extends:
        pq = body_quat[:, parent_idx]
        off_w = quat_rotate(pq, np.broadcast_to(offset_local.astype(np.float32), pq[:, :3].shape))
        epos.append(body_pos[:, parent_idx] + off_w)
        equat.append(pq)
        evel.append(body_vel[:, parent_idx] + np.cross(body_ang_vel[:, parent_idx], off_w))
        eang.append(body_ang_vel[:, parent_idx])
    body_pos = np.concatenate([body_pos, np.stack(epos, axis=1)], axis=1)
    body_quat = np.concatenate([body_quat, np.stack(equat, axis=1)], axis=1)
    body_vel = np.concatenate([body_vel, np.stack(evel, axis=1)], axis=1)
    body_ang_vel = np.concatenate([body_ang_vel, np.stack(eang, axis=1)], axis=1)
    return body_pos, body_quat, body_vel, body_ang_vel


def compute_max_local_self(
    body_pos: np.ndarray,
    body_quat: np.ndarray,
    body_vel: np.ndarray,
    body_ang_vel: np.ndarray,
    contact_binary: np.ndarray,
    root_height_obs: bool = True,
) -> np.ndarray:

    B, N = body_pos.shape[0], body_pos.shape[1]
    root_pos = body_pos[:, 0, :]
    root_rot = body_quat[:, 0, :]
    root_h = root_pos[:, 2:3]

    heading_inv = calc_heading_quat_inv(root_rot)
    heading_inv_exp = np.repeat(heading_inv[:, None, :], N, axis=1)

    local_body_pos = body_pos - root_pos[:, None, :]
    local_body_pos = quat_rotate(heading_inv_exp, local_body_pos)
    local_body_pos = local_body_pos.reshape(B, N * 3)[:, 3:]

    local_body_rot = quat_mul(heading_inv_exp, body_quat)
    local_body_rot = quat_to_tan_norm(local_body_rot).reshape(B, N * 6)

    local_body_vel = quat_rotate(heading_inv_exp, body_vel).reshape(B, N * 3)
    local_body_ang_vel = quat_rotate(heading_inv_exp, body_ang_vel).reshape(B, N * 3)

    _ = (contact_binary, root_height_obs)
    parts = [root_h, local_body_pos, local_body_rot, local_body_vel, local_body_ang_vel]
    return np.concatenate(parts, axis=-1).astype(np.float32)


HISTORY_LEN = 4

HISTORY_KEYS = ("actions", "base_ang_vel", "dof_pos", "dof_vel", "projected_gravity")

OBS_SCALE_ANG_VEL = 0.25


class BfmObsBuilder:
    def __init__(self, num_envs: int, num_dof: int, history_len: int = HISTORY_LEN):
        self.num_envs = num_envs
        self.num_dof = num_dof
        self.history_len = history_len
        self._dims = {
            k: (3 if k in ("base_ang_vel", "projected_gravity") else num_dof) for k in HISTORY_KEYS
        }
        self._hist = {
            k: np.zeros((num_envs, history_len, self._dims[k]), np.float32) for k in HISTORY_KEYS
        }

    def reset_envs(self, env_ids: np.ndarray) -> None:
        for k in HISTORY_KEYS:
            self._hist[k][env_ids] = 0.0

    def push(self, base_ang_vel, projected_gravity, dof_pos, dof_vel, actions) -> None:

        cur = {
            "base_ang_vel": base_ang_vel * OBS_SCALE_ANG_VEL,
            "projected_gravity": projected_gravity,
            "dof_pos": dof_pos,
            "dof_vel": dof_vel,
            "actions": actions,
        }
        for k in HISTORY_KEYS:
            self._hist[k] = np.roll(self._hist[k], 1, axis=1)
            self._hist[k][:, 0, :] = cur[k].astype(np.float32)

    def _history_actor(self) -> np.ndarray:

        cols = [self._hist[k][:, ::-1].reshape(self.num_envs, -1) for k in HISTORY_KEYS]
        return np.concatenate(cols, axis=-1).astype(np.float32)

    def obs(
        self,
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
        projected_gravity: np.ndarray,
        base_ang_vel: np.ndarray,
        last_action: np.ndarray,
        max_local_self: np.ndarray,
    ) -> dict[str, np.ndarray]:

        state = np.concatenate(
            [dof_pos, dof_vel, projected_gravity, base_ang_vel * OBS_SCALE_ANG_VEL], axis=-1
        )
        return {
            "state": state.astype(np.float32),
            "last_action": last_action.astype(np.float32),
            "privileged_state": max_local_self.astype(np.float32),
            "history_actor": self._history_actor(),
        }
