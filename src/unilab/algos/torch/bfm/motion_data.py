from __future__ import annotations

import numpy as np

from .bfm_obs import quat_mul, quat_rotate_inverse
from .lafan_loader import _ang_vel_from_quat, _resample_to_ctrl_rate, _wxyz2xyzw, _xyzw2wxyz


class MotionBank:
    def __init__(
        self, pkl_path: str, num_dof: int, max_motions: int | None = None, ctrl_dt: float = 0.02
    ):
        import joblib

        motions = joblib.load(pkl_path)
        keys = list(motions.keys())[:max_motions] if max_motions else list(motions.keys())
        self.num_dof = num_dof
        ctrl_fps = 1.0 / float(ctrl_dt)
        self.qpos: list[np.ndarray] = []
        self.qvel: list[np.ndarray] = []
        self.names: list[str] = []
        self.fps: list[float] = []

        for k in keys:
            m = motions[k]
            fps = float(np.asarray(m["fps"]).item()) if m.get("fps") is not None else 30.0
            rp = np.asarray(m["root_trans_offset"], np.float32)
            rr_xyzw = np.asarray(m["root_rot"], np.float32)
            dof = np.asarray(m["dof"], np.float32)

            rp, rr_xyzw, dof = _resample_to_ctrl_rate(rp, rr_xyzw, dof, fps, ctrl_dt)
            qpos = np.concatenate([rp, _xyzw2wxyz(rr_xyzw), dof], axis=-1).astype(np.float32)
            root_lin = np.concatenate(
                [(rp[1:] - rp[:-1]) * ctrl_fps, np.zeros((1, 3), np.float32)], 0
            )
            root_ang_w = _ang_vel_from_quat(rr_xyzw, ctrl_fps)

            root_ang = quat_rotate_inverse(rr_xyzw, root_ang_w)
            dof_vel = np.concatenate(
                [(dof[1:] - dof[:-1]) * ctrl_fps, np.zeros((1, dof.shape[1]), np.float32)], 0
            )
            qvel = np.concatenate([root_lin, root_ang, dof_vel], axis=-1).astype(np.float32)
            self.qpos.append(qpos)
            self.qvel.append(qvel)
            self.names.append(str(k))
            self.fps.append(ctrl_fps)
        self.num_motions = len(self.qpos)
        self._weights: np.ndarray | None = None

    def set_sampling_weights(self, weights) -> None:

        w = np.asarray(weights, np.float32).reshape(-1)
        assert w.shape[0] == self.num_motions, (w.shape, self.num_motions)
        s = float(w.sum())
        self._weights = (w / s) if s > 0 else None

    def sample_reset(
        self, n: int, lie_down_prob: float = 0.0, rng: np.random.RandomState | None = None
    ):

        rng = rng or np.random.RandomState()
        nq = 7 + self.num_dof
        nv = 6 + self.num_dof
        qpos = np.zeros((n, nq), np.float32)
        qvel = np.zeros((n, nv), np.float32)
        for i in range(n):
            mi = (
                int(rng.choice(self.num_motions, p=self._weights))
                if self._weights is not None
                else rng.randint(self.num_motions)
            )
            T = self.qpos[mi].shape[0]
            f = rng.randint(T)
            qpos[i] = self.qpos[mi][f]
            qvel[i] = self.qvel[mi][f]
        if lie_down_prob > 0:
            mask = rng.uniform(size=n) < lie_down_prob
            if mask.any():
                qpos[mask, 2] = 0.5

                sign = 1.0 if rng.uniform() < 0.5 else -1.0
                half = 0.5 * sign * (-np.pi / 2)
                rot_xyzw = np.array([np.sin(half), 0.0, 0.0, np.cos(half)], np.float32)
                q_xyzw = _wxyz2xyzw(qpos[mask, 3:7])
                new_xyzw = quat_mul(np.broadcast_to(rot_xyzw, q_xyzw.shape), q_xyzw)
                qpos[mask, 3:7] = _xyzw2wxyz(new_xyzw)
        return qpos, qvel
