from __future__ import annotations

import glob
from pathlib import Path

import numpy as np

from .bfm_obs import (
    BfmObsBuilder,
    add_extend_bodies,
    compute_max_local_self,
    quat_rotate,
    quat_rotate_inverse,
)


def _wxyz2xyzw(q: np.ndarray) -> np.ndarray:
    return np.concatenate([q[..., 1:4], q[..., 0:1]], axis=-1)


def _foot_contact(foot_pos: np.ndarray, foot_vel: np.ndarray) -> np.ndarray:
    speed = np.linalg.norm(foot_vel, axis=-1)
    return ((speed < 0.4) & (foot_pos[..., 2] < 0.07)).astype(np.float32)


def _build_motion_obs(
    npz: dict,
    body_idx: np.ndarray,
    foot_local_idx: np.ndarray,
    default_dof: np.ndarray,
    num_dof: int,
    root_height_obs: bool = True,
    extends: list | None = None,
) -> list[dict[str, np.ndarray]]:

    bpos = npz["body_pos_w"][:, body_idx]
    bquat = _wxyz2xyzw(npz["body_quat_w"][:, body_idx])
    bvel = npz["body_lin_vel_w"][:, body_idx]
    bang = npz["body_ang_vel_w"][:, body_idx]
    if extends:
        bpos, bquat, bvel, bang = add_extend_bodies(bpos, bquat, bvel, bang, extends)
    dof_pos_raw = npz["joint_pos"]
    dof_vel = npz["joint_vel"]
    T = bpos.shape[0]

    root_quat = bquat[:, 0]
    g = np.tile(np.array([0, 0, -1.0], np.float32), (T, 1))
    proj_g = quat_rotate_inverse(root_quat, g)
    base_ang_vel = quat_rotate_inverse(root_quat, bang[:, 0])
    dof_pos = dof_pos_raw - default_dof

    builder = BfmObsBuilder(num_envs=1, num_dof=num_dof)
    out = []
    for t in range(T):
        contact = _foot_contact(bpos[t : t + 1, foot_local_idx], bvel[t : t + 1, foot_local_idx])
        max_self = compute_max_local_self(
            bpos[t : t + 1],
            bquat[t : t + 1],
            bvel[t : t + 1],
            bang[t : t + 1],
            contact,
            root_height_obs,
        )

        builder.push(
            base_ang_vel[t : t + 1],
            proj_g[t : t + 1],
            dof_pos[t : t + 1],
            dof_vel[t : t + 1],
            dof_pos[t : t + 1],
        )
        obs = builder.obs(
            dof_pos[t : t + 1],
            dof_vel[t : t + 1],
            proj_g[t : t + 1],
            base_ang_vel[t : t + 1],
            dof_pos[t : t + 1],
            max_self,
        )
        out.append({k: v[0] for k, v in obs.items()})
    return out


def build_expert_buffer(
    motion_glob: str,
    body_names: tuple[str, ...],
    foot_names: tuple[str, ...],
    body_name_to_idx: dict[str, int],
    default_dof: np.ndarray,
    num_dof: int,
    device: str,
    capacity: int,
    extends_cfg: tuple = (),
):

    import torch

    from .fb_core.buffers.transition import DictBuffer

    paths = (
        sorted(glob.glob(motion_glob)) if any(c in motion_glob for c in "*?[") else [motion_glob]
    )
    paths = [p for p in paths if Path(p).exists()]
    if not paths:
        raise FileNotFoundError(f"No motion NPZ found for: {motion_glob}")

    body_idx = np.array([body_name_to_idx[n] for n in body_names], dtype=np.int64)
    foot_local_idx = np.array([body_names.index(n) for n in foot_names], dtype=np.int64)
    extends = [(body_names.index(pn), np.asarray(off, np.float32)) for pn, off in extends_cfg]

    buf = DictBuffer(capacity=capacity, device=device)
    n_added = 0
    for p in paths:
        npz = {k: np.asarray(v) for k, v in np.load(p, allow_pickle=True).items()}
        frames = _build_motion_obs(
            npz, body_idx, foot_local_idx, default_dof, num_dof, extends=extends
        )
        for t in range(len(frames) - 1):
            o, no = frames[t], frames[t + 1]
            data = {
                "action": torch.as_tensor(
                    o["last_action"][None], dtype=torch.float32, device=device
                ),
                "observation": {
                    k: torch.as_tensor(v[None], dtype=torch.float32, device=device)
                    for k, v in o.items()
                },
                "next": {
                    "observation": {
                        k: torch.as_tensor(v[None], dtype=torch.float32, device=device)
                        for k, v in no.items()
                    }
                },
            }
            buf.extend(data)
            n_added += 1
            if n_added >= capacity:
                return buf
    return buf
