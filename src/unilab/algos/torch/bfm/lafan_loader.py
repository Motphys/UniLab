from __future__ import annotations

import os

import numpy as np

from .bfm_obs import BfmObsBuilder, add_extend_bodies, compute_max_local_self, quat_rotate_inverse


def _xyzw2wxyz(q: np.ndarray) -> np.ndarray:
    return np.concatenate([q[..., 3:4], q[..., 0:3]], axis=-1)


def _wxyz2xyzw(q: np.ndarray) -> np.ndarray:
    return np.concatenate([q[..., 1:4], q[..., 0:1]], axis=-1)


def _ang_vel_from_quat(q_xyzw: np.ndarray, fps: float) -> np.ndarray:

    from .bfm_obs import quat_mul

    qa, qb = q_xyzw[:-1], q_xyzw[1:]
    qa_inv = qa.copy()
    qa_inv[..., :3] *= -1.0
    dq = quat_mul(qb, qa_inv)

    dq = np.where(dq[..., 3:4] < 0, -dq, dq)
    ang = 2.0 * dq[..., :3] * fps
    return np.concatenate([ang, ang[-1:]], axis=0)


def _slerp_np(q0: np.ndarray, q1: np.ndarray, t: np.ndarray) -> np.ndarray:

    t = np.asarray(t, np.float32).reshape(-1, 1)
    q0 = np.asarray(q0, np.float32)
    q1 = np.asarray(q1, np.float32).copy()
    cos_half = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(cos_half < 0, -q1, q1)
    cos_half = np.clip(np.abs(cos_half), -1.0, 1.0)
    half = np.arccos(cos_half)
    sin_half = np.sqrt(1.0 - cos_half * cos_half)
    safe = np.maximum(sin_half, 1e-6)
    ratio_a = np.sin((1.0 - t) * half) / safe
    ratio_b = np.sin(t * half) / safe
    out = ratio_a * q0 + ratio_b * q1
    lin = 0.5 * q0 + 0.5 * q1
    out = np.where(np.abs(sin_half) < 1e-3, lin, out)
    out = np.where(cos_half >= 1.0, q0, out)
    return out.astype(np.float32)


def _resample_to_ctrl_rate(
    rt: np.ndarray, rr_xyzw: np.ndarray, dof: np.ndarray, fps: float, ctrl_dt: float
):

    T = dof.shape[0]
    motion_dt = 1.0 / float(fps)
    motion_len = (T - 1) * motion_dt
    n = max(1, int(np.ceil(motion_len / ctrl_dt)))
    times = np.arange(n, dtype=np.float32) * ctrl_dt
    x = np.clip(times * float(fps), 0.0, T - 1)
    i0 = np.floor(x).astype(np.int64)
    i1 = np.minimum(i0 + 1, T - 1)
    blend = np.clip(x - i0, 0.0, 1.0)[:, None].astype(np.float32)
    rt_r = (1.0 - blend) * rt[i0] + blend * rt[i1]
    dof_r = (1.0 - blend) * dof[i0] + blend * dof[i1]
    rr_r = _slerp_np(rr_xyzw[i0], rr_xyzw[i1], blend[:, 0])
    return rt_r.astype(np.float32), rr_r.astype(np.float32), dof_r.astype(np.float32)


def _blend_to_ctrl(T: int, fps: float, ctrl_dt: float):

    motion_len = (T - 1) / float(fps)
    n = max(1, int(np.ceil(motion_len / ctrl_dt)))
    x = np.clip(np.arange(n, dtype=np.float32) * ctrl_dt * float(fps), 0.0, T - 1)
    i0 = np.floor(x).astype(np.int64)
    i1 = np.minimum(i0 + 1, T - 1)
    blend = np.clip(x - i0, 0.0, 1.0).astype(np.float32)
    return i0, i1, blend


def _lerp(a: np.ndarray, b: np.ndarray, blend: np.ndarray) -> np.ndarray:

    bl = blend.reshape((-1,) + (1,) * (a.ndim - 1)).astype(np.float32)
    return ((1.0 - bl) * a + bl * b).astype(np.float32)


def _slerp_bodies(q0: np.ndarray, q1: np.ndarray, blend: np.ndarray) -> np.ndarray:

    n, nb, _ = q0.shape
    out = _slerp_np(q0.reshape(-1, 4), q1.reshape(-1, 4), np.repeat(blend, nb))
    return out.reshape(n, nb, 4).astype(np.float32)


def _fk_motion(model, data, mj_body_ids, root_trans, root_rot_xyzw, dof, fps):

    import mujoco

    T = dof.shape[0]
    nb = len(mj_body_ids)
    pos = np.zeros((T, nb, 3), np.float32)
    quat_wxyz = np.zeros((T, nb, 4), np.float32)
    root_wxyz = _xyzw2wxyz(root_rot_xyzw)
    for t in range(T):
        data.qpos[:] = 0.0
        data.qpos[0:3] = root_trans[t]
        data.qpos[3:7] = root_wxyz[t]
        data.qpos[7 : 7 + dof.shape[1]] = dof[t]
        mujoco.mj_forward(model, data)
        pos[t] = data.xpos[mj_body_ids]
        quat_wxyz[t] = data.xquat[mj_body_ids]
    quat = _wxyz2xyzw(quat_wxyz)

    from scipy.ndimage import gaussian_filter1d

    lin_vel = np.gradient(pos, axis=0) * fps
    lin_vel = gaussian_filter1d(lin_vel, 2.0, axis=0, mode="nearest").astype(np.float32)
    ang_vel = np.stack([_ang_vel_from_quat(quat[:, i], fps) for i in range(nb)], axis=1)
    ang_vel = gaussian_filter1d(ang_vel, 2.0, axis=0, mode="nearest").astype(np.float32)
    return pos, quat, lin_vel, ang_vel


def _motion_frames(
    model,
    data,
    mj_body_ids,
    foot_local,
    extends,
    rt,
    rr_xyzw,
    dof,
    fps,
    default_dof,
    num_dof,
    ctrl_dt,
    root_height_obs=True,
) -> list[dict[str, np.ndarray]]:

    pos_n, quat_n, lvel_n, avel_n = _fk_motion(model, data, mj_body_ids, rt, rr_xyzw, dof, fps)
    pos_n, quat_n, lvel_n, avel_n = add_extend_bodies(pos_n, quat_n, lvel_n, avel_n, extends)
    Tn = dof.shape[0]
    if Tn > 1:
        dvel_n = np.concatenate([(dof[1:] - dof[:-1]) * fps, (dof[-1:] - dof[-2:-1]) * fps], axis=0)
    else:
        dvel_n = np.zeros_like(dof)

    i0, i1, blend = _blend_to_ctrl(Tn, fps, ctrl_dt)
    pos = _lerp(pos_n[i0], pos_n[i1], blend)
    quat = _slerp_bodies(quat_n[i0], quat_n[i1], blend)
    lvel = _lerp(lvel_n[i0], lvel_n[i1], blend)
    avel = _lerp(avel_n[i0], avel_n[i1], blend)
    dof_r = _lerp(dof[i0], dof[i1], blend)
    dof_vel = _lerp(dvel_n[i0], dvel_n[i1], blend)
    T = dof_r.shape[0]
    root_quat = quat[:, 0]
    g = np.tile(np.array([0, 0, -1.0], np.float32), (T, 1))
    proj_g = quat_rotate_inverse(root_quat, g)
    base_ang_vel = quat_rotate_inverse(root_quat, avel[:, 0])
    dof_pos = dof_r - default_dof
    builder = BfmObsBuilder(num_envs=1, num_dof=num_dof)
    zero_action = np.zeros((1, num_dof), np.float32)
    frames = []
    for t in range(T):
        speed = np.linalg.norm(lvel[t : t + 1, foot_local], axis=-1)
        contact = ((speed < 0.4) & (pos[t : t + 1, foot_local, 2] < 0.07)).astype(np.float32)
        max_self = compute_max_local_self(
            pos[t : t + 1],
            quat[t : t + 1],
            lvel[t : t + 1],
            avel[t : t + 1],
            contact,
            root_height_obs,
        )
        builder.push(
            base_ang_vel[t : t + 1],
            proj_g[t : t + 1],
            dof_pos[t : t + 1],
            dof_vel[t : t + 1],
            zero_action,
        )
        o = builder.obs(
            dof_pos[t : t + 1],
            dof_vel[t : t + 1],
            proj_g[t : t + 1],
            base_ang_vel[t : t + 1],
            zero_action,
            max_self,
        )
        if os.environ.get("UNILAB_BFM_B_INPUT", "default") in ("novel", "novel_only"):
            s = o["state"]
            o["state_novel"] = np.concatenate(
                [s[:, :num_dof], s[:, 2 * num_dof : 2 * num_dof + 6]], axis=-1
            ).astype(np.float32)
        frames.append({k: v[0] for k, v in o.items()})
    return frames


def _motion_fields(m):

    fps = float(np.asarray(m["fps"]).item()) if m.get("fps") is not None else 30.0
    rt = np.asarray(m["root_trans_offset"], np.float32)
    rr = np.asarray(m["root_rot"], np.float32)
    dof = np.asarray(m["dof"], np.float32)
    return rt, rr, dof, fps


def _make_model_and_ids(mj_xml, body_names, foot_names, extends_cfg):
    import mujoco

    model = mujoco.MjModel.from_xml_path(mj_xml)
    data = mujoco.MjData(model)
    mj_body_ids = np.array(
        [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in body_names], dtype=np.int64
    )
    foot_local = np.array([body_names.index(n) for n in foot_names], dtype=np.int64)
    extends = [(body_names.index(pn), np.asarray(off, np.float32)) for pn, off in extends_cfg]
    return model, data, mj_body_ids, foot_local, extends


def single_motion_obs_sequence(
    pkl_path: str,
    motion_idx: int,
    body_names: tuple[str, ...],
    foot_names: tuple[str, ...],
    extends_cfg: tuple,
    mj_xml: str,
    default_dof: np.ndarray,
    num_dof: int,
    root_height_obs: bool = True,
    ctrl_dt: float = 0.02,
):

    import joblib

    motions = joblib.load(pkl_path)
    key = list(motions.keys())[motion_idx]
    rt, rr, dof, fps = _motion_fields(motions[key])
    model, data, mj_body_ids, foot_local, extends = _make_model_and_ids(
        mj_xml, body_names, foot_names, extends_cfg
    )
    frames = _motion_frames(
        model,
        data,
        mj_body_ids,
        foot_local,
        extends,
        rt,
        rr,
        dof,
        fps,
        default_dof,
        num_dof,
        ctrl_dt,
        root_height_obs,
    )
    seqs: dict[str, list] = {}
    for fr in frames:
        for k, v in fr.items():
            seqs.setdefault(k, []).append(v)
    return {k: np.stack(v).astype(np.float32) for k, v in seqs.items()}


def build_expert_trajectory_buffer(
    pkl_path: str,
    body_names: tuple[str, ...],
    foot_names: tuple[str, ...],
    extends_cfg: tuple,
    mj_xml: str,
    default_dof: np.ndarray,
    num_dof: int,
    device: str,
    seq_length: int,
    max_motions: int | None = None,
    root_height_obs: bool = True,
    ctrl_dt: float = 0.02,
):

    import joblib
    import torch

    from .fb_core.buffers.trajectory import TrajectoryDictBuffer

    motions = joblib.load(pkl_path)
    keys = list(motions.keys())[:max_motions] if max_motions else list(motions.keys())
    model, data, mj_body_ids, foot_local, extends = _make_model_and_ids(
        mj_xml, body_names, foot_names, extends_cfg
    )

    episodes = []
    for mid, key in enumerate(keys):
        rt, rr, dof, fps = _motion_fields(motions[key])
        frames = _motion_frames(
            model,
            data,
            mj_body_ids,
            foot_local,
            extends,
            rt,
            rr,
            dof,
            fps,
            default_dof,
            num_dof,
            ctrl_dt,
            root_height_obs,
        )
        T = len(frames)
        obs_seq = {
            k: torch.as_tensor(
                np.stack([fr[k] for fr in frames]), dtype=torch.float32, device=device
            )
            for k in frames[0]
        }
        ep = {
            "observation": obs_seq,
            "motion_id": torch.full((T, 1), mid, dtype=torch.long, device=device),
        }
        episodes.append(ep)

    return TrajectoryDictBuffer(
        episodes,
        device=device,
        seq_length=seq_length,
        output_key_t=["observation"],
        output_key_tp1=["observation"],
        end_key="done",
        motion_id_key="motion_id",
    )


def build_expert_buffer_from_lafan(
    pkl_path: str,
    body_names: tuple[str, ...],
    foot_names: tuple[str, ...],
    extends_cfg: tuple,
    mj_xml: str,
    default_dof: np.ndarray,
    num_dof: int,
    device: str,
    capacity: int,
    max_motions: int | None = None,
    root_height_obs: bool = True,
    ctrl_dt: float = 0.02,
):
    import joblib
    import torch

    from .fb_core.buffers.transition import DictBuffer

    motions = joblib.load(pkl_path)
    keys = list(motions.keys())[:max_motions] if max_motions else list(motions.keys())
    model, data, mj_body_ids, foot_local, extends = _make_model_and_ids(
        mj_xml, body_names, foot_names, extends_cfg
    )

    buf = DictBuffer(capacity=capacity, device=device)
    n_added = 0
    for key in keys:
        rt, rr, dof, fps = _motion_fields(motions[key])
        frames = _motion_frames(
            model,
            data,
            mj_body_ids,
            foot_local,
            extends,
            rt,
            rr,
            dof,
            fps,
            default_dof,
            num_dof,
            ctrl_dt,
            root_height_obs,
        )
        for t in range(len(frames) - 1):
            o, no = frames[t], frames[t + 1]
            data_t = {
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
            buf.extend(data_t)
            n_added += 1
            if n_added >= capacity:
                return buf
    return buf
