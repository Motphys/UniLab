from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import json
from pathlib import Path

import numpy as np


def _qpos_from_backend(backend, num_dof: int) -> np.ndarray:
    base_pos = np.asarray(backend.get_base_pos())[0]
    base_quat = np.asarray(backend.get_base_quat())[0]
    dof = np.asarray(backend.get_dof_pos())[0]
    return np.concatenate([base_pos, base_quat, dof]).astype(np.float64)


def _make_renderer(scene_xml: str, height: int, width: int):
    import mujoco

    rmodel = mujoco.MjModel.from_xml_path(scene_xml)
    rdata = mujoco.MjData(rmodel)
    renderer = mujoco.Renderer(rmodel, height, width)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = mujoco.mj_name2id(rmodel, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    cam.distance, cam.elevation, cam.azimuth = 3.5, -10.0, 90.0
    return rmodel, rdata, renderer, cam


def _load(model_folder: str, device: str):
    from .fb_core.load_utils import load_model_from_checkpoint_dir

    mf = Path(model_folder)
    model = load_model_from_checkpoint_dir(str(mf / "checkpoint"), device=device)
    model.to(device)
    model.eval()
    cfg = json.load(open(mf / "config.json"))
    return model, cfg, mf


def _pkl_keys(pkl_path: str) -> list[str]:

    import joblib

    return [str(k) for k in joblib.load(pkl_path).keys()]


def _build_env(cfg: dict, max_motions: int | None, lafan_pkl: str | None = None):
    from unilab.base.registry import make
    from unilab.training import ensure_registries

    ensure_registries()
    lafan = lafan_pkl or (cfg.get("expert") or {}).get("lafan_pkl")
    ov = {}
    if lafan:
        ov["lafan_pkl"] = lafan
        if max_motions is not None:
            ov["max_motions"] = max_motions
    env = make("G1Bfm", sim_backend="mujoco", num_envs=1, env_cfg_override=ov or None)
    env.init_state()
    return env, lafan


def track(
    model_folder: str,
    motion_idx: int = 25,
    n_steps: int | None = None,
    out: str | None = None,
    device: str = "cuda",
    height: int = 480,
    width: int = 640,
    z_pkl: str | None = None,
    lafan_pkl: str | None = None,
    full: bool = False,
    motion_key: str | None = None,
    max_seconds: float | None = 30.0,
    stride: int = 1,
):

    import mediapy as media
    import mujoco
    import torch

    from unilab.envs.locomotion.g1.bfm import BODY_NAMES, EXTEND_BODIES, FOOT_NAMES

    from .lafan_loader import single_motion_obs_sequence

    model, cfg, mf = _load(model_folder, device)

    cfg_pkl = (cfg.get("expert") or {}).get("lafan_pkl")
    if lafan_pkl:
        eval_pkl = lafan_pkl
    elif full:
        if not cfg_pkl:
            raise SystemExit(
                "--full needs the run config's expert.lafan_pkl to locate the sibling "
                "lafan_29dof.pkl; pass --lafan_pkl <path> instead."
            )
        eval_pkl = str(Path(cfg_pkl).with_name("lafan_29dof.pkl"))
    else:
        eval_pkl = cfg_pkl
    if eval_pkl and not Path(eval_pkl).exists():
        try:
            from unilab.assets.hub import resolve_motion_files

            eval_pkl = resolve_motion_files(eval_pkl)
        except Exception as exc:
            raise SystemExit(
                f"eval motion pkl not found locally and HF download failed: {exc}\n"
                f"  path tried: {eval_pkl!r}\n"
                "  pass --lafan_pkl <path> to specify the file directly."
            ) from None
    if not eval_pkl or not Path(eval_pkl).exists():
        raise SystemExit(
            f"eval motion pkl not found: {eval_pkl!r} (use --lafan_pkl <path> / --full)"
        )

    if motion_key is not None:
        keys = _pkl_keys(eval_pkl)
        if motion_key not in keys:
            raise SystemExit(
                f"motion_key {motion_key!r} not in {eval_pkl} "
                f"({len(keys)} motions; e.g. {', '.join(keys[:6])}, ...)"
            )
        target_idx = keys.index(motion_key)
    else:
        target_idx = motion_idx

    env, lafan = _build_env(cfg, max_motions=max(target_idx + 1, 8), lafan_pkl=eval_pkl)
    if env._motion_bank is None:
        raise SystemExit(f"MotionBank did not load from {eval_pkl!r}; check the path / pkl format.")
    if not (0 <= target_idx < env._motion_bank.num_motions):
        raise SystemExit(
            f"motion index {target_idx} out of range "
            f"({env._motion_bank.num_motions} motions loaded from {eval_pkl})"
        )
    nd = env._num_action
    resolved_key = env._motion_bank.names[target_idx]
    print(f"[bfm-infer] pkl={eval_pkl}")
    print(f"[bfm-infer] motion_idx={target_idx}  key={resolved_key!r}")

    if z_pkl is not None:
        import joblib

        z_ext = np.asarray(joblib.load(z_pkl), np.float32)
        z_seq = torch.as_tensor(z_ext, dtype=torch.float32, device=device)
        print(
            f"[bfm-infer] using EXTERNAL z from {z_pkl}  shape={tuple(z_seq.shape)} "
            f"(||z||/frame~{float(torch.linalg.norm(z_seq, dim=-1).mean()):.2f})"
        )
    else:
        seq = single_motion_obs_sequence(
            lafan,
            target_idx,
            tuple(env._cfg.body_names),
            tuple(env._cfg.foot_names),
            EXTEND_BODIES,
            env._cfg.scene.model_file,
            env._default_dof,
            nd,
            ctrl_dt=float(env._cfg.ctrl_dt),
        )
        with torch.no_grad():
            obs_t = {
                k: torch.as_tensor(v, dtype=torch.float32, device=device) for k, v in seq.items()
            }

            z_seq = model.tracking_inference(obs_t)

    qpos0 = env._motion_bank.qpos[target_idx][0]
    qvel0 = env._motion_bank.qvel[target_idx][0]
    env._backend.set_state(np.array([0]), qpos0[None], qvel0[None])
    env._obs_builder.reset_envs(np.array([0]))
    env._last_action[:] = 0.0

    rmodel, rdata, renderer, cam = _make_renderer(env._cfg.scene.model_file, height, width)
    nq = rmodel.nq
    scale = env._cfg.control_config.action_scale
    substeps = env._cfg.sim_substeps
    ctrl_dt = float(env._cfg.ctrl_dt)

    Tref = z_seq.shape[0]
    expert_qpos = env._motion_bank.qpos[target_idx]
    Texp = expert_qpos.shape[0]
    motion_fps = float(env._motion_bank.fps[target_idx])
    n = min(Tref, Texp)
    if max_seconds is not None and max_seconds > 0:
        n = min(n, int(round(max_seconds / ctrl_dt)))
    if n_steps is not None:
        if n_steps > Tref:
            print(
                f"[bfm-infer] --n_steps {n_steps} > motion length {Tref}; clamped to motion end (no loop)"
            )
        n = min(n, n_steps)
    n = max(1, n)
    stride = max(1, int(stride))
    print(
        f"[bfm-infer] rendering {n} steps = {n * ctrl_dt:.1f}s  "
        f"(motion {Tref} frames = {Tref * ctrl_dt:.1f}s, stride={stride})"
    )

    policy_qpos = []
    with torch.no_grad():
        for t in range(n):
            obs = env._compute_obs(push=True)
            o = {k: torch.as_tensor(obs[k], dtype=torch.float32, device=device) for k in obs}
            zt = z_seq[min(t, Tref - 1)].unsqueeze(0)
            a = model.act(o, zt, mean=True).cpu().numpy().astype(np.float32)
            a_norm = np.clip(a * env.NORMALIZE_ACTION_TO, -env.ACTION_CLIP, env.ACTION_CLIP)
            env._last_action = a_norm
            env._backend.step(a_norm * scale * env._act_rescale + env._default_dof, substeps)
            policy_qpos.append(_qpos_from_backend(env._backend, nd))

    def _render_qpos(q):
        rdata.qpos[:] = q[:nq] if q.shape[0] >= nq else np.pad(q, (0, nq - q.shape[0]))
        mujoco.mj_forward(rmodel, rdata)
        renderer.update_scene(rdata, camera=cam)
        return renderer.render()

    out = out or str(mf / "inference" / f"tracking_motion{target_idx}.mp4")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    render_fps = max(1, round(1.0 / ctrl_dt / stride))
    ts = list(range(0, n, stride))
    with media.VideoWriter(out, shape=(height, width * 2), fps=render_fps) as w:
        for t in ts:
            ei = min(int(round(t * ctrl_dt * motion_fps)), Texp - 1)
            left = _render_qpos(expert_qpos[ei])
            right = _render_qpos(policy_qpos[t])
            w.add_image(np.concatenate([left, right], axis=1))
    print(f"[bfm-infer] saved {out}  ({len(ts)} frames, LEFT=expert | RIGHT=policy)")
    return out


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="BFM inference + render")
    p.add_argument("mode", choices=["track"], help="inference mode (goal/reward coming next)")
    p.add_argument("--model_folder", required=True)
    p.add_argument("--motion_idx", type=int, default=25)
    p.add_argument(
        "--motion_key",
        default=None,
        help="select motion by key name (stable across pkls; overrides --motion_idx)",
    )
    p.add_argument(
        "--n_steps",
        type=int,
        default=None,
        help="render length in steps; default = full motion (capped by --max_seconds), never loops",
    )
    p.add_argument(
        "--max_seconds",
        type=float,
        default=30.0,
        help="cap render length in seconds (default 30; <=0 = full motion)",
    )
    p.add_argument(
        "--stride",
        type=int,
        default=1,
        help="render every Nth control step (subsample very long motions; playback stays wall-clock)",
    )
    p.add_argument(
        "--lafan_pkl",
        default=None,
        help="override eval motion pkl (e.g. the full-length lafan_29dof.pkl)",
    )
    p.add_argument(
        "--full",
        action="store_true",
        help="use the sibling un-clipped lafan_29dof.pkl instead of the run's (10s-clipped) config pkl",
    )
    p.add_argument("--out", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=640)
    p.add_argument(
        "--z_pkl",
        default=None,
        help="optional external per-frame z (e.g. BFM-Zero tracking_inference/zs_<idx>.pkl) "
        "to isolate a z-computation bug from an actor-obs bug",
    )
    a = p.parse_args()
    if a.mode == "track":
        track(
            a.model_folder,
            motion_idx=a.motion_idx,
            n_steps=a.n_steps,
            out=a.out,
            device=a.device,
            height=a.height,
            width=a.width,
            z_pkl=a.z_pkl,
            lafan_pkl=a.lafan_pkl,
            full=a.full,
            motion_key=a.motion_key,
            max_seconds=a.max_seconds,
            stride=a.stride,
        )
