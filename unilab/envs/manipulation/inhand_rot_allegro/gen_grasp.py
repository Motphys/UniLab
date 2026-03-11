"""Generate stable grasp states for AllegroInhandRotation.

Algorithm:
  1. Create AllegroRotationMj env with many parallel envs.
  2. Run with zero actions so the PD controller holds each env at its
     canonical pre-grasp reset pose.
  3. An episode that survives the full length (truncated, not terminated)
     is a stable grasp.  Capture its physics state *before* auto-reset.
  4. Optionally filter by fingertip-ball distance.
  5. Save collected states as (N, 23) float32 numpy array:
         [hand_qpos(16), ball_pos(3), ball_quat(4)]

Usage:
    # From UniLab root:
    python unilab/envs/manipulation/inhand_rot_allegro/gen_grasp.py

    # Debug with live viewer (16 envs, watch env[0]):
    python unilab/envs/manipulation/inhand_rot_allegro/gen_grasp.py \\
        --num_envs 16 --viewer

    python unilab/envs/manipulation/inhand_rot_allegro/pre_grasp.py \\
        --num_envs 2048 --target 50000 --quality_check
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import pkgutil
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import mediapy as media

# ── Path setup ──────────────────────────────────────────────────────────────
# gen_grasp.py lives at: UniLab/unilab/envs/manipulation/inhand_rot_allegro/
ROOT_DIR = Path(__file__).parents[4]
sys.path.insert(0, str(ROOT_DIR))


# ── Registry discovery ───────────────────────────────────────────────────────

def ensure_registries():
    for pkg_name in ("unilab.envs.locomotion", "unilab.envs.manipulation"):
        try:
            package = importlib.import_module(pkg_name)
            if hasattr(package, "__path__"):
                for _, name, _ in pkgutil.walk_packages(
                    package.__path__, package.__name__ + "."
                ):
                    try:
                        importlib.import_module(name)
                    except Exception:
                        pass
        except ImportError:
            pass


ensure_registries()

from unilab.envs import registry  # noqa: E402  (after sys.path setup)
from unilab.utils import render_many  # noqa: E402


# ── Quality-check helpers ────────────────────────────────────────────────────

def _get_body_ids(model, names: list[str]) -> list[int]:
    ids = []
    for name in names:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            raise ValueError(f"Body '{name}' not found in model")
        ids.append(bid)
    return ids


def check_grasp_quality(
    physics_states: np.ndarray,
    model,
    fingertip_ids: list[int],
    ball_body_id: int,
    max_tip_dist: float = 0.1,
    min_close_tips: int = 2,
    close_threshold: float = 0.05,
) -> np.ndarray:
    """Return boolean mask indicating quality grasps.

    Conditions (mirror HORA allegro_hand_grasp.py):
      1. All 4 fingertips within *max_tip_dist* of the ball centre.
      2. At least *min_close_tips* fingertips within *close_threshold* of ball.

    Args:
        physics_states: (N, nstate) float physics states.
    Returns:
        Boolean mask of shape (N,).
    """
    n = len(physics_states)
    state_spec = mujoco.mjtState.mjSTATE_FULLPHYSICS
    mask = np.zeros(n, dtype=bool)

    tmp = mujoco.MjData(model)
    for i, ps in enumerate(physics_states):
        mujoco.mj_setState(model, tmp, ps.astype(np.float64), state_spec)
        mujoco.mj_forward(model, tmp)

        ball_pos = tmp.xpos[ball_body_id]                          # (3,)
        tip_pos  = np.stack([tmp.xpos[tid] for tid in fingertip_ids])  # (4, 3)
        dists    = np.linalg.norm(tip_pos - ball_pos, axis=1)     # (4,)

        cond1 = bool(np.all(dists < max_tip_dist))
        cond2 = int(np.sum(dists < close_threshold)) >= min_close_tips
        mask[i] = cond1 and cond2

    return mask


# ── Main collection loop ─────────────────────────────────────────────────────

def collect_grasps(args) -> None:
    env = registry.make(
        "AllegroInhandRotation", num_envs=args.num_envs, sim_backend="mujoco"
    )

    # Override joint noise to the exploration value before init_state().
    # reset() reads this at runtime, so all episode resets will use it.
    env.cfg.domain_rand.joint_noise = args.joint_noise
    env.cfg.domain_rand.ball_z_offset = 0.01
    # Enable gen_grasp mode to disable cache loading.
    env.cfg.gen_grasp = True
    # Override episode length for pre-grasp collection
    env.cfg.max_episode_seconds = 3.0

    env.init_state()
    env.reset(np.arange(args.num_envs))

    # Zero actions → PD holds at whatever prev_ctrl was set to during reset
    # (= canonical pre-grasp keyframe pose).
    zero_actions = np.zeros(
        (args.num_envs, env.action_space.shape[0]), dtype=env._np_dtype
    )

    # Body IDs for quality filtering.
    fingertip_ids = _get_body_ids(env._model, ["ff_tip", "mf_tip", "rf_tip", "th_tip"])
    ball_body_id  = _get_body_ids(env._model, ["ball"])[0]

    max_ep_steps = env.cfg.max_episode_steps
    print(
        f"[gen_grasp] Collecting {args.target:,} stable grasps "
        f"({args.num_envs} parallel envs, "
        f"{max_ep_steps} steps/episode = {env.cfg.max_episode_seconds}s)"
    )
    if args.quality_check:
        print("[gen_grasp] Quality check ON (fingertip-ball distance)")
    if args.viewer:
        print("[gen_grasp] Viewer ON — displaying env[0]; close window to stop early")

    cache: list[np.ndarray] = []
    video_states: list[np.ndarray] = [] if args.record_video else None
    step_idx = 0
    state_spec = mujoco.mjtState.mjSTATE_FULLPHYSICS

    # Separate MjData for the viewer so rollout workers are untouched.
    viz_data = mujoco.MjData(env._model) if args.viewer else None
    viewer_ctx = (
        mujoco.viewer.launch_passive(env._model, viz_data)
        if args.viewer
        else contextlib.nullcontext()
    )

    with viewer_ctx as viewer:

        while (
            sum(len(s) for s in cache) < args.target
            and (viewer is None or viewer.is_running())
        ):
            t0 = time.perf_counter()

            # ── Manual step (mirrors env.step() but defers auto-reset) ────────
            env._pre_step()
            env._state.ctrl[:] = env.apply_action(zero_actions, env._state)
            env._step_core()
            env._state = env.update_state(env._state, obs_required=False)
            env._state.info["steps"] += 1
            env._update_truncate()

            # ── Capture states BEFORE reset ────────────────────────────────────
            truncated  = env._state.truncated   # (N,) bool
            terminated = env._state.terminated  # (N,) bool

            # Success = full-episode survival without ball drop.
            success_mask = truncated & ~terminated
            if success_mask.any():
                success_idx = np.where(success_mask)[0]
                ps = env._state.physics_state[success_idx]   # (k, nstate)

                if args.quality_check:
                    quality = check_grasp_quality(
                        ps, env._model, fingertip_ids, ball_body_id
                    )
                    ps = ps[quality]

                if len(ps) > 0:
                    iq = env._idx_qpos
                    hand_qpos = ps[:, iq               : iq + 16           ]   # (k, 16)
                    ball_pos  = ps[:, env._ps_ball_pos : env._ps_ball_pos+3]   # (k,  3)
                    ball_quat = ps[:, env._ps_ball_quat: env._ps_ball_quat+4]  # (k,  4)
                    states = np.concatenate(
                        [hand_qpos, ball_pos, ball_quat], axis=1
                    ).astype(np.float32)   # (k, 23)
                    cache.append(states)

                    total = sum(len(s) for s in cache)
                    print(
                        f"  step {step_idx:>8d}  "
                        f"new: {len(states):>4d}  "
                        f"total: {total:>6d} / {args.target:,}"
                    )

            # ── Now let the framework reset done envs ──────────────────────────
            env._reset_done_envs()

            # ── Record video states ────────────────────────────────────────────
            if video_states is not None and step_idx < args.video_steps:
                video_states.append(env._state.physics_state.copy())

            step_idx += 1

            # ── Viewer refresh (env[0] only) ───────────────────────────────────
            if viewer is not None:
                mujoco.mj_setState(
                    env._model, viz_data,
                    env._state.physics_state[0].astype(np.float64),
                    state_spec,
                )
                mujoco.mj_forward(env._model, viz_data)
                viewer.sync()
                # Pace to real-time.
                elapsed = time.perf_counter() - t0
                if env.cfg.ctrl_dt - elapsed > 0:
                    time.sleep(env.cfg.ctrl_dt - elapsed)

    # ── Save ──────────────────────────────────────────────────────────────
    if not cache:
        print("[gen_grasp] No grasps collected.")
        return
    cache_arr = np.concatenate(cache, axis=0)[: args.target]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(output), cache_arr)
    print(f"\n[gen_grasp] Saved {len(cache_arr):,} grasps → {output}")

    # ── Render video ──────────────────────────────────────────────────────
    if video_states:
        video_path = output.parent / "pre_grasp_video.mp4"
        print(f"Rendering video to {video_path}...")
        downsampled = video_states[::2]  # 50Hz -> 25Hz
        frames = render_many.render_states_get_frames(
            downsampled, env.cfg.model_file, width=1280, height=720, camera_id=-1
        )
        media.write_video(str(video_path), frames, fps=25)
        print(f"Video saved to {video_path}")


# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    default_output = str(
        Path(__file__).parent / "grasps" / "grasp_50k.npy"
    )
    parser = argparse.ArgumentParser(
        description="Generate stable AllegroInhandRotation grasp states"
    )
    parser.add_argument(
        "--num_envs", type=int, default=2048,
        help="Number of parallel MuJoCo envs (default: 2048)",
    )
    parser.add_argument(
        "--target", type=int, default=50_000,
        help="Target number of grasps to collect (default: 50 000)",
    )
    parser.add_argument(
        "--output", type=str, default=default_output,
        help=f"Output .npy path (default: {default_output})",
    )
    parser.add_argument(
        "--joint_noise", type=float, default=0.25,
        help="Joint noise range ±rad used at each reset for diverse exploration (default: 0.25, matches HORA)",
    )
    parser.add_argument(
        "--viewer", action="store_true",
        help="Open a live MuJoCo viewer showing env[0] (useful for debugging with --num_envs 16)",
    )
    parser.add_argument(
        "--quality_check", action="store_true",
        help="Filter grasps by fingertip-ball distance",
    )
    parser.add_argument(
        "--record_video", action="store_true",
        help="Record video of the 16 parallel envs",
    )
    parser.add_argument(
        "--video_steps", type=int, default=1500,
        help="Number of steps to record for video (default: 150)",
    )
    args = parser.parse_args()
    collect_grasps(args)


if __name__ == "__main__":
    main()
