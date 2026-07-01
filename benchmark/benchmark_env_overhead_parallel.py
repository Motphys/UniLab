"""Standalone probe: parallelization ceiling for `NpEnv.update_state`.

Companion to issue #663 / #665. Measures how much speedup is achievable when
parallelizing the obs / reward / termination portion of `NpEnv.step` — the
"Env overhead" bucket reported by `benchmark_offpolicy_collector_active.py`.

The probe is intentionally decoupled from UniLab (no imports from `unilab`).
It mirrors real training shapes: `num_envs=8192`, `act_dim=29`,
`obs_dim=98/101`, `dtype=float32`, 15 active reward terms, 4 noise buffers,
per-reward log means, 2% `_reset_done_envs` copy. RNG is excluded from the hot
path (noise buffers pre-baked) so the reported ceiling reflects pure compute
scaling; RNG parallelization is a separate axis.

Three variants are benchmarked:
  * numpy vectorized baseline
  * `ThreadPoolExecutor + numpy shards` (explicit distribute / worker / aggregate)
  * `Numba prange` fused kernel (`@njit(parallel=True)`, per-thread log scratch)

Run:
    uv run --with numpy --with numba python benchmark/benchmark_env_overhead_parallel.py

    # Larger batch to amortize parallel overhead:
    PROBE_NUM_ENVS=32768 uv run --with numpy --with numba \\
        python benchmark/benchmark_env_overhead_parallel.py

    # Pin to one NUMA socket on multi-socket boxes (Xeon 8568Y+ etc.):
    PROBE_NUM_ENVS=32768 numactl -N 0 --membind 0 \\
        uv run --with numpy --with numba python benchmark/benchmark_env_overhead_parallel.py
"""

from __future__ import annotations

import concurrent.futures as _cf
import os
import time
from dataclasses import dataclass

import numpy as np

NUM_ENVS = int(os.environ.get("PROBE_NUM_ENVS", "8192"))
ACT_DIM = 29
OBS_DIM = 98
CRITIC_DIM = 101
CTRL_DT = 0.02
DTYPE = np.float32
DONE_FRAC = 0.02

N_WARMUP = 5
N_MEASURE = 100


def make_inputs(num_envs: int, act_dim: int, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    inp = dict(
        gyro=rng.standard_normal((num_envs, 3)).astype(DTYPE),
        gravity=rng.standard_normal((num_envs, 3)).astype(DTYPE),
        dof_pos=rng.standard_normal((num_envs, act_dim)).astype(DTYPE),
        dof_vel=rng.standard_normal((num_envs, act_dim)).astype(DTYPE),
        default_angles=rng.standard_normal((act_dim,)).astype(DTYPE),
        last_actions=rng.standard_normal((num_envs, act_dim)).astype(DTYPE),
        current_actions=rng.standard_normal((num_envs, act_dim)).astype(DTYPE),
        commands=rng.standard_normal((num_envs, 3)).astype(DTYPE),
        gait_phase=rng.uniform(0, 2 * np.pi, size=(num_envs, 2)).astype(DTYPE),
        pose_weights=np.array(
            [0.01, 2.0, 5.0, 0.01, 5.0, 5.0, 0.01, 2.0, 5.0, 0.01, 5.0, 5.0]
            + [50.0] * 17, dtype=DTYPE,
        ),
        linvel=rng.standard_normal((num_envs, 3)).astype(DTYPE),
        left_foot_pos=rng.standard_normal((num_envs, 3)).astype(DTYPE),
        right_foot_pos=rng.standard_normal((num_envs, 3)).astype(DTYPE),
        left_foot_quat=rng.standard_normal((num_envs, 4)).astype(DTYPE),
        right_foot_quat=rng.standard_normal((num_envs, 4)).astype(DTYPE),
        left_contact=(rng.random((num_envs,)) > 0.5).astype(np.bool_),
        right_contact=(rng.random((num_envs,)) > 0.5).astype(np.bool_),
        base_height=rng.uniform(0.5, 0.9, size=(num_envs,)).astype(DTYPE),
        target_pose=rng.standard_normal((num_envs, act_dim)).astype(DTYPE),
        target_lin_vel=rng.standard_normal((num_envs, 2)).astype(DTYPE),
        target_ang_vel=rng.standard_normal((num_envs,)).astype(DTYPE),
    )
    # Pre-baked noise buffers — reused every step so RNG never runs in hot path.
    inp["noise_gyro"] = (rng.standard_normal((num_envs, 3)) * NOISE_SCALES["gyro"]).astype(DTYPE)
    inp["noise_gravity"] = (rng.standard_normal((num_envs, 3)) * NOISE_SCALES["gravity"]).astype(DTYPE)
    inp["noise_dof_pos"] = (rng.standard_normal((num_envs, act_dim)) * NOISE_SCALES["dof_pos"]).astype(DTYPE)
    inp["noise_dof_vel"] = (rng.standard_normal((num_envs, act_dim)) * NOISE_SCALES["dof_vel"]).astype(DTYPE)
    return inp


NOISE_SCALES = dict(gyro=0.2, gravity=0.05, dof_pos=0.03, dof_vel=1.5)

# scales close to sac/g1_motion_tracking active set (15 terms) — magnitudes not
# calibrated for behavior; only counts and shapes matter for perf.
REWARD_SCALES: dict[str, float] = {
    "tracking_lin_vel": 1.5,
    "tracking_ang_vel": 0.8,
    "tracking_pose": 2.0,
    "penalty_ang_vel_xy": -1.0,
    "penalty_orientation": -8.0,
    "penalty_action_rate": -2.5,
    "penalty_action_smooth": -1.0,
    "penalty_feet_ori": -3.0,
    "penalty_close_feet_xy": -2.0,
    "feet_phase": 5.0,
    "feet_phase_contrast": 2.0,
    "feet_phase_contact": 3.0,
    "feet_double_stance": -1.5,
    "base_height": -20.0,
    "alive": 10.0,
}


# =====================================================================
# Realistic numpy baseline
# =====================================================================


def _obs_noise(rng, arr: np.ndarray, scale: float) -> np.ndarray:
    return arr + rng.standard_normal(arr.shape).astype(DTYPE) * DTYPE(scale)


def _feet_phase_target(gait_phase_col: np.ndarray, swing_h: float) -> np.ndarray:
    phi = np.fmod(gait_phase_col + np.pi, 2 * np.pi) - np.pi
    x = (phi + np.pi) / (2 * np.pi)
    tl = 2 * x
    stance = swing_h * (tl**3 + 3 * tl**2 * (1 - tl))
    tl2 = 2 * x - 1
    swing = swing_h + (-swing_h) * (tl2**3 + 3 * tl2**2 * (1 - tl2))
    return np.where(x <= 0.5, stance, swing).astype(DTYPE)


def numpy_env_overhead(
    inp: dict,
    log_out: dict,
    enable_log: bool,
    reward_buf: np.ndarray,
    actor_buf: np.ndarray,
    critic_buf: np.ndarray,
    term_buf: np.ndarray,
    final_obs_buf: dict,
    done_mask_scratch: np.ndarray,
) -> None:
    dof_pos = inp["dof_pos"]
    dof_vel = inp["dof_vel"]
    default_angles = inp["default_angles"]
    gyro = inp["gyro"]
    gravity = inp["gravity"]
    linvel = inp["linvel"]
    last_actions = inp["last_actions"]
    current_actions = inp["current_actions"]
    commands = inp["commands"]
    gait_phase = inp["gait_phase"]
    pose_weights = inp["pose_weights"]
    left_foot_pos = inp["left_foot_pos"]
    right_foot_pos = inp["right_foot_pos"]
    left_foot_quat = inp["left_foot_quat"]
    right_foot_quat = inp["right_foot_quat"]
    left_contact = inp["left_contact"]
    right_contact = inp["right_contact"]
    target_pose = inp["target_pose"]
    target_lin_vel = inp["target_lin_vel"]
    target_ang_vel = inp["target_ang_vel"]

    diff = dof_pos - default_angles
    max_tilt_rad = np.deg2rad(65.0)
    tilt = np.arccos(np.clip(gravity[:, 2], -1, 1))
    np.logical_or(tilt > max_tilt_rad, inp["base_height"] < 0.3, out=term_buf)

    # --- reward terms ---
    sigma = 0.25
    r = {}
    r["tracking_lin_vel"] = np.exp(
        -np.sum(np.square(linvel[:, :2] - target_lin_vel), axis=1) / sigma
    )
    r["tracking_ang_vel"] = np.exp(-np.square(gyro[:, 2] - target_ang_vel) / sigma)
    r["tracking_pose"] = np.exp(
        -np.sum(pose_weights * np.square(dof_pos - target_pose), axis=1) / (10 * sigma)
    )
    r["penalty_ang_vel_xy"] = np.sum(np.square(gyro[:, :2]), axis=1)
    r["penalty_orientation"] = np.sum(np.square(gravity[:, :2]), axis=1)
    r["penalty_action_rate"] = np.sum(np.square(current_actions - last_actions), axis=1)
    r["penalty_action_smooth"] = np.sum(np.square(current_actions), axis=1)
    r["penalty_feet_ori"] = (
        np.square(left_foot_quat[:, 1]) + np.square(left_foot_quat[:, 2])
        + np.square(right_foot_quat[:, 1]) + np.square(right_foot_quat[:, 2])
    )
    feet_dist = np.linalg.norm(left_foot_pos[:, :2] - right_foot_pos[:, :2], axis=1)
    r["penalty_close_feet_xy"] = np.where(
        feet_dist < 0.15, np.square(feet_dist - 0.15), DTYPE(0.0)
    )
    swing_h = 0.09
    lt = _feet_phase_target(gait_phase[:, 0], swing_h)
    rt = _feet_phase_target(gait_phase[:, 1], swing_h)
    r["feet_phase"] = np.exp(
        -(np.square(left_foot_pos[:, 2] - lt) + np.square(right_foot_pos[:, 2] - rt))
        / 0.008
    )
    r["feet_phase_contrast"] = np.exp(
        -np.square((left_foot_pos[:, 2] - right_foot_pos[:, 2]) - (lt - rt)) / 0.008
    )
    left_target_contact = lt <= swing_h * 0.5
    right_target_contact = rt <= swing_h * 0.5
    r["feet_phase_contact"] = 0.5 * (
        (left_contact == left_target_contact).astype(DTYPE)
        + (right_contact == right_target_contact).astype(DTYPE)
    )
    r["feet_double_stance"] = np.logical_and(left_contact, right_contact).astype(DTYPE)
    r["base_height"] = np.square(inp["base_height"] - 0.754)
    r["alive"] = np.ones(dof_pos.shape[0], dtype=DTYPE)

    reward_buf.fill(0.0)
    for name, rew in r.items():
        scale = REWARD_SCALES[name]
        weighted = rew * DTYPE(scale)
        reward_buf += weighted
        if enable_log:
            log_out[f"reward/{name}"] = float(np.mean(weighted))
    reward_buf *= DTYPE(CTRL_DT)

    # --- obs (noise added from pre-baked buffers, no rng call) ---
    noisy_gyro = gyro + inp["noise_gyro"]
    noisy_gravity = gravity + inp["noise_gravity"]
    noisy_diff = diff + inp["noise_dof_pos"]
    noisy_dof_vel = dof_vel + inp["noise_dof_vel"]
    np.concatenate(
        [
            noisy_gyro * DTYPE(0.25), -noisy_gravity, noisy_diff,
            noisy_dof_vel * DTYPE(0.05), current_actions, commands, gait_phase,
        ],
        axis=1, out=actor_buf,
    )
    np.concatenate(
        [
            gyro * DTYPE(0.25), -gravity, diff, dof_vel * DTYPE(0.05),
            current_actions, commands, gait_phase, linvel * DTYPE(2.0),
        ],
        axis=1, out=critic_buf,
    )

    # --- reset-done shape: final_observation copy on 2% of envs ---
    done_mask_scratch[:] = False
    n_done = max(1, int(DONE_FRAC * dof_pos.shape[0]))
    done_mask_scratch[:n_done] = True
    done_idx = np.flatnonzero(done_mask_scratch).astype(np.int32)
    final_obs_buf["obs"][done_idx] = actor_buf[done_idx]
    final_obs_buf["critic"][done_idx] = critic_buf[done_idx]


# =====================================================================
# ThreadPool numpy shard variant: explicit distribute + aggregate
# =====================================================================


def numpy_shard_worker(
    inp: dict,
    slice_start: int,
    slice_end: int,
    partial_log_sum: dict,
    reward_buf: np.ndarray,
    actor_buf: np.ndarray,
    critic_buf: np.ndarray,
    term_buf: np.ndarray,
    enable_log: bool,
) -> None:
    s = slice(slice_start, slice_end)
    dof_pos = inp["dof_pos"][s]
    dof_vel = inp["dof_vel"][s]
    default_angles = inp["default_angles"]
    gyro = inp["gyro"][s]
    gravity = inp["gravity"][s]
    linvel = inp["linvel"][s]
    last_actions = inp["last_actions"][s]
    current_actions = inp["current_actions"][s]
    commands = inp["commands"][s]
    gait_phase = inp["gait_phase"][s]
    pose_weights = inp["pose_weights"]
    left_foot_pos = inp["left_foot_pos"][s]
    right_foot_pos = inp["right_foot_pos"][s]
    left_foot_quat = inp["left_foot_quat"][s]
    right_foot_quat = inp["right_foot_quat"][s]
    left_contact = inp["left_contact"][s]
    right_contact = inp["right_contact"][s]
    target_pose = inp["target_pose"][s]
    target_lin_vel = inp["target_lin_vel"][s]
    target_ang_vel = inp["target_ang_vel"][s]
    base_h = inp["base_height"][s]

    diff = dof_pos - default_angles
    max_tilt_rad = np.deg2rad(65.0)
    tilt = np.arccos(np.clip(gravity[:, 2], -1, 1))
    term_buf[s] = np.logical_or(tilt > max_tilt_rad, base_h < 0.3)

    sigma = 0.25
    r = {}
    r["tracking_lin_vel"] = np.exp(
        -np.sum(np.square(linvel[:, :2] - target_lin_vel), axis=1) / sigma
    )
    r["tracking_ang_vel"] = np.exp(-np.square(gyro[:, 2] - target_ang_vel) / sigma)
    r["tracking_pose"] = np.exp(
        -np.sum(pose_weights * np.square(dof_pos - target_pose), axis=1) / (10 * sigma)
    )
    r["penalty_ang_vel_xy"] = np.sum(np.square(gyro[:, :2]), axis=1)
    r["penalty_orientation"] = np.sum(np.square(gravity[:, :2]), axis=1)
    r["penalty_action_rate"] = np.sum(np.square(current_actions - last_actions), axis=1)
    r["penalty_action_smooth"] = np.sum(np.square(current_actions), axis=1)
    r["penalty_feet_ori"] = (
        np.square(left_foot_quat[:, 1]) + np.square(left_foot_quat[:, 2])
        + np.square(right_foot_quat[:, 1]) + np.square(right_foot_quat[:, 2])
    )
    feet_dist = np.linalg.norm(left_foot_pos[:, :2] - right_foot_pos[:, :2], axis=1)
    r["penalty_close_feet_xy"] = np.where(
        feet_dist < 0.15, np.square(feet_dist - 0.15), DTYPE(0.0)
    )
    swing_h = 0.09
    lt = _feet_phase_target(gait_phase[:, 0], swing_h)
    rt = _feet_phase_target(gait_phase[:, 1], swing_h)
    r["feet_phase"] = np.exp(
        -(np.square(left_foot_pos[:, 2] - lt) + np.square(right_foot_pos[:, 2] - rt))
        / 0.008
    )
    r["feet_phase_contrast"] = np.exp(
        -np.square((left_foot_pos[:, 2] - right_foot_pos[:, 2]) - (lt - rt)) / 0.008
    )
    left_target_contact = lt <= swing_h * 0.5
    right_target_contact = rt <= swing_h * 0.5
    r["feet_phase_contact"] = 0.5 * (
        (left_contact == left_target_contact).astype(DTYPE)
        + (right_contact == right_target_contact).astype(DTYPE)
    )
    r["feet_double_stance"] = np.logical_and(left_contact, right_contact).astype(DTYPE)
    r["base_height"] = np.square(base_h - 0.754)
    r["alive"] = np.ones(dof_pos.shape[0], dtype=DTYPE)

    rew_slice = reward_buf[s]
    rew_slice.fill(0.0)
    for name, rew in r.items():
        scale = REWARD_SCALES[name]
        weighted = rew * DTYPE(scale)
        rew_slice += weighted
        if enable_log:
            partial_log_sum[name] = partial_log_sum.get(name, 0.0) + float(
                weighted.sum()
            )
    rew_slice *= DTYPE(CTRL_DT)

    noisy_gyro = gyro + inp["noise_gyro"][s]
    noisy_gravity = gravity + inp["noise_gravity"][s]
    noisy_diff = diff + inp["noise_dof_pos"][s]
    noisy_dof_vel = dof_vel + inp["noise_dof_vel"][s]
    np.concatenate(
        [
            noisy_gyro * DTYPE(0.25), -noisy_gravity, noisy_diff,
            noisy_dof_vel * DTYPE(0.05), current_actions, commands, gait_phase,
        ],
        axis=1, out=actor_buf[s],
    )
    np.concatenate(
        [
            gyro * DTYPE(0.25), -gravity, diff, dof_vel * DTYPE(0.05),
            current_actions, commands, gait_phase, linvel * DTYPE(2.0),
        ],
        axis=1, out=critic_buf[s],
    )


@dataclass
class ShardTiming:
    distribute_ms: float
    worker_wall_ms: float
    aggregate_ms: float
    total_ms: float


def run_shard_pool(
    pool: _cf.ThreadPoolExecutor,
    n_shards: int,
    inp: dict,
    reward_buf: np.ndarray,
    actor_buf: np.ndarray,
    critic_buf: np.ndarray,
    term_buf: np.ndarray,
    final_obs_buf: dict,
    done_mask_scratch: np.ndarray,
    log_out: dict,
    enable_log: bool,
) -> ShardTiming:
    n = reward_buf.shape[0]
    chunk = (n + n_shards - 1) // n_shards

    t_total = time.perf_counter()
    t0 = time.perf_counter()
    partial_logs: list[dict] = [{} for _ in range(n_shards)]
    futures = []
    for k in range(n_shards):
        start = k * chunk
        end = min(start + chunk, n)
        if start >= end:
            break
        futures.append(pool.submit(
            numpy_shard_worker,
            inp, start, end, partial_logs[k],
            reward_buf, actor_buf, critic_buf, term_buf, enable_log,
        ))
    distribute_ms = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    for f in futures:
        f.result()
    worker_wall_ms = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    if enable_log:
        for name in REWARD_SCALES:
            total = 0.0
            for pl in partial_logs:
                total += pl.get(name, 0.0)
            log_out[f"reward/{name}"] = total / n
    done_mask_scratch[:] = False
    n_done = max(1, int(DONE_FRAC * n))
    done_mask_scratch[:n_done] = True
    done_idx = np.flatnonzero(done_mask_scratch).astype(np.int32)
    final_obs_buf["obs"][done_idx] = actor_buf[done_idx]
    final_obs_buf["critic"][done_idx] = critic_buf[done_idx]
    aggregate_ms = (time.perf_counter() - t0) * 1000.0

    total_ms = (time.perf_counter() - t_total) * 1000.0
    return ShardTiming(distribute_ms, worker_wall_ms, aggregate_ms, total_ms)


# =====================================================================
# Numba variant — one prange over env shards, inline everything (no noise)
# =====================================================================

from numba import get_thread_id, njit, prange  # noqa: E402


@njit(parallel=True, cache=True, fastmath=True, boundscheck=False)
def numba_env_overhead(
    gyro, gravity, dof_pos, dof_vel, default_angles,
    last_actions, current_actions, commands, gait_phase,
    pose_weights, linvel,
    left_foot_pos, right_foot_pos, left_foot_quat, right_foot_quat,
    left_contact, right_contact, target_pose, target_lin_vel, target_ang_vel,
    base_height,
    noise_gyro, noise_gravity, noise_diff, noise_dof_vel,
    reward_out, actor_out, critic_out, terminated_out, log_thread_sum,
    sigma, feet_sigma, swing_h, ctrl_dt, max_tilt_rad, min_base_h,
    obs_dim,
):
    """Single fused kernel. log_thread_sum has shape (num_threads, num_terms)
    — each thread accumulates into its own row to avoid false sharing / races.
    Caller sums along axis=0 and divides by N to get per-term mean."""
    N = dof_pos.shape[0]
    A = dof_pos.shape[1]
    two_pi = np.float32(2.0 * np.pi)
    pi = np.float32(np.pi)

    for i in prange(N):
        tid = get_thread_id()
        # ---- termination
        g2 = gravity[i, 2]
        if g2 > 1.0:
            g2 = 1.0
        elif g2 < -1.0:
            g2 = -1.0
        tilt = np.arccos(g2)
        terminated_out[i] = (tilt > max_tilt_rad) or (base_height[i] < min_base_h)

        # ---- pose accumulators
        r_pose_track = np.float32(0.0)
        for j in range(A):
            d = dof_pos[i, j] - target_pose[i, j]
            r_pose_track += pose_weights[j] * d * d
        r_track_pose = np.exp(-r_pose_track / (10 * sigma))

        # ---- reward terms
        e0 = linvel[i, 0] - target_lin_vel[i, 0]
        e1 = linvel[i, 1] - target_lin_vel[i, 1]
        r_track_lin = np.exp(-(e0 * e0 + e1 * e1) / sigma)

        e2 = gyro[i, 2] - target_ang_vel[i]
        r_track_ang = np.exp(-(e2 * e2) / sigma)

        r_ang_xy = gyro[i, 0] * gyro[i, 0] + gyro[i, 1] * gyro[i, 1]
        r_ori = gravity[i, 0] * gravity[i, 0] + gravity[i, 1] * gravity[i, 1]

        r_ar = np.float32(0.0)
        r_asm = np.float32(0.0)
        for j in range(A):
            dd = current_actions[i, j] - last_actions[i, j]
            r_ar += dd * dd
            r_asm += current_actions[i, j] * current_actions[i, j]

        r_feet_ori = (
            left_foot_quat[i, 1] * left_foot_quat[i, 1]
            + left_foot_quat[i, 2] * left_foot_quat[i, 2]
            + right_foot_quat[i, 1] * right_foot_quat[i, 1]
            + right_foot_quat[i, 2] * right_foot_quat[i, 2]
        )

        dx = left_foot_pos[i, 0] - right_foot_pos[i, 0]
        dy = left_foot_pos[i, 1] - right_foot_pos[i, 1]
        feet_d = np.sqrt(dx * dx + dy * dy)
        if feet_d < 0.15:
            r_close = (feet_d - 0.15) * (feet_d - 0.15)
        else:
            r_close = np.float32(0.0)

        phi_l = ((gait_phase[i, 0] + pi) % two_pi) - pi
        xl = (phi_l + pi) / two_pi
        if xl <= 0.5:
            tl = 2 * xl
            lt = swing_h * (tl * tl * tl + 3 * tl * tl * (1 - tl))
        else:
            tl2 = 2 * xl - 1
            lt = swing_h - swing_h * (tl2 * tl2 * tl2 + 3 * tl2 * tl2 * (1 - tl2))
        phi_r = ((gait_phase[i, 1] + pi) % two_pi) - pi
        xr = (phi_r + pi) / two_pi
        if xr <= 0.5:
            tr = 2 * xr
            rt = swing_h * (tr * tr * tr + 3 * tr * tr * (1 - tr))
        else:
            tr2 = 2 * xr - 1
            rt = swing_h - swing_h * (tr2 * tr2 * tr2 + 3 * tr2 * tr2 * (1 - tr2))
        lz = left_foot_pos[i, 2] - lt
        rz = right_foot_pos[i, 2] - rt
        r_feet_phase = np.exp(-(lz * lz + rz * rz) / feet_sigma)

        actual_dz = left_foot_pos[i, 2] - right_foot_pos[i, 2]
        target_dz = lt - rt
        edz = actual_dz - target_dz
        r_feet_phase_contrast = np.exp(-(edz * edz) / feet_sigma)

        lt_contact = lt <= swing_h * 0.5
        rt_contact = rt <= swing_h * 0.5
        r_feet_phase_contact = np.float32(0.0)
        if left_contact[i] == lt_contact:
            r_feet_phase_contact += 0.5
        if right_contact[i] == rt_contact:
            r_feet_phase_contact += 0.5

        r_double_stance = np.float32(0.0)
        if left_contact[i] and right_contact[i]:
            r_double_stance = np.float32(1.0)

        r_bh = (base_height[i] - 0.754) * (base_height[i] - 0.754)
        r_alive = np.float32(1.0)

        w_track_lin = np.float32(1.5) * r_track_lin
        w_track_ang = np.float32(0.8) * r_track_ang
        w_track_pose = np.float32(2.0) * r_track_pose
        w_ang_xy = np.float32(-1.0) * r_ang_xy
        w_ori = np.float32(-8.0) * r_ori
        w_ar = np.float32(-2.5) * r_ar
        w_asm = np.float32(-1.0) * r_asm
        w_feet_ori = np.float32(-3.0) * r_feet_ori
        w_close = np.float32(-2.0) * r_close
        w_feet_phase = np.float32(5.0) * r_feet_phase
        w_feet_phase_contrast = np.float32(2.0) * r_feet_phase_contrast
        w_feet_phase_contact = np.float32(3.0) * r_feet_phase_contact
        w_double_stance = np.float32(-1.5) * r_double_stance
        w_bh = np.float32(-20.0) * r_bh
        w_alive = np.float32(10.0) * r_alive

        reward_out[i] = ctrl_dt * (
            w_track_lin + w_track_ang + w_track_pose
            + w_ang_xy + w_ori + w_ar + w_asm + w_feet_ori
            + w_close + w_feet_phase + w_feet_phase_contrast
            + w_feet_phase_contact + w_double_stance + w_bh + w_alive
        )

        # per-thread log accumulation (no false sharing across threads)
        log_thread_sum[tid, 0] += w_track_lin
        log_thread_sum[tid, 1] += w_track_ang
        log_thread_sum[tid, 2] += w_track_pose
        log_thread_sum[tid, 3] += w_ang_xy
        log_thread_sum[tid, 4] += w_ori
        log_thread_sum[tid, 5] += w_ar
        log_thread_sum[tid, 6] += w_asm
        log_thread_sum[tid, 7] += w_feet_ori
        log_thread_sum[tid, 8] += w_close
        log_thread_sum[tid, 9] += w_feet_phase
        log_thread_sum[tid, 10] += w_feet_phase_contrast
        log_thread_sum[tid, 11] += w_feet_phase_contact
        log_thread_sum[tid, 12] += w_double_stance
        log_thread_sum[tid, 13] += w_bh
        log_thread_sum[tid, 14] += w_alive

        # ---- actor obs (with noise added element-wise)
        actor_out[i, 0] = (gyro[i, 0] + noise_gyro[i, 0]) * 0.25
        actor_out[i, 1] = (gyro[i, 1] + noise_gyro[i, 1]) * 0.25
        actor_out[i, 2] = (gyro[i, 2] + noise_gyro[i, 2]) * 0.25
        actor_out[i, 3] = -(gravity[i, 0] + noise_gravity[i, 0])
        actor_out[i, 4] = -(gravity[i, 1] + noise_gravity[i, 1])
        actor_out[i, 5] = -(gravity[i, 2] + noise_gravity[i, 2])
        off = 6
        for j in range(A):
            d = dof_pos[i, j] - default_angles[j] + noise_diff[i, j]
            actor_out[i, off + j] = d
        off += A
        for j in range(A):
            actor_out[i, off + j] = (dof_vel[i, j] + noise_dof_vel[i, j]) * 0.05
        off += A
        for j in range(A):
            actor_out[i, off + j] = current_actions[i, j]
        off += A
        actor_out[i, off + 0] = commands[i, 0]
        actor_out[i, off + 1] = commands[i, 1]
        actor_out[i, off + 2] = commands[i, 2]
        off += 3
        actor_out[i, off + 0] = gait_phase[i, 0]
        actor_out[i, off + 1] = gait_phase[i, 1]

        # ---- critic: gyro(3)*0.25, -gravity(3), diff(29), dof_vel*0.05(29),
        #              current_actions(29), commands(3), gait_phase(2), linvel*2(3)
        critic_out[i, 0] = gyro[i, 0] * 0.25
        critic_out[i, 1] = gyro[i, 1] * 0.25
        critic_out[i, 2] = gyro[i, 2] * 0.25
        critic_out[i, 3] = -gravity[i, 0]
        critic_out[i, 4] = -gravity[i, 1]
        critic_out[i, 5] = -gravity[i, 2]
        off = 6
        for j in range(A):
            critic_out[i, off + j] = dof_pos[i, j] - default_angles[j]
        off += A
        for j in range(A):
            critic_out[i, off + j] = dof_vel[i, j] * 0.05
        off += A
        for j in range(A):
            critic_out[i, off + j] = current_actions[i, j]
        off += A
        critic_out[i, off + 0] = commands[i, 0]
        critic_out[i, off + 1] = commands[i, 1]
        critic_out[i, off + 2] = commands[i, 2]
        off += 3
        critic_out[i, off + 0] = gait_phase[i, 0]
        critic_out[i, off + 1] = gait_phase[i, 1]
        off += 2
        critic_out[i, off + 0] = linvel[i, 0] * 2.0
        critic_out[i, off + 1] = linvel[i, 1] * 2.0
        critic_out[i, off + 2] = linvel[i, 2] * 2.0


def bench(fn, iters: int, name: str) -> float:
    for _ in range(N_WARMUP):
        fn()
    times_ms = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        times_ms.append((time.perf_counter() - t0) * 1000.0)
    arr = np.asarray(times_ms)
    print(
        f"  {name:44s}  mean={arr.mean():7.3f}  median={np.median(arr):7.3f}  "
        f"min={arr.min():7.3f}  max={arr.max():7.3f}  (ms)"
    )
    return float(arr.mean())


def main() -> None:
    import numba

    print(f"host cpu_count = {os.cpu_count()}")
    print(f"numba version  = {numba.__version__}")
    print(
        f"config: num_envs={NUM_ENVS}  act_dim={ACT_DIM}  obs_dim={OBS_DIM}  "
        f"critic_dim={CRITIC_DIM}  dtype={DTYPE.__name__}  done_frac={DONE_FRAC}"
    )
    print(f"reward terms: {len(REWARD_SCALES)}  |  noise arrays: 4  |  log: enabled")
    # NUMA hint: for two-socket boxes, pin to one node via `numactl -N 0 --membind 0 ...`
    print()

    inp = make_inputs(NUM_ENVS, ACT_DIM)
    reward_buf = np.zeros((NUM_ENVS,), dtype=DTYPE)
    actor_buf = np.zeros((NUM_ENVS, OBS_DIM), dtype=DTYPE)
    critic_buf = np.zeros((NUM_ENVS, CRITIC_DIM), dtype=DTYPE)
    term_buf = np.zeros((NUM_ENVS,), dtype=np.bool_)
    final_obs = {
        "obs": np.zeros_like(actor_buf),
        "critic": np.zeros_like(critic_buf),
    }
    done_mask_scratch = np.zeros((NUM_ENVS,), dtype=np.bool_)
    log_out: dict = {}

    # === Numpy baseline ===
    def run_np() -> None:
        numpy_env_overhead(
            inp, log_out, True,
            reward_buf, actor_buf, critic_buf, term_buf,
            final_obs, done_mask_scratch,
        )

    print(f"benchmarks (N_MEASURE={N_MEASURE}):  [RNG excluded — noise buffers pre-baked]")
    ms_np = bench(run_np, N_MEASURE, "numpy baseline (no rng, 15 rewards, log)")

    # === ThreadPool numpy shards ===
    print()
    print("ThreadPool numpy shards (distribute / worker / aggregate breakdown):")
    for n_shards in [2, 4, 8, 16, 24, 32, 48, 64, 96, 128]:
        if n_shards > (os.cpu_count() or 1):
            continue
        pool = _cf.ThreadPoolExecutor(max_workers=n_shards)

        # warmup
        for _ in range(N_WARMUP):
            run_shard_pool(
                pool, n_shards, inp,
                reward_buf, actor_buf, critic_buf, term_buf,
                final_obs, done_mask_scratch, log_out, True,
            )

        dist, worker, agg, total = [], [], [], []
        for _ in range(N_MEASURE):
            t = run_shard_pool(
                pool, n_shards, inp,
                reward_buf, actor_buf, critic_buf, term_buf,
                final_obs, done_mask_scratch, log_out, True,
            )
            dist.append(t.distribute_ms)
            worker.append(t.worker_wall_ms)
            agg.append(t.aggregate_ms)
            total.append(t.total_ms)
        pool.shutdown(wait=True)
        d, w, a, tt = (np.mean(x) for x in (dist, worker, agg, total))
        print(
            f"  K={n_shards:3d}  total={tt:7.3f} ms  "
            f"[distribute={d:.3f}  worker_wall={w:.3f}  aggregate={a:.3f}]  "
            f"speedup={ms_np/tt:.2f}x"
        )

    # === Numba prange fused kernel ===
    print()
    print("Numba prange fused kernel (noise buffers pre-baked, no rng in hot path):")
    max_threads = numba.config.NUMBA_NUM_THREADS
    log_thread_sum = np.zeros((max_threads, len(REWARD_SCALES)), dtype=np.float64)

    t0 = time.perf_counter()
    numba_env_overhead(
        inp["gyro"], inp["gravity"], inp["dof_pos"], inp["dof_vel"],
        inp["default_angles"], inp["last_actions"], inp["current_actions"],
        inp["commands"], inp["gait_phase"], inp["pose_weights"], inp["linvel"],
        inp["left_foot_pos"], inp["right_foot_pos"],
        inp["left_foot_quat"], inp["right_foot_quat"],
        inp["left_contact"], inp["right_contact"],
        inp["target_pose"], inp["target_lin_vel"], inp["target_ang_vel"],
        inp["base_height"],
        inp["noise_gyro"], inp["noise_gravity"],
        inp["noise_dof_pos"], inp["noise_dof_vel"],
        reward_buf, actor_buf, critic_buf, term_buf, log_thread_sum,
        np.float32(0.25), np.float32(0.008), np.float32(0.09),
        np.float32(CTRL_DT), np.float32(np.deg2rad(65.0)), np.float32(0.3),
        OBS_DIM,
    )
    print(f"  JIT compile + first call: {(time.perf_counter() - t0) * 1000:.1f} ms  "
          f"(threading layer: {numba.threading_layer()})")

    def run_nb_full() -> dict:
        t_all = time.perf_counter()
        log_thread_sum.fill(0.0)
        t_k = time.perf_counter()
        numba_env_overhead(
            inp["gyro"], inp["gravity"], inp["dof_pos"], inp["dof_vel"],
            inp["default_angles"], inp["last_actions"], inp["current_actions"],
            inp["commands"], inp["gait_phase"], inp["pose_weights"], inp["linvel"],
            inp["left_foot_pos"], inp["right_foot_pos"],
            inp["left_foot_quat"], inp["right_foot_quat"],
            inp["left_contact"], inp["right_contact"],
            inp["target_pose"], inp["target_lin_vel"], inp["target_ang_vel"],
            inp["base_height"],
            inp["noise_gyro"], inp["noise_gravity"],
            inp["noise_dof_pos"], inp["noise_dof_vel"],
            reward_buf, actor_buf, critic_buf, term_buf, log_thread_sum,
            np.float32(0.25), np.float32(0.008), np.float32(0.09),
            np.float32(CTRL_DT), np.float32(np.deg2rad(65.0)), np.float32(0.3),
            OBS_DIM,
        )
        kernel_ms = (time.perf_counter() - t_k) * 1000.0

        t_ag = time.perf_counter()
        reduced = log_thread_sum.sum(axis=0)
        for k, name in enumerate(REWARD_SCALES):
            log_out[f"reward/{name}"] = float(reduced[k]) / NUM_ENVS
        done_mask_scratch[:] = False
        n_done = max(1, int(DONE_FRAC * NUM_ENVS))
        done_mask_scratch[:n_done] = True
        done_idx = np.flatnonzero(done_mask_scratch).astype(np.int32)
        final_obs["obs"][done_idx] = actor_buf[done_idx]
        final_obs["critic"][done_idx] = critic_buf[done_idx]
        agg_ms = (time.perf_counter() - t_ag) * 1000.0

        total_ms = (time.perf_counter() - t_all) * 1000.0
        return dict(kernel=kernel_ms, agg=agg_ms, total=total_ms)

    for nthreads in [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128]:
        if nthreads > (os.cpu_count() or 1):
            continue
        numba.set_num_threads(nthreads)
        for _ in range(N_WARMUP):
            run_nb_full()
        acc = {"kernel": [], "agg": [], "total": []}
        for _ in range(N_MEASURE):
            t = run_nb_full()
            for k in acc:
                acc[k].append(t[k])
        means = {k: float(np.mean(v)) for k, v in acc.items()}
        print(
            f"  nthreads={nthreads:3d}  total={means['total']:7.3f} ms  "
            f"[kernel={means['kernel']:.3f}  agg={means['agg']:.3f}]  "
            f"speedup={ms_np/means['total']:.2f}x"
        )


if __name__ == "__main__":
    main()
