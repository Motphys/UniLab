import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from unilab.envs import registry
import unilab.envs.locomotion.go1.joystick  # noqa: F401


NUM_ENVS = 4096
OUTPUT_PATH = Path(
    "benchmark/outputs/mps_rollout_pipeline/benchmark_mps_rollout_pipeline_4096.json"
)
DEVICE = "mps"


def _sync_mps():
    if torch.backends.mps.is_available():
        torch.mps.synchronize()


def bench_env_step_breakdown():
    env = registry.make("Go1JoystickFlatTerrain", num_envs=NUM_ENVS)
    env.init_state()

    for _ in range(2):
        actions = np.random.uniform(
            -1, 1, (env.num_envs, env.action_space.shape[0])
        ).astype(np.float32)
        env.step(actions)

    n_iter = 8
    rollout_t = 0.0
    update_t = 0.0
    total_t = 0.0

    for _ in range(n_iter):
        actions = np.random.uniform(
            -1, 1, (env.num_envs, env.action_space.shape[0])
        ).astype(np.float32)
        t0 = time.perf_counter()
        env._prev_physics_step()
        env._state.ctrl[:] = env.apply_action(actions, env._state)
        t1 = time.perf_counter()
        env.physics_step()
        t2 = time.perf_counter()
        env._state = env.update_state(env._state, obs_required=True)
        t3 = time.perf_counter()
        env._state.info["steps"] += 1
        env._update_truncate()
        env._reset_done_envs()
        t4 = time.perf_counter()

        rollout_t += t2 - t1
        update_t += t3 - t2
        total_t += t4 - t0

    env.close()
    return {
        "num_envs": NUM_ENVS,
        "iterations": n_iter,
        "avg_total_ms": total_t / n_iter * 1000.0,
        "avg_rollout_ms": rollout_t / n_iter * 1000.0,
        "avg_update_state_ms": update_t / n_iter * 1000.0,
        "rollout_ratio": rollout_t / total_t,
        "update_state_ratio": update_t / total_t,
    }


def _numpy_postprocess(sd, idx):
    lin = sd[:, idx["lin"]]
    gyr = sd[:, idx["gyro"]]
    glin = sd[:, idx["glin"]]
    gang = sd[:, idx["gang"]]
    up = sd[:, idx["up"]]
    contacts = sd[:, idx["contacts"]] > 0.1
    feet = np.stack([sd[:, it] for it in idx["feet_linvel"]], axis=1)
    foot_pos = np.stack([sd[:, it] for it in idx["feet_pos"]], axis=1)
    v = np.sum(np.square(glin[:, :2] - lin[:, :2]), axis=1)
    a = np.square(gyr[:, 2] - gang[:, 2])
    o = np.sum(np.square(up[:, :2]), axis=1)
    slip = np.sum(np.sum(np.square(feet[..., :2]), axis=-1) * contacts, axis=1)
    z = foot_pos[..., 2].mean(axis=1)
    rew = np.exp(-v) + np.exp(-a) - 0.1 * o - 0.01 * slip + 0.001 * z
    obs = np.hstack([lin, gyr, up, glin, gang])
    return rew, obs


def _mps_postprocess_with_cpu_copyback(sd, idx):
    t = torch.as_tensor(sd, device=DEVICE, dtype=torch.float32)
    lin = t[:, idx["lin"]]
    gyr = t[:, idx["gyro"]]
    glin = t[:, idx["glin"]]
    gang = t[:, idx["gang"]]
    up = t[:, idx["up"]]
    contacts = t[:, idx["contacts"]] > 0.1
    feet = torch.stack([t[:, it] for it in idx["feet_linvel"]], dim=1)
    foot_pos = torch.stack([t[:, it] for it in idx["feet_pos"]], dim=1)
    v = torch.sum((glin[:, :2] - lin[:, :2]) ** 2, dim=1)
    a = (gyr[:, 2] - gang[:, 2]) ** 2
    o = torch.sum(up[:, :2] ** 2, dim=1)
    slip = torch.sum(torch.sum(feet[..., :2] ** 2, dim=-1) * contacts, dim=1)
    z = foot_pos[..., 2].mean(dim=1)
    rew = torch.exp(-v) + torch.exp(-a) - 0.1 * o - 0.01 * slip + 0.001 * z
    obs = torch.hstack([lin, gyr, up, glin, gang])
    rew_np = rew.cpu().numpy()
    obs_np = obs.cpu().numpy()
    return rew_np, obs_np


def _mps_only_postprocess(sd):
    t = torch.as_tensor(sd, device=DEVICE, dtype=torch.float32)
    a = t[:, :3]
    b = t[:, 3:6]
    c = t[:, 6:9]
    d = t[:, 9:12]
    e = t[:, 12:15]
    f = t[:, 15:19] > 0.1
    vals = torch.stack([t[:, 19:22], t[:, 22:25], t[:, 25:28], t[:, 28:31]], dim=1)
    rew = (
        torch.sum((a - b) ** 2, dim=1)
        + torch.sum(c[:, :2] ** 2, dim=1)
        + torch.sum(vals[..., :2] ** 2, dim=(-1, -2))
    )
    obs = torch.hstack([a, b, c, d, e])
    return rew, obs, f


def bench_micro_postprocess():
    idx = {
        "lin": [0, 1, 2],
        "gyro": [3, 4, 5],
        "glin": [6, 7, 8],
        "gang": [9, 10, 11],
        "up": [12, 13, 14],
        "contacts": [15, 16, 17, 18],
        "feet_linvel": [[19, 20, 21], [22, 23, 24], [25, 26, 27], [28, 29, 30]],
        "feet_pos": [[31, 32, 33], [34, 35, 36], [37, 38, 39], [40, 41, 42]],
    }
    sd = np.random.randn(NUM_ENVS, 56).astype(np.float32)

    for _ in range(10):
        _numpy_postprocess(sd, idx)
    for _ in range(20):
        _mps_postprocess_with_cpu_copyback(sd, idx)
    _sync_mps()

    n = 300
    t0 = time.perf_counter()
    for _ in range(n):
        _numpy_postprocess(sd, idx)
    t1 = time.perf_counter()

    t2 = time.perf_counter()
    for _ in range(n):
        _mps_postprocess_with_cpu_copyback(sd, idx)
    _sync_mps()
    t3 = time.perf_counter()

    np_ms = (t1 - t0) / n * 1000.0
    mps_ms = (t3 - t2) / n * 1000.0
    return {
        "num_envs": NUM_ENVS,
        "iterations": n,
        "numpy_ms": np_ms,
        "mps_ms_with_cpu_copyback": mps_ms,
        "speedup_numpy_div_mps": np_ms / mps_ms,
    }


def bench_mps_only():
    sd = np.random.randn(NUM_ENVS, 56).astype(np.float32)
    for _ in range(20):
        _mps_only_postprocess(sd)
    _sync_mps()

    n = 400
    t0 = time.perf_counter()
    for _ in range(n):
        _mps_only_postprocess(sd)
    _sync_mps()
    t1 = time.perf_counter()
    return {
        "num_envs": NUM_ENVS,
        "iterations": n,
        "mps_only_ms": (t1 - t0) / n * 1000.0,
    }


def bench_action_to_numpy():
    action = torch.randn(NUM_ENVS, 12, device=DEVICE)
    for _ in range(20):
        _ = action.cpu().numpy()
    _sync_mps()

    n = 500
    t0 = time.perf_counter()
    for _ in range(n):
        _ = action.cpu().numpy()
    _sync_mps()
    t1 = time.perf_counter()
    return {
        "num_envs": NUM_ENVS,
        "iterations": n,
        "action_mps_to_numpy_ms": (t1 - t0) / n * 1000.0,
    }


def bench_obs_reward_done_to_mps():
    obs = np.random.randn(NUM_ENVS, 45).astype(np.float32)
    rew = np.random.randn(NUM_ENVS).astype(np.float32)
    done = np.random.randint(0, 2, size=(NUM_ENVS,), dtype=np.bool_)

    for _ in range(20):
        _ = torch.tensor(obs, device=DEVICE, dtype=torch.float32)
        _ = torch.tensor(rew, device=DEVICE, dtype=torch.float32)
        _ = torch.tensor(done, device=DEVICE, dtype=torch.bool)
    _sync_mps()

    n = 500
    t0 = time.perf_counter()
    for _ in range(n):
        _ = torch.tensor(obs, device=DEVICE, dtype=torch.float32)
        _ = torch.tensor(rew, device=DEVICE, dtype=torch.float32)
        _ = torch.tensor(done, device=DEVICE, dtype=torch.bool)
    _sync_mps()
    t1 = time.perf_counter()
    return {
        "num_envs": NUM_ENVS,
        "iterations": n,
        "obs_reward_done_numpy_to_mps_ms": (t1 - t0) / n * 1000.0,
    }


def bench_rollout_arrays_to_mps():
    # Match Go1 dimensions measured in this repo.
    sensor_np = np.random.randn(NUM_ENVS, 56).astype(np.float64)
    physics_np = np.random.randn(NUM_ENVS, 38).astype(np.float64)

    for _ in range(20):
        _ = torch.as_tensor(sensor_np, device=DEVICE, dtype=torch.float32)
        _ = torch.as_tensor(physics_np, device=DEVICE, dtype=torch.float32)
    _sync_mps()

    n = 500
    t0 = time.perf_counter()
    for _ in range(n):
        _ = torch.as_tensor(sensor_np, device=DEVICE, dtype=torch.float32)
        _ = torch.as_tensor(physics_np, device=DEVICE, dtype=torch.float32)
    _sync_mps()
    t1 = time.perf_counter()
    return {
        "num_envs": NUM_ENVS,
        "iterations": n,
        "rollout_sensor_plus_physics_numpy_to_mps_ms": (t1 - t0) / n * 1000.0,
    }


def build_projected_pipeline_analysis(
    env_step_breakdown: dict,
    rollout_arrays_to_mps: dict,
    mps_only_postprocess: dict,
    action_to_numpy: dict,
    obs_reward_done_to_mps: dict,
):
    env_total = env_step_breakdown["avg_total_ms"]
    env_update = env_step_breakdown["avg_update_state_ms"]
    rollout_to_mps = rollout_arrays_to_mps["rollout_sensor_plus_physics_numpy_to_mps_ms"]
    mps_compute = mps_only_postprocess["mps_only_ms"]
    action_conv = action_to_numpy["action_mps_to_numpy_ms"]
    obs_conv = obs_reward_done_to_mps["obs_reward_done_numpy_to_mps_ms"]

    # Baseline (today): env.step on numpy + wrapper conversion + action conversion.
    baseline_training_step_ms = env_total + obs_conv + action_conv

    # Projected (your proposed pipeline):
    # rollout -> one-shot numpy->mps -> mps obs/reward -> obs direct to policy.
    # Replace numpy update_state with (numpy->mps + mps compute), and remove obs conversion.
    projected_env_step_ms = env_total - env_update + rollout_to_mps + mps_compute
    projected_training_step_ms = projected_env_step_ms + action_conv

    return {
        "assumptions": [
            "projected_env_step_ms replaces current numpy update_state by mps conversion + mps compute",
            "obs/reward/done numpy->mps conversion at wrapper side is removed when obs is produced on mps",
            "action mps->numpy conversion is still required for mujoco rollout input",
        ],
        "baseline_training_step_ms": baseline_training_step_ms,
        "projected_training_step_ms": projected_training_step_ms,
        "projected_env_step_ms": projected_env_step_ms,
        "estimated_speedup_x": baseline_training_step_ms / projected_training_step_ms,
        "estimated_reduction_ms": baseline_training_step_ms - projected_training_step_ms,
    }


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    env_step_breakdown = bench_env_step_breakdown()
    micro_postprocess_numpy_vs_mps = bench_micro_postprocess()
    micro_postprocess_mps_only = bench_mps_only()
    conversion_action_mps_to_numpy = bench_action_to_numpy()
    conversion_obs_reward_done_numpy_to_mps = bench_obs_reward_done_to_mps()
    conversion_rollout_arrays_numpy_to_mps = bench_rollout_arrays_to_mps()

    projected_pipeline_analysis = build_projected_pipeline_analysis(
        env_step_breakdown=env_step_breakdown,
        rollout_arrays_to_mps=conversion_rollout_arrays_numpy_to_mps,
        mps_only_postprocess=micro_postprocess_mps_only,
        action_to_numpy=conversion_action_mps_to_numpy,
        obs_reward_done_to_mps=conversion_obs_reward_done_numpy_to_mps,
    )

    results = {
        "meta": {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "num_envs": NUM_ENVS,
            "device": DEVICE,
            "torch_version": torch.__version__,
            "mps_built": bool(torch.backends.mps.is_built()),
            "mps_available": bool(torch.backends.mps.is_available()),
        },
        "env_step_breakdown": env_step_breakdown,
        "micro_postprocess_numpy_vs_mps": micro_postprocess_numpy_vs_mps,
        "micro_postprocess_mps_only": micro_postprocess_mps_only,
        "conversion_action_mps_to_numpy": conversion_action_mps_to_numpy,
        "conversion_obs_reward_done_numpy_to_mps": conversion_obs_reward_done_numpy_to_mps,
        "conversion_rollout_arrays_numpy_to_mps": conversion_rollout_arrays_numpy_to_mps,
        "projected_pipeline_analysis": projected_pipeline_analysis,
    }

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Saved benchmark results to: {OUTPUT_PATH}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
