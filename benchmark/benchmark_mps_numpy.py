import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


DEFAULT_ENV_LIST = [256, 512, 1024, 2048, 4096]
DEFAULT_ITERS = 200
OUTPUT_DIR = Path("benchmark/outputs/mps_rollout_pipeline")
OUTPUT_JSON = OUTPUT_DIR / "benchmark_mps_numpy_real_mac.json"
OUTPUT_PNG = OUTPUT_DIR / "mps_pipeline_same_task_two_modes_real_mac.png"
DEVICE = "mps"


def sync_device():
    if torch.backends.mps.is_available():
        torch.mps.synchronize()


def parse_env_list(raw: str) -> list[int]:
    if not raw:
        return DEFAULT_ENV_LIST
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def build_idx():
    return {
        "lin": [0, 1, 2],
        "gyro": [3, 4, 5],
        "glin": [6, 7, 8],
        "gang": [9, 10, 11],
        "up": [12, 13, 14],
        "contacts": [15, 16, 17, 18],
        "feet_linvel": [[19, 20, 21], [22, 23, 24], [25, 26, 27], [28, 29, 30]],
        "feet_pos": [[31, 32, 33], [34, 35, 36], [37, 38, 39], [40, 41, 42]],
    }


def numpy_postprocess(sd: np.ndarray, idx: dict):
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
    obs = np.hstack([lin, gyr, up, glin, gang]).astype(np.float32, copy=False)
    done = (contacts.sum(axis=1) <= 0).astype(np.bool_)
    return obs, rew.astype(np.float32, copy=False), done


def mps_postprocess(sensor_t: torch.Tensor, idx_t: dict):
    lin = sensor_t[:, idx_t["lin"]]
    gyr = sensor_t[:, idx_t["gyro"]]
    glin = sensor_t[:, idx_t["glin"]]
    gang = sensor_t[:, idx_t["gang"]]
    up = sensor_t[:, idx_t["up"]]
    contacts = sensor_t[:, idx_t["contacts"]] > 0.1
    feet = torch.stack([sensor_t[:, it] for it in idx_t["feet_linvel"]], dim=1)
    foot_pos = torch.stack([sensor_t[:, it] for it in idx_t["feet_pos"]], dim=1)
    v = torch.sum((glin[:, :2] - lin[:, :2]) ** 2, dim=1)
    a = (gyr[:, 2] - gang[:, 2]) ** 2
    o = torch.sum(up[:, :2] ** 2, dim=1)
    slip = torch.sum(torch.sum(feet[..., :2] ** 2, dim=-1) * contacts, dim=1)
    z = foot_pos[..., 2].mean(dim=1)
    rew = torch.exp(-v) + torch.exp(-a) - 0.1 * o - 0.01 * slip + 0.001 * z
    obs = torch.hstack([lin, gyr, up, glin, gang])
    done = contacts.sum(dim=1) <= 0
    return obs, rew, done


def bench_one_envnum(num_envs: int, iters: int, idx: dict):
    sd = np.random.randn(num_envs, 56).astype(np.float32)
    idx_t = {
        "lin": torch.as_tensor(idx["lin"], device=DEVICE, dtype=torch.long),
        "gyro": torch.as_tensor(idx["gyro"], device=DEVICE, dtype=torch.long),
        "glin": torch.as_tensor(idx["glin"], device=DEVICE, dtype=torch.long),
        "gang": torch.as_tensor(idx["gang"], device=DEVICE, dtype=torch.long),
        "up": torch.as_tensor(idx["up"], device=DEVICE, dtype=torch.long),
        "contacts": torch.as_tensor(idx["contacts"], device=DEVICE, dtype=torch.long),
        "feet_linvel": [torch.as_tensor(v, device=DEVICE, dtype=torch.long) for v in idx["feet_linvel"]],
        "feet_pos": [torch.as_tensor(v, device=DEVICE, dtype=torch.long) for v in idx["feet_pos"]],
    }

    for _ in range(20):
        obs_np, rew_np, done_np = numpy_postprocess(sd, idx)
        _ = torch.as_tensor(obs_np, device=DEVICE, dtype=torch.float32)
        _ = torch.as_tensor(rew_np, device=DEVICE, dtype=torch.float32)
        _ = torch.as_tensor(done_np, device=DEVICE, dtype=torch.bool)
    for _ in range(20):
        sensor_t = torch.as_tensor(sd, device=DEVICE, dtype=torch.float32)
        _ = mps_postprocess(sensor_t, idx_t)
    sync_device()

    cpu_compute = 0.0
    cpu_transfer = 0.0
    mps_transfer = 0.0
    mps_compute = 0.0

    for _ in range(iters):
        t0 = time.perf_counter()
        obs_np, rew_np, done_np = numpy_postprocess(sd, idx)
        t1 = time.perf_counter()
        _ = torch.as_tensor(obs_np, device=DEVICE, dtype=torch.float32)
        _ = torch.as_tensor(rew_np, device=DEVICE, dtype=torch.float32)
        _ = torch.as_tensor(done_np, device=DEVICE, dtype=torch.bool)
        sync_device()
        t2 = time.perf_counter()
        cpu_compute += t1 - t0
        cpu_transfer += t2 - t1

    for _ in range(iters):
        t0 = time.perf_counter()
        sensor_t = torch.as_tensor(sd, device=DEVICE, dtype=torch.float32)
        sync_device()
        t1 = time.perf_counter()
        _ = mps_postprocess(sensor_t, idx_t)
        sync_device()
        t2 = time.perf_counter()
        mps_transfer += t1 - t0
        mps_compute += t2 - t1

    cpu_compute_ms = cpu_compute / iters * 1000.0
    cpu_transfer_ms = cpu_transfer / iters * 1000.0
    mps_transfer_ms = mps_transfer / iters * 1000.0
    mps_compute_ms = mps_compute / iters * 1000.0
    cpu_total_ms = cpu_compute_ms + cpu_transfer_ms
    mps_total_ms = mps_transfer_ms + mps_compute_ms
    return {
        "num_envs": num_envs,
        "iterations": iters,
        "cpu_mode": {
            "compute_numpy_ms": cpu_compute_ms,
            "transfer_obs_rew_done_to_mps_ms": cpu_transfer_ms,
            "total_ms": cpu_total_ms,
        },
        "mps_mode": {
            "transfer_sensor_to_mps_ms": mps_transfer_ms,
            "compute_on_mps_ms": mps_compute_ms,
            "total_ms": mps_total_ms,
        },
        "speedup_cpu_div_mps": cpu_total_ms / mps_total_ms if mps_total_ms > 0 else 0.0,
    }


def plot_results(results: list[dict], output_png: Path):
    envs = [r["num_envs"] for r in results]
    x = np.arange(len(envs), dtype=float)
    w = 0.34
    cpu_compute = np.array([r["cpu_mode"]["compute_numpy_ms"] for r in results])
    cpu_transfer = np.array([r["cpu_mode"]["transfer_obs_rew_done_to_mps_ms"] for r in results])
    mps_transfer = np.array([r["mps_mode"]["transfer_sensor_to_mps_ms"] for r in results])
    mps_compute = np.array([r["mps_mode"]["compute_on_mps_ms"] for r in results])

    fig, ax = plt.subplots(figsize=(11, 6))
    x_cpu = x - w / 2
    x_mps = x + w / 2
    ax.bar(x_cpu, cpu_compute, w, label="CPU mode: numpy compute", color="#D99A9A")
    ax.bar(
        x_cpu,
        cpu_transfer,
        w,
        bottom=cpu_compute,
        label="CPU mode: obs/rew/done -> mps",
        color="#EBCED6",
    )
    ax.bar(x_mps, mps_transfer, w, label="MPS mode: sensor -> mps", color="#8FB3CC")
    ax.bar(
        x_mps,
        mps_compute,
        w,
        bottom=mps_transfer,
        label="MPS mode: mps compute",
        color="#A8CFAE",
    )

    cpu_total = cpu_compute + cpu_transfer
    mps_total = mps_transfer + mps_compute
    for i in range(len(envs)):
        ax.text(x_cpu[i], cpu_total[i] + 0.05, f"{cpu_total[i]:.2f}", ha="center", va="bottom", fontsize=9)
        ax.text(x_mps[i], mps_total[i] + 0.05, f"{mps_total[i]:.2f}", ha="center", va="bottom", fontsize=9)

    ax.set_title("Mac MPS: same-task two compute modes (measured)")
    ax.set_xlabel("num_envs")
    ax.set_ylabel("Time per step (ms)")
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in envs])
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0, fontsize=9)
    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_list", type=str, default=",".join(str(v) for v in DEFAULT_ENV_LIST))
    parser.add_argument("--iters", type=int, default=DEFAULT_ITERS)
    parser.add_argument("--output_json", type=str, default=str(OUTPUT_JSON))
    parser.add_argument("--output_png", type=str, default=str(OUTPUT_PNG))
    args = parser.parse_args()

    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available on this machine.")

    env_list = parse_env_list(args.env_list)
    idx = build_idx()
    all_results = []
    for nenv in env_list:
        one = bench_one_envnum(nenv, args.iters, idx)
        all_results.append(one)
        print(
            f"[{nenv}] CPU={one['cpu_mode']['total_ms']:.3f} ms, "
            f"MPS={one['mps_mode']['total_ms']:.3f} ms, "
            f"speedup_cpu_div_mps={one['speedup_cpu_div_mps']:.3f}"
        )

    output_json = Path(args.output_json)
    output_png = Path(args.output_png)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "device": DEVICE,
            "torch_version": torch.__version__,
            "mps_built": bool(torch.backends.mps.is_built()),
            "mps_available": bool(torch.backends.mps.is_available()),
            "env_list": env_list,
            "iters": args.iters,
            "note": "Measured directly. No projected model.",
        },
        "results": all_results,
    }
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    plot_results(all_results, output_png)
    print(f"Saved JSON: {output_json}")
    print(f"Saved PNG:  {output_png}")


if __name__ == "__main__":
    main()
    raise SystemExit(0)
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
