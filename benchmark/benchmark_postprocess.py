import argparse
import importlib
import json
import math
import pkgutil
import time
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import mlx.core as mx

from unilab.envs import registry

try:
    from mujoco import mlx_step as mj_mlx_step
except Exception:
    mj_mlx_step = None

def ensure_registries() -> None:
    """Import locomotion env modules so they are registered."""
try:
    from benchmark.core.device_info import get_device_info_dict, get_device_info_line
except ModuleNotFoundError:
    from core.device_info import get_device_info_dict, get_device_info_line

    try:
        import unilab.envs.locomotion

        package = unilab.envs.locomotion
        if hasattr(package, "__path__"):
            for _, name, _ in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
                try:
                    importlib.import_module(name)
                except Exception:
                    pass
    except ImportError:
        pass

DEFAULT_ENV_LIST = [256, 512, 1024, 2048, 4096]
DEFAULT_ITERS = 200
OUTPUT_DIR = Path("benchmark/outputs/postprocess")
OUTPUT_JSON = OUTPUT_DIR / "latest_postprocess_benchmark.json"
OUTPUT_PNG = OUTPUT_DIR / "latest_postprocess_latency.png"
TORCH_DEVICE = "mps"

def sync_torch_mps():
    if torch.backends.mps.is_available():
        torch.mps.synchronize()

def parse_env_list(raw: str) -> list[int]:
    if not raw:
        return DEFAULT_ENV_LIST
    return [int(x.strip()) for x in raw.split(",") if x.strip()]

def geomean(values: list[float]) -> float:
    vals = [v for v in values if v > 0.0]
    if not vals:
        return 0.0
    return float(math.exp(sum(math.log(v) for v in vals) / len(vals)))

def build_go1_layout() -> dict:
    env = registry.make("Go1JoystickFlatTerrain", num_envs=1, sim_backend="mujoco")
    layout = {
        "sensor_dim": int(env.model.nsensordata),
        "physics_dim": int(env.physics_state_dim),
        "num_action": int(env.action_space.shape[0]),
        "idx_linvel": env.idx_linvel,
        "idx_gyro": env.idx_gyro,
        "idx_global_linvel": env.idx_global_linvel,
        "idx_upvector": env.idx_upvector,
        "idx_qpos": int(env._idx_qpos),
        "idx_qvel": int(env._idx_qvel),
        "nq": int(env.nq),
        "nv": int(env.nv),
        "default_angles": np.asarray(env.default_angles, dtype=np.float32).copy(),
        "tracking_sigma": float(env.cfg.reward_config.tracking_sigma),
        "base_height_target": float(env.cfg.reward_config.base_height_target),
        "ctrl_dt": float(env.cfg.ctrl_dt),
        "reward_scales": dict(env.cfg.reward_config.scales),
        "obs_dim": int(env.observation_space.shape[0]),
        "command_low": np.asarray(env.cfg.commands.vel_limit[0], dtype=np.float32),
        "command_high": np.asarray(env.cfg.commands.vel_limit[1], dtype=np.float32),
    }
    env.close()
    return layout

def measure_physics_step_ms(num_envs: int, iters: int) -> float:
    env = registry.make("Go1JoystickFlatTerrain", num_envs=num_envs, sim_backend="mujoco")
    try:
        initial_state, _, _ = env.reset(np.arange(env.num_envs))
        action_low = env.action_space.low.astype(np.float32)
        action_high = env.action_space.high.astype(np.float32)
        actions = np.random.uniform(action_low, action_high, size=(env.num_envs, env.action_space.shape[0])).astype(
            np.float32
        )

        for _ in range(20):
            _ = env.step(actions)

        elapsed = 0.0
        for _ in range(iters):
            t0 = time.perf_counter()
            _ = env.step(actions)
            t1 = time.perf_counter()
            elapsed += t1 - t0
        return elapsed / iters * 1000.0
    finally:
        env.close()

def _unpack_rollout_out(step_out):
    state_mx, sensor_mx = step_out
    return state_mx[:, -1, :], sensor_mx[:, -1, :]

def measure_physics_step_mlx_native_ms(num_envs: int, iters: int) -> float:
    """Measure pure MuJoCo physics stepping via native mujoco.mlx_step."""
    if mj_mlx_step is None:
        raise RuntimeError("Native mujoco.mlx_step is unavailable.")
    env = registry.make("Go1JoystickFlatTerrain", num_envs=num_envs, sim_backend="mujoco")
    try:
        initial_state, _, _ = env.reset(np.arange(env.num_envs))
        action_low = env.action_space.low.astype(np.float32)
        action_high = env.action_space.high.astype(np.float32)
        actions_np = np.random.uniform(
            action_low,
            action_high,
            size=(env.num_envs, env.action_space.shape[0]),
        ).astype(np.float32)
        actions_mx = mx.array(actions_np, dtype=mx.float32)
        control_mx = mx.broadcast_to(
            actions_mx[:, None, :],
            (env.num_envs, env.cfg.sim_substeps, env.action_space.shape[0]),
        )
        model_batch = [env._model] * env.num_envs
        initial_state = mx.array(initial_state, dtype=mx.float32)
        with mj_mlx_step.MlxStepRunner(nthread=env._n_threads) as runner:
            for _ in range(20):
                step_out = runner.step(
                    model=model_batch,
                    data=env._worker_data,
                    initial_state=initial_state,
                    control=control_mx,
                    nstep=env.cfg.sim_substeps,
                    out_dtype=mx.float32,
                )
                last_state_mx, last_sensor_mx = _unpack_rollout_out(step_out)
                mx.eval(last_state_mx, last_sensor_mx)
                initial_state = last_state_mx

            elapsed = 0.0
            for _ in range(iters):
                t0 = time.perf_counter()
                step_out = runner.step(
                    model=model_batch,
                    data=env._worker_data,
                    initial_state=initial_state,
                    control=control_mx,
                    nstep=env.cfg.sim_substeps,
                    out_dtype=mx.float32,
                )
                last_state_mx, last_sensor_mx = _unpack_rollout_out(step_out)
                mx.eval(last_state_mx, last_sensor_mx)
                t1 = time.perf_counter()
                elapsed += t1 - t0
                initial_state = last_state_mx
        return elapsed / iters * 1000.0
    finally:
        env.close()

def measure_rollout_bridge_mlx_pipeline_ms(
    num_envs: int,
    iters: int,
    idx_mx: dict,
    scalars_mx: dict,
) -> dict:
    """End-to-end: MLX action -> MuJoCo rollout -> MLX sensordata/reward."""
    if mj_mlx_step is None:
        raise RuntimeError("Native mujoco.mlx_step is unavailable.")
    env = registry.make("Go1JoystickFlatTerrain", num_envs=num_envs, sim_backend="mujoco")
    try:
        initial_state, _, reset_info = env.reset(np.arange(env.num_envs))
        action_low = env.action_space.low.astype(np.float32)
        action_high = env.action_space.high.astype(np.float32)
        actions_np = np.random.uniform(
            action_low, action_high, size=(env.num_envs, env.action_space.shape[0])
        ).astype(np.float32)
        actions_mx = mx.array(actions_np, dtype=mx.float32)
        last_actions_mx = mx.zeros_like(actions_mx)
        commands_mx = mx.array(reset_info["commands"], dtype=mx.float32)

        # Zero-order hold across MuJoCo substeps.
        control_mx = mx.broadcast_to(actions_mx[:, None, :], (env.num_envs, env.cfg.sim_substeps, env.action_space.shape[0]))
        mx.eval(control_mx, commands_mx, last_actions_mx)
        model_batch = [env._model] * env.num_envs
        with mj_mlx_step.MlxStepRunner(nthread=env._n_threads) as runner:

            for _ in range(10):
                rollout_out = runner.step(
                    model=model_batch,
                    data=env._worker_data,
                    initial_state=initial_state,
                    control=control_mx,
                    nstep=env.cfg.sim_substeps,
                    out_dtype=mx.float32,
                )
                last_state_mx, last_sensor_mx = _unpack_rollout_out(rollout_out)
                obs_mx, rew_mx, done_mx = mlx_postprocess_go1(
                    sensor_mx=last_sensor_mx,
                    physics_mx=last_state_mx,
                    current_mx=actions_mx,
                    last_mx=last_actions_mx,
                    commands_mx=commands_mx,
                    idx_mx=idx_mx,
                    scalars=scalars_mx,
                )
                mx.eval(obs_mx, rew_mx, done_mx)
                initial_state = last_state_mx

            rollout_elapsed = 0.0
            post_elapsed = 0.0
            for _ in range(iters):
                t0 = time.perf_counter()
                rollout_out = runner.step(
                    model=model_batch,
                    data=env._worker_data,
                    initial_state=initial_state,
                    control=control_mx,
                    nstep=env.cfg.sim_substeps,
                    out_dtype=mx.float32,
                )
                t1 = time.perf_counter()
                last_state_mx, last_sensor_mx = _unpack_rollout_out(rollout_out)
                obs_mx, rew_mx, done_mx = mlx_postprocess_go1(
                    sensor_mx=last_sensor_mx,
                    physics_mx=last_state_mx,
                    current_mx=actions_mx,
                    last_mx=last_actions_mx,
                    commands_mx=commands_mx,
                    idx_mx=idx_mx,
                    scalars=scalars_mx,
                )
                mx.eval(obs_mx, rew_mx, done_mx)
                t2 = time.perf_counter()
                rollout_elapsed += t1 - t0
                post_elapsed += t2 - t1
                initial_state = last_state_mx

            rollout_ms = rollout_elapsed / iters * 1000.0
            post_ms = post_elapsed / iters * 1000.0
            return {
                "rollout_with_mlx_action_ms": rollout_ms,
                "mlx_postprocess_from_rollout_ms": post_ms,
                "total_ms": rollout_ms + post_ms,
            }
    finally:
        env.close()

def numpy_postprocess_go1(
    sensor_data: np.ndarray,
    physics_state: np.ndarray,
    current_actions: np.ndarray,
    last_actions: np.ndarray,
    commands: np.ndarray,
    layout: dict,
):
    idx_lin = layout["idx_linvel"]
    idx_gyro = layout["idx_gyro"]
    idx_glin = layout["idx_global_linvel"]
    idx_up = layout["idx_upvector"]
    idx_qpos = layout["idx_qpos"]
    idx_qvel = layout["idx_qvel"]
    nq = layout["nq"]
    nv = layout["nv"]
    default_angles = layout["default_angles"]
    scales = layout["reward_scales"]

    linear_vel = sensor_data[:, idx_lin]
    gyro = sensor_data[:, idx_gyro]
    global_linvel = sensor_data[:, idx_glin]
    upvector = sensor_data[:, idx_up]
    local_gravity = -upvector

    dof_pos = physics_state[:, idx_qpos + 7 : idx_qpos + nq]
    dof_vel = physics_state[:, idx_qvel + 6 : idx_qvel + nv]
    diff = dof_pos - default_angles

    obs = np.hstack(
        [
            linear_vel,
            gyro,
            local_gravity,
            diff,
            dof_vel,
            current_actions,
            commands,
        ]
    ).astype(np.float32, copy=False)

    tracking_sigma = layout["tracking_sigma"]
    tracking_lin_vel = np.exp(
        -np.sum(np.square(commands[:, :2] - linear_vel[:, :2]), axis=1) / tracking_sigma
    )
    tracking_ang_vel = np.exp(-np.square(commands[:, 2] - gyro[:, 2]) / tracking_sigma)
    lin_vel_z = np.square(global_linvel[:, 2])
    ang_vel_xy = np.sum(np.square(gyro[:, :2]), axis=1)
    base_height = np.square(physics_state[:, idx_qpos + 2] - layout["base_height_target"])
    action_rate = np.sum(np.square(current_actions - last_actions), axis=1)
    similar_to_default = np.sum(np.abs(diff), axis=1)

    reward = (
        scales.get("tracking_lin_vel", 0.0) * tracking_lin_vel
        + scales.get("tracking_ang_vel", 0.0) * tracking_ang_vel
        + scales.get("lin_vel_z", 0.0) * lin_vel_z
        + scales.get("ang_vel_xy", 0.0) * ang_vel_xy
        + scales.get("base_height", 0.0) * base_height
        + scales.get("action_rate", 0.0) * action_rate
        + scales.get("similar_to_default", 0.0) * similar_to_default
    )
    reward = (reward * layout["ctrl_dt"]).astype(np.float32, copy=False)
    done = (upvector[:, 2] <= 0.5).astype(np.bool_)
    return obs, reward, done

def torch_postprocess_go1(
    sensor_t: torch.Tensor,
    physics_t: torch.Tensor,
    current_t: torch.Tensor,
    last_t: torch.Tensor,
    commands_t: torch.Tensor,
    idx_t: dict,
    scalars_t: dict,
):
    linear_vel = sensor_t[:, idx_t["lin"]]
    gyro = sensor_t[:, idx_t["gyro"]]
    global_linvel = sensor_t[:, idx_t["glin"]]
    upvector = sensor_t[:, idx_t["up"]]
    local_gravity = -upvector

    dof_pos = physics_t[:, idx_t["qpos_start"] : idx_t["qpos_end"]]
    dof_vel = physics_t[:, idx_t["qvel_start"] : idx_t["qvel_end"]]
    diff = dof_pos - idx_t["default_angles"]

    obs = torch.hstack([linear_vel, gyro, local_gravity, diff, dof_vel, current_t, commands_t])

    tracking_lin_vel = torch.exp(
        -torch.sum((commands_t[:, :2] - linear_vel[:, :2]) ** 2, dim=1) / scalars_t["tracking_sigma"]
    )
    tracking_ang_vel = torch.exp(-((commands_t[:, 2] - gyro[:, 2]) ** 2) / scalars_t["tracking_sigma"])
    lin_vel_z = global_linvel[:, 2] ** 2
    ang_vel_xy = torch.sum(gyro[:, :2] ** 2, dim=1)
    base_height = (physics_t[:, idx_t["base_height_idx"]] - scalars_t["base_height_target"]) ** 2
    action_rate = torch.sum((current_t - last_t) ** 2, dim=1)
    similar_to_default = torch.sum(torch.abs(diff), dim=1)

    reward = (
        scalars_t["tracking_lin_vel"] * tracking_lin_vel
        + scalars_t["tracking_ang_vel"] * tracking_ang_vel
        + scalars_t["lin_vel_z"] * lin_vel_z
        + scalars_t["ang_vel_xy"] * ang_vel_xy
        + scalars_t["base_height"] * base_height
        + scalars_t["action_rate"] * action_rate
        + scalars_t["similar_to_default"] * similar_to_default
    ) * scalars_t["ctrl_dt"]

    done = upvector[:, 2] <= 0.5
    return obs, reward, done

def mlx_postprocess_go1(
    sensor_mx: mx.array,
    physics_mx: mx.array,
    current_mx: mx.array,
    last_mx: mx.array,
    commands_mx: mx.array,
    idx_mx: dict,
    scalars: dict,
):
    linear_vel = sensor_mx[:, idx_mx["lin"]]
    gyro = sensor_mx[:, idx_mx["gyro"]]
    global_linvel = sensor_mx[:, idx_mx["glin"]]
    upvector = sensor_mx[:, idx_mx["up"]]
    local_gravity = -upvector

    dof_pos = physics_mx[:, idx_mx["qpos_start"] : idx_mx["qpos_end"]]
    dof_vel = physics_mx[:, idx_mx["qvel_start"] : idx_mx["qvel_end"]]
    diff = dof_pos - idx_mx["default_angles"]

    obs = mx.concatenate([linear_vel, gyro, local_gravity, diff, dof_vel, current_mx, commands_mx], axis=1)

    tracking_lin_vel = mx.exp(
        -mx.sum((commands_mx[:, :2] - linear_vel[:, :2]) ** 2, axis=1) / scalars["tracking_sigma"]
    )
    tracking_ang_vel = mx.exp(-((commands_mx[:, 2] - gyro[:, 2]) ** 2) / scalars["tracking_sigma"])
    lin_vel_z = global_linvel[:, 2] ** 2
    ang_vel_xy = mx.sum(gyro[:, :2] ** 2, axis=1)
    base_height = (physics_mx[:, idx_mx["base_height_idx"]] - scalars["base_height_target"]) ** 2
    action_rate = mx.sum((current_mx - last_mx) ** 2, axis=1)
    similar_to_default = mx.sum(mx.abs(diff), axis=1)

    reward = (
        scalars["tracking_lin_vel"] * tracking_lin_vel
        + scalars["tracking_ang_vel"] * tracking_ang_vel
        + scalars["lin_vel_z"] * lin_vel_z
        + scalars["ang_vel_xy"] * ang_vel_xy
        + scalars["base_height"] * base_height
        + scalars["action_rate"] * action_rate
        + scalars["similar_to_default"] * similar_to_default
    ) * scalars["ctrl_dt"]
    done = upvector[:, 2] <= 0.5
    return obs, reward, done

def bench_one_envnum(num_envs: int, iters: int, layout: dict):
    try:
        physics_step_env_wrapper_ms = measure_physics_step_ms(num_envs=num_envs, iters=iters)
    except Exception:
        physics_step_env_wrapper_ms = 0.0
    physics_step_mlx_native_ms = measure_physics_step_mlx_native_ms(num_envs=num_envs, iters=iters)

    sensor_np = np.random.randn(num_envs, layout["sensor_dim"]).astype(np.float32)
    physics_np = np.random.randn(num_envs, layout["physics_dim"]).astype(np.float32)
    current_actions_np = np.random.uniform(-1.0, 1.0, size=(num_envs, layout["num_action"])).astype(np.float32)
    last_actions_np = np.random.uniform(-1.0, 1.0, size=(num_envs, layout["num_action"])).astype(np.float32)
    commands_np = np.random.uniform(
        low=layout["command_low"],
        high=layout["command_high"],
        size=(num_envs, 3),
    ).astype(np.float32)

    idx_t = {
        "lin": torch.as_tensor(layout["idx_linvel"], device=TORCH_DEVICE, dtype=torch.long),
        "gyro": torch.as_tensor(layout["idx_gyro"], device=TORCH_DEVICE, dtype=torch.long),
        "glin": torch.as_tensor(layout["idx_global_linvel"], device=TORCH_DEVICE, dtype=torch.long),
        "up": torch.as_tensor(layout["idx_upvector"], device=TORCH_DEVICE, dtype=torch.long),
        "qpos_start": int(layout["idx_qpos"] + 7),
        "qpos_end": int(layout["idx_qpos"] + layout["nq"]),
        "qvel_start": int(layout["idx_qvel"] + 6),
        "qvel_end": int(layout["idx_qvel"] + layout["nv"]),
        "base_height_idx": int(layout["idx_qpos"] + 2),
        "default_angles": torch.as_tensor(layout["default_angles"], device=TORCH_DEVICE, dtype=torch.float32),
    }
    scalars_t = {
        "tracking_sigma": torch.tensor(layout["tracking_sigma"], device=TORCH_DEVICE, dtype=torch.float32),
        "base_height_target": torch.tensor(layout["base_height_target"], device=TORCH_DEVICE, dtype=torch.float32),
        "ctrl_dt": torch.tensor(layout["ctrl_dt"], device=TORCH_DEVICE, dtype=torch.float32),
        "tracking_lin_vel": torch.tensor(layout["reward_scales"].get("tracking_lin_vel", 0.0), device=TORCH_DEVICE),
        "tracking_ang_vel": torch.tensor(layout["reward_scales"].get("tracking_ang_vel", 0.0), device=TORCH_DEVICE),
        "lin_vel_z": torch.tensor(layout["reward_scales"].get("lin_vel_z", 0.0), device=TORCH_DEVICE),
        "ang_vel_xy": torch.tensor(layout["reward_scales"].get("ang_vel_xy", 0.0), device=TORCH_DEVICE),
        "base_height": torch.tensor(layout["reward_scales"].get("base_height", 0.0), device=TORCH_DEVICE),
        "action_rate": torch.tensor(layout["reward_scales"].get("action_rate", 0.0), device=TORCH_DEVICE),
        "similar_to_default": torch.tensor(
            layout["reward_scales"].get("similar_to_default", 0.0),
            device=TORCH_DEVICE,
        ),
    }

    idx_mx = {
        "lin": mx.array(layout["idx_linvel"], dtype=mx.int32),
        "gyro": mx.array(layout["idx_gyro"], dtype=mx.int32),
        "glin": mx.array(layout["idx_global_linvel"], dtype=mx.int32),
        "up": mx.array(layout["idx_upvector"], dtype=mx.int32),
        "qpos_start": int(layout["idx_qpos"] + 7),
        "qpos_end": int(layout["idx_qpos"] + layout["nq"]),
        "qvel_start": int(layout["idx_qvel"] + 6),
        "qvel_end": int(layout["idx_qvel"] + layout["nv"]),
        "base_height_idx": int(layout["idx_qpos"] + 2),
        "default_angles": mx.array(layout["default_angles"], dtype=mx.float32),
    }
    scalars_mx = {
        "tracking_sigma": mx.array(layout["tracking_sigma"], dtype=mx.float32),
        "base_height_target": mx.array(layout["base_height_target"], dtype=mx.float32),
        "ctrl_dt": mx.array(layout["ctrl_dt"], dtype=mx.float32),
        "tracking_lin_vel": mx.array(layout["reward_scales"].get("tracking_lin_vel", 0.0), dtype=mx.float32),
        "tracking_ang_vel": mx.array(layout["reward_scales"].get("tracking_ang_vel", 0.0), dtype=mx.float32),
        "lin_vel_z": mx.array(layout["reward_scales"].get("lin_vel_z", 0.0), dtype=mx.float32),
        "ang_vel_xy": mx.array(layout["reward_scales"].get("ang_vel_xy", 0.0), dtype=mx.float32),
        "base_height": mx.array(layout["reward_scales"].get("base_height", 0.0), dtype=mx.float32),
        "action_rate": mx.array(layout["reward_scales"].get("action_rate", 0.0), dtype=mx.float32),
        "similar_to_default": mx.array(layout["reward_scales"].get("similar_to_default", 0.0), dtype=mx.float32),
    }
    try:
        bridge_mlx = measure_rollout_bridge_mlx_pipeline_ms(
            num_envs=num_envs,
            iters=iters,
            idx_mx=idx_mx,
            scalars_mx=scalars_mx,
        )
    except Exception:
        bridge_mlx = {
            "rollout_with_mlx_action_ms": 0.0,
            "mlx_postprocess_from_rollout_ms": 0.0,
            "total_ms": 0.0,
        }

    # Warmup
    for _ in range(20):
        obs_np, rew_np, done_np = numpy_postprocess_go1(
            sensor_np, physics_np, current_actions_np, last_actions_np, commands_np, layout
        )
        _ = torch.as_tensor(obs_np, device=TORCH_DEVICE, dtype=torch.float32)
        _ = torch.as_tensor(rew_np, device=TORCH_DEVICE, dtype=torch.float32)
        _ = torch.as_tensor(done_np, device=TORCH_DEVICE, dtype=torch.bool)
    for _ in range(20):
        sensor_t = torch.as_tensor(sensor_np, device=TORCH_DEVICE, dtype=torch.float32)
        physics_t = torch.as_tensor(physics_np, device=TORCH_DEVICE, dtype=torch.float32)
        cur_t = torch.as_tensor(current_actions_np, device=TORCH_DEVICE, dtype=torch.float32)
        last_t = torch.as_tensor(last_actions_np, device=TORCH_DEVICE, dtype=torch.float32)
        cmd_t = torch.as_tensor(commands_np, device=TORCH_DEVICE, dtype=torch.float32)
        _ = torch_postprocess_go1(sensor_t, physics_t, cur_t, last_t, cmd_t, idx_t, scalars_t)
    for _ in range(20):
        sensor_mx = mx.array(sensor_np, dtype=mx.float32)
        physics_mx = mx.array(physics_np, dtype=mx.float32)
        cur_mx = mx.array(current_actions_np, dtype=mx.float32)
        last_mx = mx.array(last_actions_np, dtype=mx.float32)
        cmd_mx = mx.array(commands_np, dtype=mx.float32)
        _ = mlx_postprocess_go1(sensor_mx, physics_mx, cur_mx, last_mx, cmd_mx, idx_mx, scalars_mx)
    sync_torch_mps()

    cpu_compute = 0.0
    cpu_transfer = 0.0
    cpu_to_mlx_compute = 0.0
    cpu_to_mlx_transfer = 0.0
    torch_transfer = 0.0
    torch_compute = 0.0
    mlx_transfer = 0.0
    mlx_compute = 0.0
    mlx_torch_transfer = 0.0
    mlx_torch_compute = 0.0
    mlx_to_torch_transfer = 0.0

    for _ in range(iters):
        t0 = time.perf_counter()
        obs_np, rew_np, done_np = numpy_postprocess_go1(
            sensor_np, physics_np, current_actions_np, last_actions_np, commands_np, layout
        )
        t1 = time.perf_counter()
        _ = torch.as_tensor(obs_np, device=TORCH_DEVICE, dtype=torch.float32)
        _ = torch.as_tensor(rew_np, device=TORCH_DEVICE, dtype=torch.float32)
        _ = torch.as_tensor(done_np, device=TORCH_DEVICE, dtype=torch.bool)
        sync_torch_mps()
        t2 = time.perf_counter()
        cpu_compute += t1 - t0
        cpu_transfer += t2 - t1

    for _ in range(iters):
        t0 = time.perf_counter()
        obs_np, rew_np, done_np = numpy_postprocess_go1(
            sensor_np, physics_np, current_actions_np, last_actions_np, commands_np, layout
        )
        t1 = time.perf_counter()
        obs_mx = mx.array(obs_np, dtype=mx.float32)
        rew_mx = mx.array(rew_np, dtype=mx.float32)
        done_mx = mx.array(done_np, dtype=mx.bool_)
        mx.eval(obs_mx, rew_mx, done_mx)
        t2 = time.perf_counter()
        cpu_to_mlx_compute += t1 - t0
        cpu_to_mlx_transfer += t2 - t1

    for _ in range(iters):
        t0 = time.perf_counter()
        sensor_t = torch.as_tensor(sensor_np, device=TORCH_DEVICE, dtype=torch.float32)
        physics_t = torch.as_tensor(physics_np, device=TORCH_DEVICE, dtype=torch.float32)
        cur_t = torch.as_tensor(current_actions_np, device=TORCH_DEVICE, dtype=torch.float32)
        last_t = torch.as_tensor(last_actions_np, device=TORCH_DEVICE, dtype=torch.float32)
        cmd_t = torch.as_tensor(commands_np, device=TORCH_DEVICE, dtype=torch.float32)
        sync_torch_mps()
        t1 = time.perf_counter()
        obs_t, rew_t, done_t = torch_postprocess_go1(sensor_t, physics_t, cur_t, last_t, cmd_t, idx_t, scalars_t)
        _ = obs_t, rew_t, done_t
        sync_torch_mps()
        t2 = time.perf_counter()
        torch_transfer += t1 - t0
        torch_compute += t2 - t1

    for _ in range(iters):
        t0 = time.perf_counter()
        sensor_mx = mx.array(sensor_np, dtype=mx.float32)
        physics_mx = mx.array(physics_np, dtype=mx.float32)
        cur_mx = mx.array(current_actions_np, dtype=mx.float32)
        last_mx = mx.array(last_actions_np, dtype=mx.float32)
        cmd_mx = mx.array(commands_np, dtype=mx.float32)
        mx.eval(sensor_mx, physics_mx, cur_mx, last_mx, cmd_mx)
        t1 = time.perf_counter()
        obs_mx, rew_mx, done_mx = mlx_postprocess_go1(
            sensor_mx, physics_mx, cur_mx, last_mx, cmd_mx, idx_mx, scalars_mx
        )
        mx.eval(obs_mx, rew_mx, done_mx)
        t2 = time.perf_counter()
        mlx_transfer += t1 - t0
        mlx_compute += t2 - t1

    for _ in range(iters):
        t0 = time.perf_counter()
        sensor_mx = mx.array(sensor_np, dtype=mx.float32)
        physics_mx = mx.array(physics_np, dtype=mx.float32)
        cur_mx = mx.array(current_actions_np, dtype=mx.float32)
        last_mx = mx.array(last_actions_np, dtype=mx.float32)
        cmd_mx = mx.array(commands_np, dtype=mx.float32)
        mx.eval(sensor_mx, physics_mx, cur_mx, last_mx, cmd_mx)
        t1 = time.perf_counter()
        obs_mx, rew_mx, done_mx = mlx_postprocess_go1(
            sensor_mx, physics_mx, cur_mx, last_mx, cmd_mx, idx_mx, scalars_mx
        )
        mx.eval(obs_mx, rew_mx, done_mx)
        t2 = time.perf_counter()
        _ = torch.as_tensor(np.array(obs_mx), device=TORCH_DEVICE, dtype=torch.float32)
        _ = torch.as_tensor(np.array(rew_mx), device=TORCH_DEVICE, dtype=torch.float32)
        _ = torch.as_tensor(np.array(done_mx), device=TORCH_DEVICE, dtype=torch.bool)
        sync_torch_mps()
        t3 = time.perf_counter()
        mlx_torch_transfer += t1 - t0
        mlx_torch_compute += t2 - t1
        mlx_to_torch_transfer += t3 - t2

    cpu_compute_ms = cpu_compute / iters * 1000.0
    cpu_transfer_ms = cpu_transfer / iters * 1000.0
    cpu_to_mlx_compute_ms = cpu_to_mlx_compute / iters * 1000.0
    cpu_to_mlx_transfer_ms = cpu_to_mlx_transfer / iters * 1000.0
    torch_transfer_ms = torch_transfer / iters * 1000.0
    torch_compute_ms = torch_compute / iters * 1000.0
    mlx_transfer_ms = mlx_transfer / iters * 1000.0
    mlx_compute_ms = mlx_compute / iters * 1000.0
    mlx_torch_transfer_ms = mlx_torch_transfer / iters * 1000.0
    mlx_torch_compute_ms = mlx_torch_compute / iters * 1000.0
    mlx_to_torch_transfer_ms = mlx_to_torch_transfer / iters * 1000.0

    cpu_total_ms = cpu_compute_ms + cpu_transfer_ms
    cpu_to_mlx_total_ms = cpu_to_mlx_compute_ms + cpu_to_mlx_transfer_ms
    torch_total_ms = torch_transfer_ms + torch_compute_ms
    mlx_total_ms = mlx_transfer_ms + mlx_compute_ms
    mlx_to_torch_total_ms = mlx_torch_transfer_ms + mlx_torch_compute_ms + mlx_to_torch_transfer_ms

    return {
        "num_envs": num_envs,
        "iterations": iters,
        "layout": {
            "obs_dim": layout["obs_dim"],
            "sensor_dim": layout["sensor_dim"],
            "physics_dim": layout["physics_dim"],
            "num_action": layout["num_action"],
        },
        "physics_step_mode": {
            "physics_step_mlx_native_ms": physics_step_mlx_native_ms,
            "physics_step_env_wrapper_ms": physics_step_env_wrapper_ms,
        },
        "mlx_rollout_bridge_mode": {
            "rollout_with_mlx_action_ms": bridge_mlx["rollout_with_mlx_action_ms"],
            "mlx_postprocess_from_rollout_ms": bridge_mlx["mlx_postprocess_from_rollout_ms"],
            "total_ms": bridge_mlx["total_ms"],
        },
        "cpu_numpy_mode": {
            "compute_numpy_ms": cpu_compute_ms,
            "transfer_obs_rew_done_to_torch_mps_ms": cpu_transfer_ms,
            "total_ms": cpu_total_ms,
            "total_with_physics_ms": cpu_total_ms + physics_step_mlx_native_ms,
        },
        "cpu_numpy_to_mlx_mode": {
            "compute_numpy_ms": cpu_to_mlx_compute_ms,
            "transfer_obs_rew_done_to_mlx_ms": cpu_to_mlx_transfer_ms,
            "total_ms": cpu_to_mlx_total_ms,
            "total_with_physics_ms": cpu_to_mlx_total_ms + physics_step_mlx_native_ms,
        },
        "torch_mps_mode": {
            "transfer_all_numpy_to_torch_mps_ms": torch_transfer_ms,
            "compute_postprocess_on_torch_mps_ms": torch_compute_ms,
            "total_ms": torch_total_ms,
            "total_with_physics_ms": torch_total_ms + physics_step_mlx_native_ms,
        },
        "mlx_mode": {
            "transfer_all_numpy_to_mlx_ms": mlx_transfer_ms,
            "compute_postprocess_on_mlx_ms": mlx_compute_ms,
            "total_ms": mlx_total_ms,
            "total_with_physics_ms": mlx_total_ms + physics_step_mlx_native_ms,
        },
        "mlx_to_torch_mps_mode": {
            "transfer_all_numpy_to_mlx_ms": mlx_torch_transfer_ms,
            "compute_postprocess_on_mlx_ms": mlx_torch_compute_ms,
            "transfer_postprocess_result_mlx_to_torch_mps_ms": mlx_to_torch_transfer_ms,
            "total_ms": mlx_to_torch_total_ms,
            "total_with_physics_ms": mlx_to_torch_total_ms + physics_step_mlx_native_ms,
        },
        "speedup_cpu_div_torch_mps": cpu_total_ms / torch_total_ms if torch_total_ms > 0 else 0.0,
        "speedup_cpu_div_mlx": cpu_total_ms / mlx_total_ms if mlx_total_ms > 0 else 0.0,
        "speedup_cpu_to_mlx_div_mlx": cpu_to_mlx_total_ms / mlx_total_ms if mlx_total_ms > 0 else 0.0,
        "speedup_torch_mps_div_mlx": torch_total_ms / mlx_total_ms if mlx_total_ms > 0 else 0.0,
        "speedup_cpu_div_mlx_to_torch_mps": cpu_total_ms / mlx_to_torch_total_ms if mlx_to_torch_total_ms > 0 else 0.0,
        "speedup_torch_mps_div_mlx_to_torch_mps": torch_total_ms / mlx_to_torch_total_ms
        if mlx_to_torch_total_ms > 0
        else 0.0,
    }

def plot_results(results: list[dict], output_png: Path):
    envs = [r["num_envs"] for r in results]
    x = np.arange(len(envs), dtype=float)
    w = 0.15

    cpu_compute = np.array([r["cpu_numpy_mode"]["compute_numpy_ms"] for r in results])
    cpu_transfer = np.array([r["cpu_numpy_mode"]["transfer_obs_rew_done_to_torch_mps_ms"] for r in results])
    cpu_to_mlx_compute = np.array([r["cpu_numpy_to_mlx_mode"]["compute_numpy_ms"] for r in results])
    cpu_to_mlx_transfer = np.array([r["cpu_numpy_to_mlx_mode"]["transfer_obs_rew_done_to_mlx_ms"] for r in results])
    torch_transfer = np.array([r["torch_mps_mode"]["transfer_all_numpy_to_torch_mps_ms"] for r in results])
    torch_compute = np.array([r["torch_mps_mode"]["compute_postprocess_on_torch_mps_ms"] for r in results])
    mlx_transfer = np.array([r["mlx_mode"]["transfer_all_numpy_to_mlx_ms"] for r in results])
    mlx_compute = np.array([r["mlx_mode"]["compute_postprocess_on_mlx_ms"] for r in results])
    mlx_to_torch_transfer = np.array(
        [r["mlx_to_torch_mps_mode"]["transfer_postprocess_result_mlx_to_torch_mps_ms"] for r in results]
    )
    physics_step = np.array([r["physics_step_mode"]["physics_step_mlx_native_ms"] for r in results])

    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(13, 10), sharex=True)
    x_cpu = x - 2.0 * w
    x_cpu_mlx = x - 1.0 * w
    x_torch = x
    x_mlx = x + 1.0 * w
    x_mlx_torch = x + 2.0 * w

    ax_top.bar(x_cpu, physics_step, w, label="Shared: MuJoCo physics step", color="#9AA0A6")
    ax_top.bar(x_cpu_mlx, physics_step, w, label="_nolegend_", color="#9AA0A6")
    ax_top.bar(x_torch, physics_step, w, label="_nolegend_", color="#9AA0A6")
    ax_top.bar(x_mlx, physics_step, w, label="_nolegend_", color="#9AA0A6")
    ax_top.bar(x_mlx_torch, physics_step, w, label="_nolegend_", color="#9AA0A6")

    ax_top.bar(x_cpu, cpu_compute, w, bottom=physics_step, label="Path A (CPU->Torch): numpy compute", color="#D99A9A")
    ax_top.bar(
        x_cpu,
        cpu_transfer,
        w,
        bottom=physics_step + cpu_compute,
        label="Path A (CPU->Torch): obs/rew/done -> torch.mps",
        color="#EBCED6",
    )
    ax_top.bar(
        x_cpu_mlx,
        cpu_to_mlx_compute,
        w,
        bottom=physics_step,
        label="Path B (CPU->MLX): numpy compute",
        color="#D7B78B",
    )
    ax_top.bar(
        x_cpu_mlx,
        cpu_to_mlx_transfer,
        w,
        bottom=physics_step + cpu_to_mlx_compute,
        label="Path B (CPU->MLX): obs/rew/done -> mlx",
        color="#EFD9B7",
    )
    ax_top.bar(
        x_torch,
        torch_transfer,
        w,
        bottom=physics_step,
        label="Path C (Torch native): numpy -> torch.mps",
        color="#8FB3CC",
    )
    ax_top.bar(
        x_torch,
        torch_compute,
        w,
        bottom=physics_step + torch_transfer,
        label="Path C (Torch native): torch.mps compute",
        color="#A8CFAE",
    )
    ax_top.bar(x_mlx, mlx_transfer, w, bottom=physics_step, label="Path D (MLX native): numpy -> mlx", color="#E1B15A")
    ax_top.bar(
        x_mlx,
        mlx_compute,
        w,
        bottom=physics_step + mlx_transfer,
        label="Path D (MLX native): mlx compute",
        color="#F3D9A5",
    )
    ax_top.bar(
        x_mlx_torch,
        mlx_transfer,
        w,
        bottom=physics_step,
        label="Path E (MLX->Torch): numpy -> mlx",
        color="#B78BD0",
    )
    ax_top.bar(
        x_mlx_torch,
        mlx_compute,
        w,
        bottom=physics_step + mlx_transfer,
        label="Path E (MLX->Torch): mlx compute",
        color="#CDA9E4",
    )
    ax_top.bar(
        x_mlx_torch,
        mlx_to_torch_transfer,
        w,
        bottom=physics_step + mlx_transfer + mlx_compute,
        label="Path E (MLX->Torch): mlx result -> torch.mps",
        color="#E7D0F4",
    )

    ax_bottom.bar(x_cpu, cpu_compute, w, label="Path A (CPU->Torch): numpy compute", color="#D99A9A")
    ax_bottom.bar(
        x_cpu,
        cpu_transfer,
        w,
        bottom=cpu_compute,
        label="Path A (CPU->Torch): obs/rew/done -> torch.mps",
        color="#EBCED6",
    )
    ax_bottom.bar(x_cpu_mlx, cpu_to_mlx_compute, w, label="Path B (CPU->MLX): numpy compute", color="#D7B78B")
    ax_bottom.bar(
        x_cpu_mlx,
        cpu_to_mlx_transfer,
        w,
        bottom=cpu_to_mlx_compute,
        label="Path B (CPU->MLX): obs/rew/done -> mlx",
        color="#EFD9B7",
    )
    ax_bottom.bar(x_torch, torch_transfer, w, label="Path C (Torch native): numpy -> torch.mps", color="#8FB3CC")
    ax_bottom.bar(
        x_torch,
        torch_compute,
        w,
        bottom=torch_transfer,
        label="Path C (Torch native): torch.mps compute",
        color="#A8CFAE",
    )
    ax_bottom.bar(x_mlx, mlx_transfer, w, label="Path D (MLX native): numpy -> mlx", color="#E1B15A")
    ax_bottom.bar(
        x_mlx,
        mlx_compute,
        w,
        bottom=mlx_transfer,
        label="Path D (MLX native): mlx compute",
        color="#F3D9A5",
    )
    ax_bottom.bar(x_mlx_torch, mlx_transfer, w, label="Path E (MLX->Torch): numpy -> mlx", color="#B78BD0")
    ax_bottom.bar(
        x_mlx_torch,
        mlx_compute,
        w,
        bottom=mlx_transfer,
        label="Path E (MLX->Torch): mlx compute",
        color="#CDA9E4",
    )
    ax_bottom.bar(
        x_mlx_torch,
        mlx_to_torch_transfer,
        w,
        bottom=mlx_transfer + mlx_compute,
        label="Path E (MLX->Torch): mlx result -> torch.mps",
        color="#E7D0F4",
    )

    cpu_total = physics_step + cpu_compute + cpu_transfer
    cpu_to_mlx_total = physics_step + cpu_to_mlx_compute + cpu_to_mlx_transfer
    torch_total = physics_step + torch_transfer + torch_compute
    mlx_total = physics_step + mlx_transfer + mlx_compute
    mlx_to_torch_total = physics_step + mlx_transfer + mlx_compute + mlx_to_torch_transfer
    cpu_post_total = cpu_compute + cpu_transfer
    cpu_to_mlx_post_total = cpu_to_mlx_compute + cpu_to_mlx_transfer
    torch_post_total = torch_transfer + torch_compute
    mlx_post_total = mlx_transfer + mlx_compute
    mlx_to_torch_post_total = mlx_transfer + mlx_compute + mlx_to_torch_transfer

    for i in range(len(envs)):
        ax_top.text(x_cpu[i], cpu_total[i] + 0.03, f"{cpu_total[i]:.2f}", ha="center", va="bottom", fontsize=8)
        ax_top.text(
            x_cpu_mlx[i],
            cpu_to_mlx_total[i] + 0.03,
            f"{cpu_to_mlx_total[i]:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
        ax_top.text(x_torch[i], torch_total[i] + 0.03, f"{torch_total[i]:.2f}", ha="center", va="bottom", fontsize=8)
        ax_top.text(x_mlx[i], mlx_total[i] + 0.03, f"{mlx_total[i]:.2f}", ha="center", va="bottom", fontsize=8)
        ax_top.text(
            x_mlx_torch[i],
            mlx_to_torch_total[i] + 0.03,
            f"{mlx_to_torch_total[i]:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

        ax_bottom.text(x_cpu[i], cpu_post_total[i] + 0.03, f"{cpu_post_total[i]:.2f}", ha="center", va="bottom", fontsize=8)
        ax_bottom.text(
            x_cpu_mlx[i],
            cpu_to_mlx_post_total[i] + 0.03,
            f"{cpu_to_mlx_post_total[i]:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
        ax_bottom.text(x_torch[i], torch_post_total[i] + 0.03, f"{torch_post_total[i]:.2f}", ha="center", va="bottom", fontsize=8)
        ax_bottom.text(x_mlx[i], mlx_post_total[i] + 0.03, f"{mlx_post_total[i]:.2f}", ha="center", va="bottom", fontsize=8)
        ax_bottom.text(
            x_mlx_torch[i],
            mlx_to_torch_post_total[i] + 0.03,
            f"{mlx_to_torch_post_total[i]:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax_top.set_title(
        "Go1 latest step+postprocess (Top: with shared physics; Bottom: postprocess only)\n"
        f"{get_device_info_line()}"
    )
    ax_top.set_ylabel("Time per step (ms)")
    ax_top.grid(axis="y", alpha=0.25)
    ax_top.legend(
        title="Execution paths",
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0.0,
        fontsize=8,
        title_fontsize=9,
    )

    ax_bottom.set_xlabel("num_envs")
    ax_bottom.set_ylabel("Postprocess time (ms)")
    ax_bottom.set_xticks(x)
    ax_bottom.set_xticklabels([str(v) for v in envs])
    ax_bottom.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=180)
    plt.close(fig)

def main():
    ensure_registries()
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_list", type=str, default=",".join(str(v) for v in DEFAULT_ENV_LIST))
    parser.add_argument("--iters", type=int, default=DEFAULT_ITERS)
    parser.add_argument("--output_json", type=str, default=str(OUTPUT_JSON))
    parser.add_argument("--output_png", type=str, default=str(OUTPUT_PNG))
    args = parser.parse_args()

    if not torch.backends.mps.is_available():
        raise RuntimeError("Torch MPS is not available on this machine.")

    env_list = parse_env_list(args.env_list)
    layout = build_go1_layout()
    all_results = []
    for nenv in env_list:
        one = bench_one_envnum(nenv, args.iters, layout)
        all_results.append(one)
        print(
            f"[{nenv}] Physics(mlx_native)={one['physics_step_mode']['physics_step_mlx_native_ms']:.3f} ms, "
            f"Physics(env_wrapper)={one['physics_step_mode']['physics_step_env_wrapper_ms']:.3f} ms, "
            f"CPU={one['cpu_numpy_mode']['total_with_physics_ms']:.3f} ms, "
            f"CPU->MLX={one['cpu_numpy_to_mlx_mode']['total_with_physics_ms']:.3f} ms, "
            f"Torch.MPS={one['torch_mps_mode']['total_with_physics_ms']:.3f} ms, "
            f"MLX={one['mlx_mode']['total_with_physics_ms']:.3f} ms, "
            f"MLX->Torch.MPS={one['mlx_to_torch_mps_mode']['total_with_physics_ms']:.3f} ms, "
            f"MLXBridge(rollout+post)={one['mlx_rollout_bridge_mode']['total_ms']:.3f} ms, "
            f"cpu/torch={one['speedup_cpu_div_torch_mps']:.3f}, "
            f"cpu/mlx={one['speedup_cpu_div_mlx']:.3f}, "
            f"cpu->mlx/mlx={one['speedup_cpu_to_mlx_div_mlx']:.3f}, "
            f"torch/mlx={one['speedup_torch_mps_div_mlx']:.3f}, "
            f"cpu/mlx->torch={one['speedup_cpu_div_mlx_to_torch_mps']:.3f}, "
            f"torch/mlx->torch={one['speedup_torch_mps_div_mlx_to_torch_mps']:.3f}"
        )

    output_json = Path(args.output_json)
    output_png = Path(args.output_png)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "device_info": get_device_info_dict(),
            "device_torch": TORCH_DEVICE,
            "device_mlx": "mlx",
            "torch_version": torch.__version__,
            "mps_built": bool(torch.backends.mps.is_built()),
            "mps_available": bool(torch.backends.mps.is_available()),
            "env_list": env_list,
            "iters": args.iters,
            "task": "Go1JoystickFlatTerrain",
            "note": "Postprocess logic synchronized with latest Go1JoystickFlatTerrain; physics step uses native mujoco.mlx_step.",
        },
        "summary": {
            "geomean_speedup_cpu_over_mlx_postprocess": geomean(
                [r["speedup_cpu_div_mlx"] for r in all_results]
            ),
            "geomean_speedup_cpu_to_mlx_over_mlx_postprocess": geomean(
                [r["speedup_cpu_to_mlx_div_mlx"] for r in all_results]
            ),
            "geomean_speedup_torch_over_mlx_postprocess": geomean(
                [r["speedup_torch_mps_div_mlx"] for r in all_results]
            ),
            "geomean_env_wrapper_over_mlx_native_physics": geomean(
                [
                    r["physics_step_mode"]["physics_step_env_wrapper_ms"]
                    / max(r["physics_step_mode"]["physics_step_mlx_native_ms"], 1e-12)
                    for r in all_results
                ]
            ),
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
