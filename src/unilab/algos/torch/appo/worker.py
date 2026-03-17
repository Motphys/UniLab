"""APPO Rollout Worker — runs in a subprocess.

Collects on-policy rollouts and writes to SharedOnPolicyStorage.
"""

from __future__ import annotations

import statistics
import sys
import time
from collections import defaultdict
from typing import Any, Dict

import numpy as np
import torch
from rsl_rl.utils import resolve_callable

from unilab.utils.algo_utils import ensure_registries
from unilab.utils.obs_utils import split_obs_dict


def appo_collector_fn(
    stop_event: Any,
    env_name: str,
    rl_cfg: dict,
    num_envs: int,
    steps_per_env: int,
    shm_storage_name: Dict[str, str],
    sync_primitives: tuple,
    obs_dim: int,
    action_dim: int,
    privileged_dim: int,
    weight_sync_name: str,
    weight_param_shapes: dict,
    metrics_queue: Any,
    collector_device: str = "cpu",
    sim_backend: str = "mujoco",
    env_cfg_override: dict | None = None,
):
    """Entry point for the APPO collector subprocess.

    Creates environment + policy, collects rollouts, writes to SharedOnPolicyStorage.
    """
    from copy import deepcopy

    from tensordict import TensorDict

    from unilab.base import registry
    from unilab.ipc import SharedOnPolicyStorage, SharedWeightSync
    from unilab.utils.rsl_rl_compat import convert_config_v3_to_v4, is_rsl_rl_v4, is_rsl_rl_v5

    ensure_registries()

    # Connect to shared memory
    storage = SharedOnPolicyStorage(
        num_envs=num_envs,
        num_steps=steps_per_env,
        obs_dim=obs_dim,
        action_dim=action_dim,
        privileged_dim=privileged_dim,
        create=False,
        shm_name_prefix=shm_storage_name,
    )
    storage.attach_sync_primitives(*sync_primitives)  # (write_ptr, read_ptr)
    weight_sync = SharedWeightSync(weight_param_shapes, create=False, shm_name=weight_sync_name)

    # Create environment
    env: Any = registry.make(
        env_name, num_envs=num_envs, sim_backend=sim_backend, env_cfg_override=env_cfg_override
    )

    # Build actor (stochastic MLPModel — mirrors runner._build_learner)
    cfg = dict(rl_cfg)
    if is_rsl_rl_v5():
        pass  # appo_config is already v5-compatible (actor/critic format)
    elif is_rsl_rl_v4():
        cfg = convert_config_v3_to_v4(cfg)

    obs_example = torch.zeros((num_envs, obs_dim), device=collector_device)
    td_example = TensorDict({"policy": obs_example}, batch_size=num_envs)

    # deepcopy so MLPModel.__init__'s distribution_cfg.pop("class_name") doesn't
    # mutate the shared rl_cfg dict.
    actor_cfg = deepcopy(cfg["actor"])
    actor_cls = resolve_callable(actor_cfg.pop("class_name"))
    actor_cfg.pop("num_actions", None)
    actor = actor_cls(
        td_example,
        cfg.get("obs_groups", {"actor": {"policy": obs_dim}}),
        "actor",
        action_dim,
        **actor_cfg,
    )
    actor = actor.to(collector_device)
    actor.eval()

    # Load initial weights
    sd = dict(actor.state_dict())
    weight_sync.read_weights_into(sd)
    actor.load_state_dict(sd)
    local_weight_version = weight_sync.version

    # Reset environment
    env_indices = np.arange(num_envs, dtype=np.int32)
    try:
        obs_out, _ = env.reset(env_indices)
    except TypeError:
        obs_out, _ = env.reset()

    def to_float32_np(x):
        if hasattr(x, "cpu"):
            x = x.cpu().numpy()
        return np.asarray(x, dtype=np.float32)

    obs_np, priv_np = split_obs_dict(obs_out)
    obs_np = to_float32_np(obs_np)
    if priv_np is not None:
        priv_np = to_float32_np(priv_np)

    # Pre-allocate obs TensorDict once; update in-place each step to avoid
    # repeated TensorDict construction overhead in the hot loop.
    obs_torch = torch.zeros((num_envs, obs_dim), dtype=torch.float32, device=collector_device)
    obs_td = TensorDict({"policy": obs_torch}, batch_size=num_envs, device=collector_device)

    total_steps = 0
    ep_rewards = []
    ep_lengths = []
    current_ep_rewards = np.zeros(num_envs, dtype=np.float32)
    current_ep_lengths = np.zeros(num_envs, dtype=np.int32)
    ep_reward_components = defaultdict(list)

    # Episode completion mode counters (reset after each metrics report)
    ep_timeouts = 0
    ep_terminates = 0

    # Collector timing EMA (milliseconds, α=0.1 → slow-moving average)
    _EMA = 0.1
    ema_mlp_infer_ms: float = 0.0
    ema_env_step_ms: float = 0.0

    try:
        while not stop_event.is_set():
            # Pull latest weights from learner
            if weight_sync.version > local_weight_version:
                sd = dict(actor.state_dict())
                local_weight_version = weight_sync.read_weights_into(sd)
                actor.load_state_dict(sd)

            # Collect one rollout of length steps_per_env
            write_buf = storage.write_buffer
            for step in range(steps_per_env):
                # --- MLP inference (timed) ---
                t_mlp = time.perf_counter()
                with torch.no_grad():
                    obs_torch.copy_(torch.from_numpy(obs_np))
                    actions_torch = actor(obs_td, stochastic_output=True)
                    log_probs_torch = actor.get_output_log_prob(actions_torch)
                    actions_np = actions_torch.cpu().numpy()
                ema_mlp_infer_ms = (1 - _EMA) * ema_mlp_infer_ms + _EMA * (
                    (time.perf_counter() - t_mlp) * 1000
                )

                write_buf["obs"][:, step, :] = obs_np
                if priv_np is not None:
                    write_buf["privileged"][:, step, :] = priv_np
                write_buf["actions"][:, step, :] = actions_np
                write_buf["log_probs"][:, step] = log_probs_torch.cpu().numpy().ravel()

                # --- Env step (timed) ---
                t_env = time.perf_counter()
                state = env.step(actions_np)
                ema_env_step_ms = (1 - _EMA) * ema_env_step_ms + _EMA * (
                    (time.perf_counter() - t_env) * 1000
                )

                next_obs_raw = state.obs
                reward_raw = np.asarray(state.reward, dtype=np.float32).ravel()
                done_raw = np.asarray(state.terminated, dtype=np.float32).ravel()
                truncated_raw = np.asarray(state.truncated, dtype=np.float32).ravel()

                write_buf["rewards"][:, step] = reward_raw
                write_buf["dones"][:, step] = done_raw
                write_buf["truncated"][:, step] = truncated_raw

                obs_np, priv_np = split_obs_dict(next_obs_raw)
                obs_np = to_float32_np(obs_np)
                if priv_np is not None:
                    priv_np = to_float32_np(priv_np)

                # Bootstrap from true terminal obs so next rollout starts cleanly
                if "_final_observation" in state.info:
                    has_final = np.asarray(state.info["_final_observation"], dtype=bool)
                    if np.any(has_final):
                        final_obs_flat = flatten_obs_dict(state.info["final_observation"])
                        obs_np[has_final] = to_float32_np(final_obs_flat)[has_final]

                # Episode tracking (vectorized)
                total_steps += num_envs
                current_ep_rewards += reward_raw
                current_ep_lengths += 1
                combined_dones = np.clip(done_raw + truncated_raw, 0, 1)
                reset_indices = np.where(combined_dones > 0.5)[0]
                if len(reset_indices) > 0:
                    ep_rewards.extend(current_ep_rewards[reset_indices].tolist())
                    ep_lengths.extend(current_ep_lengths[reset_indices].tolist())
                    current_ep_rewards[reset_indices] = 0.0
                    current_ep_lengths[reset_indices] = 0
                    # Count episode completion modes for timeout/terminated rates
                    ep_timeouts += int(np.sum(truncated_raw[reset_indices] > 0.5))
                    ep_terminates += int(np.sum(truncated_raw[reset_indices] <= 0.5))

                log_info = state.info.get("log", {})
                for k, v in log_info.items():
                    if k.startswith("reward/"):
                        ep_reward_components[k].append(v)

                if metrics_queue is not None and total_steps % (num_envs * 10) == 0 and ep_rewards:
                    try:
                        msg = {
                            "total_steps": total_steps,
                            "mean_ep_reward": statistics.mean(ep_rewards[-100:]),
                            "mean_ep_length": statistics.mean(ep_lengths[-100:])
                            if ep_lengths
                            else 0.0,
                        }
                        # Episode completion mode rates
                        total_ep = ep_timeouts + ep_terminates
                        if total_ep > 0:
                            msg["timeout_rate"] = ep_timeouts / total_ep
                            msg["terminated_rate"] = ep_terminates / total_ep
                            ep_timeouts = 0
                            ep_terminates = 0
                        # Collector-side timing breakdown
                        msg["collector_timing_ms"] = {
                            "mlp_infer_ms": ema_mlp_infer_ms,
                            "env_step_total_ms": ema_env_step_ms,
                        }
                        if ep_reward_components:
                            msg["reward_components"] = {
                                k: statistics.mean(v) for k, v in ep_reward_components.items() if v
                            }
                            ep_reward_components.clear()
                        metrics_queue.put_nowait(msg)
                    except Exception as e:
                        print(f"[APPOWorker] metrics enqueue error: {e}", file=sys.stderr)

            write_buf["last_obs"][:] = obs_np
            if priv_np is not None:
                write_buf["last_privileged"][:] = priv_np
            storage.signal_write_done()  # atomic increment, non-blocking

    except Exception as e:
        import traceback

        print(f"\n[APPO WORKER CRASH]: {e}\n", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        if metrics_queue is not None:
            try:
                metrics_queue.put_nowait({"error": str(e)})
            except Exception:
                pass
        stop_event.set()
        raise

    storage.close()
    weight_sync.close()
    env.close()
