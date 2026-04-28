"""HORA-owned APPO rollout worker."""

from __future__ import annotations

import statistics
import sys
import time
from collections import defaultdict
from typing import Any, Dict

import numpy as np
import torch
from rsl_rl.utils import resolve_callable

from unilab.base.final_observation import resolve_terminal_observation_contract
from unilab.base.registry import ensure_registries

from .observations import split_hora_obs_with_priv_info


def compute_hora_timeout_bootstrap_correction(
    critic: Any,
    collector_device: str,
    gamma: float,
    timeout_mask: np.ndarray,
    final_obs: np.ndarray,
    final_critic: np.ndarray | None = None,
    final_priv_info: np.ndarray | None = None,
) -> np.ndarray:
    """Compute timeout bootstrap values for grouped HORA observations."""
    corrections = np.zeros(timeout_mask.shape, dtype=np.float32)
    if not np.any(timeout_mask):
        return corrections

    from tensordict import TensorDict

    if final_priv_info is not None:
        actor_input = torch.from_numpy(final_obs[timeout_mask]).to(collector_device)
        priv_input = torch.from_numpy(final_priv_info[timeout_mask]).to(collector_device)
        critic_td = TensorDict(
            {"actor": actor_input, "priv_info": priv_input},
            batch_size=actor_input.shape[0],
            device=collector_device,
        )
    else:
        critic_input_np = final_critic if final_critic is not None else final_obs
        critic_input = torch.from_numpy(critic_input_np[timeout_mask]).to(collector_device)
        critic_td = TensorDict(
            {"policy": critic_input},
            batch_size=critic_input.shape[0],
            device=collector_device,
        )
    with torch.no_grad():
        bootstrap = critic(critic_td).squeeze(-1).cpu().numpy().astype(np.float32, copy=False)
    corrections[timeout_mask] = float(gamma) * bootstrap
    return corrections


def hora_appo_collector_fn(
    stop_event: Any,
    env_name: str,
    rl_cfg: dict,
    num_envs: int,
    steps_per_env: int,
    shm_storage_name: Dict[str, str],
    sync_primitives: tuple,
    obs_dim: int,
    action_dim: int,
    critic_dim: int,
    actor_weight_sync_name: str,
    actor_weight_param_shapes: dict,
    critic_weight_sync_name: str,
    critic_weight_param_shapes: dict,
    metrics_queue: Any,
    collector_device: str = "cpu",
    sim_backend: str = "mujoco",
    env_cfg_override: dict | None = None,
    priv_info_dim: int = 0,
):
    """Collect grouped HORA APPO rollouts into shared storage."""
    from copy import deepcopy

    from tensordict import TensorDict

    from unilab.algos.torch.hora.rsl_rl_compat import (
        convert_config_v3_to_v4,
        is_rsl_rl_v4,
        is_rsl_rl_v5,
    )
    from unilab.base import registry
    from unilab.ipc import SharedOnPolicyStorage, SharedWeightSync

    ensure_registries()

    storage = SharedOnPolicyStorage(
        num_envs=num_envs,
        num_steps=steps_per_env,
        obs_dim=obs_dim,
        action_dim=action_dim,
        critic_dim=critic_dim,
        priv_info_dim=priv_info_dim,
        create=False,
        shm_name_prefix=shm_storage_name,
    )
    storage.attach_sync_primitives(*sync_primitives)
    actor_weight_sync = SharedWeightSync(
        actor_weight_param_shapes,
        create=False,
        shm_name=actor_weight_sync_name,
    )
    critic_weight_sync = SharedWeightSync(
        critic_weight_param_shapes,
        create=False,
        shm_name=critic_weight_sync_name,
    )

    env: Any = registry.make(
        env_name,
        num_envs=num_envs,
        sim_backend=sim_backend,
        env_cfg_override=env_cfg_override,
    )

    cfg = dict(rl_cfg)
    if is_rsl_rl_v5():
        pass
    elif is_rsl_rl_v4():
        cfg = convert_config_v3_to_v4(cfg)

    if priv_info_dim <= 0:
        raise ValueError("HORA APPO collector requires priv_info_dim > 0")

    obs_example = torch.zeros((num_envs, obs_dim), device=collector_device)
    priv_info_example = torch.zeros((num_envs, priv_info_dim), device=collector_device)
    td_example = TensorDict(
        {"actor": obs_example, "priv_info": priv_info_example},
        batch_size=num_envs,
        device=collector_device,
    )

    actor_cfg = deepcopy(cfg["actor"])
    actor_cls = resolve_callable(actor_cfg.pop("class_name"))
    actor_cfg.pop("num_actions", None)
    actor = actor_cls(td_example, cfg["obs_groups"], "actor", action_dim, **actor_cfg)
    actor = actor.to(collector_device)
    actor.eval()

    critic_cfg = deepcopy(cfg.get("critic") or cfg.get("actor") or {})
    critic_cls = resolve_callable(critic_cfg.pop("class_name", "rsl_rl.models.MLPModel"))
    critic_cfg.pop("num_actions", None)
    critic_cfg.pop("distribution_cfg", None)
    critic = critic_cls(td_example, cfg["obs_groups"], "critic", 1, **critic_cfg)
    critic = critic.to(collector_device)
    critic.eval()

    actor_sd = dict(actor.state_dict())
    actor_weight_sync.read_weights_into(actor_sd)
    actor.load_state_dict(actor_sd)
    local_actor_weight_version = actor_weight_sync.version

    critic_sd = dict(critic.state_dict())
    critic_weight_sync.read_weights_into(critic_sd)
    critic.load_state_dict(critic_sd)
    local_critic_weight_version = critic_weight_sync.version

    env_indices = np.arange(num_envs, dtype=np.int32)
    try:
        obs_out, info_out = env.reset(env_indices)
    except TypeError:
        obs_out, info_out = env.reset()

    def to_float32_np(x):
        if hasattr(x, "cpu"):
            x = x.cpu().numpy()
        return np.asarray(x, dtype=np.float32)

    obs_np, critic_np, priv_info_np = split_hora_obs_with_priv_info(obs_out, info_out)
    obs_np = to_float32_np(obs_np)
    if critic_np is not None:
        critic_np = to_float32_np(critic_np)
    if priv_info_np is not None:
        priv_info_np = to_float32_np(priv_info_np)
    if priv_info_np is None:
        raise ValueError("HORA APPO collector did not receive privileged info from env reset.")

    obs_torch = torch.zeros((num_envs, obs_dim), dtype=torch.float32, device=collector_device)
    priv_info_torch = torch.zeros(
        (num_envs, priv_info_dim),
        dtype=torch.float32,
        device=collector_device,
    )
    obs_td = TensorDict(
        {"actor": obs_torch, "priv_info": priv_info_torch},
        batch_size=num_envs,
        device=collector_device,
    )

    total_steps = 0
    ep_rewards = []
    ep_lengths = []
    current_ep_rewards = np.zeros(num_envs, dtype=np.float32)
    current_ep_lengths = np.zeros(num_envs, dtype=np.int32)
    ep_reward_components = defaultdict(list)
    ep_timeouts = 0
    ep_terminates = 0

    _EMA = 0.1
    ema_mlp_infer_ms = 0.0
    ema_env_step_ms = 0.0

    try:
        while not stop_event.is_set():
            if actor_weight_sync.version > local_actor_weight_version:
                actor_sd = dict(actor.state_dict())
                local_actor_weight_version = actor_weight_sync.read_weights_into(actor_sd)
                actor.load_state_dict(actor_sd)
            if critic_weight_sync.version > local_critic_weight_version:
                critic_sd = dict(critic.state_dict())
                local_critic_weight_version = critic_weight_sync.read_weights_into(critic_sd)
                critic.load_state_dict(critic_sd)

            write_buf = storage.write_buffer
            for step in range(steps_per_env):
                t_mlp = time.perf_counter()
                with torch.no_grad():
                    obs_torch.copy_(torch.from_numpy(obs_np))
                    priv_info_torch.copy_(torch.from_numpy(priv_info_np))
                    actions_torch = actor(obs_td, stochastic_output=True)
                    log_probs_torch = actor.get_output_log_prob(actions_torch)
                    actions_np = actions_torch.cpu().numpy()
                ema_mlp_infer_ms = (1 - _EMA) * ema_mlp_infer_ms + _EMA * (
                    (time.perf_counter() - t_mlp) * 1000
                )

                write_buf["obs"][:, step, :] = obs_np
                if critic_np is not None:
                    write_buf["critic"][:, step, :] = critic_np
                write_buf["priv_info"][:, step, :] = priv_info_np
                write_buf["actions"][:, step, :] = actions_np
                write_buf["log_probs"][:, step] = log_probs_torch.cpu().numpy().ravel()

                t_env = time.perf_counter()
                state = env.step(actions_np)
                ema_env_step_ms = (1 - _EMA) * ema_env_step_ms + _EMA * (
                    (time.perf_counter() - t_env) * 1000
                )

                reward_raw = np.asarray(state.reward, dtype=np.float32).ravel()
                terminated_raw = np.asarray(state.terminated, dtype=np.float32).ravel()
                truncated_raw = np.asarray(state.truncated, dtype=np.float32).ravel()
                combined_done_raw = np.clip(terminated_raw + truncated_raw, 0, 1)

                next_actor_obs_np, next_critic_np, next_priv_info_np = (
                    split_hora_obs_with_priv_info(
                        state.obs,
                        state.info,
                    )
                )
                next_actor_obs_np = to_float32_np(next_actor_obs_np)
                if next_critic_np is not None:
                    next_critic_np = to_float32_np(next_critic_np)
                if next_priv_info_np is not None:
                    next_priv_info_np = to_float32_np(next_priv_info_np)
                if next_priv_info_np is None:
                    raise ValueError(
                        "HORA APPO collector did not receive privileged info after step."
                    )

                terminal_contract = resolve_terminal_observation_contract(
                    next_obs_batch_size=next_actor_obs_np.shape[0],
                    final_observation=getattr(state, "final_observation", None),
                    done=combined_done_raw > 0.5,
                    info=state.info,
                    truncated=truncated_raw,
                )
                terminal_priv_info = None
                if (
                    terminal_contract.terminal_obs is not None
                    and terminal_contract.terminal_critic is not None
                ):
                    _, _, terminal_priv_info = split_hora_obs_with_priv_info(
                        {
                            "obs": terminal_contract.terminal_obs,
                            "critic": terminal_contract.terminal_critic,
                        }
                    )
                    if terminal_priv_info is not None:
                        terminal_priv_info = to_float32_np(terminal_priv_info)

                reward_raw += compute_hora_timeout_bootstrap_correction(
                    critic=critic,
                    collector_device=collector_device,
                    gamma=float(cfg["algorithm"].get("gamma", 0.99)),
                    timeout_mask=terminal_contract.timeout_terminal_mask,
                    final_obs=(
                        terminal_contract.terminal_obs
                        if terminal_contract.terminal_obs is not None
                        else next_actor_obs_np
                    ),
                    final_critic=(
                        terminal_contract.terminal_critic
                        if terminal_contract.terminal_critic is not None
                        else next_critic_np
                    ),
                    final_priv_info=(
                        terminal_priv_info if terminal_priv_info is not None else next_priv_info_np
                    ),
                )

                write_buf["rewards"][:, step] = reward_raw
                write_buf["dones"][:, step] = combined_done_raw
                write_buf["truncated"][:, step] = truncated_raw

                total_steps += num_envs
                current_ep_rewards += reward_raw
                current_ep_lengths += 1
                reset_indices = np.where(combined_done_raw > 0.5)[0]
                if len(reset_indices) > 0:
                    ep_rewards.extend(current_ep_rewards[reset_indices].tolist())
                    ep_lengths.extend(current_ep_lengths[reset_indices].tolist())
                    current_ep_rewards[reset_indices] = 0.0
                    current_ep_lengths[reset_indices] = 0
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
                        total_ep = ep_timeouts + ep_terminates
                        if total_ep > 0:
                            msg["timeout_rate"] = ep_timeouts / total_ep
                            msg["terminated_rate"] = ep_terminates / total_ep
                            ep_timeouts = 0
                            ep_terminates = 0
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
                        print(f"[HoraAPPOWorker] metrics enqueue error: {e}", file=sys.stderr)

                obs_np = next_actor_obs_np
                critic_np = next_critic_np
                priv_info_np = next_priv_info_np

            write_buf["last_obs"][:] = obs_np
            if critic_np is not None:
                write_buf["last_critic"][:] = critic_np
            write_buf["last_priv_info"][:] = priv_info_np
            storage.signal_write_done()

    except Exception as e:
        import traceback

        print(f"\n[HORA APPO WORKER CRASH]: {e}\n", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        if metrics_queue is not None:
            try:
                metrics_queue.put_nowait({"error": str(e)})
            except Exception:
                pass
        stop_event.set()
        raise

    storage.close()
    actor_weight_sync.close()
    critic_weight_sync.close()
    env.close()
