"""Async PPO collector worker."""

import time as _time

import numpy as np
import torch
from rsl_rl.algorithms import PPO
from tensordict import TensorDict

from unilab.utils.algo_utils import ensure_registries


def async_ppo_collector_fn(
    stop_event,
    env_name: str,
    rl_cfg: dict,
    num_envs: int,
    steps_per_env: int,
    buffer,
    weight_sync_name: str,
    weight_param_shapes: dict,
    weight_sync_lock,
    metrics_queue,
    collector_device: str = "cpu",
):
    """Collect rollouts using PPO.act()."""
    from unilab.base import registry
    from unilab.ipc import SharedWeightSync
    from unilab.utils.rsl_rl_compat import convert_config_v3_to_v4, is_rsl_rl_v4

    ensure_registries()

    # Convert config
    cfg = dict(rl_cfg)
    if is_rsl_rl_v4():
        cfg = convert_config_v3_to_v4(cfg)

    # Create environment
    env = registry.make(env_name, num_envs=num_envs, sim_backend="mujoco")
    obs_dim = env.observation_space.shape[0]  # type: ignore[index]
    action_dim = env.action_space.shape[0]  # type: ignore[index]

    # Build PPO (inference only)
    if not hasattr(env, 'num_actions'):
        env.num_actions = action_dim

    obs_example = torch.zeros((num_envs, obs_dim), device=collector_device)
    td_example = TensorDict({"policy": obs_example}, batch_size=num_envs)

    ppo = PPO.construct_algorithm(
        env=env,
        obs=td_example,
        cfg=cfg,
        device=collector_device,
    )
    ppo.actor.eval()
    ppo.critic.eval()

    # Weight sync
    weight_sync = SharedWeightSync(
        weight_param_shapes, create=False, shm_name=weight_sync_name, lock=weight_sync_lock
    )
    sd = {
        f"actor.{k}": v for k, v in ppo.actor.state_dict().items()
    } | {
        f"critic.{k}": v for k, v in ppo.critic.state_dict().items()
    }
    weight_sync.read_weights_into(sd)
    ppo.actor.load_state_dict({k: sd[f"actor.{k}"] for k in ppo.actor.state_dict().keys()})
    ppo.critic.load_state_dict({k: sd[f"critic.{k}"] for k in ppo.critic.state_dict().keys()})
    local_version = weight_sync.version

    # Reset env
    env_indices = np.arange(num_envs, dtype=np.int32)
    try:
        _, obs_out, _ = env.reset(env_indices)  # type: ignore[attr-defined]
    except TypeError:
        obs_out, _ = env.reset()  # type: ignore[attr-defined]

    obs = torch.from_numpy(np.asarray(obs_out, dtype=np.float32)).to(collector_device)

    rollout_count = 0
    timing_accum_ms = {}
    timing_count = 0
    episode_lengths = []
    episode_rewards = []
    episode_timeouts = 0
    episode_terminates = 0
    episode_count = 0
    current_episode_rewards = np.zeros(num_envs, dtype=np.float32)
    current_episode_lengths = np.zeros(num_envs, dtype=np.int32)
    from collections import defaultdict
    ep_reward_components = defaultdict(list)
    try:
        while not stop_event.is_set():
            # Sync weights
            if weight_sync.version > local_version:
                sd = {
                    f"actor.{k}": v for k, v in ppo.actor.state_dict().items()
                } | {
                    f"critic.{k}": v for k, v in ppo.critic.state_dict().items()
                }
                local_version = weight_sync.read_weights_into(sd)
                ppo.actor.load_state_dict({k: sd[f"actor.{k}"] for k in ppo.actor.state_dict().keys()})
                ppo.critic.load_state_dict({k: sd[f"critic.{k}"] for k in ppo.critic.state_dict().keys()})

            # Collect rollout - write directly to shared buffer to avoid copy
            obs_buf = torch.zeros(steps_per_env, num_envs, obs_dim, device=collector_device)
            actions_buf = torch.zeros(steps_per_env, num_envs, action_dim, device=collector_device)
            rewards_buf = torch.zeros(steps_per_env, num_envs, device=collector_device)
            dones_buf = torch.zeros(steps_per_env, num_envs, device=collector_device)
            log_probs_buf = torch.zeros(steps_per_env, num_envs, device=collector_device)
            values_buf = torch.zeros(steps_per_env, num_envs, device=collector_device)

            with torch.no_grad():
                for step in range(steps_per_env):
                    obs_td = TensorDict(
                        {"policy": obs}, batch_size=num_envs, device=collector_device
                    )

                    # Use PPO.act()
                    _t_infer = _time.perf_counter()
                    actions = ppo.act(obs_td)
                    timing_accum_ms["mlp_infer_ms"] = timing_accum_ms.get("mlp_infer_ms", 0.0) + (
                        _time.perf_counter() - _t_infer
                    ) * 1000

                    # Extract from transition
                    log_probs = ppo.transition.actions_log_prob
                    values = ppo.transition.values

                    obs_buf[step] = obs
                    actions_buf[step] = actions
                    log_probs_buf[step] = log_probs.squeeze(-1)
                    values_buf[step] = values.squeeze(-1)

                    # Step env
                    state = env.step(actions.cpu().numpy())  # type: ignore[attr-defined]

                    # Extract timing from env
                    timing_info = state.info.get("timing", {}) if hasattr(state, "info") else {}
                    if timing_info:
                        for key in ("env_step_total_ms", "step_core_ms", "update_state_ms", "reset_done_ms"):
                            if key in timing_info:
                                timing_accum_ms[key] = timing_accum_ms.get(key, 0.0) + float(timing_info[key])
                    timing_count += 1

                    # Collect reward components from each step
                    if hasattr(state, "info") and isinstance(state.info, dict):
                        log_info = state.info.get("log", {})
                        for k, v in log_info.items():
                            if k.startswith("reward/"):
                                ep_reward_components[k].append(v)

                    rewards = torch.from_numpy(np.asarray(state.reward, dtype=np.float32)).to(
                        collector_device
                    )
                    dones = torch.from_numpy(np.asarray(state.terminated, dtype=np.float32)).to(
                        collector_device
                    )

                    rewards_buf[step] = rewards
                    dones_buf[step] = dones

                    # Track episode stats
                    current_episode_rewards += rewards.cpu().numpy()
                    current_episode_lengths += 1

                    for i in range(num_envs):
                        if dones[i] > 0:
                            episode_count += 1
                            episode_rewards.append(float(current_episode_rewards[i]))
                            episode_lengths.append(int(current_episode_lengths[i]))
                            current_episode_rewards[i] = 0
                            current_episode_lengths[i] = 0

                            # Check if timeout or terminated
                            if hasattr(state, "info") and isinstance(state.info, dict):
                                if state.info.get("TimeLimit.truncated", False):
                                    episode_timeouts += 1
                                else:
                                    episode_terminates += 1
                            else:
                                episode_terminates += 1

                    obs = torch.from_numpy(np.asarray(state.obs, dtype=np.float32)).to(
                        collector_device
                    )

            # Zero-copy write to buffer (already in shared memory)
            rollout = {
                "observations": obs_buf if obs_buf.is_shared() else obs_buf.cpu(),
                "actions": actions_buf if actions_buf.is_shared() else actions_buf.cpu(),
                "rewards": rewards_buf if rewards_buf.is_shared() else rewards_buf.cpu(),
                "dones": dones_buf if dones_buf.is_shared() else dones_buf.cpu(),
                "log_probs": log_probs_buf if log_probs_buf.is_shared() else log_probs_buf.cpu(),
                "values": values_buf if values_buf.is_shared() else values_buf.cpu(),
                "last_obs": obs if obs.is_shared() else obs.cpu(),
            }
            buffer.add_rollout(rollout)
            rollout_count += 1

            # Send metrics every rollout
            if metrics_queue is not None:
                try:
                    avg_timing = {}
                    if timing_count > 0:
                        avg_timing = {k: v / timing_count for k, v in timing_accum_ms.items()}

                    episode_metrics = {}
                    if episode_count > 0:
                        episode_metrics["timeout_rate"] = episode_timeouts / episode_count
                        episode_metrics["terminated_rate"] = episode_terminates / episode_count
                    if episode_rewards:
                        episode_metrics["mean_reward"] = float(np.mean(episode_rewards))
                    if episode_lengths:
                        episode_metrics["mean_length"] = float(np.mean(episode_lengths))

                    msg = {
                        "collector_timing_ms": avg_timing,
                        "episode_stats": episode_metrics,
                    }

                    if ep_reward_components:
                        import statistics
                        components_mean = {}
                        for k, vals in ep_reward_components.items():
                            if vals:
                                components_mean[k] = statistics.mean(vals)
                        msg["reward_components"] = components_mean
                        ep_reward_components.clear()

                    metrics_queue.put_nowait(msg)
                    timing_accum_ms.clear()
                    timing_count = 0
                    episode_timeouts = 0
                    episode_terminates = 0
                    episode_count = 0
                    episode_rewards.clear()
                    episode_lengths.clear()
                except Exception:
                    pass

    except KeyboardInterrupt:
        pass
