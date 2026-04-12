"""Off-policy collector for SAC and TD3.

Collects (obs, action, reward, next_obs, done) transitions using the current
actor policy. Runs in a subprocess; writes to ReplayBuffer.
"""

import queue
import sys

import numpy as np
import torch

from unilab.utils.algo_utils import build_actor, ensure_registries
from unilab.utils.final_observation import resolve_terminal_observation_contract
from unilab.utils.obs_utils import split_obs_dict


def resolve_collector_actor_dims(
    env,
    obs_dim: int | None = None,
    action_dim: int | None = None,
) -> tuple[int, int]:
    """Resolve actor dims for the collector.

    Prefer explicit dims from the parent process so learner and collector
    build identical actor shapes on override-heavy env paths.
    """
    if obs_dim is None:
        from unilab.utils.obs_utils import get_obs_dims

        obs_dim, _ = get_obs_dims(env.obs_groups_spec)

    if action_dim is None:
        assert env.action_space.shape is not None
        action_dim = env.action_space.shape[0]

    assert obs_dim is not None
    assert action_dim is not None
    return obs_dim, action_dim


def off_policy_collector_fn(
    stop_event,
    env_name: str,
    num_envs: int,
    replay_buffer,
    weight_sync_name: str,
    weight_param_shapes: dict,
    algo_type: str = "sac",
    actor_hidden_dim: int = 512,
    use_layer_norm: bool = True,
    warmup_steps: int = 5000,
    metrics_queue=None,
    weight_sync_lock=None,
    sync_collection: bool = False,
    collection_ready_queue=None,
    trainer_done_queue=None,
    env_steps_per_sync: int = 1,
    obs_normalization: bool = False,
    shared_obs_normalizer_stats=None,
    sim_backend: str = "mujoco",
    env_cfg_override: dict | None = None,
    obs_dim: int | None = None,
    action_dim: int | None = None,
    **kwargs,
):
    """Entry point for the off-policy collector subprocess."""
    import sys
    import traceback

    try:
        print("[Collector] Entry point called", file=sys.stderr, flush=True)
        _run_collector(
            stop_event=stop_event,
            env_name=env_name,
            num_envs=num_envs,
            replay_buffer=replay_buffer,
            weight_sync_name=weight_sync_name,
            weight_param_shapes=weight_param_shapes,
            algo_type=algo_type,
            actor_hidden_dim=actor_hidden_dim,
            use_layer_norm=use_layer_norm,
            warmup_steps=warmup_steps,
            metrics_queue=metrics_queue,
            weight_sync_lock=weight_sync_lock,
            sync_collection=sync_collection,
            collection_ready_queue=collection_ready_queue,
            trainer_done_queue=trainer_done_queue,
            env_steps_per_sync=env_steps_per_sync,
            obs_normalization=obs_normalization,
            shared_obs_normalizer_stats=shared_obs_normalizer_stats,
            sim_backend=sim_backend,
            env_cfg_override=env_cfg_override,
            obs_dim=obs_dim,
            action_dim=action_dim,
        )
    except Exception as e:
        print(f"[Collector] Exception: {e}", file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
        if metrics_queue is not None:
            try:
                metrics_queue.put_nowait({"error": str(e)})
            except Exception:
                pass


def _run_collector(
    stop_event,
    env_name,
    num_envs,
    replay_buffer,
    weight_sync_name,
    weight_param_shapes,
    algo_type,
    actor_hidden_dim,
    use_layer_norm,
    warmup_steps,
    metrics_queue,
    weight_sync_lock,
    sync_collection,
    collection_ready_queue,
    trainer_done_queue,
    env_steps_per_sync,
    obs_normalization,
    shared_obs_normalizer_stats,
    sim_backend,
    env_cfg_override,
    obs_dim,
    action_dim,
):
    from unilab.base import registry
    from unilab.ipc import SharedWeightSync

    ensure_registries()

    # Initialize environment
    env = registry.make(
        env_name, num_envs=num_envs, sim_backend=sim_backend, env_cfg_override=env_cfg_override
    )
    if env.state is None:
        env.init_state()

    # Connect to weight sync
    weight_sync = SharedWeightSync(
        weight_param_shapes, create=False, shm_name=weight_sync_name, lock=weight_sync_lock
    )

    # Build actor (always on CPU for env interaction)
    obs_dim, action_dim = resolve_collector_actor_dims(
        env,
        obs_dim=obs_dim,
        action_dim=action_dim,
    )
    actor = build_actor(
        algo_type, obs_dim, action_dim, actor_hidden_dim, use_layer_norm, "cpu", num_envs
    )
    actor.eval()

    # Load initial weights
    sd = dict(actor.state_dict())
    weight_sync.read_weights_into(sd)
    actor.load_state_dict(sd)
    local_weight_version = weight_sync.version

    total_steps = 0
    ep_rewards = []
    ep_lengths = []
    current_ep_rewards = np.zeros(num_envs, dtype=np.float32)
    current_ep_lengths = np.zeros(num_envs, dtype=np.int32)
    from collections import defaultdict

    ep_reward_components = defaultdict(list)
    timing_accum_ms = defaultdict(float)
    timing_count = 0
    done_count_window = 0
    timeout_count_window = 0
    terminated_count_window = 0

    # Initial step to get first observation
    actions_np = np.zeros((num_envs, action_dim), dtype=np.float32)
    state = env.step(actions_np)
    obs_np, priv_np = split_obs_dict(state.obs)
    obs_np = np.asarray(obs_np, dtype=np.float32)
    if priv_np is not None:
        priv_np = np.asarray(priv_np, dtype=np.float32)
    prev_dones_np = np.zeros(num_envs, dtype=np.float32)
    max_episode_steps = getattr(getattr(env, "cfg", None), "max_episode_steps", None)
    if max_episode_steps is not None and int(max_episode_steps) > 0:
        step_offsets = np.random.randint(
            0, int(max_episode_steps), size=(num_envs,), dtype=np.uint32
        )
        if (
            hasattr(env, "state")
            and env.state is not None
            and isinstance(getattr(env.state, "info", None), dict)
        ):
            if "steps" in env.state.info:
                env.state.info["steps"][:] = step_offsets
        if isinstance(getattr(state, "info", None), dict) and "steps" in state.info:
            state.info["steps"][:] = step_offsets
    import time as _time

    _last_log_time = _time.time()

    # Track env.step calls collected since the last learner phase.
    env_steps_since_sync = 0

    # Collection loop
    while not stop_event.is_set():
        # Check for weight updates
        if weight_sync.version > local_weight_version:
            sd = dict(actor.state_dict())
            local_weight_version = weight_sync.read_weights_into(sd)
            actor.load_state_dict(sd)

            # Update normalizer stats
            if obs_normalization and shared_obs_normalizer_stats is not None:
                stats = shared_obs_normalizer_stats.get()
                if stats is not None:
                    # Apply stats to a local normalizer if needed, or directly to actor
                    pass  # Handled by EmpiricalNormalization in learner if actor possesses it. We need a local normalizer.

        # Normalize obs_np
        obs_np_input = obs_np
        if obs_normalization and shared_obs_normalizer_stats is not None:
            stats = shared_obs_normalizer_stats.get()
            if stats is not None:
                mean, std = stats
                obs_np_input = (obs_np - mean) / (std + 1e-8)

        # Select action
        with torch.no_grad():
            if total_steps < warmup_steps:
                actions_np = np.random.uniform(-1, 1, (num_envs, action_dim)).astype(np.float32)
            else:
                _t_infer = _time.perf_counter()
                obs_torch = torch.from_numpy(obs_np_input)
                if algo_type in ("sac", "td3"):
                    dones_torch = torch.from_numpy(prev_dones_np)
                    actions_torch = actor.explore(obs_torch, dones=dones_torch, deterministic=False)
                else:
                    actions_torch = torch.zeros((num_envs, action_dim))
                actions_np = actions_torch.numpy()
                timing_accum_ms["mlp_infer_ms"] += (_time.perf_counter() - _t_infer) * 1000

        # Step environment
        state = env.step(actions_np)

        timing_info = state.info.get("timing", {}) if hasattr(state, "info") else {}
        if timing_info:
            for key in ("env_step_total_ms", "step_core_ms", "update_state_ms", "reset_done_ms"):
                if key in timing_info:
                    timing_accum_ms[key] += float(timing_info[key])
            timing_count += 1

        # Extract data as numpy
        next_obs_np, next_priv_np = split_obs_dict(state.obs)
        next_obs_np = np.asarray(next_obs_np, dtype=np.float32)
        if next_priv_np is not None:
            next_priv_np = np.asarray(next_priv_np, dtype=np.float32)
        rewards_np = np.asarray(state.reward, dtype=np.float32).ravel()

        terminated_np = (
            np.asarray(state.terminated, dtype=np.float32).ravel()
            if state.terminated is not None
            else np.zeros(num_envs, dtype=np.float32)
        )
        truncated_np = (
            np.asarray(state.truncated, dtype=np.float32).ravel()
            if state.truncated is not None
            else np.zeros(num_envs, dtype=np.float32)
        )
        combined_dones = np.clip(terminated_np + truncated_np, 0, 1)
        prev_dones_np = combined_dones.astype(np.float32, copy=False)
        done_mask_np = combined_dones > 0.5
        timeout_mask_np = truncated_np > 0.5
        terminated_mask_np = np.logical_and(terminated_np > 0.5, ~timeout_mask_np)

        done_count_window += int(np.count_nonzero(done_mask_np))
        timeout_count_window += int(np.count_nonzero(timeout_mask_np))
        terminated_count_window += int(np.count_nonzero(terminated_mask_np))

        terminal_contract = resolve_terminal_observation_contract(
            next_obs_batch_size=next_obs_np.shape[0],
            final_observation=getattr(state, "final_observation", None),
            done=done_mask_np,
            info=state.info,
            truncated=truncated_np,
        )

        # Write to replay buffer
        replay_buffer.add(
            torch.from_numpy(obs_np),
            torch.from_numpy(actions_np),
            torch.from_numpy(rewards_np),
            torch.from_numpy(next_obs_np),
            torch.from_numpy(terminated_np),
            torch.from_numpy(truncated_np),
            torch.from_numpy(priv_np) if priv_np is not None else None,
            torch.from_numpy(next_priv_np) if next_priv_np is not None else None,
            terminal_mask=torch.from_numpy(terminal_contract.terminal_mask),
            terminal_next_obs=(
                torch.from_numpy(terminal_contract.terminal_obs)
                if terminal_contract.terminal_obs is not None
                else None
            ),
            terminal_next_privileged=(
                torch.from_numpy(terminal_contract.terminal_privileged)
                if terminal_contract.terminal_privileged is not None
                else None
            ),
        )

        # Track episode rewards - vectorized
        current_ep_rewards += rewards_np
        current_ep_lengths += 1
        reset_mask = combined_dones > 0.5
        reset_indices = np.where(reset_mask)[0]
        if len(reset_indices) > 0:
            ep_rewards.extend(current_ep_rewards[reset_indices].tolist())
            ep_lengths.extend(current_ep_lengths[reset_indices].tolist())
            current_ep_rewards[reset_indices] = 0.0
            current_ep_lengths[reset_indices] = 0

        obs_np = next_obs_np
        priv_np = next_priv_np
        total_steps += num_envs
        env_steps_since_sync += 1

        # Signal the learner once this collection chunk is ready.
        if (
            sync_collection
            and collection_ready_queue is not None
            and trainer_done_queue is not None
        ):
            if env_steps_since_sync >= env_steps_per_sync:
                collection_ready_queue.put(1)
                while not stop_event.is_set():
                    try:
                        trainer_done_queue.get(timeout=1.0)
                        break
                    except queue.Empty:
                        continue
                env_steps_since_sync = 0

        # Progress log every 2 seconds
        now = _time.time()
        if now - _last_log_time > 2.0:
            _last_log_time = now

        # Extract reward components from env info
        log_info = state.info.get("log", {})
        if log_info:
            for k, v in log_info.items():
                if k.startswith("reward/"):
                    ep_reward_components[k].append(v)

        # Send metrics periodically
        if metrics_queue is not None and total_steps % (num_envs * 10) == 0 and ep_rewards:
            import statistics

            try:
                msg = {
                    "total_steps": total_steps,
                    "mean_ep_reward": statistics.mean(ep_rewards[-100:]),
                    "mean_ep_length": statistics.mean(ep_lengths[-100:]) if ep_lengths else 0.0,
                    "buffer_size": int(replay_buffer.size[0]),
                }
                # Add mean reward components
                if ep_reward_components:
                    components_mean = {}
                    for k, vals in ep_reward_components.items():
                        if vals:
                            components_mean[k] = statistics.mean(vals)
                    msg["reward_components"] = components_mean
                    ep_reward_components.clear()  # reset after sending

                if timing_count > 0:
                    msg["collector_timing_ms"] = {
                        k: (v / timing_count) for k, v in timing_accum_ms.items()
                    }
                    timing_accum_ms.clear()
                    timing_count = 0

                if done_count_window > 0:
                    msg["timeout_rate"] = timeout_count_window / done_count_window
                    msg["terminated_rate"] = terminated_count_window / done_count_window
                    done_count_window = 0
                    timeout_count_window = 0
                    terminated_count_window = 0

                metrics_queue.put_nowait(msg)
            except Exception as e:
                print(f"[OffPolicyWorker] metrics enqueue error: {e}", file=sys.stderr)

    weight_sync.close()
