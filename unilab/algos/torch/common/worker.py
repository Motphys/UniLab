"""Off-policy collector for SAC and TD3.

Collects (obs, action, reward, next_obs, done) transitions using the current
actor policy.  Runs in a subprocess; writes to a SharedReplayBuffer.

Data flow:
    env.step(actions_np) → state (numpy-like) → np.asarray() → replay buffer (numpy)
    obs (numpy-like) → np.asarray → torch.from_numpy → actor → actions (torch) → numpy → env
"""

import torch
import numpy as np
import pkgutil
import importlib
def ensure_registries():
    """Import all env modules so they are registered."""
    try:
        import unilab.envs.locomotion
        package = unilab.envs.locomotion
        if hasattr(package, "__path__"):
            for _, name, ispkg in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
                try:
                    importlib.import_module(name)
                except Exception:
                    pass
    except ImportError:
        pass


def _build_actor(algo_type, obs_dim, action_dim, actor_hidden_dim, use_layer_norm, device, num_envs=1, obs_normalization=False):
    """Build the correct actor model based on algorithm type."""
    if algo_type == "sac":
        from unilab.algos.torch.fast_sac.learner import SACActor
        return SACActor(obs_dim=obs_dim, action_dim=action_dim,
                        hidden_dim=actor_hidden_dim, use_layer_norm=use_layer_norm,
                        device=device)
    elif algo_type == "td3":
        from unilab.algos.torch.fast_td3.learner import TD3Actor
        return TD3Actor(n_obs=obs_dim, n_act=action_dim, num_envs=num_envs,
                        hidden_dim=actor_hidden_dim, init_scale=0.01,
                        std_min=0.4, std_max=1.0, device=device)
    else:
        raise ValueError(f"Unknown algo_type: {algo_type}")


def off_policy_collector_fn(
    stop_event,
    env_name: str,
    num_envs: int,
    shm_buffer_name: str,
    buffer_capacity: int,
    obs_dim: int,
    action_dim: int,
    weight_sync_name: str,
    weight_param_shapes: dict,
    algo_type: str = "sac",
    actor_hidden_dim: int = 512,
    use_layer_norm: bool = True,
    collector_device: str = "cpu",
    exploration_noise: float = 0.1,
    warmup_steps: int = 5000,
    metrics_queue=None,
    buffer_lock=None,
    weight_sync_lock=None,
    sync_collection: bool = False,
    collection_ready_queue=None,
    trainer_done_queue=None,
    env_steps_per_sync: int = 1,
    obs_normalization: bool = False,
    shared_obs_normalizer_stats=None,
    **kwargs,
):
    """Entry point for the off-policy collector subprocess."""
    import traceback
    import sys
    try:
        print("[Collector] Entry point called", file=sys.stderr, flush=True)
        _run_collector(
            stop_event=stop_event,
            env_name=env_name,
            num_envs=num_envs, shm_buffer_name=shm_buffer_name,
            buffer_capacity=buffer_capacity, obs_dim=obs_dim, action_dim=action_dim,
            weight_sync_name=weight_sync_name, weight_param_shapes=weight_param_shapes,
            algo_type=algo_type, actor_hidden_dim=actor_hidden_dim,
            use_layer_norm=use_layer_norm, collector_device=collector_device,
            exploration_noise=exploration_noise, warmup_steps=warmup_steps,
            metrics_queue=metrics_queue, buffer_lock=buffer_lock,
            weight_sync_lock=weight_sync_lock, sync_collection=sync_collection,
            collection_ready_queue=collection_ready_queue, trainer_done_queue=trainer_done_queue,
            env_steps_per_sync=env_steps_per_sync,
            obs_normalization=obs_normalization,
            shared_obs_normalizer_stats=shared_obs_normalizer_stats,
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
    env_name, num_envs,
    shm_buffer_name, buffer_capacity, obs_dim, action_dim,
    weight_sync_name, weight_param_shapes,
    algo_type, actor_hidden_dim, use_layer_norm, collector_device,
    exploration_noise, warmup_steps, metrics_queue, buffer_lock,
    weight_sync_lock, sync_collection, collection_ready_queue, trainer_done_queue, env_steps_per_sync,
    obs_normalization, shared_obs_normalizer_stats
):
    from unilab.ipc import SharedReplayBuffer, SharedWeightSync
    from unilab.envs import registry

    ensure_registries()
    
    # Initialize environment
    env = registry.make(env_name, num_envs=num_envs, sim_backend="mujoco")
    if env.state is None:
        env.init_state()

    # Connect to shared memory
    replay_buffer = SharedReplayBuffer(
        buffer_capacity, obs_dim, action_dim, create=False, shm_name=shm_buffer_name,
        lock=buffer_lock,
    )
    # Connect to weight sync
    weight_sync = SharedWeightSync(
        weight_param_shapes, create=False, shm_name=weight_sync_name, lock=weight_sync_lock
    )

    # Build actor
    actor = _build_actor(algo_type, obs_dim, action_dim, actor_hidden_dim, use_layer_norm, collector_device, num_envs, obs_normalization)
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
    obs_np = np.asarray(state.obs, dtype=np.float32)
    max_episode_steps = getattr(getattr(env, "cfg", None), "max_episode_steps", None)
    if max_episode_steps is not None and int(max_episode_steps) > 0:
        step_offsets = np.random.randint(0, int(max_episode_steps), size=(num_envs,), dtype=np.uint32)
        if hasattr(env, "state") and env.state is not None and isinstance(getattr(env.state, "info", None), dict):
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
                    pass # Handled by EmpiricalNormalization in learner if actor possesses it. We need a local normalizer.
                    
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
                obs_torch = torch.from_numpy(obs_np_input).to(collector_device)
                if algo_type == "sac":
                    actions_torch = actor.explore(obs_torch)
                elif algo_type == "td3":
                    actions_torch = actor(obs_torch)
                    noise = torch.randn_like(actions_torch) * exploration_noise
                    actions_torch = (actions_torch + noise).clamp(-1, 1)
                else:
                    actions_torch = torch.zeros((num_envs, action_dim), device=collector_device)
                actions_np = actions_torch.cpu().numpy()

        # Step environment
        state = env.step(actions_np)

        timing_info = state.info.get("timing", {}) if hasattr(state, "info") else {}
        if timing_info:
            for key in ("env_step_total_ms", "step_core_ms", "update_state_ms", "reset_done_ms"):
                if key in timing_info:
                    timing_accum_ms[key] += float(timing_info[key])
            timing_count += 1

        # Extract data as numpy
        next_obs_np = np.asarray(state.obs, dtype=np.float32)
        rewards_np = np.asarray(state.reward, dtype=np.float32).ravel()

        terminated_np = np.asarray(state.terminated, dtype=np.float32).ravel() if state.terminated is not None else np.zeros(num_envs, dtype=np.float32)
        truncated_np = np.asarray(state.truncated, dtype=np.float32).ravel() if state.truncated is not None else np.zeros(num_envs, dtype=np.float32)
        combined_dones = np.clip(terminated_np + truncated_np, 0, 1)
        done_mask_np = combined_dones > 0.5
        timeout_mask_np = truncated_np > 0.5
        terminated_mask_np = np.logical_and(terminated_np > 0.5, ~timeout_mask_np)

        done_count_window += int(np.count_nonzero(done_mask_np))
        timeout_count_window += int(np.count_nonzero(timeout_mask_np))
        terminated_count_window += int(np.count_nonzero(terminated_mask_np))

        # Handle true terminal observations
        if "_final_observation" in state.info:
            has_final = state.info["_final_observation"]
            has_final_np = np.asarray(has_final, dtype=bool)
            if np.any(has_final_np):
                final_obs_np = np.asarray(state.info["final_observation"], dtype=np.float32)
                next_obs_np[has_final_np] = final_obs_np[has_final_np]

        # Write to replay buffer
        replay_buffer.add_batch(obs_np, actions_np, rewards_np, next_obs_np, terminated_np, truncated_np)

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
        total_steps += num_envs
        env_steps_since_sync += 1

        # Signal the learner once this collection chunk is ready.
        if sync_collection and collection_ready_queue is not None and trainer_done_queue is not None:
            if env_steps_since_sync >= env_steps_per_sync:
                collection_ready_queue.put(1)
                trainer_done_queue.get()  # Wait for trainer
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
                    "buffer_size": replay_buffer.size,
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
            except Exception:
                pass

    replay_buffer.close()
    weight_sync.close()
