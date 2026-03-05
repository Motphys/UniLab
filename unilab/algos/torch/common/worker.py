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


def _build_actor(algo_type, obs_dim, action_dim, actor_hidden_dim, use_layer_norm, device):
    """Build the correct actor model based on algorithm type."""
    if algo_type == "sac":
        from unilab.algos.torch.fast_sac.learner import SACActor
        return SACActor(obs_dim=obs_dim, action_dim=action_dim,
                        hidden_dim=actor_hidden_dim, use_layer_norm=use_layer_norm,
                        device=device)
    elif algo_type == "td3":
        from unilab.algos.torch.fast_td3.learner import TD3Actor
        return TD3Actor(obs_dim=obs_dim, action_dim=action_dim,
                        hidden_dim=actor_hidden_dim, use_layer_norm=use_layer_norm,
                        device=device)
    else:
        raise ValueError(f"Unknown algo_type: {algo_type}")


def off_policy_collector_fn(
    stop_event,
    env_name: str,
    env_cfg_overrides: dict,
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
    **kwargs,
):
    """Entry point for the off-policy collector subprocess."""
    import traceback
    try:
        _run_collector(
            stop_event=stop_event,
            env_name=env_name, env_cfg_overrides=env_cfg_overrides,
            num_envs=num_envs, shm_buffer_name=shm_buffer_name,
            buffer_capacity=buffer_capacity, obs_dim=obs_dim, action_dim=action_dim,
            weight_sync_name=weight_sync_name, weight_param_shapes=weight_param_shapes,
            algo_type=algo_type, actor_hidden_dim=actor_hidden_dim,
            use_layer_norm=use_layer_norm, collector_device=collector_device,
            exploration_noise=exploration_noise, warmup_steps=warmup_steps,
            metrics_queue=metrics_queue, buffer_lock=buffer_lock,
        )
    except Exception as e:
        traceback.print_exc()
        if metrics_queue is not None:
            try:
                metrics_queue.put_nowait({"error": str(e)})
            except Exception:
                pass


def _run_collector(
    stop_event,
    env_name, env_cfg_overrides, num_envs,
    shm_buffer_name, buffer_capacity, obs_dim, action_dim,
    weight_sync_name, weight_param_shapes,
    algo_type, actor_hidden_dim, use_layer_norm, collector_device,
    exploration_noise, warmup_steps, metrics_queue, buffer_lock,
):
    from unilab.ipc import SharedReplayBuffer, SharedWeightSync
    from unilab.envs import registry

    ensure_registries()

    # Connect to shared memory
    replay_buffer = SharedReplayBuffer(
        buffer_capacity, obs_dim, action_dim, create=False, shm_name=shm_buffer_name
    )
    replay_buffer._lock = buffer_lock
    weight_sync = SharedWeightSync(
        weight_param_shapes, create=False, shm_name=weight_sync_name
    )

    # Create environment - use numpy backend for PyTorch algorithms
    env = registry.make(env_name, num_envs=num_envs, sim_backend="mujoco")

    # Build actor
    actor = _build_actor(algo_type, obs_dim, action_dim, actor_hidden_dim, use_layer_norm, collector_device)
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

    # Initial step to get first observation
    actions_np = np.zeros((num_envs, action_dim), dtype=np.float32)
    state = env.step(actions_np)
    obs_np = np.asarray(state.obs, dtype=np.float32)
    import time as _time
    _last_log_time = _time.time()

    # Collection loop
    while not stop_event.is_set():
        # Check for weight updates
        if weight_sync.version > local_weight_version:
            sd = dict(actor.state_dict())
            local_weight_version = weight_sync.read_weights_into(sd)
            actor.load_state_dict(sd)

        # Select action
        with torch.no_grad():
            if total_steps < warmup_steps:
                actions_np = np.random.uniform(-1, 1, (num_envs, action_dim)).astype(np.float32)
            else:
                obs_torch = torch.from_numpy(obs_np).to(collector_device)
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

        # Extract data as numpy
        next_obs_np = np.asarray(state.obs, dtype=np.float32)
        rewards_np = np.asarray(state.reward, dtype=np.float32).ravel()
        terminated_np = np.asarray(state.terminated, dtype=np.float32).ravel() if state.terminated is not None else np.zeros(num_envs, dtype=np.float32)
        truncated_np = np.asarray(state.truncated, dtype=np.float32).ravel() if state.truncated is not None else np.zeros(num_envs, dtype=np.float32)
        combined_dones = np.clip(terminated_np + truncated_np, 0, 1)

        # Handle true terminal observations
        if "_final_observation" in state.info:
            has_final = state.info["_final_observation"]
            has_final_np = np.asarray(has_final, dtype=bool)
            if np.any(has_final_np):
                final_obs_np = np.asarray(state.info["final_observation"], dtype=np.float32)
                next_obs_np[has_final_np] = final_obs_np[has_final_np]

        # Write to replay buffer
        replay_buffer.add_batch(obs_np, actions_np, rewards_np, next_obs_np, combined_dones)

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

        # Progress log every 2 seconds
        now = _time.time()
        if now - _last_log_time > 2.0:
            _last_log_time = now
            phase = "warmup" if total_steps < warmup_steps else "policy"
            mean_r = np.mean(ep_rewards[-50:]) if ep_rewards else 0.0

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

                metrics_queue.put_nowait(msg)
            except Exception:
                pass

    replay_buffer.close()
    weight_sync.close()
