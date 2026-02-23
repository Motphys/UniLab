"""Off-policy collector for SAC and TD3 (no Ray dependency).

Collects (obs, action, reward, next_obs, done) transitions using the current
actor policy.  Runs in a subprocess; writes to a SharedReplayBuffer.
"""

import torch
import numpy as np
import pkgutil
import importlib


def _to_numpy_f32(x) -> np.ndarray:
    """Convert mlx array / numpy array / scalar to numpy float32."""
    return np.array(x, copy=False).astype(np.float32)


# Ensure all environment modules are imported so they are registered
def ensure_registries():
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

    try:
        import unilab.envs.locomotion.walking

        package = unilab.envs.locomotion.walking
        if hasattr(package, "__path__"):
            for _, name, ispkg in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
                try:
                    importlib.import_module(name)
                except Exception:
                    pass
    except ImportError:
        pass


def _build_actor(algo_type, obs_dim, action_dim, actor_hidden_dim, use_layer_norm, collector_device):
    """Build the correct actor model based on algorithm type."""
    if algo_type == "sac":
        from unilab.algos.torch.fast_sac.learner import SACActor
        actor = SACActor(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=actor_hidden_dim,
            use_layer_norm=use_layer_norm,
            device=collector_device,
        )
    elif algo_type == "td3":
        from unilab.algos.torch.fast_td3.learner import TD3Actor
        actor = TD3Actor(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=actor_hidden_dim,
            use_layer_norm=use_layer_norm,
            device=collector_device,
        )
    else:
        raise ValueError(f"Unknown algo_type: {algo_type}")
    return actor


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
    **kwargs,
):
    """Entry point for the off-policy collector subprocess.

    Creates the environment + actor, collects transitions, and writes
    them to the SharedReplayBuffer via shared memory.
    """
    from unilab.algos.torch.common.async_runner import SharedReplayBuffer, SharedWeightSync
    from unilab.envs import registry

    ensure_registries()

    # --- Connect to shared memory ---
    replay_buffer = SharedReplayBuffer(
        buffer_capacity, obs_dim, action_dim, create=False, shm_name=shm_buffer_name
    )
    weight_sync = SharedWeightSync(
        weight_param_shapes, create=False, shm_name=weight_sync_name
    )

    # --- Create environment ---
    env = registry.make(env_name, num_envs=num_envs, sim_backend="mujoco")

    # --- Build actor model ---
    actor = _build_actor(algo_type, obs_dim, action_dim, actor_hidden_dim, use_layer_norm, collector_device)
    actor.eval()

    # --- Load initial weights from shared memory ---
    sd = dict(actor.state_dict())
    weight_sync.read_weights_into(sd)
    actor.load_state_dict(sd)
    local_weight_version = weight_sync.version

    # --- Reset environment ---
    try:
        import mlx.core as mx
        env_indices = mx.arange(num_envs, dtype=mx.int32)
    except ImportError:
        env_indices = np.arange(num_envs)

    try:
        _, obs_out, _ = env.reset(env_indices)
    except TypeError:
        obs_out, _ = env.reset()

    obs_np = _to_numpy_f32(obs_out)

    total_steps = 0
    ep_rewards = []
    current_ep_rewards = np.zeros(num_envs, dtype=np.float32)

    # --- Collection loop ---
    while not stop_event.is_set():
        # Check for weight updates
        if weight_sync.version > local_weight_version:
            sd = dict(actor.state_dict())
            local_weight_version = weight_sync.read_weights_into(sd)
            actor.load_state_dict(sd)

        # Select action
        with torch.no_grad():
            if total_steps < warmup_steps:
                # Random exploration during warmup
                actions_np = np.random.uniform(-1, 1, (num_envs, action_dim)).astype(np.float32)
            else:
                obs_torch = torch.from_numpy(obs_np).to(collector_device)

                if algo_type == "sac":
                    # SAC: stochastic actor — use explore()
                    actions_torch = actor.explore(obs_torch)
                elif algo_type == "td3":
                    # TD3: deterministic + exploration noise
                    actions_torch = actor(obs_torch)
                    noise = torch.randn_like(actions_torch) * exploration_noise
                    actions_torch = (actions_torch + noise).clamp(-1, 1)
                else:
                    actions_torch = torch.zeros((num_envs, action_dim), device=collector_device)

                actions_np = actions_torch.cpu().numpy().astype(np.float32)

        # Step environment (convert actions to mlx if needed)
        try:
            import mlx.core as mx
            actions_mx = mx.array(actions_np)
            ctrl = env.apply_action(actions_mx, env._state)
            state = env._step_physics(ctrl)
            state = env.update_state(state)
            env._state = state
        except Exception:
            # Fallback: use env.step if available
            state = env.step(actions_np)

        # Extract observations, rewards, dones
        if hasattr(state, "obs") and state.obs is not None:
            next_obs_raw = state.obs
            reward_raw = state.reward if hasattr(state, "reward") and state.reward is not None else np.zeros(num_envs)
            done_raw = state.terminated if hasattr(state, "terminated") and state.terminated is not None else np.zeros(num_envs)
            truncated_raw = state.truncated if hasattr(state, "truncated") and state.truncated is not None else np.zeros(num_envs)
        elif isinstance(state, tuple):
            next_obs_raw = state[0]
            reward_raw = state[1] if len(state) > 1 else np.zeros(num_envs)
            done_raw = state[2] if len(state) > 2 else np.zeros(num_envs)
            truncated_raw = state[3] if len(state) > 3 else np.zeros(num_envs)
        else:
            continue

        # Convert to numpy (handles both mlx and numpy)
        next_obs_np = _to_numpy_f32(next_obs_raw)
        rewards_np = _to_numpy_f32(reward_raw).ravel()
        dones_np = _to_numpy_f32(done_raw).ravel()
        truncated_np = _to_numpy_f32(truncated_raw).ravel()

        # Combined done = terminated | truncated
        combined_dones = np.clip(dones_np + truncated_np, 0, 1)

        # Write to shared replay buffer
        replay_buffer.add_batch(obs_np, actions_np, rewards_np, next_obs_np, combined_dones)

        # Track episode rewards
        current_ep_rewards += rewards_np
        reset_mask = combined_dones > 0.5

        # Handle resets
        if np.any(reset_mask):
            for i in range(num_envs):
                if reset_mask[i]:
                    ep_rewards.append(float(current_ep_rewards[i]))
                    current_ep_rewards[i] = 0.0

            # Reset terminated environments
            try:
                import mlx.core as mx
                reset_indices = mx.array(np.where(reset_mask)[0], dtype=mx.int32)
                if len(reset_indices) > 0:
                    new_physics, new_obs, new_info = env.reset(reset_indices)
                    reset_idx_np = np.where(reset_mask)[0]
                    next_obs_np[reset_idx_np] = _to_numpy_f32(new_obs)
            except Exception:
                pass

        total_steps += num_envs

        # Send metrics periodically
        if metrics_queue is not None and total_steps % (num_envs * 10) == 0 and ep_rewards:
            import statistics
            try:
                metrics_queue.put_nowait({
                    "total_steps": total_steps,
                    "mean_ep_reward": statistics.mean(ep_rewards[-100:]),
                    "buffer_size": replay_buffer.size,
                })
            except Exception:
                pass

        obs_np = next_obs_np

    # Cleanup
    replay_buffer.close()
    weight_sync.close()
