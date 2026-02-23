"""Off-policy collector for TD3 and SAC (no Ray dependency).

Collects (obs, action, reward, next_obs, done) transitions using the current
actor policy.  Runs in a subprocess; writes to a SharedReplayBuffer.
"""

import torch
import numpy as np
import pkgutil
import importlib


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


def off_policy_collector_fn(
    stop_event,
    env_name: str,
    env_cfg_overrides: dict,
    rl_cfg: dict,
    num_envs: int,
    steps_per_env: int,
    shm_buffer_name: str,
    buffer_capacity: int,
    obs_dim: int,
    action_dim: int,
    weight_sync_name: str,
    weight_param_shapes: dict,
    collector_device: str = "cpu",
    exploration_noise: float = 0.1,
    warmup_steps: int = 5000,
    metrics_queue=None,
    algo_type: str = "sac",  # "sac" or "td3"
):
    """Entry point for the off-policy collector subprocess.

    Creates the environment + actor, collects transitions, and writes
    them to the SharedReplayBuffer via shared memory.
    """
    from unilab.algos.torch.common.async_runner import SharedReplayBuffer, SharedWeightSync
    from unilab.envs import registry
    from tensordict import TensorDict
    from rsl_rl.utils import resolve_callable

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
    from unilab.utils.rsl_rl_compat import convert_config_v3_to_v4, is_rsl_rl_v4

    cfg = dict(rl_cfg)
    if is_rsl_rl_v4():
        cfg = convert_config_v3_to_v4(cfg)

    obs_example = torch.zeros((num_envs, obs_dim), device=collector_device)
    td_example = TensorDict({"policy": obs_example}, batch_size=num_envs)

    actor_cfg = cfg["actor"].copy()
    actor_cls = resolve_callable(actor_cfg.pop("class_name"))
    actor = actor_cls(td_example, cfg["obs_groups"], "actor", action_dim, **actor_cfg)
    actor = actor.to(collector_device)
    actor.eval()

    # --- Load initial weights ---
    weight_sync.read_weights_into(dict(actor.state_dict()))
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

    # Convert obs
    if hasattr(obs_out, "__array__"):
        obs_np = np.array(obs_out, dtype=np.float32)
    else:
        obs_np = obs_out.astype(np.float32)

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
                obs_td = TensorDict({"policy": obs_torch}, batch_size=num_envs, device=collector_device)
                actions_torch = actor(obs_td)

                if algo_type == "td3":
                    # TD3: deterministic + exploration noise
                    actions_torch = torch.tanh(actions_torch)
                    noise = torch.randn_like(actions_torch) * exploration_noise
                    actions_torch = (actions_torch + noise).clamp(-1, 1)
                else:
                    # SAC: stochastic policy (actor already samples)
                    actions_torch = torch.tanh(actions_torch)

                actions_np = actions_torch.cpu().numpy().astype(np.float32)

        # Step environment
        state = env.step(actions_np)

        if hasattr(state, "obs"):
            next_obs_raw = state.obs
            reward_raw = state.reward if hasattr(state, "reward") else np.zeros(num_envs)
            done_raw = state.terminated if hasattr(state, "terminated") else np.zeros(num_envs)
        else:
            next_obs_raw = state[0]
            reward_raw = state[1] if len(state) > 1 else np.zeros(num_envs)
            done_raw = state[2] if len(state) > 2 else np.zeros(num_envs)

        # Convert to numpy
        if hasattr(next_obs_raw, "__array__"):
            next_obs_np = np.array(next_obs_raw, dtype=np.float32)
        else:
            next_obs_np = next_obs_raw.astype(np.float32)

        rewards_np = np.array(reward_raw, dtype=np.float32).ravel()
        dones_np = np.array(done_raw, dtype=np.float32).ravel()

        # Write to shared replay buffer
        replay_buffer.add_batch(obs_np, actions_np, rewards_np, next_obs_np, dones_np)

        # Track metrics
        current_ep_rewards += rewards_np
        for i in range(num_envs):
            if dones_np[i] > 0.5:
                ep_rewards.append(float(current_ep_rewards[i]))
                current_ep_rewards[i] = 0.0

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
    env.close()
