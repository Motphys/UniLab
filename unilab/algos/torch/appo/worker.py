"""APPO Rollout Worker — runs in a subprocess (no Ray dependency).

Collects on-policy rollouts and writes to SharedOnPolicyStorage.
"""

import torch
import numpy as np
import pkgutil
import importlib
from rsl_rl.models import MLPModel
from rsl_rl.utils import resolve_callable


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


def appo_collector_fn(
    stop_event,
    env_name: str,
    env_cfg_overrides: dict,
    rl_cfg: dict,
    num_envs: int,
    steps_per_env: int,
    shm_storage_name: str,
    obs_dim: int,
    action_dim: int,
    weight_sync_name: str,
    weight_param_shapes: dict,
    collector_device: str = "cpu",
):
    """Entry point for the APPO collector subprocess.

    Creates environment + policy, collects rollouts, writes to SharedOnPolicyStorage.
    """
    from unilab.algos.torch.common.async_runner import SharedOnPolicyStorage, SharedWeightSync
    from unilab.envs import registry
    from tensordict import TensorDict
    from unilab.utils.rsl_rl_compat import convert_config_v3_to_v4, is_rsl_rl_v4

    ensure_registries()

    # Connect to shared memory
    storage = SharedOnPolicyStorage(
        num_envs=num_envs,
        num_steps=steps_per_env,
        obs_dim=obs_dim,
        action_dim=action_dim,
        create=False,
        shm_name_prefix=shm_storage_name,
    )
    weight_sync = SharedWeightSync(
        weight_param_shapes, create=False, shm_name=weight_sync_name
    )

    # Create environment
    env = registry.make(env_name, num_envs=num_envs, sim_backend="mujoco")

    # Build actor
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

    # Load initial weights
    sd = dict(actor.state_dict())
    weight_sync.read_weights_into(sd)
    actor.load_state_dict(sd)
    local_weight_version = weight_sync.version

    # Reset environment
    try:
        import mlx.core as mx
        env_indices = mx.arange(num_envs, dtype=mx.int32)
    except ImportError:
        env_indices = np.arange(num_envs)

    try:
        _, obs_out, _ = env.reset(env_indices)
    except TypeError:
        obs_out, _ = env.reset()

    if hasattr(obs_out, "__array__"):
        obs_np = np.array(obs_out, dtype=np.float32)
    else:
        obs_np = obs_out.astype(np.float32)

    # Collection loop
    while not stop_event.is_set():
        # Check for weight updates
        if weight_sync.version > local_weight_version:
            sd = dict(actor.state_dict())
            local_weight_version = weight_sync.read_weights_into(sd)
            actor.load_state_dict(sd)

        # Collect one rollout
        write_buf = storage.write_buffer
        for step in range(steps_per_env):
            with torch.no_grad():
                obs_torch = torch.from_numpy(obs_np).to(collector_device)
                obs_td = TensorDict({"policy": obs_torch}, batch_size=num_envs, device=collector_device)
                actions_torch = actor(obs_td)
                actions_np = actions_torch.cpu().numpy().astype(np.float32)

            # Store in shared storage
            write_buf["obs"][:, step, :] = obs_np
            write_buf["actions"][:, step, :] = actions_np

            # Step environment
            state = env.step(actions_np)

            if hasattr(state, "obs"):
                next_obs_raw = state.obs
                reward_raw = state.reward if hasattr(state, "reward") else np.zeros(num_envs)
                done_raw = state.terminated if hasattr(state, "terminated") else np.zeros(num_envs)
                truncated_raw = state.truncated if hasattr(state, "truncated") else np.zeros(num_envs)
            else:
                next_obs_raw = state[0]
                reward_raw = state[1] if len(state) > 1 else np.zeros(num_envs)
                done_raw = state[2] if len(state) > 2 else np.zeros(num_envs)
                truncated_raw = state[3] if len(state) > 3 else np.zeros(num_envs)

            if hasattr(next_obs_raw, "__array__"):
                obs_np = np.array(next_obs_raw, dtype=np.float32)
            else:
                obs_np = next_obs_raw.astype(np.float32)

            write_buf["rewards"][:, step] = np.array(reward_raw, dtype=np.float32).ravel()
            write_buf["dones"][:, step] = np.array(done_raw, dtype=np.float32).ravel()
            write_buf["truncated"][:, step] = np.array(truncated_raw, dtype=np.float32).ravel()

        # Store last obs
        write_buf["last_obs"][:] = obs_np

        # Signal data ready
        storage.signal_write_done()

    # Cleanup
    storage.close()
    weight_sync.close()
    env.close()
