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
    sync_primitives: tuple,
    obs_dim: int,
    action_dim: int,
    weight_sync_name: str,
    weight_param_shapes: dict,
    metrics_queue,
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
    storage.attach_sync_primitives(*sync_primitives)
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
    actor_core = actor_cls(td_example, rl_cfg.get("obs_groups", {"actor": {"policy": obs_dim}}), "actor", action_dim, **actor_cfg)
    
    from unilab.algos.torch.appo.learner import APPOActorWrapper
    actor = APPOActorWrapper(actor_core, action_dim)
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
    def to_float32_np(x):
        if hasattr(x, "cpu"):
            x = x.cpu().numpy()
        try:
            return np.array(x, dtype=np.float32)
        except Exception:
            pass
        try:
            return np.array(x, copy=False).astype(np.float32)
        except Exception:
            return np.array(x).astype(np.float32)

    obs_np = to_float32_np(obs_out)

    total_steps = 0
    ep_rewards = []
    ep_lengths = []
    current_ep_rewards = np.zeros(num_envs, dtype=np.float32)
    current_ep_lengths = np.zeros(num_envs, dtype=np.int32)
    
    from collections import defaultdict
    ep_reward_components = defaultdict(list)

    import sys
    import time as _time
    _last_log_time = _time.time()
    
    # Collection loop
    try:
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
                    actions_torch = actor(obs_td, stochastic_output=True)
                    log_probs_torch = actor.get_output_log_prob(actions_torch)
                    actions_np = actions_torch.cpu().numpy().astype(np.float32)

                # Store in shared storage
                write_buf["obs"][:, step, :] = obs_np
                write_buf["actions"][:, step, :] = actions_np
                write_buf["log_probs"][:, step] = log_probs_torch.cpu().numpy().astype(np.float32).ravel()

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

                obs_np = to_float32_np(next_obs_raw)

                # Handle true terminal observations
                if hasattr(state, "info") and "_final_observation" in state.info:
                    has_final = state.info["_final_observation"]
                    if hasattr(has_final, "item"):
                        has_final_np = to_float32_np(has_final).astype(bool)
                    else:
                        has_final_np = np.asarray(has_final, dtype=bool)
                    if np.any(has_final_np):
                        final_obs_np = to_float32_np(state.info["final_observation"])
                        obs_np[has_final_np] = final_obs_np[has_final_np]

                # Track metrics
                total_steps += num_envs
                current_ep_rewards += reward_raw
                current_ep_lengths += 1
                
                combined_dones = np.clip(done_raw + truncated_raw, 0, 1)
                reset_mask = combined_dones > 0.5
                if np.any(reset_mask):
                    for i in range(num_envs):
                        if reset_mask[i]:
                            ep_rewards.append(float(current_ep_rewards[i]))
                            ep_lengths.append(float(current_ep_lengths[i]))
                            current_ep_rewards[i] = 0.0
                            current_ep_lengths[i] = 0
                            
                log_info = getattr(state, "info", {}).get("log", {})
                if log_info:
                    for k, v in log_info.items():
                        if k.startswith("reward/"):
                            ep_reward_components[k].append(v)
                            
                if metrics_queue is not None and total_steps % (num_envs * 10) == 0 and ep_rewards:
                    import statistics
                    try:
                        msg = {
                            "total_steps": total_steps,
                            "mean_ep_reward": statistics.mean(ep_rewards[-100:]),
                            "mean_ep_length": statistics.mean(ep_lengths[-100:]) if ep_lengths else 0.0,
                        }
                        if ep_reward_components:
                            components_mean = {}
                            for k, vals in ep_reward_components.items():
                                if vals:
                                    components_mean[k] = statistics.mean(vals)
                            msg["reward_components"] = components_mean
                            ep_reward_components.clear()
                            
                        metrics_queue.put_nowait(msg)
                    except Exception:
                        pass
                        
            # Store last obs
            write_buf["last_obs"][:] = obs_np

            # Signal data ready
            storage.signal_write_done()
    except Exception as e:
        import traceback
        import sys
        print(f"\n[APPO WORKER CRASH]: {e}\n", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        stop_event.set()
        raise e

    # Cleanup
    storage.close()
    weight_sync.close()
    env.close()
