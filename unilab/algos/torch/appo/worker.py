"""APPO Rollout Worker — runs in a subprocess.

Collects on-policy rollouts and writes to SharedOnPolicyStorage.
"""

import sys
from collections import defaultdict
import statistics

import torch
import numpy as np
from rsl_rl.utils import resolve_callable

from unilab.utils.algo_utils import ensure_registries


def appo_collector_fn(
    stop_event,
    env_name: str,
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
    from unilab.ipc import SharedOnPolicyStorage, SharedWeightSync
    from unilab.base import registry
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
    actor_core = actor_cls(
        td_example,
        cfg.get("obs_groups", {"actor": {"policy": obs_dim}}),
        "actor",
        action_dim,
        **actor_cfg,
    )

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
    env_indices = np.arange(num_envs, dtype=np.int32)
    try:
        _, obs_out, _ = env.reset(env_indices)
    except TypeError:
        obs_out, _ = env.reset()

    def to_float32_np(x):
        if hasattr(x, "cpu"):
            x = x.cpu().numpy()
        return np.asarray(x, dtype=np.float32)

    obs_np = to_float32_np(obs_out)

    total_steps = 0
    ep_rewards = []
    ep_lengths = []
    current_ep_rewards = np.zeros(num_envs, dtype=np.float32)
    current_ep_lengths = np.zeros(num_envs, dtype=np.int32)
    ep_reward_components = defaultdict(list)

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
                with torch.no_grad():
                    obs_torch = torch.from_numpy(obs_np).to(collector_device)
                    obs_td = TensorDict(
                        {"policy": obs_torch}, batch_size=num_envs, device=collector_device
                    )
                    actions_torch = actor(obs_td, stochastic_output=True)
                    log_probs_torch = actor.get_output_log_prob(actions_torch)
                    actions_np = actions_torch.cpu().numpy().astype(np.float32)

                write_buf["obs"][:, step, :] = obs_np
                write_buf["actions"][:, step, :] = actions_np
                write_buf["log_probs"][:, step] = (
                    log_probs_torch.cpu().numpy().astype(np.float32).ravel()
                )

                state = env.step(actions_np)

                next_obs_raw = state.obs
                reward_raw = np.asarray(state.reward, dtype=np.float32).ravel()
                done_raw = np.asarray(state.terminated, dtype=np.float32).ravel()
                truncated_raw = np.asarray(state.truncated, dtype=np.float32).ravel()

                write_buf["rewards"][:, step] = reward_raw
                write_buf["dones"][:, step] = done_raw
                write_buf["truncated"][:, step] = truncated_raw

                obs_np = to_float32_np(next_obs_raw)

                # Bootstrap from true terminal obs so next rollout starts cleanly
                if "_final_observation" in state.info:
                    has_final = np.asarray(state.info["_final_observation"], dtype=bool)
                    if np.any(has_final):
                        obs_np[has_final] = to_float32_np(state.info["final_observation"])[has_final]

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

                log_info = state.info.get("log", {})
                for k, v in log_info.items():
                    if k.startswith("reward/"):
                        ep_reward_components[k].append(v)

                if metrics_queue is not None and total_steps % (num_envs * 10) == 0 and ep_rewards:
                    try:
                        msg = {
                            "total_steps": total_steps,
                            "mean_ep_reward": statistics.mean(ep_rewards[-100:]),
                            "mean_ep_length": statistics.mean(ep_lengths[-100:]) if ep_lengths else 0.0,
                        }
                        if ep_reward_components:
                            msg["reward_components"] = {
                                k: statistics.mean(v) for k, v in ep_reward_components.items() if v
                            }
                            ep_reward_components.clear()
                        metrics_queue.put_nowait(msg)
                    except Exception:
                        pass

            write_buf["last_obs"][:] = obs_np
            storage.signal_write_done()

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
