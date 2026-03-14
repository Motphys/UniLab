"""Async PPO collector worker."""

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
    metrics_queue,
    collector_device: str = "cpu",
):
    """Collect rollouts using PPO.act()."""
    from unilab.base import registry
    from unilab.ipc import SharedWeightSync

    ensure_registries()

    # Create environment
    env = registry.make(env_name, num_envs=num_envs, sim_backend="mujoco")
    obs_dim = env.observation_space.shape[0]  # type: ignore[index]
    action_dim = env.action_space.shape[0]  # type: ignore[index]

    # Build PPO (inference only)
    obs_example = torch.zeros((num_envs, obs_dim), device=collector_device)
    td_example = TensorDict({"policy": obs_example}, batch_size=num_envs)

    ppo = PPO.construct_algorithm(
        env=env,
        obs=td_example,
        cfg=rl_cfg,
        device=collector_device,
    )
    ppo.actor.eval()
    ppo.critic.eval()

    # Weight sync
    weight_sync = SharedWeightSync(weight_param_shapes, create=False, shm_name=weight_sync_name)
    sd = {**ppo.actor.state_dict(), **ppo.critic.state_dict()}
    weight_sync.read_weights_into(sd)
    ppo.actor.load_state_dict({k: v for k, v in sd.items() if k in ppo.actor.state_dict()})
    ppo.critic.load_state_dict({k: v for k, v in sd.items() if k in ppo.critic.state_dict()})
    local_version = weight_sync.version

    # Reset env
    env_indices = np.arange(num_envs, dtype=np.int32)
    try:
        _, obs_out, _ = env.reset(env_indices)  # type: ignore[attr-defined]
    except TypeError:
        obs_out, _ = env.reset()  # type: ignore[attr-defined]

    obs = torch.from_numpy(np.asarray(obs_out, dtype=np.float32)).to(collector_device)

    try:
        while not stop_event.is_set():
            # Sync weights
            if weight_sync.version > local_version:
                sd = {**ppo.actor.state_dict(), **ppo.critic.state_dict()}
                local_version = weight_sync.read_weights_into(sd)
                ppo.actor.load_state_dict(
                    {k: v for k, v in sd.items() if k in ppo.actor.state_dict()}
                )
                ppo.critic.load_state_dict(
                    {k: v for k, v in sd.items() if k in ppo.critic.state_dict()}
                )

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
                    actions = ppo.act(obs_td, stochastic_output=True)

                    # Extract from transition
                    log_probs = ppo.transition.actions_log_prob
                    values = ppo.transition.values

                    obs_buf[step] = obs
                    actions_buf[step] = actions
                    log_probs_buf[step] = log_probs.squeeze(-1)
                    values_buf[step] = values.squeeze(-1)

                    # Step env
                    state = env.step(actions.cpu().numpy())  # type: ignore[attr-defined]
                    rewards = torch.from_numpy(np.asarray(state.reward, dtype=np.float32)).to(
                        collector_device
                    )
                    dones = torch.from_numpy(np.asarray(state.terminated, dtype=np.float32)).to(
                        collector_device
                    )

                    rewards_buf[step] = rewards
                    dones_buf[step] = dones

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

    except KeyboardInterrupt:
        pass
