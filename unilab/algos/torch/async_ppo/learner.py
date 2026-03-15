"""Learner adapter for async PPO."""

from typing import Any

from tensordict import TensorDict


class AsyncPPOLearner:
    """Adapts buffer data to PPO's RolloutStorage."""

    def __init__(self, ppo, buffer):
        self.ppo = ppo
        self.buffer = buffer

    def update(self) -> dict[str, Any]:
        """Perform PPO update using latest rollout."""
        import time

        t0 = time.perf_counter()
        rollout = self.buffer.get_latest()
        t1 = time.perf_counter()

        self._fill_storage(rollout)
        t2 = time.perf_counter()

        # Calculate env steps from this rollout
        num_steps = rollout["observations"].shape[0]
        num_envs = rollout["observations"].shape[1]
        env_steps = num_steps * num_envs

        # Mark rollout as consumed
        self.buffer.count[0] = 0

        # Recompute values with current critic to fix stale-value GAE corruption.
        # The collector uses an older copy of the critic; mixing stale st.values with
        # the current last_values in compute_returns inflates advantages and breaks training.
        import torch
        st = self.ppo.storage
        with torch.no_grad():
            obs_flat = rollout["observations"].flatten(0, 1)  # [T*N, obs_dim]
            obs_td = TensorDict({"policy": obs_flat}, device=self.ppo.device)
            fresh_values = self.ppo.critic(obs_td)  # [T*N, 1]
            st.values[:] = fresh_values.view(st.num_transitions_per_env, st.num_envs, 1)

        last_obs_td = TensorDict({"policy": rollout["last_obs"]}, device=self.ppo.device)
        self.ppo.compute_returns(last_obs_td)
        t3 = time.perf_counter()

        metrics = self.ppo.update()
        t4 = time.perf_counter()

        self.ppo.storage.clear()

        # Add env steps to metrics
        metrics["env_steps"] = env_steps
        metrics["timing/get_latest_ms"] = (t1 - t0) * 1000
        metrics["timing/fill_storage_ms"] = (t2 - t1) * 1000
        metrics["timing/compute_returns_ms"] = (t3 - t2) * 1000
        metrics["timing/ppo_update_ms"] = (t4 - t3) * 1000

        return metrics  # type: ignore[no-any-return]

    def _fill_storage(self, rollout):
        """Fill PPO storage from rollout dict."""
        st = self.ppo.storage
        st.observations[:] = rollout["observations"]
        st.actions[:] = rollout["actions"]
        st.rewards[:] = rollout["rewards"].unsqueeze(-1)
        st.dones[:] = rollout["dones"].unsqueeze(-1).byte()
        st.actions_log_prob[:] = rollout["log_probs"].unsqueeze(-1)
        st.values[:] = rollout["values"].unsqueeze(-1)
        st.step = st.num_transitions_per_env

        if "dist_params" in rollout:
            # dist_params: [num_steps, num_envs, 2*action_dim] packed as (mean, std)
            action_dim = st.actions.shape[-1]
            mean = rollout["dist_params"][..., :action_dim].clone()
            std = rollout["dist_params"][..., action_dim:].clone()
            st.distribution_params = (mean, std)
        else:
            # Fallback: recompute distribution_params using current policy
            import torch
            from tensordict import TensorDict
            with torch.no_grad():
                obs_flat = rollout["observations"].flatten(0, 1)
                obs_td = TensorDict({"policy": obs_flat}, device=self.ppo.device)
                _ = self.ppo.actor(obs_td, stochastic_output=True)
                dist_params = self.ppo.actor.distribution.params
                st.distribution_params = tuple(
                    p.view(st.num_transitions_per_env, st.num_envs, *p.shape[1:]).clone()
                    for p in dist_params
                )
