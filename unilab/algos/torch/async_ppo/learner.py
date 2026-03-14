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
        rollout = self.buffer.get_latest()
        self._fill_storage(rollout)

        # Mark rollout as consumed
        self.buffer.count[0] = 0

        last_obs_td = TensorDict({"policy": rollout["last_obs"]}, device=self.ppo.device)
        self.ppo.compute_returns(last_obs_td)

        metrics = self.ppo.update()
        self.ppo.storage.clear()

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

        # Initialize distribution_params by re-running actor forward pass
        import torch
        from tensordict import TensorDict
        with torch.no_grad():
            obs_flat = rollout["observations"].flatten(0, 1)
            obs_td = TensorDict({"policy": obs_flat}, device=self.ppo.device)

            # Get MLP output and update distribution
            latent = self.ppo.actor.get_latent(obs_td)
            mlp_output = self.ppo.actor.mlp(latent)
            self.ppo.actor.distribution.update(mlp_output)

            # Get distribution params and clone to avoid view issues
            dist_params = self.ppo.actor.distribution.params
            st.distribution_params = tuple(
                p.view(st.num_transitions_per_env, st.num_envs, *p.shape[1:]).clone()
                for p in dist_params
            )
