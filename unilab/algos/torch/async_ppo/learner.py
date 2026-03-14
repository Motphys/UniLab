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
