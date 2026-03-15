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
