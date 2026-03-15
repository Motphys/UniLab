"""Base collector class for shared functionality."""

from collections import defaultdict

import numpy as np

from unilab.ipc import SharedWeightSync


class BaseCollector:
    """Base class for collectors with common weight sync and episode tracking."""

    def __init__(
        self,
        env_name: str,
        num_envs: int,
        weight_sync_name: str,
        weight_sync_lock,
        weight_param_shapes: dict,
        metrics_queue,
        stop_event,
    ):
        self.env_name = env_name
        self.num_envs = num_envs
        self.metrics_queue = metrics_queue
        self.stop_event = stop_event

        # Weight sync
        self.weight_sync = SharedWeightSync(
            weight_param_shapes, create=False, shm_name=weight_sync_name, lock=weight_sync_lock
        )
        self.local_weight_version = 0

        # Episode tracking
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_rewards = np.zeros(num_envs, dtype=np.float32)
        self.current_episode_lengths = np.zeros(num_envs, dtype=np.int32)
        self.ep_reward_components = defaultdict(list)

        # Timing
        self.timing_accum_ms = defaultdict(float)
        self.timing_count = 0

    def sync_weights_if_needed(self):
        """Check and sync weights if updated."""
        if self.weight_sync.version > self.local_weight_version:
            sd = self._get_state_dict_template()
            self.local_weight_version = self.weight_sync.read_weights_into(sd)
            self._load_state_dict(sd)

    def track_episode(self, rewards, dones, state=None):
        """Track episode statistics."""
        self.current_episode_rewards += rewards
        self.current_episode_lengths += 1

        for i in range(self.num_envs):
            if dones[i] > 0:
                self.episode_rewards.append(float(self.current_episode_rewards[i]))
                self.episode_lengths.append(int(self.current_episode_lengths[i]))
                self.current_episode_rewards[i] = 0
                self.current_episode_lengths[i] = 0

    # Abstract methods
    def _get_state_dict_template(self) -> dict:
        raise NotImplementedError

    def _load_state_dict(self, sd: dict):
        raise NotImplementedError
