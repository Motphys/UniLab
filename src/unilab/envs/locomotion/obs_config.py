from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ObsConfig:
    """Observation layout descriptor.

    The ``obs_dict`` key order defines the concatenation order for the full
    observation. ``actor_obs`` selects which terms are exposed to actor policy.
    """

    obs_dict: dict[str, int]
    actor_obs: list[str]

    @property
    def total_dim(self) -> int:
        return int(sum(self.obs_dict.values()))

    @property
    def actor_indices(self) -> list[int]:
        indices: list[int] = []
        offset = 0
        actor_keys = set(self.actor_obs)

        for key, dim in self.obs_dict.items():
            end = offset + int(dim)
            if key in actor_keys:
                indices.extend(range(offset, end))
            offset = end

        return indices
