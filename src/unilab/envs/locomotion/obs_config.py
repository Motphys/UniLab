from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ObsConfig:
    """Observation layout for locomotion environments.

    ``obs_dict`` maps observation name → dimension in the order the
    fields are concatenated into the full observation vector.
    ``actor_obs`` lists the subset of keys exposed to the policy.
    ``actor_indices`` derives the corresponding flat index slice
    automatically, so callers never have to compute offsets by hand.
    """

    obs_dict: dict[str, int] = field(default_factory=dict)
    actor_obs: list[str] = field(default_factory=list)

    @property
    def total_dim(self) -> int:
        return sum(self.obs_dict.values())

    @property
    def actor_indices(self) -> list[int]:
        """Flat indices into the full obs vector for the ``actor_obs`` keys."""
        offsets: dict[str, int] = {}
        s = 0
        for k, v in self.obs_dict.items():
            offsets[k] = s
            s += v
        indices: list[int] = []
        for key in self.actor_obs:
            if key in offsets:
                start = offsets[key]
                indices.extend(range(start, start + self.obs_dict[key]))
        return indices
