from __future__ import annotations

import numpy as np


def flatten_obs_dict(obs: dict[str, np.ndarray]) -> np.ndarray:
    """Concatenate obs groups in insertion order -> flat (N, total_dim) array."""
    return np.concatenate(list(obs.values()), axis=1)


def flatten_policy_obs_dict(obs: dict[str, np.ndarray]) -> np.ndarray:
    """Build actor-policy inputs from the single actor observation group."""
    return obs["obs"]


def split_obs_dict(obs: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Split observation dict into (actor_obs, critic_obs).

    When no separate critic group exists, critic_obs == actor_obs.
    """
    actor = obs["obs"]
    return actor, obs.get("critic", actor)


def get_obs_dims(obs_groups_spec: dict[str, int]) -> tuple[int, int]:
    """Extract (actor_obs_dim, critic_obs_dim) from obs_groups_spec.

    When no separate critic group exists, critic_obs_dim == actor_obs_dim.
    """
    obs_dim = obs_groups_spec.get("obs", 0)
    return obs_dim, obs_groups_spec.get("critic", obs_dim)


def get_critic_base_dim(obs_groups_spec: dict[str, int]) -> int:
    """Get critic observation dim, falling back to actor obs when absent."""
    critic_dim = obs_groups_spec.get("critic", 0)
    return critic_dim if critic_dim > 0 else obs_groups_spec.get("obs", 0)
