from __future__ import annotations

import numpy as np


def flatten_obs_dict(obs: dict[str, np.ndarray]) -> np.ndarray:
    """Concatenate obs groups in insertion order -> flat (N, total_dim) array."""
    return np.concatenate(list(obs.values()), axis=1)


def flatten_policy_obs_dict(obs: dict[str, np.ndarray]) -> np.ndarray:
    """Build actor-policy inputs from the single actor observation group."""
    return obs["obs"]


def split_obs_dict(obs: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray | None]:
    """Split observation dict into (actor_obs, critic_obs)."""
    return obs["obs"], obs.get("critic")


def get_obs_dims(obs_groups_spec: dict[str, int]) -> tuple[int, int]:
    """Extract (actor_obs_dim, critic_obs_dim) from obs_groups_spec."""
    return obs_groups_spec.get("obs", 0), obs_groups_spec.get("critic", 0)


def get_critic_base_dim(obs_groups_spec: dict[str, int]) -> int:
    """Get critic observation dim, falling back to actor obs when absent."""
    critic_dim = obs_groups_spec.get("critic", 0)
    return critic_dim if critic_dim > 0 else obs_groups_spec.get("obs", 0)
