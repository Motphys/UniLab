from __future__ import annotations

import numpy as np


def flatten_obs_dict(obs: dict[str, np.ndarray]) -> np.ndarray:
    """Concatenate obs groups in insertion order -> flat (N, total_dim) array."""
    return np.concatenate(list(obs.values()), axis=1)


def split_obs_dict(obs: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray | None]:
    """Split obs dict into (obs, privileged).

    Args:
        obs: Dict with "obs" key (required) and optional "privileged" key

    Returns:
        obs_arr: (N, obs_dim) array from "obs" key
        privileged_arr: (N, priv_dim) array from "privileged" key, or None if missing
    """
    obs_arr = obs["obs"]
    privileged_arr = obs.get("privileged", None)
    return obs_arr, privileged_arr


def split_obs_dict_with_critic(
    obs: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Split obs dict into (obs, privileged, critic).

    Args:
        obs: Dict with "obs" key (required) and optional "privileged" / "critic" keys

    Returns:
        obs_arr: (N, obs_dim) array from "obs" key
        privileged_arr: (N, priv_dim) array from "privileged" key, or None if missing
        critic_arr: (N, critic_dim) array from "critic" key, or None if missing
    """
    obs_arr = obs["obs"]
    privileged_arr = obs.get("privileged", None)
    critic_arr = obs.get("critic", None)
    return obs_arr, privileged_arr, critic_arr


def get_obs_dims(obs_groups_spec: dict[str, int]) -> tuple[int, int]:
    """Extract obs_dim and privileged_dim from obs_groups_spec.

    Args:
        obs_groups_spec: Dict mapping group names to dimensions

    Returns:
        obs_dim: Dimension of "obs" group
        privileged_dim: Dimension of "privileged" group (0 if missing)
    """
    obs_dim = obs_groups_spec.get("obs", 0)
    privileged_dim = obs_groups_spec.get("privileged", 0)
    return obs_dim, privileged_dim


def get_obs_dims_with_critic(obs_groups_spec: dict[str, int]) -> tuple[int, int, int]:
    """Extract (obs_dim, privileged_dim, critic_dim) from obs_groups_spec.

    Args:
        obs_groups_spec: Dict mapping group names to dimensions

    Returns:
        obs_dim: Dimension of "obs" group
        privileged_dim: Dimension of "privileged" group (0 if missing)
        critic_dim: Dimension of "critic" group (0 if missing)
    """
    obs_dim = obs_groups_spec.get("obs", 0)
    privileged_dim = obs_groups_spec.get("privileged", 0)
    critic_dim = obs_groups_spec.get("critic", 0)
    return obs_dim, privileged_dim, critic_dim
