from __future__ import annotations

from typing import Any

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


def split_obs_with_priv_info(
    obs: dict[str, np.ndarray],
    info: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Split env outputs into actor obs, critic obs, and privileged info.

    Args:
        obs: Environment observation dict that follows the UniLab env contract.
        info: Optional env info dict. When present, ``info["critic_info"]`` is
            the preferred source of privileged info for separated-observation
            tasks such as HORA.

    Returns:
        Tuple of ``(actor_obs, critic_obs, priv_info)``. ``priv_info`` is
        derived from ``info["critic_info"]`` when available, otherwise from the
        extra tail of ``critic_obs`` when that observation is wider than
        ``actor_obs``.
    """
    actor_obs, critic_obs = split_obs_dict(obs)

    priv_info: np.ndarray | None = None
    if isinstance(info, dict):
        candidate = info.get("critic_info")
        if isinstance(candidate, np.ndarray) and candidate.shape[0] == actor_obs.shape[0]:
            priv_info = candidate

    if (
        priv_info is None
        and critic_obs is not None
        and critic_obs.ndim == 2
        and actor_obs.ndim == 2
        and critic_obs.shape[0] == actor_obs.shape[0]
        and critic_obs.shape[1] > actor_obs.shape[1]
    ):
        priv_info = critic_obs[:, actor_obs.shape[1] :]

    return actor_obs, critic_obs, priv_info


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
