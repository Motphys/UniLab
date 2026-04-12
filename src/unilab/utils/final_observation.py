from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from unilab.utils.obs_utils import split_obs_dict


@dataclass(frozen=True)
class TransitionBootstrapContract:
    actor_next_obs: np.ndarray
    actor_next_privileged: np.ndarray | None
    storage_next_obs: np.ndarray
    storage_next_privileged: np.ndarray | None
    final_mask: np.ndarray
    timeout_final_mask: np.ndarray


def patch_transition_next_obs(
    next_obs: np.ndarray,
    next_privileged: np.ndarray | None,
    info: dict[str, Any] | None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """Patch transition next obs with final_observation without mutating actor inputs."""
    if not isinstance(info, dict):
        return next_obs, next_privileged, np.zeros((next_obs.shape[0],), dtype=bool)

    final_mask = np.asarray(info.get("_final_observation"), dtype=bool)
    final_obs_dict = info.get("final_observation")
    if (
        final_mask.shape != (next_obs.shape[0],)
        or not np.any(final_mask)
        or not isinstance(final_obs_dict, dict)
    ):
        return next_obs, next_privileged, np.zeros((next_obs.shape[0],), dtype=bool)

    final_obs, final_privileged = split_obs_dict(final_obs_dict)
    storage_next_obs = next_obs.copy()
    storage_next_obs[final_mask] = np.asarray(final_obs, dtype=next_obs.dtype)[final_mask]

    storage_next_privileged = next_privileged
    if next_privileged is not None and final_privileged is not None:
        storage_next_privileged = next_privileged.copy()
        storage_next_privileged[final_mask] = np.asarray(
            final_privileged, dtype=next_privileged.dtype
        )[final_mask]

    return storage_next_obs, storage_next_privileged, final_mask


def resolve_transition_bootstrap_contract(
    next_obs: np.ndarray,
    next_privileged: np.ndarray | None,
    info: dict[str, Any] | None,
    truncated: np.ndarray | None = None,
) -> TransitionBootstrapContract:
    """Resolve actor/storage observations and timeout bootstrap masks for a step."""
    storage_next_obs, storage_next_privileged, final_mask = patch_transition_next_obs(
        next_obs, next_privileged, info
    )
    timeout_final_mask = final_mask
    if truncated is not None:
        timeout_final_mask = np.logical_and(final_mask, np.asarray(truncated, dtype=bool).ravel())
    return TransitionBootstrapContract(
        actor_next_obs=next_obs,
        actor_next_privileged=next_privileged,
        storage_next_obs=storage_next_obs,
        storage_next_privileged=storage_next_privileged,
        final_mask=final_mask,
        timeout_final_mask=timeout_final_mask,
    )
