from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from unilab.utils.obs_utils import split_obs_dict


@dataclass(frozen=True)
class TransitionBootstrapContract:
    actor_next_obs: np.ndarray
    actor_next_privileged: np.ndarray | None
    transition_next_obs: np.ndarray
    transition_next_privileged: np.ndarray | None
    terminal_mask: np.ndarray
    timeout_terminal_mask: np.ndarray


@dataclass(frozen=True)
class TerminalObservationContract:
    terminal_obs: np.ndarray | None
    terminal_privileged: np.ndarray | None
    terminal_mask: np.ndarray
    timeout_terminal_mask: np.ndarray


def patch_transition_next_obs(
    next_obs: np.ndarray,
    next_privileged: np.ndarray | None,
    final_observation: dict[str, Any] | None = None,
    done: np.ndarray | None = None,
    info: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """Patch transition next obs with final_observation without mutating actor inputs."""
    terminal_contract = resolve_terminal_observation_contract(
        next_obs_batch_size=next_obs.shape[0],
        final_observation=final_observation,
        done=done,
        info=info,
    )
    if not np.any(terminal_contract.terminal_mask) or terminal_contract.terminal_obs is None:
        return next_obs, next_privileged, np.zeros((next_obs.shape[0],), dtype=bool)

    transition_next_obs = next_obs.copy()
    transition_next_obs[terminal_contract.terminal_mask] = np.asarray(
        terminal_contract.terminal_obs, dtype=next_obs.dtype
    )[terminal_contract.terminal_mask]

    transition_next_privileged = next_privileged
    if next_privileged is not None and terminal_contract.terminal_privileged is not None:
        transition_next_privileged = next_privileged.copy()
        transition_next_privileged[terminal_contract.terminal_mask] = np.asarray(
            terminal_contract.terminal_privileged, dtype=next_privileged.dtype
        )[terminal_contract.terminal_mask]

    return transition_next_obs, transition_next_privileged, terminal_contract.terminal_mask


def resolve_transition_bootstrap_contract(
    next_obs: np.ndarray,
    next_privileged: np.ndarray | None,
    info: dict[str, Any] | None = None,
    final_observation: dict[str, Any] | None = None,
    done: np.ndarray | None = None,
    truncated: np.ndarray | None = None,
) -> TransitionBootstrapContract:
    """Resolve actor/storage observations and timeout bootstrap masks for a step."""
    transition_next_obs, transition_next_privileged, terminal_mask = patch_transition_next_obs(
        next_obs,
        next_privileged,
        final_observation=final_observation,
        done=done,
        info=info,
    )
    timeout_terminal_mask = terminal_mask
    if truncated is not None:
        timeout_terminal_mask = np.logical_and(
            terminal_mask, np.asarray(truncated, dtype=bool).ravel()
        )
    return TransitionBootstrapContract(
        actor_next_obs=next_obs,
        actor_next_privileged=next_privileged,
        transition_next_obs=transition_next_obs,
        transition_next_privileged=transition_next_privileged,
        terminal_mask=terminal_mask,
        timeout_terminal_mask=timeout_terminal_mask,
    )


def resolve_terminal_observation_contract(
    next_obs_batch_size: int,
    final_observation: dict[str, Any] | None = None,
    done: np.ndarray | None = None,
    info: dict[str, Any] | None = None,
    truncated: np.ndarray | None = None,
) -> TerminalObservationContract:
    """Resolve terminal observation facts without constructing patched next obs."""
    terminal_mask = _resolve_terminal_mask(next_obs_batch_size, done, info)
    resolved_final_observation = _resolve_final_observation(final_observation, info)

    terminal_obs: np.ndarray | None = None
    terminal_privileged: np.ndarray | None = None
    if np.any(terminal_mask) and isinstance(resolved_final_observation, dict):
        terminal_obs, terminal_privileged = split_obs_dict(resolved_final_observation)

    timeout_terminal_mask = terminal_mask
    if truncated is not None:
        timeout_terminal_mask = np.logical_and(
            terminal_mask, np.asarray(truncated, dtype=bool).ravel()
        )

    return TerminalObservationContract(
        terminal_obs=terminal_obs,
        terminal_privileged=terminal_privileged,
        terminal_mask=terminal_mask,
        timeout_terminal_mask=timeout_terminal_mask,
    )


def _resolve_final_observation(
    final_observation: dict[str, Any] | None,
    info: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if isinstance(final_observation, dict):
        return final_observation
    if isinstance(info, dict):
        final_obs = info.get("final_observation")
        if isinstance(final_obs, dict):
            return final_obs
    return None


def _resolve_terminal_mask(
    next_obs_batch_size: int,
    done: np.ndarray | None,
    info: dict[str, Any] | None,
) -> np.ndarray:
    if done is not None:
        done_mask = np.asarray(done, dtype=bool).ravel()
        if done_mask.shape == (next_obs_batch_size,):
            return done_mask
        return np.zeros((next_obs_batch_size,), dtype=bool)
    if isinstance(info, dict):
        terminal_mask = np.asarray(info.get("_final_observation"), dtype=bool)
        if terminal_mask.shape == (next_obs_batch_size,):
            return terminal_mask
    return np.zeros((next_obs_batch_size,), dtype=bool)
