"""Observation-group helpers for the env contract.

Generic helpers over the ``obs_groups_spec`` / observation-dict contract
defined in ``unilab.base.contracts.env``. Observation values are Torch tensors at
the env boundary (see ``env_contract``); the helpers stay value-agnostic so
host-side diagnostics with NumPy carriers can reuse them.
"""

from __future__ import annotations

from typing import TypeVar

T = TypeVar("T")


def split_obs_dict(obs: dict[str, T]) -> tuple[T, T]:
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
