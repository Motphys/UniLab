"""Probe env dims through an injected env factory (issue #1479)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from unilab.base.contracts.env import EnvFactory, get_produces_completion_receipts
from unilab.base.contracts.observations import get_obs_dims as _get_obs_dims


@dataclass(frozen=True)
class EnvContractProbe:
    """Cold-path env contract probe result."""

    obs_dim: int
    action_dim: int
    critic_obs_dim: int
    produces_completion_receipts: bool


def probe_env_contract(
    env_factory: EnvFactory,
    env_cfg_override: Mapping[str, Any] | None = None,
) -> EnvContractProbe:
    """Probe one env instance for dims and completion-receipt semantics.

    Builds a one-env probe through the injected factory, reads the env
    contract, and closes the probe again. ``env_cfg_override`` is an opaque
    mapping forwarded to the factory. ``produces_completion_receipts``
    defaults to ``True`` (fail closed) when the env does not declare it.
    """
    env = env_factory(1, env_cfg_override)
    try:
        obs_dim, critic_dim = _get_obs_dims(dict(env.obs_groups_spec))
        action_shape = env.action_space.shape
        assert action_shape is not None
        action_dim = action_shape[0]
        produces_completion_receipts = get_produces_completion_receipts(env)
    finally:
        env.close()
    return EnvContractProbe(
        obs_dim=obs_dim,
        action_dim=action_dim,
        critic_obs_dim=critic_dim,
        produces_completion_receipts=produces_completion_receipts,
    )


__all__ = ["EnvContractProbe", "probe_env_contract"]
