"""Config-driven runtime selection helpers for HORA teacher-policy RL."""

from __future__ import annotations

from typing import Any

HORA_APPO_RUNTIME_IMPL = "hora_appo"
HORA_PPO_RUNTIME_IMPL = "hora_ppo"


def resolve_hora_runtime_impl(rl_cfg: dict[str, Any]) -> str | None:
    """Return the explicit HORA runtime marker from a resolved algo config.

    Args:
        rl_cfg: Resolved algorithm config dictionary from Hydra composition.

    Returns:
        Runtime marker string when the owner YAML selected one, otherwise ``None``.
    """
    runtime_impl = rl_cfg.get("runtime_impl")
    if runtime_impl in (None, ""):
        return None
    return str(runtime_impl)


def is_hora_appo_runtime(rl_cfg: dict[str, Any]) -> bool:
    """Check whether the resolved algo config selects the HORA APPO runtime.

    Args:
        rl_cfg: Resolved algorithm config dictionary from Hydra composition.

    Returns:
        ``True`` when the config explicitly selects the HORA APPO runtime.
    """
    return resolve_hora_runtime_impl(rl_cfg) == HORA_APPO_RUNTIME_IMPL


def is_hora_ppo_runtime(rl_cfg: dict[str, Any]) -> bool:
    """Check whether the resolved algo config selects the HORA PPO runtime.

    Args:
        rl_cfg: Resolved algorithm config dictionary from Hydra composition.

    Returns:
        ``True`` when the config explicitly selects the HORA PPO runtime.
    """
    return resolve_hora_runtime_impl(rl_cfg) == HORA_PPO_RUNTIME_IMPL
