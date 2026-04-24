"""Utility functions for reward config handling."""

from typing import Any, cast

from omegaconf import DictConfig, OmegaConf

RewardDict = dict[str, Any]


def _to_reward_dict(value: object, *, error_message: str) -> RewardDict:
    """Convert an OmegaConf container into a plain reward dictionary."""
    resolved = OmegaConf.to_container(value, resolve=True)
    if not isinstance(resolved, dict):
        raise ValueError(error_message)
    # Some reward configs are mounted as a full `reward:` section.
    # Env config injection expects the inner reward mapping.
    if set(resolved) == {"reward"} and isinstance(resolved["reward"], dict):
        return cast(RewardDict, resolved["reward"])
    return cast(RewardDict, resolved)


def resolve_reward_dict(cfg: DictConfig) -> RewardDict:
    """Resolve the reward config from the final composed config."""
    reward_cfg = OmegaConf.select(cfg, "reward")
    if not reward_cfg:
        raise ValueError(
            "Missing 'reward' config in Hydra. Reward config must be explicitly provided."
        )

    reward_dict = _to_reward_dict(
        reward_cfg,
        error_message="Reward config must resolve to a mapping.",
    )
    if not reward_dict:
        raise ValueError(
            "Reward config resolved to empty. Please select a non-default reward override."
        )

    return reward_dict


def extract_reward_config(cfg: DictConfig) -> dict[str, RewardDict]:
    """Extract and validate reward config from Hydra config.

    Args:
        cfg: Hydra DictConfig containing reward section

    Returns:
        Dictionary with reward_config key for env_cfg_override

    Raises:
        ValueError: If reward config is missing
    """
    return {"reward_config": resolve_reward_dict(cfg)}
