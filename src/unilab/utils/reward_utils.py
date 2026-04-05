"""Utility functions for reward config handling."""

from omegaconf import DictConfig, OmegaConf


def resolve_reward_dict(cfg: DictConfig) -> dict:
    """Resolve the reward config, preferring backend-specific overrides when present."""
    sim_backend = OmegaConf.select(cfg, "training.sim_backend")
    if sim_backend:
        backend_reward = OmegaConf.select(cfg, f"reward_{sim_backend}")
        if backend_reward:
            reward_dict = OmegaConf.to_container(backend_reward, resolve=True)
            if reward_dict:
                return reward_dict

    reward_cfg = OmegaConf.select(cfg, "reward")
    if not reward_cfg:
        raise ValueError(
            "Missing 'reward' config in Hydra. Reward config must be explicitly provided."
        )

    reward_dict = OmegaConf.to_container(reward_cfg, resolve=True)
    if not reward_dict:
        raise ValueError(
            "Reward config resolved to empty. Please select a non-default reward override."
        )

    return reward_dict


def extract_reward_config(cfg: DictConfig) -> dict:
    """Extract and validate reward config from Hydra config.

    Args:
        cfg: Hydra DictConfig containing reward section

    Returns:
        Dictionary with reward_config key for env_cfg_override

    Raises:
        ValueError: If reward config is missing
    """
    return {"reward_config": resolve_reward_dict(cfg)}
