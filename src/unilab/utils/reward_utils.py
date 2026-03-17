"""Utility functions for reward config handling."""

from omegaconf import DictConfig, OmegaConf


def extract_reward_config(cfg: DictConfig) -> dict:
    """Extract and validate reward config from Hydra config.

    Args:
        cfg: Hydra DictConfig containing reward section

    Returns:
        Dictionary with reward_config key for env_cfg_override

    Raises:
        ValueError: If reward config is missing
    """
    if not hasattr(cfg, "reward") or not cfg.reward:
        raise ValueError("Missing 'reward' config in Hydra. Reward config must be explicitly provided.")

    reward_dict = OmegaConf.to_container(cfg.reward, resolve=True)
    return {"reward_config": reward_dict}
