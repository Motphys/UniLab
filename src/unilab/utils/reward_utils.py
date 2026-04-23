from __future__ import annotations

import warnings

from unilab.config.reward import RewardDict, extract_reward_config, resolve_reward_dict

warnings.warn(
    "`unilab.utils.reward_utils` is deprecated and will be removed in 0.2.0; "
    "use `unilab.config.reward` instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["RewardDict", "extract_reward_config", "resolve_reward_dict"]
