from __future__ import annotations

import warnings

from unilab.base.observations import (
    flatten_obs_dict,
    flatten_policy_obs_dict,
    get_critic_base_dim,
    get_obs_dims,
    split_obs_dict,
)

warnings.warn(
    "`unilab.utils.obs_utils` is deprecated and will be removed in 0.2.0; "
    "use `unilab.base.observations` instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "flatten_obs_dict",
    "flatten_policy_obs_dict",
    "get_critic_base_dim",
    "get_obs_dims",
    "split_obs_dict",
]
