from __future__ import annotations

import warnings

from unilab.algos.torch.rsl_rl.vec_env_wrapper import RslRlVecEnvWrapper

warnings.warn(
    "`unilab.utils.rsl_rl_vec_env_wrapper` is deprecated and will be removed in 0.2.0; "
    "use `unilab.algos.torch.rsl_rl.vec_env_wrapper` instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["RslRlVecEnvWrapper"]
