"""MLX RL base modules.

This package contains framework-level building blocks that are reused by
algorithm implementations (e.g. PPO).
"""

from .distributions import diag_gaussian_entropy, diag_gaussian_log_prob
from .mlp import MLP
from .normalization import EmpiricalDiscountedVariationNormalization, EmpiricalNormalization
from .rollout_storage import RolloutBuffer

__all__ = [
    "MLP",
    "EmpiricalNormalization",
    "EmpiricalDiscountedVariationNormalization",
    "RolloutBuffer",
    "diag_gaussian_log_prob",
    "diag_gaussian_entropy",
]
