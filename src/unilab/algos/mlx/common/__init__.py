"""MLX RL base modules.

This package contains framework-level building blocks that are reused by
algorithm implementations (e.g. PPO).
"""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "MLP": (".mlp", "MLP"),
    "EmpiricalNormalization": (".normalization", "EmpiricalNormalization"),
    "EmpiricalDiscountedVariationNormalization": (
        ".normalization",
        "EmpiricalDiscountedVariationNormalization",
    ),
    "RolloutBuffer": (".rollout_storage", "RolloutBuffer"),
    "diag_gaussian_log_prob": (".distributions", "diag_gaussian_log_prob"),
    "diag_gaussian_entropy": (".distributions", "diag_gaussian_entropy"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    value = getattr(import_module(module_name, __name__), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
