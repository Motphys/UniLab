"""Activation helpers for MLX models."""

from __future__ import annotations

from typing import Callable

import mlx.core as mx


def get_activation(name: str | None) -> Callable[[mx.array], mx.array]:
    """Resolve a string activation name to a callable."""
    if name is None:
        return lambda x: x

    name = name.lower()
    if name == "relu":
        return lambda x: mx.maximum(x, 0.0)
    if name == "elu":
        return lambda x: mx.where(x > 0.0, x, mx.exp(x) - 1.0)
    if name == "tanh":
        return mx.tanh
    if name == "sigmoid":
        return mx.sigmoid
    if name == "swish":
        return lambda x: x * mx.sigmoid(x)
    if name == "identity":
        return lambda x: x
    raise ValueError(f"Unsupported activation: {name}")
