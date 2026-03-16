"""MLP module used by MLX RL algorithms."""

from __future__ import annotations

import math
from typing import Sequence

import mlx.core as mx
import mlx.nn as nn

from .activations import get_activation


class MLP(nn.Module):
    """Simple feed-forward MLP with configurable activations."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Sequence[int],
        activation: str = "elu",
        last_activation: str | None = None,
    ) -> None:
        super().__init__()
        dims = [int(input_dim)] + [int(h) for h in hidden_dims] + [int(output_dim)]
        self.layers = [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        self.activation = get_activation(activation)
        self.last_activation = (
            get_activation(last_activation) if last_activation is not None else None
        )

    def __call__(self, x: mx.array) -> mx.array:
        for idx, layer in enumerate(self.layers):
            x = layer(x)
            is_last = idx == (len(self.layers) - 1)
            if not is_last:
                x = self.activation(x)
            elif self.last_activation is not None:
                x = self.last_activation(x)
        return x

    def init_orthogonal(
        self, hidden_gain: float = math.sqrt(2.0), output_gain: float = 1.0
    ) -> None:
        """Orthogonally initialize linear layers with separate output gain."""
        num_layers = len(self.layers)
        for idx, layer in enumerate(self.layers):
            gain = output_gain if idx == (num_layers - 1) else hidden_gain
            layer.weight = nn.init.orthogonal(gain=gain)(layer.weight)
            layer.bias = mx.zeros_like(layer.bias)
