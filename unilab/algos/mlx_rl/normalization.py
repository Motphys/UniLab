"""Running-stat normalization utilities for MLX RL."""

from __future__ import annotations

import mlx.core as mx


class EmpiricalNormalization:
    """Normalize features using running mean/std over batch axis."""

    def __init__(self, shape: int, eps: float = 1e-2) -> None:
        self.eps = float(eps)
        self.mean = mx.zeros((1, shape), dtype=mx.float32)
        self.var = mx.ones((1, shape), dtype=mx.float32)
        self.std = mx.ones((1, shape), dtype=mx.float32)
        self.count = mx.array(0.0, dtype=mx.float32)

    def __call__(self, x: mx.array) -> mx.array:
        return (x - self.mean) / (self.std + self.eps)

    def update(self, x: mx.array) -> None:
        x = mx.array(x, dtype=mx.float32)
        batch_count = mx.array(float(x.shape[0]), dtype=mx.float32)
        batch_mean = mx.mean(x, axis=0, keepdims=True)
        batch_var = mx.var(x, axis=0, keepdims=True)

        total = self.count + batch_count
        rate = batch_count / (total + 1e-8)
        delta = batch_mean - self.mean

        self.mean = self.mean + rate * delta
        self.var = self.var + rate * (batch_var - self.var + delta * (batch_mean - self.mean))
        self.std = mx.sqrt(mx.maximum(self.var, 1e-8))
        self.count = total
        mx.eval(self.mean, self.var, self.std, self.count)


class EmpiricalDiscountedVariationNormalization:
    """Reward normalization with running std of discounted returns."""

    def __init__(self, eps: float = 1e-2, gamma: float = 0.99) -> None:
        self.emp_norm = EmpiricalNormalization(shape=1, eps=eps)
        self.gamma = float(gamma)
        self.avg = None

    def __call__(self, rew: mx.array) -> mx.array:
        """Normalize reward tensor of shape [N] or [N, 1]."""
        if rew.ndim == 1:
            rew = mx.expand_dims(rew, axis=-1)
        rew = mx.array(rew, dtype=mx.float32)
        if self.avg is None:
            self.avg = rew
        else:
            self.avg = self.avg * self.gamma + rew
        self.emp_norm.update(self.avg)
        return rew / (self.emp_norm.std + self.emp_norm.eps)
