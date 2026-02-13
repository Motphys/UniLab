"""Distribution utilities for RL policies."""

from __future__ import annotations

import math

import mlx.core as mx


def diag_gaussian_log_prob(actions: mx.array, mean: mx.array, log_std: mx.array) -> mx.array:
    """Log-probability under a diagonal Gaussian."""
    var = mx.exp(2.0 * log_std)
    log_probs = -0.5 * (((actions - mean) ** 2) / var + 2.0 * log_std + math.log(2.0 * math.pi))
    return mx.sum(log_probs, axis=-1)


def diag_gaussian_entropy(log_std: mx.array) -> mx.array:
    """Entropy of a diagonal Gaussian."""
    return mx.sum(log_std + 0.5 * (1.0 + math.log(2.0 * math.pi)), axis=-1)
