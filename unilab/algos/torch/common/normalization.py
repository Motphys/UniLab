"""Observation normalization for RL training."""

from __future__ import annotations

import torch
import torch.nn as nn


class EmpiricalNormalization(nn.Module):
    """Normalize mean and variance of observations using running statistics."""

    _mean: torch.Tensor
    _var: torch.Tensor
    _std: torch.Tensor
    count: torch.Tensor

    def __init__(self, shape, device, eps=1e-2):
        super().__init__()
        self.eps = eps
        self.device = device
        self.register_buffer("_mean", torch.zeros(shape).unsqueeze(0).to(device))
        self.register_buffer("_var", torch.ones(shape).unsqueeze(0).to(device))
        self.register_buffer("_std", torch.ones(shape).unsqueeze(0).to(device))
        self.register_buffer("count", torch.tensor(0, dtype=torch.long).to(device))

    @property
    def mean(self):
        return self._mean.squeeze(0).clone()

    @property
    def std(self):
        return self._std.squeeze(0).clone()

    @torch.no_grad()
    def forward(self, x: torch.Tensor, center: bool = True, update: bool = True) -> torch.Tensor:
        if self.training and update:
            self.update(x)
        if center:
            return torch.as_tensor((x - self._mean) / (self._std + self.eps))
        else:
            return torch.as_tensor(x / (self._std + self.eps))

    def update(self, x):
        batch_size = x.shape[0]
        batch_mean = torch.mean(x, dim=0, keepdim=True)
        batch_var = torch.var(x, dim=0, keepdim=True, unbiased=False)

        new_count = self.count + batch_size

        # Welford's online algorithm
        delta = batch_mean - self._mean
        self._mean.copy_(self._mean + delta * (batch_size / new_count))
        delta2 = batch_mean - self._mean
        m_a = self._var * self.count
        m_b = batch_var * batch_size
        M2 = m_a + m_b + delta2.pow(2) * (self.count * batch_size / new_count)
        self._var.copy_(M2 / new_count)
        self._std.copy_(self._var.sqrt())
        self.count.copy_(new_count)

    def inverse(self, y):
        return y * (self._std + self.eps) + self._mean
