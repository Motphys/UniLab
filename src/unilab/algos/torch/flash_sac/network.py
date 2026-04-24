"""FlashSAC actor, critic, and temperature modules."""

from __future__ import annotations

import math
from typing import cast

import torch
import torch.nn as nn

from unilab.algos.torch.flash_sac.layers import (
    EnsembleCategoricalValue,
    EnsembleFlashSACBlock,
    EnsembleFlashSACEmbedder,
    EnsembleUnitRMSNorm,
    FlashSACBlock,
    FlashSACEmbedder,
    NormalTanhPolicy,
    UnitRMSNorm,
)


def _normalize_module_tree(module: nn.Module) -> None:
    with torch.no_grad():
        for child in module.modules():
            if child is module:
                continue
            normalize = getattr(child, "normalize_parameters", None)
            if callable(normalize):
                normalize()


class FlashSACActor(nn.Module):
    zeta_cdf: torch.Tensor
    _noise: torch.Tensor
    _repeat_count: torch.Tensor
    _repeat_target: torch.Tensor

    def __init__(
        self,
        num_blocks: int,
        input_dim: int,
        hidden_dim: int,
        action_dim: int,
        noise_zeta_mu: float = 2.0,
        noise_zeta_max: int = 16,
        device: str | torch.device = "cpu",
    ):
        super().__init__()
        self.embedder = FlashSACEmbedder(input_dim=input_dim, hidden_dim=hidden_dim)
        self.encoder = nn.ModuleList([FlashSACBlock(hidden_dim) for _ in range(num_blocks)])
        self.post_norm = UnitRMSNorm(hidden_dim)
        self.predictor = NormalTanhPolicy(hidden_dim=hidden_dim, action_dim=action_dim)
        self.noise_zeta_mu = noise_zeta_mu
        self.noise_zeta_max = noise_zeta_max
        ns = torch.arange(1, noise_zeta_max + 1, dtype=torch.float32)
        pmf = ns.pow(-noise_zeta_mu)
        self.register_buffer("zeta_cdf", torch.cumsum(pmf / pmf.sum(), dim=0))
        self.register_buffer("_noise", torch.zeros(0), persistent=False)
        self.register_buffer("_repeat_count", torch.zeros(0, dtype=torch.int32), persistent=False)
        self.register_buffer("_repeat_target", torch.zeros(0, dtype=torch.int32), persistent=False)
        self.to(device)
        self.normalize_parameters()

    def normalize_parameters(self) -> None:
        _normalize_module_tree(self)

    def _encode(self, observations: torch.Tensor, training: bool) -> torch.Tensor:
        x = self.embedder(observations, training=training)
        for block in self.encoder:
            x = block(x, training=training)
        return cast(torch.Tensor, self.post_norm(x))

    def get_mean_and_std(
        self, observations: torch.Tensor, training: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self._encode(observations, training=training)
        return self.predictor.get_mean_and_std(encoded)

    def forward(
        self, observations: torch.Tensor, training: bool
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        encoded = self._encode(observations, training=training)
        return cast(tuple[torch.Tensor, dict[str, torch.Tensor]], self.predictor(encoded))

    def as_export_module(self) -> "nn.Module":
        """Return a single-input/single-output wrapper suitable for torch.onnx.export."""
        actor = self

        class _Wrapper(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.base = actor

            def forward(self, obs: torch.Tensor) -> torch.Tensor:
                action, _ = self.base(obs, training=False)
                return cast(torch.Tensor, action)

        return _Wrapper()

    def _ensure_exploration_state(
        self, batch_size: int, action_dim: int, device: torch.device, dtype: torch.dtype
    ) -> None:
        if (
            self._noise.numel() == batch_size * action_dim
            and self._noise.device == device
            and self._noise.dtype == dtype
        ):
            return
        self._noise = torch.zeros(batch_size, action_dim, device=device, dtype=dtype)
        self._repeat_count = torch.zeros(batch_size, device=device, dtype=torch.int32)
        self._repeat_target = torch.zeros(batch_size, device=device, dtype=torch.int32)

    def _sample_repeat_targets(self, batch_size: int, device: torch.device) -> torch.Tensor:
        draws = torch.rand(batch_size, device=device)
        cdf = self.zeta_cdf.to(device)
        return cast(torch.Tensor, torch.searchsorted(cdf, draws).to(torch.int32) + 1)

    @torch.no_grad()
    def explore(
        self,
        obs: torch.Tensor,
        dones: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> torch.Tensor:
        if isinstance(dones, bool):
            deterministic = dones
            dones = None
        mean, std = self.get_mean_and_std(obs, training=False)
        if deterministic:
            return torch.tanh(mean)

        batch_size, action_dim = mean.shape
        self._ensure_exploration_state(batch_size, action_dim, mean.device, mean.dtype)

        if dones is None:
            done_mask = torch.zeros(batch_size, device=mean.device, dtype=torch.bool)
        else:
            done_mask = dones.to(device=mean.device).reshape(-1) > 0.5

        reinit = done_mask | (self._repeat_count <= 0) | (self._repeat_count >= self._repeat_target)
        if torch.any(reinit):
            new_noise = torch.randn_like(mean)
            new_target = self._sample_repeat_targets(batch_size, mean.device)
            self._noise = torch.where(reinit.unsqueeze(-1), new_noise, self._noise)
            self._repeat_target = torch.where(reinit, new_target, self._repeat_target)
            self._repeat_count = torch.where(
                reinit, torch.zeros_like(self._repeat_count), self._repeat_count
            )

        actions = torch.tanh(mean + std * self._noise)
        self._repeat_count = self._repeat_count + 1
        return actions


class FlashSACDoubleCritic(nn.Module):
    def __init__(
        self,
        num_blocks: int,
        input_dim: int,
        hidden_dim: int,
        num_bins: int,
        min_v: float,
        max_v: float,
        num_qs: int = 2,
        device: str | torch.device = "cpu",
    ):
        super().__init__()
        self.embedder = EnsembleFlashSACEmbedder(num_qs, input_dim, hidden_dim)
        self.encoder = nn.ModuleList(
            [EnsembleFlashSACBlock(num_qs, hidden_dim) for _ in range(num_blocks)]
        )
        self.post_norm = EnsembleUnitRMSNorm(num_qs, hidden_dim)
        self.predictor = EnsembleCategoricalValue(
            num_ensemble=num_qs,
            hidden_dim=hidden_dim,
            num_bins=num_bins,
            min_v=min_v,
            max_v=max_v,
        )
        self.to(device)
        self.normalize_parameters()

    def normalize_parameters(self) -> None:
        _normalize_module_tree(self)

    def forward(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        training: bool,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        x = torch.cat((observations, actions), dim=-1)
        x = x.unsqueeze(0).expand(self.predictor.logit_w.weight.shape[0], -1, -1)
        x = self.embedder(x, training=training)
        for block in self.encoder:
            x = block(x, training=training)
        x = self.post_norm(x)
        return cast(tuple[torch.Tensor, dict[str, torch.Tensor]], self.predictor(x))


class FlashSACTemperature(nn.Module):
    def __init__(self, initial_value: float = 0.01):
        super().__init__()
        self.log_temp = nn.Parameter(torch.tensor([math.log(initial_value)], dtype=torch.float32))

    def forward(self) -> torch.Tensor:
        return torch.exp(self.log_temp)
