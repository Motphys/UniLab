# Derived from mujocolab/mjlab v1.6.0 (0fb8a681), src/mjlab/utils/noise/noise_cfg.py.
# Copyright 2025, The mjlab Developers.
# Modified by UniLab for Torch-only Manager noise; licensed under Apache-2.0.
from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import ClassVar, Literal

import torch
from typing_extensions import override

from unilab.managers._noise import noise_model

# Type alias for noise parameters: scalar or per-dimension values.
NoiseParam = float | tuple[float, ...]


@dataclass(kw_only=True)
class NoiseCfg(abc.ABC):
    """Base configuration for Torch observation noise."""

    operation: Literal["add", "scale", "abs"] = "add"

    @abc.abstractmethod
    def apply(
        self,
        data: torch.Tensor,
        *,
        torch_rng: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Apply noise to a Torch carrier on its existing device."""

    @staticmethod
    def _as_torch(
        value: NoiseParam,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        return torch.as_tensor(value, dtype=dtype, device=device)

    @staticmethod
    def _require_generator(torch_rng: torch.Generator | None) -> torch.Generator:
        if torch_rng is None:
            raise ValueError("noise sampling requires an env-owned Torch generator")
        return torch_rng

    def _apply_noise(self, data: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        if self.operation == "add":
            return data + noise
        if self.operation == "scale":
            return data * noise
        if self.operation == "abs":
            return noise
        raise ValueError(f"Unsupported noise operation: {self.operation}")

    def _cached_torch(
        self,
        value: NoiseParam,
        *,
        dtype: torch.dtype,
        device: torch.device,
        cache_name: str,
    ) -> torch.Tensor:
        cached = getattr(self, cache_name, None)
        if cached is not None and cached.device == device and cached.dtype == dtype:
            return cached
        cached = self._as_torch(value, dtype=dtype, device=device)
        setattr(self, cache_name, cached)
        return cached


@dataclass
class UniformNoiseCfg(NoiseCfg):
    n_min: NoiseParam = -1.0
    n_max: NoiseParam = 1.0

    def __post_init__(self):
        if isinstance(self.n_min, float) and isinstance(self.n_max, float):
            if self.n_min >= self.n_max:
                raise ValueError(f"n_min ({self.n_min}) must be less than n_max ({self.n_max})")

    @override
    def apply(
        self,
        data: torch.Tensor,
        *,
        torch_rng: torch.Generator | None = None,
    ) -> torch.Tensor:
        if not isinstance(data, torch.Tensor):
            raise TypeError(f"Noise data must be torch.Tensor, got {type(data).__name__}")
        generator = self._require_generator(torch_rng)
        if generator.device != data.device:
            raise ValueError("Torch noise generator and observation device do not match.")
        n_min = self._cached_torch(
            self.n_min, dtype=data.dtype, device=data.device, cache_name="_n_min_torch"
        )
        n_max = self._cached_torch(
            self.n_max, dtype=data.dtype, device=data.device, cache_name="_n_max_torch"
        )
        unit = torch.rand(
            tuple(data.shape), dtype=data.dtype, device=data.device, generator=generator
        )
        return self._apply_noise(data, unit * (n_max - n_min) + n_min)


##
# Noise models.
##


@dataclass(kw_only=True)
class NoiseModelCfg:
    """Configuration for a noise model."""

    noise_cfg: NoiseCfg

    class_type: ClassVar[type[noise_model.NoiseModel]] = noise_model.NoiseModel

    def __init_subclass__(cls, class_type: type[noise_model.NoiseModel]):
        cls.class_type = class_type


@dataclass(kw_only=True)
class NoiseModelWithAdditiveBiasCfg(
    NoiseModelCfg, class_type=noise_model.NoiseModelWithAdditiveBias
):
    """Configuration for an additive Gaussian noise with bias model."""

    bias_noise_cfg: NoiseCfg | None = None
    sample_bias_per_component: bool = True

    def __post_init__(self):
        if self.bias_noise_cfg is None:
            raise ValueError("bias_noise_cfg must be specified for NoiseModelWithAdditiveBiasCfg")
