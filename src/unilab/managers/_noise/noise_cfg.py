# Derived from mujocolab/mjlab v1.6.0 (0fb8a681), src/mjlab/utils/noise/noise_cfg.py.
# Copyright 2025, The mjlab Developers.
# Modified by UniLab for Torch-only Manager noise; licensed under Apache-2.0.
from __future__ import annotations

import abc
import math
from dataclasses import dataclass
from typing import ClassVar, Literal

import torch
from typing_extensions import override

from unilab.managers._noise import noise_model

# Type alias for noise parameters: scalar or per-dimension values.
NoiseParam = float | tuple[float, ...]

SegmentRange = tuple[float, float]


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
class ConstantNoiseCfg(NoiseCfg):
    bias: NoiseParam = 0.0

    @override
    def apply(
        self,
        data: torch.Tensor,
        *,
        torch_rng: torch.Generator | None = None,
    ) -> torch.Tensor:
        del torch_rng
        if not isinstance(data, torch.Tensor):
            raise TypeError(f"Noise data must be torch.Tensor, got {type(data).__name__}")
        bias = self._as_torch(self.bias, dtype=data.dtype, device=data.device)
        if self.operation == "add":
            return data + bias
        if self.operation == "scale":
            return data * bias
        if self.operation == "abs":
            return torch.zeros_like(data) + bias
        raise ValueError(f"Unsupported noise operation: {self.operation}")


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


@dataclass(kw_only=True)
class SegmentwiseUniformNoiseCfg(NoiseCfg):
    """Independent additive uniform noise with a bound for each final column."""

    ranges: tuple[SegmentRange, ...]
    _lower_torch: torch.Tensor | None = None
    _upper_torch: torch.Tensor | None = None

    def __post_init__(self):
        self.ranges = tuple((float(lower), float(upper)) for lower, upper in self.ranges)
        if not self.ranges:
            raise ValueError("SegmentwiseUniformNoiseCfg requires at least one range")
        if any(lower > upper for lower, upper in self.ranges):
            raise ValueError(
                f"Each SegmentwiseUniformNoiseCfg range must satisfy min <= max; got {self.ranges}"
            )

    def _bounds(
        self, *, dtype: torch.dtype, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            self._lower_torch is not None
            and self._upper_torch is not None
            and self._lower_torch.device == device
            and self._lower_torch.dtype == dtype
            and self._upper_torch.device == device
            and self._upper_torch.dtype == dtype
        ):
            return self._lower_torch, self._upper_torch
        self._lower_torch = torch.as_tensor(
            [range[0] for range in self.ranges], dtype=dtype, device=device
        )
        self._upper_torch = torch.as_tensor(
            [range[1] for range in self.ranges], dtype=dtype, device=device
        )
        return self._lower_torch, self._upper_torch

    @override
    def apply(
        self,
        data: torch.Tensor,
        *,
        torch_rng: torch.Generator | None = None,
    ) -> torch.Tensor:
        if not isinstance(data, torch.Tensor):
            raise TypeError(f"Noise data must be torch.Tensor, got {type(data).__name__}")
        if data.ndim != 2 or data.shape[-1] != len(self.ranges):
            raise ValueError(
                "SegmentwiseUniformNoiseCfg expected a two-dimensional carrier with "
                f"{len(self.ranges)} final columns; got {tuple(data.shape)}"
            )
        generator = self._require_generator(torch_rng)
        if generator.device != data.device:
            raise ValueError("Torch noise generator and observation device do not match.")
        lower, upper = self._bounds(dtype=data.dtype, device=data.device)
        unit = torch.rand(
            tuple(data.shape), dtype=data.dtype, device=data.device, generator=generator
        )
        return self._apply_noise(data, unit * (upper - lower) + lower)


@dataclass
class GaussianNoiseCfg(NoiseCfg):
    mean: NoiseParam = 0.0
    std: NoiseParam = 1.0
    # Optional symmetric clamp on the standard-normal draw, in units of sigma
    # (e.g. clamp=3.0 bounds the additive noise to ±3*std). None keeps the
    # unbounded Gaussian.
    clamp: float | None = None

    def __post_init__(self):
        if isinstance(self.std, float) and self.std <= 0:
            raise ValueError(f"std ({self.std}) must be positive")
        if self.clamp is not None:
            if isinstance(self.clamp, bool) or not isinstance(self.clamp, (int, float)):
                raise TypeError(f"clamp ({self.clamp!r}) must be a real number or None")
            if not math.isfinite(self.clamp) or self.clamp <= 0:
                raise ValueError(f"clamp ({self.clamp}) must be finite and positive")

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
        mean = self._as_torch(self.mean, dtype=data.dtype, device=data.device)
        std = self._as_torch(self.std, dtype=data.dtype, device=data.device)
        unit = torch.randn(
            tuple(data.shape), dtype=data.dtype, device=data.device, generator=generator
        )
        if self.clamp is not None:
            unit.clamp_(min=-self.clamp, max=self.clamp)
        return self._apply_noise(data, mean + std * unit)


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
