# Derived from mujocolab/mjlab v1.6.0 (0fb8a681), src/mjlab/utils/noise/noise_cfg.py.
# Copyright 2025, The mjlab Developers.
# Modified by UniLab for NumPy and UniLab contracts; licensed under Apache-2.0.
from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import ClassVar, Literal

import numpy as np
import torch
from typing_extensions import override

from unilab.managers._noise import noise_model

# Type alias for noise parameters: scalar or per-dimension values.
NoiseParam = float | tuple[float, ...]

SegmentRange = tuple[float, float]


@dataclass(kw_only=True)
class NoiseCfg(abc.ABC):
    """Base configuration for a noise term."""

    operation: Literal["add", "scale", "abs"] = "add"

    @abc.abstractmethod
    def apply(
        self,
        data: np.ndarray | torch.Tensor,
        *,
        rng: np.random.Generator | None = None,
        torch_rng: torch.Generator | None = None,
    ) -> np.ndarray | torch.Tensor:
        """Apply noise to NumPy or Torch data on its existing carrier/device."""

    @staticmethod
    def _as_torch(
        value: NoiseParam,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        return torch.as_tensor(value, dtype=dtype, device=device)

    @staticmethod
    def _require_generator(rng: np.random.Generator | None) -> np.random.Generator:
        if rng is None:
            raise ValueError("noise sampling requires an env-owned Torch or NumPy generator")
        return rng

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
        data: np.ndarray | torch.Tensor,
        *,
        rng: np.random.Generator | None = None,
        torch_rng: torch.Generator | None = None,
    ) -> np.ndarray | torch.Tensor:
        del rng
        del torch_rng
        if not isinstance(data, torch.Tensor):
            bias = np.asarray(self.bias, dtype=data.dtype)
            if self.operation == "add":
                return data + bias
            if self.operation == "scale":
                return data * bias
            if self.operation == "abs":
                return np.zeros_like(data) + bias
            raise ValueError(f"Unsupported noise operation: {self.operation}")

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
        data: np.ndarray | torch.Tensor,
        *,
        rng: np.random.Generator | None = None,
        torch_rng: torch.Generator | None = None,
    ) -> np.ndarray | torch.Tensor:
        if rng is None and torch_rng is None:
            raise ValueError("UniformNoiseCfg requires an env-owned Torch or NumPy generator.")
        if not isinstance(data, torch.Tensor):
            generator = self._require_generator(rng)
            n_min = np.asarray(self.n_min, dtype=data.dtype)
            n_max = np.asarray(self.n_max, dtype=data.dtype)

            # Generate uniform noise in [0, 1) and transform the generated array
            # in place.  Float32 data draws directly in float32 (Generator.random
            # dtype fast path), which is ~2x faster than drawing float64 and
            # casting; bit-level noise values differ from the float64 path,
            # which the issue #1348 RNG-stream parity removal allows.
            if data.dtype == np.float32:
                noise = generator.random(data.shape, dtype=np.float32)
            else:
                noise = generator.random(data.shape).astype(data.dtype, copy=False)
            np.multiply(noise, n_max - n_min, out=noise)
            np.add(noise, n_min, out=noise)

            if self.operation == "add":
                np.add(data, noise, out=noise)
                return noise
            if self.operation == "scale":
                np.multiply(data, noise, out=noise)
                return noise
            if self.operation == "abs":
                return noise
            raise ValueError(f"Unsupported noise operation: {self.operation}")

        n_min = self._cached_torch(
            self.n_min, dtype=data.dtype, device=data.device, cache_name="_n_min_torch"
        )
        n_max = self._cached_torch(
            self.n_max, dtype=data.dtype, device=data.device, cache_name="_n_max_torch"
        )
        if torch_rng is not None and torch_rng.device != data.device:
            raise ValueError("Torch noise generator and observation device do not match.")
        if torch_rng is not None:
            unit = torch.rand(
                tuple(data.shape), dtype=data.dtype, device=data.device, generator=torch_rng
            )
            noise = unit * (n_max - n_min) + n_min
        else:
            generator = self._require_generator(rng)
            if data.dtype == torch.float32:
                unit = generator.random(tuple(data.shape), dtype=np.float32)
            else:
                unit = generator.random(tuple(data.shape)).astype(np.float32, copy=False)
            # The env-owned NumPy RNG remains authoritative in the default path.
            unit_torch = torch.from_numpy(unit).to(device=data.device, dtype=data.dtype)
            noise = unit_torch * (n_max - n_min) + n_min
        if self.operation == "add":
            return data + noise
        if self.operation == "scale":
            return data * noise
        if self.operation == "abs":
            return noise
        raise ValueError(f"Unsupported noise operation: {self.operation}")


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
        self, *, dtype: np.dtype | torch.dtype, device: torch.device | None = None
    ) -> tuple[np.ndarray, np.ndarray] | tuple[torch.Tensor, torch.Tensor]:
        if isinstance(dtype, torch.dtype):
            if device is None:
                raise ValueError("Torch segment noise requires a device")
            if (
                self._lower_torch is not None
                and self._upper_torch is not None
                and self._lower_torch.device == device
                and self._lower_torch.dtype == dtype
                and self._upper_torch.device == device
                and self._upper_torch.dtype == dtype
            ):
                lower_torch = self._lower_torch
                upper_torch = self._upper_torch
                return lower_torch, upper_torch
            self._lower_torch = torch.as_tensor(
                [range[0] for range in self.ranges], dtype=dtype, device=device
            )
            self._upper_torch = torch.as_tensor(
                [range[1] for range in self.ranges], dtype=dtype, device=device
            )
            return self._lower_torch, self._upper_torch
        return (
            np.asarray([range[0] for range in self.ranges], dtype=dtype),
            np.asarray([range[1] for range in self.ranges], dtype=dtype),
        )

    @override
    def apply(
        self,
        data: np.ndarray | torch.Tensor,
        *,
        rng: np.random.Generator | None = None,
        torch_rng: torch.Generator | None = None,
    ) -> np.ndarray | torch.Tensor:
        if data.ndim != 2 or data.shape[-1] != len(self.ranges):
            raise ValueError(
                "SegmentwiseUniformNoiseCfg expected a two-dimensional carrier with "
                f"{len(self.ranges)} final columns; got {tuple(data.shape)}"
            )
        if not isinstance(data, torch.Tensor):
            generator = self._require_generator(rng)
            lower, upper = self._bounds(dtype=data.dtype)
            assert isinstance(lower, np.ndarray) and isinstance(upper, np.ndarray)
            if data.dtype == np.float32:
                noise = generator.random(data.shape, dtype=np.float32)
            else:
                noise = generator.random(data.shape).astype(data.dtype, copy=False)
            noise *= upper - lower
            noise += lower
            if self.operation == "add":
                noise += data
                return noise
            if self.operation == "scale":
                noise *= data
                return noise
            if self.operation == "abs":
                return noise
            raise ValueError(f"Unsupported noise operation: {self.operation}")

        if torch_rng is not None and torch_rng.device != data.device:
            raise ValueError("Torch noise generator and observation device do not match.")
        lower, upper = self._bounds(dtype=data.dtype, device=data.device)
        assert isinstance(lower, torch.Tensor) and isinstance(upper, torch.Tensor)
        if torch_rng is not None:
            unit = torch.rand(
                tuple(data.shape), dtype=data.dtype, device=data.device, generator=torch_rng
            )
        else:
            generator = self._require_generator(rng)
            if data.dtype == torch.float32:
                host_unit = generator.random(tuple(data.shape), dtype=np.float32)
            else:
                host_unit = generator.random(tuple(data.shape)).astype(np.float32, copy=False)
            unit = torch.from_numpy(host_unit).to(device=data.device, dtype=data.dtype)
        noise = unit * (upper - lower) + lower
        if self.operation == "add":
            return data + noise
        if self.operation == "scale":
            return data * noise
        if self.operation == "abs":
            return noise
        raise ValueError(f"Unsupported noise operation: {self.operation}")


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
            if not np.isfinite(self.clamp) or self.clamp <= 0:
                raise ValueError(f"clamp ({self.clamp}) must be finite and positive")

    @override
    def apply(
        self,
        data: np.ndarray | torch.Tensor,
        *,
        rng: np.random.Generator | None = None,
        torch_rng: torch.Generator | None = None,
    ) -> np.ndarray | torch.Tensor:
        if rng is None and torch_rng is None:
            raise ValueError("GaussianNoiseCfg requires an env-owned Torch or NumPy generator.")
        if not isinstance(data, torch.Tensor):
            generator = self._require_generator(rng)
            mean = np.asarray(self.mean, dtype=data.dtype)
            std = np.asarray(self.std, dtype=data.dtype)

            # Generate standard normal noise and scale.  Float32 data draws
            # directly in float32 (same fast path as UniformNoiseCfg).
            if data.dtype == np.float32:
                noise = generator.standard_normal(data.shape, dtype=np.float32)
            else:
                noise = generator.standard_normal(data.shape).astype(data.dtype, copy=False)
            if self.clamp is not None:
                np.clip(noise, -self.clamp, self.clamp, out=noise)
            noise = mean + std * noise

            if self.operation == "add":
                return data + noise
            if self.operation == "scale":
                return data * noise
            if self.operation == "abs":
                return noise
            raise ValueError(f"Unsupported noise operation: {self.operation}")

        mean = self._as_torch(self.mean, dtype=data.dtype, device=data.device)
        std = self._as_torch(self.std, dtype=data.dtype, device=data.device)
        if torch_rng is not None:
            unit_torch = torch.randn(
                tuple(data.shape), dtype=data.dtype, device=data.device, generator=torch_rng
            )
        elif data.dtype == torch.float32:
            generator = self._require_generator(rng)
            unit = generator.standard_normal(tuple(data.shape), dtype=np.float32)
            unit_torch = torch.from_numpy(unit).to(device=data.device, dtype=data.dtype)
        else:
            generator = self._require_generator(rng)
            unit = generator.standard_normal(tuple(data.shape)).astype(np.float32, copy=False)
            unit_torch = torch.from_numpy(unit).to(device=data.device, dtype=data.dtype)
        if self.clamp is not None:
            unit_torch.clamp_(min=-self.clamp, max=self.clamp)
        noise = mean + std * unit_torch
        if self.operation == "add":
            return data + noise
        if self.operation == "scale":
            return data * noise
        if self.operation == "abs":
            return noise
        raise ValueError(f"Unsupported noise operation: {self.operation}")


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
