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


@dataclass
class ConstantNoiseCfg(NoiseCfg):
    bias: NoiseParam = 0.0

    @override
    def apply(
        self,
        data: np.ndarray | torch.Tensor,
        *,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray | torch.Tensor:
        del rng
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
    ) -> np.ndarray | torch.Tensor:
        if rng is None:
            raise ValueError("UniformNoiseCfg requires an env-owned NumPy generator.")
        if not isinstance(data, torch.Tensor):
            n_min = np.asarray(self.n_min, dtype=data.dtype)
            n_max = np.asarray(self.n_max, dtype=data.dtype)

            # Generate uniform noise in [0, 1) and transform the generated array
            # in place.  Float32 data draws directly in float32 (Generator.random
            # dtype fast path), which is ~2x faster than drawing float64 and
            # casting; bit-level noise values differ from the float64 path,
            # which the issue #1348 RNG-stream parity removal allows.
            if data.dtype == np.float32:
                noise = rng.random(data.shape, dtype=np.float32)
            else:
                noise = rng.random(data.shape).astype(data.dtype, copy=False)
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

        n_min = self._as_torch(self.n_min, dtype=data.dtype, device=data.device)
        n_max = self._as_torch(self.n_max, dtype=data.dtype, device=data.device)
        if data.dtype == torch.float32:
            unit = rng.random(tuple(data.shape), dtype=np.float32)
        else:
            unit = rng.random(tuple(data.shape)).astype(np.float32, copy=False)
        # The env-owned NumPy RNG remains authoritative. This unit draw is the
        # only host transfer in the tensor path; arithmetic and output stay on
        # the observation device.
        unit_torch = torch.from_numpy(unit).to(device=data.device, dtype=data.dtype)
        noise = unit_torch * (n_max - n_min) + n_min
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

    def __post_init__(self):
        if isinstance(self.std, float) and self.std <= 0:
            raise ValueError(f"std ({self.std}) must be positive")

    @override
    def apply(
        self,
        data: np.ndarray | torch.Tensor,
        *,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray | torch.Tensor:
        if rng is None:
            raise ValueError("GaussianNoiseCfg requires an env-owned NumPy generator.")
        if not isinstance(data, torch.Tensor):
            mean = np.asarray(self.mean, dtype=data.dtype)
            std = np.asarray(self.std, dtype=data.dtype)

            # Generate standard normal noise and scale.  Float32 data draws
            # directly in float32 (same fast path as UniformNoiseCfg).
            if data.dtype == np.float32:
                noise = rng.standard_normal(data.shape, dtype=np.float32)
            else:
                noise = rng.standard_normal(data.shape).astype(data.dtype, copy=False)
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
        if data.dtype == torch.float32:
            unit = rng.standard_normal(tuple(data.shape), dtype=np.float32)
        else:
            unit = rng.standard_normal(tuple(data.shape)).astype(np.float32, copy=False)
        unit_torch = torch.from_numpy(unit).to(device=data.device, dtype=data.dtype)
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
