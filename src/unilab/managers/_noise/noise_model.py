# Derived from mujocolab/mjlab v1.6.0 (0fb8a681), src/mjlab/utils/noise/noise_model.py.
# Copyright 2025, The mjlab Developers.
# Modified by UniLab for NumPy and UniLab contracts; licensed under Apache-2.0.
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from typing_extensions import override

if TYPE_CHECKING:
    from unilab.managers._noise import noise_cfg


class NoiseModel:
    """Base class for noise models."""

    def __init__(
        self,
        noise_model_cfg: noise_cfg.NoiseModelCfg,
        num_envs: int,
        rng: np.random.Generator,
    ):
        self._noise_model_cfg = noise_model_cfg
        self._num_envs = num_envs
        self._rng = rng

        # Validate configuration.
        if not hasattr(noise_model_cfg, "noise_cfg") or noise_model_cfg.noise_cfg is None:
            raise ValueError("NoiseModelCfg must have a valid noise_cfg")

    def reset(self, env_ids: np.ndarray | slice | None = None) -> None:
        """Reset noise model state. Override in subclasses if needed."""

    def __call__(self, data: np.ndarray) -> np.ndarray:
        """Apply noise to input data."""
        assert self._noise_model_cfg.noise_cfg is not None
        return self._noise_model_cfg.noise_cfg.apply(data, rng=self._rng)


class NoiseModelWithAdditiveBias(NoiseModel):
    """Noise model with additional additive bias that is constant for the duration
    of the entire episode."""

    def __init__(
        self,
        noise_model_cfg: noise_cfg.NoiseModelWithAdditiveBiasCfg,
        num_envs: int,
        rng: np.random.Generator,
    ):
        super().__init__(noise_model_cfg, num_envs, rng)

        # Validate bias configuration.
        if not hasattr(noise_model_cfg, "bias_noise_cfg") or noise_model_cfg.bias_noise_cfg is None:
            raise ValueError("NoiseModelWithAdditiveBiasCfg must have a valid bias_noise_cfg")

        self._bias_noise_cfg = noise_model_cfg.bias_noise_cfg
        self._sample_bias_per_component = noise_model_cfg.sample_bias_per_component

        # Shape is materialized from the first observation so scalar and
        # higher-rank terms broadcast without a device-specific convention.
        self._bias = np.zeros((num_envs, 1), dtype=np.float32)
        self._bias_initialized = False

    @override
    def reset(self, env_ids: np.ndarray | slice | None = None) -> None:
        """Reset bias values for specified environments."""
        indices = slice(None) if env_ids is None else env_ids
        # Sample new bias values.
        self._bias[indices] = self._bias_noise_cfg.apply(self._bias[indices], rng=self._rng)

    def _initialize_bias_shape(self, data: np.ndarray) -> None:
        """Initialize bias tensor shape based on data and configuration."""
        if not self._bias_initialized:
            if data.ndim == 0 or data.shape[0] != self._num_envs:
                raise ValueError(
                    f"NoiseModel expected leading dimension {self._num_envs}, "
                    f"received shape {data.shape}."
                )
            if self._sample_bias_per_component:
                bias_shape = data.shape
            else:
                bias_shape = (self._num_envs, *([1] * (data.ndim - 1)))
            self._bias = np.zeros(bias_shape, dtype=data.dtype)
            self._bias_initialized = True
            self.reset()
        elif self._bias.shape != data.shape and self._sample_bias_per_component:
            raise ValueError(
                f"NoiseModel observation shape changed from {self._bias.shape} to {data.shape}."
            )

    @override
    def __call__(self, data: np.ndarray) -> np.ndarray:
        """Apply noise and additive bias to input data."""
        self._initialize_bias_shape(data)
        noisy_data = super().__call__(data)
        return noisy_data + self._bias
