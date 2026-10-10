# Derived from mujocolab/mjlab v1.6.0 (0fb8a681), src/mjlab/utils/noise/noise_model.py.
# Copyright 2025, The mjlab Developers.
# Modified by UniLab for NumPy and UniLab contracts; licensed under Apache-2.0.
from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from typing_extensions import override

if TYPE_CHECKING:
    from unilab.managers._noise import noise_cfg


class NoiseModel:
    """Base class for tensor-native noise models."""

    def __init__(
        self,
        noise_model_cfg: noise_cfg.NoiseModelCfg,
        num_envs: int,
        *,
        torch_rng: torch.Generator,
        device: torch.device | str | None = None,
    ):
        self._noise_model_cfg = noise_model_cfg
        self._num_envs = num_envs
        self._torch_rng = torch_rng
        self._device = torch.device(device) if device is not None else torch.device("cpu")

        # Validate configuration.
        if not hasattr(noise_model_cfg, "noise_cfg") or noise_model_cfg.noise_cfg is None:
            raise ValueError("NoiseModelCfg must have a valid noise_cfg")

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        """Reset noise model state. Override in subclasses if needed."""

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """Apply noise to input data on its existing device."""
        assert self._noise_model_cfg.noise_cfg is not None
        return cast(
            "torch.Tensor",
            self._noise_model_cfg.noise_cfg.apply(data, torch_rng=self._torch_rng),
        )


class NoiseModelWithAdditiveBias(NoiseModel):
    """Noise model with additional additive bias that is constant for the duration
    of the entire episode."""

    def __init__(
        self,
        noise_model_cfg: noise_cfg.NoiseModelWithAdditiveBiasCfg,
        num_envs: int,
        *,
        torch_rng: torch.Generator,
        device: torch.device | str | None = None,
    ):
        super().__init__(noise_model_cfg, num_envs, torch_rng=torch_rng, device=device)

        # Validate bias configuration.
        if not hasattr(noise_model_cfg, "bias_noise_cfg") or noise_model_cfg.bias_noise_cfg is None:
            raise ValueError("NoiseModelWithAdditiveBiasCfg must have a valid bias_noise_cfg")

        self._bias_noise_cfg = noise_model_cfg.bias_noise_cfg
        self._sample_bias_per_component = noise_model_cfg.sample_bias_per_component

        # Shape is materialized from the first observation so scalar and
        # higher-rank terms broadcast without a device-specific convention.
        self._bias = torch.zeros((num_envs, 1), dtype=torch.float32, device=self._device)
        self._bias_initialized = False

    @override
    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        """Reset bias values for specified environments."""
        indices = slice(None) if env_ids is None else env_ids
        replacement = self._bias_noise_cfg.apply(self._bias[indices], torch_rng=self._torch_rng)
        self._bias[indices] = cast("torch.Tensor", replacement)

    def _initial_bias(self, data: torch.Tensor) -> torch.Tensor:
        shape = tuple(data.shape)
        if self._sample_bias_per_component:
            bias_shape = shape
        else:
            bias_shape = (self._num_envs, *([1] * (len(shape) - 1)))
        return torch.zeros(bias_shape, dtype=data.dtype, device=data.device)

    def _initialize_bias_shape(self, data: torch.Tensor) -> None:
        """Initialize bias carrier shape based on data and configuration."""
        if not self._bias_initialized:
            if data.ndim == 0 or data.shape[0] != self._num_envs:
                raise ValueError(
                    f"NoiseModel expected leading dimension {self._num_envs}, "
                    f"received shape {tuple(data.shape)}."
                )
            self._bias = self._initial_bias(data)
            self._bias_initialized = True
            self.reset()
        elif tuple(self._bias.shape) != tuple(data.shape) and self._sample_bias_per_component:
            raise ValueError(
                f"NoiseModel observation shape changed from {tuple(self._bias.shape)} "
                f"to {tuple(data.shape)}."
            )

    @override
    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """Apply noise and additive bias to input data."""
        self._initialize_bias_shape(data)
        noisy_data = super().__call__(data)
        return noisy_data + self._bias
