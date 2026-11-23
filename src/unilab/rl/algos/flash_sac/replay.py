"""Age-biased replay sampling for FlashSAC.

The device ring remains owned by the generic device replay pipeline. This
subclass changes only index selection: zero decay uses uniform sampling,
while positive and negative decays bias toward recent and older transitions
respectively. Bucketed sampling preserves that bias approximately for large
device-resident replay buffers.
"""

from __future__ import annotations

import torch

from unilab.rl.runtime.ipc.replay_buffer import ReplayBuffer
from unilab.rl.runtime.ipc.replay_pipelines.device import DeviceReplayPipeline


def _linear_age_weights(
    ages: torch.Tensor,
    *,
    decay_step: int,
    min_weight: float,
) -> torch.Tensor:
    span = max(abs(decay_step), 1)
    if decay_step > 0:
        return torch.clamp(1.0 - ages / span, min=min_weight)
    return torch.clamp(min_weight + ages / span, max=1.0)


def _with_zero_weight_fallback(weights: torch.Tensor) -> torch.Tensor:
    """Fall back to uniform weights when all biased weights vanish."""
    return torch.where(weights.sum() > 0, weights, torch.ones_like(weights))


def sample_biased_replay_indices(
    *,
    visible_size: int,
    capacity: int,
    current_ptr: int,
    sample_count: int,
    decay_step: int,
    min_weight: float,
    num_buckets: int,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    """Return physical indices in the authoritative replay ring.

    Logical indices are ordered oldest-to-newest.  Their age is therefore
    ``visible_size - logical_index`` both before and after the physical ring
    wraps, eliminating the need to store a timestamp per transition.
    """
    if decay_step == 0:
        return torch.randint(
            0,
            visible_size,
            (sample_count,),
            generator=generator,
            device=device,
        )

    if num_buckets > 0:
        bucket_size = (visible_size + num_buckets - 1) // num_buckets
        starts = torch.arange(0, visible_size, bucket_size, device=device)
        ends = torch.clamp(starts + bucket_size, max=visible_size)
        midpoints = (starts + ends - 1).div(2, rounding_mode="floor")
        bucket_weights = _linear_age_weights(
            visible_size - midpoints,
            decay_step=decay_step,
            min_weight=min_weight,
        )
        bucket_weights = _with_zero_weight_fallback(bucket_weights)
        sampled_buckets = torch.multinomial(
            bucket_weights,
            sample_count,
            replacement=True,
            generator=generator,
        )
        widths = (ends[sampled_buckets] - starts[sampled_buckets]).float()
        offsets = (torch.rand(sample_count, generator=generator, device=device) * widths).long()
        logical_indices = starts[sampled_buckets] + offsets
    else:
        logical_indices = torch.arange(visible_size, device=device)
        weights = _linear_age_weights(
            visible_size - logical_indices,
            decay_step=decay_step,
            min_weight=min_weight,
        )
        weights = _with_zero_weight_fallback(weights)
        logical_indices = torch.multinomial(
            weights,
            sample_count,
            replacement=True,
            generator=generator,
        )

    if current_ptr < capacity:
        return logical_indices
    ring_start = current_ptr - visible_size
    return (ring_start + logical_indices) % capacity


class AgeBiasedReplayPipeline(DeviceReplayPipeline):
    """Device-resident replay with a linear age-bias sampler."""

    def __init__(
        self,
        replay_buffer: ReplayBuffer,
        *,
        decay_step: int = 0,
        min_weight: float = 0.1,
        num_buckets: int = 2000,
        **kwargs,
    ) -> None:
        self.decay_step = int(decay_step)
        self.min_weight = float(min_weight)
        self.num_buckets = int(num_buckets)
        if not 0.0 <= self.min_weight <= 1.0:
            raise ValueError(f"min_weight must be in [0, 1], got {self.min_weight}")
        if self.num_buckets < 0:
            raise ValueError(f"num_buckets must be non-negative, got {self.num_buckets}")
        super().__init__(replay_buffer, **kwargs)

    def _gather_rows(self, *, visible_size: int, slot: int, gen: torch.Generator) -> None:
        indices = sample_biased_replay_indices(
            visible_size=visible_size,
            capacity=self._capacity,
            current_ptr=int(self._visible_ptr),
            sample_count=self._sample_count,
            decay_step=self.decay_step,
            min_weight=self.min_weight,
            num_buckets=self.num_buckets,
            device=self._device,
            generator=gen,
        )
        self._gather_packed_rows(indices, slot)


__all__ = ["AgeBiasedReplayPipeline"]
