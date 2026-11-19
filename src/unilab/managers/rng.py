"""Manager RNG adapter for owners.

The Manager environment owns one explicit CPU or CUDA ``torch.Generator`` and
publishes this adapter as ``ManagerBasedRlEnv.rng``. It never converts
drawn tensors back to NumPy, so a CUDA owner can keep randomization on its
declared device.
"""

from __future__ import annotations

from typing import cast

import torch


def _shape(size: int | tuple[int, ...]) -> tuple[int, ...]:
    if isinstance(size, bool):
        raise TypeError(f"ManagerRng shape must be an int or tuple of ints, got {size!r}")
    if isinstance(size, int):
        return (size,)
    if not isinstance(size, tuple) or not all(
        isinstance(value, int) and not isinstance(value, bool) for value in size
    ):
        raise TypeError(f"ManagerRng shape must be an int or tuple of ints, got {size!r}")
    return size


def _bounds(
    value: float | int | torch.Tensor,
    *,
    name: str,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor | float:
    if isinstance(value, (float, int)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, torch.Tensor):
        tensor = value.to(device=device, dtype=dtype)
    else:
        raise TypeError(f"ManagerRng {name} must be a scalar or Tensor bounds, got {value!r}")
    if tensor.numel() == 0:
        raise ValueError(f"ManagerRng {name} cannot be empty")
    return tensor


def _sampled_value(
    value: torch.Tensor | float,
    shape: tuple[int, ...],
    *,
    device: torch.device,
) -> torch.Tensor:
    if not isinstance(value, float):
        expanded = cast(torch.Tensor, value)
        return expanded.expand(shape)
    return torch.full(shape, value, device=device)


def _sampled_bounds(
    low: torch.Tensor | float,
    high: torch.Tensor | float,
    shape: tuple[int, ...],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _sampled_value(low, shape, device=device), _sampled_value(high, shape, device=device)


class ManagerRng:
    """Small explicit RNG surface backed by one ``torch.Generator``.

    The method names mirror the NumPy Generator calls used by Manager terms so
    owner code can be compared side by side. Results are not bit-compatible with
    NumPy: this is a device-resident replacement, not a stream compatibility
    adapter.
    """

    def __init__(self, *, device: str | torch.device, generator: torch.Generator | None = None):
        resolved = torch.device(device)
        if resolved.type not in {"cpu", "cuda"}:
            raise ValueError(f"ManagerRng supports CPU/CUDA devices only, got {resolved}")
        if resolved.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(f"ManagerRng CUDA device is unavailable: {resolved}")
            if resolved.index is None:
                resolved = torch.device("cuda", index=torch.cuda.current_device())
        if generator is None:
            generator = torch.Generator(device=resolved)
        elif generator.device != resolved:
            raise ValueError(
                f"ManagerRng generator device {generator.device} does not match {resolved}"
            )
        self.generator = generator
        self.device = resolved

    @classmethod
    def seeded(cls, seed: int, *, device: str | torch.device = "cpu") -> ManagerRng:
        return cls(device=device).manual_seed(seed)

    def manual_seed(self, seed: int) -> ManagerRng:
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError(f"ManagerRng seed must be a non-negative integer, got {seed!r}")
        self.generator.manual_seed(int(seed))
        return self

    def random(
        self, size: int | tuple[int, ...], *, dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        return torch.rand(_shape(size), dtype=dtype, device=self.device, generator=self.generator)

    def uniform(
        self,
        low: float | int | torch.Tensor,
        high: float | int | torch.Tensor,
        size: int | tuple[int, ...],
        *,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        shape = _shape(size)
        low_t = _bounds(low, name="low", device=self.device, dtype=dtype)
        high_t = _bounds(high, name="high", device=self.device, dtype=dtype)
        unit = torch.rand(shape, dtype=dtype, device=self.device, generator=self.generator)
        sampled_low, sampled_high = _sampled_bounds(low_t, high_t, shape, device=self.device)
        return unit * (sampled_high - sampled_low) + sampled_low

    def standard_normal(
        self, size: int | tuple[int, ...], *, dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        return torch.randn(_shape(size), dtype=dtype, device=self.device, generator=self.generator)

    def normal(
        self,
        mean: float | int | torch.Tensor,
        std: float | int | torch.Tensor,
        size: int | tuple[int, ...],
        *,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        shape = _shape(size)
        mean_t = _bounds(mean, name="mean", device=self.device, dtype=dtype)
        std_t = _bounds(std, name="std", device=self.device, dtype=dtype)
        unit = torch.randn(shape, dtype=dtype, device=self.device, generator=self.generator)
        sampled_mean, sampled_std = _sampled_bounds(mean_t, std_t, shape, device=self.device)
        return sampled_mean + sampled_std * unit

    def integers(
        self,
        low: int,
        high: int,
        size: int | tuple[int, ...],
        *,
        dtype: torch.dtype = torch.int64,
    ) -> torch.Tensor:
        if isinstance(low, bool) or not isinstance(low, int) or low < 0:
            raise ValueError(f"ManagerRng integers low must be a non-negative integer, got {low!r}")
        if isinstance(high, bool) or not isinstance(high, int) or high <= low:
            raise ValueError(
                f"ManagerRng integers high must be greater than low, got {low!r}, {high!r}"
            )
        shape = _shape(size)
        if dtype not in {torch.int32, torch.int64}:
            raise TypeError(f"ManagerRng integers supports int32/int64, got {dtype}")
        return torch.randint(
            int(low),
            int(high),
            shape,
            dtype=dtype,
            device=self.device,
            generator=self.generator,
        )

    def choice(
        self,
        a: int,
        size: int | tuple[int, ...],
        *,
        p: torch.Tensor,
    ) -> torch.Tensor:
        """Sample category indices from ``{0, ..., a - 1}`` with probabilities ``p``.

        Sampling is always with replacement. The method performs no host
        reads: it is sync-free and safe on fixed-shape autoreset and CUDA
        graph hot paths. ``p`` must therefore be trusted by the caller:
        finite, non-negative, with a positive sum. An invalid ``p`` fails in
        ``torch.multinomial`` itself — a clean ``RuntimeError`` on CPU, but a
        device-side assert with a poisoned CUDA context on CUDA.
        """
        if isinstance(a, bool) or not isinstance(a, int) or a <= 0:
            raise ValueError(f"ManagerRng choice population must be a positive int, got {a}")
        if not isinstance(p, torch.Tensor):
            raise TypeError("ManagerRng choice probabilities must be a torch.Tensor")
        if p.ndim != 1 or p.numel() != a:
            raise ValueError(
                f"ManagerRng choice probabilities must have shape ({a},), got {tuple(p.shape)}"
            )
        if p.device != self.device:
            raise ValueError(
                f"ManagerRng choice probabilities must live on {self.device}, got {p.device}"
            )
        count = _shape(size)
        if len(count) != 1:
            raise TypeError("ManagerRng choice size must be one-dimensional")
        return torch.multinomial(p, count[0], replacement=True, generator=self.generator)

    def get_state(self) -> torch.Tensor:
        return self.generator.get_state()

    def set_state(self, state: torch.Tensor) -> None:
        if state.device.type != "cpu":
            raise ValueError("ManagerRng state must be a CPU Torch state tensor")
        self.generator.set_state(state)


__all__ = ["ManagerRng"]
