"""Reusable tensor-runtime pieces for motion-tracking task owners.

These helpers deliberately do not construct environments or discover backend
capabilities. A task owner remains responsible for validating its semantic
contract and negotiating UniSim's public tensor lifecycle before using them.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, is_dataclass
from dataclasses import fields as dataclass_fields
from typing import Any, Mapping

import numpy as np
import torch

from unilab.managers._noise.noise_cfg import UniformNoiseCfg


def _qualified_name(value: Any) -> str:
    if isinstance(value, type):
        return f"{value.__module__}.{value.__qualname__}"
    return f"{type(value).__module__}.{type(value).__qualname__}"


def semantic_token(value: Any) -> Any:
    """Normalize an owner configuration value without unknown escapes."""
    if value is None:
        return "none"
    if isinstance(value, slice):
        return "slice", value.start, value.stop, value.step
    if is_dataclass(value) and not isinstance(value, type):
        excluded = {"fixed_variant_plan"} if type(value).__name__ == "SceneCfg" else set()
        excluded |= getattr(value, "_semantic_fingerprint_excludes", frozenset())
        return (
            _qualified_name(value),
            tuple(
                (item.name, semantic_token(getattr(value, item.name)))
                for item in dataclass_fields(value)
                if item.name not in excluded
            ),
        )
    if isinstance(value, (str, bytes, bool, int, float)):
        if isinstance(value, float) and not bool(np.isfinite(value)):
            raise ValueError("tensor task owner identity contains a non-finite scalar")
        return type(value).__name__, value
    if isinstance(value, np.generic):
        if isinstance(value, (float, np.floating)) and not bool(np.isfinite(value)):
            raise ValueError("tensor task owner identity contains a non-finite scalar")
        return type(value).__name__, value.item()
    if isinstance(value, np.ndarray):
        return _qualified_name(value), str(value.dtype), value.shape, value.tobytes(order="C")
    if isinstance(value, Mapping):
        return tuple((str(key), semantic_token(item)) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return tuple(semantic_token(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return tuple(sorted((semantic_token(item) for item in value), key=repr))
    if callable(value):
        return "callable", _qualified_name(value)
    raise TypeError(f"unsupported tensor task owner identity value: {_qualified_name(value)}")


def semantic_fingerprint(domain: str, value: Any) -> str:
    """Return a stable SHA-256 fingerprint for a task-owner semantic token."""
    if not domain:
        raise ValueError("tensor task semantic fingerprint domain must be non-empty")
    token = semantic_token((domain, value))
    return hashlib.sha256(repr(token).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class TensorObservationNoise:
    """Uniform additive observation noise with owner-declared bounds."""

    lower: torch.Tensor
    upper: torch.Tensor

    def __post_init__(self) -> None:
        if self.lower.shape != self.upper.shape or self.lower.ndim != 1:
            raise ValueError("observation noise bounds must be matching one-dimensional tensors")
        if self.lower.device != self.upper.device:
            raise ValueError("observation noise bounds must live on one device")
        if self.lower.numel() == 0:
            raise ValueError("observation noise bounds cannot be empty")
        if not bool(torch.isfinite(self.lower).all() and torch.isfinite(self.upper).all()):
            raise ValueError("observation noise bounds must be finite")
        if bool((self.upper < self.lower).any()):
            raise ValueError("observation noise upper bounds must not be below lower bounds")

    @classmethod
    def from_uniform_terms(
        cls,
        terms: tuple[UniformNoiseCfg, ...],
        widths: tuple[int, ...],
        device: torch.device,
    ) -> TensorObservationNoise:
        if len(terms) != len(widths) or not terms or min(widths) <= 0:
            raise ValueError("observation noise terms and widths must be non-empty and aligned")
        lower_values = tuple(term.n_min for term in terms if isinstance(term.n_min, float))
        upper_values = tuple(term.n_max for term in terms if isinstance(term.n_max, float))
        if len(lower_values) != len(terms) or len(upper_values) != len(terms):
            raise TypeError("motion-tracking tensor runtime requires scalar uniform noise bounds")
        if any(not np.isfinite(value) for value in (*lower_values, *upper_values)):
            raise ValueError("observation noise bounds must be finite")
        if any(upper < lower for lower, upper in zip(lower_values, upper_values)):
            raise ValueError("observation noise upper bounds must not be below lower bounds")
        width_tensor = torch.tensor(widths, device=device, dtype=torch.int64)
        lower = torch.tensor(lower_values, device=device, dtype=torch.float32).repeat_interleave(
            width_tensor
        )
        upper = torch.tensor(upper_values, device=device, dtype=torch.float32).repeat_interleave(
            width_tensor
        )
        return cls(lower, upper)

    def apply(
        self,
        observations: torch.Tensor,
        *,
        cursor: int,
        generator: torch.Generator,
    ) -> torch.Tensor:
        width = self.lower.numel()
        if cursor < 0 or cursor + width > observations.shape[-1]:
            raise ValueError("observation noise range does not fit the observation tensor")
        noise = torch.rand(
            (*observations.shape[:-1], width), device=self.lower.device, generator=generator
        )
        observations[..., cursor : cursor + width] += noise * (self.upper - self.lower) + self.lower
        return observations


@dataclass
class TensorResetPlan:
    """Device-side done union and selected reset-row planner."""

    terminated: torch.Tensor
    truncated: torch.Tensor
    _rows: torch.Tensor | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.terminated.shape != self.truncated.shape or self.terminated.ndim != 1:
            raise ValueError("tensor reset masks must be matching one-dimensional tensors")
        if self.terminated.dtype != torch.bool or self.truncated.dtype != torch.bool:
            raise TypeError("tensor reset masks must be boolean tensors")
        if self.terminated.device != self.truncated.device:
            raise ValueError("tensor reset masks must live on one device")

    @property
    def done(self) -> torch.Tensor:
        return self.terminated | self.truncated

    @property
    def rows(self) -> torch.Tensor:
        rows = self._rows
        if rows is None:
            rows = self.done.nonzero(as_tuple=False).flatten().to(torch.int64)
            object.__setattr__(self, "_rows", rows)
        return rows


@dataclass
class TensorEpisodeMetrics:
    """Device-resident current episode accumulators.

    ``snapshot`` and ``finished_values`` return tensors; callers decide when a
    logging or IPC boundary is allowed to copy them to host.
    """

    rewards: torch.Tensor
    lengths: torch.Tensor

    @classmethod
    def create(cls, num_envs: int, device: torch.device) -> TensorEpisodeMetrics:
        if num_envs <= 0:
            raise ValueError("tensor episode metrics require a positive environment count")
        return cls(
            rewards=torch.zeros(num_envs, dtype=torch.float32, device=device),
            lengths=torch.zeros(num_envs, dtype=torch.int64, device=device),
        )

    def update(self, reward: torch.Tensor, done: torch.Tensor) -> None:
        if reward.shape != done.shape or reward.shape != self.rewards.shape:
            raise ValueError("episode metric tensors must have environment shape")
        if done.dtype != torch.bool:
            raise TypeError("episode metric done values must be a boolean tensor")
        # The reward argument is the terminal transition reward. Include it,
        # capture finished rows, and only then reset those rows.
        self.rewards += reward
        self.lengths += 1

    def finished_values(self, rows: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            (self.rewards[rows].to(torch.float64), self.lengths[rows].to(torch.float64)), dim=1
        )

    def reset(self, rows: torch.Tensor) -> None:
        self.rewards[rows] = 0.0
        self.lengths[rows] = 0

    def snapshot(self) -> dict[str, torch.Tensor]:
        return {"reward": self.rewards, "length": self.lengths}


__all__ = [
    "TensorEpisodeMetrics",
    "TensorObservationNoise",
    "TensorResetPlan",
    "semantic_fingerprint",
    "semantic_token",
]
