"""Fixed-shape selected-row reset (``ResetSelection``).

The public reset contract speaks IsaacLab/mjlab ``env_ids`` row indices. Row
indices have a data-dependent shape: building them from a device done mask
needs ``nonzero`` (a host synchronization), and every consumer then launches
kernels whose shape depends on how many rows reset. Neither is compatible with
a host-sync-free step or with CUDA graph capture.

``ResetSelection`` is the fixed-shape alternative. It carries a persistent
full-width bool ``mask`` that the environment rewrites in place before each
reset; every participant applies its reset as a masked full-width update, so
shapes never depend on the selected row count and no host read decides them.

The selection is opt-in per participant. A manager, term or buffer declares
support with the ``supports_reset_selection`` attribute (``True``) and accepts
a ``ResetSelection`` wherever it accepts ``env_ids``. Participants that do not
declare support never receive one: the environment negotiates support before
it enables the masked autoreset and reports every blocker by name. Code that
receives a selection by mistake fails closed, because a ``ResetSelection`` is
not a tensor and cannot be used as an index.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, TypeVar

import torch

SUPPORTS_RESET_SELECTION = "supports_reset_selection"

_T = TypeVar("_T")


@dataclass(frozen=True, eq=False)
class ResetSelection:
    """Full-width reset selection: row ``i`` resets when ``mask[i]`` is true.

    ``mask`` is owned by the environment and has a fixed address for the
    environment's lifetime; its contents change between resets. ``row_ids`` is
    the persistent ``arange(num_envs)``; it is *not* the set of reset rows and
    exists for full-width gathers.
    """

    mask: torch.Tensor
    row_ids: torch.Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.mask, torch.Tensor) or self.mask.dtype != torch.bool:
            raise TypeError("ResetSelection.mask must be a bool torch.Tensor")
        if self.mask.ndim != 1:
            raise ValueError(f"ResetSelection.mask must be 1-D; got shape {tuple(self.mask.shape)}")
        if (
            not isinstance(self.row_ids, torch.Tensor)
            or self.row_ids.dtype != torch.int64
            or tuple(self.row_ids.shape) != tuple(self.mask.shape)
            or self.row_ids.device != self.mask.device
        ):
            raise ValueError("ResetSelection.row_ids must be int64 arange(num_envs) on mask.device")

    @classmethod
    def allocate(cls, num_envs: int, device: torch.device | str) -> ResetSelection:
        """Allocate a persistent selection with an all-false mask."""
        device = torch.device(device)
        return cls(
            mask=torch.zeros(num_envs, dtype=torch.bool, device=device),
            row_ids=torch.arange(num_envs, dtype=torch.int64, device=device),
        )

    @property
    def num_envs(self) -> int:
        return int(self.mask.shape[0])

    @property
    def device(self) -> torch.device:
        return self.mask.device

    def row_mask(self, ndim: int, *, row_dim: int = 0) -> torch.Tensor:
        """Return ``mask`` shaped to broadcast over a ``ndim`` tensor's row dim."""
        if not 0 <= row_dim < ndim:
            raise ValueError(f"row_dim {row_dim} outside a {ndim}-D tensor")
        shape = [1] * ndim
        shape[row_dim] = self.num_envs
        return self.mask.view(shape)


def is_reset_selection(selector: object) -> bool:
    return isinstance(selector, ResetSelection)


def supports_reset_selection(participant: object) -> bool:
    """Whether ``participant`` declared that it accepts a ``ResetSelection``.

    For class-based participants the declaration must sit at or below the
    class that defines the effective ``reset``: a subclass that overrides
    ``reset`` does not inherit its parent's declaration and has to opt in
    again, so an unconverted override is reported instead of failing later.
    """
    if getattr(participant, SUPPORTS_RESET_SELECTION, False) is not True:
        return False
    cls = participant if isinstance(participant, type) else type(participant)
    if not isinstance(participant, type) and SUPPORTS_RESET_SELECTION in vars(participant):
        return True  # declared on the instance itself (function terms)
    mro = cls.__mro__
    declared_by = next((c for c in mro if SUPPORTS_RESET_SELECTION in vars(c)), None)
    reset_by = next((c for c in mro if "reset" in vars(c)), None)
    if declared_by is None or reset_by is None:
        return True
    return issubclass(declared_by, reset_by)


def reset_selection_capable(participant: _T) -> _T:
    """Mark a function-style term (for example a reset event) as selection-capable.

    Use only for callables whose body handles a ``ResetSelection`` argument;
    class-based participants set ``supports_reset_selection = True`` instead.
    """
    setattr(participant, SUPPORTS_RESET_SELECTION, True)
    return participant


def selection_blockers(owner: str, participants: Iterable[tuple[str, object]]) -> list[str]:
    """Name every participant that has not declared ``ResetSelection`` support."""
    return [
        f"{owner}: {name} ({_describe(participant)})"
        for name, participant in participants
        if not supports_reset_selection(participant)
    ]


def _describe(participant: object) -> str:
    # Functions and classes carry __qualname__; instances report their type.
    return getattr(participant, "__qualname__", None) or type(participant).__qualname__


def fill_rows_(
    tensor: torch.Tensor,
    rows: torch.Tensor | slice | ResetSelection | None,
    value: float | int | bool,
    *,
    row_dim: int = 0,
) -> None:
    """``tensor[rows] = value`` along ``row_dim`` for any reset selector."""
    if isinstance(rows, ResetSelection):
        tensor.masked_fill_(rows.row_mask(tensor.ndim, row_dim=row_dim), value)
        return
    index: tuple[Any, ...] = (slice(None),) * row_dim + (slice(None) if rows is None else rows,)
    tensor[index] = value


def assign_rows_(
    tensor: torch.Tensor,
    rows: torch.Tensor | slice | ResetSelection | None,
    values: torch.Tensor | Callable[[int], torch.Tensor],
    *,
    row_dim: int = 0,
) -> None:
    """Write new values into the selected rows.

    ``values`` is either a full-width tensor (row ``i`` of ``values`` lands in
    row ``i`` of ``tensor``) or a factory ``values(n)`` producing ``n`` rows.
    Under a ``ResetSelection`` the factory is called with ``num_envs`` and the
    result is merged through the mask, so its shape stays fixed.
    """
    if isinstance(rows, ResetSelection):
        full = values(rows.num_envs) if callable(values) else values
        tensor.copy_(torch.where(rows.row_mask(tensor.ndim, row_dim=row_dim), full, tensor))
        return
    index: tuple[Any, ...] = (slice(None),) * row_dim + (slice(None) if rows is None else rows,)
    if callable(values):
        if rows is None:
            count = tensor.shape[row_dim]
        elif isinstance(rows, slice):
            count = len(range(*rows.indices(tensor.shape[row_dim])))
        else:
            count = int(rows.numel())
        tensor[index] = values(count)
        return
    tensor[index] = values[index]


__all__ = [
    "SUPPORTS_RESET_SELECTION",
    "ResetSelection",
    "assign_rows_",
    "fill_rows_",
    "is_reset_selection",
    "reset_selection_capable",
    "selection_blockers",
    "supports_reset_selection",
]
