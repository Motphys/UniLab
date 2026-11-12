"""Reset row-selector conversion between the public ``env_ids`` and mask forms.

The task/manager reset contract speaks IsaacLab/mjlab-style ``env_ids`` row
indices. Backend transactions keep consuming fixed-width bool masks
(``SimBackend.apply_reset``/packed reset). These helpers are the single,
deliberately simple conversion point between the two forms.

Hot reset paths must not allocate a fresh mask per conversion
(roadmap #2461): each conversion site owns a ``ResetMaskBuffer`` whose storage
is allocated once on the env/manager/term and rewritten in place per call. The
module-level helpers keep allocating and remain the owned-copy form for
callers without a stable owner (tests, fixtures).
"""

from __future__ import annotations

from typing import overload

import torch

from unilab.base.reset_selection import ResetSelection


class ResetMaskBuffer:
    """Reusable fixed-width bool reset mask owned by one conversion site.

    ``scatter``/``resolve`` rewrite the internal storage in place and return
    it; the returned tensor aliases that storage and stays valid only until
    the next call on the same buffer. Every site whose masks may be live at
    the same time owns a separate instance: the env reset-transaction entry,
    the env-owned event-term scratch, each manager, and each class-based term.
    Buffer lifetime follows its owner (the env or manager instance), so
    separate env instances never share storage.
    """

    def __init__(self, num_envs: int, device: torch.device) -> None:
        self._mask = torch.zeros(num_envs, dtype=torch.bool, device=device)

    @property
    def num_envs(self) -> int:
        return int(self._mask.shape[0])

    @property
    def device(self) -> torch.device:
        return self._mask.device

    def scatter(self, env_ids: torch.Tensor | ResetSelection) -> torch.Tensor:
        """Rewrite the buffer from row indices and return the aliased mask.

        A ``ResetSelection`` is copied without scattering (fixed shape, no
        host read).
        """
        if isinstance(env_ids, ResetSelection):
            self._mask.copy_(env_ids.mask)
            return self._mask
        self._mask.zero_()
        if env_ids.numel():
            self._mask[env_ids.to(device=self._mask.device, dtype=torch.int64)] = True
        return self._mask

    def resolve(
        self, selector: torch.Tensor | slice | ResetSelection | None, *, owner: str
    ) -> torch.Tensor:
        """Resolve a public reset selector like ``resolve_reset_mask``.

        Every selector is normalized to ``env_ids`` and scattered into this
        buffer.
        """
        env_ids = resolve_reset_env_ids(
            selector, num_envs=self.num_envs, device=self.device, owner=owner
        )
        return self.scatter(env_ids)


@overload
def resolve_reset_env_ids(
    selector: ResetSelection,
    *,
    num_envs: int,
    device: torch.device,
    owner: str,
) -> ResetSelection: ...


@overload
def resolve_reset_env_ids(
    selector: torch.Tensor | slice | None,
    *,
    num_envs: int,
    device: torch.device,
    owner: str,
) -> torch.Tensor: ...


def resolve_reset_env_ids(
    selector: torch.Tensor | slice | ResetSelection | None,
    *,
    num_envs: int,
    device: torch.device,
    owner: str,
) -> torch.Tensor | ResetSelection:
    """Normalize a public reset selector to an int64 row-index tensor.

    ``None`` selects every row; a slice selects ``arange(num_envs)[slice]``; an
    integer tensor is validated and moved onto ``device``. Fixed-width bool
    masks are rejected: the v1.4 mask form was removed in v1.5.

    A ``ResetSelection`` is returned unchanged after a host-only structural
    check: it is the fixed-shape selector of selection-capable participants
    (see ``unilab.base.reset_selection``), valid by construction.
    """
    if isinstance(selector, ResetSelection):
        if selector.num_envs != num_envs or not _same_device(selector.device, device):
            raise ValueError(
                f"{owner} ResetSelection covers {selector.num_envs} rows on "
                f"{selector.device}; expected {num_envs} rows on {device}"
            )
        return selector
    if selector is None:
        return torch.arange(num_envs, dtype=torch.int64, device=device)
    if isinstance(selector, slice):
        return torch.arange(num_envs, dtype=torch.int64, device=device)[selector]
    if not isinstance(selector, torch.Tensor):
        raise TypeError(
            f"{owner} env_ids must be a torch.Tensor, slice, or None; got {type(selector).__name__}"
        )
    if selector.dtype == torch.bool:
        raise TypeError(
            f"{owner} env_ids must be a 1-D integer row-index tensor; got a bool "
            "mask (the v1.4 fixed-width mask form was removed in v1.5)"
        )
    if selector.ndim != 1:
        raise ValueError(
            f"{owner} env_ids must be a 1-D row-index tensor; got shape {tuple(selector.shape)}"
        )
    ids = selector.to(device=device, dtype=torch.int64)
    if ids.numel():
        # One fused device reduction and one host read per validation; separate
        # lower/upper checks would synchronize twice per caller.
        if bool(((ids < 0) | (ids >= num_envs)).any()):
            raise ValueError(f"{owner} env_ids contain rows outside [0, {num_envs})")
    return ids


def _same_device(left: torch.device, right: torch.device | str) -> bool:
    right = torch.device(right)
    return left.type == right.type and (
        left.index is None or right.index is None or left.index == right.index
    )


def reset_env_ids_mask(
    env_ids: torch.Tensor,
    *,
    num_envs: int,
    device: torch.device,
) -> torch.Tensor:
    """Scatter row indices into a freshly allocated fixed-width bool mask."""
    return ResetMaskBuffer(num_envs, device).scatter(env_ids)


def resolve_reset_mask(
    selector: torch.Tensor | slice | ResetSelection | None,
    *,
    num_envs: int,
    device: torch.device,
    owner: str,
) -> torch.Tensor:
    """Resolve a public reset selector to the internal fixed-width bool mask.

    The selector is normalized to ``env_ids`` and scattered into a freshly
    allocated mask.
    """
    env_ids = resolve_reset_env_ids(selector, num_envs=num_envs, device=device, owner=owner)
    return reset_env_ids_mask(env_ids, num_envs=num_envs, device=device)


__all__ = [
    "ResetMaskBuffer",
    "reset_env_ids_mask",
    "resolve_reset_env_ids",
    "resolve_reset_mask",
]
