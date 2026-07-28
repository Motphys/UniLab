"""Cold-path entity resolution through the public ``SimBackend`` contract.

The manager compiler needs immutable IDs, while a task must not import a
concrete backend or reach into its model. This adapter is deliberately
narrow: it maps only exact selector kinds for which the base backend contract
has an unambiguous public ID query. Unsupported kinds and regex selection
fail before a plan is compiled rather than becoming a runtime fallback.
"""

from __future__ import annotations

import numpy as np

from unilab.base.backend.base import SimBackend

from .entities import EntityKind, EntitySelector, ManagerContractError, SelectorMode


class BackendEntityResolver:
    """Resolve exact manager selectors through cold public backend metadata."""

    def __init__(self, backend: SimBackend) -> None:
        if not isinstance(backend, SimBackend):
            raise ManagerContractError("backend entity resolver requires a SimBackend")
        self._backend = backend

    @staticmethod
    def _normalize_ids(
        ids: object,
        *,
        selector: EntitySelector,
    ) -> tuple[int, ...]:
        if not isinstance(ids, np.ndarray):
            raise ManagerContractError(
                f"backend selector {selector.key!r} did not return a numpy id array"
            )
        if ids.ndim != 1 or ids.shape != (len(selector.expressions),):
            raise ManagerContractError(
                f"backend selector {selector.key!r} returned ids with shape {ids.shape}, "
                f"expected {(len(selector.expressions),)}"
            )
        if not np.issubdtype(ids.dtype, np.integer):
            raise ManagerContractError(
                f"backend selector {selector.key!r} returned non-integer ids"
            )
        normalized = tuple(int(value) for value in ids)
        if any(value < 0 for value in normalized) or len(set(normalized)) != len(normalized):
            raise ManagerContractError(
                f"backend selector {selector.key!r} returned invalid or duplicate ids"
            )
        return normalized

    def resolve(self, selector: EntitySelector) -> tuple[int, ...]:
        """Resolve one exact selector once during task compilation."""

        if not isinstance(selector, EntitySelector):
            raise ManagerContractError("backend entity resolver requires an EntitySelector")
        if selector.mode is not SelectorMode.EXACT:
            raise ManagerContractError(
                f"backend entity resolver supports exact selectors only, got {selector.mode.value!r}"
            )
        if selector.kind is EntityKind.ROOT:
            if len(selector.expressions) != 1:
                raise ManagerContractError("a backend root selector must name exactly one body")
            resolve = self._backend.get_body_ids
        elif selector.kind is EntityKind.BODY:
            resolve = self._backend.get_body_ids
        elif selector.kind is EntityKind.DOF:
            # This intentionally selects single-DoF qpos coordinates. The
            # backend typed binder validates each field's velocity/position
            # semantics before physics; ball/free and unsupported layouts
            # remain fail-closed.
            resolve = self._backend.get_joint_dof_pos_indices
        elif selector.kind is EntityKind.SENSOR:
            resolve = self._backend.get_sensor_ids
        else:
            raise ManagerContractError(
                f"backend entity resolver does not support selector kind {selector.kind.value!r}"
            )
        try:
            ids = resolve(selector.expressions)
        except (NotImplementedError, ValueError) as exc:
            raise ManagerContractError(
                f"failed to resolve backend selector {selector.key!r}: {exc}"
            ) from exc
        return self._normalize_ids(ids, selector=selector)


__all__ = ["BackendEntityResolver"]
