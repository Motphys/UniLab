"""Typed control-flow signals shared by environments and entrypoints."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType


class RunComplete(Exception):  # noqa: N818 - this is a normal control signal, not an error
    """Signal that a run reached its intended non-error completion condition."""

    def __init__(
        self,
        *,
        reason: str,
        summary: Mapping[str, object] | None = None,
    ) -> None:
        if not reason:
            raise ValueError("RunComplete reason must be non-empty")
        self.reason = reason
        self._summary = MappingProxyType(dict(summary or {}))
        super().__init__(reason)

    @property
    def summary(self) -> Mapping[str, object]:
        """Return the completion facts without allowing caller mutation."""
        return self._summary
