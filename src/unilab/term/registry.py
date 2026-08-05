"""Explicit registry for trusted term implementations."""

from __future__ import annotations

from .errors import TermPlanError, TermRegistrationError
from .spec import TermDefinition


class TermRegistry:
    def __init__(self) -> None:
        self._definitions: dict[str, TermDefinition] = {}

    def register(self, definition: TermDefinition) -> TermDefinition:
        if definition.key in self._definitions:
            raise TermRegistrationError(f"term key {definition.key!r} is already registered")
        self._definitions[definition.key] = definition
        return definition

    def resolve(self, key: str) -> TermDefinition:
        try:
            return self._definitions[key]
        except KeyError as exc:
            raise TermPlanError(f"unknown term key {key!r}") from exc
