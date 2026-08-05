"""Cold-path registry for managed term definitions."""

from __future__ import annotations

from dataclasses import dataclass, field

from .entities import ManagerContractError
from .spec import TermDefinition


@dataclass
class TermRegistry:
    """Mutable build-time registry that never escapes into a compiled plan."""

    _definitions: dict[str, TermDefinition] = field(default_factory=dict, init=False, repr=False)
    _lookup_count: int = field(default=0, init=False, repr=False)
    _frozen: bool = field(default=False, init=False, repr=False)

    def register(self, definition: TermDefinition) -> None:
        if self._frozen:
            raise ManagerContractError("term registry is frozen")
        if not isinstance(definition, TermDefinition):
            raise ManagerContractError("registry entries must be TermDefinition values")
        if definition.key in self._definitions:
            raise ManagerContractError(f"term definition {definition.key!r} is already registered")
        self._definitions[definition.key] = definition

    def resolve(self, key: str) -> TermDefinition:
        self._lookup_count += 1
        try:
            return self._definitions[key]
        except KeyError as exc:
            raise ManagerContractError(f"term definition {key!r} is not registered") from exc

    def freeze(self) -> None:
        self._frozen = True

    @property
    def lookup_count(self) -> int:
        return self._lookup_count

    @property
    def is_frozen(self) -> bool:
        return self._frozen

    def definitions(self) -> tuple[TermDefinition, ...]:
        return tuple(self._definitions[key] for key in sorted(self._definitions))
