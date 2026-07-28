"""Cold-path semantic entity selection for managed tasks."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol


class ManagerContractError(ValueError):
    """Raised when a managed task violates its compile-time contract."""


class EntityKind(str, Enum):
    ROOT = "root"
    BODY = "body"
    JOINT = "joint"
    DOF = "dof"
    SENSOR = "sensor"
    SITE = "site"
    GEOM = "geom"
    ACTUATOR = "actuator"
    TASK = "task"
    GLOBAL = "global"


class SelectorMode(str, Enum):
    EXACT = "exact"
    REGEX = "regex"


def _non_empty(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ManagerContractError(f"{name} must be a non-empty string")
    return value.strip()


@dataclass(frozen=True)
class EntitySelector:
    """Backend-agnostic selector resolved exactly once during task compilation."""

    key: str
    entity: str
    kind: EntityKind
    expressions: tuple[str, ...]
    mode: SelectorMode = SelectorMode.EXACT

    def __post_init__(self) -> None:
        object.__setattr__(self, "key", _non_empty(self.key, "selector key"))
        object.__setattr__(self, "entity", _non_empty(self.entity, "selector entity"))
        if not isinstance(self.kind, EntityKind):
            raise ManagerContractError("selector kind must be an EntityKind")
        if not isinstance(self.mode, SelectorMode):
            raise ManagerContractError("selector mode must be a SelectorMode")
        if not isinstance(self.expressions, tuple) or not self.expressions:
            raise ManagerContractError("selector expressions must be a non-empty tuple")
        normalized = tuple(_non_empty(item, "selector expression") for item in self.expressions)
        if len(set(normalized)) != len(normalized):
            raise ManagerContractError("selector expressions must be unique")
        object.__setattr__(self, "expressions", normalized)


class EntityResolver(Protocol):
    """Explicit cold-path resolver implemented by scene/backend materialization."""

    def resolve(self, selector: EntitySelector) -> tuple[int, ...]: ...


@dataclass(frozen=True)
class CompiledSelector:
    key: str
    entity: str
    kind: EntityKind
    mode: SelectorMode
    expressions: tuple[str, ...]
    entity_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "key", _non_empty(self.key, "compiled selector key"))
        object.__setattr__(self, "entity", _non_empty(self.entity, "compiled selector entity"))
        if not isinstance(self.kind, EntityKind):
            raise ManagerContractError("compiled selector kind must be an EntityKind")
        if not isinstance(self.mode, SelectorMode):
            raise ManagerContractError("compiled selector mode must be a SelectorMode")
        if not isinstance(self.expressions, tuple) or not self.expressions:
            raise ManagerContractError("compiled selector expressions must be a non-empty tuple")
        if not isinstance(self.entity_ids, tuple) or not self.entity_ids:
            raise ManagerContractError("compiled selector must contain at least one entity id")
        if any(
            isinstance(entity_id, bool) or not isinstance(entity_id, int) or entity_id < 0
            for entity_id in self.entity_ids
        ):
            raise ManagerContractError("compiled selector ids must be non-negative integers")
        if len(set(self.entity_ids)) != len(self.entity_ids):
            raise ManagerContractError("compiled selector ids must be unique")

    @classmethod
    def bind(cls, selector: EntitySelector, resolver: EntityResolver) -> CompiledSelector:
        try:
            ids = resolver.resolve(selector)
        except ManagerContractError:
            raise
        except (KeyError, ValueError) as exc:
            raise ManagerContractError(
                f"failed to resolve selector {selector.key!r}: {exc}"
            ) from exc
        if not isinstance(ids, tuple):
            raise ManagerContractError(
                f"resolver must return a tuple for selector {selector.key!r}"
            )
        return cls(
            key=selector.key,
            entity=selector.entity,
            kind=selector.kind,
            mode=selector.mode,
            expressions=selector.expressions,
            entity_ids=ids,
        )
