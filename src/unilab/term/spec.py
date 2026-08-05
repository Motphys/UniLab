"""Data-only contracts for config-resolved numeric terms."""

from __future__ import annotations

import math
import numbers
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import TypeAlias

import numpy as np
from numpy.typing import DTypeLike

from .errors import TermRegistrationError

_NAME = re.compile(r"^[a-z][a-z0-9_.-]*$")
ParameterValue: TypeAlias = bool | int | float | str | tuple[bool | int | float | str, ...]


class TermKind(str, Enum):
    REWARD = "reward"
    OBSERVATION = "observation"
    TERMINATION = "termination"


class ParameterKind(str, Enum):
    FLOAT = "float"
    INTEGER = "integer"
    BOOLEAN = "boolean"
    STRING = "string"


class _Missing:
    pass


_MISSING = _Missing()


def _valid_name(value: str, label: str) -> None:
    if not _NAME.fullmatch(value):
        raise TermRegistrationError(f"{label} must match {_NAME.pattern!r}; got {value!r}")


def _scalar(value: object, kind: ParameterKind, label: str) -> bool | int | float | str:
    if kind is ParameterKind.BOOLEAN and isinstance(value, bool):
        return value
    if kind is ParameterKind.STRING and isinstance(value, str):
        return value
    if kind is ParameterKind.INTEGER:
        if not isinstance(value, bool) and isinstance(value, numbers.Integral):
            return int(value)
    if kind is ParameterKind.FLOAT:
        if not isinstance(value, bool) and isinstance(value, numbers.Real):
            result = float(value)
            if math.isfinite(result):
                return result
    raise TermRegistrationError(f"{label} must be {kind.value}")


def _config_value(value: object, label: str) -> ParameterValue:
    if isinstance(value, (list, tuple)):
        items = tuple(_config_value(item, label) for item in value)
        if any(isinstance(item, tuple) for item in items):
            raise TermRegistrationError(f"{label} must be a flat scalar tuple")
        return items  # type: ignore[return-value]
    if isinstance(value, (bool, str)):
        return value
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real) and math.isfinite(float(value)):
        return float(value)
    raise TermRegistrationError(f"{label} must be a finite scalar or flat scalar tuple")


@dataclass(frozen=True)
class TensorSpec:
    """Batched tensor shape without the leading environment dimension."""

    shape: tuple[int, ...]
    dtype: DTypeLike

    def __post_init__(self) -> None:
        shape = tuple(self.shape)
        if any(
            isinstance(dim, bool) or not isinstance(dim, numbers.Integral) or dim <= 0
            for dim in shape
        ):
            raise TermRegistrationError(f"tensor dimensions must be positive: {shape!r}")
        try:
            dtype = np.dtype(self.dtype)
        except TypeError as exc:
            raise TermRegistrationError(f"invalid tensor dtype {self.dtype!r}") from exc
        if dtype != np.dtype(np.bool_) and not np.issubdtype(dtype, np.number):
            raise TermRegistrationError(f"tensor dtype must be numeric or bool; got {dtype.name}")
        object.__setattr__(self, "shape", tuple(int(dim) for dim in shape))
        object.__setattr__(self, "dtype", dtype.str)

    @property
    def numpy_dtype(self) -> np.dtype:
        return np.dtype(self.dtype)


@dataclass(frozen=True)
class NamedTensorSpec:
    name: str
    tensor: TensorSpec

    def __post_init__(self) -> None:
        _valid_name(self.name, "tensor name")


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    kind: ParameterKind
    tuple_value: bool = False
    default: ParameterValue | _Missing = _MISSING

    def __post_init__(self) -> None:
        _valid_name(self.name, "parameter name")
        if not isinstance(self.kind, ParameterKind):
            raise TermRegistrationError(f"invalid parameter kind {self.kind!r}")
        if not isinstance(self.default, _Missing):
            object.__setattr__(self, "default", self.normalize(self.default))

    @property
    def required(self) -> bool:
        return isinstance(self.default, _Missing)

    def normalize(self, value: object) -> ParameterValue:
        label = f"parameter {self.name!r}"
        if not self.tuple_value:
            return _scalar(value, self.kind, label)
        if not isinstance(value, (list, tuple)):
            raise TermRegistrationError(f"{label} must be a tuple of {self.kind.value}")
        return tuple(_scalar(item, self.kind, label) for item in value)


@dataclass(frozen=True)
class NumpyTermContext:
    inputs: Mapping[str, np.ndarray]
    parameters: Mapping[str, ParameterValue]
    output: np.ndarray
    workspace: Mapping[str, np.ndarray]


NumpyTermFn: TypeAlias = Callable[[NumpyTermContext], None]
NumbaItemFn: TypeAlias = Callable[..., None]


@dataclass(frozen=True)
class TermDefinition:
    """Trusted implementation plus cold-path requirements.

    A future Numba item callable receives inputs, parameters, output, workspace,
    and row index in declaration order. It is metadata in Tier 1.
    """

    key: str
    kind: TermKind
    numpy_fn: NumpyTermFn
    output: TensorSpec
    inputs: tuple[NamedTensorSpec, ...] = field(default_factory=tuple)
    workspace: tuple[NamedTensorSpec, ...] = field(default_factory=tuple)
    parameters: tuple[ParameterSpec, ...] = field(default_factory=tuple)
    numba_item_fn: NumbaItemFn | None = None
    implementation_version: int = 1

    def __post_init__(self) -> None:
        _valid_name(self.key, "term key")
        if not isinstance(self.kind, TermKind) or not callable(self.numpy_fn):
            raise TermRegistrationError(f"term {self.key!r} has an invalid kind or numpy_fn")
        if self.numba_item_fn is not None and not callable(self.numba_item_fn):
            raise TermRegistrationError(f"term {self.key!r} numba_item_fn must be callable")
        if (
            isinstance(self.implementation_version, bool)
            or not isinstance(self.implementation_version, numbers.Integral)
            or self.implementation_version < 1
        ):
            raise TermRegistrationError(f"term {self.key!r} version must be a positive integer")
        object.__setattr__(self, "implementation_version", int(self.implementation_version))
        object.__setattr__(self, "inputs", tuple(self.inputs))
        object.__setattr__(self, "workspace", tuple(self.workspace))
        object.__setattr__(self, "parameters", tuple(self.parameters))
        for label, items in (
            ("input", self.inputs),
            ("workspace", self.workspace),
            ("parameter", self.parameters),
        ):
            names = [item.name for item in items]
            if len(names) != len(set(names)):
                raise TermRegistrationError(f"term {self.key!r} declares duplicate {label}")
        overlap = {item.name for item in self.inputs} & {item.name for item in self.workspace}
        if overlap:
            raise TermRegistrationError(f"term {self.key!r} overlaps input/workspace {overlap}")
        dtype = self.output.numpy_dtype
        if self.kind is TermKind.TERMINATION and dtype != np.dtype(np.bool_):
            raise TermRegistrationError(f"termination term {self.key!r} output must use bool dtype")
        if self.kind is not TermKind.TERMINATION and not np.issubdtype(dtype, np.floating):
            raise TermRegistrationError(
                f"{self.kind.value} term {self.key!r} output must use a floating dtype"
            )


@dataclass(frozen=True)
class TermConfig:
    name: str
    term_key: str
    scale: float = 1.0
    parameters: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _valid_name(self.name, "term instance name")
        _valid_name(self.term_key, "term key")
        if isinstance(self.scale, bool) or not isinstance(self.scale, numbers.Real):
            raise TermRegistrationError(f"term {self.name!r} scale must be finite")
        scale = float(self.scale)
        if not math.isfinite(scale):
            raise TermRegistrationError(f"term {self.name!r} scale must be finite")
        parameters = {
            name: _config_value(value, f"parameter {name!r}")
            for name, value in self.parameters.items()
        }
        for name in parameters:
            _valid_name(name, "parameter name")
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "parameters", MappingProxyType(parameters))
