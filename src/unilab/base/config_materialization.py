"""Cold-path materialization of Hydra-owned typed configuration."""

from __future__ import annotations

import dataclasses
import inspect
import types
from collections.abc import Mapping
from typing import Any, Union, get_args, get_origin, get_type_hints

from hydra.utils import get_object, instantiate
from omegaconf import OmegaConf

from .config_overrides import (
    CONFIG_MAPPING_POLICY_KEY,
    MANAGER_PARAMS_MAPPING_POLICY,
    MANAGER_TERM_MAPPING_POLICY,
)

HYDRA_TARGET_KEY = "_target_"
_MISSING = object()


def _plain(value: Any) -> Any:
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _hints(target_type: type[Any]) -> dict[str, Any]:
    try:
        return get_type_hints(target_type)
    except (NameError, TypeError):
        return {field.name: field.type for field in dataclasses.fields(target_type)}


def _dataclass_types(annotation: Any) -> tuple[type[Any], ...]:
    if annotation in (Any, None):
        return ()
    origin = get_origin(annotation)
    if origin in (types.UnionType, Union):
        return tuple(
            target
            for item in get_args(annotation)
            if item is not type(None)
            for target in _dataclass_types(item)
        )
    if isinstance(annotation, type) and dataclasses.is_dataclass(annotation):
        return (annotation,)
    return ()


def _dict_value_type(annotation: Any) -> Any:
    origin = get_origin(annotation)
    if origin is dict:
        args = get_args(annotation)
        return args[1] if len(args) == 2 else Any
    if origin in (types.UnionType, Union):
        for item in get_args(annotation):
            value_type = _dict_value_type(item)
            if value_type is not _MISSING:
                return value_type
    return _MISSING


def _resolve(reference: Any, *, path: str) -> Any:
    if not isinstance(reference, str) or not reference.strip():
        raise TypeError(f"Config field '{path}' must be a non-empty dotted string")
    try:
        return get_object(reference)
    except Exception as exc:
        raise ValueError(
            f"Config field '{path}' could not resolve dotted reference {reference!r}: {exc}"
        ) from exc


def _resolve_target(
    reference: Any,
    *,
    expected: Any,
    path: str,
) -> type[Any]:
    target = _resolve(reference, path=f"{path}.{HYDRA_TARGET_KEY}")
    if not isinstance(target, type) or not dataclasses.is_dataclass(target):
        raise TypeError(
            f"Config field '{path}.{HYDRA_TARGET_KEY}' must resolve to a dataclass type"
        )
    expected_types = _dataclass_types(expected)
    if expected_types and not any(issubclass(target, item) for item in expected_types):
        names = ", ".join(item.__qualname__ for item in expected_types)
        raise TypeError(
            f"Config field '{path}.{HYDRA_TARGET_KEY}' resolved to {target.__qualname__}, "
            f"expected a subclass of {names}"
        )
    if inspect.isabstract(target):
        raise TypeError(
            f"Config field '{path}.{HYDRA_TARGET_KEY}' resolved to abstract config "
            f"{target.__qualname__}; select a concrete term config"
        )
    return target


def _target_path(target: type[Any]) -> str:
    return f"{target.__module__}.{target.__qualname__}"


def _prepare_value(value: Any, *, annotation: Any, path: str) -> Any:
    value = _plain(value)
    if isinstance(value, Mapping):
        values = dict(value)
        if HYDRA_TARGET_KEY in values:
            return _prepare_dataclass(values, expected=annotation, path=path, require_target=True)

        candidates = _dataclass_types(annotation)
        if len(candidates) == 1 and not inspect.isabstract(candidates[0]):
            return _prepare_dataclass(values, expected=annotation, path=path, require_target=False)

        value_type = _dict_value_type(annotation)
        return {
            key: _prepare_value(
                item,
                annotation=Any if value_type is _MISSING else value_type,
                path=f"{path}.{key}",
            )
            for key, item in values.items()
        }
    if isinstance(value, list):
        return [
            _prepare_value(item, annotation=Any, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    return value


def _prepare_manager_mapping(value: Any, *, annotation: Any, path: str) -> dict[str, Any]:
    value = _plain(value)
    if not isinstance(value, Mapping):
        raise TypeError(f"Config field '{path}' must be a mapping")
    value_type = _dict_value_type(annotation)
    if value_type is _MISSING:
        raise TypeError(f"Config field '{path}' manager policy requires a typed dict")

    result: dict[str, Any] = {}
    for name, raw_entry in value.items():
        if not isinstance(name, str) or not name:
            raise TypeError(f"Config field '{path}' term names must be non-empty strings")
        entry_path = f"{path}.{name}"
        entry = _plain(raw_entry)
        if entry is None:
            result[name] = None
        elif not isinstance(entry, Mapping):
            raise TypeError(f"Config field '{entry_path}' must be a mapping or None")
        else:
            result[name] = _prepare_dataclass(
                dict(entry),
                expected=value_type,
                path=entry_path,
                require_target=True,
            )
    return result


def _prepare_dataclass(
    values: Mapping[str, Any],
    *,
    expected: Any,
    path: str,
    require_target: bool,
) -> dict[str, Any]:
    values = dict(values)
    reference = values.pop(HYDRA_TARGET_KEY, _MISSING)
    if reference is _MISSING:
        candidates = _dataclass_types(expected)
        if require_target:
            raise ValueError(
                f"Config field '{path}' is a new Manager-Based entry and must declare "
                f"'{HYDRA_TARGET_KEY}'"
            )
        if len(candidates) != 1 or inspect.isabstract(candidates[0]):
            raise ValueError(
                f"Config field '{path}' cannot infer one concrete dataclass type; "
                f"declare '{HYDRA_TARGET_KEY}'"
            )
        target = candidates[0]
        reference = _target_path(target)
    else:
        target = _resolve_target(reference, expected=expected, path=path)

    fields = {field.name: field for field in dataclasses.fields(target) if field.init}
    unknown = [name for name in values if name not in fields]
    if unknown:
        raise ValueError(
            f"Config field '{path}' target {target.__qualname__} has no fields {unknown}"
        )

    hints = _hints(target)
    prepared: dict[str, Any] = {HYDRA_TARGET_KEY: reference}
    for name, raw_value in values.items():
        field = fields[name]
        field_path = f"{path}.{name}"
        annotation = hints.get(name, field.type)
        policy = field.metadata.get(CONFIG_MAPPING_POLICY_KEY)
        if policy == MANAGER_TERM_MAPPING_POLICY:
            prepared[name] = _prepare_manager_mapping(
                raw_value,
                annotation=annotation,
                path=field_path,
            )
        elif name == "func":
            resolved = _resolve(raw_value, path=field_path)
            if not callable(resolved):
                raise TypeError(
                    f"Config field '{field_path}' resolved to {type(resolved).__name__}, "
                    "expected a callable"
                )
            prepared[name] = resolved
        else:
            prepared[name] = _prepare_value(raw_value, annotation=annotation, path=field_path)
    return prepared


def _materialize_entry(value: Mapping[str, Any], *, expected: Any, path: str) -> Any:
    prepared = _prepare_dataclass(value, expected=expected, path=path, require_target=True)
    try:
        result = instantiate(prepared, _convert_="all")
    except Exception as exc:
        raise TypeError(
            f"Config field '{path}' could not construct its typed config: {exc}"
        ) from exc
    expected_types = _dataclass_types(expected)
    if expected_types and not isinstance(result, expected_types):
        raise TypeError(
            f"Config field '{path}' materialized {type(result).__name__}, expected "
            f"{', '.join(item.__qualname__ for item in expected_types)}"
        )
    return result


def _policy(target_obj: Any, name: str) -> str | None:
    if not dataclasses.is_dataclass(target_obj) or isinstance(target_obj, type):
        return None
    field = next((item for item in dataclasses.fields(target_obj) if item.name == name), None)
    value = None if field is None else field.metadata.get(CONFIG_MAPPING_POLICY_KEY)
    return str(value) if value is not None else None


def _is_term_cfg(target_obj: Any) -> bool:
    return (
        dataclasses.is_dataclass(target_obj)
        and not isinstance(target_obj, type)
        and any(
            field.metadata.get(CONFIG_MAPPING_POLICY_KEY) == MANAGER_PARAMS_MAPPING_POLICY
            for field in dataclasses.fields(target_obj)
        )
    )


def _apply_manager_mapping(
    target_obj: Any,
    name: str,
    overrides: Any,
    *,
    annotation: Any,
    policy: str,
) -> None:
    owner = f"{type(target_obj).__name__}.{name}"
    existing = getattr(target_obj, name)
    overrides = _plain(overrides)
    if not isinstance(existing, dict) or not isinstance(overrides, Mapping):
        raise TypeError(f"Config field '{owner}' manager value and override must be mappings")

    if policy == MANAGER_PARAMS_MAPPING_POLICY:
        for param_name, raw_value in overrides.items():
            path = f"{owner}.{param_name}"
            value = _plain(raw_value)
            current = existing.get(param_name, _MISSING)
            if isinstance(value, Mapping) and HYDRA_TARGET_KEY in value:
                existing[param_name] = _materialize_entry(value, expected=Any, path=path)
            elif isinstance(value, Mapping) and dataclasses.is_dataclass(current):
                apply_cfg_overrides(current, value, _path=path)
            else:
                existing[param_name] = _prepare_value(value, annotation=Any, path=path)
        return
    if policy != MANAGER_TERM_MAPPING_POLICY:
        raise ValueError(f"Config field '{owner}' has unknown mapping policy {policy!r}")

    value_type = _dict_value_type(annotation)
    if value_type is _MISSING:
        raise TypeError(f"Config field '{owner}' manager policy requires a typed dict")
    for term_name, raw_value in overrides.items():
        path = f"{owner}.{term_name}"
        value = _plain(raw_value)
        if value is None:
            existing[term_name] = None
        elif not isinstance(value, Mapping):
            raise TypeError(f"Config field '{path}' must be a field mapping or None")
        elif HYDRA_TARGET_KEY in value:
            existing[term_name] = _materialize_entry(value, expected=value_type, path=path)
        else:
            current = existing.get(term_name, _MISSING)
            if current is _MISSING or current is None:
                raise ValueError(
                    f"Config field '{path}' is a new Manager-Based entry and must declare "
                    f"'{HYDRA_TARGET_KEY}'"
                )
            if not dataclasses.is_dataclass(current) or isinstance(current, type):
                raise TypeError(f"Config field '{path}' does not contain a dataclass config")
            apply_cfg_overrides(current, value, _path=path)


def apply_cfg_overrides(
    target_obj: Any,
    overrides: Mapping[str, Any],
    *,
    _path: str | None = None,
) -> None:
    """Apply overrides and materialize explicitly typed Manager-Based entries."""
    overrides = _plain(overrides)
    if not isinstance(overrides, Mapping):
        raise TypeError(f"Config overrides for {type(target_obj).__name__} must be a mapping")
    hints = _hints(type(target_obj)) if dataclasses.is_dataclass(target_obj) else {}
    fields = (
        {field.name: field for field in dataclasses.fields(target_obj)}
        if dataclasses.is_dataclass(target_obj) and not isinstance(target_obj, type)
        else {}
    )
    owner_path = _path or type(target_obj).__name__

    for key, raw_value in overrides.items():
        if not isinstance(key, str) or not hasattr(target_obj, key):
            raise ValueError(f"Config class '{type(target_obj).__name__}' has no attribute '{key}'")
        value = _plain(raw_value)
        existing = getattr(target_obj, key)
        annotation = hints.get(key, fields[key].type if key in fields else Any)
        policy = _policy(target_obj, key)
        if policy is not None:
            _apply_manager_mapping(
                target_obj,
                key,
                value,
                annotation=annotation,
                policy=policy,
            )
            continue
        if key == "func" and _is_term_cfg(target_obj):
            raise ValueError(
                f"Config field '{type(target_obj).__name__}.func' belongs to the typed term "
                "declaration and cannot be changed by a partial override"
            )

        path = f"{owner_path}.{key}"
        if isinstance(value, Mapping):
            if HYDRA_TARGET_KEY in value:
                setattr(target_obj, key, _materialize_entry(value, expected=annotation, path=path))
                continue
            if dataclasses.is_dataclass(existing) and not isinstance(existing, type):
                apply_cfg_overrides(existing, value, _path=path)
                continue
            candidates = _dataclass_types(annotation)
            if existing is None and len(candidates) == 1:
                prepared = _prepare_dataclass(
                    value,
                    expected=annotation,
                    path=path,
                    require_target=False,
                )
                try:
                    setattr(target_obj, key, instantiate(prepared, _convert_="all"))
                except Exception as exc:
                    raise TypeError(
                        f"Config field '{path}' could not construct its typed config: {exc}"
                    ) from exc
                continue
        setattr(target_obj, key, _prepare_value(value, annotation=Any, path=path))


__all__ = ["HYDRA_TARGET_KEY", "apply_cfg_overrides"]
