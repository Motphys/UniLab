# Derived from mujocolab/mjlab v1.6.0 (0fb8a681),
# src/mjlab/envs/mdp/events.py.
# Copyright 2025, The mjlab Developers.
# Modified by UniLab for NumPy reset transactions; Apache-2.0.
# The mass/CoM/gravity public names and signatures follow Isaac Lab v2.2.0;
# their implementation here is UniLab's original payload adapter, not vendored PhysX code.
"""Community-style reset event terms for UniLab's NumPy manager runtime."""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal, cast

import torch

from unilab.managers.event_manager import EventTermCfg
from unilab.managers.manager_base import ManagerTermBase
from unilab.managers.scene_entity_config import SceneEntityCfg
from unilab.managers.torch_rng import TorchManagerRng

if TYPE_CHECKING:
    from unilab.base.entity import Entity
    from unilab.managers._types import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")
_SE3_KEYS = ("x", "y", "z", "roll", "pitch", "yaw")
_XYZ_KEYS = ("x", "y", "z")
_PD_GAIN_PARAM_NAMES = frozenset(("kp_range", "kd_range", "asset_cfg", "distribution", "operation"))
_DISTRIBUTIONS = ("uniform", "log_uniform", "gaussian")
_OPERATIONS = ("add", "scale", "abs")


def _gain_range(
    value: Any,
    *,
    name: str,
    distribution: Literal["uniform", "log_uniform"],
) -> tuple[float, float]:
    try:
        if isinstance(value, (str, bytes)) or len(tuple(value)) != 2:
            raise ValueError(f"pd_gains {name} must have shape (2,)")
        lower, upper = float(value[0]), float(value[1])
    except (TypeError, ValueError) as exc:
        raise TypeError(f"pd_gains {name} must be a numeric (min, max) pair") from exc
    if not (math.isfinite(lower) and math.isfinite(upper)):
        raise ValueError(f"pd_gains {name} must contain only finite values")
    if lower > upper:
        raise ValueError(f"pd_gains {name} minimum {lower} exceeds maximum {upper}")
    if distribution == "log_uniform" and lower <= 0.0:
        raise ValueError(f"pd_gains {name} must be positive for log_uniform sampling")
    return lower, upper


def _gain_choice(
    value: Any,
    *,
    name: str,
    choices: tuple[str, ...],
) -> str:
    if not isinstance(value, str):
        raise TypeError(f"pd_gains {name} must be a string, got {type(value).__name__}")
    if value not in choices:
        raise ValueError(f"pd_gains {name} must be one of {choices}, got {value!r}")
    return value


def _sample_gain_range(
    rng: TorchManagerRng,
    bounds: tuple[float, float],
    shape: tuple[int, int],
    distribution: Literal["uniform", "log_uniform"],
) -> torch.Tensor:
    if distribution == "uniform":
        return rng.uniform(bounds[0], bounds[1], shape, dtype=torch.float64)
    return rng.uniform(math.log(bounds[0]), math.log(bounds[1]), shape, dtype=torch.float64).exp()


def _sample_se3_range(
    range_dict: dict[str, tuple[float, float]] | None,
    shape: tuple[int, ...],
    rng: TorchManagerRng,
) -> torch.Tensor:
    """Sample uniform ``[x, y, z, roll, pitch, yaw]`` offsets with Torch RNG."""
    if not shape or shape[-1] != len(_SE3_KEYS):
        raise ValueError(
            f"reset_root_state_uniform SE(3) sample shape must end in 6; received {shape}"
        )
    try:
        ranges = torch.tensor(
            [(range_dict or {}).get(key, (0.0, 0.0)) for key in _SE3_KEYS],
            dtype=torch.float64,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "reset_root_state_uniform ranges must map each SE(3) key to a numeric (min, max) pair"
        ) from exc
    if tuple(ranges.shape) != (len(_SE3_KEYS), 2):
        raise ValueError(
            "reset_root_state_uniform ranges must map each SE(3) key to a "
            f"(min, max) pair; received shape {ranges.shape}"
        )
    if not bool(torch.isfinite(ranges).all()):
        raise ValueError("reset_root_state_uniform ranges must contain only finite values")
    invalid = ranges[:, 0] > ranges[:, 1]
    if bool(invalid.any()):
        keys = [_SE3_KEYS[index] for index in torch.nonzero(invalid).flatten().tolist()]
        raise ValueError(f"reset_root_state_uniform range minimum exceeds maximum for keys {keys}")
    lower = ranges[:, 0].to(device=rng.device)
    upper = ranges[:, 1].to(device=rng.device)
    return rng.uniform(lower, upper, shape, dtype=torch.float64)


def _event_choice(value: Any, *, term_name: str, name: str, choices: tuple[str, ...]) -> str:
    if not isinstance(value, str):
        raise TypeError(
            f"EventManager term '{term_name}' parameter '{name}' must be a string, "
            f"got {type(value).__name__}"
        )
    if value not in choices:
        raise ValueError(
            f"EventManager term '{term_name}' parameter '{name}' must be one of "
            f"{choices}, got {value!r}"
        )
    return value


def _distribution_parameters(
    value: Any,
    *,
    term_name: str,
    name: str,
    width: int | None = None,
    distribution: str,
) -> torch.Tensor:
    try:
        params = torch.as_tensor(value, dtype=torch.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"EventManager term '{term_name}' parameter '{name}' must contain numeric values"
        ) from exc
    expected = (2,) if width is None else (2, width)
    if tuple(params.shape) != expected:
        raise ValueError(
            f"EventManager term '{term_name}' parameter '{name}' has shape {params.shape}; "
            f"expected {expected}"
        )
    if not bool(torch.isfinite(params).all()):
        raise ValueError(
            f"EventManager term '{term_name}' parameter '{name}' must contain only finite values"
        )
    if distribution == "gaussian":
        if bool((params[1] < 0.0).any()):
            raise ValueError(
                f"EventManager term '{term_name}' parameter '{name}' standard deviation "
                "must be non-negative"
            )
    else:
        if bool((params[0] > params[1]).any()):
            raise ValueError(
                f"EventManager term '{term_name}' parameter '{name}' lower bound exceeds upper bound"
            )
        if distribution == "log_uniform" and bool((params <= 0.0).any()):
            raise ValueError(
                f"EventManager term '{term_name}' parameter '{name}' must be positive "
                "for log_uniform sampling"
            )
    return params.clone()


def _sample_distribution(
    rng: TorchManagerRng,
    params: torch.Tensor,
    shape: tuple[int, ...],
    distribution: str,
) -> torch.Tensor:
    low = params[0].to(device=rng.device, dtype=torch.float64)
    high = params[1].to(device=rng.device, dtype=torch.float64)
    if distribution == "gaussian":
        return rng.normal(low, high, shape, dtype=torch.float64)
    if distribution == "log_uniform":
        return rng.uniform(torch.log(low), torch.log(high), shape, dtype=torch.float64).exp()
    return rng.uniform(low, high, shape, dtype=torch.float64)


def _selected_reset_defaults_tensor(
    defaults: torch.Tensor,
    env_ids: torch.Tensor,
    *,
    canonical_ndim: int,
    device: torch.device,
) -> torch.Tensor:
    """Select entity-local event defaults without a host row conversion."""
    table = defaults
    rows = env_ids.to(dtype=torch.int64, device=device)
    selected = table.to(dtype=torch.float64, device=device)
    if table.ndim == canonical_ndim:
        return selected.expand((rows.numel(), *selected.shape)).clone()
    if table.ndim == canonical_ndim + 1:
        return selected.index_select(0, rows)
    raise ValueError(
        f"Selected event default table has shape {tuple(table.shape)}; expected "
        f"a {canonical_ndim}-D selected-column table or a per-environment "
        f"(*, {canonical_ndim}) table"
    )


def _model_field_default_rows(
    defaults: torch.Tensor,
    env_ids: torch.Tensor,
    *,
    field_width: int,
    device: torch.device,
) -> torch.Tensor:
    """Select entity-local model-field defaults without a host row conversion."""
    table = defaults
    canonical_ndim = 1 if field_width == 1 else 2
    canonical = table.ndim == canonical_ndim
    rows = env_ids.to(dtype=torch.int64, device=device)
    if canonical:
        selected = table.to(device=device, dtype=torch.float64)
    elif table.ndim == canonical_ndim + 1:
        selected = table.to(device=device, dtype=torch.float64).index_select(0, rows)
    else:
        raise ValueError(
            f"Selected model-field default table has shape {tuple(table.shape)}; expected "
            f"a {canonical_ndim}-D canonical table or a per-environment "
            f"(*, {canonical_ndim}) table"
        )
    if canonical:
        return selected.expand((rows.numel(), *selected.shape)).clone()
    return selected


def _require_torch_rng(env: ManagerBasedRlEnv, term_name: str) -> TorchManagerRng:
    rng = getattr(env, "torch_rng", None)
    if not isinstance(rng, TorchManagerRng):
        raise RuntimeError(
            f"EventManager term '{term_name}' requires the Manager-owned Torch generator"
        )
    return rng


def _axis_ranges(
    value: Any,
    *,
    term_name: str,
    name: str,
    keys: tuple[str, ...],
) -> torch.Tensor:
    if not isinstance(value, dict):
        raise TypeError(f"EventManager term '{term_name}' parameter '{name}' must be a dict")
    unknown = sorted(set(value) - set(keys))
    if unknown:
        raise ValueError(
            f"EventManager term '{term_name}' parameter '{name}' has unknown axes {unknown}"
        )
    parameters = [
        [value.get(key, (0.0, 0.0))[0] for key in keys],
        [value.get(key, (0.0, 0.0))[1] for key in keys],
    ]
    return _distribution_parameters(
        parameters,
        term_name=term_name,
        name=name,
        width=len(keys),
        distribution="uniform",
    ).T


def _validate_event_term(
    cfg: EventTermCfg,
    *,
    term_name: str,
    mode: str | tuple[str, ...],
    allowed_params: frozenset[str],
    required_params: tuple[str, ...],
) -> None:
    supported_modes = (mode,) if isinstance(mode, str) else mode
    if cfg.mode not in supported_modes:
        raise NotImplementedError(
            "EventManager term "
            f"'{term_name}' only supports mode='{'/'.join(supported_modes)}' on the UniLab runtime"
        )
    unknown = sorted(set(cfg.params) - allowed_params)
    if unknown:
        raise ValueError(f"EventManager term '{term_name}' has unknown parameters {unknown}")
    missing = [name for name in required_params if name not in cfg.params]
    if missing:
        raise ValueError(f"EventManager term '{term_name}' is missing parameters {missing}")


def resolve_env_ids(env: ManagerBasedRlEnv, env_ids: torch.Tensor | None) -> torch.Tensor:
    """Return CPU Torch rows for the legacy host event boundary."""
    if env_ids is None:
        return torch.arange(env.num_envs, dtype=torch.int64)
    if isinstance(env_ids, torch.Tensor):
        return env_ids.detach().cpu().to(dtype=torch.int64)
    return torch.as_tensor(env_ids, dtype=torch.int64)


def _device_reset_table(values: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Materialize one immutable default/parameter table on a device."""
    return values.to(device=device, dtype=torch.float32)


def _sample_distribution_tensor(
    rng: TorchManagerRng,
    low: torch.Tensor,
    high: torch.Tensor,
    shape: tuple[int, ...],
    distribution: str,
) -> torch.Tensor:
    """Sample one DR distribution on the Torch RNG's device."""
    if distribution == "gaussian":
        return rng.normal(low, high, shape)
    if distribution == "log_uniform":
        return rng.uniform(low.log(), high.log(), shape).exp()
    return rng.uniform(low, high, shape)


def _apply_randomization_operation_tensor(
    default: torch.Tensor,
    samples: torch.Tensor,
    operation: str,
) -> torch.Tensor:
    """Compose sampled multipliers/offsets with default rows on device."""
    if operation == "add":
        return default + samples
    if operation == "scale":
        return default * samples
    return samples


def _sample_axis_tensor(
    rng: TorchManagerRng,
    parameters: torch.Tensor,
    count: int,
    *,
    distribution: str,
    shared_random: bool,
) -> torch.Tensor:
    if not shared_random:
        return _sample_distribution_tensor(
            rng,
            parameters[:, 0],
            parameters[:, 1],
            (count, parameters.shape[0]),
            distribution,
        )
    samples = torch.empty(
        (count, parameters.shape[0]), dtype=parameters.dtype, device=parameters.device
    )
    groups: dict[tuple[float, float], list[int]] = {}
    for index, pair in enumerate(parameters.detach().cpu().tolist()):
        groups.setdefault((float(pair[0]), float(pair[1])), []).append(index)
    for low, high, indices in _shared_parameter_groups(parameters, groups):
        shared = _sample_distribution_tensor(
            rng,
            low,
            high,
            (count, 1),
            distribution,
        ).to(dtype=samples.dtype)
        samples[:, indices] = shared
    return samples


def _shared_parameter_groups(
    parameters: torch.Tensor,
    groups: Mapping[tuple[float, float], Sequence[int]],
) -> Iterable[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Turn host-scalar shared bounds into device broadcast tensors."""
    device = parameters.device
    for (low, high), indices in groups.items():
        yield (
            torch.full((1,), low, dtype=parameters.dtype, device=device),
            torch.full((1,), high, dtype=parameters.dtype, device=device),
            torch.as_tensor(list(indices), dtype=torch.int64, device=device),
        )


class _ResetRandomizationTerm(ManagerTermBase):
    """Reset DR base delegating the mask reset to the packed transaction.

    The environment owns the packed reset transaction
    (``ResetStateTransaction.scoped_device_event_tensor``); ``reset`` resolves
    the selected rows from the fixed-width mask and reuses the term's
    ``__call__`` sampling and facade staging, so the whole lifecycle composes
    into the same exactly-once commit. Backends missing a field's reset
    capability still fail closed at construction binding.
    """

    def reset(
        self,
        env: ManagerBasedRlEnv,
        reset_mask: torch.Tensor,
        **kwargs: Any,
    ) -> None:
        rows = torch.nonzero(reset_mask, as_tuple=False).flatten()
        self(env, rows, **kwargs)


class _ModelFieldRandomizer(_ResetRandomizationTerm):
    """Pinned mjlab-style model-field DR terms with Torch sampling tables."""

    _term_name = "model_field"
    _field_width = 1
    _default_axes: tuple[int, ...] = (0,)
    _valid_axes: tuple[int, ...] = (0,)
    _PARAMS = frozenset(
        ("ranges", "asset_cfg", "distribution", "operation", "axes", "shared_random")
    )

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        _validate_event_term(
            cfg,
            term_name=self._term_name,
            mode="reset",
            allowed_params=self._PARAMS,
            required_params=("ranges",),
        )
        self._distribution = _event_choice(
            cfg.params.get("distribution", "uniform"),
            term_name=self._term_name,
            name="distribution",
            choices=_DISTRIBUTIONS,
        )
        self._operation = _event_choice(
            cfg.params.get("operation", "abs"),
            term_name=self._term_name,
            name="operation",
            choices=_OPERATIONS,
        )
        shared_random = cfg.params.get("shared_random", False)
        if not isinstance(shared_random, bool):
            raise TypeError(
                f"EventManager term '{self._term_name}' parameter 'shared_random' must be bool"
            )
        self._shared_random = shared_random
        asset_cfg = cfg.params.get("asset_cfg", _DEFAULT_ASSET_CFG)
        if not isinstance(asset_cfg, SceneEntityCfg):
            raise TypeError(
                f"EventManager term '{self._term_name}' asset_cfg must be SceneEntityCfg, "
                f"got {type(asset_cfg).__name__}"
            )
        entity = cast("Entity", env.scene[asset_cfg.name])
        local_ids, defaults, names = self._bind(entity, asset_cfg)
        ranges = cfg.params["ranges"]
        local_ids, defaults, names, ranges = self._select_string_ranges(
            local_ids,
            defaults,
            names,
            ranges,
        )
        self._entity = entity
        self._local_ids = local_ids
        self._defaults = defaults
        self._axes = self._resolve_axes(cfg.params.get("axes"), ranges)
        self._ranges = self._resolve_ranges(ranges, names)
        self._device_ranges: torch.Tensor | None = None
        self._device_axis_groups: dict[
            int, list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
        ] = {}

    def _bind(
        self,
        entity: Entity,
        asset_cfg: SceneEntityCfg,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[str, ...]]:
        raise NotImplementedError

    def _write(
        self,
        values: torch.Tensor,
        env_ids: torch.Tensor,
    ) -> None:
        raise NotImplementedError

    def _write_tensor(self, values: torch.Tensor, env_ids: torch.Tensor) -> None:
        raise NotImplementedError

    def _select_string_ranges(
        self,
        local_ids: torch.Tensor,
        defaults: torch.Tensor,
        names: tuple[str, ...],
        ranges: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[str, ...], Any]:
        if not isinstance(ranges, dict) or not ranges:
            return local_ids, defaults, names, ranges
        keys = tuple(ranges)
        if not all(isinstance(key, str) for key in keys):
            if any(isinstance(key, str) for key in keys):
                raise TypeError(
                    f"EventManager term '{self._term_name}' ranges cannot mix string and integer keys"
                )
            return local_ids, defaults, names, ranges

        assigned: list[Any | None] = [None] * len(names)
        for pattern, bounds in ranges.items():
            try:
                matched = [index for index, name in enumerate(names) if re.fullmatch(pattern, name)]
            except re.error as exc:
                raise ValueError(
                    f"EventManager term '{self._term_name}' ranges contains invalid regex "
                    f"{pattern!r}: {exc}"
                ) from exc
            if not matched:
                raise ValueError(
                    f"EventManager term '{self._term_name}' ranges pattern {pattern!r} "
                    f"matched no selected names; available={list(names)}"
                )
            for index in matched:
                if assigned[index] is not None:
                    raise ValueError(
                        f"EventManager term '{self._term_name}' ranges patterns overlap for "
                        f"selected name '{names[index]}'"
                    )
                assigned[index] = bounds
        selected = [index for index, value in enumerate(assigned) if value is not None]
        selected_tensor = torch.tensor(selected, dtype=torch.int64)
        canonical_ndim = 1 if self._field_width == 1 else 2
        if defaults.ndim == canonical_ndim:
            selected_defaults = defaults.index_select(0, selected_tensor)
        elif defaults.ndim == canonical_ndim + 1:
            selected_defaults = defaults.index_select(1, selected_tensor)
        else:
            raise ValueError(
                f"EventManager term '{self._term_name}' bound an invalid default table with "
                f"shape {defaults.shape}"
            )
        return (
            local_ids.index_select(0, selected_tensor),
            selected_defaults,
            tuple(names[index] for index in selected),
            [assigned[index] for index in selected],
        )

    def _resolve_axes(self, value: Any, ranges: Any) -> tuple[int, ...]:
        if value is None and isinstance(ranges, dict) and ranges:
            value = list(ranges)
        if value is None:
            axes = self._default_axes
        else:
            if not isinstance(value, (list, tuple)):
                raise TypeError(
                    f"EventManager term '{self._term_name}' parameter 'axes' must be a sequence"
                )
            if any(isinstance(axis, bool) or not isinstance(axis, int) for axis in value):
                raise TypeError(
                    f"EventManager term '{self._term_name}' parameter 'axes' must contain integers"
                )
            axes = tuple(int(axis) for axis in value)
        if not axes:
            raise ValueError(
                f"EventManager term '{self._term_name}' parameter 'axes' cannot be empty"
            )
        if len(set(axes)) != len(axes):
            raise ValueError(
                f"EventManager term '{self._term_name}' parameter 'axes' contains duplicates"
            )
        invalid = sorted(set(axes) - set(self._valid_axes))
        if invalid:
            raise ValueError(
                f"EventManager term '{self._term_name}' has invalid axes {invalid}; "
                f"valid axes are {list(self._valid_axes)}"
            )
        return axes

    def _resolve_ranges(self, value: Any, names: tuple[str, ...]) -> torch.Tensor:
        per_entity: list[Any]
        if (
            isinstance(value, list)
            and len(value) == len(names)
            and any(isinstance(item, (tuple, list, torch.Tensor)) for item in value)
        ):
            per_entity = list(value)
            value = None
        else:
            per_entity = []

        result = torch.full(
            (len(names), self._field_width, 2),
            float("nan"),
            dtype=torch.float64,
        )
        if per_entity:
            if len(self._axes) != 1:
                raise ValueError(
                    f"EventManager term '{self._term_name}' string-keyed ranges require one axis"
                )
            for index, bounds in enumerate(per_entity):
                result[index, self._axes[0]] = _distribution_parameters(
                    bounds,
                    term_name=self._term_name,
                    name=f"ranges[{names[index]}]",
                    distribution=self._distribution,
                )
        elif isinstance(value, dict):
            unknown = sorted(set(value) - set(self._axes))
            missing = sorted(set(self._axes) - set(value))
            if unknown or missing:
                raise ValueError(
                    f"EventManager term '{self._term_name}' ranges axes mismatch; "
                    f"missing={missing}, unknown={unknown}"
                )
            for axis in self._axes:
                result[:, axis] = _distribution_parameters(
                    value[axis],
                    term_name=self._term_name,
                    name=f"ranges[{axis}]",
                    distribution=self._distribution,
                )
        else:
            parameters = _distribution_parameters(
                value,
                term_name=self._term_name,
                name="ranges",
                distribution=self._distribution,
            )
            for axis in self._axes:
                result[:, axis] = parameters
        return result

    def _build_axis_groups(
        self, axis: int
    ) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Group selected columns sharing identical bounds for one axis on device."""
        ranges = self._device_ranges
        assert ranges is not None
        parameters = self._ranges[:, axis].to(device=ranges.device)
        groups: dict[tuple[float, float], list[int]] = {}
        for index, pair in enumerate(parameters.detach().cpu().tolist()):
            groups.setdefault((float(pair[0]), float(pair[1])), []).append(index)
        return list(_shared_parameter_groups(parameters, groups))

    def _sample_axis_tensor(self, rng: TorchManagerRng, axis: int, count: int) -> torch.Tensor:
        """Sample one axis on the RNG device, honoring shared_random groups."""
        ranges = self._device_ranges
        assert ranges is not None
        parameters = ranges[:, axis]
        if not self._shared_random:
            return _sample_distribution_tensor(
                rng,
                parameters[:, 0],
                parameters[:, 1],
                (count, parameters.shape[0]),
                self._distribution,
            )
        groups = self._device_axis_groups.get(axis)
        if groups is None:
            groups = self._build_axis_groups(axis)
            self._device_axis_groups[axis] = groups
        samples = torch.empty(
            (count, parameters.shape[0]), dtype=torch.float32, device=parameters.device
        )
        for low, high, indices in groups:
            shared = _sample_distribution_tensor(rng, low, high, (count, 1), self._distribution)
            samples[:, indices] = shared
        return samples

    def _sample_device(self, env: ManagerBasedRlEnv) -> torch.device:
        return _require_torch_rng(env, self.name).device

    def _sample_axis(self, env: ManagerBasedRlEnv, axis: int, count: int) -> torch.Tensor:
        parameters = self._ranges[:, axis].to(device=self._sample_device(env))
        rng = _require_torch_rng(env, self.name)
        return _sample_axis_tensor(
            rng,
            parameters,
            count,
            distribution=self._distribution,
            shared_random=self._shared_random,
        )

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        env_ids: torch.Tensor | None,
        ranges: Any,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
        distribution: str = "uniform",
        operation: str = "abs",
        axes: list[int] | None = None,
        shared_random: bool = False,
    ) -> None:
        del ranges, asset_cfg, distribution, operation, axes, shared_random
        ids = resolve_env_ids(env, env_ids)
        default_values = _model_field_default_rows(
            self._defaults,
            ids,
            field_width=self._field_width,
            device=self._sample_device(env),
        )
        sampled_axis = (
            self._sample_axis(env, self._axes[0], ids.numel()).to(device=default_values.device)
            if self._axes
            else None
        )
        values = default_values.clone()
        for index, axis in enumerate(self._axes):
            samples = sampled_axis if index == 0 else self._sample_axis(env, axis, ids.numel())
            if self._field_width == 1:
                values = _apply_randomization_operation_tensor(
                    default_values,
                    samples,
                    self._operation,
                )
            else:
                values[..., axis] = _apply_randomization_operation_tensor(
                    default_values[..., axis],
                    samples,
                    self._operation,
                )
        if bool((values < 0.0).any()) or not bool(torch.isfinite(values).all()):
            raise ValueError(
                f"EventManager term '{self._term_name}' produced negative, NaN, or Inf values"
            )
        self._write(values.cpu(), ids)


class GeomFriction(_ModelFieldRandomizer):
    """Pinned mjlab-style geom friction randomization through the reset payload."""

    _term_name = "geom_friction"
    _field_width = 3
    _default_axes = (0,)
    _valid_axes = (0, 1, 2)

    def _bind(
        self,
        entity: Entity,
        asset_cfg: SceneEntityCfg,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[str, ...]]:
        local_ids, defaults = entity.bind_geom_friction_write(
            asset_cfg.geom_ids,
            term_name=self._term_name,
        )
        names = tuple(entity.geom_names[int(index)] for index in local_ids)
        return local_ids, defaults, names

    def _write(self, values: torch.Tensor, env_ids: torch.Tensor) -> None:
        self._entity.write_geom_friction_to_sim(
            values,
            self._local_ids,
            env_ids,
            term_name=self._term_name,
        )


class JointArmature(_ModelFieldRandomizer):
    """Pinned mjlab-style joint armature randomization through the reset payload."""

    _term_name = "joint_armature"

    def _bind(
        self,
        entity: Entity,
        asset_cfg: SceneEntityCfg,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[str, ...]]:
        local_ids, defaults = entity.bind_joint_armature_write(
            asset_cfg.joint_ids,
            term_name=self._term_name,
        )
        names = tuple(entity.joint_names[int(index)] for index in local_ids)
        return local_ids, defaults, names

    def _write(self, values: torch.Tensor, env_ids: torch.Tensor) -> None:
        self._entity.write_joint_armature_to_sim(
            values,
            self._local_ids,
            env_ids,
            term_name=self._term_name,
        )


geom_friction = GeomFriction
joint_armature = JointArmature
dof_armature = joint_armature


class PdGains(_ResetRandomizationTerm):
    """Pinned-mjlab-compatible PD gain randomization on UniLab reset payloads."""

    mask_reset_supported = True

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        if cfg.mode != "reset":
            raise NotImplementedError(
                "EventManager term 'pd_gains' only supports mode='reset' on the UniLab "
                "set_state transaction; startup/interval/step model-field mutation is unavailable"
            )
        unknown = sorted(set(cfg.params) - _PD_GAIN_PARAM_NAMES)
        if unknown:
            raise ValueError(f"EventManager term 'pd_gains' has unknown parameters {unknown}")
        missing = [name for name in ("kp_range", "kd_range") if name not in cfg.params]
        if missing:
            raise ValueError(f"EventManager term 'pd_gains' is missing parameters {missing}")

        distribution = cast(
            Literal["uniform", "log_uniform"],
            _gain_choice(
                cfg.params.get("distribution", "uniform"),
                name="distribution",
                choices=("uniform", "log_uniform"),
            ),
        )
        self._operation = cast(
            Literal["scale", "abs"],
            _gain_choice(
                cfg.params.get("operation", "scale"),
                name="operation",
                choices=("scale", "abs"),
            ),
        )
        self._distribution = distribution
        self._kp_range = _gain_range(
            cfg.params["kp_range"],
            name="kp_range",
            distribution=distribution,
        )
        self._kd_range = _gain_range(
            cfg.params["kd_range"],
            name="kd_range",
            distribution=distribution,
        )
        asset_cfg = cfg.params.get("asset_cfg", _DEFAULT_ASSET_CFG)
        if not isinstance(asset_cfg, SceneEntityCfg):
            raise TypeError(
                "EventManager term 'pd_gains' asset_cfg must be SceneEntityCfg, got "
                f"{type(asset_cfg).__name__}"
            )
        self._entity = cast("Entity", env.scene[asset_cfg.name])
        self._actuator_ids, self._default_kp, self._default_kd = (
            self._entity.bind_actuator_gain_write(
                asset_cfg.actuator_ids,
                term_name="pd_gains",
            )
        )

    def reset(
        self,
        env: ManagerBasedRlEnv,
        reset_mask: torch.Tensor,
        **kwargs: object,
    ) -> None:
        """Sample full-width final gains and commit only masked rows."""
        del kwargs
        ids = torch.arange(env.num_envs, dtype=torch.int64, device=reset_mask.device)
        shape = (ids.numel(), len(self._actuator_ids))
        kp = _sample_gain_range(
            _require_torch_rng(env, "pd_gains"), self._kp_range, shape, self._distribution
        )
        kd = _sample_gain_range(
            _require_torch_rng(env, "pd_gains"), self._kd_range, shape, self._distribution
        )
        if self._operation == "scale":
            default_kp = _selected_reset_defaults_tensor(
                self._default_kp,
                ids,
                canonical_ndim=1,
                device=kp.device,
            )
            default_kd = _selected_reset_defaults_tensor(
                self._default_kd,
                ids,
                canonical_ndim=1,
                device=kd.device,
            )
            kp = kp * default_kp.to(dtype=kp.dtype, device=kp.device)
            kd = kd * default_kd.to(dtype=kd.dtype, device=kd.device)
        kp = kp.to(dtype=torch.float32).contiguous()
        kd = kd.to(dtype=torch.float32).contiguous()
        self._entity.write_actuator_gains_tensor_mask(
            reset_mask,
            kp,
            kd,
            actuator_ids=self._actuator_ids,
            term_name="pd_gains",
        )

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        env_ids: torch.Tensor | None,
        kp_range: tuple[float, float],
        kd_range: tuple[float, float],
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
        distribution: Literal["uniform", "log_uniform"] = "uniform",
        operation: Literal["scale", "abs"] = "scale",
    ) -> None:
        del kp_range, kd_range, asset_cfg, distribution, operation
        ids = resolve_env_ids(env, env_ids)
        shape = (ids.numel(), len(self._actuator_ids))
        kp = _sample_gain_range(
            _require_torch_rng(env, "pd_gains"), self._kp_range, shape, self._distribution
        )
        kd = _sample_gain_range(
            _require_torch_rng(env, "pd_gains"), self._kd_range, shape, self._distribution
        )
        if self._operation == "scale":
            default_kp = _selected_reset_defaults_tensor(
                self._default_kp,
                ids,
                canonical_ndim=1,
                device=kp.device,
            )
            default_kd = _selected_reset_defaults_tensor(
                self._default_kd,
                ids,
                canonical_ndim=1,
                device=kd.device,
            )
            kp = kp * default_kp.to(dtype=kp.dtype, device=kp.device)
            kd = kd * default_kd.to(dtype=kd.dtype, device=kd.device)
        self._entity.write_actuator_gains_to_sim(
            kp.cpu(),
            kd.cpu(),
            actuator_ids=self._actuator_ids,
            env_ids=ids,
            term_name="pd_gains",
        )


pd_gains = PdGains


class RandomizeRigidBodyMass(_ResetRandomizationTerm):
    """Community-compatible body-mass randomization via reset/startup payloads."""

    _PARAMS = frozenset(
        (
            "asset_cfg",
            "mass_distribution_params",
            "operation",
            "distribution",
            "recompute_inertia",
            "min_mass",
        )
    )
    uses_startup_rows = True

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        term_name = "randomize_rigid_body_mass"
        _validate_event_term(
            cfg,
            term_name=term_name,
            mode=("reset", "startup"),
            allowed_params=self._PARAMS,
            required_params=("asset_cfg", "mass_distribution_params", "operation"),
        )
        asset_cfg = cfg.params["asset_cfg"]
        if not isinstance(asset_cfg, SceneEntityCfg):
            raise TypeError(
                f"EventManager term '{term_name}' asset_cfg must be SceneEntityCfg, "
                f"got {type(asset_cfg).__name__}"
            )
        recompute_inertia = cfg.params.get("recompute_inertia", True)
        if not isinstance(recompute_inertia, bool):
            raise TypeError(f"EventManager term '{term_name}' recompute_inertia must be bool")
        if recompute_inertia:
            raise NotImplementedError(
                f"EventManager term '{term_name}' recompute_inertia=True is unavailable: "
                "ResetRandomizationPayload has no inertia-recomputation contract; set it "
                "explicitly to false or do not configure this term"
            )
        self._operation = _event_choice(
            cfg.params["operation"],
            term_name=term_name,
            name="operation",
            choices=_OPERATIONS,
        )
        self._distribution = _event_choice(
            cfg.params.get("distribution", "uniform"),
            term_name=term_name,
            name="distribution",
            choices=_DISTRIBUTIONS,
        )
        self._distribution_params = _distribution_parameters(
            cfg.params["mass_distribution_params"],
            term_name=term_name,
            name="mass_distribution_params",
            distribution=self._distribution,
        )
        min_mass = cfg.params.get("min_mass", 1e-6)
        if isinstance(min_mass, bool) or not isinstance(min_mass, (int, float)):
            raise TypeError(f"EventManager term '{term_name}' min_mass must be numeric")
        self._min_mass = float(min_mass)
        if not math.isfinite(self._min_mass) or self._min_mass < 1e-6:
            raise ValueError(
                f"EventManager term '{term_name}' min_mass must be finite and at least 1e-6"
            )
        self._entity = cast("Entity", env.scene[asset_cfg.name])
        self._body_ids, self._default_mass = self._entity.bind_body_mass_write(
            asset_cfg.body_ids,
            term_name=term_name,
        )
        self._startup = cfg.mode == "startup"
        self._startup_values: torch.Tensor | None = None

    def _apply_startup(self, env: ManagerBasedRlEnv, env_ids: torch.Tensor | None) -> None:
        """Sample every world once and replay the immutable startup table.

        Manager startup always supplies every environment row. The event is
        full-width so the backend pays its derived-constant refresh once and
        later mask resets can omit this field entirely.
        """
        ids = resolve_env_ids(env, env_ids)
        if not torch.equal(ids, torch.arange(env.num_envs, dtype=torch.int64)):
            raise ValueError(
                "EventManager term 'randomize_rigid_body_mass' startup mode requires all "
                f"environment rows; got {ids.tolist()} for {env.num_envs} environments"
            )
        if self._startup_values is None:
            sampled = _sample_distribution(
                _require_torch_rng(env, "randomize_rigid_body_mass"),
                self._distribution_params,
                (env.num_envs, self._body_ids.size),
                self._distribution,
            )
            default_mass = _selected_reset_defaults_tensor(
                self._default_mass,
                ids,
                canonical_ndim=1,
                device=sampled.device,
            )
            values = _apply_randomization_operation_tensor(
                default_mass,
                sampled,
                self._operation,
            ).clamp_min(self._min_mass)
            if not bool(torch.isfinite(values).all()):
                raise ValueError(
                    "EventManager term 'randomize_rigid_body_mass' produced NaN or Inf "
                    "startup values"
                )
            self._startup_values = values
        self._entity.record_startup_body_mass_to_sim(
            self._startup_values.cpu(),
            body_ids=self._body_ids,
            env_ids=ids,
            term_name="randomize_rigid_body_mass",
        )

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg,
        mass_distribution_params: tuple[float, float],
        operation: Literal["add", "scale", "abs"],
        distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
        recompute_inertia: bool = True,
        min_mass: float = 1e-6,
    ) -> None:
        del (
            asset_cfg,
            mass_distribution_params,
            operation,
            distribution,
            recompute_inertia,
            min_mass,
        )
        if self._startup:
            self._apply_startup(env, env_ids)
            return
        ids = resolve_env_ids(env, env_ids)
        sampled = _sample_distribution(
            _require_torch_rng(env, "randomize_rigid_body_mass"),
            self._distribution_params,
            (ids.numel(), self._body_ids.size),
            self._distribution,
        )
        default_mass = _selected_reset_defaults_tensor(
            self._default_mass,
            ids,
            canonical_ndim=1,
            device=sampled.device,
        )
        values = (
            _apply_randomization_operation_tensor(
                default_mass,
                sampled,
                self._operation,
            )
            .clamp_min(self._min_mass)
            .cpu()
        )
        self._entity.write_body_mass_to_sim(
            values,
            body_ids=self._body_ids,
            env_ids=ids,
            term_name="randomize_rigid_body_mass",
        )


randomize_rigid_body_mass = RandomizeRigidBodyMass


class RandomizeBodyMassInertia(_ResetRandomizationTerm):
    """Startup-style mass+inertia scaling via one shared per-env factor.

    NumPy adapter for the alpha-only slice of mjlab's ``dr.pseudo_inertia``:
    mass and principal inertia of the selected bodies are multiplied by the
    same factor s = e^{2α} with α ~ U(ln(lo)/2, ln(hi)/2) — i.e. s is
    log-uniform in ``scale_range``; the CoM (``body_ipos``) and the principal
    frame (``body_iquat``) stay untouched.

    Upstream runs this as a startup event (fixed per env for the whole run).
    UniLab startup events have no reset-transaction write path, so this term
    runs in reset mode but samples the factor only once — at the first reset —
    and reapplies the cached per-env values on every later reset. Both writes
    re-derive from immutable backend-authoritative defaults, so reapplication
    is idempotent and non-accumulating.
    """

    _PARAMS = frozenset(("asset_cfg", "scale_range"))

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        term_name = "randomize_body_mass_inertia"
        _validate_event_term(
            cfg,
            term_name=term_name,
            mode="reset",
            allowed_params=self._PARAMS,
            required_params=("asset_cfg", "scale_range"),
        )
        asset_cfg = cfg.params["asset_cfg"]
        if not isinstance(asset_cfg, SceneEntityCfg):
            raise TypeError(
                f"EventManager term '{term_name}' asset_cfg must be SceneEntityCfg, "
                f"got {type(asset_cfg).__name__}"
            )
        bounds = _distribution_parameters(
            cfg.params["scale_range"],
            term_name=term_name,
            name="scale_range",
            distribution="log_uniform",
        )
        self._scale_lo = float(bounds[0])
        self._scale_hi = float(bounds[1])
        self._entity = cast("Entity", env.scene[asset_cfg.name])
        self._body_ids, self._default_mass = self._entity.bind_body_mass_write(
            asset_cfg.body_ids,
            term_name=term_name,
        )
        inertia_ids, self._default_inertia = self._entity.bind_body_inertia_write(
            asset_cfg.body_ids,
            term_name=term_name,
        )
        if self._body_ids.tolist() != inertia_ids.tolist():
            raise RuntimeError(
                f"EventManager term '{term_name}' mass/inertia body bindings diverged: "
                f"{self._body_ids.tolist()} != {inertia_ids.tolist()}"
            )
        if self._default_mass.ndim != self._default_inertia.ndim - 1:
            raise ValueError(
                f"EventManager term '{term_name}' mass and inertia default tables use "
                "incompatible canonical/per-environment layouts"
            )
        self._scales: torch.Tensor | None = None

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg,
        scale_range: tuple[float, float],
    ) -> None:
        del asset_cfg, scale_range
        ids = resolve_env_ids(env, env_ids)
        if self._scales is None:
            alpha = _require_torch_rng(env, "randomize_body_mass_inertia").uniform(
                math.log(self._scale_lo) / 2.0,
                math.log(self._scale_hi) / 2.0,
                (env.num_envs, self._body_ids.size),
                dtype=torch.float64,
            )
            self._scales = torch.exp(2.0 * alpha)
        rows = ids.to(dtype=torch.int64)
        scales = self._scales.index_select(0, rows).to(device=self._scales.device)
        default_mass = _selected_reset_defaults_tensor(
            self._default_mass,
            ids,
            canonical_ndim=1,
            device=self._scales.device,
        )
        default_inertia = _selected_reset_defaults_tensor(
            self._default_inertia,
            ids,
            canonical_ndim=2,
            device=self._scales.device,
        )
        self._entity.write_body_mass_to_sim(
            (default_mass * scales).cpu(),
            body_ids=self._body_ids,
            env_ids=ids,
            term_name="randomize_body_mass_inertia",
        )
        self._entity.write_body_inertia_to_sim(
            (default_inertia * scales[..., None]).cpu(),
            body_ids=self._body_ids,
            env_ids=ids,
            term_name="randomize_body_mass_inertia",
        )


randomize_body_mass_inertia = RandomizeBodyMassInertia


class RandomizeRigidBodyCom(_ResetRandomizationTerm):
    """Community-compatible additive rigid-body CoM randomization.

    The ``com_range`` param is re-resolved from the live term params on every
    apply (not cached at construction), so step-staged curricula can widen the
    range by updating the event term config between resets.

    In startup mode the declared range is sampled once for every environment and
    the resulting full-width table is replayed; later curriculum changes do not
    retroactively alter the already-fixed per-env values.
    """

    uses_startup_rows = True

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        term_name = "randomize_rigid_body_com"
        _validate_event_term(
            cfg,
            term_name=term_name,
            mode=("reset", "startup"),
            allowed_params=frozenset(("com_range", "asset_cfg")),
            required_params=("com_range", "asset_cfg"),
        )
        asset_cfg = cfg.params["asset_cfg"]
        if not isinstance(asset_cfg, SceneEntityCfg):
            raise TypeError(
                f"EventManager term '{term_name}' asset_cfg must be SceneEntityCfg, "
                f"got {type(asset_cfg).__name__}"
            )
        # Fail-closed validation of the declared range at construction; the
        # live value is re-resolved per apply so curricula can stage it.
        _axis_ranges(
            cfg.params["com_range"],
            term_name=term_name,
            name="com_range",
            keys=_XYZ_KEYS,
        )
        self._entity = cast("Entity", env.scene[asset_cfg.name])
        self._body_ids, self._default_ipos = self._entity.bind_body_ipos_write(
            asset_cfg.body_ids,
            term_name=term_name,
        )
        self._startup = cfg.mode == "startup"
        self._startup_values: torch.Tensor | None = None

    def _apply_startup(
        self,
        env: ManagerBasedRlEnv,
        env_ids: torch.Tensor | None,
        com_range: dict[str, tuple[float, float]],
    ) -> None:
        """Sample every world once and commit the immutable startup table."""
        ids = resolve_env_ids(env, env_ids)
        if not torch.equal(ids, torch.arange(env.num_envs, dtype=torch.int64)):
            raise ValueError(
                "EventManager term 'randomize_rigid_body_com' startup mode requires all "
                f"environment rows; got {ids.tolist()} for {env.num_envs} environments"
            )
        if self._startup_values is None:
            ranges = _axis_ranges(
                com_range,
                term_name="randomize_rigid_body_com",
                name="com_range",
                keys=_XYZ_KEYS,
            )
            offsets = _require_torch_rng(env, "randomize_rigid_body_com").uniform(
                torch.as_tensor(ranges[:, 0], dtype=torch.float64),
                torch.as_tensor(ranges[:, 1], dtype=torch.float64),
                (env.num_envs, 3),
                dtype=torch.float64,
            )
            default_ipos = _selected_reset_defaults_tensor(
                self._default_ipos,
                ids,
                canonical_ndim=2,
                device=offsets.device,
            )
            values = default_ipos + offsets[:, None, :]
            if not bool(torch.isfinite(values).all()):
                raise ValueError(
                    "EventManager term 'randomize_rigid_body_com' produced NaN or Inf "
                    "startup values"
                )
            self._startup_values = values
        self._entity.record_startup_body_ipos_to_sim(
            self._startup_values.cpu(),
            body_ids=self._body_ids,
            env_ids=ids,
            term_name="randomize_rigid_body_com",
        )

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        env_ids: torch.Tensor | None,
        com_range: dict[str, tuple[float, float]],
        asset_cfg: SceneEntityCfg,
    ) -> None:
        del asset_cfg
        if self._startup:
            self._apply_startup(env, env_ids, com_range)
            return
        ranges = _axis_ranges(
            com_range,
            term_name="randomize_rigid_body_com",
            name="com_range",
            keys=_XYZ_KEYS,
        )
        ids = resolve_env_ids(env, env_ids)
        offsets = _require_torch_rng(env, "randomize_rigid_body_com").uniform(
            torch.as_tensor(ranges[:, 0], dtype=torch.float64),
            torch.as_tensor(ranges[:, 1], dtype=torch.float64),
            (ids.numel(), 3),
            dtype=torch.float64,
        )
        default_ipos = _selected_reset_defaults_tensor(
            self._default_ipos,
            ids,
            canonical_ndim=2,
            device=offsets.device,
        )
        values = (default_ipos + offsets[:, None, :]).cpu()
        self._entity.write_body_ipos_to_sim(
            values,
            body_ids=self._body_ids,
            env_ids=ids,
            term_name="randomize_rigid_body_com",
        )


randomize_rigid_body_com = RandomizeRigidBodyCom


class RandomizePhysicsSceneGravity(_ResetRandomizationTerm):
    """Community-compatible gravity randomization through reset transactions."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        term_name = "randomize_physics_scene_gravity"
        _validate_event_term(
            cfg,
            term_name=term_name,
            mode="reset",
            allowed_params=frozenset(("gravity_distribution_params", "operation", "distribution")),
            required_params=("gravity_distribution_params", "operation"),
        )
        self._operation = _event_choice(
            cfg.params["operation"],
            term_name=term_name,
            name="operation",
            choices=_OPERATIONS,
        )
        self._distribution = _event_choice(
            cfg.params.get("distribution", "uniform"),
            term_name=term_name,
            name="distribution",
            choices=_DISTRIBUTIONS,
        )
        self._distribution_params = _distribution_parameters(
            cfg.params["gravity_distribution_params"],
            term_name=term_name,
            name="gravity_distribution_params",
            width=3,
            distribution=self._distribution,
        )
        self._default_gravity = env.scene.bind_gravity_write_torch(
            term_name=term_name, device=getattr(env, "device", torch.device("cpu"))
        )

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        env_ids: torch.Tensor | None,
        gravity_distribution_params: tuple[list[float], list[float]],
        operation: Literal["add", "scale", "abs"],
        distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
    ) -> None:
        del gravity_distribution_params, operation, distribution
        ids = resolve_env_ids(env, env_ids)
        sampled = _sample_distribution(
            _require_torch_rng(env, "randomize_gravity"),
            self._distribution_params,
            (ids.numel(), 3),
            self._distribution,
        )
        if self._default_gravity.ndim == 1:
            default_gravity = (
                self._default_gravity.expand(ids.numel(), 3).clone()
                if self._default_gravity.shape[0] == 1
                else self._default_gravity
            ).to(device=sampled.device)
        else:
            default_gravity = _selected_reset_defaults_tensor(
                self._default_gravity,
                ids,
                canonical_ndim=1,
                device=sampled.device,
            )
        values = _apply_randomization_operation_tensor(
            default_gravity,
            sampled,
            self._operation,
        )
        env.scene.write_gravity_to_sim(
            values.cpu(),
            ids,
            term_name="randomize_physics_scene_gravity",
        )


randomize_physics_scene_gravity = RandomizePhysicsSceneGravity


class PushBySettingVelocity(ManagerTermBase):
    """Pinned community velocity kick dispatched through the interval plan.

    Linear (``x``/``y``/``z``) and angular (``roll``/``pitch``/``yaw``) ranges
    are world-frame deltas, matching the pinned mjlab semantics. Angular kicks
    require the backend's interval angular-velocity capability; backends
    without it fail closed at construction.

    ``__call__`` re-reads the live ``velocity_range`` on every apply so
    ``event_curriculum`` range stages take effect (the documented manager
    contract); construction still parses the initial ranges to decide which
    backend capabilities to bind. A curriculum that activates angular ranges
    on a term constructed without them fails closed at apply time.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        term_name = "push_by_setting_velocity"
        _validate_event_term(
            cfg,
            term_name=term_name,
            mode="interval",
            allowed_params=frozenset(("velocity_range", "asset_cfg")),
            required_params=("velocity_range",),
        )
        asset_cfg = cfg.params.get("asset_cfg", _DEFAULT_ASSET_CFG)
        if not isinstance(asset_cfg, SceneEntityCfg):
            raise TypeError(
                f"EventManager term '{term_name}' asset_cfg must be SceneEntityCfg, "
                f"got {type(asset_cfg).__name__}"
            )
        ranges = _axis_ranges(
            cfg.params["velocity_range"],
            term_name=term_name,
            name="velocity_range",
            keys=_SE3_KEYS,
        )
        self._angular_active = bool((ranges[3:] != 0.0).any())
        self._entity = cast("Entity", env.scene[asset_cfg.name])
        self._entity.bind_root_linear_velocity_delta(term_name=term_name)
        if self._angular_active:
            self._entity.bind_root_angular_velocity_delta(term_name=term_name)

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        env_ids: torch.Tensor | None,
        velocity_range: dict[str, tuple[float, float]],
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    ) -> None:
        del asset_cfg
        ids = resolve_env_ids(env, env_ids)
        ranges = _axis_ranges(
            velocity_range,
            term_name="push_by_setting_velocity",
            name="velocity_range",
            keys=_SE3_KEYS,
        )
        linear_ranges = ranges[:3]
        angular_ranges = ranges[3:]
        linear_delta = self._host_uniform(
            linear_ranges[:, 0],
            linear_ranges[:, 1],
            size=(ids.numel(), 3),
            env=env,
        )
        angular_delta = None
        if bool((angular_ranges != 0.0).any()):
            if not self._angular_active:
                raise NotImplementedError(
                    "EventManager term 'push_by_setting_velocity' velocity_range activated "
                    "angular axes after construction, but the backend angular-velocity "
                    "capability was never bound; declare non-zero angular ranges in the "
                    "initial params"
                )
            angular_delta = self._host_uniform(
                angular_ranges[:, 0],
                angular_ranges[:, 1],
                size=(ids.numel(), 3),
                env=env,
            )
        self._entity.apply_root_velocity_delta_to_sim(
            linear_delta,
            angular_delta,
            env_ids=ids,
            term_name="push_by_setting_velocity",
        )

    def _host_uniform(
        self,
        lower: torch.Tensor,
        upper: torch.Tensor,
        *,
        size: tuple[int, ...],
        env: ManagerBasedRlEnv,
    ) -> torch.Tensor:
        rng = _require_torch_rng(env, "push_by_setting_velocity")
        unit = torch.rand(size, generator=rng.generator, device=rng.device)
        scale = torch.as_tensor(upper - lower, dtype=unit.dtype, device=unit.device)
        offset = torch.as_tensor(lower, dtype=unit.dtype, device=unit.device)
        # The interval host plan consumes CPU tensors; sampling follows the
        # Manager generator device, so move the result to the host.
        return (unit * scale + offset).cpu()


push_by_setting_velocity = PushBySettingVelocity


def _time_range(
    value: Any,
    *,
    term_name: str,
    name: str,
) -> tuple[float, float]:
    try:
        if isinstance(value, (str, bytes)) or len(tuple(value)) != 2:
            raise ValueError
        lower, upper = float(value[0]), float(value[1])
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"EventManager term '{term_name}' parameter '{name}' must be a numeric (min, max) pair"
        ) from exc
    if not (math.isfinite(lower) and math.isfinite(upper)):
        raise ValueError(
            f"EventManager term '{term_name}' parameter '{name}' must contain only finite values"
        )
    if lower < 0.0 or lower > upper:
        raise ValueError(
            f"EventManager term '{term_name}' parameter '{name}' must satisfy "
            f"0 <= min <= max, got ({lower}, {upper})"
        )
    return lower, upper


class ApplyBodyImpulse(ManagerTermBase):
    """Transient random body impulses with a cooldown->trigger->sustain->expire lifecycle.

    NumPy adaptation of the pinned mjlab ``apply_body_impulse`` term. Each
    environment runs an independent timer: after a sampled cooldown, a
    uniformly sampled world-frame wrench is staged on the selected bodies and
    re-staged every step for a sampled duration, then expires back into
    cooldown. Use with ``mode="step"``.

    ``body_point_offset`` shifts the force application point in the body link
    frame; the resulting ``cross(offset_w, force)`` torque is added at trigger
    time and held for the impulse duration. Torque channels (explicit
    ``torque_range`` or an offset) require the backend's interval body-torque
    capability; force-only impulses only require interval body force. Backends
    without the required capability fail closed at construction.
    """

    _PARAMS = frozenset(
        (
            "force_range",
            "torque_range",
            "duration_s",
            "cooldown_s",
            "asset_cfg",
            "body_point_offset",
        )
    )

    @property
    def entity_name(self) -> str:
        return self._asset_cfg_name

    @property
    def tensor_body_names(self) -> tuple[str, ...]:
        return self._body_names if self._body_point_offset is not None else ()

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        term_name = "apply_body_impulse"
        _validate_event_term(
            cfg,
            term_name=term_name,
            mode="step",
            allowed_params=self._PARAMS,
            required_params=("force_range", "torque_range", "duration_s", "cooldown_s"),
        )
        self._force_range = _distribution_parameters(
            cfg.params["force_range"],
            term_name=term_name,
            name="force_range",
            distribution="uniform",
        )
        self._torque_range = _distribution_parameters(
            cfg.params["torque_range"],
            term_name=term_name,
            name="torque_range",
            distribution="uniform",
        )
        self._duration_s = _time_range(
            cfg.params["duration_s"], term_name=term_name, name="duration_s"
        )
        self._cooldown_s = _time_range(
            cfg.params["cooldown_s"], term_name=term_name, name="cooldown_s"
        )
        asset_cfg = cfg.params.get("asset_cfg", _DEFAULT_ASSET_CFG)
        if not isinstance(asset_cfg, SceneEntityCfg):
            raise TypeError(
                f"EventManager term '{term_name}' asset_cfg must be SceneEntityCfg, "
                f"got {type(asset_cfg).__name__}"
            )
        offset = cfg.params.get("body_point_offset")
        if offset is not None:
            try:
                if isinstance(offset, (str, bytes)) or len(tuple(offset)) != 3:
                    raise ValueError
                x, y, z = (float(component) for component in offset)
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    f"EventManager term '{term_name}' body_point_offset must be numeric"
                ) from exc
            if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
                raise ValueError(
                    f"EventManager term '{term_name}' body_point_offset must be a finite "
                    f"(x, y, z) triple, got {offset!r}"
                )
            self._body_point_offset: tuple[float, float, float] | None = (x, y, z)
        else:
            self._body_point_offset = None

        self._entity = cast("Entity", env.scene[asset_cfg.name])
        self._asset_cfg_name = asset_cfg.name
        self._uses_torque = (
            bool((self._torque_range != 0.0).any()) or self._body_point_offset is not None
        )
        self._local_body_ids, self._backend_body_ids = self._entity.bind_body_wrench(
            asset_cfg.body_ids,
            torque=self._uses_torque,
            term_name=term_name,
        )
        self._body_names = tuple(
            self._entity.body_names[int(index)] for index in self._local_body_ids
        )
        self._step_dt = float(env.step_dt)
        if not math.isfinite(self._step_dt) or self._step_dt <= 0.0:
            raise ValueError(
                f"EventManager term '{term_name}' requires a positive env step_dt, "
                f"got {self._step_dt}"
            )
        num_bodies = self._backend_body_ids.size
        self._device = _require_torch_rng(self._env, "apply_body_impulse").device
        self._time_remaining = torch.zeros(self.num_envs, dtype=torch.float64, device=self._device)
        self._active = torch.zeros(self.num_envs, dtype=torch.bool, device=self._device)
        self._active_forces = torch.zeros(
            (self.num_envs, num_bodies, 3), dtype=torch.float64, device=self._device
        )
        self._active_torques = torch.zeros(
            (self.num_envs, num_bodies, 3), dtype=torch.float64, device=self._device
        )
        # Pre-sample the initial cooldown so the first impulse is preceded by a
        # cooldown rather than firing immediately at t=0.
        self._interval_time_left = self._sample_cooldown(self.num_envs)

    def _sample_cooldown(self, count: int) -> torch.Tensor:
        return self._host_uniform(self._cooldown_s[0], self._cooldown_s[1], count)

    def _host_uniform(self, lower: float, upper: float, count: int) -> torch.Tensor:
        rng = _require_torch_rng(self._env, "apply_body_impulse")
        samples = torch.rand((count,), generator=rng.generator, device=rng.device)
        return (samples * float(upper - lower) + float(lower)).to(dtype=torch.float64)

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        env_ids: torch.Tensor | None,
        force_range: tuple[float, float],
        torque_range: tuple[float, float],
        duration_s: tuple[float, float],
        cooldown_s: tuple[float, float],
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
        body_point_offset: tuple[float, float, float] | None = None,
    ) -> None:
        del env_ids, force_range, torque_range, duration_s, cooldown_s, asset_cfg
        dt = self._step_dt

        # Decrement active impulse timers, then expire finished impulses.
        self._time_remaining[self._active] -= dt
        expired = self._active & (self._time_remaining <= 0.0)
        self._active &= ~expired
        self._active_forces[expired] = 0.0
        self._active_torques[expired] = 0.0
        self._time_remaining[expired] = 0.0
        if bool(expired.any()):
            expired_ids = torch.nonzero(expired, as_tuple=False).flatten()
            self._interval_time_left[expired_ids] = self._sample_cooldown(expired_ids.numel())

        # Decrement cooldown timers, then trigger eligible envs.
        self._interval_time_left -= dt
        eligible = (~self._active) & (self._interval_time_left <= 0.0)
        if bool(eligible.any()):
            trigger_ids = torch.nonzero(eligible, as_tuple=False).flatten()
            count = trigger_ids.numel()
            num_bodies = self._backend_body_ids.size
            forces = self._host_uniform(
                self._force_range[0], self._force_range[1], count * num_bodies * 3
            ).reshape(count, num_bodies, 3)
            torques = self._host_uniform(
                self._torque_range[0], self._torque_range[1], count * num_bodies * 3
            ).reshape(count, num_bodies, 3)
            if self._body_point_offset is not None:
                read_plan = getattr(self._env.scene, "_tensor_read_plan", None)
                if read_plan is None:
                    raise NotImplementedError(
                        "EventManager term 'apply_body_impulse' requires the packed "
                        "Torch body-state read plan for body_point_offset"
                    )
                quats = read_plan.body_tensor_view(self._asset_cfg_name, self._body_names).quat_w
                quat_rows = self._selected_body_quats(
                    quats.to(device=self._device, dtype=torch.float64), trigger_ids
                )
                offsets = (
                    torch.tensor(self._body_point_offset)
                    .to(dtype=quat_rows.dtype)
                    .expand(count * num_bodies, 3)
                )
                offset_w = _tensor_quat_apply(quat_rows, offsets).reshape(count, num_bodies, 3)
                torques = torques + torch.linalg.cross(offset_w, forces)
            self._active_forces[trigger_ids] = forces
            self._active_torques[trigger_ids] = torques
            self._time_remaining[trigger_ids] = self._host_uniform(
                self._duration_s[0], self._duration_s[1], count
            )
            self._active[trigger_ids] = True
            self._interval_time_left[trigger_ids] = self._sample_cooldown(count)

        # Re-stage the full-width wrench while any impulse is active: backends
        # with one-shot external-force channels (MuJoCo) need the sustain
        # re-stage, and absolute channels (Motrix) treat it as an idempotent
        # target update. Newly expired rows stage zeros, which actively clears
        # persistent channels. Idle envs skip the call entirely so the term
        # never clobbers another producer's staged forces.
        if bool(self._active.any() or expired.any()):
            self._entity.apply_body_wrench_to_sim(
                self._active_forces.detach().cpu(),
                self._active_torques.detach().cpu() if self._uses_torque else None,
                self._backend_body_ids,
                env_ids=None,
                term_name="apply_body_impulse",
            )

    def reset(self, reset_mask: torch.Tensor) -> None:
        was_active = self._active.any()
        forces = self._active_forces.clone()
        torques = self._active_torques.clone()
        keep_mask = ~reset_mask
        self._active.copy_(self._active & keep_mask)
        self._active_forces.copy_(self._active_forces * keep_mask[:, None, None])
        self._active_torques.copy_(self._active_torques * keep_mask[:, None, None])
        self._time_remaining.copy_(self._time_remaining * keep_mask)
        self._interval_time_left.copy_(
            torch.where(
                reset_mask,
                self._sample_cooldown(self.num_envs),
                self._interval_time_left,
            )
        )
        if bool(was_active):
            zeros = torch.zeros_like(self._active_forces)
            full_forces = torch.where(reset_mask[:, None, None], zeros, forces)
            full_torques = torch.where(reset_mask[:, None, None], zeros, torques)
            self._entity.apply_body_wrench_to_sim(
                full_forces,
                full_torques if self._uses_torque else None,
                self._backend_body_ids,
                env_ids=None,
                term_name="apply_body_impulse",
            )

    def _selected_body_quats(
        self,
        quats: torch.Tensor,
        trigger_ids: torch.Tensor,
    ) -> torch.Tensor:
        rows = quats.to(device=self._device)
        rows = rows.index_select(0, trigger_ids.to(device=self._device))
        columns = self._local_body_ids.to(device=self._device)
        return rows.index_select(1, columns).reshape(-1, 4)


apply_body_impulse = ApplyBodyImpulse


class RandomizeEncoderBias(ManagerTermBase):
    """Per-reset joint encoder calibration bias through the Entity data surface."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        term_name = "randomize_encoder_bias"
        _validate_event_term(
            cfg,
            term_name=term_name,
            mode="reset",
            allowed_params=frozenset(("bias_range", "asset_cfg")),
            required_params=("bias_range", "asset_cfg"),
        )
        asset_cfg = cfg.params["asset_cfg"]
        if not isinstance(asset_cfg, SceneEntityCfg):
            raise TypeError(
                f"EventManager term '{term_name}' asset_cfg must be SceneEntityCfg, "
                f"got {type(asset_cfg).__name__}"
            )
        try:
            raw_bias_range = cfg.params["bias_range"]
            if isinstance(raw_bias_range, (str, bytes)) or len(tuple(raw_bias_range)) != 2:
                raise ValueError
            bias_lower, bias_upper = float(raw_bias_range[0]), float(raw_bias_range[1])
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"EventManager term '{term_name}' bias_range must be a numeric (min, max) pair"
            ) from exc
        if not (math.isfinite(bias_lower) and math.isfinite(bias_upper)):
            raise ValueError(f"EventManager term '{term_name}' bias_range must be a finite pair")
        if bias_lower > bias_upper:
            raise ValueError(f"EventManager term '{term_name}' bias_range minimum exceeds maximum")
        self._range = (bias_lower, bias_upper)
        self._entity = cast("Entity", env.scene[asset_cfg.name])
        all_joint_ids = tuple(range(self._entity.num_joints))
        if isinstance(asset_cfg.joint_ids, slice):
            selected_ids: Sequence[int] = range(len(all_joint_ids))[asset_cfg.joint_ids]
        else:
            selected_ids = asset_cfg.joint_ids
        joint_ids = tuple(all_joint_ids[index] for index in selected_ids)
        if not joint_ids:
            raise ValueError(
                f"EventManager term '{term_name}' asset_cfg must select at least one joint"
            )
        self._joint_ids = joint_ids

    def reset(
        self,
        env: ManagerBasedRlEnv,
        reset_mask: torch.Tensor,
        bias_range: tuple[float, float],
        asset_cfg: SceneEntityCfg,
    ) -> None:
        del bias_range, asset_cfg
        control = getattr(self._entity.data, "control_buffer", None)
        if isinstance(control, torch.Tensor):
            columns = torch.tensor(self._joint_ids, dtype=torch.int64, device=control.device)
            rng_owner = getattr(env, "torch_rng", None)
            if rng_owner is None:
                raise NotImplementedError(
                    "Tensor encoder-bias reset requires the Manager Torch RNG"
                )
            lower, upper = self._range
            sampled = rng_owner.uniform(
                lower,
                upper,
                (self.num_envs, columns.numel()),
                dtype=torch.float32,
            )
            self._entity.data.update_encoder_bias_tensor_mask(reset_mask, columns, sampled)
            return
        raise NotImplementedError(
            "EventManager term 'randomize_encoder_bias' requires a Torch control plane"
        )


randomize_encoder_bias = RandomizeEncoderBias


def reset_scene_to_default(env: ManagerBasedRlEnv, reset_mask: torch.Tensor) -> None:
    """Reset all materialized scene entities to backend default qpos/qvel."""
    if not env.scene.entities:
        return
    env.scene.reset_to_default(reset_mask, term_name="reset_scene_to_default")


setattr(reset_scene_to_default, "reset", reset_scene_to_default)


def reset_root_state_uniform(
    env: ManagerBasedRlEnv,
    reset_mask: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]] | None = None,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
    """Reset a floating root from defaults plus uniformly sampled SE(3) offsets.

    This is the NumPy adaptation of the pinned mjlab event. UniLab currently has
    no public mocap-pose write contract, so fixed-base/mocap requests fail through
    the entity's cached floating-root capability instead of falling back.
    """
    asset = cast("Entity", env.scene[asset_cfg.name])
    if reset_mask.shape != (env.num_envs,) or reset_mask.dtype != torch.bool:
        raise ValueError("reset_root_state_uniform requires a full-width bool reset mask")
    try:
        default_root_state = asset.data.default_root_state_torch(torch.device(env.device))
        root_states = default_root_state.clone()
    except NotImplementedError as exc:
        raise NotImplementedError(
            "EventManager term 'reset_root_state_uniform' requires a floating-root "
            f"state for entity '{asset_cfg.name}'; fixed-base/mocap root reset is "
            f"unsupported without a formal backend contract: {exc}"
        ) from exc

    pose_samples = _sample_se3_range(
        pose_range, (env.num_envs, 6), _require_torch_rng(env, "reset_root_state_uniform")
    )
    root_states[:, 0:3] = (
        root_states[:, 0:3]
        + pose_samples[:, 0:3].to(dtype=root_states.dtype)
        + env.scene.env_origins_torch(torch.device(env.device))
    )
    orientation_delta = _tensor_quat_from_euler_xyz(
        pose_samples[:, 3], pose_samples[:, 4], pose_samples[:, 5]
    )
    root_states[:, 3:7] = _tensor_quat_mul(
        root_states[:, 3:7], orientation_delta.to(root_states.dtype)
    )

    velocity_samples = _sample_se3_range(
        velocity_range, (env.num_envs, 6), _require_torch_rng(env, "reset_root_state_uniform")
    )
    root_states[:, 7:13] = root_states[:, 7:13] + velocity_samples.to(dtype=root_states.dtype)
    asset.write_root_state_tensor_to_sim(root_states, reset_mask)


setattr(reset_root_state_uniform, "reset", reset_root_state_uniform)


def _tensor_se3_bounds(range_dict: dict[str, tuple[float, float]] | None) -> torch.Tensor:
    try:
        values = [
            [
                float((range_dict or {}).get(key, (0.0, 0.0))[0]),
                float((range_dict or {}).get(key, (0.0, 0.0))[1]),
            ]
            for key in _SE3_KEYS
        ]
        bounds = torch.tensor(values, dtype=torch.float32)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError(
            "tensor reset SE(3) ranges must map each key to a numeric (min, max) pair"
        ) from exc
    if bounds.shape != (len(_SE3_KEYS), 2) or not bool(torch.isfinite(bounds).all()):
        raise ValueError("tensor reset SE(3) ranges must be finite (min, max) pairs")
    if bool((bounds[:, 0] > bounds[:, 1]).any()):
        raise ValueError("tensor reset SE(3) range minimum exceeds maximum")
    return bounds


def _tensor_quat_from_euler_xyz(
    roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor
) -> torch.Tensor:
    """Convert XYZ Euler angles to device-resident wxyz quaternions."""
    cos_roll, sin_roll = torch.cos(0.5 * roll), torch.sin(0.5 * roll)
    cos_pitch, sin_pitch = torch.cos(0.5 * pitch), torch.sin(0.5 * pitch)
    cos_yaw, sin_yaw = torch.cos(0.5 * yaw), torch.sin(0.5 * yaw)
    return torch.stack(
        (
            cos_roll * cos_pitch * cos_yaw + sin_roll * sin_pitch * sin_yaw,
            sin_roll * cos_pitch * cos_yaw - cos_roll * sin_pitch * sin_yaw,
            cos_roll * sin_pitch * cos_yaw + sin_roll * cos_pitch * sin_yaw,
            cos_roll * cos_pitch * sin_yaw - sin_roll * sin_pitch * cos_yaw,
        ),
        dim=-1,
    )


def _tensor_quat_apply(quat_wxyz: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Apply broadcast-compatible wxyz quaternions to device-resident vectors."""
    quat_xyz = quat_wxyz[..., 1:4]
    if vector.ndim < quat_xyz.ndim:
        vector = vector.expand_as(quat_xyz)
    cross = torch.linalg.cross(quat_xyz, vector, dim=-1)
    second_cross = torch.linalg.cross(quat_xyz, cross, dim=-1)
    return vector + 2.0 * quat_wxyz[..., 0:1] * cross + 2.0 * second_cross


def _tensor_quat_mul(left_wxyz: torch.Tensor, right_wxyz: torch.Tensor) -> torch.Tensor:
    """Multiply broadcast-compatible device-resident wxyz quaternions."""
    left_w, left_xyz = left_wxyz[..., 0:1], left_wxyz[..., 1:4]
    right_w, right_xyz = right_wxyz[..., 0:1], right_wxyz[..., 1:4]
    cross = torch.linalg.cross(left_xyz, right_xyz, dim=-1)
    return torch.cat(
        (
            left_w * right_w - (left_xyz * right_xyz).sum(dim=-1, keepdim=True),
            left_w * right_xyz + right_w * left_xyz + cross,
        ),
        dim=-1,
    )


def _tensor_root_reset_kernel(
    default_rows: torch.Tensor,
    origins_rows: torch.Tensor | None,
    pose_delta: torch.Tensor,
    velocity_delta: torch.Tensor,
    output: torch.Tensor,
) -> None:
    """Build one sampled floating-root state in a single compiled Torch graph."""
    output.copy_(default_rows)
    if origins_rows is not None:
        output[:, 0:3] += origins_rows[:, 0:3]
    output[:, 0:3] += pose_delta[:, 0:3]
    orientation_delta = _tensor_quat_from_euler_xyz(
        pose_delta[:, 3], pose_delta[:, 4], pose_delta[:, 5]
    )
    output[:, 3:7] = _tensor_quat_mul(output[:, 3:7], orientation_delta)
    output[:, 7:13] += velocity_delta


def reset_root_state_uniform_tensor(
    env: ManagerBasedRlEnv,
    reset_mask: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]] | None = None,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
    """Full-width root reset."""
    if env.torch_rng is None:
        raise NotImplementedError(
            "reset_root_state_uniform_tensor requires env.torch_rng on a CUDA "
            "Manager tensor runtime"
        )
    if reset_mask.device != torch.device(env.device):
        raise ValueError("reset_root_state_uniform_tensor mask must live on env.device")
    asset = cast("Entity", env.scene[asset_cfg.name])
    device = torch.device(env.device)
    asset.data.default_root_state_torch(device)
    bounds_key = (
        (asset_cfg.name, tuple(sorted(pose_range.items())), tuple(sorted(velocity_range.items())))
        if velocity_range is not None
        else (asset_cfg.name, tuple(sorted(pose_range.items())), None)
    )
    pose_bounds_tensor = getattr(env, "_tensor_reset_pose_bounds", None)
    velocity_bounds_tensor = getattr(env, "_tensor_reset_velocity_bounds", None)
    if pose_bounds_tensor is None or getattr(env, "_tensor_reset_bounds_key", None) != bounds_key:
        new_pose_bounds = _tensor_se3_bounds(pose_range).to(device=device)
        new_velocity_bounds = _tensor_se3_bounds(velocity_range).to(device=device)
        env._tensor_reset_pose_bounds = new_pose_bounds
        env._tensor_reset_velocity_bounds = new_velocity_bounds
        env._tensor_reset_bounds_key = bounds_key
        pose_bounds_tensor = new_pose_bounds
        velocity_bounds_tensor = new_velocity_bounds
    reset_rows = torch.nonzero(reset_mask, as_tuple=False).flatten()
    reset_unit = torch.rand(
        (reset_rows.shape[0], 2 * len(_SE3_KEYS)),
        dtype=torch.float32,
        device=device,
        generator=env.torch_rng.generator,
    )
    bounds = torch.cat((pose_bounds_tensor, velocity_bounds_tensor), dim=0)
    deltas = torch.addcmul(
        bounds[None, :, 0],
        reset_unit,
        bounds[None, :, 1] - bounds[None, :, 0],
    )
    pose_delta, velocity_delta = torch.split(deltas, len(_SE3_KEYS), dim=1)
    default_tensor = asset.data.default_root_state_torch(device)
    origins = env.scene.env_origins_torch(device)
    default_rows = default_tensor.index_select(0, reset_rows)
    origin_rows = origins.index_select(0, reset_rows) if origins is not None else None
    reset_rows_state = torch.empty_like(default_rows)
    _tensor_root_reset_kernel(
        default_rows,
        origin_rows,
        pose_delta,
        velocity_delta,
        reset_rows_state,
    )
    root_state = default_tensor.clone()
    root_state.index_copy_(0, reset_rows, reset_rows_state)
    asset.write_root_state_tensor_to_sim(root_state, reset_mask)


setattr(reset_root_state_uniform_tensor, "reset", reset_root_state_uniform_tensor)


def reset_scene_to_default_tensor(
    env: ManagerBasedRlEnv,
    reset_mask: torch.Tensor,
) -> None:
    """Mask-preserving no-op hook for the canonical MJWarp reset scene term."""
    del env, reset_mask


setattr(reset_scene_to_default_tensor, "reset", reset_scene_to_default_tensor)


__all__ = [
    "apply_body_impulse",
    "dof_armature",
    "geom_friction",
    "joint_armature",
    "pd_gains",
    "push_by_setting_velocity",
    "randomize_body_mass_inertia",
    "randomize_encoder_bias",
    "randomize_physics_scene_gravity",
    "randomize_rigid_body_com",
    "randomize_rigid_body_mass",
    "reset_root_state_uniform",
    "reset_root_state_uniform_tensor",
    "reset_scene_to_default",
    "reset_scene_to_default_tensor",
    "resolve_env_ids",
]
