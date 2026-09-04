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
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np

from unilab.managers.event_manager import EventTermCfg
from unilab.managers.manager_base import ManagerTermBase
from unilab.managers.scene_entity_config import SceneEntityCfg
from unilab.utils.rotation import np_quat_apply_batched, np_quat_from_euler_xyz, np_quat_mul

if TYPE_CHECKING:
    from unilab.base.entity import Entity
    from unilab.envs.manager_based_rl_env import ManagerBasedRlEnv as ManagerBasedRlEnvImpl
    from unilab.managers._types import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")
_SE3_KEYS = ("x", "y", "z", "roll", "pitch", "yaw")
_XYZ_KEYS = ("x", "y", "z")
_PD_GAIN_PARAM_NAMES = frozenset(("kp_range", "kd_range", "asset_cfg", "distribution", "operation"))
_DISTRIBUTIONS = ("uniform", "log_uniform", "gaussian")
_OPERATIONS = ("add", "scale", "abs")
_REPO_ROOT = Path(__file__).resolve().parents[4]


def _gain_range(
    value: Any,
    *,
    name: str,
    distribution: Literal["uniform", "log_uniform"],
) -> tuple[float, float]:
    try:
        bounds = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"pd_gains {name} must be a numeric (min, max) pair") from exc
    if bounds.shape != (2,):
        raise ValueError(f"pd_gains {name} must have shape (2,), got {bounds.shape}")
    if not np.isfinite(bounds).all():
        raise ValueError(f"pd_gains {name} must contain only finite values")
    lower, upper = float(bounds[0]), float(bounds[1])
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
    rng: np.random.Generator,
    bounds: tuple[float, float],
    shape: tuple[int, int],
    distribution: Literal["uniform", "log_uniform"],
) -> np.ndarray:
    if distribution == "uniform":
        return rng.uniform(bounds[0], bounds[1], size=shape)
    return np.exp(rng.uniform(np.log(bounds[0]), np.log(bounds[1]), size=shape))


def _sample_se3_range(
    range_dict: dict[str, tuple[float, float]] | None,
    shape: tuple[int, ...],
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample uniform ``[x, y, z, roll, pitch, yaw]`` offsets with NumPy."""
    if not shape or shape[-1] != len(_SE3_KEYS):
        raise ValueError(
            f"reset_root_state_uniform SE(3) sample shape must end in 6; received {shape}"
        )
    try:
        ranges = np.asarray(
            [(range_dict or {}).get(key, (0.0, 0.0)) for key in _SE3_KEYS],
            dtype=np.float64,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "reset_root_state_uniform ranges must map each SE(3) key to a numeric (min, max) pair"
        ) from exc
    if ranges.shape != (len(_SE3_KEYS), 2):
        raise ValueError(
            "reset_root_state_uniform ranges must map each SE(3) key to a "
            f"(min, max) pair; received shape {ranges.shape}"
        )
    if not np.isfinite(ranges).all():
        raise ValueError("reset_root_state_uniform ranges must contain only finite values")
    invalid = ranges[:, 0] > ranges[:, 1]
    if np.any(invalid):
        keys = [_SE3_KEYS[index] for index in np.flatnonzero(invalid)]
        raise ValueError(f"reset_root_state_uniform range minimum exceeds maximum for keys {keys}")
    return rng.uniform(ranges[:, 0], ranges[:, 1], size=shape)


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
) -> np.ndarray:
    try:
        params = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"EventManager term '{term_name}' parameter '{name}' must contain numeric values"
        ) from exc
    expected = (2,) if width is None else (2, width)
    if params.shape != expected:
        raise ValueError(
            f"EventManager term '{term_name}' parameter '{name}' has shape {params.shape}; "
            f"expected {expected}"
        )
    if not np.isfinite(params).all():
        raise ValueError(
            f"EventManager term '{term_name}' parameter '{name}' must contain only finite values"
        )
    if distribution == "gaussian":
        if np.any(params[1] < 0.0):
            raise ValueError(
                f"EventManager term '{term_name}' parameter '{name}' standard deviation "
                "must be non-negative"
            )
    else:
        if np.any(params[0] > params[1]):
            raise ValueError(
                f"EventManager term '{term_name}' parameter '{name}' lower bound exceeds upper bound"
            )
        if distribution == "log_uniform" and np.any(params <= 0.0):
            raise ValueError(
                f"EventManager term '{term_name}' parameter '{name}' must be positive "
                "for log_uniform sampling"
            )
    result = np.array(params, copy=True)
    result.setflags(write=False)
    return result


def _sample_distribution(
    rng: np.random.Generator,
    params: np.ndarray,
    shape: tuple[int, ...],
    distribution: str,
) -> np.ndarray:
    if distribution == "gaussian":
        return rng.normal(params[0], params[1], size=shape)
    if distribution == "log_uniform":
        return np.exp(rng.uniform(np.log(params[0]), np.log(params[1]), size=shape))
    return rng.uniform(params[0], params[1], size=shape)


def _apply_randomization_operation(
    default: np.ndarray,
    samples: np.ndarray,
    operation: str,
) -> np.ndarray:
    if operation == "add":
        return default + samples
    if operation == "scale":
        return default * samples
    return samples


def _axis_ranges(
    value: Any,
    *,
    term_name: str,
    name: str,
    keys: tuple[str, ...],
) -> np.ndarray:
    if not isinstance(value, dict):
        raise TypeError(f"EventManager term '{term_name}' parameter '{name}' must be a dict")
    unknown = sorted(set(value) - set(keys))
    if unknown:
        raise ValueError(
            f"EventManager term '{term_name}' parameter '{name}' has unknown axes {unknown}"
        )
    parameters = np.asarray([value.get(key, (0.0, 0.0)) for key in keys], dtype=object).T
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
    mode: str,
    allowed_params: frozenset[str],
    required_params: tuple[str, ...],
) -> None:
    if cfg.mode != mode:
        raise NotImplementedError(
            f"EventManager term '{term_name}' only supports mode='{mode}' on the UniLab runtime"
        )
    unknown = sorted(set(cfg.params) - allowed_params)
    if unknown:
        raise ValueError(f"EventManager term '{term_name}' has unknown parameters {unknown}")
    missing = [name for name in required_params if name not in cfg.params]
    if missing:
        raise ValueError(f"EventManager term '{term_name}' is missing parameters {missing}")


def resolve_env_ids(env: ManagerBasedRlEnv, env_ids: np.ndarray | None) -> np.ndarray:
    """Return concrete NumPy environment IDs, preserving community sentinel semantics."""
    if env_ids is None:
        return np.arange(env.num_envs, dtype=np.int32)
    return env_ids


class _ModelFieldRandomizer(ManagerTermBase):
    """Cold-path-bound NumPy adapter for pinned mjlab model-field DR terms."""

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

    def _bind(
        self,
        entity: Entity,
        asset_cfg: SceneEntityCfg,
    ) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
        raise NotImplementedError

    def _write(
        self,
        values: np.ndarray,
        env_ids: np.ndarray,
    ) -> None:
        raise NotImplementedError

    def _select_string_ranges(
        self,
        local_ids: np.ndarray,
        defaults: np.ndarray,
        names: tuple[str, ...],
        ranges: Any,
    ) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], Any]:
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
        selected = np.asarray(
            [index for index, value in enumerate(assigned) if value is not None],
            dtype=np.intp,
        )
        return (
            local_ids[selected],
            defaults[selected],
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
            if any(
                isinstance(axis, bool) or not isinstance(axis, (int, np.integer)) for axis in value
            ):
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

    def _resolve_ranges(self, value: Any, names: tuple[str, ...]) -> np.ndarray:
        per_entity: list[Any]
        if (
            isinstance(value, list)
            and len(value) == len(names)
            and any(isinstance(item, (tuple, list, np.ndarray)) for item in value)
        ):
            per_entity = list(value)
            value = None
        else:
            per_entity = []

        result = np.empty((len(names), self._field_width, 2), dtype=np.float64)
        result[:] = np.nan
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
        result.setflags(write=False)
        return result

    def _sample_axis(self, env: ManagerBasedRlEnv, axis: int, count: int) -> np.ndarray:
        parameters = self._ranges[:, axis]
        if self._shared_random:
            samples = np.empty((count, len(parameters)), dtype=np.float64)
            groups: dict[tuple[float, float], list[int]] = {}
            for index, pair in enumerate(parameters):
                groups.setdefault((float(pair[0]), float(pair[1])), []).append(index)
            for (first, second), indices in groups.items():
                if self._distribution == "gaussian":
                    shared = env.rng.normal(first, second, size=(count, 1))
                elif self._distribution == "log_uniform":
                    shared = np.exp(env.rng.uniform(np.log(first), np.log(second), size=(count, 1)))
                else:
                    shared = env.rng.uniform(first, second, size=(count, 1))
                samples[:, indices] = shared
            return samples
        if self._distribution == "gaussian":
            return env.rng.normal(parameters[:, 0], parameters[:, 1], size=(count, len(parameters)))
        if self._distribution == "log_uniform":
            return np.exp(
                env.rng.uniform(
                    np.log(parameters[:, 0]),
                    np.log(parameters[:, 1]),
                    size=(count, len(parameters)),
                )
            )
        return env.rng.uniform(
            parameters[:, 0],
            parameters[:, 1],
            size=(count, len(parameters)),
        )

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        env_ids: np.ndarray | None,
        ranges: Any,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
        distribution: str = "uniform",
        operation: str = "abs",
        axes: list[int] | None = None,
        shared_random: bool = False,
    ) -> None:
        del ranges, asset_cfg, distribution, operation, axes, shared_random
        ids = resolve_env_ids(env, env_ids)
        defaults = self._defaults
        scalar = defaults.ndim == 1
        default_values = defaults[:, None] if scalar else defaults
        values = np.broadcast_to(
            default_values,
            (len(ids), *default_values.shape),
        ).copy()
        for axis in self._axes:
            samples = self._sample_axis(env, axis, len(ids))
            values[:, :, axis] = _apply_randomization_operation(
                default_values[None, :, axis],
                samples,
                self._operation,
            )
        if np.any(values < 0.0) or not np.isfinite(values).all():
            raise ValueError(
                f"EventManager term '{self._term_name}' produced negative, NaN, or Inf values"
            )
        self._write(values[:, :, 0] if scalar else values, ids)


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
    ) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
        local_ids, defaults = entity.bind_geom_friction_write(
            asset_cfg.geom_ids,
            term_name=self._term_name,
        )
        names = tuple(entity.geom_names[int(index)] for index in local_ids)
        return local_ids, defaults, names

    def _write(self, values: np.ndarray, env_ids: np.ndarray) -> None:
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
    ) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
        local_ids, defaults = entity.bind_joint_armature_write(
            asset_cfg.joint_ids,
            term_name=self._term_name,
        )
        names = tuple(entity.joint_names[int(index)] for index in local_ids)
        return local_ids, defaults, names

    def _write(self, values: np.ndarray, env_ids: np.ndarray) -> None:
        self._entity.write_joint_armature_to_sim(
            values,
            self._local_ids,
            env_ids,
            term_name=self._term_name,
        )


geom_friction = GeomFriction
joint_armature = JointArmature
dof_armature = joint_armature


class PdGains(ManagerTermBase):
    """Pinned-mjlab-compatible PD gain randomization on UniLab reset payloads."""

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

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        env_ids: np.ndarray | None,
        kp_range: tuple[float, float],
        kd_range: tuple[float, float],
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
        distribution: Literal["uniform", "log_uniform"] = "uniform",
        operation: Literal["scale", "abs"] = "scale",
    ) -> None:
        del kp_range, kd_range, asset_cfg, distribution, operation
        ids = resolve_env_ids(env, env_ids)
        shape = (len(ids), len(self._actuator_ids))
        kp = _sample_gain_range(env.rng, self._kp_range, shape, self._distribution)
        kd = _sample_gain_range(env.rng, self._kd_range, shape, self._distribution)
        if self._operation == "scale":
            kp *= self._default_kp[None, :]
            kd *= self._default_kd[None, :]
        self._entity.write_actuator_gains_to_sim(
            kp,
            kd,
            actuator_ids=self._actuator_ids,
            env_ids=ids,
            term_name="pd_gains",
        )


pd_gains = PdGains


class RandomizeRigidBodyMass(ManagerTermBase):
    """Community-compatible body-mass randomization via the reset payload."""

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

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        term_name = "randomize_rigid_body_mass"
        _validate_event_term(
            cfg,
            term_name=term_name,
            mode="reset",
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
        if not np.isfinite(self._min_mass) or self._min_mass < 1e-6:
            raise ValueError(
                f"EventManager term '{term_name}' min_mass must be finite and at least 1e-6"
            )
        self._entity = cast("Entity", env.scene[asset_cfg.name])
        self._body_ids, self._default_mass = self._entity.bind_body_mass_write(
            asset_cfg.body_ids,
            term_name=term_name,
        )

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        env_ids: np.ndarray | None,
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
        ids = resolve_env_ids(env, env_ids)
        samples = _sample_distribution(
            env.rng,
            self._distribution_params,
            (ids.size, self._body_ids.size),
            self._distribution,
        )
        values = _apply_randomization_operation(
            self._default_mass[None, :],
            samples,
            self._operation,
        )
        np.maximum(values, self._min_mass, out=values)
        self._entity.write_body_mass_to_sim(
            values,
            body_ids=self._body_ids,
            env_ids=ids,
            term_name="randomize_rigid_body_mass",
        )


randomize_rigid_body_mass = RandomizeRigidBodyMass


def _scene_inertial_defaults(
    env: ManagerBasedRlEnv,
    *,
    term_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Compile the configured MJCF scene on the cold path for inertial defaults.

    Returns the full ``(nbody,)`` body-mass and ``(nbody, 3)`` principal-inertia
    tables in model body order. ``SimBackend`` exposes no body-inertia getter,
    so the defaults come from the same scene file the MuJoCo-family backends
    compile; the reset transaction cross-validates the mass table against the
    backend's authoritative values before trusting the inertia rows.
    """
    try:
        import mujoco
    except ImportError as exc:
        raise NotImplementedError(
            f"EventManager term '{term_name}' requires the mujoco package to compile "
            "the scene model for inertial defaults"
        ) from exc
    scene = cast("ManagerBasedRlEnvImpl", env).cfg.scene
    if scene is None:
        raise ValueError(f"EventManager term '{term_name}' requires a configured scene model file")
    model_file = str(scene.model_file)
    candidates = [Path(model_file)]
    if not Path(model_file).is_absolute():
        candidates.append(_REPO_ROOT / model_file)
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        raise ValueError(
            f"EventManager term '{term_name}' cannot locate scene model file "
            f"{model_file!r} (tried {', '.join(str(candidate) for candidate in candidates)})"
        )
    model = mujoco.MjModel.from_xml_path(str(path))
    mass = np.asarray(model.body_mass, dtype=np.float64)
    inertia = np.asarray(model.body_inertia, dtype=np.float64)
    return mass, inertia


class RandomizeBodyMassInertia(ManagerTermBase):
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
    re-derive from immutable compile-time defaults, so reapplication is
    idempotent and non-accumulating.
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
        default_mass, default_inertia = _scene_inertial_defaults(env, term_name=term_name)
        self._body_ids, self._default_mass = self._entity.bind_body_mass_write(
            asset_cfg.body_ids,
            term_name=term_name,
        )
        inertia_ids, self._default_inertia = self._entity.bind_body_inertia_write(
            asset_cfg.body_ids,
            default=default_inertia,
            default_mass=default_mass,
            term_name=term_name,
        )
        if not np.array_equal(self._body_ids, inertia_ids):
            raise RuntimeError(
                f"EventManager term '{term_name}' mass/inertia body bindings diverged: "
                f"{self._body_ids.tolist()} != {inertia_ids.tolist()}"
            )
        self._scales: np.ndarray | None = None

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        env_ids: np.ndarray | None,
        asset_cfg: SceneEntityCfg,
        scale_range: tuple[float, float],
    ) -> None:
        del asset_cfg, scale_range
        ids = resolve_env_ids(env, env_ids)
        if self._scales is None:
            alpha = env.rng.uniform(
                math.log(self._scale_lo) / 2.0,
                math.log(self._scale_hi) / 2.0,
                size=(env.num_envs, self._body_ids.size),
            )
            self._scales = np.exp(2.0 * alpha)
        scales = self._scales[ids]
        self._entity.write_body_mass_to_sim(
            self._default_mass[None, :] * scales,
            body_ids=self._body_ids,
            env_ids=ids,
            term_name="randomize_body_mass_inertia",
        )
        self._entity.write_body_inertia_to_sim(
            self._default_inertia[None, :, :] * scales[:, :, None],
            body_ids=self._body_ids,
            env_ids=ids,
            term_name="randomize_body_mass_inertia",
        )


randomize_body_mass_inertia = RandomizeBodyMassInertia


class RandomizeRigidBodyCom(ManagerTermBase):
    """Community-compatible additive rigid-body CoM randomization.

    The ``com_range`` param is re-resolved from the live term params on every
    apply (not cached at construction), so step-staged curricula can widen the
    range by updating the event term config between resets.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        term_name = "randomize_rigid_body_com"
        _validate_event_term(
            cfg,
            term_name=term_name,
            mode="reset",
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

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        env_ids: np.ndarray | None,
        com_range: dict[str, tuple[float, float]],
        asset_cfg: SceneEntityCfg,
    ) -> None:
        del asset_cfg
        ranges = _axis_ranges(
            com_range,
            term_name="randomize_rigid_body_com",
            name="com_range",
            keys=_XYZ_KEYS,
        )
        ids = resolve_env_ids(env, env_ids)
        offsets = env.rng.uniform(
            ranges[:, 0],
            ranges[:, 1],
            size=(ids.size, 3),
        )
        values = self._default_ipos[None, :, :] + offsets[:, None, :]
        self._entity.write_body_ipos_to_sim(
            values,
            body_ids=self._body_ids,
            env_ids=ids,
            term_name="randomize_rigid_body_com",
        )


randomize_rigid_body_com = RandomizeRigidBodyCom


class RandomizePhysicsSceneGravity(ManagerTermBase):
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
        self._default_gravity = env.scene.bind_gravity_write(term_name=term_name)

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        env_ids: np.ndarray | None,
        gravity_distribution_params: tuple[list[float], list[float]],
        operation: Literal["add", "scale", "abs"],
        distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
    ) -> None:
        del gravity_distribution_params, operation, distribution
        ids = resolve_env_ids(env, env_ids)
        samples = _sample_distribution(
            env.rng,
            self._distribution_params,
            (ids.size, 3),
            self._distribution,
        )
        values = _apply_randomization_operation(
            self._default_gravity[None, :],
            samples,
            self._operation,
        )
        env.scene.write_gravity_to_sim(
            values,
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
        self._angular_active = bool(np.any(ranges[3:] != 0.0))
        self._entity = cast("Entity", env.scene[asset_cfg.name])
        self._entity.bind_root_linear_velocity_delta(term_name=term_name)
        if self._angular_active:
            self._entity.bind_root_angular_velocity_delta(term_name=term_name)

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        env_ids: np.ndarray | None,
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
        linear_delta = env.rng.uniform(
            linear_ranges[:, 0],
            linear_ranges[:, 1],
            size=(ids.size, 3),
        )
        angular_delta = None
        if np.any(angular_ranges != 0.0):
            if not self._angular_active:
                raise NotImplementedError(
                    "EventManager term 'push_by_setting_velocity' velocity_range activated "
                    "angular axes after construction, but the backend angular-velocity "
                    "capability was never bound; declare non-zero angular ranges in the "
                    "initial params"
                )
            angular_delta = env.rng.uniform(
                angular_ranges[:, 0],
                angular_ranges[:, 1],
                size=(ids.size, 3),
            )
        self._entity.apply_root_velocity_delta_to_sim(
            linear_delta,
            angular_delta,
            env_ids=ids,
            term_name="push_by_setting_velocity",
        )


push_by_setting_velocity = PushBySettingVelocity


def _time_range(
    value: Any,
    *,
    term_name: str,
    name: str,
) -> tuple[float, float]:
    try:
        bounds = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"EventManager term '{term_name}' parameter '{name}' must be a numeric (min, max) pair"
        ) from exc
    if bounds.shape != (2,):
        raise ValueError(
            f"EventManager term '{term_name}' parameter '{name}' must have shape (2,), "
            f"got {bounds.shape}"
        )
    if not np.isfinite(bounds).all():
        raise ValueError(
            f"EventManager term '{term_name}' parameter '{name}' must contain only finite values"
        )
    lower, upper = float(bounds[0]), float(bounds[1])
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
            offset_array = np.asarray(offset, dtype=np.float64)
            if offset_array.shape != (3,) or not np.isfinite(offset_array).all():
                raise ValueError(
                    f"EventManager term '{term_name}' body_point_offset must be a finite "
                    f"(x, y, z) triple, got {offset!r}"
                )
            self._body_point_offset: np.ndarray | None = np.array(offset_array)
        else:
            self._body_point_offset = None

        self._entity = cast("Entity", env.scene[asset_cfg.name])
        self._uses_torque = (
            bool(np.any(self._torque_range != 0.0)) or self._body_point_offset is not None
        )
        self._local_body_ids, self._backend_body_ids = self._entity.bind_body_wrench(
            asset_cfg.body_ids,
            torque=self._uses_torque,
            term_name=term_name,
        )
        self._step_dt = float(env.step_dt)
        if not np.isfinite(self._step_dt) or self._step_dt <= 0.0:
            raise ValueError(
                f"EventManager term '{term_name}' requires a positive env step_dt, "
                f"got {self._step_dt}"
            )
        num_bodies = self._backend_body_ids.size
        self._time_remaining = np.zeros(self.num_envs, dtype=np.float64)
        self._active = np.zeros(self.num_envs, dtype=np.bool_)
        self._active_forces = np.zeros((self.num_envs, num_bodies, 3), dtype=np.float64)
        self._active_torques = np.zeros((self.num_envs, num_bodies, 3), dtype=np.float64)
        # Pre-sample the initial cooldown so the first impulse is preceded by a
        # cooldown rather than firing immediately at t=0.
        self._interval_time_left = self._sample_cooldown(self.num_envs)

    def _sample_cooldown(self, count: int) -> np.ndarray:
        return self._env.rng.uniform(self._cooldown_s[0], self._cooldown_s[1], size=count)

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        env_ids: np.ndarray | None,
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
        if np.any(expired):
            expired_ids = np.flatnonzero(expired)
            self._active[expired_ids] = False
            self._active_forces[expired_ids] = 0.0
            self._active_torques[expired_ids] = 0.0
            self._time_remaining[expired_ids] = 0.0
            self._interval_time_left[expired_ids] = self._sample_cooldown(len(expired_ids))

        # Decrement cooldown timers, then trigger eligible envs.
        self._interval_time_left -= dt
        eligible = (~self._active) & (self._interval_time_left <= 0.0)
        if np.any(eligible):
            trigger_ids = np.flatnonzero(eligible)
            count = len(trigger_ids)
            num_bodies = self._backend_body_ids.size
            forces = env.rng.uniform(
                self._force_range[0], self._force_range[1], size=(count, num_bodies, 3)
            )
            torques = env.rng.uniform(
                self._torque_range[0], self._torque_range[1], size=(count, num_bodies, 3)
            )
            if self._body_point_offset is not None:
                quats = self._entity.data.body_link_quat_w[trigger_ids][:, self._local_body_ids]
                offset_w = np_quat_apply_batched(
                    quats.reshape(-1, 4),
                    np.broadcast_to(self._body_point_offset, (count * num_bodies, 3)),
                ).reshape(count, num_bodies, 3)
                torques = torques + np.cross(offset_w, forces)
            self._active_forces[trigger_ids] = forces
            self._active_torques[trigger_ids] = torques
            self._time_remaining[trigger_ids] = env.rng.uniform(
                self._duration_s[0], self._duration_s[1], size=count
            )
            self._active[trigger_ids] = True
            self._interval_time_left[trigger_ids] = self._sample_cooldown(count)

        # Re-stage the full-width wrench while any impulse is active: backends
        # with one-shot external-force channels (MuJoCo) need the sustain
        # re-stage, and absolute channels (Motrix) treat it as an idempotent
        # target update. Newly expired rows stage zeros, which actively clears
        # persistent channels. Idle envs skip the call entirely so the term
        # never clobbers another producer's staged forces.
        if np.any(self._active) or np.any(expired):
            self._entity.apply_body_wrench_to_sim(
                self._active_forces,
                self._active_torques if self._uses_torque else None,
                self._backend_body_ids,
                env_ids=None,
                term_name="apply_body_impulse",
            )

    def reset(self, env_ids: np.ndarray | slice | None = None) -> None:
        ids = (
            np.arange(self.num_envs, dtype=np.intp)
            if env_ids is None
            else np.arange(self.num_envs, dtype=np.intp)[env_ids]
            if isinstance(env_ids, slice)
            else np.asarray(env_ids, dtype=np.intp)
        )
        was_active = ids[self._active[ids]]
        self._active[ids] = False
        self._active_forces[ids] = 0.0
        self._active_torques[ids] = 0.0
        self._time_remaining[ids] = 0.0
        self._interval_time_left[ids] = self._sample_cooldown(len(ids))
        if was_active.size:
            zeros = np.zeros((len(was_active), self._backend_body_ids.size, 3), dtype=np.float64)
            self._entity.apply_body_wrench_to_sim(
                zeros,
                zeros.copy() if self._uses_torque else None,
                self._backend_body_ids,
                env_ids=was_active,
                term_name="apply_body_impulse",
            )


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
        bias_range = np.asarray(cfg.params["bias_range"], dtype=np.float64)
        if bias_range.shape != (2,) or not np.isfinite(bias_range).all():
            raise ValueError(f"EventManager term '{term_name}' bias_range must be a finite pair")
        if bias_range[0] > bias_range[1]:
            raise ValueError(f"EventManager term '{term_name}' bias_range minimum exceeds maximum")
        self._range = (float(bias_range[0]), float(bias_range[1]))
        self._entity = cast("Entity", env.scene[asset_cfg.name])
        joint_ids = np.arange(self._entity.num_joints, dtype=np.intp)[asset_cfg.joint_ids]
        if joint_ids.ndim != 1 or joint_ids.size == 0:
            raise ValueError(
                f"EventManager term '{term_name}' asset_cfg must select at least one joint"
            )
        self._joint_ids = joint_ids

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        env_ids: np.ndarray | None,
        bias_range: tuple[float, float],
        asset_cfg: SceneEntityCfg,
    ) -> None:
        del bias_range, asset_cfg
        ids = resolve_env_ids(env, env_ids)
        self._entity.data.encoder_bias[np.ix_(ids, self._joint_ids)] = env.rng.uniform(
            self._range[0],
            self._range[1],
            size=(ids.size, self._joint_ids.size),
        )


randomize_encoder_bias = RandomizeEncoderBias


def reset_scene_to_default(env: ManagerBasedRlEnv, env_ids: np.ndarray | None) -> None:
    """Reset all materialized scene entities to backend default qpos/qvel."""
    ids = resolve_env_ids(env, env_ids)
    if not env.scene.entities:
        return
    env.scene.reset_to_default(ids, term_name="reset_scene_to_default")


def reset_root_state_uniform(
    env: ManagerBasedRlEnv,
    env_ids: np.ndarray | None,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]] | None = None,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
    """Reset a floating root from defaults plus uniformly sampled SE(3) offsets.

    This is the NumPy adaptation of the pinned mjlab event. UniLab currently has
    no public mocap-pose write contract, so fixed-base/mocap requests fail through
    the entity's cached floating-root capability instead of falling back.
    """
    ids = resolve_env_ids(env, env_ids)
    asset = cast("Entity", env.scene[asset_cfg.name])
    try:
        root_states = np.array(asset.data.default_root_state[ids], copy=True)
    except NotImplementedError as exc:
        raise NotImplementedError(
            "EventManager term 'reset_root_state_uniform' requires a floating-root "
            f"state for entity '{asset_cfg.name}'; fixed-base/mocap root reset is "
            f"unsupported without a formal backend contract: {exc}"
        ) from exc

    pose_samples = _sample_se3_range(pose_range, (len(ids), 6), env.rng)
    root_states[:, 0:3] = root_states[:, 0:3] + pose_samples[:, 0:3] + env.scene.env_origins[ids]
    orientation_delta = np_quat_from_euler_xyz(
        pose_samples[:, 3], pose_samples[:, 4], pose_samples[:, 5]
    )
    root_states[:, 3:7] = np_quat_mul(root_states[:, 3:7], orientation_delta)

    velocity_samples = _sample_se3_range(velocity_range, (len(ids), 6), env.rng)
    root_states[:, 7:13] = root_states[:, 7:13] + velocity_samples
    asset.write_root_state_to_sim(root_states, env_ids=ids)


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
    "reset_scene_to_default",
    "resolve_env_ids",
]
