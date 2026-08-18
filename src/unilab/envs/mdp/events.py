# Derived from mujocolab/mjlab v1.6.0 (0fb8a681),
# src/mjlab/envs/mdp/events.py.
# Copyright 2025, The mjlab Developers.
# Modified by UniLab for NumPy reset transactions; Apache-2.0.
# The mass/CoM/gravity public names and signatures follow Isaac Lab v2.2.0;
# their implementation here is UniLab's original payload adapter, not vendored PhysX code.
"""Community-style reset event terms for UniLab's NumPy manager runtime."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np

from unilab.managers.event_manager import EventTermCfg
from unilab.managers.manager_base import ManagerTermBase
from unilab.managers.scene_entity_config import SceneEntityCfg
from unilab.utils.rotation import np_quat_from_euler_xyz, np_quat_mul

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
    if mode == "reset" and cfg.min_step_count_between_reset != 0:
        raise NotImplementedError(
            f"EventManager term '{term_name}' requires min_step_count_between_reset=0 "
            "because sparse reset payload rows cannot be represented"
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


class PdGains(ManagerTermBase):
    """Pinned-mjlab-compatible PD gain randomization on UniLab reset payloads."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        if cfg.mode != "reset":
            raise NotImplementedError(
                "EventManager term 'pd_gains' only supports mode='reset' on the UniLab "
                "set_state transaction; startup/interval/step model-field mutation is unavailable"
            )
        if cfg.min_step_count_between_reset != 0:
            raise NotImplementedError(
                "EventManager term 'pd_gains' requires min_step_count_between_reset=0 "
                "because sparse per-field reset rows cannot be represented by the current "
                "SimBackend.set_state payload"
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


class RandomizeRigidBodyCom(ManagerTermBase):
    """Community-compatible additive rigid-body CoM randomization."""

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
        self._ranges = _axis_ranges(
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
        del com_range, asset_cfg
        ids = resolve_env_ids(env, env_ids)
        offsets = env.rng.uniform(
            self._ranges[:, 0],
            self._ranges[:, 1],
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
    """Pinned community velocity kick dispatched through the interval plan."""

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
        if np.any(ranges[3:] != 0.0):
            raise NotImplementedError(
                f"EventManager term '{term_name}' angular velocity ranges are unsupported: "
                "IntervalRandomizationPlan only declares body_linear_velocity_delta"
            )
        self._ranges = ranges[:3]
        self._entity = cast("Entity", env.scene[asset_cfg.name])
        self._entity.bind_root_linear_velocity_delta(term_name=term_name)

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        env_ids: np.ndarray | None,
        velocity_range: dict[str, tuple[float, float]],
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    ) -> None:
        del velocity_range, asset_cfg
        ids = resolve_env_ids(env, env_ids)
        delta = env.rng.uniform(
            self._ranges[:, 0],
            self._ranges[:, 1],
            size=(ids.size, 3),
        )
        self._entity.apply_root_linear_velocity_delta_to_sim(
            delta,
            env_ids=ids,
            term_name="push_by_setting_velocity",
        )


push_by_setting_velocity = PushBySettingVelocity


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
    "pd_gains",
    "push_by_setting_velocity",
    "randomize_physics_scene_gravity",
    "randomize_rigid_body_com",
    "randomize_rigid_body_mass",
    "reset_root_state_uniform",
    "reset_scene_to_default",
    "resolve_env_ids",
]
