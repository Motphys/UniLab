# Derived from mujocolab/mjlab v1.6.0 (0fb8a681),
# src/mjlab/envs/mdp/events.py.
# Copyright 2025, The mjlab Developers.
# Modified by UniLab for NumPy reset transactions; Apache-2.0.
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
_PD_GAIN_PARAM_NAMES = frozenset(("kp_range", "kd_range", "asset_cfg", "distribution", "operation"))


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


__all__ = ["pd_gains", "reset_root_state_uniform", "reset_scene_to_default", "resolve_env_ids"]
