# Derived from mujocolab/mjlab v1.6.0 (0fb8a681),
# src/mjlab/envs/mdp/observations.py.
# Copyright 2025, The mjlab Developers.
# Modified by UniLab for the base-owned entity facade; Apache-2.0.
"""Community-style observation terms for the Manager runtime."""

from __future__ import annotations

import math
import weakref
from typing import TYPE_CHECKING, Any, cast

import torch

from unilab.base.entity import EntityTensorStateView
from unilab.managers.manager_base import ManagerTermBase, ManagerTermBaseCfg
from unilab.managers.rng import ManagerRng
from unilab.managers.scene_entity_config import SceneEntityCfg
from unilab.utils.math import quat_apply, quat_apply_inverse, quat_from_angle_axis

if TYPE_CHECKING:
    from unilab.base.entity import Entity
    from unilab.managers.types import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def require_read_plan(env: ManagerBasedRlEnv, term_name: str) -> Any:
    """Return the scene's packed Torch read plan or fail closed for ``term_name``."""
    read_plan = getattr(env.scene, "read_plan", None)
    if read_plan is None:
        raise NotImplementedError(
            f"Observation term '{term_name}' requires the Manager packed Torch read plan"
        )
    return read_plan


def _root_body_name(asset: "Entity", term_name: str) -> str:
    root_body_name = getattr(asset, "root_body_name", None)
    if not isinstance(root_body_name, str) or not root_body_name:
        raise NotImplementedError(
            f"Observation term '{term_name}' requires entity '{asset.name}' "
            "to declare root_body_name"
        )
    return root_body_name


def root_body_view(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    term_name: str,
) -> tuple[Entity, Any]:
    """Return the entity and its packed root-body tensor view for ``term_name``."""
    asset = cast("Entity", env.scene[asset_cfg.name])
    read_plan = require_read_plan(env, term_name)
    return asset, read_plan.body_tensor_view(asset, (_root_body_name(asset, term_name),))


def joint_view(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg, term_name: str) -> Any:
    """Return the packed joint tensor view for the configured entity.

    Before the packed read plan is compiled, returns a zero-valued probe view
    with the declared joint width so shape probing works during manager
    construction.
    """
    asset = env.scene[asset_cfg.name]
    read_plan = getattr(env.scene, "read_plan", None)
    if read_plan is None:
        # ObservationManager probes shape before cold packed-read compilation.
        # Return a zeros-shaped probe from the immutable declared joint namespace.
        width = len(asset.joint_names)
        return EntityTensorStateView(
            joint_pos=torch.zeros(
                (env.num_envs, width),
                dtype=torch.float32,
                device=env.device,
            ),
            joint_vel=torch.zeros(
                (env.num_envs, width),
                dtype=torch.float32,
                device=env.device,
            ),
        )
    return require_read_plan(env, term_name).joint_tensor_view(asset)


class _NamedSensorObservation(ManagerTermBase):
    """Cold-path binding shared by pinned named-sensor observation terms."""

    _term_name = "named_sensor"
    _tensor_sensor_names: tuple[str, ...] = ()

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        """Resolve the ``sensor_name``/``entity_name`` params and bind the sensor view."""
        super().__init__(env)
        sensor_name = cfg.params.get("sensor_name")
        if not isinstance(sensor_name, str) or not sensor_name:
            raise ValueError(
                f"Observation term '{self._term_name}' capability 'named sensor' "
                "requires a non-empty sensor_name"
            )
        self._sensor_name = sensor_name
        self._entity_name = self._resolve_entity_name(cfg)
        self._tensor_sensor_names = (sensor_name,)
        try:
            self._view = env.scene.bind_sensor_data((sensor_name,))
        except (KeyError, TypeError, ValueError, NotImplementedError) as exc:
            raise type(exc)(
                f"Observation term '{self._term_name}' capability 'named sensor "
                f"{sensor_name}' could not be materialized: {exc}"
            ) from exc

    def _resolve_entity_name(self, cfg: ManagerTermBaseCfg) -> str:
        entity_name = cfg.params.get("entity_name")
        if not isinstance(entity_name, str) or not entity_name:
            return "robot"
        return entity_name

    def _validate_call_name(self, sensor_name: str) -> None:
        if sensor_name != self._sensor_name:
            raise ValueError(
                f"Observation term '{self._term_name}' was bound to sensor "
                f"'{self._sensor_name}', received '{sensor_name}'"
            )

    def _read(self) -> Any:
        try:
            return self._view.read()
        except (KeyError, TypeError, ValueError, NotImplementedError) as exc:
            raise type(exc)(
                f"Observation term '{self._term_name}' capability 'named sensor "
                f"{self._sensor_name}' failed on backend '{self._view.backend_type}': {exc}"
            ) from exc

    def _read_tensor(self, env: ManagerBasedRlEnv) -> torch.Tensor:
        read_plan = getattr(env.scene, "read_plan", None)
        if read_plan is None:
            # Manager construction probes immutable observation shape before
            # cold packed-read compilation. This is not a hot-path read.
            device = env.device
            return torch.as_tensor(self._read(), device=device)
        read_plan = require_read_plan(env, self._term_name)
        entity = env.scene[self._entity_name]
        sensor_names = self._tensor_sensor_names
        compiled = set(read_plan.sensor_names.get(self._entity_name, ()))
        if not set(sensor_names).issubset(compiled):
            raise NotImplementedError(
                f"Observation term '{self._term_name}' requires entity "
                f"'{self._entity_name}' sensors {sensor_names} in the packed Torch "
                f"read plan; compiled={tuple(sorted(compiled))}"
            )
        views = read_plan.sensor_tensor_views(entity, sensor_names)
        value = views.values[self._sensor_name]
        if getattr(env, "defer_finite_validation", False):
            record = getattr(env, "record_finite_validation", None)
            if callable(record):
                record(value, f"observation/{self._term_name}")
            elif not bool(torch.isfinite(value).all()):
                raise ValueError(f"Observation term '{self._term_name}' received NaN or Inf")
        elif not bool(torch.isfinite(value).all()):
            raise ValueError(
                f"Observation term '{self._term_name}' capability "
                f"'named sensor {self._sensor_name}' on backend "
                f"'{self._view.backend_type}' returned NaN or Inf"
            )
        return value

    @property
    def tensor_sensor_names(self) -> tuple[str, ...]:
        """Sensor names this term requires in the packed Torch read plan."""
        return self._tensor_sensor_names

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Row boundary for stateless sensor observations."""
        del env_ids


class builtin_sensor(_NamedSensorObservation):
    """Read one existing backend sensor through a cached NumPy view."""

    _term_name = "builtin_sensor"

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        sensor_name: str,
        entity_name: str | None = None,
    ) -> torch.Tensor:
        """Return the bound sensor's latest finite tensor reading."""
        del entity_name
        self._validate_call_name(sensor_name)
        return self._read_tensor(env)


class projected_gravity_from_sensor(_NamedSensorObservation):
    """Negate a cached 3-D up-vector sensor to obtain projected gravity."""

    _term_name = "projected_gravity_from_sensor"

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        """Bind the sensor view and require a 3-D up-vector output."""
        super().__init__(cfg, env)
        if self._view.dimensions != (3,):
            raise ValueError(
                "Observation term 'projected_gravity_from_sensor' capability "
                f"'3-D named sensor {self._sensor_name}' received dimensions "
                f"{self._view.dimensions} on backend '{self._view.backend_type}'"
            )

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        sensor_name: str,
        entity_name: str | None = None,
    ) -> torch.Tensor:
        """Return the negated 3-D up-vector reading (projected gravity)."""
        del entity_name
        self._validate_call_name(sensor_name)
        return -self._read_tensor(env)


def base_lin_vel(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Root body linear velocity expressed in the root body frame.

    Args:
        env: Manager-based RL environment providing the packed read plan.
        asset_cfg: Scene entity selection; defaults to the ``"robot"`` entity.

    Returns:
        A ``(num_envs, 3)`` tensor with the world-frame linear velocity
        rotated into the root body frame.
    """
    _, view = root_body_view(env, asset_cfg, "base_lin_vel")
    return quat_apply_inverse(view.quat_w[:, 0, :], view.lin_vel_w[:, 0, :])


def base_ang_vel(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Root body angular velocity expressed in the root body frame.

    Args:
        env: Manager-based RL environment providing the packed read plan.
        asset_cfg: Scene entity selection; defaults to the ``"robot"`` entity.

    Returns:
        A ``(num_envs, 3)`` tensor with the world-frame angular velocity
        rotated into the root body frame.
    """
    _, view = root_body_view(env, asset_cfg, "base_ang_vel")
    return quat_apply_inverse(view.quat_w[:, 0, :], view.ang_vel_w[:, 0, :])


def projected_gravity(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Gravity vector projected into the root body frame.

    Args:
        env: Manager-based RL environment providing the packed read plan.
        asset_cfg: Scene entity selection; defaults to the ``"robot"`` entity.

    Returns:
        A ``(num_envs, 3)`` tensor with the world gravity vector expressed in
        the root body frame.
    """
    asset, view = root_body_view(env, asset_cfg, "projected_gravity")
    gravity = asset.data.gravity_vec_w(env.device)[0]
    return quat_apply_inverse(view.quat_w[:, 0, :], gravity.expand((view.quat_w.shape[0], 3)))


# Per-env constant IMU mounting-misalignment quaternions, keyed by env instance
# and max_angle_rad. Sampled once per (env, angle) at term construction and
# shared by every misaligned term of that env, so the gyroscope and gravity
# observations see the SAME mounting error (upstream microduck recipe:
# microduck_rl/src/mjlab_microduck/tasks/mdp.py:3618-3660). Weak keys keep the
# cache from outliving the env.
_IMU_MISALIGNMENT_QUATS: weakref.WeakKeyDictionary[ManagerBasedRlEnv, dict[float, torch.Tensor]] = (
    weakref.WeakKeyDictionary()
)


def _imu_misalignment_quat(env: ManagerBasedRlEnv, max_angle_rad: float) -> torch.Tensor:
    """Per-env constant IMU mounting-misalignment rotation, sampled once.

    Models a fixed small mounting/calibration error of the IMU on each robot:
    a rotation about an axis uniform on the sphere with magnitude uniform in
    [0, max_angle_rad]. Sampled from ``env.rng`` on first use and cached for the
    whole run (a startup-style systematic per-robot bias, not per-step noise and
    not resampled on episode reset) — the behavior pinned from the upstream
    microduck reference above.

    Returns (num_envs, 4) unit quaternions (w, x, y, z) in float32.
    """
    per_env = _IMU_MISALIGNMENT_QUATS.get(env)
    if per_env is None:
        per_env = {}
        _IMU_MISALIGNMENT_QUATS[env] = per_env
    quat = per_env.get(max_angle_rad)
    if quat is None:
        rng_owner = getattr(env, "rng", None)
        if not isinstance(rng_owner, ManagerRng):
            raise RuntimeError(
                "IMU misalignment sampling requires the Manager-owned Torch generator"
            )
        axis = rng_owner.standard_normal((env.num_envs, 3))
        angle = rng_owner.uniform(0.0, max_angle_rad, (env.num_envs,), dtype=torch.float32)
        quat = quat_from_angle_axis(angle, axis).detach().to(dtype=torch.float32)
        per_env[max_angle_rad] = quat
    return quat


class _ImuMisalignedObservation(ManagerTermBase):
    """Cold-path binding shared by IMU mounting-misalignment observation terms."""

    _term_name = "imu_misaligned"

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        """Validate ``max_angle_deg`` and bind the per-env misalignment quaternion."""
        super().__init__(env)
        max_angle_deg = cfg.params.get("max_angle_deg", 1.0)
        if isinstance(max_angle_deg, bool) or not isinstance(max_angle_deg, (int, float)):
            raise ValueError(
                f"Observation term '{self._term_name}' capability 'IMU misalignment' "
                f"requires a real max_angle_deg, got {max_angle_deg!r}"
            )
        max_angle_deg = float(max_angle_deg)
        if not math.isfinite(max_angle_deg) or max_angle_deg < 0.0:
            raise ValueError(
                f"Observation term '{self._term_name}' capability 'IMU misalignment' "
                f"requires a finite non-negative max_angle_deg, got {max_angle_deg}"
            )
        self._max_angle_deg = max_angle_deg
        self._quat = _imu_misalignment_quat(env, math.radians(max_angle_deg))

    def _validate_max_angle(self, max_angle_deg: float) -> None:
        if float(max_angle_deg) != self._max_angle_deg:
            raise ValueError(
                f"Observation term '{self._term_name}' was bound to max_angle_deg "
                f"{self._max_angle_deg}, received {max_angle_deg}"
            )

    def _rotate(self, values: torch.Tensor) -> torch.Tensor:
        device = self._env.device
        vectors = values.to(dtype=torch.float32, device=device)
        quat = self._quat.to(device=device)
        return quat_apply(quat, vectors)


class base_ang_vel_imu_misaligned(_ImuMisalignedObservation):
    """Base angular velocity rotated by the per-env constant IMU misalignment."""

    _term_name = "base_ang_vel_imu_misaligned"

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        max_angle_deg: float = 1.0,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    ) -> torch.Tensor:
        """Return the base angular velocity rotated by the bound IMU misalignment."""
        self._validate_max_angle(max_angle_deg)
        _, view = root_body_view(env, asset_cfg, self._term_name)
        return self._rotate(quat_apply_inverse(view.quat_w[:, 0, :], view.ang_vel_w[:, 0, :]))


class projected_gravity_imu_misaligned(_ImuMisalignedObservation):
    """Projected gravity rotated by the SAME per-env IMU misalignment as the gyro."""

    _term_name = "projected_gravity_imu_misaligned"

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        max_angle_deg: float = 1.0,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    ) -> torch.Tensor:
        """Return projected gravity rotated by the bound IMU misalignment."""
        self._validate_max_angle(max_angle_deg)
        asset, view = root_body_view(env, asset_cfg, self._term_name)
        gravity = asset.data.gravity_vec_w(env.device)[0]
        projected = quat_apply_inverse(
            view.quat_w[:, 0, :], gravity.expand((view.quat_w.shape[0], 3))
        )
        return self._rotate(projected)


def joint_pos_rel(
    env: ManagerBasedRlEnv,
    biased: bool = False,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Joint positions relative to the default joint positions.

    Args:
        env: Manager-based RL environment.
        biased: When True, add the per-joint encoder bias to the relative
            positions.
        asset_cfg: Scene entity selection; defaults to the ``"robot"`` entity.

    Returns:
        A ``(num_envs, len(asset_cfg.joint_ids))`` tensor of relative joint
        positions.

    Raises:
        TypeError: If ``biased`` is not a ``bool``.
    """
    if not isinstance(biased, bool):
        raise TypeError(f"joint_pos_rel biased must be bool, got {type(biased).__name__}")
    joint_ids = asset_cfg.joint_ids
    device = env.device
    asset = env.scene[asset_cfg.name]
    state = joint_view(env, asset_cfg, "joint_pos_rel")
    result = state.joint_pos[:, joint_ids] - asset.data.default_joint_pos(device)[:, joint_ids]
    return (
        result
        if not biased
        else result + asset.data.encoder_bias_tensor.to(result.device)[:, joint_ids]
    )


def joint_vel_rel(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Joint velocities relative to the default joint velocities.

    Args:
        env: Manager-based RL environment.
        asset_cfg: Scene entity selection; defaults to the ``"robot"`` entity.

    Returns:
        A ``(num_envs, len(asset_cfg.joint_ids))`` tensor of relative joint
        velocities.
    """
    joint_ids = asset_cfg.joint_ids
    device = env.device
    asset = env.scene[asset_cfg.name]
    state = joint_view(env, asset_cfg, "joint_vel_rel")
    return state.joint_vel[:, joint_ids] - asset.data.default_joint_vel(device)[:, joint_ids]


def last_action(env: ManagerBasedRlEnv, action_name: str | None = None) -> torch.Tensor:
    """Latest raw action tensor published by the action manager.

    Args:
        env: Manager-based RL environment.
        action_name: When given, return only this action term's raw action
            instead of the manager's combined action buffer.

    Returns:
        A ``(num_envs, action_dim)`` tensor with the most recent raw actions.

    Raises:
        KeyError: If ``action_name`` does not name an action term.
    """
    if action_name is None:
        return env.action_manager.action
    try:
        action = env.action_manager.get_term(action_name).raw_action
    except KeyError as exc:
        raise KeyError(f"Action term '{action_name}' not found") from exc
    return action


def generated_commands(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Latest command tensor generated by the named command term.

    Args:
        env: Manager-based RL environment.
        command_name: Name of the command term to read.

    Returns:
        The command term's current command tensor.

    Raises:
        KeyError: If ``command_name`` does not name an active command term.
    """
    try:
        command = env.command_manager.get_command(command_name)
    except KeyError as exc:
        raise KeyError(f"Command term '{command_name}' not found") from exc
    if command is None:
        raise KeyError(f"Command term '{command_name}' not found")
    return command


__all__ = [
    "base_ang_vel",
    "base_ang_vel_imu_misaligned",
    "base_lin_vel",
    "builtin_sensor",
    "generated_commands",
    "joint_pos_rel",
    "joint_vel_rel",
    "joint_view",
    "last_action",
    "projected_gravity",
    "projected_gravity_from_sensor",
    "projected_gravity_imu_misaligned",
    "require_read_plan",
    "root_body_view",
]
