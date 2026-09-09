"""Joint-target terms using the public NumPy entity/reset contracts."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

if TYPE_CHECKING:
    from unilab.base.entity import Entity
    from unilab.managers import ManagerTermBaseCfg
    from unilab.managers._types import ManagerBasedRlEnv


class JointTargetObservation:
    """Bind one ordered joint target on the manager construction path."""

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv) -> None:
        self._entity = cast("Entity", env.scene[cfg.params["entity_name"]])
        self._target = np.asarray(cfg.params["target"], dtype=np.float32)
        expected = (self._entity.num_joints,)
        if self._target.shape != expected or not np.isfinite(self._target).all():
            raise ValueError(f"FR3 joint target must be finite with shape {expected}")
        self._target.setflags(write=False)

    def __call__(self, env: ManagerBasedRlEnv, entity_name: str, target: list[float]) -> np.ndarray:
        return self._entity.data.joint_pos - self._target


class JointTargetReward(JointTargetObservation):
    """Reward current joint accuracy with a configured squared-error scale."""

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv) -> None:
        super().__init__(cfg, env)
        std = cfg.params["std"]
        if (
            isinstance(std, bool)
            or not isinstance(std, (int, float))
            or not np.isfinite(std)
            or std <= 0
        ):
            raise ValueError("FR3 joint target reward std must be finite and positive")
        self._variance = float(std) ** 2

    def __call__(
        self, env: ManagerBasedRlEnv, entity_name: str, target: list[float], std: float = 0.5
    ) -> np.ndarray:
        error = self._entity.data.joint_pos - self._target
        return np.exp(-np.sum(np.square(error), axis=-1) / self._variance)


class ResetJointOffsets:
    """Stage selected joint resets without requiring a floating root."""

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv) -> None:
        self._entity = cast("Entity", env.scene[cfg.params["entity_name"]])
        self._ranges = []
        for name in ("position_range", "velocity_range"):
            values = np.asarray(cfg.params[name], dtype=np.float64)
            if values.shape != (2,) or not np.isfinite(values).all() or values[0] > values[1]:
                raise ValueError(f"FR3 reset {name} must be a finite ordered pair")
            self._ranges.append((float(values[0]), float(values[1])))

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        env_ids: np.ndarray | None,
        entity_name: str,
        position_range: list[float],
        velocity_range: list[float],
    ) -> None:
        if env_ids is None:
            raise ValueError("FR3 reset requires explicit environment IDs")
        positions = self._entity.data.default_joint_pos[env_ids]
        velocities = self._entity.data.default_joint_vel[env_ids]
        self._entity.write_joint_state_to_sim(
            positions + env.rng.uniform(*self._ranges[0], size=positions.shape),
            velocities + env.rng.uniform(*self._ranges[1], size=velocities.shape),
            env_ids=env_ids,
        )
