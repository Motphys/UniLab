"""Shared named-sensor binding base for locomotion manager terms.

Both the G1 biped and the quadruped families port legacy reward/observation
equations that read IMU-style XML sensors by name.  This module owns the
common cold-path binding and fail-closed read contract so family modules only
declare which sensors they need.  Reward terms built on this base live in
``sensor_reward_terms.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import numpy as np

from unilab.managers.manager_base import ManagerTermBase, ManagerTermBaseCfg

if TYPE_CHECKING:
    from unilab.managers._types import ManagerBasedRlEnv, ManagerSensorView


class SensorTermBase(ManagerTermBase):
    """Cold-path named-sensor binding shared by locomotion manager terms."""

    _allowed_params: ClassVar[frozenset[str]] = frozenset()

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        unexpected = set(cfg.params) - self._allowed_params
        if unexpected:
            raise TypeError(f"{self.name} received unsupported parameters: {sorted(unexpected)}")

    def _bind(self, sensor_names: tuple[str, ...]) -> ManagerSensorView:
        try:
            return self._env.scene.bind_sensor_data(sensor_names)
        except (KeyError, TypeError, ValueError, NotImplementedError) as exc:
            raise type(exc)(
                f"Manager term '{self.name}' named-sensor capability could not be "
                f"materialized for {sensor_names}: {exc}"
            ) from exc

    @staticmethod
    def _read(view: ManagerSensorView, term: str) -> np.ndarray:
        try:
            return view.read()
        except (KeyError, TypeError, ValueError, NotImplementedError) as exc:
            raise type(exc)(
                f"Manager term '{term}' named-sensor capability failed on "
                f"backend '{view.backend_type}': {exc}"
            ) from exc


__all__ = ["SensorTermBase"]
