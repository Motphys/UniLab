"""Typing-only contracts used by the standalone manager package.

The production environment and scene adapters implement these structural protocols in
later integration layers.  Keeping them here prevents the manager core from importing
an environment, backend, runner, or IPC implementation.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np


class ManagerEntity(Protocol):
    """Cold-path entity metadata required by :class:`SceneEntityCfg`."""

    joint_names: list[str]
    body_names: list[str]
    geom_names: list[str]
    site_names: list[str]
    actuator_names: list[str]
    tendon_names: list[str]
    camera_names: list[str]
    light_names: list[str]
    material_names: list[str]
    texture_names: list[str]
    pair_names: list[str]


class ManagerScene(Protocol):
    """Minimal name-addressable scene surface consumed by managers."""

    def __getitem__(self, name: str) -> ManagerEntity: ...


class ManagerBasedRlEnv(Protocol):
    """Structural context visible to manager terms.

    Additional task-owned state is intentionally not enumerated: term callables may use
    their concrete environment type, while the manager core depends only on this seam.
    """

    num_envs: int
    rng: np.random.Generator
    scene: ManagerScene
    max_episode_length_s: float

    def __getattr__(self, name: str) -> Any: ...


DebugVisualizer = Any
