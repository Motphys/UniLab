"""Typing-only contracts used by the standalone manager package.

The production environment and scene adapters implement these structural protocols in
later integration layers.  Keeping them here prevents the manager core from importing
an environment, backend, runner, or IPC implementation.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np


class ManagerEntity(Protocol):
    """Cold-path entity metadata required by :class:`SceneEntityCfg`."""

    @property
    def joint_names(self) -> Sequence[str]: ...

    @property
    def body_names(self) -> Sequence[str]: ...

    @property
    def geom_names(self) -> Sequence[str]: ...

    @property
    def site_names(self) -> Sequence[str]: ...

    @property
    def actuator_names(self) -> Sequence[str]: ...

    @property
    def tendon_names(self) -> Sequence[str]: ...

    @property
    def camera_names(self) -> Sequence[str]: ...

    @property
    def light_names(self) -> Sequence[str]: ...

    @property
    def material_names(self) -> Sequence[str]: ...

    @property
    def texture_names(self) -> Sequence[str]: ...

    @property
    def pair_names(self) -> Sequence[str]: ...

    @property
    def num_joints(self) -> int: ...

    @property
    def num_bodies(self) -> int: ...

    @property
    def num_geoms(self) -> int: ...

    @property
    def num_sites(self) -> int: ...

    @property
    def num_actuators(self) -> int: ...

    @property
    def num_tendons(self) -> int: ...

    @property
    def num_cameras(self) -> int: ...

    @property
    def num_lights(self) -> int: ...

    @property
    def num_materials(self) -> int: ...

    @property
    def num_textures(self) -> int: ...

    @property
    def num_pairs(self) -> int: ...

    def find_joints(
        self, keys: str | Sequence[str], preserve_order: bool = False
    ) -> tuple[list[int], list[str]]: ...

    def find_bodies(
        self, keys: str | Sequence[str], preserve_order: bool = False
    ) -> tuple[list[int], list[str]]: ...

    def find_geoms(
        self, keys: str | Sequence[str], preserve_order: bool = False
    ) -> tuple[list[int], list[str]]: ...

    def find_sites(
        self, keys: str | Sequence[str], preserve_order: bool = False
    ) -> tuple[list[int], list[str]]: ...

    def find_actuators(
        self, keys: str | Sequence[str], preserve_order: bool = False
    ) -> tuple[list[int], list[str]]: ...

    def find_tendons(
        self, keys: str | Sequence[str], preserve_order: bool = False
    ) -> tuple[list[int], list[str]]: ...

    def find_cameras(
        self, keys: str | Sequence[str], preserve_order: bool = False
    ) -> tuple[list[int], list[str]]: ...

    def find_lights(
        self, keys: str | Sequence[str], preserve_order: bool = False
    ) -> tuple[list[int], list[str]]: ...

    def find_materials(
        self, keys: str | Sequence[str], preserve_order: bool = False
    ) -> tuple[list[int], list[str]]: ...

    def find_textures(
        self, keys: str | Sequence[str], preserve_order: bool = False
    ) -> tuple[list[int], list[str]]: ...

    def find_pairs(
        self, keys: str | Sequence[str], preserve_order: bool = False
    ) -> tuple[list[int], list[str]]: ...


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
