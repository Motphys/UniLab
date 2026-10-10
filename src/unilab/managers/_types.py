"""Typing-only contracts used by the standalone manager package.

The production environment and scene adapters implement these structural protocols in
later integration layers.  Keeping them here prevents the manager core from importing
an environment, backend, runner, or IPC implementation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol

import numpy as np
import torch
from unisim.backend.base import SimBackend

from unilab.managers.torch_rng import TorchManagerRng


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
    def data(self) -> Any: ...

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

    def write_joint_state_to_sim(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        joint_ids: np.ndarray | Sequence[int] | slice | None = None,
        env_ids: np.ndarray | slice | None = None,
    ) -> None: ...

    def write_root_state_tensor_to_sim(
        self,
        root_state: torch.Tensor,
        env_ids: torch.Tensor | np.ndarray | slice | None = None,
    ) -> None: ...

    def write_motion_state_tensor_to_sim(
        self,
        root_state: torch.Tensor,
        position: torch.Tensor,
        velocity: torch.Tensor,
        env_ids: torch.Tensor,
    ) -> None: ...


class ManagerSensorView(Protocol):
    """Backend-owned named-sensor view retained by a manager term."""

    @property
    def backend_type(self) -> str: ...

    @property
    def names(self) -> tuple[str, ...]: ...

    @property
    def dimensions(self) -> tuple[int, ...]: ...

    @property
    def data(self) -> np.ndarray: ...

    def read(self) -> np.ndarray: ...


class ManagerScene(Protocol):
    """Minimal name-addressable scene surface consumed by managers."""

    @property
    def entities(self) -> Mapping[str, ManagerEntity]: ...

    def compile_tensor_reads(self, device: str | torch.device, specs: Sequence[Any]) -> Any: ...

    @property
    def env_origins(self) -> np.ndarray: ...

    def __getitem__(self, name: str) -> ManagerEntity: ...

    def bind_sensor_data(self, names: Sequence[str]) -> ManagerSensorView: ...

    def reset_to_default(self, env_ids: np.ndarray, *, term_name: str) -> None: ...

    def bind_gravity_write(self, *, term_name: str) -> np.ndarray: ...

    def write_gravity_to_sim(
        self, values: np.ndarray, env_ids: np.ndarray, *, term_name: str
    ) -> None: ...

    @property
    def _tensor_read_plan(self) -> Any: ...


class ManagerActionTerm(Protocol):
    @property
    def raw_action(self) -> torch.Tensor: ...

    def process_actions(self, actions: torch.Tensor) -> None: ...


class ManagerActionManager(Protocol):
    @property
    def action(self) -> torch.Tensor: ...

    @property
    def prev_action(self) -> torch.Tensor: ...

    @property
    def prev_prev_action(self) -> torch.Tensor: ...

    def get_term(self, name: str) -> ManagerActionTerm: ...


class ManagerCommandManager(Protocol):
    def get_command(self, name: str) -> torch.Tensor | None: ...

    def get_term(self, name: str) -> Any: ...

    def get_term_cfg(self, name: str) -> Any: ...


class ManagerEventManager(Protocol):
    """Cold-path event term config surface consumed by curriculum terms."""

    def get_term_cfg(self, term_name: str) -> Any: ...


class ManagerTerminationManager(Protocol):
    @property
    def terminated(self) -> torch.Tensor: ...

    def get_term_cfg(self, term_name: str) -> Any: ...


class ManagerRewardManager(Protocol):
    """Cold-path reward term config surface consumed by curriculum terms."""

    def get_term_cfg(self, term_name: str) -> Any: ...


class ManagerBasedRlEnv(Protocol):
    """Structural context visible to manager terms.

    Additional task-owned state is intentionally not enumerated: term callables may use
    their concrete environment type, while the manager core depends only on this seam.
    """

    @property
    def num_envs(self) -> int: ...

    @property
    def device(self) -> torch.device: ...

    @property
    def rng(self) -> np.random.Generator: ...

    @property
    def torch_rng(self) -> TorchManagerRng: ...

    @property
    def cfg(self) -> Any: ...

    @property
    def backend(self) -> SimBackend: ...

    _tensor_reset_default_root_state: torch.Tensor | None
    _tensor_reset_env_origins: torch.Tensor | None
    _tensor_reset_pose_bounds: torch.Tensor | None
    _tensor_reset_velocity_bounds: torch.Tensor | None

    @property
    def physics_dt(self) -> float: ...

    @property
    def step_dt(self) -> float: ...

    @property
    def scene(self) -> ManagerScene: ...

    @property
    def action_manager(self) -> ManagerActionManager: ...

    @property
    def command_manager(self) -> ManagerCommandManager: ...

    @property
    def event_manager(self) -> ManagerEventManager: ...

    @property
    def termination_manager(self) -> ManagerTerminationManager: ...

    @property
    def reward_manager(self) -> ManagerRewardManager: ...

    @property
    def metrics_manager(self) -> Any: ...

    @property
    def episode_length_buf(self) -> torch.Tensor: ...

    @property
    def reset_buf(self) -> torch.Tensor: ...

    @property
    def common_step_counter(self) -> int: ...

    @property
    def max_episode_length(self) -> int: ...

    @property
    def max_episode_length_s(self) -> float: ...

    # Concrete task terms may still type their own richer env subclass.  The
    # standalone manager core deliberately depends only on the properties above.


DebugVisualizer = Any
