"""Typing-only contracts used by the standalone manager package.

The production environment and scene adapters implement these structural protocols in
later integration layers.  Keeping them here prevents the manager core from importing
an environment, backend, runner, or IPC implementation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Protocol

import torch
from unisim.backend.base import SimBackend

from unilab.base.reset_selectors import ResetMaskBuffer
from unilab.managers.rng import ManagerRng

if TYPE_CHECKING:
    from unilab.base.terrain import Terrain


class ManagerEntity(Protocol):
    """Cold-path entity metadata required by :class:`SceneEntityCfg`."""

    @property
    def joint_names(self) -> Sequence[str]:
        """Declared joint names in model order."""
        ...

    @property
    def body_names(self) -> Sequence[str]:
        """Declared body names in model order."""
        ...

    @property
    def geom_names(self) -> Sequence[str]:
        """Declared geom names in model order."""
        ...

    @property
    def site_names(self) -> Sequence[str]:
        """Declared site names in model order."""
        ...

    @property
    def actuator_names(self) -> Sequence[str]:
        """Declared actuator names in model order."""
        ...

    @property
    def tendon_names(self) -> Sequence[str]:
        """Declared tendon names in model order."""
        ...

    @property
    def camera_names(self) -> Sequence[str]:
        """Declared camera names in model order."""
        ...

    @property
    def light_names(self) -> Sequence[str]:
        """Declared light names in model order."""
        ...

    @property
    def material_names(self) -> Sequence[str]:
        """Declared material names in model order."""
        ...

    @property
    def texture_names(self) -> Sequence[str]:
        """Declared texture names in model order."""
        ...

    @property
    def pair_names(self) -> Sequence[str]:
        """Declared body-pair names in model order."""
        ...

    @property
    def data(self) -> Any:
        """Entity data facade consumed by manager terms."""
        ...

    @property
    def num_joints(self) -> int:
        """Number of declared joints."""
        ...

    @property
    def num_bodies(self) -> int:
        """Number of declared bodies."""
        ...

    @property
    def num_geoms(self) -> int:
        """Number of declared geoms."""
        ...

    @property
    def num_sites(self) -> int:
        """Number of declared sites."""
        ...

    @property
    def num_actuators(self) -> int:
        """Number of declared actuators."""
        ...

    @property
    def num_tendons(self) -> int:
        """Number of declared tendons."""
        ...

    @property
    def num_cameras(self) -> int:
        """Number of declared cameras."""
        ...

    @property
    def num_lights(self) -> int:
        """Number of declared lights."""
        ...

    @property
    def num_materials(self) -> int:
        """Number of declared materials."""
        ...

    @property
    def num_textures(self) -> int:
        """Number of declared textures."""
        ...

    @property
    def num_pairs(self) -> int:
        """Number of declared body pairs."""
        ...

    def find_joints(
        self, keys: str | Sequence[str], preserve_order: bool = False
    ) -> tuple[list[int], list[str]]:
        """Resolve name or regex keys to ``(ids, names)``; ``preserve_order`` keeps key order."""
        ...

    def find_bodies(
        self, keys: str | Sequence[str], preserve_order: bool = False
    ) -> tuple[list[int], list[str]]:
        """Resolve name or regex keys to ``(ids, names)``; ``preserve_order`` keeps key order."""
        ...

    def find_geoms(
        self, keys: str | Sequence[str], preserve_order: bool = False
    ) -> tuple[list[int], list[str]]:
        """Resolve name or regex keys to ``(ids, names)``; ``preserve_order`` keeps key order."""
        ...

    def find_sites(
        self, keys: str | Sequence[str], preserve_order: bool = False
    ) -> tuple[list[int], list[str]]:
        """Resolve name or regex keys to ``(ids, names)``; ``preserve_order`` keeps key order."""
        ...

    def find_actuators(
        self, keys: str | Sequence[str], preserve_order: bool = False
    ) -> tuple[list[int], list[str]]:
        """Resolve name or regex keys to ``(ids, names)``; ``preserve_order`` keeps key order."""
        ...

    def find_tendons(
        self, keys: str | Sequence[str], preserve_order: bool = False
    ) -> tuple[list[int], list[str]]:
        """Resolve name or regex keys to ``(ids, names)``; ``preserve_order`` keeps key order."""
        ...

    def find_cameras(
        self, keys: str | Sequence[str], preserve_order: bool = False
    ) -> tuple[list[int], list[str]]:
        """Resolve name or regex keys to ``(ids, names)``; ``preserve_order`` keeps key order."""
        ...

    def find_lights(
        self, keys: str | Sequence[str], preserve_order: bool = False
    ) -> tuple[list[int], list[str]]:
        """Resolve name or regex keys to ``(ids, names)``; ``preserve_order`` keeps key order."""
        ...

    def find_materials(
        self, keys: str | Sequence[str], preserve_order: bool = False
    ) -> tuple[list[int], list[str]]:
        """Resolve name or regex keys to ``(ids, names)``; ``preserve_order`` keeps key order."""
        ...

    def find_textures(
        self, keys: str | Sequence[str], preserve_order: bool = False
    ) -> tuple[list[int], list[str]]:
        """Resolve name or regex keys to ``(ids, names)``; ``preserve_order`` keeps key order."""
        ...

    def find_pairs(
        self, keys: str | Sequence[str], preserve_order: bool = False
    ) -> tuple[list[int], list[str]]:
        """Resolve name or regex keys to ``(ids, names)``; ``preserve_order`` keeps key order."""
        ...

    def write_joint_state_to_sim(
        self,
        position: torch.Tensor,
        velocity: torch.Tensor,
        joint_ids: torch.Tensor | Sequence[int] | slice | None = None,
        env_ids: torch.Tensor | slice | None = None,
    ) -> None:
        """Write joint position/velocity columns for the selected rows into the sim."""
        ...

    def write_root_state_tensor_to_sim(
        self,
        root_state: torch.Tensor,
        reset_mask: torch.Tensor,
    ) -> None:
        """Write a 13-D root state for the rows selected by ``reset_mask``."""
        ...

    def write_motion_state_tensor_to_sim(
        self,
        root_state: torch.Tensor,
        position: torch.Tensor,
        velocity: torch.Tensor,
        reset_mask: torch.Tensor,
    ) -> None:
        """Write root and joint motion state for the rows selected by ``reset_mask``."""
        ...


class ManagerSensorView(Protocol):
    """Backend-owned named-sensor view retained by a manager term."""

    @property
    def backend_type(self) -> str:
        """Backend type identifier of the owning view."""
        ...

    @property
    def names(self) -> tuple[str, ...]:
        """Bound sensor names."""
        ...

    @property
    def dimensions(self) -> tuple[int, ...]:
        """Per-reading dimensions of the bound sensor."""
        ...

    def read(self) -> Any:
        """Return the latest sensor reading."""
        ...


class ManagerScene(Protocol):
    """Minimal name-addressable scene surface consumed by managers."""

    @property
    def entities(self) -> Mapping[str, ManagerEntity]:
        """Name-addressable scene entities."""
        ...

    @property
    def terrain(self) -> Terrain:
        """Scene terrain provider."""
        ...

    def compile_tensor_reads(
        self,
        device: str | torch.device,
        specs: Sequence[Any],
        body_pair_specs: Sequence[Any] = (),
    ) -> Any:
        """Compile the packed Torch read plan for ``specs`` on ``device``."""
        ...

    def env_origins(self, device: str | torch.device) -> torch.Tensor:
        """Return per-environment origin positions on ``device``."""
        ...

    def __getitem__(self, name: str) -> ManagerEntity:
        """Return the entity registered under ``name``."""
        ...

    def bind_sensor_data(self, names: Sequence[str]) -> ManagerSensorView:
        """Bind a named-sensor view retained by a manager term."""
        ...

    def reset_to_default(self, env_ids: torch.Tensor, *, term_name: str) -> None:
        """Restore default state for the rows selected by ``env_ids``."""
        ...

    def bind_gravity_write_tensor(
        self, *, term_name: str, device: str | torch.device
    ) -> torch.Tensor:
        """Bind the per-environment gravity write table on ``device``."""
        ...

    def write_gravity_to_sim(
        self, values: torch.Tensor, env_ids: torch.Tensor, *, term_name: str
    ) -> None:
        """Write per-row gravity vectors into the sim."""
        ...

    @property
    def read_plan(self) -> Any:
        """Compiled packed Torch read plan, when available."""
        ...


class ManagerActionTerm(Protocol):
    """Action-term surface consumed by observation terms."""

    @property
    def raw_action(self) -> torch.Tensor:
        """Latest raw action tensor received by the term."""
        ...

    def process_actions(self, actions: torch.Tensor) -> None:
        """Process one batch of raw policy actions."""
        ...


class ManagerActionManager(Protocol):
    """Action-manager surface consumed by manager terms."""

    @property
    def action(self) -> torch.Tensor:
        """Combined latest action tensor across terms."""
        ...

    @property
    def prev_action(self) -> torch.Tensor:
        """Action tensor from the previous step."""
        ...

    @property
    def prev_prev_action(self) -> torch.Tensor:
        """Action tensor from two steps ago."""
        ...

    def get_term(self, name: str) -> ManagerActionTerm:
        """Return the action term registered under ``name``."""
        ...

    def clear_action_state(self, env_ids: torch.Tensor | None = None) -> None:
        """Clear buffered action state for the selected rows."""
        ...


class ManagerCommandManager(Protocol):
    """Command-manager surface consumed by observation terms."""

    def get_command(self, name: str) -> torch.Tensor | None:
        """Return the named term's current command tensor, or ``None``."""
        ...

    def get_term(self, name: str) -> Any:
        """Return the command term registered under ``name``."""
        ...

    def get_term_cfg(self, name: str) -> Any:
        """Return the live config of the command term registered under ``name``."""
        ...


class ManagerEventManager(Protocol):
    """Cold-path event term config surface consumed by curriculum terms."""

    def get_term_cfg(self, term_name: str) -> Any:
        """Return the live config of the event term registered under ``term_name``."""
        ...


class ManagerTerminationManager(Protocol):
    """Termination-manager surface consumed by manager terms."""

    @property
    def terminated(self) -> torch.Tensor:
        """Per-row termination flags for the current step."""
        ...

    def get_term_cfg(self, term_name: str) -> Any:
        """Return the live config of the termination term registered under ``term_name``."""
        ...


class ManagerRewardManager(Protocol):
    """Cold-path reward term config surface consumed by curriculum terms."""

    def get_term_cfg(self, term_name: str) -> Any:
        """Return the live config of the reward term registered under ``term_name``."""
        ...


class ManagerBasedRlEnv(Protocol):
    """Structural context visible to manager terms.

    Additional task-owned state is intentionally not enumerated: term callables may use
    their concrete environment type, while the manager core depends only on this seam.
    """

    @property
    def num_envs(self) -> int:
        """Number of vectorized environment rows."""
        ...

    @property
    def device(self) -> torch.device:
        """Torch device owning the environment tensors."""
        ...

    @property
    def rng(self) -> ManagerRng:
        """Manager-owned Torch random generator."""
        ...

    @property
    def cfg(self) -> Any:
        """Resolved environment configuration."""
        ...

    @property
    def backend(self) -> SimBackend:
        """Physics backend bound to this environment."""
        ...

    tensor_reset_pose_bounds: torch.Tensor | None
    tensor_reset_velocity_bounds: torch.Tensor | None
    tensor_reset_bounds_key: tuple[Any, ...] | None
    # Short-lived ids→mask scratch shared by event terms within one reset
    # dispatch; the returned mask is valid only until the next scatter.
    event_term_mask_buffer: ResetMaskBuffer

    @property
    def physics_dt(self) -> float:
        """Physics integration timestep in seconds."""
        ...

    @property
    def step_dt(self) -> float:
        """Policy step interval in seconds."""
        ...

    @property
    def scene(self) -> ManagerScene:
        """Name-addressable scene surface."""
        ...

    @property
    def action_manager(self) -> ManagerActionManager:
        """Action manager instance."""
        ...

    @property
    def command_manager(self) -> ManagerCommandManager:
        """Command manager instance."""
        ...

    @property
    def observation_manager(self) -> Any:
        """Observation manager instance."""
        ...

    @property
    def event_manager(self) -> ManagerEventManager:
        """Event manager instance."""
        ...

    @property
    def termination_manager(self) -> ManagerTerminationManager:
        """Termination manager instance."""
        ...

    @property
    def reward_manager(self) -> ManagerRewardManager:
        """Reward manager instance."""
        ...

    @property
    def metrics_manager(self) -> Any:
        """Metrics manager instance."""
        ...

    @property
    def episode_length_buf(self) -> torch.Tensor:
        """Per-row episode length counters."""
        ...

    @property
    def defer_finite_validation(self) -> bool:
        """Whether finite checks accumulate on the device between polls."""
        ...

    def record_finite_validation(self, value: torch.Tensor, label: str = "") -> None:
        """Accumulate one finite verdict into the env's deferred poll boundary."""
        ...

    @property
    def reset_buf(self) -> torch.Tensor:
        """Per-row reset flags."""
        ...

    @property
    def common_step_counter(self) -> int:
        """Total number of environment steps taken."""
        ...

    @property
    def max_episode_length(self) -> int:
        """Maximum episode length in steps."""
        ...

    @property
    def max_episode_length_s(self) -> float:
        """Maximum episode length in seconds."""
        ...

    # Concrete task terms may still type their own richer env subclass.  The
    # standalone manager core deliberately depends only on the properties above.
