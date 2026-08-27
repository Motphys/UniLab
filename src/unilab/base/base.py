import abc
from collections.abc import Callable
from dataclasses import dataclass
from os import PathLike
from typing import Any, Optional

import gymnasium as gym
import numpy as np

from unilab.base.backend.base import BackendPlayRenderPlan

from .scene import SceneCfg


@dataclass(frozen=True)
class EnvPlayCapabilities:
    """Env-facing play/render capabilities consumed by training entrypoints."""

    supports_native_interactive_renderer: bool = False
    supports_physics_state_playback: bool = False
    supports_native_video_capture: bool = False


@dataclass
class EnvCfg:
    """
    Config for the environment

    """

    scene: SceneCfg | None = None
    sim_dt: float = 0.01
    max_episode_seconds: Optional[float] = None
    ctrl_dt: float = 0.01
    render_spacing: float = 1.0
    render_offset_mode: str = "grid"
    drake_backend_mode: str = "batch"
    drake_nthread: int = 0
    motrix_max_iterations: Optional[int] = None
    post_step_forward_sensor: bool = False
    adaptive_chunk_size: bool = True
    chunk_size: Optional[int] = None
    # Explicit CPU block owned by this env's process (Linux affinity only).
    # ``cpu_ids[i]`` pins MuJoCo BatchEnvPool worker thread ``i`` to one CPU;
    # env construction also confines the owning process to the same block and
    # sizes Numba's parallel pool to ``len(cpu_ids)`` so host-side post-step
    # compute stays inside the rank's partition. ``None`` keeps the default
    # OS scheduling behavior.
    cpu_ids: Optional[list[int]] = None
    # ``mjwarp`` owns contact/constraint storage independently from MuJoCo.
    # Keep its capacity knobs explicit in the task owner configuration so a
    # device profile never relies on an implicit backend-wide allocation size.
    mjwarp_nconmax: Optional[int] = None
    mjwarp_njmax: Optional[int] = None
    # ``isaacgym`` runs physics in a Python 3.8 worker subprocess (Preview 4 is
    # EOL and incompatible with the main environment). ``None`` keeps the
    # backend defaults (device 0, generous handshake/step timeout).
    isaacgym_device_id: Optional[int] = None
    isaacgym_worker_timeout_s: Optional[float] = None

    @property
    def max_episode_steps(self) -> Optional[int]:
        """
        return the max episode steps
        """
        if self.max_episode_seconds is None:
            return None
        return int(self.max_episode_seconds / self.ctrl_dt)

    @property
    def sim_substeps(self) -> int:
        """
        return the number of simulation steps per control step
        """
        return int(round(self.ctrl_dt / self.sim_dt))

    def validate(self):
        """
        validate the config
        """
        if self.sim_dt > self.ctrl_dt:
            raise ValueError("sim_dt must be less than or equal to ctrl_dt")
        for name, value in (
            ("mjwarp_nconmax", self.mjwarp_nconmax),
            ("mjwarp_njmax", self.mjwarp_njmax),
        ):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
            ):
                raise ValueError(f"{name} must be a positive integer or None, got {value!r}")
        if self.isaacgym_device_id is not None and (
            isinstance(self.isaacgym_device_id, bool)
            or not isinstance(self.isaacgym_device_id, int)
            or self.isaacgym_device_id < 0
        ):
            raise ValueError(
                "isaacgym_device_id must be a non-negative integer or None, "
                f"got {self.isaacgym_device_id!r}"
            )
        if self.isaacgym_worker_timeout_s is not None and (
            not isinstance(self.isaacgym_worker_timeout_s, (int, float))
            or isinstance(self.isaacgym_worker_timeout_s, bool)
            or self.isaacgym_worker_timeout_s <= 0
        ):
            raise ValueError(
                "isaacgym_worker_timeout_s must be a positive number or None, "
                f"got {self.isaacgym_worker_timeout_s!r}"
            )
        if self.cpu_ids is not None:
            ids = list(self.cpu_ids)
            if not ids:
                raise ValueError("cpu_ids must be a non-empty sequence of CPU ids or None")
            for cpu_id in ids:
                if isinstance(cpu_id, bool) or not isinstance(cpu_id, int) or cpu_id < 0:
                    raise ValueError(
                        f"cpu_ids entries must be non-negative integers, got {cpu_id!r}"
                    )
            if len(set(ids)) != len(ids):
                raise ValueError(f"cpu_ids entries must be unique, got {ids!r}")


class ABEnv(abc.ABC):
    @property
    def play_capabilities(self) -> EnvPlayCapabilities:
        """Return env-facing play/render capabilities."""
        return EnvPlayCapabilities()

    def resolve_play_render_plan(
        self,
        *,
        play_render_mode: str | None,
        play_steps: int | None,
        output_video: str | PathLike[str] | None,
    ) -> BackendPlayRenderPlan:
        """Resolve high-level playback mode through the backend contract."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not define playback render mode semantics"
        )

    def run_playback(
        self,
        *,
        initialize: Callable[[], Any],
        step: Callable[[Any], Any],
        num_steps: int | None,
        output_video: str | PathLike[str] | None = None,
        render_spacing: float | None = None,
        render_offset_mode: str | None = None,
        headless: bool | None = None,
        record_video: bool | None = None,
        frame_state_getter: Callable[[], np.ndarray] | None = None,
        camera_kwargs: dict[str, Any] | None = None,
        extra_data_getter: Callable[[], np.ndarray | None] | None = None,
    ) -> str | None:
        """Execute playback through the backend contract."""
        raise NotImplementedError(f"{self.__class__.__name__} does not support playback execution")

    def run_playback_mode(
        self,
        *,
        play_render_mode: str | None,
        play_steps: int | None,
        output_video: str | PathLike[str] | None,
        initialize: Callable[[], Any],
        step: Callable[[Any], Any],
        render_spacing: float | None = None,
        render_offset_mode: str | None = None,
        frame_state_getter: Callable[[], np.ndarray] | None = None,
        camera_kwargs: dict[str, Any] | None = None,
        extra_data_getter: Callable[[], np.ndarray | None] | None = None,
        on_plan: Callable[[BackendPlayRenderPlan], None] | None = None,
    ) -> str | None:
        """Resolve configured playback mode and execute it through the backend contract."""
        plan = self.resolve_play_render_plan(
            play_render_mode=play_render_mode,
            play_steps=play_steps,
            output_video=output_video,
        )
        if on_plan is not None:
            on_plan(plan)
        if plan.mode == "none":
            return None
        return self.run_playback(
            initialize=initialize,
            step=step,
            num_steps=plan.num_steps,
            output_video=plan.output_video,
            render_spacing=render_spacing,
            render_offset_mode=render_offset_mode,
            headless=plan.headless,
            record_video=plan.record_video,
            frame_state_getter=frame_state_getter,
            camera_kwargs=camera_kwargs,
            extra_data_getter=extra_data_getter,
        )

    @property
    @abc.abstractmethod
    def num_envs(self) -> int:
        """
        return the size of the env if it is vectorized
        """

    @property
    @abc.abstractmethod
    def cfg(self) -> EnvCfg:
        """
        The configuration of the environment
        """

    @property
    @abc.abstractmethod
    def observation_space(self) -> gym.Space:
        """Observation space"""

    @property
    @abc.abstractmethod
    def action_space(self) -> gym.Space:
        """Action space"""

    @property
    @abc.abstractmethod
    def obs_groups_spec(self) -> dict[str, int]:
        """Map from observation group name to its dimension."""

    @property
    @abc.abstractmethod
    def state(self) -> Any:
        """Current environment state (None before first reset)"""

    @abc.abstractmethod
    def init_state(self) -> Any:
        """Initialize environment and return initial state"""

    @abc.abstractmethod
    def step(self, actions: np.ndarray) -> Any:
        """Step the environment with given actions, return new state"""

    @abc.abstractmethod
    def close(self) -> None:
        """Clean up environment resources"""

    def init_play_renderer(
        self,
        render_spacing: float | None = None,
        render_offset_mode: str | None = None,
        *,
        headless: bool = False,
        capture: bool = False,
        width: int = 1280,
        height: int = 720,
        camera_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Initialize env-facing playback rendering when supported."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support native playback rendering"
        )

    def render_play_frame(self) -> None:
        """Render one frame through the env-facing interactive playback contract."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support native interactive playback"
        )

    def capture_play_video_frame(self) -> np.ndarray:
        """Capture one RGB frame through the env-facing video contract."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support native video capture"
        )

    def get_physics_state_snapshot(self) -> np.ndarray:
        """Return a physics snapshot for offline playback/video export."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support physics-state playback"
        )

    def get_playback_model(self, env_index: int | None = None) -> Any:
        """Return a model object suitable for backend-specific playback tooling.

        Args:
            env_index: Optional vectorized environment index whose playback model
                should be returned when backend model variants differ across envs.

        Returns:
            A backend-specific playback model object.
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not expose a playback model")
