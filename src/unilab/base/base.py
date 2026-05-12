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
    motrix_max_iterations: Optional[int] = None

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
