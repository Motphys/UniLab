"""Curriculum learning for adaptive difficulty adjustment."""

from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from unilab.envs.locomotion.g1.joystick import G1JoystickEnv


class EpisodeLengthTracker:
    """Track moving average of episode length."""

    def __init__(self, num_envs: int, window_size: int = 1000):
        self.num_envs = num_envs
        self.window_size = max(1, int(window_size * num_envs / 4096))
        self.average_length = 0.0

    def update(self, episode_lengths: np.ndarray) -> None:
        """Update average with new episode lengths."""
        if len(episode_lengths) == 0:
            return
        current_avg = float(np.mean(episode_lengths))
        weight = min(len(episode_lengths) / self.window_size, 1.0)
        self.average_length = self.average_length * (1 - weight) + current_avg * weight


class PenaltyCurriculum:
    """Adaptive penalty scaling based on episode length."""

    def __init__(
        self,
        env: G1JoystickEnv,
        enabled: bool = True,
        initial_scale: float = 0.5,
        min_scale: float = 0.5,
        max_scale: float = 1.0,
        level_down_threshold: float = 150.0,
        level_up_threshold: float = 750.0,
        degree: float = 0.001,
    ):
        self.env = env
        self.enabled = enabled
        self.current_scale = initial_scale
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.level_down_threshold = level_down_threshold
        self.level_up_threshold = level_up_threshold
        self.degree = degree

        # Store original penalty weights
        self.penalty_names = []
        self.original_weights = {}

        if enabled:
            self._identify_penalties()
            self._apply_initial_scale()

    def _identify_penalties(self) -> None:
        """Identify penalty rewards (negative scales)."""
        for name, scale in self.env.cfg.reward_config.scales.items():
            if scale < 0:
                self.penalty_names.append(name)
                self.original_weights[name] = scale

    def _apply_initial_scale(self) -> None:
        """Apply initial penalty scaling."""
        for name in self.penalty_names:
            self.env.cfg.reward_config.scales[name] = (
                self.original_weights[name] * self.current_scale
            )

    def update(self, average_episode_length: float) -> None:
        """Update penalty scale based on average episode length."""
        if not self.enabled:
            return

        # Adjust scale
        if average_episode_length < self.level_down_threshold:
            self.current_scale *= 1.0 - self.degree
        elif average_episode_length > self.level_up_threshold:
            self.current_scale *= 1.0 + self.degree

        # Clamp
        self.current_scale = float(np.clip(self.current_scale, self.min_scale, self.max_scale))

        # Apply to all penalty rewards
        for name in self.penalty_names:
            self.env.cfg.reward_config.scales[name] = (
                self.original_weights[name] * self.current_scale
            )
