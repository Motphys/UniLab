"""Curriculum learning for adaptive difficulty adjustment."""

from __future__ import annotations

import numpy as np


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
