"""Reusable recorder terms for the NumPy Manager-Based runtime."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from unilab.managers.recorder_manager import RecorderTerm, RecorderTermCfg

if TYPE_CHECKING:
    from unilab.managers._types import ManagerBasedRlEnv


class LifecycleCounterRecorder(RecorderTerm):
    """Count recorder lifecycle calls without performing I/O."""

    def __init__(self, cfg: RecorderTermCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self.pre_reset_count = 0
        self.post_reset_count = 0
        self.post_step_count = 0

    def record_pre_reset(self, env_ids: np.ndarray) -> None:
        self.pre_reset_count += int(len(env_ids))

    def record_post_reset(self, env_ids: np.ndarray) -> None:
        self.post_reset_count += int(len(env_ids))

    def record_post_step(self) -> None:
        self.post_step_count += 1


__all__ = ["LifecycleCounterRecorder"]
