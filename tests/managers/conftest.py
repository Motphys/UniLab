# Derived from mujocolab/mjlab v1.6.0 (0fb8a681), manager test fixtures.
# Modified by UniLab for NumPy and the standalone manager contract; Apache-2.0.

from __future__ import annotations

import re
from types import SimpleNamespace

import numpy as np
import pytest


class FakeEntity:
    def __init__(self) -> None:
        self.joint_names = ["hip", "knee", "ankle"]
        self.body_names = ["base", "foot"]
        self.num_joints = len(self.joint_names)
        self.num_bodies = len(self.body_names)

    @staticmethod
    def _find(
        all_names: list[str], patterns: list[str], preserve_order: bool
    ) -> tuple[list[int], list[str]]:
        if preserve_order:
            found = [
                name for pattern in patterns for name in all_names if re.fullmatch(pattern, name)
            ]
        else:
            found = [name for name in all_names if any(re.fullmatch(p, name) for p in patterns)]
        return [all_names.index(name) for name in found], found

    def find_joints(
        self, patterns: list[str], *, preserve_order: bool = False
    ) -> tuple[list[int], list[str]]:
        return self._find(self.joint_names, patterns, preserve_order)

    def find_bodies(
        self, patterns: list[str], *, preserve_order: bool = False
    ) -> tuple[list[int], list[str]]:
        return self._find(self.body_names, patterns, preserve_order)


class FakeEnv:
    def __init__(self, seed: int = 7, num_envs: int = 4) -> None:
        self.num_envs = num_envs
        self.rng = np.random.default_rng(seed)
        self.scene = {"robot": FakeEntity()}
        self.max_episode_length_s = 2.0
        self.value = np.arange(num_envs, dtype=np.float32)
        self.obs = np.arange(num_envs * 2, dtype=np.float32).reshape(num_envs, 2)
        self.calls: list[tuple[str, np.ndarray | None]] = []
        self.obs_buf: dict[str, np.ndarray] = {}
        self.reset_buf = np.zeros(num_envs, dtype=np.bool_)

    def _mba_getter_total_ms(self) -> float:
        """Stand-in for ManagerBasedRlEnv's leaf getter timing probe."""
        return 0.0


@pytest.fixture
def fake_env() -> FakeEnv:
    return FakeEnv()
