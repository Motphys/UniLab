# Derived from mujocolab/mjlab v1.6.0 (0fb8a681),
# src/mjlab/envs/mdp/events.py.
# Copyright 2025, The mjlab Developers.
# Modified by UniLab for NumPy reset transactions; Apache-2.0.
"""Community-style reset event terms for UniLab's NumPy manager runtime."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from unilab.managers._types import ManagerBasedRlEnv


def resolve_env_ids(env: ManagerBasedRlEnv, env_ids: np.ndarray | None) -> np.ndarray:
    """Return concrete NumPy environment IDs, preserving community sentinel semantics."""
    if env_ids is None:
        return np.arange(env.num_envs, dtype=np.int32)
    return env_ids


def reset_scene_to_default(env: ManagerBasedRlEnv, env_ids: np.ndarray | None) -> None:
    """Reset all materialized scene entities to backend default qpos/qvel."""
    ids = resolve_env_ids(env, env_ids)
    if not env.scene.entities:
        return
    env.scene.reset_to_default(ids, term_name="reset_scene_to_default")


__all__ = ["reset_scene_to_default", "resolve_env_ids"]
