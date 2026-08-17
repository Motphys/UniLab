# Derived from mujocolab/mjlab v1.6.0 (0fb8a681),
# src/mjlab/envs/mdp/events.py.
# Copyright 2025, The mjlab Developers.
# Modified by UniLab for NumPy reset transactions; Apache-2.0.
"""Community-style reset event terms for UniLab's NumPy manager runtime."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

from unilab.managers.scene_entity_config import SceneEntityCfg
from unilab.utils.rotation import np_quat_from_euler_xyz, np_quat_mul

if TYPE_CHECKING:
    from unilab.base.entity import Entity
    from unilab.managers._types import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")
_SE3_KEYS = ("x", "y", "z", "roll", "pitch", "yaw")


def _sample_se3_range(
    range_dict: dict[str, tuple[float, float]] | None,
    shape: tuple[int, ...],
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample uniform ``[x, y, z, roll, pitch, yaw]`` offsets with NumPy."""
    if not shape or shape[-1] != len(_SE3_KEYS):
        raise ValueError(
            f"reset_root_state_uniform SE(3) sample shape must end in 6; received {shape}"
        )
    try:
        ranges = np.asarray(
            [(range_dict or {}).get(key, (0.0, 0.0)) for key in _SE3_KEYS],
            dtype=np.float64,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "reset_root_state_uniform ranges must map each SE(3) key to a numeric (min, max) pair"
        ) from exc
    if ranges.shape != (len(_SE3_KEYS), 2):
        raise ValueError(
            "reset_root_state_uniform ranges must map each SE(3) key to a "
            f"(min, max) pair; received shape {ranges.shape}"
        )
    if not np.isfinite(ranges).all():
        raise ValueError("reset_root_state_uniform ranges must contain only finite values")
    invalid = ranges[:, 0] > ranges[:, 1]
    if np.any(invalid):
        keys = [_SE3_KEYS[index] for index in np.flatnonzero(invalid)]
        raise ValueError(f"reset_root_state_uniform range minimum exceeds maximum for keys {keys}")
    return rng.uniform(ranges[:, 0], ranges[:, 1], size=shape)


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


def reset_root_state_uniform(
    env: ManagerBasedRlEnv,
    env_ids: np.ndarray | None,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]] | None = None,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
    """Reset a floating root from defaults plus uniformly sampled SE(3) offsets.

    This is the NumPy adaptation of the pinned mjlab event. UniLab currently has
    no public mocap-pose write contract, so fixed-base/mocap requests fail through
    the entity's cached floating-root capability instead of falling back.
    """
    ids = resolve_env_ids(env, env_ids)
    asset = cast("Entity", env.scene[asset_cfg.name])
    try:
        root_states = np.array(asset.data.default_root_state[ids], copy=True)
    except NotImplementedError as exc:
        raise NotImplementedError(
            "EventManager term 'reset_root_state_uniform' requires a floating-root "
            f"state for entity '{asset_cfg.name}'; fixed-base/mocap root reset is "
            f"unsupported without a formal backend contract: {exc}"
        ) from exc

    pose_samples = _sample_se3_range(pose_range, (len(ids), 6), env.rng)
    root_states[:, 0:3] = root_states[:, 0:3] + pose_samples[:, 0:3] + env.scene.env_origins[ids]
    orientation_delta = np_quat_from_euler_xyz(
        pose_samples[:, 3], pose_samples[:, 4], pose_samples[:, 5]
    )
    root_states[:, 3:7] = np_quat_mul(root_states[:, 3:7], orientation_delta)

    velocity_samples = _sample_se3_range(velocity_range, (len(ids), 6), env.rng)
    root_states[:, 7:13] = root_states[:, 7:13] + velocity_samples
    asset.write_root_state_to_sim(root_states, env_ids=ids)


__all__ = ["reset_root_state_uniform", "reset_scene_to_default", "resolve_env_ids"]
