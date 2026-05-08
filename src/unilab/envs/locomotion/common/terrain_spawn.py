"""Spawn-origin managers for locomotion envs.

``BaseSpawnManager`` is a no-op default: every env spawns at the world origin
(plus the existing per-env xy jitter from the dr_provider). Used whenever the
env has no procedural terrain — flat scenes don't need spatial separation

``TerrainSpawnManager`` overrides this for terrain scenes: it indexes
``terrain_origins[level, type_col]`` so each env spawns on a specific cell, and
optionally promotes/demotes ``level`` per-env on episode end. With
``enabled=True`` levels start at 0; with ``enabled=False`` levels are uniformly
distributed and never change — but spawn still uses cell-aware xyz so robots
land on the correct surface height.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class BaseSpawnManager:
    """Default no-op spawn manager: returns zeros, records nothing."""

    def origins_for(self, env_ids: np.ndarray) -> np.ndarray:
        return np.zeros((env_ids.shape[0], 3), dtype=np.float64)

    def record_episode_start(self, env_ids: np.ndarray, qpos_xyz: np.ndarray) -> None:
        del env_ids, qpos_xyz

    def update_on_done(self, done_indices: np.ndarray, current_xyz: np.ndarray) -> dict[str, float]:
        del done_indices, current_xyz
        return {}


@dataclass
class TerrainCurriculumCfg:
    enabled: bool = False
    """If True, levels start at 0 and evolve via promote/demote."""
    promote_frac: float = 0.5
    """Walked distance > promote_frac * cell_size promotes one level."""
    demote_frac: float = 0.25
    """Walked distance < demote_frac * cell_size demotes one level."""
    cycle_top_frac: float = 0.5
    """When level overflows the top row, resample uniformly in
    ``[num_rows * cycle_top_frac, num_rows - 1]``."""
    spawn_height_margin: float = 0.05
    """Extra z added on top of terrain_origins[..., 2] to absorb the surface
    approximation in random_rough / wave_terrain (their stored spawn z is the
    cell midline, not the cell-center actual surface)."""
    seed: int | None = None


class TerrainSpawnManager(BaseSpawnManager):
    def __init__(
        self,
        num_envs: int,
        terrain_origins: np.ndarray,
        cell_size: float,
        cfg: TerrainCurriculumCfg,
    ) -> None:
        if terrain_origins.ndim != 3 or terrain_origins.shape[2] != 3:
            raise ValueError(
                f"terrain_origins must have shape (num_rows, num_cols, 3); "
                f"got {terrain_origins.shape}"
            )
        num_rows, num_cols, _ = terrain_origins.shape
        if cfg.enabled and num_rows < 2:
            raise ValueError(
                f"Curriculum requires terrain_generator.num_rows >= 2; got {num_rows}."
            )

        self._terrain_origins = terrain_origins.astype(np.float64, copy=False)
        self._num_rows = num_rows
        self._num_cols = num_cols
        self._cell_size = float(cell_size)
        self._cfg = cfg
        self._rng = np.random.default_rng(cfg.seed)

        self.type_cols = self._rng.integers(0, num_cols, size=num_envs).astype(np.int32)
        if cfg.enabled:
            self.levels = np.zeros(num_envs, dtype=np.int32)
        else:
            self.levels = self._rng.integers(0, num_rows, size=num_envs).astype(np.int32)

        self._episode_start_xyz = np.zeros((num_envs, 3), dtype=np.float64)
        self._has_started = np.zeros(num_envs, dtype=bool)

    @property
    def enabled(self) -> bool:
        return self._cfg.enabled

    def origins_for(self, env_ids: np.ndarray) -> np.ndarray:
        rows = self.levels[env_ids]
        cols = self.type_cols[env_ids]
        out = self._terrain_origins[rows, cols].copy()
        out[:, 2] += self._cfg.spawn_height_margin
        return out

    def record_episode_start(self, env_ids: np.ndarray, qpos_xyz: np.ndarray) -> None:
        self._episode_start_xyz[env_ids] = qpos_xyz
        self._has_started[env_ids] = True

    def update_on_done(self, done_indices: np.ndarray, current_xyz: np.ndarray) -> dict[str, float]:
        active_mask = self._has_started[done_indices]
        active = done_indices[active_mask]
        num_skipped = int((~active_mask).sum())

        if active.size == 0:
            return {
                "mean_level": float(self.levels.mean()),
                "max_level": float(self.levels.max()),
                "mean_walked": 0.0,
                "num_promoted": 0,
                "num_demoted": 0,
                "num_skipped": num_skipped,
            }

        starts = self._episode_start_xyz[active, :2]
        ends = current_xyz[active_mask, :2]
        walked = np.linalg.norm(ends - starts, axis=1)

        num_promoted = 0
        num_demoted = 0
        if self._cfg.enabled:
            promote_threshold = self._cfg.promote_frac * self._cell_size
            demote_threshold = self._cfg.demote_frac * self._cell_size
            promote_mask = walked > promote_threshold
            demote_mask = walked < demote_threshold

            promote_ids = active[promote_mask]
            demote_ids = active[demote_mask]
            num_promoted = int(promote_ids.size)
            num_demoted = int(demote_ids.size)

            self.levels[promote_ids] += 1
            self.levels[demote_ids] -= 1

            overflow_mask = self.levels[promote_ids] >= self._num_rows
            if overflow_mask.any():
                lo = int(self._num_rows * self._cfg.cycle_top_frac)
                lo = min(max(lo, 0), self._num_rows - 1)
                overflow_ids = promote_ids[overflow_mask]
                self.levels[overflow_ids] = self._rng.integers(
                    lo, self._num_rows, size=overflow_ids.size
                ).astype(np.int32)

            np.clip(self.levels, 0, self._num_rows - 1, out=self.levels)

        return {
            "mean_level": float(self.levels.mean()),
            "max_level": float(self.levels.max()),
            "mean_walked": float(walked.mean()),
            "num_promoted": num_promoted,
            "num_demoted": num_demoted,
            "num_skipped": num_skipped,
        }
