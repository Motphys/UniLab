"""Stable Torch terrain spawn state for the Manager scene runtime."""

from __future__ import annotations

import math

import torch
from unisim.backend.base import SimBackend
from unisim.scene import TerrainSceneCfg


def _proportional_types(
    num_envs: int, proportions: list[float], *, device: torch.device
) -> torch.Tensor:
    total = sum(proportions)
    if not all(math.isfinite(value) and value >= 0.0 for value in proportions) or total <= 0.0:
        raise ValueError("Terrain sub_terrain proportions must be non-negative with a positive sum")
    normalized = [value / total for value in proportions]
    positive = [index for index, value in enumerate(proportions) if value > 0.0]
    counts = [0] * len(proportions)
    if num_envs >= len(positive):
        for index in positive:
            counts[index] = 1
    remaining = num_envs - sum(counts)
    ideal = [
        normalized[index] * remaining if index in positive else 0.0
        for index in range(len(proportions))
    ]
    floors = [int(value) for value in ideal]
    counts = [value + floor for value, floor in zip(counts, floors, strict=True)]
    leftover = remaining - sum(floors)
    order = sorted(positive, key=lambda index: (-(ideal[index] - floors[index]), index))
    for index in order[:leftover]:
        counts[index] += 1
    return torch.tensor(
        [column for column, count in enumerate(counts) for _ in range(count)],
        dtype=torch.int64,
        device=device,
    )


def _balanced_types(num_envs: int, num_cols: int, *, device: torch.device) -> torch.Tensor:
    return torch.tensor(
        [index * num_cols // num_envs for index in range(num_envs)],
        dtype=torch.int64,
        device=device,
    )


class Terrain:
    """Backend-materialized tile origins and mutable per-environment selections."""

    def __init__(
        self,
        cfg: TerrainSceneCfg | None,
        backend: SimBackend,
        *,
        device: str | torch.device,
    ) -> None:
        resolved = torch.device(device)
        if resolved.type == "cuda" and resolved.index is None:
            resolved = torch.device("cuda", index=torch.cuda.current_device())
        self._device = resolved
        self._curriculum = bool(
            cfg is not None and cfg.generator is not None and cfg.generator.curriculum
        )
        if cfg is None or cfg.generator is None:
            origins = torch.zeros((1, 1, 3), dtype=torch.float32, device=resolved)
            types = torch.zeros(backend.num_envs, dtype=torch.int64, device=resolved)
        else:
            generator = cfg.generator
            spawn_data = backend.get_terrain_spawn_data()
            if spawn_data is None:
                raise NotImplementedError(
                    f"Backend '{backend.backend_type}' declared generated scene terrain but "
                    "did not provide SimBackend.get_terrain_spawn_data metadata"
                )
            origins = torch.tensor(spawn_data.terrain_origins, dtype=torch.float32, device=resolved)
            expected_cols = (
                len(generator.sub_terrains) if generator.curriculum else generator.num_cols
            )
            expected = (generator.num_rows, expected_cols, 3)
            if tuple(origins.shape) != expected:
                raise ValueError(
                    f"Backend '{backend.backend_type}' terrain_origins has shape "
                    f"{tuple(origins.shape)}; expected {expected} from cfg.scene.terrain"
                )
            if not bool(torch.isfinite(origins).all()):
                raise ValueError("Backend terrain_origins must contain only finite values")
            if generator.curriculum:
                types = _proportional_types(
                    backend.num_envs,
                    [sub_terrain.proportion for sub_terrain in generator.sub_terrains.values()],
                    device=resolved,
                )
            else:
                types = _balanced_types(backend.num_envs, expected_cols, device=resolved)

        self.terrain_origins = origins
        self.terrain_levels = torch.zeros(backend.num_envs, dtype=torch.int64, device=resolved)
        self.terrain_types = types
        self.env_origins = origins[0, types].clone()

    @property
    def is_curriculum(self) -> bool:
        return self._curriculum

    def select_levels(self, env_ids: torch.Tensor, levels: torch.Tensor) -> None:
        """Select curriculum rows in place without changing types or unselected rows."""
        if not self._curriculum:
            raise NotImplementedError(
                "Terrain level selection requires cfg.scene.terrain.generator.curriculum=True"
            )
        if env_ids.ndim != 1 or env_ids.dtype != torch.int64 or env_ids.device != self._device:
            raise TypeError(f"Terrain env_ids must be int64 tensors on {self._device}")
        if (
            levels.shape != env_ids.shape
            or levels.dtype != torch.int64
            or levels.device != self._device
        ):
            raise TypeError(
                f"Terrain levels must be int64 tensors on {self._device} with env_ids shape"
            )
        if not env_ids.numel():
            return
        num_rows = self.terrain_origins.shape[0]
        if bool(((levels < 0) | (levels >= num_rows)).any()):
            raise IndexError(f"Terrain levels must be in [0, {num_rows})")
        self.terrain_levels.index_copy_(0, env_ids, levels)
        selected_types = self.terrain_types.index_select(0, env_ids)
        self.env_origins.index_copy_(0, env_ids, self.terrain_origins[levels, selected_types])


__all__ = ["Terrain"]
