"""UniLab configuration owner layered over the package-neutral UniSim scene."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from unisim.scene import (
    SceneCfg as _UniSimSceneCfg,
)
from unisim.scene import (
    TerrainSceneCfg,
    resolve_scene_default_qpos,
    resolve_scene_fragment_path,
)

from unilab.base.entity import EntityCfg


@dataclass
class SceneCfg(_UniSimSceneCfg):
    """Task-owned scene declaration with UniLab entity materialization.

    Physics scene fields and all backend-facing behavior are implemented by
    UniSim.  UniLab only converts Hydra/OmegaConf entity mappings into the
    manager facade's :class:`EntityCfg` records on the cold configuration path.
    """

    model_file: str
    fragment_files: list[str] = field(default_factory=list)
    terrain: TerrainSceneCfg | None = None
    entities: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        materialized: dict[str, object] = {}
        for name, value in self.entities.items():
            if isinstance(value, EntityCfg):
                materialized[name] = value
            elif isinstance(value, Mapping):
                materialized[name] = EntityCfg(**value)
            else:
                materialized[name] = value
        self.entities = materialized


__all__ = [
    "SceneCfg",
    "TerrainSceneCfg",
    "resolve_scene_default_qpos",
    "resolve_scene_fragment_path",
]
