"""UniLab configuration owner layered over the package-neutral UniSim scene."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
from unisim.dr.types import FixedVariantLayout, FixedVariantPlan, ModelSourceDescriptor
from unisim.entities import EntityInitialState, EntityVariantBinding, SceneEntitySpec
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

    model_file: str = ""
    fragment_files: list[str] = field(default_factory=list)
    terrain: TerrainSceneCfg | None = None
    entities: dict[str, object] = field(default_factory=dict)
    entity_assets: tuple[SceneEntitySpec, ...] = ()
    entity_variant: EntityVariantBinding | None = None
    primary_entity: str | None = None

    def materialize_entities(self) -> None:
        """Convert task/Hydra physical records without importing a physics SDK."""
        physical = []
        for raw in self.entity_assets:
            if isinstance(raw, SceneEntitySpec):
                physical.append(raw)
                continue
            if not isinstance(raw, Mapping):
                raise TypeError("entity_assets entries must be SceneEntitySpec or mappings")
            values = dict(raw)
            source = values.get("source")
            if isinstance(source, Mapping):
                values["source"] = ModelSourceDescriptor(**source)
            state = values.get("initial_state")
            if isinstance(state, Mapping):
                values["initial_state"] = EntityInitialState(
                    **{key: tuple(value) for key, value in state.items()}
                )
            physical.append(SceneEntitySpec(**values))
        self.entity_assets = tuple(physical)
        if isinstance(self.entity_variant, Mapping):
            binding = dict(self.entity_variant)
            plan = binding.get("plan")
            if isinstance(plan, Mapping):
                data = dict(plan)
                data["assignment"] = np.asarray(data["assignment"])
                data["variants"] = tuple(
                    ModelSourceDescriptor(**source) if isinstance(source, Mapping) else source
                    for source in data["variants"]
                )
                if "layout" in data:
                    data["layout"] = FixedVariantLayout(data["layout"])
                binding["plan"] = FixedVariantPlan(**data)
            self.entity_variant = EntityVariantBinding(**binding)
        if (
            self.entity_assets
            and self.primary_entity is not None
            and self.primary_entity not in {entity.name for entity in self.entity_assets}
        ):
            raise ValueError("primary_entity must name a declared physical entity")
        self.validate_composition()

    def __post_init__(self) -> None:
        self.materialize_entities()
        materialized: dict[str, object] = {}
        for name, value in self.entities.items():
            if isinstance(value, EntityCfg):
                materialized[name] = value
            elif isinstance(value, Mapping):
                materialized[name] = EntityCfg(**value)
            else:
                materialized[name] = value
        self.entities = materialized
        super().__post_init__()


__all__ = [
    "SceneCfg",
    "TerrainSceneCfg",
    "resolve_scene_default_qpos",
    "resolve_scene_fragment_path",
]
