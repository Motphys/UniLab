from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from unilab.base.entity import EntityCfg
from unilab.terrains.terrain_generator import TerrainGeneratorCfg

if TYPE_CHECKING:
    from unilab.base.backend.base import SimBackend


def resolve_scene_fragment_path(fragment_file: str, model_file: Path) -> Path:
    """Resolve a ``SceneCfg.fragment_files`` entry against the scene model file.

    Single resolution rule shared by the MuJoCo and Motrix scene
    materializers: absolute paths pass through; relative paths that exist
    resolve against the CWD; anything else resolves relative to the model
    file's directory.
    """
    path = Path(fragment_file)
    if path.is_absolute():
        return path
    if path.is_file():
        return path.resolve()
    return (model_file.parent / path).resolve()


@dataclass
class TerrainSceneCfg:
    """Backend-agnostic terrain slot declaration for a scene."""

    generator: TerrainGeneratorCfg | None = None
    hfield_name: str = "terrain_hfield"
    geom_name: str | None = None


@dataclass
class SceneCfg:
    """Scene source and optional cold-path composition configuration."""

    model_file: str
    fragment_files: list[str] = field(default_factory=list)
    terrain: TerrainSceneCfg | None = None
    entities: dict[str, EntityCfg] = field(default_factory=dict)
    """Logical entity partitions materialized by the base-owned manager facade."""
    # Optional render-only model override. When set, offline playback/video
    # export renders this XML instead of ``model_file`` while physics keeps
    # using ``model_file``. Used to give the renderer a visual twin of the
    # scene (e.g. a per-env replicable obstacle) without touching the trained
    # collision model. ``None`` => render with ``model_file`` (unchanged).
    visual_model_file: str | None = None
    default_keyframe_name: str | None = None
    """Optional named keyframe used as the Manager-Based default state."""


def resolve_scene_default_qpos(cfg: SceneCfg, backend: SimBackend) -> np.ndarray | None:
    """Resolve one named default-qpos snapshot without changing the qpos0 path."""
    keyframe_name = cfg.default_keyframe_name
    if keyframe_name is not None and not isinstance(keyframe_name, str):
        raise TypeError(
            "SceneCfg default_keyframe_name must be a non-empty string or None, "
            f"got {type(keyframe_name).__name__}"
        )
    if keyframe_name == "":
        raise ValueError("SceneCfg default_keyframe_name must be a non-empty string or None")
    if keyframe_name is None:
        return None

    capability = f"default keyframe {keyframe_name!r} qpos"
    try:
        value = backend.get_keyframe_qpos(keyframe_name)
    except (AttributeError, NotImplementedError) as exc:
        raise NotImplementedError(
            f"Manager scene default keyframe {keyframe_name!r} is unavailable on "
            f"backend '{backend.backend_type}': {exc}"
        ) from exc
    except (KeyError, ValueError) as exc:
        raise ValueError(
            f"Manager scene could not resolve default keyframe {keyframe_name!r} on "
            f"backend '{backend.backend_type}': {exc}"
        ) from exc

    if not isinstance(value, np.ndarray):
        raise TypeError(
            f"Manager scene {capability} on backend '{backend.backend_type}' must return "
            f"np.ndarray, got {type(value).__name__}"
        )
    if value.ndim != 1:
        raise ValueError(
            f"Manager scene {capability} on backend '{backend.backend_type}' returned shape "
            f"{value.shape}; expected 1-D"
        )
    if not np.issubdtype(value.dtype, np.floating):
        raise TypeError(
            f"Manager scene {capability} on backend '{backend.backend_type}' must be "
            f"floating, got {value.dtype}"
        )
    if not np.isfinite(value).all():
        raise ValueError(
            f"Manager scene {capability} on backend '{backend.backend_type}' returned NaN or Inf"
        )
    resolved = np.array(value, copy=True)
    resolved.setflags(write=False)
    return resolved
