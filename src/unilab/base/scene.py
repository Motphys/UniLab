from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from unilab.terrains.terrain_generator import TerrainGeneratorCfg


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
    # Optional render-only model override. When set, offline playback/video
    # export renders this XML instead of ``model_file`` while physics keeps
    # using ``model_file``. Used to give the renderer a visual twin of the
    # scene (e.g. a per-env replicable obstacle) without touching the trained
    # collision model. ``None`` => render with ``model_file`` (unchanged).
    visual_model_file: str | None = None
