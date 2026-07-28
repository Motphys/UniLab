"""Cold-path scene materialization owned by the independent ``mjwarp`` backend."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from unilab.base.scene import SceneCfg


class _TemporarySceneCleanup:
    """Own the one temporary XML created while merging scene fragments."""

    def __init__(self, path: str) -> None:
        self._path = path
        self._cleaned = False

    def cleanup(self) -> None:
        if self._cleaned:
            return
        self._cleaned = True
        try:
            os.remove(self._path)
        except FileNotFoundError:
            pass


@dataclass(frozen=True)
class MjwarpSceneContext:
    """Cold-path scene source and cleanup ownership for one backend instance."""

    source_model_file: str
    diagnostic_model_file: str
    cleanup_handle: Any | None = None


def materialize_mjwarp_scene(scene: SceneCfg) -> MjwarpSceneContext:
    """Resolve a flat/fragments scene before CUDA model upload.

    Height-field terrain construction is intentionally rejected in the first
    correctness profile.  The rejection happens before model upload so an
    unsupported owner cannot silently fall back to a different terrain path.
    """
    if scene is None or not scene.model_file:
        raise ValueError("MjwarpBackend requires SceneCfg.model_file")
    if scene.terrain is not None:
        raise NotImplementedError(
            "mjwarp host_numpy profile does not support generated terrain or height-field "
            "scanners; select a flat owner YAML or a backend with terrain support."
        )
    if not scene.fragment_files:
        return MjwarpSceneContext(
            source_model_file=str(scene.model_file),
            diagnostic_model_file=str(scene.model_file),
        )

    # This is intentionally in a cold-path-only module.  The shared XML
    # composition helper is not a sibling runtime backend dependency.
    from unilab.base.backend.mujoco.xml import materialize_scene_fragments

    materialized = materialize_scene_fragments(
        str(scene.model_file),
        fragment_files=scene.fragment_files,
    )
    return MjwarpSceneContext(
        source_model_file=materialized,
        diagnostic_model_file=str(scene.model_file),
        cleanup_handle=_TemporarySceneCleanup(materialized),
    )
