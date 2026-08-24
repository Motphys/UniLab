"""Cold-path scene materialization for the independent ``mjwarp`` backend."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from unilab.base.scene import SceneCfg


class _TemporarySceneCleanup:
    """Own the temporary XMLs created while materializing one scene."""

    def __init__(self, *paths: str) -> None:
        self._paths = paths
        self._cleaned = False

    def cleanup(self) -> None:
        if self._cleaned:
            return
        self._cleaned = True
        for path in self._paths:
            try:
                os.remove(path)
            except FileNotFoundError:
                pass


@dataclass(frozen=True)
class MjwarpSceneContext:
    """Cold-path scene source and cleanup ownership for one backend instance."""

    source_model_file: str
    diagnostic_model_file: str
    cleanup_handle: Any | None = None
    tracked_body_names: tuple[str, ...] = ()


def materialize_mjwarp_scene(
    scene: SceneCfg,
    *,
    add_body_sensors: bool = False,
) -> MjwarpSceneContext:
    """Resolve a flat/fragments scene before CUDA model upload.

    Height-field terrain construction is intentionally rejected in the first
    correctness profile.  The rejection happens before model upload so an
    unsupported owner cannot silently fall back to a different terrain path.

    When ``add_body_sensors`` is set, world-frame body tracking sensors are
    injected into the resolved model on the cold path (same helper as the
    MuJoCo backend) so the host profile can serve body kinematics from its
    per-step sensor cache without extra device transfers.
    """
    if scene is None or not scene.model_file:
        raise ValueError("MjwarpBackend requires SceneCfg.model_file")
    if scene.terrain is not None:
        raise NotImplementedError(
            "mjwarp host_numpy profile does not support generated terrain or height-field "
            "scanners; select a flat owner YAML or a backend with terrain support."
        )
    temp_paths: list[str] = []
    if not scene.fragment_files:
        source_model_file = str(scene.model_file)
    else:
        # This is intentionally in a cold-path-only module.  The shared XML
        # composition helper is not a sibling runtime backend dependency.
        from unilab.base.backend.mujoco.xml import materialize_scene_fragments

        source_model_file = materialize_scene_fragments(
            str(scene.model_file),
            fragment_files=scene.fragment_files,
        )
        temp_paths.append(source_model_file)

    tracked_body_names: tuple[str, ...] = ()
    if add_body_sensors:
        from unilab.base.backend.mujoco.xml import inject_mujoco_tracking_sensors

        source_model_file, _tracked_body_ids, valid_bnames = inject_mujoco_tracking_sensors(
            source_model_file
        )
        temp_paths.append(source_model_file)
        tracked_body_names = tuple(valid_bnames)

    return MjwarpSceneContext(
        source_model_file=source_model_file,
        diagnostic_model_file=str(scene.model_file),
        cleanup_handle=_TemporarySceneCleanup(*temp_paths) if temp_paths else None,
        tracked_body_names=tracked_body_names,
    )
