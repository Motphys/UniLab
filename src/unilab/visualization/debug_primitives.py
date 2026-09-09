"""Interactive-viewer injection for :class:`unisim.backend.base.DebugPrimitive`.

The offline record pipeline renders debug overlays inside unisim's render
workers.  Interactive MuJoCo viewers instead own a live ``viewer.user_scn``
(mjvScene), so this module converts the same typed primitives into user-scene
geoms for that path.  It is UniLab-owned until unisim exposes a public
interactive-path helper (single env, already-loaded model, caller-owned
mjvScene); see ADR-0008 and unilabsim/wuji_unilab#21.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from unisim.backend.base import DebugPrimitive


def quat_from_z_axis(direction: Sequence[float]) -> tuple[float, float, float, float]:
    """Return the unit wxyz quaternion rotating the +z axis onto ``direction``."""
    vec = np.asarray(direction, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(vec))
    if norm < 1e-12:
        raise ValueError("direction must be non-zero")
    vec /= norm
    # axis = z × d, w = 1 + z·d (normalized); degenerates when d ≈ -z.
    w = 1.0 + float(vec[2])
    if w < 1e-9:
        return (0.0, 1.0, 0.0, 0.0)
    quat = np.array([w, -vec[1], vec[0], 0.0], dtype=np.float64)
    quat /= np.linalg.norm(quat)
    return (float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))


def segment_arrow(
    p0: Sequence[float],
    p1: Sequence[float],
    *,
    rgba: Sequence[float],
    min_length: float = 1e-6,
) -> DebugPrimitive | None:
    """Build an arrow primitive spanning ``p0`` → ``p1`` (``None`` when degenerate)."""
    start = np.asarray(p0, dtype=np.float64).reshape(3)
    end = np.asarray(p1, dtype=np.float64).reshape(3)
    delta = end - start
    length = float(np.linalg.norm(delta))
    if length < min_length:
        return None
    return DebugPrimitive(
        kind="arrow",
        pos=tuple(float(v) for v in start),
        quat=quat_from_z_axis(delta),
        size=(length,),
        rgba=tuple(float(v) for v in rgba),
    )


def append_debug_primitives_to_scene(
    scene,
    primitives: Sequence[DebugPrimitive],
    *,
    mesh_ids: Mapping[str, int] | None = None,
) -> int:
    """Inject debug primitives into a caller-owned mjvScene (``viewer.user_scn``).

    ``mesh_ids`` resolves ``ghost_geom`` mesh assets to mesh ids in the loaded
    viewer model; a ghost primitive without a resolvable mesh raises
    ``ValueError``.  ``text`` primitives are a documented no-op (mjvScene has
    no text channel), matching the offline renderer.  Returns the number of
    geoms appended; injection stops at ``scene.maxgeom``.
    """
    import mujoco

    added = 0
    for primitive in primitives:
        pos = np.array(primitive.pos, dtype=np.float64)
        if primitive.quat is not None:
            quat = np.array(primitive.quat, dtype=np.float64)
            quat /= np.linalg.norm(quat)
            mat = np.empty(9, dtype=np.float64)
            mujoco.mju_quat2Mat(mat, quat)
        else:
            mat = np.eye(3, dtype=np.float64).flatten()
        rgba = np.array(primitive.rgba, dtype=np.float64)

        kind = primitive.kind
        if kind in ("sphere", "box"):
            if scene.ngeom >= scene.maxgeom:
                break
            size = (
                np.array([primitive.size[0], 0.0, 0.0])
                if kind == "sphere"
                else np.array(primitive.size)
            )
            geom_type = (
                mujoco.mjtGeom.mjGEOM_SPHERE if kind == "sphere" else mujoco.mjtGeom.mjGEOM_BOX
            )
            mujoco.mjv_initGeom(
                scene.geoms[scene.ngeom],
                type=geom_type,
                size=size,
                pos=pos,
                mat=mat,
                rgba=rgba,
            )
            scene.ngeom += 1
            added += 1
        elif kind == "ghost_geom":
            assert primitive.mesh_asset is not None
            mesh_id = (mesh_ids or {}).get(primitive.mesh_asset, -1)
            if mesh_id < 0:
                raise ValueError(
                    f"ghost_geom mesh asset {primitive.mesh_asset!r} is not resolvable in the "
                    "interactive viewer; pass mesh_ids mapping the asset to a mesh id in the "
                    "loaded viewer model"
                )
            if scene.ngeom >= scene.maxgeom:
                break
            scale = primitive.size[0] if primitive.size else 1.0
            geom = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(
                geom,
                type=mujoco.mjtGeom.mjGEOM_MESH,
                size=np.array([scale, scale, scale]),
                pos=pos,
                mat=mat,
                rgba=rgba,
            )
            geom.dataid = mesh_id
            scene.ngeom += 1
            added += 1
        elif kind in ("frame", "arrow"):
            length = float(primitive.size[0])
            width = max(1e-3, 0.02 * length)
            rotation = np.asarray(mat, dtype=np.float64).reshape(3, 3)
            axes: list[tuple[np.ndarray, tuple[float, float, float, float]]]
            if kind == "arrow":
                axes = [
                    (
                        rotation[:, 2],
                        (float(rgba[0]), float(rgba[1]), float(rgba[2]), float(rgba[3])),
                    )
                ]
            else:
                alpha = float(rgba[3])
                axes = [
                    (rotation[:, axis], (*color, alpha))
                    for axis, color in enumerate(
                        ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
                    )
                ]
            for direction, color in axes:
                if scene.ngeom >= scene.maxgeom:
                    return added
                geom = scene.geoms[scene.ngeom]
                mujoco.mjv_connector(
                    geom,
                    mujoco.mjtGeom.mjGEOM_ARROW,
                    width,
                    pos,
                    pos + direction * length,
                )
                geom.rgba[:] = color
                scene.ngeom += 1
                added += 1
        elif kind == "text":
            continue
        else:  # pragma: no cover - DebugPrimitive validates kinds
            raise ValueError(f"unsupported debug primitive kind {kind!r}")
    return added


__all__ = [
    "append_debug_primitives_to_scene",
    "quat_from_z_axis",
    "segment_arrow",
]
