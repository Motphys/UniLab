# pyright: reportMissingImports=false
"""MuJoCo-to-viser scene adapter for interactive web-based 3D visualization.

This module renders MuJoCo scenes via a viser web server, providing browser-based
interactive 3D viewing without requiring a local display or GLFW.  It is gated
behind the ``viser`` optional-dependency group and is **not** imported by default.

Usage (from ``scripts/play_viser.py``)::

    from unilab.utils.viser_scene import MujocoViserScene, VISER_AVAILABLE
"""

from __future__ import annotations

import math
from typing import Any

import mujoco
import numpy as np

try:
    import trimesh
    import viser

    VISER_AVAILABLE = True
except ImportError:
    VISER_AVAILABLE = False


# --------------------------------------------------------------------------- #
# Rotation helpers (pure numpy, no scipy dependency)                          #
# --------------------------------------------------------------------------- #


def _rotmat_to_wxyz(mat: np.ndarray) -> tuple[float, float, float, float]:
    """Convert a 3x3 rotation matrix to a (w, x, y, z) quaternion."""
    m = np.asarray(mat, dtype=np.float64).reshape(3, 3)
    trace = m[0, 0] + m[1, 1] + m[2, 2]

    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m[2, 1] - m[1, 2]) * s
        y = (m[0, 2] - m[2, 0]) * s
        z = (m[1, 0] - m[0, 1]) * s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s

    return (float(w), float(x), float(y), float(z))


# --------------------------------------------------------------------------- #
# Geometry extraction helpers                                                 #
# --------------------------------------------------------------------------- #


def _rgba_to_color(rgba: np.ndarray) -> tuple[int, int, int]:
    """Convert MuJoCo float RGBA [0,1] to viser int RGB [0,255]."""
    return (
        int(np.clip(rgba[0] * 255, 0, 255)),
        int(np.clip(rgba[1] * 255, 0, 255)),
        int(np.clip(rgba[2] * 255, 0, 255)),
    )


def _rgba_to_opacity(rgba: np.ndarray) -> float:
    return float(np.clip(rgba[3], 0.0, 1.0))


def _extract_mesh(model: mujoco.MjModel, geom_dataid: int) -> tuple[np.ndarray, np.ndarray]:
    """Extract vertices and faces for a MuJoCo mesh geom."""
    vert_adr = model.mesh_vertadr[geom_dataid]
    vert_num = model.mesh_vertnum[geom_dataid]
    face_adr = model.mesh_faceadr[geom_dataid]
    face_num = model.mesh_facenum[geom_dataid]

    vertices = model.mesh_vert[vert_adr : vert_adr + vert_num].copy()
    faces = model.mesh_face[face_adr : face_adr + face_num].copy()
    return vertices, faces


# --------------------------------------------------------------------------- #
# MujocoViserScene                                                           #
# --------------------------------------------------------------------------- #


class MujocoViserScene:
    """Bridges a ``mujoco.MjModel`` to a ``viser.ViserServer`` scene graph.

    Call :meth:`build` once to populate the scene with geometry handles, then
    call :meth:`update` each frame to sync body transforms from ``MjData``.
    """

    def __init__(self, server: Any, model: mujoco.MjModel) -> None:
        if not VISER_AVAILABLE:
            raise ImportError("viser is not installed. Install with: uv sync --extra viser")
        self._server: viser.ViserServer = server
        self._model = model
        self._handles: dict[int, Any] = {}
        self._build()

    # ------------------------------------------------------------------ #
    # Scene construction                                                  #
    # ------------------------------------------------------------------ #

    def _build(self) -> None:
        """Create viser scene nodes for every MuJoCo geom."""
        model = self._model
        server = self._server

        server.scene.set_up_direction("+z")

        for i in range(model.ngeom):
            geom_type = model.geom_type[i]
            size = model.geom_size[i]
            rgba = model.geom_rgba[i]
            color = _rgba_to_color(rgba)
            opacity = _rgba_to_opacity(rgba)
            name = f"/mujoco/geom/{i}"

            handle = None

            if geom_type == mujoco.mjtGeom.mjGEOM_PLANE:
                # Render ground plane as a grid
                plane_size = float(size[0]) if size[0] > 0 else 10.0
                handle = server.scene.add_grid(
                    name,
                    width=plane_size * 2,
                    height=plane_size * 2,
                    cell_size=0.5,
                )

            elif geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
                handle = server.scene.add_icosphere(
                    name,
                    radius=float(size[0]),
                    color=color,
                    opacity=opacity,
                )

            elif geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE:
                half_len = float(size[1])
                radius = float(size[0])
                mesh = trimesh.creation.capsule(height=half_len * 2, radius=radius)
                handle = server.scene.add_mesh_trimesh(name, mesh=mesh)
                # Manually set color since trimesh mesh may not carry it
                if hasattr(handle, "color"):
                    handle.color = color

            elif geom_type == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
                # Use a unit sphere mesh scaled non-uniformly
                mesh = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
                mesh.vertices *= np.array([float(size[0]), float(size[1]), float(size[2])])
                handle = server.scene.add_mesh_trimesh(name, mesh=mesh)

            elif geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
                handle = server.scene.add_cylinder(
                    name,
                    radius=float(size[0]),
                    height=float(size[1]) * 2,
                    color=color,
                    opacity=opacity,
                )

            elif geom_type == mujoco.mjtGeom.mjGEOM_BOX:
                handle = server.scene.add_box(
                    name,
                    dimensions=(
                        float(size[0]) * 2,
                        float(size[1]) * 2,
                        float(size[2]) * 2,
                    ),
                    color=color,
                    opacity=opacity,
                )

            elif geom_type == mujoco.mjtGeom.mjGEOM_MESH:
                dataid = model.geom_dataid[i]
                if dataid >= 0:
                    vertices, faces = _extract_mesh(model, dataid)
                    handle = server.scene.add_mesh_simple(
                        name,
                        vertices=vertices.astype(np.float32),
                        faces=faces.astype(np.int32),
                        color=color,
                        opacity=opacity,
                    )

            if handle is not None:
                self._handles[i] = handle

    # ------------------------------------------------------------------ #
    # Per-frame update                                                    #
    # ------------------------------------------------------------------ #

    def update(self, data: mujoco.MjData) -> None:
        """Sync all geom transforms from *data* into the viser scene."""
        with self._server.atomic():
            for i, handle in self._handles.items():
                xpos = data.geom_xpos[i]
                xmat = data.geom_xmat[i]

                handle.position = (float(xpos[0]), float(xpos[1]), float(xpos[2]))
                handle.wxyz = _rotmat_to_wxyz(xmat)
