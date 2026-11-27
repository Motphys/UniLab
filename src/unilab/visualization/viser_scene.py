# pyright: reportMissingImports=false
"""MuJoCo-to-viser scene adapter for interactive web-based 3D visualization.

This module renders MuJoCo scenes via a viser web server, providing browser-based
interactive 3D viewing without requiring a local display or GLFW.  viser is a
required UniLab dependency; this module is only imported on playback paths,
which also require the optional ``mujoco`` package for the playback shell.

Usage (from ``scripts/play_viser.py``)::

    from unilab.visualization.viser_scene import MujocoViserScene
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import mujoco
import numpy as np
import trimesh
import viser
from unisim.visualization.mujoco_geom import geom_mesh

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


# MuJoCo's default scene option (mjvOption.geomgroup) enables geom groups
# 0-2 and hides groups 3-5; match that default so auxiliary geoms such as
# collision-only markers stay hidden.
DEFAULT_VISIBLE_GEOM_GROUPS: tuple[int, ...] = (0, 1, 2)


def build_visible_env_indices(num_envs: int, visible_envs: int) -> np.ndarray:
    """Select a stable subset of env indices spread across the full batch.

    Args:
        num_envs: Total number of runtime environments.
        visible_envs: Number of env slots exposed in the viewer.

    Returns:
        A monotonically increasing array of runtime env indices.
    """
    if visible_envs <= 0:
        raise ValueError(f"visible_envs must be positive, got {visible_envs}")
    if visible_envs >= num_envs:
        return np.arange(num_envs, dtype=np.int32)
    return np.floor(np.linspace(0, num_envs, visible_envs, endpoint=False)).astype(np.int32)


# --------------------------------------------------------------------------- #
# MujocoViserScene                                                           #
# --------------------------------------------------------------------------- #


class MujocoViserScene:
    """Bridges a ``mujoco.MjModel`` to a ``viser.ViserServer`` scene graph.

    Call :meth:`build` once to populate the scene with geometry handles, then
    call :meth:`update` each frame to sync body transforms from ``MjData``.
    """

    def __init__(
        self,
        server: Any,
        model: mujoco.MjModel,
        *,
        name_prefix: str = "/mujoco",
        position_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
        render_plane: bool = True,
        visible_geom_groups: Sequence[int] = DEFAULT_VISIBLE_GEOM_GROUPS,
    ) -> None:
        self._server: viser.ViserServer = server
        self._model = model
        self._name_prefix = name_prefix.rstrip("/") or "/mujoco"
        self._position_offset = np.asarray(position_offset, dtype=np.float64)
        self._render_plane = bool(render_plane)
        self._visible_geom_groups = frozenset(int(g) for g in visible_geom_groups)
        self._handles: dict[int, Any] = {}
        self._build()

    def reset(
        self,
        model: mujoco.MjModel,
        *,
        position_offset: tuple[float, float, float] | None = None,
        render_plane: bool | None = None,
    ) -> None:
        """Rebuild the viser scene for a new MuJoCo model.

        Args:
            model: MuJoCo model whose geoms should populate the scene.
            position_offset: Optional XYZ offset applied to all geoms.
            render_plane: Optional override for whether plane geoms should be built.

        Returns:
            None.
        """
        self.close()
        self._model = model
        if position_offset is not None:
            self._position_offset = np.asarray(position_offset, dtype=np.float64)
        if render_plane is not None:
            self._render_plane = bool(render_plane)
        self._build()

    def close(self) -> None:
        """Remove all scene handles owned by this adapter."""
        for handle in self._handles.values():
            handle.remove()
        self._handles.clear()

    # ------------------------------------------------------------------ #
    # Scene construction                                                  #
    # ------------------------------------------------------------------ #

    def _build(self) -> None:
        """Create viser scene nodes for every geom in a visible geom group."""
        model = self._model
        server = self._server

        server.scene.set_up_direction("+z")

        for i in range(model.ngeom):
            if int(model.geom_group[i]) not in self._visible_geom_groups:
                continue
            geom_type = model.geom_type[i]
            size = model.geom_size[i]
            rgba = model.geom_rgba[i]
            color = _rgba_to_color(rgba)
            opacity = _rgba_to_opacity(rgba)
            name = f"{self._name_prefix}/geom/{i}"

            handle: Any | None = None

            if geom_type == mujoco.mjtGeom.mjGEOM_PLANE:
                if not self._render_plane:
                    continue
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
                mesh = geom_mesh(model, i)
                if mesh is not None:
                    vertices, faces = mesh
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
                xpos = data.geom_xpos[i] + self._position_offset
                xmat = data.geom_xmat[i]

                handle.position = (float(xpos[0]), float(xpos[1]), float(xpos[2]))
                handle.wxyz = _rotmat_to_wxyz(xmat)


class MujocoViserBatchScene:
    """Render matching MuJoCo models as batched Viser mesh instances.

    One Viser batched mesh is created for each geom index.  This keeps the
    geometry immutable while updating all visible environment transforms in a
    single array assignment per geom, instead of creating one scene node per
    environment and sending two messages per geom instance.
    """

    def __init__(
        self,
        server: Any,
        models: Sequence[mujoco.MjModel],
        *,
        name_prefix: str = "/mujoco/batch",
        position_offsets: np.ndarray | Sequence[Sequence[float]] | None = None,
        render_plane: bool = True,
        visible_geom_groups: Sequence[int] = DEFAULT_VISIBLE_GEOM_GROUPS,
    ) -> None:
        if not models:
            raise ValueError("MujocoViserBatchScene requires at least one model")
        first = models[0]
        if any(model.ngeom != first.ngeom for model in models[1:]):
            raise ValueError("Batched MuJoCo models must have the same number of geoms")
        for model in models[1:]:
            if not (
                np.array_equal(model.geom_type, first.geom_type)
                and np.array_equal(model.geom_size, first.geom_size)
                and np.array_equal(model.geom_dataid, first.geom_dataid)
                and np.array_equal(model.geom_rgba, first.geom_rgba)
                and np.array_equal(model.geom_group, first.geom_group)
            ):
                raise ValueError("Batched MuJoCo models must have matching geom geometry")

        self._server: viser.ViserServer = server
        self._models = tuple(models)
        self._name_prefix = name_prefix.rstrip("/") or "/mujoco/batch"
        self._position_offsets = np.zeros((len(models), 3), dtype=np.float64)
        if position_offsets is not None:
            offsets = np.asarray(position_offsets, dtype=np.float64)
            if offsets.shape != self._position_offsets.shape:
                raise ValueError(
                    f"position_offsets must have shape {self._position_offsets.shape}, got {offsets.shape}"
                )
            self._position_offsets[:] = offsets
        self._render_plane = bool(render_plane)
        self._visible_geom_groups = frozenset(int(g) for g in visible_geom_groups)
        self._handles: dict[int, Any] = {}
        self._grid_handles: list[Any] = []
        self._build()

    def _build(self) -> None:
        model = self._models[0]
        self._server.scene.set_up_direction("+z")
        for geom_index in range(model.ngeom):
            if int(model.geom_group[geom_index]) not in self._visible_geom_groups:
                continue
            geom_type = model.geom_type[geom_index]
            size = model.geom_size[geom_index]
            rgba = model.geom_rgba[geom_index]
            name = f"{self._name_prefix}/geom/{geom_index}"
            if geom_type == mujoco.mjtGeom.mjGEOM_PLANE:
                if self._render_plane and not self._grid_handles:
                    plane_size = float(size[0]) if size[0] > 0 else 10.0
                    grid = self._server.scene.add_grid(
                        f"{self._name_prefix}/ground",
                        width=plane_size * 2,
                        height=plane_size * 2,
                        cell_size=0.5,
                    )
                    grid.position = tuple(self._position_offsets[0])
                    self._grid_handles.append(grid)
                continue
            mesh = geom_mesh(model, geom_index)
            if mesh is None:
                continue
            vertices, faces = mesh
            count = len(self._models)
            positions = np.zeros((count, 3), dtype=np.float32)
            orientations = np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (count, 1))
            handle = self._server.scene.add_batched_meshes_simple(
                name,
                vertices=vertices.astype(np.float32),
                faces=faces.astype(np.int32),
                batched_wxyzs=orientations,
                batched_positions=positions,
                batched_colors=_rgba_to_color(rgba),
                opacity=_rgba_to_opacity(rgba),
            )
            self._handles[geom_index] = handle

    def update(self, data: Sequence[mujoco.MjData]) -> None:
        """Update all visible environment instances in one atomic batch."""
        if len(data) != len(self._models):
            raise ValueError(f"Expected {len(self._models)} MuJoCo data objects, got {len(data)}")
        with self._server.atomic():
            for geom_index, handle in self._handles.items():
                positions = np.empty((len(data), 3), dtype=np.float32)
                orientations = np.empty((len(data), 4), dtype=np.float32)
                for env_index, env_data in enumerate(data):
                    positions[env_index] = (
                        env_data.geom_xpos[geom_index] + self._position_offsets[env_index]
                    )
                    orientations[env_index] = _rotmat_to_wxyz(env_data.geom_xmat[geom_index])
                handle.batched_positions = positions
                handle.batched_wxyzs = orientations

    def close(self) -> None:
        for handle in (*self._handles.values(), *self._grid_handles):
            handle.remove()
        self._handles.clear()
        self._grid_handles.clear()
