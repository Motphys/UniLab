"""Cold-path scene materializer.

Takes a base robot XML and a TerrainGeneratorCfg, runs the procedural terrain
generator, and writes a final ``scene.xml`` + ``assets/`` directory that
unilab's existing ``model_file`` path can load. All work happens once at env
init — step/reset never touches assets.

This is intentionally a thin port: it wraps mjlab's terrain generator and spec
exporter but does not bring in mjlab's Scene / Entity / Sensor abstractions.
"""

from __future__ import annotations

import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from unilab.scene.spec_export import export_spec
from unilab.terrains.terrain_generator import (
    TerrainGenerator,
    TerrainGeneratorCfg,
)

_DEFAULT_FLOOR_GEOM = "floor"
_DEFAULT_TERRAIN_BODY = "terrain"


@dataclass
class MaterializedScene:
    """Output of :func:`compose_and_materialize`.

    Attributes:
        scene_xml: Path to the materialized ``scene.xml`` (load this with
            ``mujoco.MjModel.from_xml_path``).
        terrain_origins: ``(num_rows, num_cols, 3)`` array of per-cell spawn
            origins computed by the terrain generator. Used by tasks that
            want to distribute envs across the grid (see
            :func:`unilab.terrains.utils.compute_env_origins`).
    """

    scene_xml: Path
    terrain_origins: np.ndarray


def _strip_floor_geom(spec: mujoco.MjSpec, floor_geom: str) -> bool:
    """Remove a named geom from the spec's worldbody. Returns True if removed."""
    for geom in list(spec.worldbody.geoms):
        if geom.name == floor_geom:
            spec.delete(geom)
            return True
    return False


def _retarget_contact_sensors(
    spec: mujoco.MjSpec,
    *,
    floor_geom: str,
    terrain_body: str,
) -> int:
    """Rewrite contact sensors that referenced ``floor_geom`` so they instead
    reference ``terrain_body``. MuJoCo's ``body1`` / ``body2`` on a contact
    sensor matches contacts involving any geom under that body subtree.

    Returns the number of (objname, refname) pairs rewritten.
    """
    rewritten = 0
    for sensor in spec.sensors:
        if sensor.type != mujoco.mjtSensor.mjSENS_CONTACT:
            continue
        if sensor.objtype == mujoco.mjtObj.mjOBJ_GEOM and sensor.objname == floor_geom:
            sensor.objtype = mujoco.mjtObj.mjOBJ_BODY
            sensor.objname = terrain_body
            rewritten += 1
        if sensor.reftype == mujoco.mjtObj.mjOBJ_GEOM and sensor.refname == floor_geom:
            sensor.reftype = mujoco.mjtObj.mjOBJ_BODY
            sensor.refname = terrain_body
            rewritten += 1
    return rewritten


def _copy_external_assets(base_xml: Path, output_dir: Path) -> None:
    """Copy mesh/texture files referenced by the materialized scene.xml from
    the base XML's directory into the output ``assets/`` subdirectory.

    ``MjSpec.from_file`` references mesh / texture files by path but does not
    pre-load their bytes into ``spec.assets``. ``export_spec`` only writes the
    bytes it has, so externally-referenced files must be copied separately.
    """
    scene_xml = output_dir / "scene.xml"
    if not scene_xml.is_file():
        return
    materialized_root = ET.fromstring(scene_xml.read_text())
    referenced: set[str] = set()
    for elem in materialized_root.iter():
        file_val = elem.get("file")
        if file_val:
            referenced.add(file_val)
    if not referenced:
        return

    # MjSpec.to_xml() emits a flat <compiler meshdir="..." .../> reflecting the
    # original meshdir from the base spec (or any included file). Use that
    # directory as the primary candidate; fall back to the base XML directory
    # itself for cases without an explicit meshdir.
    base_dir = base_xml.parent
    compiler = materialized_root.find("compiler")
    out_meshdir = compiler.get("meshdir", "") if compiler is not None else ""
    out_texdir = compiler.get("texturedir", out_meshdir) if compiler is not None else ""

    assets_dir = output_dir / "assets"
    for ref in sorted(referenced):
        candidates = [
            base_dir / out_meshdir / ref,
            base_dir / out_texdir / ref,
            base_dir / ref,
        ]
        src_path: Path | None = next((p for p in candidates if p.is_file()), None)
        if src_path is None:
            continue  # Either generated procedurally or genuinely missing.
        dst = assets_dir / ref
        if dst.is_file():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src_path, dst)


def compose_and_materialize(
    base_xml: Path,
    terrain_cfg: TerrainGeneratorCfg,
    output_dir: Path,
    *,
    floor_geom: str = _DEFAULT_FLOOR_GEOM,
    terrain_body: str = _DEFAULT_TERRAIN_BODY,
) -> MaterializedScene:
    """Materialize a final ``scene.xml`` + ``assets/`` from a base robot XML
    and a terrain generator cfg.

    Steps:

    1. Load *base_xml* via ``MjSpec.from_file``.
    2. Run ``TerrainGenerator(terrain_cfg).compile(spec)`` — adds a
       ``terrain`` body with all generated geoms / hfields / lights.
    3. Retarget contact sensors that referenced the original ``floor_geom``
       so they point at the new ``terrain_body`` subtree.
    4. Strip the ``floor_geom`` from worldbody.
    5. Call :func:`export_spec` to write ``scene.xml`` plus any procedurally
       generated assets.
    6. Copy externally-referenced mesh / texture files from the base XML's
       directory into the output ``assets/`` subdirectory.

    Returns a :class:`MaterializedScene` carrying the scene XML path and the
    per-cell ``terrain_origins`` array (so callers can distribute envs
    across the grid at reset time).
    """
    base_xml = Path(base_xml)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    spec = mujoco.MjSpec.from_file(str(base_xml))
    generator = TerrainGenerator(terrain_cfg)
    generator.compile(spec)
    _retarget_contact_sensors(spec, floor_geom=floor_geom, terrain_body=terrain_body)
    _strip_floor_geom(spec, floor_geom)
    export_spec(spec, output_dir)
    _copy_external_assets(base_xml, output_dir)
    return MaterializedScene(
        scene_xml=output_dir / "scene.xml",
        terrain_origins=generator.terrain_origins.copy(),
    )
