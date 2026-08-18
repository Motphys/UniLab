from __future__ import annotations

import tempfile
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Literal, overload

import numpy as np

from unilab.base.scene import resolve_scene_fragment_path
from unilab.terrains.terrain_generator import TerrainGeneratorCfg

if TYPE_CHECKING:
    from motrixsim import SceneModel
    from motrixsim.msd import Link, World

#: Prefix for the per-geom contact-force sensors added by
#: :func:`add_motrix_contact_force_sensors`.
CONTACT_FORCE_SENSOR_PREFIX = "contact_force_"

_MAX_INCLUDE_DEPTH = 8
_CONTACT_SENSOR_MAX_NUM = 8


def contact_force_sensor_name(geom_name: str) -> str:
    """Return the contact-force sensor name that :func:`add_motrix_contact_force_sensors` uses."""
    return f"{CONTACT_FORCE_SENSOR_PREFIX}{geom_name}"


def _iter_xml_roots(model_file: str | Path) -> list[ET.Element]:
    """Return the XML roots of ``model_file`` and everything it ``<include>``s."""
    roots: list[ET.Element] = []
    visited: set[Path] = set()

    def walk(path: Path, depth: int = 0) -> None:
        if depth > _MAX_INCLUDE_DEPTH or path in visited or not path.is_file():
            return
        visited.add(path)
        try:
            root = ET.parse(path).getroot()
        except ET.ParseError:
            return
        roots.append(root)
        for include in root.findall("include"):
            included = include.get("file")
            if included:
                walk((path.parent / included).resolve(), depth + 1)

    walk(Path(model_file).resolve())
    return roots


def collect_body_collision_geoms(
    model_file: str | Path, body_names: Sequence[str]
) -> dict[str, tuple[str, ...]]:
    """Collect the named colliding geoms of each requested body from a scene XML.

    Every colliding geom must be included for a body's summed contact force to equal its
    true ground reaction force: a G1 foot carries 4 sphere geoms plus 7 capsule geoms, and
    summing only the spheres recovers ~20% of body weight (ledger §9.8).

    Geoms are treated as visual-only, and skipped, when they declare both ``contype="0"``
    and ``conaffinity="0"``, or belong to a class named ``visual``. Unnamed geoms are
    skipped: MotrixSim's ``geom_pair`` matcher needs a name.

    A body present in the XML but carrying no colliding geom maps to an empty tuple,
    distinguishing "exists, cannot collide" (force is zero) from "not in this model"
    (absent from the result, a caller error). MuJoCo reads exactly ``0.0`` for these
    bodies too, so the two backends agree on them (ledger §9.11).

    Args:
        model_file: Scene XML; ``<include>`` directives are followed.
        body_names: Bodies to collect geoms for.

    Returns:
        Body name to its colliding geom names, for the requested bodies present in the
        model; the tuple is empty when the body has no colliding geom.
    """
    wanted = set(body_names)
    found: dict[str, tuple[str, ...]] = {}

    def visit(element: ET.Element) -> None:
        for body in element.findall("body"):
            name = body.get("name")
            if name in wanted and name not in found:
                geoms: list[str] = []
                for geom in body.findall("geom"):
                    geom_name = geom.get("name")
                    if not geom_name:
                        continue
                    if geom.get("class") == "visual":
                        continue
                    if geom.get("contype") == "0" and geom.get("conaffinity") == "0":
                        continue
                    geoms.append(geom_name)
                # Recorded even when empty: the body exists but cannot collide, so its
                # contact force is a well-defined zero rather than a lookup failure.
                found[name] = tuple(geoms)
            visit(body)

    for root in _iter_xml_roots(model_file):
        for worldbody in root.findall("worldbody"):
            visit(worldbody)
    return found


def parse_actuator_force_ranges(
    model_file: str | Path, actuator_names: Sequence[str]
) -> np.ndarray | None:
    """Parse ``forcerange`` for each actuator from a MuJoCo-format scene XML.

    MotrixSim enforces ``forcerange`` internally but does not expose it, so the effort
    limits an env needs (e.g. an ``effort/kp`` action rescale) have to come from the XML.
    Cold path only: called once at backend init.

    Args:
        model_file: Scene XML; ``<include>`` directives are followed.
        actuator_names: Actuator names in model order.

    Returns:
        ``(len(actuator_names), 2)`` array of ``[low, high]``, or ``None`` when any
        actuator has no declared ``forcerange``.
    """
    found: dict[str, tuple[float, float]] = {}
    for root in _iter_xml_roots(model_file):
        for actuator in root.findall(".//actuator/*"):
            name = actuator.get("name")
            force_range = actuator.get("forcerange")
            if not name or not force_range:
                continue
            parts = force_range.split()
            if len(parts) != 2:
                continue
            try:
                found[name] = (float(parts[0]), float(parts[1]))
            except ValueError:
                continue

    if any(name not in found for name in actuator_names):
        return None
    return np.array([found[name] for name in actuator_names], dtype=np.float64)


def add_motrix_contact_force_sensors(
    world: World,
    *,
    body_geoms: Mapping[str, Sequence[str]],
    ground_geom: str = "floor",
) -> dict[str, tuple[str, ...]]:
    """Add per-geom net contact-force sensors, grouped per body.

    Stands in for MuJoCo's ``mjSENS_TOUCH`` injection, which has no MotrixSim analogue
    (``msd.TouchSensor`` over a site reads a constant zero). Each geom gets a
    ``ContactSensor`` matching it against ``ground_geom`` with ``NetForce`` reduction and
    a force report, so summing a body's magnitudes yields that body's total ground
    reaction force in newtons — the same quantity BFM-Zero thresholds.

    Every colliding geom of a body must be listed: a G1 foot carries 4 sphere geoms plus
    7 capsule geoms, and summing only the spheres recovers just ~20% of body weight
    (ledger §9.8).

    Args:
        world: MSD world under construction.
        body_geoms: Body name to the names of its colliding geoms.
        ground_geom: Geom the bodies are tested against.

    Returns:
        Body name to the created sensor names, in the order given.
    """
    import motrixsim.msd as msd

    existing = {sensor.name for sensor in world.sensors.contact if sensor.name}
    registry: dict[str, tuple[str, ...]] = {}
    for body_name, geom_names in body_geoms.items():
        created: list[str] = []
        for geom_name in geom_names:
            sensor_name = f"{CONTACT_FORCE_SENSOR_PREFIX}{geom_name}"
            created.append(sensor_name)
            if sensor_name in existing:
                continue
            report = msd.ContactSensorReport()
            report.found = False
            report.force = True
            sensor = msd.ContactSensor()
            sensor.name = sensor_name
            sensor.match_ = msd.ContactMatch.geom_pair(ground_geom, geom_name)
            sensor.reduce = msd.ContactSensorReduce.NetForce
            sensor.report = report
            sensor.max_num = _CONTACT_SENSOR_MAX_NUM
            world.sensors.contact.append(sensor)
            existing.add(sensor_name)
        registry[body_name] = tuple(created)
    return registry


def _extract_keyframes(fragment_file: Path) -> list[ET.Element]:
    """Return ``<keyframe>`` child elements declared inside ``fragment_file``."""
    root = ET.parse(fragment_file).getroot()
    return list(root.findall("keyframe"))


def _materialize_robot_with_fragment_keyframes(
    robot_path: Path, fragment_paths: Sequence[Path]
) -> Path:
    """Inject fragment ``<keyframe>`` blocks into a temporary copy of ``robot_path``.

    motrix's ``msd.from_file`` validates ``<keyframe>`` qpos against the loaded
    model. fragment XMLs only carry sensors/contacts (no body), so a fragment
    with its own keyframe fails to parse on its own. Mujoco backend already
    merges fragments into the scene XML before parsing; this helper does the
    equivalent for the keyframe block so motrix can load a robot model that
    owns the keyframe declared in a sibling fragment.

    Returns the original ``robot_path`` when no fragment has a keyframe.
    """
    fragment_keyframes: list[ET.Element] = []
    for fragment_path in fragment_paths:
        fragment_keyframes.extend(_extract_keyframes(fragment_path))
    if not fragment_keyframes:
        return robot_path

    tree = ET.parse(robot_path)
    root = tree.getroot()
    existing = root.find("keyframe")
    if existing is None:
        existing = ET.SubElement(root, "keyframe")
    for keyframe in fragment_keyframes:
        existing.extend(list(keyframe))

    tmp = tempfile.NamedTemporaryFile(
        suffix=f"_{robot_path.name}",
        dir=str(robot_path.parent),
        mode="w",
        delete=False,
    )
    tmp.close()
    tree.write(tmp.name)
    return Path(tmp.name)


def _materialize_fragment_without_keyframes(fragment_file: Path) -> Path:
    """Strip ``<keyframe>`` from a fragment XML; return original if no change."""
    tree = ET.parse(fragment_file)
    root = tree.getroot()
    keyframes = root.findall("keyframe")
    if not keyframes:
        return fragment_file
    for keyframe in keyframes:
        root.remove(keyframe)
    tmp = tempfile.NamedTemporaryFile(
        suffix=f"_{fragment_file.name}",
        dir=str(fragment_file.parent),
        mode="w",
        delete=False,
    )
    tmp.close()
    tree.write(tmp.name)
    return Path(tmp.name)


def _cleanup_temp_xml(path: Path, original: Path) -> None:
    if path == original:
        return
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def _attach_motrix_scene_fragment(world: World, fragment_file: Path) -> None:
    import motrixsim.msd as msd

    sanitized = _materialize_fragment_without_keyframes(fragment_file)
    try:
        fragment = msd.from_file(str(sanitized))
    finally:
        _cleanup_temp_xml(sanitized, fragment_file)
    world.attach(fragment)


def _iter_motrix_links(link: Link):
    yield link
    for child in link.children:
        yield from _iter_motrix_links(child)


def _motrix_world_link_names(world: World) -> list[str]:
    names: list[str] = []
    for body in world.hierarchy.bodies:
        for link in _iter_motrix_links(body.link):
            if link.name:
                names.append(link.name)
    return names


def add_motrix_tracking_frame_sensors(world: World, *, base_name: str) -> None:
    """Add Motrix-native frame sensors matching the legacy tracking sensor contract."""
    import motrixsim.msd as msd

    link_names = _motrix_world_link_names(world)
    if base_name not in link_names:
        raise ValueError(f"Base link '{base_name}' not found in Motrix scene")

    existing = {sensor.name for sensor in world.sensors.frame if sensor.name}
    sensor_specs = (
        ("track_pos_b", msd.FrameSensorType.FramePos),
        ("track_quat_b", msd.FrameSensorType.FrameQuat),
        ("track_linvel_b", msd.FrameSensorType.FrameLinVel),
        ("track_angvel_b", msd.FrameSensorType.FrameAngVel),
    )
    ref_frame = msd.FrameSensorRef.object(msd.ObjectType.link(base_name))
    for link_name in link_names:
        object_type = msd.ObjectType.link(link_name)
        for prefix, sensor_type in sensor_specs:
            sensor_name = f"{prefix}_{link_name}"
            if sensor_name in existing:
                continue
            sensor = msd.FrameSensor()
            sensor.name = sensor_name
            sensor.sensor_type = sensor_type
            sensor.object_type = object_type
            sensor.ref_frame = ref_frame
            world.sensors.frame.append(sensor)


def materialize_motrix_scene(
    *,
    model_file: str,
    fragment_files: Sequence[str] = (),
    add_body_sensors: bool = False,
    base_name: str = "base",
    contact_force_geoms: Mapping[str, Sequence[str]] | None = None,
    ground_geom: str = "floor",
) -> SceneModel:
    """Build a MotrixSim model through MSD scene composition.

    Args:
        model_file: Scene XML entry point.
        fragment_files: Extra XML fragments to attach.
        add_body_sensors: Add the per-link tracking frame sensors.
        base_name: Base link, used as the tracking sensors' reference frame.
        contact_force_geoms: Body name to its colliding geom names. When given, a
            net-contact-force sensor is added per geom; read them back with
            :func:`contact_force_sensor_name`.
        ground_geom: Geom the contact-force sensors test against.

    Returns:
        The built scene model.
    """
    import motrixsim.msd as msd

    model_path = Path(model_file).resolve()
    fragment_paths = [
        resolve_scene_fragment_path(fragment_file, model_path) for fragment_file in fragment_files
    ]
    robot_path = _materialize_robot_with_fragment_keyframes(model_path, fragment_paths)
    try:
        world = msd.from_file(str(robot_path))
        for fragment_path in fragment_paths:
            _attach_motrix_scene_fragment(world, fragment_path)
        if add_body_sensors:
            add_motrix_tracking_frame_sensors(world, base_name=base_name)
        if contact_force_geoms:
            add_motrix_contact_force_sensors(
                world, body_geoms=contact_force_geoms, ground_geom=ground_geom
            )
        return msd.build(world)
    finally:
        _cleanup_temp_xml(robot_path, model_path)


@overload
def materialize_motrix_hfield_attached_scene(
    *,
    model_file: str,
    terrain_cfg: TerrainGeneratorCfg,
    fragment_files: Sequence[str] = (),
    hfield_name: str = "terrain_hfield",
    geom_name: str = "floor",
    add_body_sensors: bool = False,
    base_name: str = "base",
    return_surface_sampler: Literal[False] = False,
) -> tuple[SceneModel, np.ndarray]: ...


@overload
def materialize_motrix_hfield_attached_scene(
    *,
    model_file: str,
    terrain_cfg: TerrainGeneratorCfg,
    fragment_files: Sequence[str] = (),
    hfield_name: str = "terrain_hfield",
    geom_name: str = "floor",
    add_body_sensors: bool = False,
    base_name: str = "base",
    return_surface_sampler: Literal[True],
) -> tuple[SceneModel, np.ndarray, object]: ...


def materialize_motrix_hfield_attached_scene(
    *,
    model_file: str,
    terrain_cfg: TerrainGeneratorCfg,
    fragment_files: Sequence[str] = (),
    hfield_name: str = "terrain_hfield",
    geom_name: str = "floor",
    add_body_sensors: bool = False,
    base_name: str = "base",
    return_surface_sampler: bool = False,
) -> tuple[SceneModel, np.ndarray] | tuple[SceneModel, np.ndarray, object]:
    """Build a MotrixSim model with generated hfield terrain and attached robot."""
    import motrixsim.msd as msd

    from unilab.terrains import TerrainGenerator

    robot_path = Path(model_file).resolve()
    generated = TerrainGenerator(terrain_cfg).generate()

    world = msd.World()
    world.name = "unilab materialized hfield scene"

    hfield = msd.HFieldSource()
    hfield.nrow = int(generated.heights_yx.shape[0])
    hfield.ncol = int(generated.heights_yx.shape[1])
    # MotrixSim's hfield source uses MuJoCo-style X/Y half extents.
    hfield.size = [float(generated.hfield_size[0]), float(generated.hfield_size[1])]
    hfield.height_scale = float(generated.height_extent)
    # MotrixSim buffers use compiled hfield row order: row 0 is the -Y side.
    hfield_data = np.ascontiguousarray(np.flipud(generated.heights_yx).astype(np.float32))
    hfield.source_type = msd.HFieldSourceType.buffer(
        hfield_data.reshape(-1),
        f"{hfield_name}_buffer",
    )
    world.assets.hfields[hfield_name] = hfield

    terrain_geom = msd.Geometry()
    terrain_geom.name = geom_name
    terrain_geom.shape = msd.ShapeType.HField
    terrain_geom.hfield = hfield_name
    terrain_geom.position = np.asarray(generated.geom_pos, dtype=np.float32)
    terrain_geom.collision_mask = msd.CollisionMask.collide_with_all()
    terrain_geom.physics_material.friction = [1.0, 0.005, 0.0001]
    world.hierarchy.geoms.append(terrain_geom)

    fragment_paths = [
        resolve_scene_fragment_path(fragment_file, robot_path) for fragment_file in fragment_files
    ]
    merged_robot_path = _materialize_robot_with_fragment_keyframes(robot_path, fragment_paths)
    try:
        robot_world = msd.from_file(str(merged_robot_path))
        world.attach(robot_world)
        # TODO(motrixsim): remove this once msd.World.attach carries keyframes.
        world.keyframes.extend(robot_world.keyframes)
    finally:
        _cleanup_temp_xml(merged_robot_path, robot_path)

    for fragment_path in fragment_paths:
        _attach_motrix_scene_fragment(world, fragment_path)
    if add_body_sensors:
        add_motrix_tracking_frame_sensors(world, base_name=base_name)

    if return_surface_sampler:
        return msd.build(world), generated.terrain_origins, generated.surface_sampler()
    return msd.build(world), generated.terrain_origins
