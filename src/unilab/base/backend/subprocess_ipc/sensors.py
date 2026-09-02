"""Cold-path MJCF scene metadata scan for MJCF subprocess backends.

The subprocess backends have no MuJoCo sensor concept, so the host backend rebuilds the sensor
contract from the tensor API.  This module parses the scene MJCF once during
``materialize()`` (the only lifecycle phase allowed to touch asset files) and
extracts:

- sensor declarations mapped to quantities computable from the shm tensor
  caches (see the kind table below),
- ``<keyframe>`` qpos snapshots (MuJoCo ``wxyz`` convention, returned as-is),
- ``<position>`` actuator parameters (kp/kv/forcerange/ctrlrange) that the
  worker needs to reproduce MuJoCo's position-actuator PD semantics, and
- per-joint dynamics (``range``/``armature``/``frictionloss``), resolved
  through MJCF default classes.

The MJCF importer is *not* trusted for actuation parameters:
empirically it reads ``kp``/``forcerange``/``armature`` but drops ``kv`` and
``frictionloss`` (and joint ranges), so every value the physics depends on is
parsed here and pushed to the worker verbatim.  Only ``<position>`` actuators
with ``gear`` unset (or 1) and symmetric ``forcerange`` are supported; any
other actuator element fails the scan closed, because ``SimBackend.step(ctrl)``
must keep a single backend-native ctrl semantics (position targets).

Supported sensor kinds and their tensor-API source:

==================  ====================  ===================================
MJCF element        kind                  source
==================  ====================  ===================================
``gyro``            ``gyro``              body ang-vel in the site frame
``velocimeter``     ``local_linvel``      body lin-vel in the site frame
``framequat``       ``framequat``         body/site frame quat (wxyz)
``framepos``        ``framepos``          body/site frame world position
``framezaxis``      ``framezaxis``        body/site frame z axis in world
``contact``         ``contact_found``     1.0 when the body's net contact force
(data=found)                              norm is positive, else 0.0
==================  ====================  ===================================

Sites are rigidly attached to their owning body, so a site frame is exact:
the cold-path scan records each site's local ``pos``/``quat`` (identity by
default) and the hot path composes them with the body's shm state.  Sites
declared with ``euler``/``axisangle``/``xyaxes``/``zaxis`` orientation
attributes fail closed — only ``quat`` (wxyz) is parsed.

Anything else (force/torque sensors, accelerometers, rangefinders, contact
sensors requesting ``force``/``dist`` data, ...) is recorded as unsupported
and fails closed with an explanatory ``NotImplementedError`` on access.
Sensor ``noise``/``cutoff`` attributes are ignored: the tensor API has no
equivalent, and UniLab applies observation noise at the env layer.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

KIND_GYRO = "gyro"
KIND_LOCAL_LINVEL = "local_linvel"
KIND_FRAMEQUAT = "framequat"
KIND_FRAMEPOS = "framepos"
KIND_FRAMEZAXIS = "framezaxis"
KIND_CONTACT_FOUND = "contact_found"

SUPPORTED_KINDS = (
    KIND_GYRO,
    KIND_LOCAL_LINVEL,
    KIND_FRAMEQUAT,
    KIND_FRAMEPOS,
    KIND_FRAMEZAXIS,
    KIND_CONTACT_FOUND,
)

_KIND_DIMS = {
    KIND_GYRO: 3,
    KIND_LOCAL_LINVEL: 3,
    KIND_FRAMEQUAT: 4,
    KIND_FRAMEPOS: 3,
    KIND_FRAMEZAXIS: 3,
    KIND_CONTACT_FOUND: 1,
}

# Orientation attributes on a <site> that this backend does not parse; their
# presence fails the referencing sensor closed instead of silently assuming
# the identity rotation.
_UNSUPPORTED_SITE_ORIENTATION_ATTRS = ("euler", "axisangle", "xyaxes", "zaxis")

_IDENTITY_QUAT = (1.0, 0.0, 0.0, 0.0)
_ZERO_POS = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class SiteFrame:
    """Rigid local transform of a site inside its owning body (cold-path data)."""

    body_name: str
    local_pos: tuple[float, float, float]
    local_quat: tuple[float, float, float, float]  # wxyz


@dataclass(frozen=True)
class SceneSensorSpec:
    """One MJCF sensor declaration resolved to its host-side quantity.

    ``local_pos``/``local_quat`` express the sensor frame (a site, or the
    identity for a body) inside ``body_name``'s frame.
    """

    name: str
    kind: str
    body_name: str
    local_pos: tuple[float, float, float] = _ZERO_POS
    local_quat: tuple[float, float, float, float] = _IDENTITY_QUAT

    @property
    def dim(self) -> int:
        return _KIND_DIMS[self.kind]


@dataclass(frozen=True)
class UnsupportedSensorSpec:
    """A declared MJCF sensor this backend cannot reproduce, with the reason."""

    name: str
    reason: str


@dataclass(frozen=True)
class ActuatorSpec:
    """One MJCF ``<position>`` actuator resolved to PD gains and limits.

    ``forcerange``/``ctrlrange`` are ``None`` when unlimited (attribute absent
    or ``0 0``, matching MuJoCo's ``*limited=false`` convention).
    ``forcerange`` is guaranteed symmetric (``[-F, F]``) by the scan.
    """

    name: str
    joint_name: str
    kp: float
    kv: float
    forcerange: tuple[float, float] | None = None
    ctrlrange: tuple[float, float] | None = None


@dataclass(frozen=True)
class SceneMetadata:
    """Cold-path scan result for one MJCF scene."""

    model_file: str
    sensors: dict[str, SceneSensorSpec] = field(default_factory=dict)
    unsupported_sensors: dict[str, UnsupportedSensorSpec] = field(default_factory=dict)
    keyframes: dict[str, np.ndarray] = field(default_factory=dict)
    joint_names: tuple[str, ...] = ()
    """Single-DoF joint names in MJCF document order (the qpos[7:] layout).

    Note: ``<include>`` splicing order is approximated by file-visit order;
    scenes whose joints live in more than one file are not supported by the
    keyframe application path.
    """
    body_names: tuple[str, ...] = ()
    """Body names in MJCF document order (depth-first pre-order, worldbody excluded)."""
    freejoint_body_name: str | None = None
    """Name of the body owning the free joint (the floating root), if any."""
    actuators: tuple[ActuatorSpec, ...] = ()
    """``<position>`` actuators in document order (transmission: one joint each)."""
    joint_ranges: tuple[tuple[float, float], ...] = ()
    """Per-joint ``range`` aligned with ``joint_names``; ``(-inf, inf)`` when undeclared."""
    joint_armature: tuple[float, ...] = ()
    """Per-joint ``armature`` aligned with ``joint_names`` (defaults-class resolved)."""
    joint_frictionloss: tuple[float, ...] = ()
    """Per-joint ``frictionloss`` aligned with ``joint_names`` (defaults-class resolved)."""


def _iter_scene_files(model_file: Path) -> list[Path]:
    """Return the scene file plus transitively included MJCF files."""
    seen: set[Path] = set()
    ordered: list[Path] = []

    def visit(path: Path) -> None:
        resolved = path.resolve()
        if resolved in seen:
            return
        seen.add(resolved)
        ordered.append(resolved)
        try:
            root = ET.parse(resolved).getroot()
        except ET.ParseError as exc:
            raise ValueError(f"failed to parse MJCF scene file {resolved}: {exc}") from exc
        for include in root.iter("include"):
            include_file = include.get("file")
            if include_file:
                visit(resolved.parent / include_file)

    visit(model_file)
    return ordered


def _parse_floats(raw: str | None, count: int, *, what: str) -> tuple[float, ...] | None:
    if raw is None:
        return None
    values = tuple(float(value) for value in raw.split())
    if len(values) != count:
        raise ValueError(f"{what} must have {count} floats, got {raw!r}")
    return values


def _collect_default_classes(root: ET.Element) -> dict[str, dict[str, dict[str, str]]]:
    """Map MJCF default class names to merged per-element attribute dicts.

    MJCF default classes inherit through nesting: ``classes[cls][tag]`` holds
    the attributes an element of type ``tag`` with ``class=cls`` starts with
    (the root class ``""`` always exists, possibly empty).  Only the attributes
    needed here matter, but the merge is tag-agnostic.
    """
    classes: dict[str, dict[str, dict[str, str]]] = {}

    def walk(default_el: ET.Element, inherited: dict[str, dict[str, str]]) -> None:
        merged = {tag: dict(attrs) for tag, attrs in inherited.items()}
        for child in default_el:
            if child.tag == "default":
                continue
            merged.setdefault(child.tag, {}).update(child.attrib)
        class_name = default_el.get("class") or ""
        classes[class_name] = merged
        for child in default_el:
            if child.tag == "default":
                walk(child, merged)

    for top_level in root.findall("default"):
        walk(top_level, {})
    classes.setdefault("", {})
    return classes


def _resolved_attrs(
    classes: dict[str, dict[str, dict[str, str]]],
    class_name: str,
    tag: str,
    own_attrib: dict[str, str],
    *,
    what: str,
) -> dict[str, str]:
    """Resolve one element's attributes through its MJCF default class."""
    if class_name not in classes:
        raise ValueError(f"{what} references unknown MJCF default class {class_name!r}")
    resolved = dict(classes[class_name].get(tag, {}))
    resolved.update(own_attrib)
    return resolved


def _scan_one_file(path: Path, metadata: dict) -> None:
    root = ET.parse(path).getroot()
    classes = _collect_default_classes(root)

    def walk_body(body: ET.Element, active_class: str) -> None:
        body_name = body.get("name", "")
        # Depth-first pre-order matches MJCF body document order (body ids).
        metadata["body_names"].append(body_name)
        # ``childclass`` sets the default class for this body's subtree.
        body_class = body.get("childclass", active_class)
        for child in body:
            if child.tag == "freejoint":
                # The first body owning a free joint is the floating root.
                metadata.setdefault("freejoint_body", body_name)
            elif child.tag == "site" and child.get("name"):
                site_name = child.get("name")
                assert site_name is not None
                metadata["site_attrs"][site_name] = dict(child.attrib)
                local_pos = (
                    _parse_floats(child.get("pos"), 3, what=f"site {site_name!r} pos") or _ZERO_POS
                )
                local_quat = (
                    _parse_floats(child.get("quat"), 4, what=f"site {site_name!r} quat")
                    or _IDENTITY_QUAT
                )
                metadata["site_frames"][site_name] = SiteFrame(
                    body_name=body_name,
                    local_pos=(local_pos[0], local_pos[1], local_pos[2]),
                    local_quat=(
                        local_quat[0],
                        local_quat[1],
                        local_quat[2],
                        local_quat[3],
                    ),
                )
            elif child.tag == "geom" and child.get("name"):
                metadata["geom_body"][child.get("name")] = body_name
            elif child.tag == "joint" and child.get("name"):
                # Single-DoF joint names in MJCF document order, matching the
                # joint section of keyframe/qpos (qpos[7:]). Free joints are
                # excluded (they are the root 7/6 columns); ball joints are
                # outside the backend's 1-dof-per-joint contract.
                joint_name = child.get("name")
                assert joint_name is not None
                attrs = _resolved_attrs(
                    classes,
                    child.get("class", body_class),
                    "joint",
                    dict(child.attrib),
                    what=f"joint {joint_name!r}",
                )
                metadata["joint_names"].append(joint_name)
                joint_range = _parse_floats(
                    attrs.get("range"), 2, what=f"joint {joint_name!r} range"
                )
                metadata["joint_ranges"].append(
                    (-np.inf, np.inf) if joint_range is None else joint_range
                )
                metadata["joint_armature"].append(float(attrs.get("armature", 0.0)))
                metadata["joint_frictionloss"].append(float(attrs.get("frictionloss", 0.0)))
            elif child.tag == "body":
                walk_body(child, body_class)

    for worldbody in root.iter("worldbody"):
        for body in worldbody:
            if body.tag == "body":
                walk_body(body, "")

    for sensor_block in root.iter("sensor"):
        for element in sensor_block:
            name = element.get("name")
            if not name:
                continue
            metadata["sensors"].append((path, element.tag, name, dict(element.attrib)))

    for actuator_block in root.iter("actuator"):
        for element in actuator_block:
            metadata["actuators"].append((path, classes, element.tag, dict(element.attrib)))

    for keyframe in root.iter("key"):
        name = keyframe.get("name")
        qpos = keyframe.get("qpos")
        if not name or qpos is None:
            continue
        metadata["keyframes"][name] = np.asarray(
            [float(value) for value in qpos.split()], dtype=np.float32
        )


def _resolve_actuator(
    tag: str,
    attrib: dict[str, str],
    classes: dict[str, dict[str, dict[str, str]]],
    source_file: Path,
    known_joints: set[str],
    backend_label: str = "subprocess",
) -> ActuatorSpec:
    """Resolve one MJCF actuator element to an ``ActuatorSpec``.

    Fails closed on anything that would make ``SimBackend.step(ctrl)`` lose
    its single position-target semantics (non-position actuator types,
    non-joint transmissions, non-unit gear, asymmetric force limits).
    """
    if tag != "position":
        raise NotImplementedError(
            f"{backend_label} supports MJCF <position> actuators only; found <{tag}> "
            f"(file: {source_file}). SimBackend.step(ctrl) requires a single "
            "backend-native ctrl semantics (position targets)."
        )
    name = attrib.get("name") or attrib.get("joint") or "<unnamed>"
    attrs = _resolved_attrs(
        classes, attrib.get("class", ""), "position", attrib, what=f"actuator {name!r}"
    )
    joint_name = attrs.get("joint")
    if not joint_name:
        raise NotImplementedError(
            f"{backend_label} position actuator {name!r} has no joint transmission "
            f"(file: {source_file}); only joint transmissions are supported"
        )
    if joint_name not in known_joints:
        raise ValueError(
            f"{backend_label} position actuator {name!r} references unknown joint "
            f"{joint_name!r} (file: {source_file})"
        )
    gear = attrs.get("gear")
    if gear is not None and float(gear) != 1.0:
        raise NotImplementedError(
            f"{backend_label} position actuator {name!r} uses gear={gear}; only gear=1 is supported"
        )
    if "kp" not in attrs:
        raise ValueError(
            f"{backend_label} position actuator {name!r} has no kp (file: {source_file}); "
            "declare kp explicitly or via a default class"
        )
    forcerange_raw = _parse_floats(attrs.get("forcerange"), 2, what=f"actuator {name!r} forcerange")
    forcerange: tuple[float, float] | None = (
        None if forcerange_raw is None else (forcerange_raw[0], forcerange_raw[1])
    )
    if forcerange is not None and forcerange[0] == 0.0 and forcerange[1] == 0.0:
        # MuJoCo convention: a 0 0 forcerange is unlimited (forcelimited=false).
        forcerange = None
    if forcerange is not None and not np.isclose(forcerange[0], -forcerange[1]):
        raise NotImplementedError(
            f"{backend_label} position actuator {name!r} uses asymmetric forcerange "
            f"{forcerange}; PhysX dof effort limits are symmetric"
        )
    ctrlrange_raw = _parse_floats(attrs.get("ctrlrange"), 2, what=f"actuator {name!r} ctrlrange")
    ctrlrange: tuple[float, float] | None = (
        None if ctrlrange_raw is None else (ctrlrange_raw[0], ctrlrange_raw[1])
    )
    if ctrlrange is not None and ctrlrange[0] == 0.0 and ctrlrange[1] == 0.0:
        ctrlrange = None
    return ActuatorSpec(
        name=name,
        joint_name=joint_name,
        kp=float(attrs["kp"]),
        kv=float(attrs.get("kv", 0.0)),
        forcerange=forcerange,
        ctrlrange=ctrlrange,
    )


def _resolve_frame_target(
    sensor_name: str,
    tag: str,
    attrib: dict[str, str],
    site_frames: dict[str, SiteFrame],
    site_attrs: dict[str, dict[str, str]],
    backend_label: str = "subprocess",
) -> SiteFrame | UnsupportedSensorSpec:
    """Resolve a body/site frame target for frame* sensors."""

    def unsupported(reason: str) -> UnsupportedSensorSpec:
        return UnsupportedSensorSpec(name=sensor_name, reason=reason)

    objtype = attrib.get("objtype", "body")
    objname = attrib.get("objname")
    if objtype == "body":
        if not objname:
            return unsupported(f"{tag} sensor {sensor_name!r} has no objname")
        return SiteFrame(body_name=objname, local_pos=_ZERO_POS, local_quat=_IDENTITY_QUAT)
    if objtype == "site":
        orientation_error = _site_orientation_error(sensor_name, tag, objname, site_attrs)
        if orientation_error is not None:
            return unsupported(orientation_error)
        frame = site_frames.get(objname or "")
        if frame is None:
            return unsupported(f"{tag} sensor {sensor_name!r} references unknown site {objname!r}")
        return frame
    return unsupported(
        f"{tag} sensor {sensor_name!r} uses unsupported objtype {objtype!r} "
        f"(objname {objname!r}); only body and site frames map to the {backend_label} "
        "rigid-body state tensor"
    )


def _site_orientation_error(
    sensor_name: str,
    tag: str,
    site_name: str | None,
    site_attrs: dict[str, dict[str, str]],
) -> str | None:
    """Fail closed when the referenced site uses an unparsed orientation attr."""
    site_attr = site_attrs.get(site_name or "", {})
    bad = [key for key in _UNSUPPORTED_SITE_ORIENTATION_ATTRS if key in site_attr]
    if not bad:
        return None
    return (
        f"{tag} sensor {sensor_name!r} references site {site_name!r} whose orientation "
        f"is declared via {bad}; only the quat attribute is parsed"
    )


def _resolve_site_sensor(
    sensor_name: str,
    tag: str,
    attrib: dict[str, str],
    site_frames: dict[str, SiteFrame],
    site_attrs: dict[str, dict[str, str]],
    backend_label: str = "subprocess",
) -> SiteFrame | UnsupportedSensorSpec:
    """Resolve a site-attached sensor (gyro/velocimeter) to its site frame."""
    site = attrib.get("site")
    orientation_error = _site_orientation_error(sensor_name, tag, site, site_attrs)
    if orientation_error is not None:
        return UnsupportedSensorSpec(name=sensor_name, reason=orientation_error)
    frame = site_frames.get(site or "")
    if frame is None:
        return UnsupportedSensorSpec(
            name=sensor_name,
            reason=f"{tag} sensor {sensor_name!r} references unknown site {site!r}",
        )
    return frame


def _resolve_sensor(
    tag: str,
    name: str,
    attrib: dict[str, str],
    site_frames: dict[str, SiteFrame],
    site_attrs: dict[str, dict[str, str]],
    geom_body: dict[str, str],
    backend_label: str = "subprocess",
) -> SceneSensorSpec | UnsupportedSensorSpec:
    """Map one MJCF sensor element onto a tensor-API-computable quantity."""

    def unsupported(reason: str) -> UnsupportedSensorSpec:
        return UnsupportedSensorSpec(name=name, reason=reason)

    def from_frame(kind: str, frame: SiteFrame) -> SceneSensorSpec:
        return SceneSensorSpec(
            name=name,
            kind=kind,
            body_name=frame.body_name,
            local_pos=frame.local_pos,
            local_quat=frame.local_quat,
        )

    if tag == "gyro":
        frame = _resolve_site_sensor(name, tag, attrib, site_frames, site_attrs, backend_label)
        if isinstance(frame, UnsupportedSensorSpec):
            return frame
        return from_frame(KIND_GYRO, frame)
    if tag == "velocimeter":
        frame = _resolve_site_sensor(name, tag, attrib, site_frames, site_attrs, backend_label)
        if isinstance(frame, UnsupportedSensorSpec):
            return frame
        return from_frame(KIND_LOCAL_LINVEL, frame)
    if tag in ("framequat", "framepos", "framezaxis"):
        frame = _resolve_frame_target(name, tag, attrib, site_frames, site_attrs, backend_label)
        if isinstance(frame, UnsupportedSensorSpec):
            return frame
        kind = {
            "framequat": KIND_FRAMEQUAT,
            "framepos": KIND_FRAMEPOS,
            "framezaxis": KIND_FRAMEZAXIS,
        }[tag]
        return from_frame(kind, frame)
    if tag == "contact":
        data = (attrib.get("data") or "found").split()
        if data != ["found"]:
            return unsupported(
                f"contact sensor {name!r} requests data={data}; only data='found' maps "
                f"to the {backend_label} net-contact-force tensor"
            )
        geom = attrib.get("geom2") or attrib.get("geom1")
        if geom is None or geom not in geom_body:
            return unsupported(f"contact sensor {name!r} references unknown geom {geom!r}")
        return SceneSensorSpec(name=name, kind=KIND_CONTACT_FOUND, body_name=geom_body[geom])
    return unsupported(f"sensor {name!r} uses unsupported MJCF sensor type {tag!r}")


def scan_scene_metadata(model_file: str, *, backend_label: str = "subprocess") -> SceneMetadata:
    """Scan one MJCF scene (with includes) for sensors and keyframes.

    Cold path only: this reads and parses asset XML and must never run on
    step/reset hot paths.
    """
    path = Path(model_file).expanduser()
    if not path.is_file():
        raise ValueError(f"{backend_label} scene model file does not exist: {path}")
    raw: dict = {
        "site_frames": {},
        "site_attrs": {},
        "geom_body": {},
        "sensors": [],
        "keyframes": {},
        "joint_names": [],
        "joint_ranges": [],
        "joint_armature": [],
        "joint_frictionloss": [],
        "body_names": [],
        "actuators": [],
    }
    for scene_file in _iter_scene_files(path):
        _scan_one_file(scene_file, raw)

    sensors: dict[str, SceneSensorSpec] = {}
    unsupported: dict[str, UnsupportedSensorSpec] = {}
    for source_file, tag, name, attrib in raw["sensors"]:
        resolved = _resolve_sensor(
            tag,
            name,
            attrib,
            raw["site_frames"],
            raw["site_attrs"],
            raw["geom_body"],
            backend_label,
        )
        if isinstance(resolved, SceneSensorSpec):
            sensors[name] = resolved
        else:
            if not resolved.reason.endswith(f"(file: {source_file})"):
                resolved = UnsupportedSensorSpec(
                    name=resolved.name, reason=f"{resolved.reason} (file: {source_file})"
                )
            unsupported[name] = resolved

    known_joints = {str(name) for name in raw["joint_names"]}
    actuators: list[ActuatorSpec] = []
    actuated_joints: set[str] = set()
    for source_file, classes, tag, attrib in raw["actuators"]:
        spec = _resolve_actuator(tag, attrib, classes, source_file, known_joints, backend_label)
        if spec.joint_name in actuated_joints:
            raise ValueError(
                f"joint {spec.joint_name!r} has more than one <position> actuator "
                f"(file: {source_file}); the backend maps one actuator per dof"
            )
        actuated_joints.add(spec.joint_name)
        actuators.append(spec)

    return SceneMetadata(
        model_file=str(path),
        sensors=sensors,
        unsupported_sensors=unsupported,
        keyframes=raw["keyframes"],
        joint_names=tuple(str(name) for name in raw["joint_names"]),
        body_names=tuple(str(name) for name in raw["body_names"]),
        freejoint_body_name=raw.get("freejoint_body"),
        actuators=tuple(actuators),
        joint_ranges=tuple(raw["joint_ranges"]),
        joint_armature=tuple(raw["joint_armature"]),
        joint_frictionloss=tuple(raw["joint_frictionloss"]),
    )


__all__ = [
    "KIND_CONTACT_FOUND",
    "KIND_FRAMEPOS",
    "KIND_FRAMEQUAT",
    "KIND_FRAMEZAXIS",
    "KIND_GYRO",
    "KIND_LOCAL_LINVEL",
    "SUPPORTED_KINDS",
    "ActuatorSpec",
    "SceneMetadata",
    "SceneSensorSpec",
    "SiteFrame",
    "UnsupportedSensorSpec",
    "scan_scene_metadata",
]
