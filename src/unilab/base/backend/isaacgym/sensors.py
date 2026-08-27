"""Cold-path MJCF scene metadata scan for the IsaacGym subprocess backend.

IsaacGym has no MuJoCo sensor concept, so the host backend rebuilds the sensor
contract from the tensor API.  This module parses the scene MJCF once during
``materialize()`` (the only lifecycle phase allowed to touch asset files) and
extracts:

- sensor declarations mapped to quantities computable from the shm tensor
  caches (see the kind table below), and
- ``<keyframe>`` qpos snapshots (MuJoCo ``wxyz`` convention, returned as-is).

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


def _scan_one_file(path: Path, metadata: dict) -> None:
    root = ET.parse(path).getroot()

    def walk_body(body: ET.Element) -> None:
        body_name = body.get("name", "")
        for child in body:
            if child.tag == "site" and child.get("name"):
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
                metadata["joint_names"].append(child.get("name"))
            elif child.tag == "body":
                walk_body(child)

    for worldbody in root.iter("worldbody"):
        for body in worldbody:
            if body.tag == "body":
                walk_body(body)

    for sensor_block in root.iter("sensor"):
        for element in sensor_block:
            name = element.get("name")
            if not name:
                continue
            metadata["sensors"].append((path, element.tag, name, dict(element.attrib)))

    for keyframe in root.iter("key"):
        name = keyframe.get("name")
        qpos = keyframe.get("qpos")
        if not name or qpos is None:
            continue
        metadata["keyframes"][name] = np.asarray(
            [float(value) for value in qpos.split()], dtype=np.float32
        )


def _resolve_frame_target(
    sensor_name: str,
    tag: str,
    attrib: dict[str, str],
    site_frames: dict[str, SiteFrame],
    site_attrs: dict[str, dict[str, str]],
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
        f"(objname {objname!r}); only body and site frames map to the IsaacGym "
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
        frame = _resolve_site_sensor(name, tag, attrib, site_frames, site_attrs)
        if isinstance(frame, UnsupportedSensorSpec):
            return frame
        return from_frame(KIND_GYRO, frame)
    if tag == "velocimeter":
        frame = _resolve_site_sensor(name, tag, attrib, site_frames, site_attrs)
        if isinstance(frame, UnsupportedSensorSpec):
            return frame
        return from_frame(KIND_LOCAL_LINVEL, frame)
    if tag in ("framequat", "framepos", "framezaxis"):
        frame = _resolve_frame_target(name, tag, attrib, site_frames, site_attrs)
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
                "to the IsaacGym net-contact-force tensor"
            )
        geom = attrib.get("geom2") or attrib.get("geom1")
        if geom is None or geom not in geom_body:
            return unsupported(f"contact sensor {name!r} references unknown geom {geom!r}")
        return SceneSensorSpec(name=name, kind=KIND_CONTACT_FOUND, body_name=geom_body[geom])
    return unsupported(f"sensor {name!r} uses unsupported MJCF sensor type {tag!r}")


def scan_scene_metadata(model_file: str) -> SceneMetadata:
    """Scan one MJCF scene (with includes) for sensors and keyframes.

    Cold path only: this reads and parses asset XML and must never run on
    step/reset hot paths.
    """
    path = Path(model_file).expanduser()
    if not path.is_file():
        raise ValueError(f"isaacgym scene model file does not exist: {path}")
    raw: dict = {
        "site_frames": {},
        "site_attrs": {},
        "geom_body": {},
        "sensors": [],
        "keyframes": {},
        "joint_names": [],
    }
    for scene_file in _iter_scene_files(path):
        _scan_one_file(scene_file, raw)

    sensors: dict[str, SceneSensorSpec] = {}
    unsupported: dict[str, UnsupportedSensorSpec] = {}
    for source_file, tag, name, attrib in raw["sensors"]:
        resolved = _resolve_sensor(
            tag, name, attrib, raw["site_frames"], raw["site_attrs"], raw["geom_body"]
        )
        if isinstance(resolved, SceneSensorSpec):
            sensors[name] = resolved
        else:
            if not resolved.reason.endswith(f"(file: {source_file})"):
                resolved = UnsupportedSensorSpec(
                    name=resolved.name, reason=f"{resolved.reason} (file: {source_file})"
                )
            unsupported[name] = resolved

    return SceneMetadata(
        model_file=str(path),
        sensors=sensors,
        unsupported_sensors=unsupported,
        keyframes=raw["keyframes"],
        joint_names=tuple(str(name) for name in raw["joint_names"]),
    )


__all__ = [
    "KIND_CONTACT_FOUND",
    "KIND_FRAMEPOS",
    "KIND_FRAMEQUAT",
    "KIND_FRAMEZAXIS",
    "KIND_GYRO",
    "KIND_LOCAL_LINVEL",
    "SUPPORTED_KINDS",
    "SceneMetadata",
    "SceneSensorSpec",
    "SiteFrame",
    "UnsupportedSensorSpec",
    "scan_scene_metadata",
]
