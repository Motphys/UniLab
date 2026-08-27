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
``gyro``            ``gyro``              body ang-vel rotated into body frame
``velocimeter``     ``local_linvel``      body lin-vel rotated into body frame
``framequat``       ``framequat``         body quat (wxyz)
``framepos``        ``framepos``          body world position
``contact``         ``contact_found``     1.0 when the body's net contact force
(data=found)                              norm is positive, else 0.0
==================  ====================  ===================================

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
KIND_CONTACT_FOUND = "contact_found"

SUPPORTED_KINDS = (
    KIND_GYRO,
    KIND_LOCAL_LINVEL,
    KIND_FRAMEQUAT,
    KIND_FRAMEPOS,
    KIND_CONTACT_FOUND,
)

_KIND_DIMS = {
    KIND_GYRO: 3,
    KIND_LOCAL_LINVEL: 3,
    KIND_FRAMEQUAT: 4,
    KIND_FRAMEPOS: 3,
    KIND_CONTACT_FOUND: 1,
}


@dataclass(frozen=True)
class SceneSensorSpec:
    """One MJCF sensor declaration resolved to its host-side quantity."""

    name: str
    kind: str
    body_name: str

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


def _scan_one_file(path: Path, metadata: dict) -> None:
    root = ET.parse(path).getroot()

    def walk_body(body: ET.Element) -> None:
        body_name = body.get("name", "")
        for child in body:
            if child.tag == "site" and child.get("name"):
                metadata["site_body"][child.get("name")] = body_name
            elif child.tag == "geom" and child.get("name"):
                metadata["geom_body"][child.get("name")] = body_name
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


def _resolve_sensor(
    tag: str,
    name: str,
    attrib: dict[str, str],
    site_body: dict[str, str],
    geom_body: dict[str, str],
) -> SceneSensorSpec | UnsupportedSensorSpec:
    """Map one MJCF sensor element onto a tensor-API-computable quantity."""

    def unsupported(reason: str) -> UnsupportedSensorSpec:
        return UnsupportedSensorSpec(name=name, reason=reason)

    if tag == "gyro":
        site = attrib.get("site")
        if site is None or site not in site_body:
            return unsupported(f"gyro sensor {name!r} references unknown site {site!r}")
        return SceneSensorSpec(name=name, kind=KIND_GYRO, body_name=site_body[site])
    if tag == "velocimeter":
        site = attrib.get("site")
        if site is None or site not in site_body:
            return unsupported(f"velocimeter sensor {name!r} references unknown site {site!r}")
        return SceneSensorSpec(name=name, kind=KIND_LOCAL_LINVEL, body_name=site_body[site])
    if tag in ("framequat", "framepos"):
        objtype = attrib.get("objtype", "body")
        objname = attrib.get("objname")
        if objtype == "body":
            body_name = objname
        elif objtype == "site" and objname in site_body:
            # Site frames are approximated by their owning body frame; the
            # IsaacGym tensor API exposes rigid-body poses only.
            body_name = site_body[objname]
        else:
            return unsupported(
                f"{tag} sensor {name!r} uses unsupported objtype {objtype!r} (objname {objname!r})"
            )
        if not body_name:
            return unsupported(f"{tag} sensor {name!r} has no resolvable target body")
        kind = KIND_FRAMEQUAT if tag == "framequat" else KIND_FRAMEPOS
        return SceneSensorSpec(name=name, kind=kind, body_name=body_name)
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
    raw: dict = {"site_body": {}, "geom_body": {}, "sensors": [], "keyframes": {}}
    for scene_file in _iter_scene_files(path):
        _scan_one_file(scene_file, raw)

    sensors: dict[str, SceneSensorSpec] = {}
    unsupported: dict[str, UnsupportedSensorSpec] = {}
    for source_file, tag, name, attrib in raw["sensors"]:
        resolved = _resolve_sensor(tag, name, attrib, raw["site_body"], raw["geom_body"])
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
    )


__all__ = [
    "KIND_CONTACT_FOUND",
    "KIND_FRAMEPOS",
    "KIND_FRAMEQUAT",
    "KIND_GYRO",
    "KIND_LOCAL_LINVEL",
    "SUPPORTED_KINDS",
    "SceneMetadata",
    "SceneSensorSpec",
    "UnsupportedSensorSpec",
    "scan_scene_metadata",
]
