"""Cold-path MjSpec export.

Slim port of mjlab/utils/spec.py — keeps only the option-field diff helper and
the `export_spec` writer used by :mod:`unilab.scene.composer`. The actuator and
mesh helpers from mjlab are intentionally omitted; unilab's robot XMLs already
declare actuators directly.
"""

from __future__ import annotations

import shutil
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import mujoco
import numpy as np

from unilab.scene.spec_xml import fix_spec_xml, strip_buffer_textures

_DEFAULT_SPEC_OPTION = mujoco.MjSpec().option

_OPTION_FIELDS = (
    "ccd_iterations",
    "ccd_tolerance",
    "cone",
    "density",
    "disableactuator",
    "disableflags",
    "enableflags",
    "gravity",
    "impratio",
    "integrator",
    "iterations",
    "jacobian",
    "ls_iterations",
    "ls_tolerance",
    "magnetic",
    "noslip_iterations",
    "noslip_tolerance",
    "o_friction",
    "o_margin",
    "o_solimp",
    "o_solref",
    "sdf_initpoints",
    "sdf_iterations",
    "sleep_tolerance",
    "solver",
    "timestep",
    "tolerance",
    "viscosity",
    "wind",
)


def non_default_option_fields(opt: mujoco._specs.MjOption) -> list[str]:
    """Return option field names that differ from MjSpec defaults."""
    diffs = []
    for name in _OPTION_FIELDS:
        default = getattr(_DEFAULT_SPEC_OPTION, name)
        value = getattr(opt, name)
        if isinstance(default, np.ndarray):
            if not np.array_equal(default, value):
                diffs.append(name)
        elif default != value:
            diffs.append(name)
    return diffs


def export_spec(
    spec: mujoco.MjSpec,
    output_dir: Path,
    *,
    zip: bool = False,
) -> None:
    """Write a spec's XML and referenced mesh/texture assets to a directory.

    Creates ``scene.xml`` and an ``assets/`` subdirectory containing only the
    assets referenced by the generated XML. When *zip* is True the directory is
    compressed into a ``.zip`` archive and removed.

    Operates on a copy of *spec* to avoid mutation.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp = spec.copy()
    strip_buffer_textures(tmp)
    xml = fix_spec_xml(tmp.to_xml(), meshdir="assets")
    (output_dir / "scene.xml").write_text(xml)

    # Collect file paths referenced in the XML.
    root = ET.fromstring(xml)
    referenced: set[str] = set()
    for elem in root.iter():
        file_val = elem.get("file")
        if file_val:
            referenced.add(file_val)

    # Match asset keys to XML file attributes by path suffix because keys may
    # carry the original meshdir prefix (e.g. ``../../meshes/robot/arm.stl`` for
    # a file attribute of ``robot/arm.stl``).
    assets_dir = output_dir / "assets"
    for ref_path in sorted(referenced):
        for key, data in tmp.assets.items():
            norm = key.replace("\\", "/")
            if norm == ref_path or norm.endswith("/" + ref_path):
                out = assets_dir / ref_path
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(data)
                break

    if zip:
        zip_path = output_dir.with_suffix(".zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for file in sorted(output_dir.rglob("*")):
                if file.is_file():
                    zf.write(file, file.relative_to(output_dir))
        shutil.rmtree(output_dir)
