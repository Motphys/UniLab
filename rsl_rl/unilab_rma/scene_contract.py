"""Cold-path XML discovery; never guess missing scene/home/geometry metadata."""

import xml.etree.ElementTree as ET
from pathlib import Path


def expanded_xml(path, stack=()):
    path = Path(path).resolve()
    if path in stack:
        raise ValueError(f"Cyclic XML include: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Restore original UniLab XML/mesh assets first: {path}")
    root = ET.parse(path).getroot()

    def expand(parent):
        for child in list(parent):
            if child.tag == "include":
                included = expanded_xml(path.parent / child.attrib["file"], (*stack, path))
                index = list(parent).index(child)
                parent.remove(child)
                for offset, element in enumerate(list(included)):
                    parent.insert(index + offset, element)
            else:
                expand(child)

    expand(root)
    return root


def prepare_scene(override, root_dir):
    scene = override["scene"]
    path = Path(scene["model_file"])
    if not path.is_absolute():
        path = Path(root_dir) / path
    tree = expanded_xml(path)
    base = next((b for b in tree.iter("body") if b.get("name") == "base"), None)
    if base is None:
        raise ValueError("Expected the Go2 base body in restored scene")
    names = [g.attrib["name"] for g in base.iter("geom") if g.get("name")]
    if not names or len(names) != len(set(names)):
        raise ValueError(
            "RMA requires unique named robot geoms; inspect/name collision geoms in XML"
        )
    scene["model_file"] = str(path.resolve())
    scene["entities"]["robot"]["geom_names"] = names
    print(f"RMA friction geom selection from XML: {names}", flush=True)
