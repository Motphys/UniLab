"""Export a materialized MuJoCo scene to XML + mesh assets."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any


def export_scene(
    model_file: str,
    output_dir: str,
    *,
    as_zip: bool = False,
) -> str:
    import mujoco as _mujoco

    mujoco: Any = _mujoco

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    spec = mujoco.MjSpec.from_file(model_file)

    xml_content = spec.to_xml()
    xml_path = output_path / "scene.xml"
    xml_path.write_text(xml_content)

    source_dir = Path(model_file).parent
    meshdir = spec.meshdir if spec.meshdir else ""
    mesh_source = source_dir / meshdir if meshdir else source_dir

    assets_dst = output_path / "assets"
    if mesh_source.is_dir():
        for ext in ("*.stl", "*.obj", "*.msh", "*.ply"):
            for f in mesh_source.glob(ext):
                assets_dst.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, assets_dst / f.name)

    if as_zip:
        zip_path = shutil.make_archive(str(output_path), "zip", str(output_path))
        return zip_path

    return str(output_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="unilab-export-scene",
        description="Export a MuJoCo model to XML + mesh assets directory.",
    )
    parser.add_argument("model_file", help="Path to the MuJoCo XML or MJB model file")
    parser.add_argument(
        "-o", "--output-dir", default="exported_scene", help="Output directory"
    )
    parser.add_argument("--zip", action="store_true", help="Also create a zip archive")
    args = parser.parse_args(argv)

    result = export_scene(args.model_file, args.output_dir, as_zip=args.zip)
    print(f"Scene exported to: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
