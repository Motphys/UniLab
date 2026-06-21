"""Generate Drake-compatible Go1 MJCF assets.

Drake's MJCF parser accepts OBJ meshes but rejects the STL visual meshes used
by the upstream Go1 MuJoCo asset. This script converts the visual STL meshes to
OBJ and writes Drake-specific XML copies that reference those OBJ files.
"""

from __future__ import annotations

import argparse
import math
import struct
from pathlib import Path

MESH_FILES = (
    "trunk.stl",
    "hip.stl",
    "thigh_mirror.stl",
    "calf.stl",
    "thigh.stl",
)


def _is_binary_stl(path: Path) -> bool:
    size = path.stat().st_size
    if size < 84:
        return False
    with path.open("rb") as f:
        f.seek(80)
        tri_count = struct.unpack("<I", f.read(4))[0]
    return size == 84 + tri_count * 50


def _read_binary_stl(path: Path) -> list[tuple[tuple[float, float, float], ...]]:
    faces: list[tuple[tuple[float, float, float], ...]] = []
    with path.open("rb") as f:
        f.seek(80)
        tri_count = struct.unpack("<I", f.read(4))[0]
        for _ in range(tri_count):
            raw = f.read(50)
            if len(raw) != 50:
                raise ValueError(f"Unexpected EOF while reading {path}")
            values = struct.unpack("<12fH", raw)
            vertices = (
                (values[3], values[4], values[5]),
                (values[6], values[7], values[8]),
                (values[9], values[10], values[11]),
            )
            faces.append(vertices)
    return faces


def _read_ascii_stl(path: Path) -> list[tuple[tuple[float, float, float], ...]]:
    vertices: list[tuple[float, float, float]] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 4 and parts[0].lower() == "vertex":
                vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
    if len(vertices) % 3 != 0:
        raise ValueError(f"ASCII STL vertex count is not divisible by 3: {path}")
    return [(vertices[i], vertices[i + 1], vertices[i + 2]) for i in range(0, len(vertices), 3)]


def _read_stl(path: Path) -> list[tuple[tuple[float, float, float], ...]]:
    return _read_binary_stl(path) if _is_binary_stl(path) else _read_ascii_stl(path)


def _face_normal(face: tuple[tuple[float, float, float], ...]) -> tuple[float, float, float]:
    ax, ay, az = face[0]
    bx, by, bz = face[1]
    cx, cy, cz = face[2]
    ux, uy, uz = bx - ax, by - ay, bz - az
    vx, vy, vz = cx - ax, cy - ay, cz - az
    nx = uy * vz - uz * vy
    ny = uz * vx - ux * vz
    nz = ux * vy - uy * vx
    norm = math.sqrt(nx * nx + ny * ny + nz * nz)
    if norm <= 1.0e-12:
        return (0.0, 0.0, 1.0)
    return (nx / norm, ny / norm, nz / norm)


def stl_to_obj(stl_path: Path, obj_path: Path) -> None:
    faces = _read_stl(stl_path)
    vertex_ids: dict[tuple[float, float, float], int] = {}
    face_ids: list[tuple[int, int, int]] = []
    normals: list[tuple[float, float, float]] = []

    for face in faces:
        ids: list[int] = []
        for vertex in face:
            vertex_id = vertex_ids.get(vertex)
            if vertex_id is None:
                vertex_id = len(vertex_ids) + 1
                vertex_ids[vertex] = vertex_id
            ids.append(vertex_id)
        face_ids.append((ids[0], ids[1], ids[2]))
        normals.append(_face_normal(face))

    vertices = [None] * len(vertex_ids)
    for vertex, vertex_id in vertex_ids.items():
        vertices[vertex_id - 1] = vertex

    obj_path.parent.mkdir(parents=True, exist_ok=True)
    with obj_path.open("w", encoding="utf-8") as f:
        f.write(f"# Generated from {stl_path.name} by prepare_go1_drake_assets.py\n")
        for vertex in vertices:
            assert vertex is not None
            f.write(f"v {vertex[0]:.9g} {vertex[1]:.9g} {vertex[2]:.9g}\n")
        for normal in normals:
            f.write(f"vn {normal[0]:.9g} {normal[1]:.9g} {normal[2]:.9g}\n")
        for normal_id, face in enumerate(face_ids, start=1):
            f.write(f"f {face[0]}//{normal_id} {face[1]}//{normal_id} {face[2]}//{normal_id}\n")


def _rewrite_go1_xml(source: Path, target: Path) -> None:
    text = source.read_text(encoding="utf-8")
    text = text.replace('meshdir="assets"', 'meshdir="assets_drake"')
    for mesh_file in MESH_FILES:
        text = text.replace(f'file="{mesh_file}"', f'file="{Path(mesh_file).stem}.obj"')
    target.write_text(text, encoding="utf-8")


def _rewrite_scene_xml(source: Path, target: Path) -> None:
    text = source.read_text(encoding="utf-8")
    text = text.replace('<include file="go1.xml"/>', '<include file="go1_drake.xml"/>')
    target.write_text(text, encoding="utf-8")


def prepare_assets(go1_dir: Path) -> None:
    assets_dir = go1_dir / "assets"
    drake_assets_dir = go1_dir / "assets_drake"
    for mesh_file in MESH_FILES:
        stl_path = assets_dir / mesh_file
        obj_path = drake_assets_dir / f"{Path(mesh_file).stem}.obj"
        if not stl_path.is_file():
            raise FileNotFoundError(stl_path)
        stl_to_obj(stl_path, obj_path)
        print(f"wrote {obj_path.relative_to(go1_dir)}")

    _rewrite_go1_xml(go1_dir / "go1.xml", go1_dir / "go1_drake.xml")
    print("wrote go1_drake.xml")
    _rewrite_scene_xml(go1_dir / "scene_flat.xml", go1_dir / "scene_flat_drake.xml")
    print("wrote scene_flat_drake.xml")


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    default_go1_dir = repo_root / "src" / "unilab" / "assets" / "robots" / "go1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--go1-dir", type=Path, default=default_go1_dir)
    args = parser.parse_args()
    prepare_assets(args.go1_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
