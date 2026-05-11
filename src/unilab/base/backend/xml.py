from __future__ import annotations

import os
import shutil
import tempfile
import xml.etree.ElementTree as ET
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, Literal, cast, overload

import numpy as np

from unilab.terrains.terrain_generator import TerrainGeneratorCfg


def _enable_discardvisual(root: ET.Element) -> None:
    compiler_tag = root.find("compiler")
    if compiler_tag is None:
        compiler_tag = ET.Element("compiler")
        root.insert(0, compiler_tag)
    compiler_tag.set("discardvisual", "true")


def _write_xml_root(root: ET.Element, output_path: Path) -> None:
    ET.indent(root, space="  ")
    output_path.write_text(ET.tostring(root, encoding="unicode"), encoding="utf-8")


def create_discardvisual_xml(model_file: str) -> str:
    tree = ET.parse(model_file)
    _enable_discardvisual(tree.getroot())
    return _write_temp_xml(tree, model_file)


def _iter_expanded_children(
    parent: ET.Element, base_dir: Path
) -> Iterator[tuple[ET.Element, Path]]:
    for child in parent:
        if child.tag != "include":
            yield child, base_dir
            continue

        include_file = child.get("file")
        if not include_file:
            raise ValueError(f"Invalid <include> without file attribute in {base_dir}")
        include_path = (base_dir / include_file).resolve()
        include_root = ET.parse(include_path).getroot()
        yield from _iter_expanded_children(include_root, include_path.parent)


def _iter_named_bodies(root: ET.Element, base_dir: Path) -> Iterator[str]:
    for child, child_base_dir in _iter_expanded_children(root, base_dir):
        if child.tag == "body":
            body_name = child.get("name")
            if body_name:
                yield body_name
        yield from _iter_named_bodies(child, child_base_dir)


def _get_named_bodies(model_file: str) -> tuple[list[int], list[str]]:
    model_path = Path(model_file).resolve()
    names = list(_iter_named_bodies(ET.parse(model_path).getroot(), model_path.parent))
    ids = list(range(1, len(names) + 1))
    return ids, names


def get_named_body_ids(model_file: str, names: Sequence[str]) -> list[int]:
    """Resolve MuJoCo-style body ids from XML without importing mujoco."""
    body_ids, body_names = _get_named_bodies(model_file)
    body_id_by_name = dict(zip(body_names, body_ids, strict=True))
    missing = [name for name in names if name not in body_id_by_name]
    if missing:
        missing_str = ", ".join(missing)
        raise ValueError(f"Bodies not found in XML '{model_file}': {missing_str}")
    return [body_id_by_name[name] for name in names]


def _mujoco_module() -> Any:
    import mujoco

    return cast(Any, mujoco)


def _materialize_spec_xml(spec, model_file: str) -> str:
    fd, output_path = tempfile.mkstemp(
        suffix=".xml", dir=os.path.dirname(os.path.abspath(model_file))
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(spec.to_xml())
    except Exception:
        os.close(fd)
        raise
    return output_path


def _add_w_sensors(spec, valid_bnames: list[str]) -> None:
    mujoco = _mujoco_module()
    for bname in valid_bnames:
        spec.add_sensor(
            name=f"track_pos_w_{bname}",
            type=mujoco.mjtSensor.mjSENS_FRAMEPOS,
            objtype=mujoco.mjtObj.mjOBJ_XBODY,
            objname=bname,
        )
    for bname in valid_bnames:
        spec.add_sensor(
            name=f"track_quat_w_{bname}",
            type=mujoco.mjtSensor.mjSENS_FRAMEQUAT,
            objtype=mujoco.mjtObj.mjOBJ_XBODY,
            objname=bname,
        )
    for bname in valid_bnames:
        spec.add_sensor(
            name=f"track_linvel_w_{bname}",
            type=mujoco.mjtSensor.mjSENS_FRAMELINVEL,
            objtype=mujoco.mjtObj.mjOBJ_XBODY,
            objname=bname,
        )
    for bname in valid_bnames:
        spec.add_sensor(
            name=f"track_angvel_w_{bname}",
            type=mujoco.mjtSensor.mjSENS_FRAMEANGVEL,
            objtype=mujoco.mjtObj.mjOBJ_XBODY,
            objname=bname,
        )


def _add_b_sensors(spec, valid_bnames: list[str], baselink_name: str) -> None:
    mujoco = _mujoco_module()
    for bname in valid_bnames:
        spec.add_sensor(
            name=f"track_pos_b_{bname}",
            type=mujoco.mjtSensor.mjSENS_FRAMEPOS,
            objtype=mujoco.mjtObj.mjOBJ_XBODY,
            objname=bname,
            reftype=mujoco.mjtObj.mjOBJ_XBODY,
            refname=baselink_name,
        )
    for bname in valid_bnames:
        spec.add_sensor(
            name=f"track_quat_b_{bname}",
            type=mujoco.mjtSensor.mjSENS_FRAMEQUAT,
            objtype=mujoco.mjtObj.mjOBJ_XBODY,
            objname=bname,
            reftype=mujoco.mjtObj.mjOBJ_XBODY,
            refname=baselink_name,
        )
    for bname in valid_bnames:
        spec.add_sensor(
            name=f"track_linvel_b_{bname}",
            type=mujoco.mjtSensor.mjSENS_FRAMELINVEL,
            objtype=mujoco.mjtObj.mjOBJ_XBODY,
            objname=bname,
            reftype=mujoco.mjtObj.mjOBJ_XBODY,
            refname=baselink_name,
        )
    for bname in valid_bnames:
        spec.add_sensor(
            name=f"track_angvel_b_{bname}",
            type=mujoco.mjtSensor.mjSENS_FRAMEANGVEL,
            objtype=mujoco.mjtObj.mjOBJ_XBODY,
            objname=bname,
            reftype=mujoco.mjtObj.mjOBJ_XBODY,
            refname=baselink_name,
        )


def _write_temp_xml(tree: ET.ElementTree[ET.Element], model_file: str) -> str:  # type: ignore[type-arg]
    fd, output_path = tempfile.mkstemp(
        suffix=".xml", dir=os.path.dirname(os.path.abspath(model_file))
    )
    os.close(fd)
    tree.write(output_path)
    return output_path


def _format_values(values: list[float] | tuple[float, ...]) -> str:
    return " ".join(str(float(value)) for value in values)


def materialize_scene_visual_override(
    source_model_file: str,
    *,
    ground_texture_file: str | None = None,
    ground_texrepeat: list[float] | tuple[float, float] | None = None,
    skybox_rgb1: list[float] | tuple[float, float, float] | None = None,
    skybox_rgb2: list[float] | tuple[float, float, float] | None = None,
) -> str:
    """Create a temporary scene XML with visual-only overrides applied."""
    tree = ET.parse(source_model_file)
    root = tree.getroot()
    asset_tag = root.find("asset")
    if asset_tag is None:
        raise ValueError(f"Scene '{source_model_file}' is missing an <asset> tag.")

    if skybox_rgb1 is not None or skybox_rgb2 is not None:
        skybox = asset_tag.find("./texture[@type='skybox']")
        if skybox is None:
            raise ValueError(f"Scene '{source_model_file}' is missing a skybox texture.")
        if skybox_rgb1 is not None:
            skybox.set("rgb1", _format_values(tuple(skybox_rgb1)))
        if skybox_rgb2 is not None:
            skybox.set("rgb2", _format_values(tuple(skybox_rgb2)))

    if ground_texture_file is not None:
        ground_texture = asset_tag.find("./texture[@name='groundplane']")
        if ground_texture is None:
            raise ValueError(f"Scene '{source_model_file}' is missing the groundplane texture.")
        for attr in ("builtin", "mark", "rgb1", "rgb2", "markrgb", "width", "height"):
            ground_texture.attrib.pop(attr, None)
        ground_texture.set("file", str(Path(ground_texture_file)))

    if ground_texrepeat is not None:
        ground_material = asset_tag.find("./material[@name='groundplane']")
        if ground_material is None:
            raise ValueError(f"Scene '{source_model_file}' is missing the groundplane material.")
        ground_material.set("texrepeat", _format_values(tuple(ground_texrepeat)))

    return _write_temp_xml(tree, source_model_file)


def materialize_scene_fragments(
    source_model_file: str,
    *,
    fragment_files: Sequence[str],
) -> str:
    """Create a temporary scene XML with task/scene fragments merged."""
    tree = ET.parse(source_model_file)
    root = tree.getroot()
    source_path = Path(source_model_file).resolve()
    for fragment_file in fragment_files:
        _merge_scene_fragment(root, _resolve_scene_fragment_path(fragment_file, source_path))
    return _write_temp_xml(tree, source_model_file)


_ATTACH_PREFIXED_ATTRS = {
    "class",
    "childclass",
    "name",
    "material",
    "texture",
    "mesh",
    "joint",
    "site",
    "geom1",
    "geom2",
    "body1",
    "body2",
    "objname",
    "refname",
    "hfield",
    "hfieldname",
    "actuator",
}


def _strip_attach_prefixes(root: ET.Element) -> None:
    for elem in root.iter():
        for attr, value in list(elem.attrib.items()):
            if attr in _ATTACH_PREFIXED_ATTRS and value.startswith("/"):
                elem.set(attr, value[1:])


def _flatten_attach_main_default(root: ET.Element) -> None:
    default = root.find("default")
    if default is None:
        return
    main = default.find("./default[@class='main']")
    if main is None:
        return
    insert_at = list(default).index(main)
    default.remove(main)
    for child in list(main):
        default.insert(insert_at, child)
        insert_at += 1


def _ensure_child(parent: ET.Element, query: str, xml: str) -> None:
    if parent.find(query) is None:
        parent.append(ET.fromstring(xml))


def _ensure_generated_hfield_scene_visuals(root: ET.Element, geom_name: str) -> None:
    asset = root.find("asset")
    if asset is None:
        asset = ET.Element("asset")
        root.insert(0, asset)

    _ensure_child(
        asset,
        "./texture[@type='skybox']",
        '<texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" '
        'rgb2="0 0 0" width="512" height="3072"/>',
    )
    _ensure_child(
        asset,
        "./texture[@name='groundplane']",
        '<texture type="2d" name="groundplane" builtin="checker" mark="edge" '
        'rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8" '
        'width="300" height="300"/>',
    )
    _ensure_child(
        asset,
        "./material[@name='groundplane']",
        '<material name="groundplane" texture="groundplane" texuniform="true" '
        'texrepeat="5 5" reflectance="0.2"/>',
    )

    if root.find("visual") is None:
        root.append(
            ET.fromstring(
                '<visual><headlight diffuse="0.8 0.8 0.8" ambient="0.45 0.45 0.45" '
                'specular="0.1 0.1 0.1"/><rgba haze="0.15 0.25 0.35 1"/>'
                '<global azimuth="-130" elevation="-20"/><quality offsamples="4"/>'
                '<map force="0.01"/></visual>'
            )
        )

    terrain_geom = root.find(f".//geom[@name='{geom_name}']")
    if terrain_geom is not None and terrain_geom.get("material") is None:
        terrain_geom.set("material", "groundplane")
    for light in root.findall(".//light"):
        light.set("directional", "true")
        light.set("castshadow", "false")


def _merge_scene_fragment(root: ET.Element, fragment_file: Path) -> None:
    fragment_root = ET.parse(fragment_file).getroot()
    if fragment_root.tag != "mujoco":
        raise ValueError(f"Scene fragment '{fragment_file}' must have a <mujoco> root.")

    for child in list(fragment_root):
        if child.tag in {"sensor", "keyframe", "actuator"}:
            existing = root.find(child.tag)
            if existing is None:
                root.append(child)
            else:
                existing.extend(list(child))
            continue
        root.append(child)


def _resolve_scene_fragment_path(fragment_file: str, model_file: Path) -> Path:
    path = Path(fragment_file)
    if path.is_absolute():
        return path
    if path.is_file():
        return path.resolve()
    return (model_file.parent / path).resolve()


def _copy_robot_asset_dir(model_file: Path, output_dir: Path) -> None:
    assets_src = model_file.parent / "assets"
    if assets_src.is_dir():
        shutil.copytree(assets_src, output_dir / "assets", dirs_exist_ok=True)


def _collect_mujoco_assets(asset_dir: Path) -> dict[str, bytes]:
    assets: dict[str, bytes] = {}
    if not asset_dir.is_dir():
        return assets
    for path in asset_dir.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(asset_dir)
        assets[str(rel)] = path.read_bytes()
        assets[str(asset_dir.name / rel)] = path.read_bytes()
    return assets


@overload
def materialize_mujoco_hfield_attached_scene(
    *,
    model_file: str,
    terrain_cfg: TerrainGeneratorCfg,
    output_dir: str | Path,
    fragment_files: Sequence[str] = (),
    hfield_name: str = "terrain_hfield",
    geom_name: str = "floor",
    return_surface_sampler: Literal[False] = False,
) -> tuple[Any, np.ndarray]: ...


@overload
def materialize_mujoco_hfield_attached_scene(
    *,
    model_file: str,
    terrain_cfg: TerrainGeneratorCfg,
    output_dir: str | Path,
    fragment_files: Sequence[str] = (),
    hfield_name: str = "terrain_hfield",
    geom_name: str = "floor",
    return_surface_sampler: Literal[True],
) -> tuple[Any, np.ndarray, Any]: ...


def materialize_mujoco_hfield_attached_scene(
    *,
    model_file: str,
    terrain_cfg: TerrainGeneratorCfg,
    output_dir: str | Path,
    fragment_files: Sequence[str] = (),
    hfield_name: str = "terrain_hfield",
    geom_name: str = "floor",
    return_surface_sampler: bool = False,
) -> tuple[Any, np.ndarray] | tuple[Any, np.ndarray, Any]:
    """Build a MuJoCo model with generated hfield terrain and attached robot spec."""
    import mujoco

    from unilab.terrains import TerrainGenerator

    robot_path = Path(model_file).resolve()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    _copy_robot_asset_dir(robot_path, output_path)

    hfield_rel = Path("hfields") / "hfield.png"
    generated = TerrainGenerator(terrain_cfg).write_png(output_path / hfield_rel)

    spec = mujoco.MjSpec()
    spec.compiler.autolimits = True
    spec.compiler.meshdir = "assets"

    spec.add_hfield(
        name=hfield_name,
        file=str((output_path / hfield_rel).resolve()),
        size=list(generated.hfield_size),
    )
    spec.worldbody.add_light(pos=[0.0, 0.0, 8.0], dir=[0.0, 0.0, -1.0])
    spec.worldbody.add_geom(
        name=geom_name,
        type=mujoco.mjtGeom.mjGEOM_HFIELD,
        hfieldname=hfield_name,
        pos=list(generated.geom_pos),
    )

    robot_spec = mujoco.MjSpec.from_file(str(robot_path))
    frame = spec.worldbody.add_frame()
    spec.attach(robot_spec, frame=frame)

    root = ET.fromstring(spec.to_xml())
    _strip_attach_prefixes(root)
    _flatten_attach_main_default(root)
    _ensure_generated_hfield_scene_visuals(root, geom_name)
    for fragment_file in fragment_files:
        _merge_scene_fragment(root, _resolve_scene_fragment_path(fragment_file, robot_path))

    scene_xml = output_path / "scene.xml"
    _write_xml_root(root, scene_xml)

    physics_root = ET.fromstring(ET.tostring(root, encoding="unicode"))
    _enable_discardvisual(physics_root)
    model = mujoco.MjSpec.from_string(
        ET.tostring(physics_root, encoding="unicode"),
        assets=_collect_mujoco_assets(output_path / "assets"),
    ).compile()
    if return_surface_sampler:
        return model, generated.terrain_origins, generated.surface_sampler()
    return model, generated.terrain_origins


def inject_mujoco_tracking_sensors(
    model_file: str,
    baselink_name: str | None = None,
) -> tuple[str, list, list]:
    """为 MuJoCo 后端注入 tracking sensors。

    注入所有 body 的世界系 (_w) sensors；若指定 baselink_name，
    同时注入相对 baselink 坐标系的 (_b) sensors。

    Returns:
        (tmp_xml_path, tracked_body_ids, valid_bnames)
    """
    mujoco = _mujoco_module()
    tracked_body_ids, valid_bnames = _get_named_bodies(model_file)

    spec = mujoco.MjSpec.from_file(model_file)
    _add_w_sensors(spec, valid_bnames)
    if baselink_name and baselink_name in valid_bnames:
        _add_b_sensors(spec, valid_bnames, baselink_name)

    return _materialize_spec_xml(spec, model_file), tracked_body_ids, valid_bnames


def processed_xml(xml_path):
    xml_dir = os.path.dirname(os.path.abspath(xml_path))

    tree = ET.parse(xml_path)
    root = tree.getroot()

    compiler = root.find("compiler")
    if compiler is not None:
        meshdir = compiler.get("meshdir")
        if meshdir:
            abs_meshdir = os.path.normpath(os.path.join(xml_dir, meshdir))
            compiler.set("meshdir", abs_meshdir)

    bodys = root.findall(".//body")

    geom_names = []
    for body in bodys:
        body_name = body.get("name", "unnamed_body")
        geoms = body.findall("geom")

        if geoms:
            filtered_geoms = []
            for geom in geoms:
                geom_class = geom.get("class")
                if geom_class != "visual":
                    filtered_geoms.append(geom)

            if filtered_geoms:
                i = 0
                for geom in filtered_geoms:
                    geom_name = geom.get("name", "unnamed_geom")
                    if geom_name == "unnamed_geom":
                        new_name = f"{body_name}_geom{i}"
                        i += 1
                        geom.set("name", new_name)
                        geom_name = new_name
                    geom_names.append(geom_name)

    new_xml_string = ET.tostring(root, encoding="unicode")
    return new_xml_string, geom_names


def add_sensor(root, sensor_type, name, **kwargs):
    """
    在 MuJoCo XML 的 sensor 节点下添加传感器的通用函数。

    参数:
    - root: XML 的根节点
    - sensor_type: 传感器标签名 (如 'gyro', 'contact', 'framepos')
    - name: 传感器的 name 属性
    - **kwargs: 其他任意属性 (如 site='imu', geom1='floor' 等)
    """
    # 1. 查找或创建 <sensor> 标签
    sensor_element = root.find("sensor")
    if sensor_element is None:
        sensor_element = ET.SubElement(root, "sensor")

    # 2. 创建具体的传感器子节点
    sensor = ET.SubElement(sensor_element, sensor_type)

    # 3. 设置必选的 name 属性
    sensor.set("name", name)

    # 4. 循环设置其他传入的属性
    for key, value in kwargs.items():
        sensor.set(key, str(value))

    return sensor
