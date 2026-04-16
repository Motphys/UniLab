from __future__ import annotations

import os
import tempfile
import xml.etree.ElementTree as ET
from collections.abc import Iterator, Sequence
from pathlib import Path

import mujoco


def _enable_discardvisual(root: ET.Element) -> None:
    compiler_tag = root.find("compiler")
    if compiler_tag is None:
        compiler_tag = ET.Element("compiler")
        root.insert(0, compiler_tag)
    compiler_tag.set("discardvisual", "true")


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


def _add_w_sensors(sensor_tag: ET.Element, valid_bnames: list[str]) -> None:
    for bname in valid_bnames:
        ET.SubElement(
            sensor_tag, "framepos", name=f"track_pos_w_{bname}", objtype="xbody", objname=bname
        )
    for bname in valid_bnames:
        ET.SubElement(
            sensor_tag, "framequat", name=f"track_quat_w_{bname}", objtype="xbody", objname=bname
        )
    for bname in valid_bnames:
        ET.SubElement(
            sensor_tag,
            "framelinvel",
            name=f"track_linvel_w_{bname}",
            objtype="xbody",
            objname=bname,
        )
    for bname in valid_bnames:
        ET.SubElement(
            sensor_tag,
            "frameangvel",
            name=f"track_angvel_w_{bname}",
            objtype="xbody",
            objname=bname,
        )


def _add_b_sensors(sensor_tag: ET.Element, valid_bnames: list[str], baselink_name: str) -> None:
    for bname in valid_bnames:
        ET.SubElement(
            sensor_tag,
            "framepos",
            name=f"track_pos_b_{bname}",
            objtype="xbody",
            objname=bname,
            reftype="xbody",
            refname=baselink_name,
        )
    for bname in valid_bnames:
        ET.SubElement(
            sensor_tag,
            "framequat",
            name=f"track_quat_b_{bname}",
            objtype="xbody",
            objname=bname,
            reftype="xbody",
            refname=baselink_name,
        )
    for bname in valid_bnames:
        ET.SubElement(
            sensor_tag,
            "framelinvel",
            name=f"track_linvel_b_{bname}",
            objtype="xbody",
            objname=bname,
            reftype="xbody",
            refname=baselink_name,
        )
    for bname in valid_bnames:
        ET.SubElement(
            sensor_tag,
            "frameangvel",
            name=f"track_angvel_b_{bname}",
            objtype="xbody",
            objname=bname,
            reftype="xbody",
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
    tracked_body_ids, valid_bnames = _get_named_bodies(model_file)

    tree = ET.parse(model_file)
    root = tree.getroot()
    sensor_tag = root.find("sensor")
    if sensor_tag is None:
        sensor_tag = ET.SubElement(root, "sensor")

    _add_w_sensors(sensor_tag, valid_bnames)
    if baselink_name and baselink_name in valid_bnames:
        _add_b_sensors(sensor_tag, valid_bnames, baselink_name)

    return _write_temp_xml(tree, model_file), tracked_body_ids, valid_bnames


def inject_motrix_tracking_sensors(model_file: str, baselink_name: str) -> tuple[str, list, list]:
    """为 MotrixSim 后端注入 tracking sensors。

    只注入相对 baselink 坐标系的 (_b) sensors。
    世界系 (_w) 数据由 motrixsim body API 直接提供，无需 sensor 注入。

    Returns:
        (tmp_xml_path, tracked_body_ids, valid_bnames)
    """
    tracked_body_ids, valid_bnames = _get_named_bodies(model_file)

    tree = ET.parse(model_file)
    root = tree.getroot()
    sensor_tag = root.find("sensor")
    if sensor_tag is None:
        sensor_tag = ET.SubElement(root, "sensor")

    _add_b_sensors(sensor_tag, valid_bnames, baselink_name)

    return _write_temp_xml(tree, model_file), tracked_body_ids, valid_bnames


def processed_xml(xml_path):
    xml_dir = os.path.dirname(os.path.abspath(xml_path))

    # 加载模型 spec
    spec = mujoco.MjSpec().from_file(xml_path)
    full_xml = spec.to_xml()
    root = ET.fromstring(full_xml)

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
        # 将 Python 的下划线命名（可选）转换为 XML 习惯（如有必要）
        # 这里直接设置即可
        sensor.set(key, str(value))

    return sensor
