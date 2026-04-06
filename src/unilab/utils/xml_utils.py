from __future__ import annotations

import os
import tempfile
import xml.etree.ElementTree as ET
from typing import Any, cast

import mujoco


def _enable_discardvisual(root: ET.Element) -> None:
    compiler_tag = root.find("compiler")
    if compiler_tag is None:
        compiler_tag = ET.Element("compiler")
        root.insert(0, compiler_tag)
    compiler_tag.set("discardvisual", "true")


def materialize_mujoco_xml(
    model_file: str,
    *,
    discard_visual: bool = False,
) -> tuple[str, bool]:
    if not discard_visual:
        return model_file, False

    tree = ET.parse(model_file)
    _enable_discardvisual(tree.getroot())
    return _write_temp_xml(tree, model_file), True


def _get_named_bodies(
    model_file: str, *, discard_visual: bool = False
) -> tuple[list[int], list[str]]:
    mj = cast(Any, mujoco)
    model_path, should_cleanup = materialize_mujoco_xml(
        model_file,
        discard_visual=discard_visual,
    )
    try:
        _m = mj.MjModel.from_xml_path(model_path)
    finally:
        if should_cleanup:
            os.remove(model_path)
    ids, names = [], []
    for i in range(1, _m.nbody):  # skip body 0 (world body)
        name = mj.mj_id2name(_m, mj.mjtObj.mjOBJ_BODY, i)
        if name:
            ids.append(i)
            names.append(name)
    return ids, names


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


def _write_temp_xml(tree: ET.ElementTree[ET.Element], model_file: str) -> str:
    fd, output_path = tempfile.mkstemp(
        suffix=".xml", dir=os.path.dirname(os.path.abspath(model_file))
    )
    os.close(fd)
    tree.write(output_path)
    return output_path


def inject_mujoco_tracking_sensors(
    model_file: str,
    baselink_name: str | None = None,
    *,
    discard_visual: bool = False,
) -> tuple[str, list, list]:
    """为 MuJoCo 后端注入 tracking sensors。

    注入所有 body 的世界系 (_w) sensors；若指定 baselink_name，
    同时注入相对 baselink 坐标系的 (_b) sensors。

    Returns:
        (tmp_xml_path, tracked_body_ids, valid_bnames)
    """
    tracked_body_ids, valid_bnames = _get_named_bodies(
        model_file,
        discard_visual=discard_visual,
    )

    tree = ET.parse(model_file)
    root = tree.getroot()
    if discard_visual:
        _enable_discardvisual(root)
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
