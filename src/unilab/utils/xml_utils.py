import os
import tempfile
import xml.etree.ElementTree as ET

import mujoco


def _get_named_bodies(model_file: str) -> tuple[list[int], list[str]]:
    _m = mujoco.MjModel.from_xml_path(model_file)
    ids, names = [], []
    for i in range(_m.nbody):
        name = mujoco.mj_id2name(_m, mujoco.mjtObj.mjOBJ_BODY, i)
        if name:
            ids.append(i)
            names.append(name)
    return ids, names


def _add_w_sensors(sensor_tag: ET.Element, valid_bnames: list[str]) -> None:
    for bname in valid_bnames:
        ET.SubElement(sensor_tag, "framepos", name=f"track_pos_w_{bname}", objtype="xbody", objname=bname)
    for bname in valid_bnames:
        ET.SubElement(sensor_tag, "framequat", name=f"track_quat_w_{bname}", objtype="xbody", objname=bname)
    for bname in valid_bnames:
        ET.SubElement(sensor_tag, "framelinvel", name=f"track_linvel_w_{bname}", objtype="xbody", objname=bname)
    for bname in valid_bnames:
        ET.SubElement(sensor_tag, "frameangvel", name=f"track_angvel_w_{bname}", objtype="xbody", objname=bname)


def _add_b_sensors(sensor_tag: ET.Element, valid_bnames: list[str], baselink_name: str) -> None:
    for bname in valid_bnames:
        ET.SubElement(sensor_tag, "framepos", name=f"track_pos_b_{bname}", objtype="xbody", objname=bname, reftype="xbody", refname=baselink_name)
    for bname in valid_bnames:
        ET.SubElement(sensor_tag, "framequat", name=f"track_quat_b_{bname}", objtype="xbody", objname=bname, reftype="xbody", refname=baselink_name)
    for bname in valid_bnames:
        ET.SubElement(sensor_tag, "framelinvel", name=f"track_linvel_b_{bname}", objtype="xbody", objname=bname, reftype="xbody", refname=baselink_name)
    for bname in valid_bnames:
        ET.SubElement(sensor_tag, "frameangvel", name=f"track_angvel_b_{bname}", objtype="xbody", objname=bname, reftype="xbody", refname=baselink_name)


def _write_temp_xml(tree: ET.ElementTree, model_file: str) -> str:
    fd, output_path = tempfile.mkstemp(
        suffix=".xml", dir=os.path.dirname(os.path.abspath(model_file))
    )
    os.close(fd)
    tree.write(output_path)
    return output_path


def inject_mujoco_tracking_sensors(
    model_file: str, baselink_name: str | None = None
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


def inject_motrix_tracking_sensors(
    model_file: str, baselink_name: str
) -> tuple[str, list, list]:
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
