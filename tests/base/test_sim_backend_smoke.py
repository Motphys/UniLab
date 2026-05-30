"""Fast smoke tests for representative SimBackend contracts.

These tests stay in the default lane to cover the owner-layer backend
contract without running the full backend matrix in `test_sim_backend.py`.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base.backend.mujoco.xml import get_named_body_ids
from unilab.base.scene import SceneCfg
from unilab.dr.types import ResetRandomizationPayload

pytest.importorskip("mujoco", reason="mujoco not installed")


def _xml(robot: str, scene: str = "scene_flat.xml") -> str:
    return str(ASSETS_ROOT_PATH / "robots" / robot / scene)


BASIC_ROBOTS = [
    pytest.param(dict(model_file=_xml("g1"), base_name="pelvis"), id="g1"),
    pytest.param(dict(model_file=_xml("go1"), base_name="trunk"), id="go1"),
    pytest.param(dict(model_file=_xml("go2"), base_name="base"), id="go2"),
]

_G1 = dict(model_file=_xml("g1"), base_name="pelvis")
_ALLEGRO = dict(model_file=_xml("allegro_hand", "scene.xml"), base_name="palm")
_SHARPA = dict(model_file=_xml("sharpa_wave", "scene.xml"), base_name="right_hand_C_MC")

NUM_ENVS = 2
SIM_DT = 0.005
_CROSS_BACKEND_ATOL = 2e-3


def _shape(arr: np.ndarray, *expected: int) -> None:
    assert arr.shape == expected, f"expected shape {expected}, got {arr.shape}"


def _mujoco_module() -> Any:
    import mujoco

    return cast(Any, mujoco)


def _unit_quat(q: np.ndarray, tag: str = "") -> None:
    np.testing.assert_allclose(
        np.linalg.norm(q, axis=-1),
        1.0,
        atol=1e-5,
        err_msg=f"quaternion not unit — {tag}",
    )


def _identity_qpos_mujoco(nq: int, xyz=(0.0, 0.0, 0.8)) -> np.ndarray:
    q = np.zeros((1, nq))
    q[0, :3] = xyz
    q[0, 3] = 1.0
    return q


def _mujoco_expected_dof_dims(model) -> tuple[int, int]:
    mujoco = _mujoco_module()

    if model.njnt > 0 and int(model.jnt_type[0]) == int(mujoco.mjtJoint.mjJNT_FREE):
        return model.nq - 7, model.nv - 6
    return model.nq, model.nv


@pytest.mark.parametrize("robot", BASIC_ROBOTS)
def test_mujoco_backend_smoke_contract(robot):
    from unilab.base.backend.mujoco.backend import MuJoCoBackend

    bkd = MuJoCoBackend(
        SceneCfg(model_file=robot["model_file"]), NUM_ENVS, SIM_DT, base_name=robot["base_name"]
    )
    bkd.materialize()

    assert bkd.num_envs == NUM_ENVS
    assert bkd.model is not None
    assert bkd._pool is not None

    bkd.step(np.zeros((NUM_ENVS, bkd.model.nu)), nsteps=2)

    qpos = _identity_qpos_mujoco(bkd.model.nq, xyz=(1.0, 2.0, 0.8))
    bkd.set_state(np.array([0]), qpos, np.zeros((1, bkd.model.nv)))
    np.testing.assert_allclose(bkd.get_base_pos()[0], (1.0, 2.0, 0.8), atol=1e-5)
    _unit_quat(bkd.get_base_quat(), "MuJoCo smoke")

    caps = bkd.get_dr_capabilities()
    assert {
        "base_mass_delta",
        "base_com_offset",
        "gravity",
        "body_iquat",
        "body_inertia",
        "body_mass",
        "dof_armature",
        "geom_friction",
        "kp",
        "kd",
    }.issubset(caps.supported_reset_terms)
    assert caps.supports_interval_push


def test_mujoco_backend_fixed_base_dof_views_do_not_skip_first_joint():
    mujoco = _mujoco_module()

    from unilab.base.backend.mujoco.backend import MuJoCoBackend

    bkd = MuJoCoBackend(
        SceneCfg(model_file=_ALLEGRO["model_file"]),
        NUM_ENVS,
        SIM_DT,
        base_name=_ALLEGRO["base_name"],
    )
    assert int(bkd.model.jnt_type[0]) != int(mujoco.mjtJoint.mjJNT_FREE)
    _shape(bkd.get_dof_pos(), NUM_ENVS, bkd.model.nq)
    _shape(bkd.get_dof_vel(), NUM_ENVS, bkd.model.nv)
    _shape(bkd.get_base_pos(), NUM_ENVS, 3)
    _shape(bkd.get_base_quat(), NUM_ENVS, 4)
    np.testing.assert_allclose(bkd.get_base_lin_vel(), 0.0, atol=1e-8)
    np.testing.assert_allclose(bkd.get_base_ang_vel(), 0.0, atol=1e-8)
    _unit_quat(bkd.get_base_quat(), "MuJoCo fixed-base smoke")


@pytest.mark.parametrize("robot", BASIC_ROBOTS)
def test_motrix_backend_smoke_contract(robot):
    pytest.importorskip("motrixsim")

    from unilab.base.backend.motrix.backend import MotrixBackend

    bkd = MotrixBackend(
        SceneCfg(model_file=robot["model_file"]), NUM_ENVS, SIM_DT, base_name=robot["base_name"]
    )

    assert bkd.num_envs == NUM_ENVS
    assert bkd.model is not None
    assert bkd.data is not None

    ctrl = np.zeros_like(bkd.data.actuator_ctrls)
    bkd.step(ctrl, nsteps=2)

    nq = bkd.get_dof_pos().shape[-1] + 7
    nv = bkd.get_dof_vel().shape[-1] + 6
    qpos = _identity_qpos_mujoco(nq, xyz=(1.0, 2.0, 0.8))
    gravity = np.asarray([[1.0, 2.0, -3.0]], dtype=np.float64)
    bkd.set_state(
        np.array([0]),
        qpos,
        np.zeros((1, nv)),
        randomization=ResetRandomizationPayload(gravity=gravity),
    )
    np.testing.assert_allclose(bkd.get_base_pos()[0], (1.0, 2.0, 0.8), atol=1e-4)
    np.testing.assert_allclose(bkd.model.get_gravity_override(bkd.data)[0], gravity[0])
    _unit_quat(bkd.get_base_quat(), "Motrix smoke")

    caps = bkd.get_dr_capabilities()
    assert {"base_mass_delta", "base_com_offset", "gravity", "kp", "kd"}.issubset(
        caps.supported_reset_terms
    )
    assert caps.supports_interval_push
    play_caps = bkd.get_play_capabilities()
    assert play_caps.supports_native_interactive_renderer
    assert play_caps.supports_native_video_capture


def test_motrix_backend_fixed_base_base_views_are_available():
    pytest.importorskip("motrixsim")

    from unilab.base.backend.motrix.backend import MotrixBackend

    bkd = MotrixBackend(
        SceneCfg(model_file=_ALLEGRO["model_file"]),
        NUM_ENVS,
        SIM_DT,
        base_name=_ALLEGRO["base_name"],
    )
    _shape(bkd.get_base_pos(), NUM_ENVS, 3)
    _shape(bkd.get_base_quat(), NUM_ENVS, 4)
    np.testing.assert_allclose(bkd.get_base_lin_vel(), 0.0, atol=1e-8)
    np.testing.assert_allclose(bkd.get_base_ang_vel(), 0.0, atol=1e-8)
    _unit_quat(bkd.get_base_quat(), "Motrix fixed-base smoke")


@pytest.mark.parametrize(
    "robot",
    [
        pytest.param(dict(model_file=_xml("g1"), base_name="pelvis"), id="g1"),
        pytest.param(dict(model_file=_xml("go2"), base_name="base"), id="go2"),
    ],
)
def test_cross_backend_base_pose_smoke(robot):
    pytest.importorskip("motrixsim")

    from unilab.base.backend.motrix.backend import MotrixBackend
    from unilab.base.backend.mujoco.backend import MuJoCoBackend

    mj = MuJoCoBackend(
        SceneCfg(model_file=robot["model_file"]), NUM_ENVS, SIM_DT, base_name=robot["base_name"]
    )
    mj.materialize()
    mx = MotrixBackend(
        SceneCfg(model_file=robot["model_file"]), NUM_ENVS, SIM_DT, base_name=robot["base_name"]
    )

    nq, nv = mj.model.nq, mj.model.nv
    qpos = np.tile(_identity_qpos_mujoco(nq), (NUM_ENVS, 1))
    qvel = np.zeros((NUM_ENVS, nv))
    env_idx = np.arange(NUM_ENVS)
    mj.set_state(env_idx, qpos, qvel)
    mx.set_state(env_idx, qpos, qvel)

    np.testing.assert_allclose(mj.get_base_pos(), mx.get_base_pos(), atol=_CROSS_BACKEND_ATOL)
    np.testing.assert_allclose(
        np.abs(mj.get_base_quat()),
        np.abs(mx.get_base_quat()),
        atol=_CROSS_BACKEND_ATOL,
    )


def test_cross_backend_model_properties_smoke():
    pytest.importorskip("motrixsim")

    from unilab.base.backend.motrix.backend import MotrixBackend
    from unilab.base.backend.mujoco.backend import MuJoCoBackend

    mj = MuJoCoBackend(
        SceneCfg(model_file=_G1["model_file"]), NUM_ENVS, SIM_DT, base_name=_G1["base_name"]
    )
    mx = MotrixBackend(
        SceneCfg(model_file=_G1["model_file"]), NUM_ENVS, SIM_DT, base_name=_G1["base_name"]
    )

    assert mj.num_actuators == mx.num_actuators
    assert mj.num_dof_vel == mx.num_dof_vel
    assert mj.get_actuator_ctrl_range().shape == mx.get_actuator_ctrl_range().shape

    body_names = ["pelvis", "torso_link"]
    expected = np.asarray(get_named_body_ids(_G1["model_file"], body_names), dtype=np.int32)
    np.testing.assert_array_equal(mj.get_motion_body_ids(body_names), expected)
    np.testing.assert_array_equal(mx.get_motion_body_ids(body_names), expected)


@pytest.mark.parametrize("backend_type", ["mujoco", "motrix"])
def test_backend_batch_sensor_data_matches_individual_sensors(backend_type):
    if backend_type == "motrix":
        pytest.importorskip("motrixsim")

    from unilab.base.backend import create_backend
    from unilab.envs.locomotion.go2w.base import JOINT_SENSOR_PREFIXES

    bkd = create_backend(
        backend_type,
        SceneCfg(model_file=_xml("go2w", "scene_flat.xml")),
        NUM_ENVS,
        SIM_DT,
        base_name="base_link",
    )
    bkd.materialize()

    names = tuple(f"{prefix}_pos" for prefix in JOINT_SENSOR_PREFIXES[:4])
    expected = np.concatenate(
        [np.asarray(bkd.get_sensor_data(name)).reshape(NUM_ENVS, -1) for name in names],
        axis=1,
    )

    np.testing.assert_allclose(bkd.get_sensor_data_batch(names), expected)
    _shape(bkd.get_sensor_data_batch(()), NUM_ENVS, 0)


def test_mujoco_model_properties_smoke():
    from unilab.base.backend.mujoco.backend import MuJoCoBackend

    bkd = MuJoCoBackend(
        SceneCfg(model_file=_G1["model_file"]), NUM_ENVS, SIM_DT, base_name=_G1["base_name"]
    )
    assert bkd.num_actuators == bkd.model.nu
    assert bkd.num_actuators > 0
    assert bkd.num_dof_vel > 0
    assert bkd.num_dof_vel <= bkd.model.nv
    ctrl_range = bkd.get_actuator_ctrl_range()
    _shape(ctrl_range, bkd.num_actuators, 2)
    _, expected_nv = _mujoco_expected_dof_dims(bkd.model)
    assert expected_nv >= bkd.num_dof_vel


def test_mujoco_metadata_getters_return_stable_copies():
    mujoco = _mujoco_module()

    from unilab.base.backend.mujoco.backend import MuJoCoBackend

    bkd = MuJoCoBackend(
        SceneCfg(model_file=_SHARPA["model_file"]), NUM_ENVS, SIM_DT, base_name=_SHARPA["base_name"]
    )
    model = bkd.model
    object_geom_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "object"))
    base_body_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, _SHARPA["base_name"]))

    assert bkd.get_geom_id("object") == object_geom_id
    assert bkd.get_body_id(_SHARPA["base_name"]) == base_body_id
    with pytest.raises(ValueError, match="Geom 'missing'"):
        bkd.get_geom_id("missing")
    with pytest.raises(ValueError, match="Body 'missing'"):
        bkd.get_body_id("missing")

    default_qpos = bkd.get_default_qpos()
    _shape(default_qpos, model.nq)
    np.testing.assert_allclose(default_qpos, model.qpos0)
    default_qpos[0] += 1.0
    assert not np.isclose(default_qpos[0], model.qpos0[0])

    geom_size = bkd.get_geom_size("object")
    _shape(geom_size, 3)
    np.testing.assert_allclose(geom_size, model.geom_size[object_geom_id])
    geom_size[0] += 1.0
    assert not np.isclose(geom_size[0], model.geom_size[object_geom_id, 0])

    geom_body_ids = bkd.get_geom_body_ids()
    _shape(geom_body_ids, model.ngeom)
    np.testing.assert_array_equal(geom_body_ids, model.geom_bodyid)
    geom_body_ids[object_geom_id] = -1
    assert int(model.geom_bodyid[object_geom_id]) != -1

    geom_contype, geom_conaffinity = bkd.get_geom_contact_masks()
    _shape(geom_contype, model.ngeom)
    _shape(geom_conaffinity, model.ngeom)
    np.testing.assert_array_equal(geom_contype, model.geom_contype)
    np.testing.assert_array_equal(geom_conaffinity, model.geom_conaffinity)

    geom_names = bkd.get_geom_names()
    assert len(geom_names) == model.ngeom
    assert geom_names[object_geom_id] == "object"
    assert base_body_id in set(int(body_id) for body_id in bkd.get_body_subtree_ids(base_body_id))

    geom_friction = bkd.get_geom_friction()
    _shape(geom_friction, model.ngeom, 3)
    np.testing.assert_allclose(geom_friction, model.geom_friction)
    geom_friction[object_geom_id, 0] += 1.0
    assert not np.isclose(geom_friction[object_geom_id, 0], model.geom_friction[object_geom_id, 0])

    gravity = bkd.get_gravity()
    _shape(gravity, 3)
    np.testing.assert_allclose(gravity, model.opt.gravity)
    gravity[2] += 1.0
    assert not np.isclose(gravity[2], model.opt.gravity[2])

    body_mass = bkd.get_body_mass()
    _shape(body_mass, model.nbody)
    np.testing.assert_allclose(body_mass, model.body_mass)
    body_mass[base_body_id] += 1.0
    assert not np.isclose(body_mass[base_body_id], model.body_mass[base_body_id])

    body_ipos = bkd.get_body_ipos()
    _shape(body_ipos, model.nbody, 3)
    np.testing.assert_allclose(body_ipos, model.body_ipos)
    body_ipos[base_body_id, 0] += 1.0
    assert not np.isclose(body_ipos[base_body_id, 0], model.body_ipos[base_body_id, 0])

    dof_armature = bkd.get_dof_armature()
    _shape(dof_armature, model.nv)
    np.testing.assert_allclose(dof_armature, model.dof_armature)
    dof_armature[-1] += 1.0
    assert not np.isclose(dof_armature[-1], model.dof_armature[-1])


def test_mujoco_copy_body_state_matches_split_queries():
    from unilab.base.backend.mujoco.backend import MuJoCoBackend

    bkd = MuJoCoBackend(
        SceneCfg(model_file=_G1["model_file"]),
        NUM_ENVS,
        SIM_DT,
        base_name=_G1["base_name"],
        add_body_sensors=True,
    )
    bkd.materialize()
    bkd.step(np.zeros((NUM_ENVS, bkd.model.nu)), nsteps=1)
    body_ids = bkd.get_body_ids(["pelvis", "torso_link"])

    expected_pos = bkd.get_body_pos_w(body_ids)
    expected_quat = bkd.get_body_quat_w(body_ids)
    expected_lin_vel = bkd.get_body_lin_vel_w(body_ids)
    expected_ang_vel = bkd.get_body_ang_vel_w(body_ids)
    out_pos = np.empty_like(expected_pos)
    out_quat = np.empty_like(expected_quat)
    out_lin_vel = np.empty_like(expected_lin_vel)
    out_ang_vel = np.empty_like(expected_ang_vel)

    result = bkd.copy_body_state_w(body_ids, out_pos, out_quat, out_lin_vel, out_ang_vel)

    assert result == (out_pos, out_quat, out_lin_vel, out_ang_vel)
    np.testing.assert_allclose(out_pos, expected_pos)
    np.testing.assert_allclose(out_quat, expected_quat)
    np.testing.assert_allclose(out_lin_vel, expected_lin_vel)
    np.testing.assert_allclose(out_ang_vel, expected_ang_vel)


def test_motrix_model_properties_smoke():
    pytest.importorskip("motrixsim")

    from unilab.base.backend.motrix.backend import MotrixBackend

    bkd = MotrixBackend(
        SceneCfg(model_file=_G1["model_file"]), NUM_ENVS, SIM_DT, base_name=_G1["base_name"]
    )
    assert bkd.num_actuators > 0
    assert bkd.num_dof_vel > 0
    ctrl_range = bkd.get_actuator_ctrl_range()
    _shape(ctrl_range, bkd.num_actuators, 2)
    assert bkd.get_default_qpos().ndim == 1
    assert bkd.get_joint_range() is None


def test_motrix_copy_body_state_matches_split_queries():
    pytest.importorskip("motrixsim")

    from unilab.base.backend.motrix.backend import MotrixBackend

    bkd = MotrixBackend(
        SceneCfg(model_file=_G1["model_file"]),
        NUM_ENVS,
        SIM_DT,
        base_name=_G1["base_name"],
        add_body_sensors=True,
    )
    body_names = ["pelvis", "torso_link"]
    body_ids = np.asarray([bkd.model.get_link_index(name) for name in body_names], dtype=np.int32)

    expected_pos = bkd.get_body_pos_w(body_ids)
    expected_quat = bkd.get_body_quat_w(body_ids)
    expected_lin_vel = bkd.get_body_lin_vel_w(body_ids)
    expected_ang_vel = bkd.get_body_ang_vel_w(body_ids)
    out_pos = np.empty_like(expected_pos)
    out_quat = np.empty_like(expected_quat)
    out_lin_vel = np.empty_like(expected_lin_vel)
    out_ang_vel = np.empty_like(expected_ang_vel)

    result = bkd.copy_body_state_w(body_ids, out_pos, out_quat, out_lin_vel, out_ang_vel)

    assert result == (out_pos, out_quat, out_lin_vel, out_ang_vel)
    np.testing.assert_allclose(out_pos, expected_pos)
    np.testing.assert_allclose(out_quat, expected_quat)
    np.testing.assert_allclose(out_lin_vel, expected_lin_vel)
    np.testing.assert_allclose(out_ang_vel, expected_ang_vel)

    row_ids = np.array([1, 0, 1], dtype=np.int32)
    np.testing.assert_allclose(
        bkd.get_sensor_data_rows("pelvis_local_linvel", row_ids),
        bkd.get_sensor_data("pelvis_local_linvel")[row_ids],
    )


def test_motrix_default_qpos_uses_mujoco_quaternion_convention():
    pytest.importorskip("motrixsim")

    from unilab.base.backend.motrix.backend import MotrixBackend

    bkd = MotrixBackend(
        SceneCfg(model_file=_SHARPA["model_file"]), NUM_ENVS, SIM_DT, base_name=_SHARPA["base_name"]
    )
    qpos = bkd.get_default_qpos()
    assert qpos.ndim == 1

    assert len(bkd._floating_base_quat_indices) > 0
    for quat_indices in bkd._floating_base_quat_indices:
        np.testing.assert_allclose(
            np.abs(qpos[quat_indices]),
            [1.0, 0.0, 0.0, 0.0],
            atol=1.0e-6,
        )

    env_ids = np.arange(NUM_ENVS, dtype=np.int32)
    bkd.set_state(
        env_ids,
        np.broadcast_to(qpos, (NUM_ENVS, qpos.shape[0])).copy(),
        np.zeros((NUM_ENVS, bkd.model.num_dof_vel), dtype=np.float64),
    )
    object_body_id = bkd.get_body_id("object")
    np.testing.assert_allclose(
        np.abs(bkd.get_body_quat_w(np.asarray([object_body_id], dtype=np.int32))[:, 0, :]),
        np.broadcast_to([1.0, 0.0, 0.0, 0.0], (NUM_ENVS, 4)),
        atol=1.0e-6,
    )
