"""Protocol-level tests for the IsaacGym subprocess backend.

These tests drive ``IsaacGymBackend`` through ``isaacgym_mock_worker.py`` — a
deterministic kinematic fake that speaks the real wire protocol over real
shared memory on the host interpreter — so they run without the IsaacGym
runtime (real-runtime coverage lives in the slow conformance tests).
"""

from __future__ import annotations

import io
import sys
import textwrap
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from unisim.backend.isaacgym import protocol
from unisim.backend.isaacgym.backend import IsaacGymBackend, IsaacGymWorkerError
from unisim.backend.isaacgym.dependencies import (
    ENV_HOME,
    ENV_PYTHON,
    IsaacGymDependencyError,
    build_worker_env,
    resolve_isaacgym_runtime,
)
from unisim.backend.isaacgym.sensors import scan_scene_metadata

from unilab.base.backend_factory import create_backend
from unilab.base.scene import SceneCfg

_MOCK_WORKER = str(Path(__file__).resolve().parent / "isaacgym_mock_worker.py")

SIM_DT = 0.005
NUM_ENVS = 2

_ROBOT_XML = """
<mujoco>
  <worldbody>
    <geom name="floor_geom" type="plane" size="0 0 0.05"/>
    <body name="base">
      <freejoint/>
      <site name="imu_site"/>
      <site name="tilted_site" pos="0.1 0 0" quat="0.7071068 0 0.7071068 0"/>
      <site name="euler_site" euler="0.1 0 0"/>
      <geom name="base_geom" size="0.1"/>
      <body name="link0">
        <joint name="j0" type="hinge" range="-1.5 1.5"/>
        <geom name="g0" size="0.1"/>
        <body name="link1">
          <joint name="j1" type="hinge" range="-1.5 1.5"/>
          <geom name="g1" size="0.1"/>
          <body name="link2">
            <joint name="j2" type="hinge" range="-1.5 1.5"/>
            <geom name="g2" size="0.1"/>
            <geom name="foot_geom" size="0.1"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="j0" joint="j0" kp="10" kv="0.5" forcerange="-100 100" ctrlrange="-1.2 1.2"/>
    <position name="j1" joint="j1" kp="20" kv="1.0" forcerange="-80 80"/>
    <position name="j2" joint="j2" kp="30"/>
  </actuator>
</mujoco>
"""

_SCENE_XML = """
<mujoco>
  <include file="robot.xml"/>
  <sensor>
    <gyro name="base_gyro" site="imu_site"/>
    <velocimeter name="base_local_linvel" site="imu_site"/>
    <framequat name="link0_quat" objtype="body" objname="link0"/>
    <framepos name="link1_pos" objtype="body" objname="link1"/>
    <framepos name="tilted_site_pos" objtype="site" objname="tilted_site"/>
    <framezaxis name="base_upvector" objtype="body" objname="base"/>
    <framezaxis name="imu_upvector" objtype="site" objname="imu_site"/>
    <framezaxis name="tilted_upvector" objtype="site" objname="tilted_site"/>
    <framezaxis name="euler_upvector" objtype="site" objname="euler_site"/>
    <framezaxis name="geom_upvector" objtype="geom" objname="base_geom"/>
    <contact name="foot_contact" geom1="floor_geom" geom2="foot_geom" data="found" num="1"/>
    <accelerometer name="base_acc" site="imu_site"/>
  </sensor>
  <keyframe>
    <key name="home" qpos="0 0 0.8 1 0 0 0 0.1 0.2 0.3"/>
  </keyframe>
</mujoco>
"""


@pytest.fixture()
def scene_file(tmp_path: Path) -> str:
    (tmp_path / "robot.xml").write_text(textwrap.dedent(_ROBOT_XML), encoding="utf-8")
    scene = tmp_path / "scene.xml"
    scene.write_text(textwrap.dedent(_SCENE_XML), encoding="utf-8")
    return str(scene)


def _make_backend(scene_file: str, **kwargs: Any) -> IsaacGymBackend:
    kwargs.setdefault("worker_command", [sys.executable, _MOCK_WORKER])
    kwargs.setdefault("worker_timeout_s", 30.0)
    base_name = kwargs.pop("base_name", "base")
    backend = create_backend(
        "isaacgym",
        SceneCfg(model_file=scene_file),
        NUM_ENVS,
        SIM_DT,
        base_name=base_name,
        **kwargs,
    )
    assert isinstance(backend, IsaacGymBackend)
    return backend


@pytest.fixture()
def backend(scene_file: str) -> Any:
    instance = _make_backend(scene_file)
    instance.materialize()
    yield instance
    instance.close()


# ---------------------------------------------------------------------------
# Protocol unit tests
# ---------------------------------------------------------------------------


def test_protocol_message_roundtrip() -> None:
    stream = io.BytesIO()
    protocol.send_message(stream, protocol.CMD_STEP, {"nsteps": 4})
    stream.seek(0)
    message = protocol.recv_message(stream)
    assert message == {"cmd": "STEP", "payload": {"nsteps": 4}}

    stream.seek(0)
    header = stream.read(protocol.HEADER_SIZE)
    assert protocol.unpack_header(header) == len(
        protocol.pack_message(protocol.CMD_STEP, {"nsteps": 4})
    )
    assert protocol.decode_message(stream.read())["cmd"] == "STEP"


def test_protocol_recv_eof_raises() -> None:
    with pytest.raises(protocol.WorkerDisconnectedError):
        protocol.recv_message(io.BytesIO(b""))
    with pytest.raises(protocol.WorkerDisconnectedError):
        protocol.recv_message(io.BytesIO(b"\x10\x00"))


def test_protocol_slot_layout() -> None:
    shapes = protocol.slot_shapes(num_envs=4, num_dof=3, num_bodies=2)
    assert shapes["ctrl"] == (4, 3)
    assert shapes["root_state"] == (4, 13)
    assert shapes["dof_state"] == (4, 3, 2)
    assert shapes["body_state"] == (4, 2, 13)
    assert shapes["contact_force"] == (4, 2, 3)
    assert shapes["reset_env_ids"] == (4,)
    assert shapes["reset_qpos"] == (4, 10)
    assert shapes["reset_qvel"] == (4, 9)
    assert protocol.slot_dtype("reset_env_ids") == np.dtype(np.int32)
    assert protocol.slot_nbytes("ctrl", shapes["ctrl"]) == 4 * 3 * 4
    with pytest.raises(ValueError, match="unknown shm slot"):
        protocol.slot_dtype("nope")
    with pytest.raises(ValueError, match="num_envs>0"):
        protocol.slot_shapes(0, 3, 2)


def test_legacy_protocol_module_reexports_canonical_public_surface() -> None:
    from unisim.backend.subprocess_ipc import protocol as shared_protocol

    assert protocol.__all__ == shared_protocol.__all__
    for name in shared_protocol.__all__:
        assert getattr(protocol, name) is getattr(shared_protocol, name)


def test_protocol_error_payload() -> None:
    try:
        raise RuntimeError("boom")
    except RuntimeError as exc:
        payload = protocol.serialize_exception(exc)
    assert payload["type"] == "RuntimeError"
    assert payload["message"] == "boom"
    assert "RuntimeError: boom" in payload["traceback"]
    rendered = protocol.format_worker_error(payload)
    assert "RuntimeError" in rendered and "boom" in rendered


def test_protocol_quat_helpers() -> None:
    xyzw = np.array([[0.1, 0.2, 0.3, 0.9]])
    wxyz = protocol.xyzw_to_wxyz(xyzw)
    np.testing.assert_allclose(wxyz, [[0.9, 0.1, 0.2, 0.3]])
    np.testing.assert_allclose(protocol.wxyz_to_xyzw(wxyz), xyzw)

    # 90-degree rotation about z maps +x to +y (and inverse maps +y to +x).
    half = np.sqrt(0.5)
    quat = np.array([[half, 0.0, 0.0, half]])
    np.testing.assert_allclose(
        protocol.quat_rotate(quat, np.array([[1.0, 0.0, 0.0]])), [[0.0, 1.0, 0.0]], atol=1e-7
    )
    np.testing.assert_allclose(
        protocol.quat_rotate_inverse(quat, np.array([[0.0, 1.0, 0.0]])),
        [[1.0, 0.0, 0.0]],
        atol=1e-7,
    )


# ---------------------------------------------------------------------------
# Dependency discovery
# ---------------------------------------------------------------------------


def _make_runtime_tree(home: Path) -> None:
    python = home / "miniconda3" / "envs" / "hsgym" / "bin" / "python3.8"
    python.parent.mkdir(parents=True)
    python.write_text("#!/bin/sh\n", encoding="utf-8")
    (home / "miniconda3" / "envs" / "hsgym" / "lib").mkdir(parents=True)
    (home / "isaacgym" / "python").mkdir(parents=True)


def test_dependencies_resolve_default_layout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(ENV_HOME, str(tmp_path))
    monkeypatch.delenv(ENV_PYTHON, raising=False)
    _make_runtime_tree(tmp_path)
    runtime = resolve_isaacgym_runtime()
    assert runtime.python == tmp_path / "miniconda3" / "envs" / "hsgym" / "bin" / "python3.8"
    assert runtime.isaacgym_python == tmp_path / "isaacgym" / "python"
    env = build_worker_env(runtime)
    assert env["LD_LIBRARY_PATH"].split(":")[0] == str(
        tmp_path / "miniconda3" / "envs" / "hsgym" / "lib"
    )


def test_dependencies_python_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_HOME, str(tmp_path))
    _make_runtime_tree(tmp_path)
    custom = tmp_path / "custom_python"
    custom.write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.setenv(ENV_PYTHON, str(custom))
    assert resolve_isaacgym_runtime().python == custom
    assert resolve_isaacgym_runtime(python_override=str(custom)).python == custom


def test_dependencies_missing_runtime_is_actionable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(ENV_HOME, str(tmp_path / "empty"))
    monkeypatch.delenv(ENV_PYTHON, raising=False)
    with pytest.raises(IsaacGymDependencyError, match="setup_isaacgym_env.sh"):
        resolve_isaacgym_runtime()


# ---------------------------------------------------------------------------
# Mock-worker protocol tests
# ---------------------------------------------------------------------------


def test_materialize_binds_metadata_and_slots(backend: IsaacGymBackend) -> None:
    assert backend.num_envs == NUM_ENVS
    assert backend.num_actuators == 3
    assert backend.num_dof_vel == 3
    assert backend.get_actuator_names() == ("j0", "j1", "j2")
    assert backend.get_actuator_joint_names() == ("j0", "j1", "j2")
    np.testing.assert_allclose(
        backend.get_actuator_ctrl_range(),
        np.array([[-1.2, 1.2], [0.0, 0.0], [0.0, 0.0]], dtype=np.float32),
    )
    kp, kd = backend.get_actuator_gains()
    np.testing.assert_allclose(kp, [10.0, 20.0, 30.0])
    np.testing.assert_allclose(kd, [0.5, 1.0, 0.0])
    np.testing.assert_allclose(
        backend.get_joint_range(), np.array([[-1.5, 1.5]] * 3, dtype=np.float32)
    )
    np.testing.assert_allclose(backend.get_gravity(), [0.0, 0.0, -9.81])
    np.testing.assert_array_equal(
        backend.get_body_ids(["base", "link1"]), np.array([0, 2], dtype=np.int32)
    )
    with pytest.raises(ValueError, match="not found"):
        backend.get_body_ids(["missing_body"])

    default_qpos = backend.get_default_qpos()
    assert default_qpos.shape == (10,)
    # The sole scene keyframe ("home") is selected as the initial/default state.
    np.testing.assert_allclose(default_qpos, [0, 0, 0.8, 1, 0, 0, 0, 0.1, 0.2, 0.3], atol=1e-7)
    np.testing.assert_allclose(backend.get_init_qvel(), np.zeros(9))
    np.testing.assert_allclose(backend.get_default_dof_pos(), [0.1, 0.2, 0.3], atol=1e-7)
    np.testing.assert_allclose(
        backend.get_keyframe_qpos("home"), [0, 0, 0.8, 1, 0, 0, 0, 0.1, 0.2, 0.3]
    )
    with pytest.raises(ValueError, match="Keyframe 'missing'"):
        backend.get_keyframe_qpos("missing")

    layout = backend.get_root_state_layout("base")
    assert layout.qpos_indices == tuple(range(7))
    assert layout.qvel_indices == tuple(range(6))
    with pytest.raises(NotImplementedError, match="root-state layout"):
        backend.get_root_state_layout("link0")

    np.testing.assert_array_equal(
        backend.get_joint_state_qpos_indices(["j1"]), np.array([8], dtype=np.int32)
    )
    np.testing.assert_array_equal(
        backend.get_joint_state_qvel_indices(["j1"]), np.array([7], dtype=np.int32)
    )

    # Initial state written by the mock at ATTACH reflects the scene keyframe:
    # root at z=0.8, dofs at the keyframe joint positions.
    np.testing.assert_allclose(backend.get_base_pos(), [[0, 0, 0.8], [0, 0, 0.8]], atol=1e-6)
    np.testing.assert_allclose(backend.get_dof_pos(), [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]], atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(backend.get_base_quat(), axis=-1), 1.0)
    assert backend.model.num_bodies == 4


def test_init_applies_scene_keyframe(scene_file: str) -> None:
    """INIT applies the scene keyframe as the initial root pose + dof pos."""
    backend = _make_backend(scene_file)
    backend.materialize()
    try:
        np.testing.assert_allclose(backend.get_dof_pos(), [[0.1, 0.2, 0.3]] * NUM_ENVS, atol=1e-6)
        np.testing.assert_allclose(backend.get_base_pos(), [[0.0, 0.0, 0.8]] * NUM_ENVS, atol=1e-6)
        np.testing.assert_allclose(
            backend.get_base_quat(), [[1.0, 0.0, 0.0, 0.0]] * NUM_ENVS, atol=1e-6
        )
        # The default-state contract matches the post-INIT worker state.
        np.testing.assert_allclose(
            backend.get_default_dof_pos(), backend.get_dof_pos()[0], atol=1e-7
        )
    finally:
        backend.close()


def test_xml_metadata_available_before_materialize(scene_file: str) -> None:
    """Keyframe/default qpos are pure parent-side XML metadata.

    Env construction (``resolve_scene_default_qpos``) queries them before
    ``materialize()`` spawns the worker; values must be identical afterwards.
    """
    backend = _make_backend(scene_file)
    expected_qpos = np.array([0, 0, 0.8, 1, 0, 0, 0, 0.1, 0.2, 0.3], dtype=np.float32)
    np.testing.assert_allclose(backend.get_keyframe_qpos("home"), expected_qpos)
    np.testing.assert_allclose(backend.get_default_qpos(), expected_qpos)
    np.testing.assert_allclose(backend.get_default_dof_pos(), [0.1, 0.2, 0.3])
    # Pure-XML metadata is available pre-materialize (env constructors read
    # these before the worker handshake).
    assert backend.num_actuators == 3
    assert backend.num_dof_vel == 3
    np.testing.assert_allclose(backend.get_init_qvel(), np.zeros(9))
    np.testing.assert_array_equal(
        backend.get_body_ids(["base", "link1"]), np.array([0, 2], dtype=np.int32)
    )
    assert backend.get_actuator_names() == ("j0", "j1", "j2")
    layout = backend.get_root_state_layout("base")
    assert layout.qpos_indices == tuple(range(7))
    # Worker-handshake-dependent surfaces materialize lazily on first use:
    # none of the pure-XML queries above spawned the worker.
    assert backend._proc is None
    _ = backend.model
    assert backend._proc is not None

    backend.materialize()
    try:
        np.testing.assert_allclose(backend.get_keyframe_qpos("home"), expected_qpos)
        np.testing.assert_allclose(backend.get_default_qpos(), expected_qpos)
        np.testing.assert_allclose(backend.get_default_dof_pos(), [0.1, 0.2, 0.3])
        np.testing.assert_allclose(backend.get_dof_pos(), [[0.1, 0.2, 0.3]] * NUM_ENVS, atol=1e-6)
    finally:
        backend.close()


def test_actuator_metadata_available_before_materialize(scene_file: str) -> None:
    """PD gains / ctrl ranges / joint ranges are pure XML metadata.

    IsaacGym's MJCF importer drops kv/frictionloss/joint ranges, so these
    surfaces are answered from the parent-side scene scan and must not spawn
    the worker.
    """
    backend = _make_backend(scene_file)
    kp, kd = backend.get_actuator_gains()
    np.testing.assert_allclose(kp, [10.0, 20.0, 30.0])
    np.testing.assert_allclose(kd, [0.5, 1.0, 0.0])
    np.testing.assert_allclose(
        backend.get_actuator_ctrl_range(),
        np.array([[-1.2, 1.2], [0.0, 0.0], [0.0, 0.0]], dtype=np.float32),
    )
    np.testing.assert_allclose(
        backend.get_joint_range(), np.array([[-1.5, 1.5]] * 3, dtype=np.float32)
    )
    assert backend._proc is None


def test_scan_resolves_joint_default_classes(tmp_path: Path) -> None:
    """Joint armature/frictionloss resolve through MJCF default classes."""
    robot = """
    <mujoco>
      <default>
        <default class="robot">
          <joint armature="0.01" frictionloss="0.3"/>
        </default>
      </default>
      <worldbody>
        <body name="base" childclass="robot">
          <freejoint/>
          <body name="link0">
            <joint name="j0" type="hinge"/>
            <body name="link1">
              <joint name="j1" type="hinge" armature="0.5"/>
            </body>
          </body>
        </body>
      </worldbody>
    </mujoco>
    """
    scene = tmp_path / "scene.xml"
    scene.write_text(textwrap.dedent(robot), encoding="utf-8")
    metadata = scan_scene_metadata(str(scene))
    assert metadata.joint_names == ("j0", "j1")
    np.testing.assert_allclose(metadata.joint_armature, [0.01, 0.5])
    np.testing.assert_allclose(metadata.joint_frictionloss, [0.3, 0.3])
    assert np.isinf(metadata.joint_ranges[0]).all()


def test_scan_fails_closed_on_non_position_actuator(tmp_path: Path) -> None:
    """<motor>/other actuator types would break the position-target ctrl contract."""
    robot = textwrap.dedent(_ROBOT_XML).replace(
        "</mujoco>",
        '<actuator><motor name="m0" joint="j0"/></actuator>\n</mujoco>',
    )
    scene = tmp_path / "scene.xml"
    scene.write_text(robot, encoding="utf-8")
    with pytest.raises(NotImplementedError, match="position"):
        scan_scene_metadata(str(scene))


def test_scan_fails_closed_on_asymmetric_forcerange(tmp_path: Path) -> None:
    """PhysX dof effort limits are symmetric; asymmetric MJCF forceranges fail."""
    robot = textwrap.dedent(_ROBOT_XML).replace(
        'forcerange="-100 100"',
        'forcerange="-50 100"',
    )
    scene = tmp_path / "scene.xml"
    scene.write_text(robot, encoding="utf-8")
    with pytest.raises(NotImplementedError, match="asymmetric forcerange"):
        scan_scene_metadata(str(scene))


def test_init_keyframe_selection_prefers_default_keyframe_name(tmp_path: Path) -> None:
    scene = textwrap.dedent(_SCENE_XML).replace(
        '<key name="home" qpos="0 0 0.8 1 0 0 0 0.1 0.2 0.3"/>',
        '<key name="home" qpos="0 0 0.8 1 0 0 0 0.1 0.2 0.3"/>'
        '<key name="crouch" qpos="0 0 0.5 1 0 0 0 -0.1 -0.2 -0.3"/>',
    )
    (tmp_path / "robot.xml").write_text(textwrap.dedent(_ROBOT_XML), encoding="utf-8")
    (tmp_path / "scene.xml").write_text(scene, encoding="utf-8")
    scene_file = str(tmp_path / "scene.xml")

    # Ambiguous keyframes without default_keyframe_name keep the zero default.
    backend = _make_backend(scene_file)
    backend.materialize()
    try:
        np.testing.assert_allclose(backend.get_default_dof_pos(), np.zeros(3))
        np.testing.assert_allclose(backend.get_dof_pos(), np.zeros((NUM_ENVS, 3)))
    finally:
        backend.close()

    # An explicit default_keyframe_name selects that keyframe.
    backend = create_backend(
        "isaacgym",
        SceneCfg(model_file=scene_file, default_keyframe_name="crouch"),
        NUM_ENVS,
        SIM_DT,
        base_name="base",
        worker_command=[sys.executable, _MOCK_WORKER],
        worker_timeout_s=30.0,
    )
    assert isinstance(backend, IsaacGymBackend)
    backend.materialize()
    try:
        np.testing.assert_allclose(
            backend.get_dof_pos(), [[-0.1, -0.2, -0.3]] * NUM_ENVS, atol=1e-6
        )
        np.testing.assert_allclose(backend.get_base_pos(), [[0.0, 0.0, 0.5]] * NUM_ENVS, atol=1e-6)
    finally:
        backend.close()

    # A missing default_keyframe_name fails fast at materialize.
    backend = create_backend(
        "isaacgym",
        SceneCfg(model_file=scene_file, default_keyframe_name="missing"),
        NUM_ENVS,
        SIM_DT,
        base_name="base",
        worker_command=[sys.executable, _MOCK_WORKER],
        worker_timeout_s=30.0,
    )
    with pytest.raises(ValueError, match="default_keyframe_name 'missing' not found"):
        backend.materialize()
    backend.close()


def test_init_keyframe_dof_name_mismatch_fails(
    scene_file: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("UNILAB_ISAACGYM_MOCK_DOF_NAMES", "j0,j1,renamed_joint")
    backend = _make_backend(scene_file)
    try:
        with pytest.raises(IsaacGymWorkerError, match="renamed_joint.*mjcf_joint_names"):
            backend.materialize()
    finally:
        backend.close()


def test_step_integrates_mock_physics(backend: IsaacGymBackend) -> None:
    ctrl = np.ones((NUM_ENVS, 3), dtype=np.float32)
    result = backend.step(ctrl, nsteps=2)
    assert "timing" in result
    assert "physics_ms" in result["timing"]

    # Mock: dof_vel += ctrl*dt; dof_pos += dof_vel*dt per substep, starting
    # from the keyframe joint positions [0.1, 0.2, 0.3].
    expected_vel = np.float32(2 * SIM_DT)
    np.testing.assert_allclose(backend.get_dof_vel(), expected_vel, atol=1e-7)
    expected_pos = np.array([[0.1, 0.2, 0.3]], dtype=np.float32) + (
        np.float32(SIM_DT * SIM_DT) + np.float32(2 * SIM_DT * SIM_DT)
    )
    np.testing.assert_allclose(
        backend.get_dof_pos(), np.broadcast_to(expected_pos, (NUM_ENVS, 3)), atol=1e-7
    )

    # Free fall from the keyframe root height: v_z after 2 substeps, z += v_z*dt.
    expected_vz = 2 * (-9.81) * SIM_DT
    expected_z = 0.8 + (-9.81 * SIM_DT * SIM_DT) + (-9.81 * 2 * SIM_DT * SIM_DT)
    np.testing.assert_allclose(backend.get_base_lin_vel()[:, 2], expected_vz, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(backend.get_base_pos()[:, 2], expected_z, rtol=1e-6, atol=1e-7)

    with pytest.raises(ValueError, match="ctrl must have shape"):
        backend.step(np.zeros((NUM_ENVS, 2), dtype=np.float32))
    with pytest.raises(ValueError, match="nsteps"):
        backend.step(ctrl, nsteps=0)


def test_set_state_roundtrip_and_cache_refresh(backend: IsaacGymBackend) -> None:
    nq = 10
    nv = 9
    rows = np.array([0], dtype=np.int32)
    qpos = np.zeros((1, nq), dtype=np.float32)
    qpos[0, :3] = [1.0, 2.0, 0.8]
    qpos[0, 3] = 1.0
    qpos[0, 7:] = [0.1, 0.2, 0.3]
    qvel = np.zeros((1, nv), dtype=np.float32)
    qvel[0, :3] = [0.5, 0.0, 0.0]
    qvel[0, 6:] = [1.0, 2.0, 3.0]

    result = backend.set_state(rows, qpos, qvel)
    assert isinstance(result, dict) and "timing" in result
    assert "set_state_internal_gap_ms" in result["timing"]

    np.testing.assert_allclose(backend.get_base_pos()[0], [1.0, 2.0, 0.8], atol=1e-6)
    np.testing.assert_allclose(backend.get_base_lin_vel()[0], [0.5, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(backend.get_dof_pos()[0], [0.1, 0.2, 0.3], atol=1e-6)
    np.testing.assert_allclose(backend.get_dof_vel()[0], [1.0, 2.0, 3.0], atol=1e-6)
    # Untouched env keeps its previous state.
    np.testing.assert_allclose(backend.get_base_pos()[1], [0, 0, 0.8], atol=1e-6)

    with pytest.raises(ValueError, match="qpos must have shape"):
        backend.set_state(rows, np.zeros((1, nq + 1), dtype=np.float32), qvel)
    with pytest.raises(ValueError, match="duplicate rows"):
        backend.set_state(np.array([0, 0], dtype=np.int32), np.zeros((2, nq)), np.zeros((2, nv)))


def test_sensor_mapping_from_scene_scan(backend: IsaacGymBackend) -> None:
    gyro = backend.get_sensor_data("base_gyro")
    assert gyro.shape == (NUM_ENVS, 3)
    np.testing.assert_allclose(gyro, 0.0, atol=1e-6)
    linvel = backend.get_sensor_data("base_local_linvel")
    assert linvel.shape == (NUM_ENVS, 3)

    quat = backend.get_sensor_data("link0_quat")
    assert quat.shape == (NUM_ENVS, 4)
    np.testing.assert_allclose(np.linalg.norm(quat, axis=-1), 1.0, atol=1e-6)

    pos = backend.get_sensor_data("link1_pos")
    np.testing.assert_allclose(pos, [[0.2, 0.0, 0.8]] * NUM_ENVS, atol=1e-6)

    # contact "found": zero while ctrl is zero, one after a nonzero torque step.
    np.testing.assert_allclose(backend.get_sensor_data("foot_contact"), 0.0)
    backend.step(np.ones((NUM_ENVS, 3), dtype=np.float32))
    np.testing.assert_allclose(backend.get_sensor_data("foot_contact"), 1.0)

    # Declared but unmappable sensor type fails closed with context.
    with pytest.raises(NotImplementedError, match="accelerometer"):
        backend.get_sensor_data("base_acc")
    with pytest.raises(ValueError, match="Sensor 'nope' not found"):
        backend.get_sensor_data("nope")

    view = backend.bind_sensor_data(("base_gyro", "base_local_linvel"))
    assert view.dimensions == (3, 3)
    values = view.read()
    assert values.shape == (NUM_ENVS, 6)
    assert np.isfinite(values).all()


def test_framezaxis_sensor_mapping(backend: IsaacGymBackend) -> None:
    """framezaxis reports the target frame's z axis in world coordinates."""
    # Identity orientations at materialize: body/site z axes point up; the
    # tilted site's local quat (90 deg about y) maps its z axis to +x.
    np.testing.assert_allclose(
        backend.get_sensor_data("base_upvector"), [[0, 0, 1]] * NUM_ENVS, atol=1e-6
    )
    np.testing.assert_allclose(
        backend.get_sensor_data("imu_upvector"), [[0, 0, 1]] * NUM_ENVS, atol=1e-6
    )
    np.testing.assert_allclose(
        backend.get_sensor_data("tilted_upvector"), [[1, 0, 0]] * NUM_ENVS, atol=1e-5
    )
    # framepos on a site includes the site's local pos offset (rigid attach).
    np.testing.assert_allclose(
        backend.get_sensor_data("tilted_site_pos"), [[0.1, 0.0, 0.8]] * NUM_ENVS, atol=1e-6
    )

    # Rotate the root 90 deg about x (wxyz): body z axis maps to -y.
    half = np.float32(np.sqrt(0.5))
    qpos = np.zeros((NUM_ENVS, 10), dtype=np.float32)
    qpos[:, 2] = 1.0
    qpos[:, 3] = half
    qpos[:, 4] = half
    qvel = np.zeros((NUM_ENVS, 9), dtype=np.float32)
    backend.set_state(np.arange(NUM_ENVS, dtype=np.int32), qpos, qvel)

    np.testing.assert_allclose(
        backend.get_sensor_data("base_upvector"), [[0, -1, 0]] * NUM_ENVS, atol=1e-5
    )
    np.testing.assert_allclose(
        backend.get_sensor_data("imu_upvector"), [[0, -1, 0]] * NUM_ENVS, atol=1e-5
    )
    # tilted site: local z->x, then the body rotation keeps +x at +x.
    np.testing.assert_allclose(
        backend.get_sensor_data("tilted_upvector"), [[1, 0, 0]] * NUM_ENVS, atol=1e-5
    )

    # Sites declared with euler/axisangle/xyaxes/zaxis fail closed.
    with pytest.raises(NotImplementedError, match="euler"):
        backend.get_sensor_data("euler_upvector")
    # Unknown frame target objtype fails closed as well.
    with pytest.raises(NotImplementedError, match="objtype"):
        backend.get_sensor_data("geom_upvector")
    metadata = backend._scene_metadata
    assert metadata is not None
    assert "euler_upvector" in metadata.unsupported_sensors
    assert "geom_upvector" in metadata.unsupported_sensors


def test_body_state_views(backend: IsaacGymBackend) -> None:
    ids = backend.get_body_ids(["base", "link1"])
    pos_w = backend.get_body_pos_w(ids)
    assert pos_w.shape == (NUM_ENVS, 2, 3)
    np.testing.assert_allclose(pos_w[:, 1], [[0.2, 0.0, 0.8]] * NUM_ENVS, atol=1e-6)
    # Base-frame link1 offset: [0.2, 0, 0] under the identity base orientation.
    pos_b = backend.get_body_pos_b(ids)
    np.testing.assert_allclose(pos_b[:, 1], [[0.2, 0.0, 0.0]] * NUM_ENVS, atol=1e-6)
    quat_b = backend.get_body_quat_b(ids)
    np.testing.assert_allclose(quat_b[:, :, 0], 1.0, atol=1e-6)
    with pytest.raises(ValueError, match="body_ids"):
        backend.get_body_pos_w(np.array([99]))


def test_dr_and_pre_step_control_fail_closed(backend: IsaacGymBackend) -> None:
    capabilities = backend.get_dr_capabilities()
    assert not capabilities.supported_reset_terms
    assert not capabilities.supports_interval_push
    assert not capabilities.supports_interval_body_force

    class _Plan:
        def is_empty(self) -> bool:
            return False

    with pytest.raises(NotImplementedError, match="interval randomization"):
        backend.apply_interval_randomization(_Plan())

    class _Payload:
        def is_empty(self) -> bool:
            return False

        def requested_terms(self) -> set[str]:
            return {"base_mass"}

    with pytest.raises(NotImplementedError, match="reset domain randomization"):
        backend.set_state(
            np.array([0], dtype=np.int32),
            np.zeros((1, 10), dtype=np.float32),
            np.zeros((1, 9), dtype=np.float32),
            randomization=_Payload(),
        )

    with pytest.raises(NotImplementedError, match="pre-step"):
        backend.set_pre_step_control(lambda backend_, ctrl: ctrl)
    backend.set_pre_step_control(None)

    # Physics-state playback export stays unsupported; native rendering has
    # its own dedicated tests below.
    with pytest.raises(NotImplementedError, match="physics-state playback"):
        backend.get_physics_state()


def test_close_reaps_worker_and_unlinks_shm(backend: IsaacGymBackend) -> None:
    proc = backend._proc
    assert proc is not None and proc.poll() is None
    shm_names = [handle.name for handle in backend._shm_handles.values()]
    backend.close()
    assert proc.poll() is not None
    from multiprocessing import shared_memory

    for name in shm_names:
        with pytest.raises(FileNotFoundError):
            shared_memory.SharedMemory(name=name, create=False)
    backend.close()  # idempotent
    with pytest.raises(IsaacGymWorkerError, match="closed or not materialized"):
        backend.step(np.zeros((NUM_ENVS, 3), dtype=np.float32))


def test_worker_init_error_propagates(scene_file: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("UNILAB_ISAACGYM_MOCK_BEHAVIOR", "fail_init")
    backend = _make_backend(scene_file)
    try:
        with pytest.raises(IsaacGymWorkerError, match="mock init failure"):
            backend.materialize()
    finally:
        backend.close()


def test_worker_death_reports_stderr_tail(scene_file: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("UNILAB_ISAACGYM_MOCK_BEHAVIOR", "die_on_step")
    backend = _make_backend(scene_file)
    backend.materialize()
    try:
        with pytest.raises(IsaacGymWorkerError, match="mock dying on step"):
            backend.step(np.zeros((NUM_ENVS, 3), dtype=np.float32))
        # Later calls fail fast with the cached death diagnostic.
        with pytest.raises(IsaacGymWorkerError, match="earlier failure"):
            backend.step(np.zeros((NUM_ENVS, 3), dtype=np.float32))
    finally:
        backend.close()


def test_worker_timeout_kills_worker(scene_file: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("UNILAB_ISAACGYM_MOCK_BEHAVIOR", "hang_on_step")
    backend = _make_backend(scene_file, worker_timeout_s=1.0)
    backend.materialize()
    try:
        with pytest.raises(IsaacGymWorkerError, match="did not answer STEP"):
            backend.step(np.zeros((NUM_ENVS, 3), dtype=np.float32))
        assert backend._proc is not None
        assert backend._proc.poll() is not None
    finally:
        backend.close()


def test_constructor_rejects_bad_arguments(scene_file: str) -> None:
    with pytest.raises(ValueError, match="num_envs"):
        create_backend("isaacgym", SceneCfg(model_file=scene_file), 0, SIM_DT)
    with pytest.raises(ValueError, match="sim_dt"):
        create_backend("isaacgym", SceneCfg(model_file=scene_file), 1, -1.0, worker_command=["x"])
    with pytest.raises(TypeError, match="worker_command"):
        _make_backend(scene_file, worker_command="not-a-list")
    with pytest.raises(NotImplementedError, match="fragments"):
        create_backend(
            "isaacgym",
            SceneCfg(model_file=scene_file, fragment_files=["extra.xml"]),
            1,
            SIM_DT,
        )
    with pytest.raises(TypeError, match="does not accept backend options"):
        create_backend("isaacgym", SceneCfg(model_file=scene_file), 1, SIM_DT, bogus_option=1)


def test_g1_scene_framezaxis_sensors_resolve_from_real_xml(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The g1_walk_flat scene declares the upvector sensors the policy needs.

    All four are ``objtype="site"`` frames in ``g1.xml``; the mock worker
    reports the owning bodies so the cold-path scan must resolve them to
    exact site-frame z axes.
    """
    from unisim.backend.isaacgym.sensors import scan_scene_metadata

    from unilab.assets import ASSETS_ROOT_PATH

    scene = str(ASSETS_ROOT_PATH / "robots" / "g1" / "scene_flat.xml")
    # Give the mock the real MJCF joint/body sets so the scene's "stand"
    # keyframe (29 dofs) maps onto the fake asset by name, as on the real
    # worker, and the XML↔worker name-order validation passes.
    scene_metadata = scan_scene_metadata(scene)
    monkeypatch.setenv("UNILAB_ISAACGYM_MOCK_DOF_NAMES", ",".join(scene_metadata.joint_names))
    monkeypatch.setenv(
        "UNILAB_ISAACGYM_MOCK_BODY_NAMES",
        ",".join(name or f"unnamed_{i}" for i, name in enumerate(scene_metadata.body_names)),
    )
    backend = _make_backend(scene, base_name="pelvis")
    backend.materialize()
    try:
        for name in (
            "pelvis_upvector",
            "torso_upvector",
            "left_foot_upvector",
            "right_foot_upvector",
        ):
            value = backend.get_sensor_data(name)
            assert value.shape == (NUM_ENVS, 3)
            # Mock state starts at the identity orientation, so every z axis
            # (these sites are identity-rotated within their bodies) points up.
            np.testing.assert_allclose(value, [[0.0, 0.0, 1.0]] * NUM_ENVS, atol=1e-6)
    finally:
        backend.close()


def test_state_access_before_materialize_lazy_spawns_worker(scene_file: str) -> None:
    """State reads/step auto-materialize: the constructor stays light, but the
    first state-dependent call performs the worker handshake (matching the
    MuJoCo backend's fully-constructed availability)."""
    backend = _make_backend(scene_file)
    try:
        assert backend._proc is None
        np.testing.assert_allclose(backend.get_dof_pos(), [[0.1, 0.2, 0.3]] * NUM_ENVS, atol=1e-6)
        assert backend._proc is not None
        # Explicit materialize() afterwards is a no-op.
        backend.materialize()
        backend.step(np.zeros((NUM_ENVS, 3), dtype=np.float32))
        assert backend.get_dof_pos().shape == (NUM_ENVS, 3)
    finally:
        backend.close()


# ---------------------------------------------------------------------------
# Native rendering / playback
# ---------------------------------------------------------------------------


def test_play_capabilities_advertise_native_rendering(backend: IsaacGymBackend) -> None:
    caps = backend.get_play_capabilities()
    assert caps.supports_native_interactive_renderer
    assert caps.supports_native_video_capture
    assert not caps.supports_physics_state_playback


def test_normalize_camera_kwargs_maps_mujoco_convention() -> None:
    from unisim.backend.isaacgym.backend import _normalize_camera_kwargs

    out = _normalize_camera_kwargs(
        {"cam_distance": 3.0, "cam_elevation": -20.0, "cam_azimuth": 90.0}
    )
    assert out == {"distance": 3.0, "elevation_deg": 20.0, "azimuth_deg": 90.0}
    assert _normalize_camera_kwargs(None) == {
        "distance": 2.0,
        "elevation_deg": 20.0,
        "azimuth_deg": 90.0,
    }


def test_resolve_play_render_plan_modes(
    scene_file: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    backend = _make_backend(scene_file)
    try:
        plan = backend.resolve_play_render_plan(
            play_render_mode="none", play_steps=10, output_video=tmp_path / "x.mp4"
        )
        assert plan.mode == "none" and plan.num_steps is None

        plan = backend.resolve_play_render_plan(
            play_render_mode="interactive", play_steps=None, output_video=None
        )
        assert plan.mode == "interactive" and not plan.headless and not plan.record_video

        plan = backend.resolve_play_render_plan(
            play_render_mode="record", play_steps=5, output_video=tmp_path / "x.mp4"
        )
        assert plan.mode == "record" and plan.headless and plan.record_video
        assert plan.num_steps == 5

        with pytest.raises(ValueError, match="play_steps"):
            backend.resolve_play_render_plan(
                play_render_mode="record", play_steps=None, output_video=tmp_path / "x.mp4"
            )
        with pytest.raises(ValueError, match="output video path"):
            backend.resolve_play_render_plan(
                play_render_mode="record", play_steps=5, output_video=None
            )

        monkeypatch.setenv("DISPLAY", ":0")
        plan = backend.resolve_play_render_plan(
            play_render_mode="auto", play_steps=5, output_video=None
        )
        assert plan.mode == "interactive"

        monkeypatch.delenv("DISPLAY", raising=False)
        monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
        plan = backend.resolve_play_render_plan(
            play_render_mode="auto", play_steps=5, output_video=tmp_path / "x.mp4"
        )
        assert plan.mode == "record" and plan.headless
    finally:
        backend.close()


def test_interactive_renderer_roundtrip(backend: IsaacGymBackend) -> None:
    backend.init_renderer(headless=False)
    backend.render()
    # Re-initializing with the same config is a no-op; a different one fails.
    backend.init_renderer(headless=False)
    with pytest.raises(RuntimeError, match="already initialized"):
        backend.init_renderer(headless=True, capture=True)
    # Capturing on an interactive-only renderer must not silently succeed.
    with pytest.raises(RuntimeError, match="already initialized"):
        backend.capture_video_frame()


def test_viewer_close_raises_render_closed(
    scene_file: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from unisim.backend.base import RenderClosedError

    monkeypatch.setenv("UNILAB_ISAACGYM_MOCK_BEHAVIOR", "close_on_render")
    backend = _make_backend(scene_file)
    backend.materialize()
    try:
        backend.init_renderer(headless=False)
        with pytest.raises(RenderClosedError, match="closed"):
            backend.render()
    finally:
        backend.close()


def test_viewer_creation_failure_is_actionable(
    scene_file: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("UNILAB_ISAACGYM_MOCK_BEHAVIOR", "viewer_fails")
    backend = _make_backend(scene_file)
    backend.materialize()
    try:
        with pytest.raises(IsaacGymWorkerError, match="create_viewer failed"):
            backend.init_renderer(headless=False)
    finally:
        backend.close()


def test_renderer_requires_graphics_context(
    scene_file: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("UNILAB_ISAACGYM_MOCK_BEHAVIOR", "no_graphics")
    backend = _make_backend(scene_file)
    backend.materialize()
    try:
        with pytest.raises(NotImplementedError, match="requires a GPU sim"):
            backend.init_renderer(headless=True, capture=True)
    finally:
        backend.close()


def test_capture_video_frame_roundtrip(backend: IsaacGymBackend) -> None:
    backend.init_renderer(headless=True, capture=True, width=64, height=48)
    frame = backend.capture_video_frame()
    assert frame.shape == (48, 64, 3)
    assert frame.dtype == np.uint8
    # Mock frame is a deterministic gradient: red follows rows, green columns.
    assert frame[10, 0, 0] == 10
    assert frame[0, 20, 1] == 20
    assert (frame[:, :, 2] == 128).all()


def test_capture_without_init_uses_record_config(backend: IsaacGymBackend) -> None:
    # capture_video_frame lazily initializes the headless capture camera.
    frame = backend.capture_video_frame()
    assert frame.shape == (720, 1280, 3)


def test_run_playback_record_writes_video(backend: IsaacGymBackend, tmp_path: Path) -> None:
    from types import SimpleNamespace

    output = tmp_path / "play.mp4"
    result = backend.run_playback(
        env=SimpleNamespace(cfg=None),
        initialize=lambda: 0,
        step=lambda obs: obs + 1,
        num_steps=3,
        output_video=output,
        headless=True,
        record_video=True,
    )
    assert result == str(output)
    assert output.exists() and output.stat().st_size > 0


def test_run_playback_interactive_finite_steps(backend: IsaacGymBackend) -> None:
    from types import SimpleNamespace

    steps: list[int] = []
    result = backend.run_playback(
        env=SimpleNamespace(cfg=None),
        initialize=lambda: 0,
        step=lambda obs: steps.append(obs) or (obs + 1),
        num_steps=2,
        headless=False,
        record_video=False,
    )
    assert result is None
    assert steps == [0, 1]


def test_run_playback_interactive_window_close_is_graceful(
    scene_file: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from types import SimpleNamespace

    monkeypatch.setenv("UNILAB_ISAACGYM_MOCK_BEHAVIOR", "close_on_render")
    backend = _make_backend(scene_file)
    backend.materialize()
    try:
        result = backend.run_playback(
            env=SimpleNamespace(cfg=None),
            initialize=lambda: 0,
            step=lambda obs: obs + 1,
            num_steps=None,
            headless=False,
            record_video=False,
        )
        assert result is None
    finally:
        backend.close()
