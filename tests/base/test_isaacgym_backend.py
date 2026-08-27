"""Protocol-level tests for the IsaacGym subprocess backend.

The development machine has no IsaacGym install (the Preview 4 tarball
requires an NVIDIA account), so these tests drive ``IsaacGymBackend`` through
``isaacgym_mock_worker.py`` — a deterministic kinematic fake that speaks the
real wire protocol over real shared memory on the host interpreter.
"""

from __future__ import annotations

import io
import sys
import textwrap
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from unilab.base.backend import create_backend
from unilab.base.backend.isaacgym import protocol
from unilab.base.backend.isaacgym.backend import IsaacGymBackend, IsaacGymWorkerError
from unilab.base.backend.isaacgym.dependencies import (
    ENV_HOME,
    ENV_PYTHON,
    IsaacGymDependencyError,
    build_worker_env,
    resolve_isaacgym_runtime,
)
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
      <geom name="base_geom" size="0.1"/>
      <body name="link0">
        <joint name="j0" type="hinge"/>
        <geom name="g0" size="0.1"/>
        <body name="link1">
          <joint name="j1" type="hinge"/>
          <geom name="foot_geom" size="0.1"/>
          <body name="link2">
            <joint name="j2" type="hinge"/>
            <geom name="g2" size="0.1"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
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
    backend = create_backend(
        "isaacgym",
        SceneCfg(model_file=scene_file),
        NUM_ENVS,
        SIM_DT,
        base_name="base",
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
        backend.get_actuator_ctrl_range(), np.array([[-100.0, 100.0]] * 3, dtype=np.float32)
    )
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
    np.testing.assert_allclose(default_qpos, [0, 0, 0, 1, 0, 0, 0, 0, 0, 0], atol=1e-7)
    np.testing.assert_allclose(backend.get_init_qvel(), np.zeros(9))
    np.testing.assert_allclose(backend.get_default_dof_pos(), np.zeros(3))
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

    # Initial state written by the mock at ATTACH.
    np.testing.assert_allclose(backend.get_base_pos(), [[0, 0, 1], [0, 0, 1]], atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(backend.get_base_quat(), axis=-1), 1.0)
    assert backend.model.num_bodies == 3


def test_step_integrates_mock_physics(backend: IsaacGymBackend) -> None:
    ctrl = np.ones((NUM_ENVS, 3), dtype=np.float32)
    result = backend.step(ctrl, nsteps=2)
    assert "timing" in result
    assert "physics_ms" in result["timing"]

    # Mock: dof_vel += ctrl*dt; dof_pos += dof_vel*dt per substep.
    expected_vel = np.float32(2 * SIM_DT)
    expected_pos = np.float32(SIM_DT * SIM_DT) + np.float32(2 * SIM_DT * SIM_DT)
    np.testing.assert_allclose(backend.get_dof_vel(), expected_vel, atol=1e-7)
    np.testing.assert_allclose(backend.get_dof_pos(), expected_pos, atol=1e-7)

    # Free fall: v_z after 2 substeps, z += v_z*dt each substep.
    expected_vz = 2 * (-9.81) * SIM_DT
    expected_z = 1.0 + (-9.81 * SIM_DT * SIM_DT) + (-9.81 * 2 * SIM_DT * SIM_DT)
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
    np.testing.assert_allclose(backend.get_base_pos()[1], [0, 0, 1], atol=1e-6)

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
    np.testing.assert_allclose(pos, [[0.2, 0.0, 1.0]] * NUM_ENVS, atol=1e-6)

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


def test_body_state_views(backend: IsaacGymBackend) -> None:
    ids = backend.get_body_ids(["base", "link1"])
    pos_w = backend.get_body_pos_w(ids)
    assert pos_w.shape == (NUM_ENVS, 2, 3)
    np.testing.assert_allclose(pos_w[:, 1], [[0.2, 0.0, 1.0]] * NUM_ENVS, atol=1e-6)
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

    with pytest.raises(NotImplementedError, match="render"):
        backend.init_renderer()
    with pytest.raises(NotImplementedError, match="playback"):
        backend.run_playback(env=None, initialize=None, step=None, num_steps=None)


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


def test_metadata_access_before_materialize_fails(scene_file: str) -> None:
    backend = _make_backend(scene_file)
    try:
        with pytest.raises(RuntimeError, match="materialize"):
            _ = backend.num_actuators
        with pytest.raises(RuntimeError, match="materialize"):
            backend.step(np.zeros((NUM_ENVS, 3), dtype=np.float32))
    finally:
        backend.close()
