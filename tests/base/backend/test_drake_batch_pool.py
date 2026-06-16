from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import textwrap

import pytest


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except ModuleNotFoundError:
        return False


def _drakeuni_package_installed() -> bool:
    return _module_available("drakeuni")


def _batch_extension_built() -> bool:
    return _module_available("drakeuni.compiled._drake_env_pool")


def _run_clean_python(code: str) -> str:
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout


def test_batch_import_diagnostic_is_preserved() -> None:
    output = _run_clean_python(
        """
        import json

        try:
            from drakeuni.batch_env import batch_available, batch_import_error
        except ImportError as exc:
            def batch_available():
                return False

            def batch_import_error():
                return exc

        error = batch_import_error()
        summary = {
            "available": bool(batch_available()),
            "error_type": None if error is None else type(error).__name__,
        }
        print(json.dumps(summary, sort_keys=True))
        """
    )
    summary = json.loads(output.strip().splitlines()[-1])
    if summary["available"]:
        assert summary["error_type"] is None
    else:
        assert summary["error_type"] == "ImportError"


def test_drake_batch_thread_policy_matches_mujoco_auto(monkeypatch: pytest.MonkeyPatch) -> None:
    from unilab.base.backend.drake import backend

    monkeypatch.setattr(backend, "cpu_count", lambda: 10)

    assert backend._resolve_batch_nthread(1024, 0) == 20
    assert backend._resolve_batch_nthread(8, 0) == 8
    assert backend._resolve_batch_nthread(1024, 4) == 4
    assert backend._resolve_batch_nthread(2, 8) == 2


def test_batch_backend_mode_rejects_existing_pydrake_module() -> None:
    output = _run_clean_python(
        """
        import json
        import sys

        from unilab.assets import ASSETS_ROOT_PATH
        from unilab.base.backend import create_backend
        from unilab.base.scene import SceneCfg

        sys.modules["pydrake"] = object()
        try:
            create_backend(
                "drake",
                SceneCfg(model_file=str(ASSETS_ROOT_PATH / "robots/go1/scene_flat_drake.xml")),
                1,
                0.01,
                base_name="trunk",
                drake_backend_mode="batch",
                robot_profile="go1",
                position_actuator_gains={"kp": 35.0, "kd": 0.5},
            )
        except ImportError as exc:
            message = str(exc)
        else:
            raise AssertionError("batch mode unexpectedly loaded with pydrake present")
        assert "pydrake" in message
        assert "fresh process" in message
        print(json.dumps({"message": message}, sort_keys=True))
        """
    )
    assert "pydrake" in output


def test_direct_drake_backend_batch_mode_rejects_existing_pydrake_module() -> None:
    output = _run_clean_python(
        """
        import json
        import sys

        from unilab.assets import ASSETS_ROOT_PATH
        from unilab.base.backend.drake.backend import DrakeBackend
        from unilab.base.scene import SceneCfg

        sys.modules["pydrake"] = object()
        try:
            DrakeBackend(
                SceneCfg(model_file=str(ASSETS_ROOT_PATH / "robots/go1/scene_flat_drake.xml")),
                1,
                0.01,
                base_name="trunk",
                drake_backend_mode="batch",
                robot_profile="go1",
                position_actuator_gains={"kp": 35.0, "kd": 0.5},
            )
        except ImportError as exc:
            message = str(exc)
        else:
            raise AssertionError("direct batch DrakeBackend unexpectedly loaded with pydrake present")
        assert "pydrake" in message
        assert "fresh process" in message
        print(json.dumps({"message": message}, sort_keys=True))
        """
    )
    assert "pydrake" in output


def test_create_backend_rejects_pydrake_mode() -> None:
    from unilab.assets import ASSETS_ROOT_PATH
    from unilab.base.backend import create_backend
    from unilab.base.scene import SceneCfg

    with pytest.raises(ValueError, match="drake_backend_mode='batch'"):
        create_backend(
            "drake",
            SceneCfg(model_file=str(ASSETS_ROOT_PATH / "robots/go1/scene_flat_drake.xml")),
            1,
            0.01,
            base_name="trunk",
            drake_backend_mode="pydrake",
            robot_profile="go1",
            position_actuator_gains={"kp": 35.0, "kd": 0.5},
        )


def test_direct_drake_backend_rejects_pydrake_mode() -> None:
    from unilab.assets import ASSETS_ROOT_PATH
    from unilab.base.backend.drake.backend import DrakeBackend
    from unilab.base.scene import SceneCfg

    with pytest.raises(ValueError, match="drake_backend_mode='batch'"):
        DrakeBackend(
            SceneCfg(model_file=str(ASSETS_ROOT_PATH / "robots/go1/scene_flat_drake.xml")),
            1,
            0.01,
            base_name="trunk",
            drake_backend_mode="pydrake",
            robot_profile="go1",
            position_actuator_gains={"kp": 35.0, "kd": 0.5},
        )


def test_drake_backend_requires_task_robot_profile() -> None:
    from unilab.assets import ASSETS_ROOT_PATH
    from unilab.base.backend import create_backend
    from unilab.base.scene import SceneCfg

    with pytest.raises(ValueError, match="robot_profile"):
        create_backend(
            "drake",
            SceneCfg(model_file=str(ASSETS_ROOT_PATH / "robots/go1/scene_flat_drake.xml")),
            1,
            0.01,
            base_name="trunk",
            drake_backend_mode="batch",
            position_actuator_gains={"kp": 35.0, "kd": 0.5},
        )


def test_drake_backend_requires_task_base_name() -> None:
    from unilab.assets import ASSETS_ROOT_PATH
    from unilab.base.backend import create_backend
    from unilab.base.scene import SceneCfg

    with pytest.raises(ValueError, match="base_name"):
        create_backend(
            "drake",
            SceneCfg(model_file=str(ASSETS_ROOT_PATH / "robots/go1/scene_flat_drake.xml")),
            1,
            0.01,
            drake_backend_mode="batch",
            robot_profile="go1",
            position_actuator_gains={"kp": 35.0, "kd": 0.5},
        )


@pytest.mark.skipif(
    not _drakeuni_package_installed(),
    reason="optional drakeuni package has not been installed",
)
def test_batch_package_direct_import_rejects_existing_pydrake_module() -> None:
    output = _run_clean_python(
        """
        import json
        import sys

        sys.modules["pydrake"] = object()
        from drakeuni.batch_env import (
            DrakeEnvPool,
            batch_available,
            batch_import_error,
        )

        error = batch_import_error()
        assert DrakeEnvPool is None
        assert not batch_available()
        assert error is not None
        assert "pydrake" in str(error)
        print(json.dumps({"message": str(error)}, sort_keys=True))
        """
    )
    assert "pydrake" in output


@pytest.mark.skipif(
    not _batch_extension_built(),
    reason="optional Drake batch extension has not been built",
)
def test_drake_batch_pool_imports_in_clean_process() -> None:
    output = _run_clean_python(
        """
        from drakeuni.batch_env import DrakeEnvPool, batch_available
        assert batch_available()
        assert DrakeEnvPool is not None
        print(DrakeEnvPool.__name__)
        """
    )
    assert "DrakeEnvPool" in output


def test_drakeuni_runtime_import_is_lazy_and_pydrake_free() -> None:
    output = _run_clean_python(
        """
        import json
        import sys

        from drakeuni.runtime import DrakeRuntimeConfig

        summary = {
            "config": DrakeRuntimeConfig.__name__,
            "compiled_loaded": any(name.startswith("drakeuni.compiled") for name in sys.modules),
            "pydrake_loaded": any(
                name == "pydrake" or name.startswith("pydrake.") for name in sys.modules
            ),
        }
        print(json.dumps(summary, sort_keys=True))
        """
    )
    assert json.loads(output.strip().splitlines()[-1]) == {
        "config": "DrakeRuntimeConfig",
        "compiled_loaded": False,
        "pydrake_loaded": False,
    }


def test_unilab_drake_public_surface_excludes_batch_backend_symbol() -> None:
    output = _run_clean_python(
        """
        import json

        import unilab.base.backend as backend_root
        import unilab.base.backend.drake as drake_pkg
        from unilab.base.backend.drake import backend as backend_module

        try:
            from unilab.base.backend.drake.backend import DrakeUniBatchBackend  # noqa: F401
        except ImportError:
            direct_import = "failed"
        else:
            direct_import = "succeeded"

        summary = {
            "direct_import": direct_import,
            "root_has_batch": hasattr(backend_root, "DrakeUniBatchBackend"),
            "subpackage_has_batch": hasattr(drake_pkg, "DrakeUniBatchBackend"),
            "module_has_batch": hasattr(backend_module, "DrakeUniBatchBackend"),
            "module_all_has_batch": "DrakeUniBatchBackend" in backend_module.__all__,
        }
        print(json.dumps(summary, sort_keys=True))
        """
    )
    assert json.loads(output.strip().splitlines()[-1]) == {
        "direct_import": "failed",
        "module_all_has_batch": False,
        "module_has_batch": False,
        "root_has_batch": False,
        "subpackage_has_batch": False,
    }


@pytest.mark.skipif(
    not _batch_extension_built(),
    reason="optional Drake batch extension has not been built",
)
def test_drake_batch_pool_go1_smoke_shapes_and_time() -> None:
    output = _run_clean_python(
        """
        import json
        import xml.etree.ElementTree as ET

        import numpy as np

        from unilab.assets import ASSETS_ROOT_PATH
        from drakeuni.batch_env import DrakeEnvPool

        model = ASSETS_ROOT_PATH / "robots/go1/scene_flat_drake.xml"
        robot = ASSETS_ROOT_PATH / "robots/go1/go1_drake.xml"
        scene_root = ET.parse(model).getroot()
        robot_root = ET.parse(robot).getroot()
        qpos = np.fromstring(
            scene_root.find('.//key[@name="home"]').attrib["qpos"],
            sep=" ",
            dtype=np.float64,
        )
        qvel = np.zeros(18, dtype=np.float64)
        ctrl_limits = []
        torque_limits = []
        for actuator in robot_root.findall(".//actuator/position"):
            ctrl_limits.append(
                np.fromstring(
                    actuator.attrib.get("ctrlrange", "-1 1"),
                    sep=" ",
                    dtype=np.float64,
                )
            )
            force = np.fromstring(
                actuator.attrib.get("forcerange", "-23.7 23.7"),
                sep=" ",
                dtype=np.float64,
            )
            torque_limits.append(float(np.max(np.abs(force))))
        ctrl_limits = np.asarray(ctrl_limits, dtype=np.float64)
        torque_limits = np.asarray(torque_limits, dtype=np.float64)
        state = np.zeros((2, 1 + qpos.size + qvel.size), dtype=np.float64)
        state[:, 1 : 1 + qpos.size] = qpos
        state[:, 1 + qpos.size :] = qvel
        pool = DrakeEnvPool(
            str(model),
            2,
            0.01,
            ctrl_limits,
            torque_limits,
            1,
            1,
            [7, 4, 13, 10],
            np.tile(np.array([[0.0, 0.0, -0.213]], dtype=np.float64), (4, 1)),
            35.0,
            0.5,
            1,
        )
        output = pool.step(state, 2, np.tile(qpos[7:], (2, 1)), None)
        summary = {
            "state_shape": list(output["state"].shape),
            "gyro_shape": list(output["sensor"]["gyro"].shape),
            "feet_shape": list(output["sensor"]["feet_pos"].shape),
            "time": output["state"][:, 0].tolist(),
            "state_finite": bool(np.all(np.isfinite(output["state"]))),
            "sensor_finite": bool(np.all(np.isfinite(output["sensor"]["feet_pos"]))),
            "nthread": pool.nthread,
            "workspace_count": pool.workspace_count,
            "num_filtered_geometries": pool.num_filtered_geometries,
        }
        print(json.dumps(summary, sort_keys=True))
        """
    )
    summary = json.loads(output.strip().splitlines()[-1])
    assert summary.pop("num_filtered_geometries") > 0
    assert summary == {
        "feet_shape": [2, 4, 3],
        "gyro_shape": [2, 3],
        "nthread": 1,
        "sensor_finite": True,
        "state_finite": True,
        "state_shape": [2, 38],
        "time": [0.02, 0.02],
        "workspace_count": 1,
    }


@pytest.mark.skipif(
    not _batch_extension_built(),
    reason="optional Drake batch extension has not been built",
)
def test_drake_batch_pool_uses_thread_workspaces_not_env_workspaces() -> None:
    output = _run_clean_python(
        """
        import json
        import xml.etree.ElementTree as ET

        import numpy as np

        from unilab.assets import ASSETS_ROOT_PATH
        from drakeuni.batch_env import DrakeEnvPool

        model = ASSETS_ROOT_PATH / "robots/go1/scene_flat_drake.xml"
        robot = ASSETS_ROOT_PATH / "robots/go1/go1_drake.xml"
        scene_root = ET.parse(model).getroot()
        robot_root = ET.parse(robot).getroot()
        qpos = np.fromstring(
            scene_root.find('.//key[@name="home"]').attrib["qpos"],
            sep=" ",
            dtype=np.float64,
        )
        qvel = np.zeros(18, dtype=np.float64)
        ctrl_limits = []
        torque_limits = []
        for actuator in robot_root.findall(".//actuator/position"):
            ctrl_limits.append(
                np.fromstring(
                    actuator.attrib.get("ctrlrange", "-1 1"),
                    sep=" ",
                    dtype=np.float64,
                )
            )
            force = np.fromstring(
                actuator.attrib.get("forcerange", "-23.7 23.7"),
                sep=" ",
                dtype=np.float64,
            )
            torque_limits.append(float(np.max(np.abs(force))))
        ctrl_limits = np.asarray(ctrl_limits, dtype=np.float64)
        torque_limits = np.asarray(torque_limits, dtype=np.float64)
        state = np.zeros((4, 1 + qpos.size + qvel.size), dtype=np.float64)
        for env_index in range(4):
            row_qpos = qpos.copy()
            row_qpos[0] += 0.05 * env_index
            state[env_index, 1 : 1 + qpos.size] = row_qpos
            state[env_index, 1 + qpos.size :] = qvel

        def make_pool(nthread):
            return DrakeEnvPool(
                str(model),
                4,
                0.01,
                ctrl_limits,
                torque_limits,
                1,
                1,
                [7, 4, 13, 10],
                np.tile(np.array([[0.0, 0.0, -0.213]], dtype=np.float64), (4, 1)),
                35.0,
                0.5,
                nthread,
            )

        control = np.tile(qpos[7:], (4, 1))
        serial_pool = make_pool(1)
        threaded_pool = make_pool(2)
        serial = serial_pool.step(state, 2, control, None)
        threaded = threaded_pool.step(state, 2, control, None)
        threaded_snapshot = threaded_pool.snapshot()

        reset_state = state[[2]].copy()
        reset_state[0, 1] = 1.23
        reset_snapshot = threaded_pool.reset(np.array([2], dtype=np.int32), reset_state)

        summary = {
            "serial_workspace_count": serial_pool.workspace_count,
            "threaded_workspace_count": threaded_pool.workspace_count,
            "parity_state": bool(np.allclose(serial["state"], threaded["state"])),
            "parity_feet": bool(np.allclose(serial["sensor"]["feet_pos"], threaded["sensor"]["feet_pos"])),
            "snapshot_state": bool(np.allclose(threaded["state"], threaded_snapshot["state"])),
            "reset_times": reset_snapshot["state"][:, 0].round(6).tolist(),
            "reset_x": reset_snapshot["state"][:, 1].round(6).tolist(),
        }
        print(json.dumps(summary, sort_keys=True))
        """
    )
    summary = json.loads(output.strip().splitlines()[-1])
    assert summary == {
        "parity_feet": True,
        "parity_state": True,
        "reset_times": [0.02, 0.02, 0.0, 0.02],
        "reset_x": [0.00101, 0.05101, 1.23, 0.15101],
        "serial_workspace_count": 1,
        "snapshot_state": True,
        "threaded_workspace_count": 2,
    }


@pytest.mark.skipif(
    not _batch_extension_built(),
    reason="optional Drake batch extension has not been built",
)
def test_create_backend_batch_mode_avoids_pydrake_and_steps() -> None:
    output = _run_clean_python(
        """
        import json
        import sys

        import numpy as np

        from unilab.assets import ASSETS_ROOT_PATH
        from unilab.base.backend import create_backend
        from unilab.base.scene import SceneCfg

        assert "pydrake" not in sys.modules
        backend = create_backend(
            "drake",
            SceneCfg(model_file=str(ASSETS_ROOT_PATH / "robots/go1/scene_flat_drake.xml")),
            2,
            0.01,
            base_name="trunk",
            drake_backend_mode="batch",
            drake_nthread=2,
            robot_profile="go1",
            position_actuator_gains={"kp": 35.0, "kd": 0.5},
        )
        assert "pydrake" not in sys.modules
        qpos = np.stack([backend.get_keyframe_qpos("home") for _ in range(2)])
        qvel = np.stack([backend.get_init_qvel() for _ in range(2)])
        backend.set_state(np.arange(2, dtype=np.int32), qpos, qvel)
        backend.step(backend.get_dof_pos(), nsteps=2)
        for _ in range(100):
            backend.step(backend.get_dof_pos(), nsteps=1)
        diagnostics = backend.diagnostics()
        foot_contact = backend.get_sensor_data("FL_foot_contact")
        summary = {
            "cls": type(backend).__name__,
            "state_shape": list(backend.get_physics_state().shape),
            "base_shape": list(backend.get_base_pos().shape),
            "foot_shape": list(backend.get_sensor_data("FL_pos").shape),
            "contact_shape": list(foot_contact.shape),
            "contact_nonzero": bool(np.max(np.abs(foot_contact)) > 0.0),
            "diagnostic_mode": diagnostics.mode,
            "nthread": backend.nthread,
            "workspace_count": diagnostics.workspace_count,
            "time": np.round(backend.get_physics_state()[:, 0], decimals=6).tolist(),
            "pydrake_loaded": "pydrake" in sys.modules,
        }
        print(json.dumps(summary, sort_keys=True))
        """
    )
    summary = json.loads(output.strip().splitlines()[-1])
    assert summary == {
        "base_shape": [2, 3],
        "cls": "DrakeBackend",
        "contact_shape": [2, 3],
        "contact_nonzero": True,
        "diagnostic_mode": "batch",
        "foot_shape": [2, 3],
        "nthread": 2,
        "pydrake_loaded": False,
        "state_shape": [2, 38],
        "time": [1.02, 1.02],
        "workspace_count": 2,
    }


@pytest.mark.skipif(
    not _batch_extension_built(),
    reason="optional Drake batch extension has not been built",
)
def test_create_go2_backend_batch_mode_avoids_pydrake_and_steps() -> None:
    output = _run_clean_python(
        """
        import json
        import sys

        import numpy as np

        from unilab.assets import ASSETS_ROOT_PATH
        from unilab.base.backend import create_backend
        from unilab.base.scene import SceneCfg

        assert "pydrake" not in sys.modules
        backend = create_backend(
            "drake",
            SceneCfg(model_file=str(ASSETS_ROOT_PATH / "robots/go2/scene_flat.xml")),
            2,
            0.01,
            base_name="base",
            drake_backend_mode="batch",
            drake_nthread=2,
            robot_profile="go2",
            position_actuator_gains={"kp": 35.0, "kd": 0.5},
        )
        assert "pydrake" not in sys.modules
        qpos = np.stack([backend.get_keyframe_qpos("home") for _ in range(2)])
        qvel = np.stack([backend.get_init_qvel() for _ in range(2)])
        backend.set_state(np.arange(2, dtype=np.int32), qpos, qvel)
        backend.step(backend.get_dof_pos(), nsteps=2)
        diagnostics = backend.diagnostics()
        summary = {
            "cls": type(backend).__name__,
            "state_shape": list(backend.get_physics_state().shape),
            "base_shape": list(backend.get_base_pos().shape),
            "foot_shape": list(backend.get_sensor_data("FL_pos").shape),
            "contact_shape": list(backend.get_sensor_data("FL_foot_contact").shape),
            "diagnostic_mode": diagnostics.mode,
            "nthread": backend.nthread,
            "workspace_count": diagnostics.workspace_count,
            "time": backend.get_physics_state()[:, 0].tolist(),
            "pydrake_loaded": "pydrake" in sys.modules,
        }
        print(json.dumps(summary, sort_keys=True))
        """
    )
    summary = json.loads(output.strip().splitlines()[-1])
    assert summary == {
        "base_shape": [2, 3],
        "cls": "DrakeBackend",
        "contact_shape": [2, 3],
        "diagnostic_mode": "batch",
        "foot_shape": [2, 3],
        "nthread": 2,
        "pydrake_loaded": False,
        "state_shape": [2, 38],
        "time": [0.02, 0.02],
        "workspace_count": 2,
    }
