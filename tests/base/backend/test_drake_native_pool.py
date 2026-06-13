from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

NATIVE_DIR = Path(__file__).resolve().parents[3] / "src/unilab/base/backend/drake/native"


def _native_extension_built() -> bool:
    return any(NATIVE_DIR.glob("_drake_env_pool*.so"))


def _run_clean_python(code: str) -> str:
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout


def test_native_import_diagnostic_is_preserved() -> None:
    output = _run_clean_python(
        """
        import json

        from unilab.base.backend.drake.native import native_available, native_import_error

        error = native_import_error()
        summary = {
            "available": bool(native_available()),
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


def test_native_drake_thread_policy_matches_mujoco_auto(monkeypatch: pytest.MonkeyPatch) -> None:
    from unilab.base.backend.drake import backend_native

    monkeypatch.setattr(backend_native, "cpu_count", lambda: 10)

    assert backend_native._resolve_native_nthread(1024, 0) == 20
    assert backend_native._resolve_native_nthread(8, 0) == 8
    assert backend_native._resolve_native_nthread(1024, 4) == 4
    assert backend_native._resolve_native_nthread(2, 8) == 2


def test_native_backend_mode_rejects_existing_pydrake_module() -> None:
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
                drake_backend_mode="native",
                position_actuator_gains={"kp": 35.0, "kd": 0.5},
            )
        except ImportError as exc:
            message = str(exc)
        else:
            raise AssertionError("native mode unexpectedly loaded with pydrake present")
        assert "pydrake" in message
        assert "fresh process" in message
        print(json.dumps({"message": message}, sort_keys=True))
        """
    )
    assert "pydrake" in output


def test_native_package_direct_import_rejects_existing_pydrake_module() -> None:
    output = _run_clean_python(
        """
        import json
        import sys

        sys.modules["pydrake"] = object()
        from unilab.base.backend.drake.native import (
            NativeDrakeEnvPool,
            native_available,
            native_import_error,
        )

        error = native_import_error()
        assert NativeDrakeEnvPool is None
        assert not native_available()
        assert error is not None
        assert "pydrake" in str(error)
        print(json.dumps({"message": str(error)}, sort_keys=True))
        """
    )
    assert "pydrake" in output


@pytest.mark.skipif(
    not _native_extension_built(),
    reason="optional Drake native extension has not been built",
)
def test_native_drake_pool_imports_in_clean_process() -> None:
    output = _run_clean_python(
        """
        from unilab.base.backend.drake.native import NativeDrakeEnvPool, native_available
        assert native_available()
        assert NativeDrakeEnvPool is not None
        print(NativeDrakeEnvPool.__name__)
        """
    )
    assert "NativeDrakeEnvPool" in output


@pytest.mark.skipif(
    not _native_extension_built(),
    reason="optional Drake native extension has not been built",
)
def test_native_drake_pool_go1_smoke_shapes_and_time() -> None:
    output = _run_clean_python(
        """
        import json
        import xml.etree.ElementTree as ET

        import numpy as np

        from unilab.assets import ASSETS_ROOT_PATH
        from unilab.base.backend.drake.native import NativeDrakeEnvPool

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
        pool = NativeDrakeEnvPool(
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
    }


@pytest.mark.skipif(
    not _native_extension_built(),
    reason="optional Drake native extension has not been built",
)
def test_create_backend_native_mode_avoids_pydrake_and_steps() -> None:
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
            drake_backend_mode="native",
            drake_nthread=2,
            position_actuator_gains={"kp": 35.0, "kd": 0.5},
        )
        assert "pydrake" not in sys.modules
        qpos = np.stack([backend.get_keyframe_qpos("home") for _ in range(2)])
        qvel = np.stack([backend.get_init_qvel() for _ in range(2)])
        backend.set_state(np.arange(2, dtype=np.int32), qpos, qvel)
        backend.step(backend.get_dof_pos(), nsteps=2)
        summary = {
            "cls": type(backend).__name__,
            "state_shape": list(backend.get_physics_state().shape),
            "base_shape": list(backend.get_base_pos().shape),
            "foot_shape": list(backend.get_sensor_data("FL_pos").shape),
            "contact_shape": list(backend.get_sensor_data("FL_foot_contact").shape),
            "nthread": backend.nthread,
            "time": backend.get_physics_state()[:, 0].tolist(),
            "pydrake_loaded": "pydrake" in sys.modules,
        }
        print(json.dumps(summary, sort_keys=True))
        """
    )
    summary = json.loads(output.strip().splitlines()[-1])
    assert summary == {
        "base_shape": [2, 3],
        "cls": "NativeDrakeBackend",
        "contact_shape": [2, 3],
        "foot_shape": [2, 3],
        "nthread": 2,
        "pydrake_loaded": False,
        "state_shape": [2, 38],
        "time": [0.02, 0.02],
    }
