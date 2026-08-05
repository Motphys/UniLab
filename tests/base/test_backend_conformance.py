"""Cross-backend conformance suite driven only through the public contract.

Every interaction goes through ``create_backend`` and the public ``SimBackend``
surface. mjwarp runs when a CUDA Warp device is available (slow lane).
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base.backend import create_backend
from unilab.base.scene import SceneCfg
from unilab.tools.backend_isolation import audit_backend_isolation

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
_FACTORY_FILE = SRC_ROOT / "unilab" / "base" / "backend" / "__init__.py"
_BACKEND_CLASS_NAMES = frozenset(
    {"MuJoCoBackend", "MotrixBackend", "DrakeBackend", "MjwarpBackend"}
)

NUM_ENVS = 2
SIM_DT = 0.005
_G1_SCENE = str(ASSETS_ROOT_PATH / "robots" / "g1" / "scene_flat.xml")


def _module_available(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def _mjwarp_cuda_available() -> bool:
    from unilab.base.backend.mjwarp.dependencies import (
        load_mjwarp_dependencies,
        mjwarp_dependencies_available,
    )

    if not mjwarp_dependencies_available():
        return False
    try:
        return bool(load_mjwarp_dependencies().warp.get_device().is_cuda)
    except Exception:
        return False


def _drake_batch_available() -> bool:
    try:
        from unilab.base.backend.drake.backend import ensure_drake_batch_available
    except ImportError:
        return False
    available, _ = ensure_drake_batch_available()
    return bool(available)


def _require_backend(backend_type: str) -> None:
    if backend_type == "mujoco":
        pytest.importorskip("mujoco", reason="mujoco not installed")
    elif backend_type == "motrix":
        pytest.importorskip("motrixsim", reason="motrixsim not installed")
    elif backend_type == "mjwarp":
        if not _mjwarp_cuda_available():
            pytest.skip("mjwarp requires an active CUDA Warp device")
    elif backend_type == "drake":
        if not _drake_batch_available():
            pytest.skip("drake batch extension not available")


_BACKEND_PARAMS = [
    pytest.param("mujoco", id="mujoco"),
    pytest.param("motrix", id="motrix"),
    pytest.param("drake", id="drake"),
    pytest.param("mjwarp", id="mjwarp", marks=pytest.mark.slow),
]


def test_backend_classes_are_only_instantiated_through_create_backend() -> None:
    offenders: list[str] = []
    for path in sorted(SRC_ROOT.rglob("*.py")):
        if path == _FACTORY_FILE or "__pycache__" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = None
            if isinstance(func, ast.Name):
                name = func.id
            elif isinstance(func, ast.Attribute):
                name = func.attr
            if name in _BACKEND_CLASS_NAMES:
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}: {name}(")
    assert not offenders, "direct backend instantiation outside create_backend:\n" + "\n".join(
        offenders
    )


def test_real_repo_backend_isolation_audit_passes() -> None:
    report = audit_backend_isolation(REPO_ROOT)
    assert report.ok, "\n".join(violation.format() for violation in report.violations)


@pytest.mark.parametrize("backend_type", _BACKEND_PARAMS)
def test_legacy_contract_step_set_state_and_state_reads(backend_type: str) -> None:
    _require_backend(backend_type)

    backend = create_backend(
        backend_type,
        SceneCfg(model_file=_G1_SCENE),
        NUM_ENVS,
        SIM_DT,
        base_name="pelvis",
    )
    backend.materialize()

    assert backend.num_envs == NUM_ENVS
    assert backend.num_actuators > 0

    backend.step(np.zeros((NUM_ENVS, backend.num_actuators)), nsteps=2)

    default_qpos = np.asarray(backend.get_default_qpos())
    qpos = np.broadcast_to(default_qpos, (NUM_ENVS, default_qpos.shape[0])).copy()
    target_xyz = np.array([1.0, 2.0, 0.8])
    qpos[:, :3] = target_xyz
    qvel = np.zeros((NUM_ENVS, len(backend.get_init_qvel())))
    backend.set_state(np.arange(NUM_ENVS, dtype=np.int32), qpos, qvel)

    np.testing.assert_allclose(
        backend.get_base_pos(), np.tile(target_xyz, (NUM_ENVS, 1)), atol=1e-4
    )
    np.testing.assert_allclose(np.linalg.norm(backend.get_base_quat(), axis=-1), 1.0, atol=1e-5)
