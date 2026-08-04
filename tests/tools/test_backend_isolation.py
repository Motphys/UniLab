from __future__ import annotations

from pathlib import Path

import pytest

from unilab.tools.backend_isolation import (
    BackendIsolationAuditError,
    audit_backend_isolation,
)


def _write_files(root: Path, files: dict[str, str]) -> None:
    for relative_path, source in files.items():
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")


def _fixture_repo(root: Path, overrides: dict[str, str] | None = None) -> Path:
    files = {
        "src/unilab/__init__.py": "",
        "src/unilab/base/__init__.py": "",
        "src/unilab/base/backend/__init__.py": "",
        "src/unilab/base/backend/base.py": """
class SimBackend:
    backend_type: str

    @property
    def model(self):
        raise NotImplementedError

    def allowed(self) -> None:
        pass
""",
        "src/unilab/base/backend/mujoco/__init__.py": "",
        "src/unilab/base/backend/mujoco/backend.py": """
from ..base import SimBackend

class MuJoCoBackend(SimBackend):
    pass
""",
        "src/unilab/base/backend/motrix/__init__.py": "",
        "src/unilab/base/backend/motrix/backend.py": """
from ..base import SimBackend

class MotrixBackend(SimBackend):
    pass
""",
        "src/unilab/base/np_env.py": """
def backend_name(self):
    return self._backend.backend_type
""",
        "src/unilab/dr/__init__.py": "",
        "src/unilab/envs/__init__.py": "",
        "src/unilab/envs/task.py": """
def use_backend(self):
    self._backend.allowed()
""",
    }
    files.update(overrides or {})
    _write_files(root, files)
    return root


def _codes(root: Path) -> set[str]:
    return {violation.code for violation in audit_backend_isolation(root).violations}


def test_clean_runtime_and_shared_module_imports_pass(tmp_path: Path) -> None:
    root = _fixture_repo(
        tmp_path,
        {
            "src/unilab/base/backend/mujoco/playback.py": (
                "from unilab.base.backend.playback_common import write_playback_video\n"
            ),
        },
    )

    report = audit_backend_isolation(root)

    assert report.ok
    assert report.backend_packages == ("motrix", "mujoco")
    # Every package file is audited, including __init__.py re-export modules.
    assert report.runtime_modules == (
        "unilab.base.backend.motrix",
        "unilab.base.backend.motrix.backend",
        "unilab.base.backend.mujoco",
        "unilab.base.backend.mujoco.backend",
        "unilab.base.backend.mujoco.playback",
    )


def test_sibling_import_in_any_package_file_fails_closed(tmp_path: Path) -> None:
    root = _fixture_repo(
        tmp_path,
        {
            "src/unilab/base/backend/mujoco/playback.py": (
                "from unilab.base.backend.motrix.backend import MotrixBackend\n"
            ),
        },
    )

    report = audit_backend_isolation(root)

    assert "sibling-runtime-import" in {violation.code for violation in report.violations}
    assert any(
        violation.path == "src/unilab/base/backend/mujoco/playback.py"
        for violation in report.violations
    )


def test_documented_sibling_cold_path_exception_passes(tmp_path: Path) -> None:
    root = _fixture_repo(
        tmp_path,
        {
            "src/unilab/base/backend/mujoco/playback.py": (
                "def run_mujoco_playback():\n    pass\n"
            ),
            "src/unilab/base/backend/motrix/playback.py": (
                "from unilab.base.backend.mujoco.playback import run_mujoco_playback\n"
            ),
        },
    )

    report = audit_backend_isolation(
        root,
        sibling_exceptions={("motrix", "unilab.base.backend.mujoco.playback")},
    )

    assert report.ok


@pytest.mark.parametrize(
    ("module_name", "symbol"),
    (
        ("batch", "BatchContract"),
        ("mutation", "MutationContract"),
    ),
)
def test_runtime_can_import_approved_shared_contract(
    tmp_path: Path, module_name: str, symbol: str
) -> None:
    root = _fixture_repo(
        tmp_path,
        {
            f"src/unilab/base/backend/{module_name}.py": f"class {symbol}:\n    pass\n",
            "src/unilab/base/backend/mujoco/backend.py": (
                f"from unilab.base.backend.{module_name} import {symbol}\n"
                "from ..base import SimBackend\n\n"
                "class MuJoCoBackend(SimBackend):\n"
                "    pass\n"
            ),
        },
    )

    report = audit_backend_isolation(root)

    assert report.ok


@pytest.mark.parametrize(
    ("source", "expected_code"),
    [
        (
            "from unilab.base.backend.motrix.backend import MotrixBackend\n",
            "sibling-runtime-import",
        ),
        (
            "from ..motrix.backend import MotrixBackend\n",
            "sibling-runtime-import",
        ),
        (
            "from .....outside import runtime\n",
            "invalid-relative-import",
        ),
        (
            "from unilab.base.backend.motrix import backend as sibling\nVALUE = sibling._PRIVATE\n",
            "sibling-private-access",
        ),
        (
            "from unilab.base.backend.motrix.backend import MotrixBackend\n"
            "class MuJoCoBackend(MotrixBackend):\n"
            "    pass\n",
            "sibling-backend-inheritance",
        ),
        (
            "from unilab.base.backend.experimental import runtime\n",
            "unapproved-runtime-dependency",
        ),
    ],
)
def test_runtime_faults_fail_closed(tmp_path: Path, source: str, expected_code: str) -> None:
    root = _fixture_repo(
        tmp_path,
        {"src/unilab/base/backend/mujoco/backend.py": source},
    )

    assert expected_code in _codes(root)


@pytest.mark.parametrize(
    ("source", "expected_code"),
    [
        ("def bad(self):\n    return self._backend._model\n", "private-backend-member"),
        ("def bad(self):\n    return self._backend.model\n", "raw-backend-object"),
        ("def bad(self):\n    return self._backend.not_declared()\n", "unknown-backend-member"),
        (
            "def bad(self):\n    return getattr(self._backend, 'allowed')\n",
            "dynamic-backend-probe",
        ),
        (
            "def bad(self):\n    return hasattr(self._backend, 'allowed')\n",
            "dynamic-backend-probe",
        ),
        (
            "def bad(self):\n    backend = self._backend\n    return backend._model\n",
            "private-backend-member",
        ),
    ],
)
def test_env_contract_faults_fail_closed(
    tmp_path: Path,
    source: str,
    expected_code: str,
) -> None:
    root = _fixture_repo(tmp_path, {"src/unilab/envs/task.py": source})

    report = audit_backend_isolation(root)

    assert expected_code in {violation.code for violation in report.violations}
    assert any(violation.path == "src/unilab/envs/task.py" for violation in report.violations)


def test_future_mjwarp_package_is_discovered_automatically(tmp_path: Path) -> None:
    root = _fixture_repo(
        tmp_path,
        {
            "src/unilab/base/backend/mjwarp/__init__.py": "",
            "src/unilab/base/backend/mjwarp/backend.py": (
                "from unilab.base.backend.mujoco.backend import MuJoCoBackend\n"
            ),
        },
    )

    report = audit_backend_isolation(root)

    assert "mjwarp" in report.backend_packages
    assert "sibling-runtime-import" in {violation.code for violation in report.violations}


def test_syntax_error_and_invalid_layout_fail_closed(tmp_path: Path) -> None:
    syntax_root = _fixture_repo(
        tmp_path / "syntax",
        {"src/unilab/base/backend/mujoco/backend.py": "def invalid(:\n"},
    )
    layout_root = _fixture_repo(tmp_path / "layout")
    (layout_root / "src/unilab/base/backend/mujoco/__init__.py").unlink()

    assert "source-syntax-error" in _codes(syntax_root)
    assert "invalid-backend-layout" in _codes(layout_root)


def test_missing_and_empty_runtime_roots_fail_closed(tmp_path: Path) -> None:
    missing_report = audit_backend_isolation(tmp_path / "missing")
    empty_root = tmp_path / "empty"
    _write_files(
        empty_root,
        {
            "src/unilab/base/backend/base.py": "class SimBackend:\n    pass\n",
            "src/unilab/base/np_env.py": "",
            "src/unilab/dr/__init__.py": "",
            "src/unilab/envs/__init__.py": "",
        },
    )
    empty_report = audit_backend_isolation(empty_root)

    assert "missing-audit-root" in {violation.code for violation in missing_report.violations}
    assert "empty-runtime-root" in {violation.code for violation in empty_report.violations}
    with pytest.raises(BackendIsolationAuditError, match="backend isolation audit failed"):
        empty_report.require_ok()


@pytest.mark.parametrize(
    "forbidden_import",
    (
        "import hydra\n",
        "from omegaconf import DictConfig\n",
        "from unilab.envs.task import Task\n",
        "from unilab.training import create_env\n",
        "from unilab.algos import ppo\n",
        "from unilab.manager import ManagedEnvState\n",
        "from unilab.ipc import async_runner\n",
        "from typing import TYPE_CHECKING\n"
        "if TYPE_CHECKING:\n"
        "    from unilab.training import create_env\n",
    ),
)
def test_physics_layer_forbidden_imports_fail_closed(
    tmp_path: Path, forbidden_import: str
) -> None:
    root = _fixture_repo(
        tmp_path,
        {"src/unilab/base/backend/mujoco/backend.py": forbidden_import},
    )

    assert "physics-layer-forbidden-import" in _codes(root)


@pytest.mark.parametrize(
    "allowed_import",
    (
        "from unilab.dr.types import ResetRandomizationPayload\n",
        "from unilab.base.scene import SceneCfg\n",
        "from unilab.terrains.terrain_generator import TerrainGeneratorCfg\n",
        "from unilab.dtype_config import get_global_dtype\n",
        "from unilab.visualization import render_many\n",
        (
            "from typing import TYPE_CHECKING\n"
            "if TYPE_CHECKING:\n"
            "    from unilab.base.base import EnvCfg\n"
        ),
    ),
)
def test_physics_layer_documented_allowlist_passes(tmp_path: Path, allowed_import: str) -> None:
    root = _fixture_repo(
        tmp_path,
        {
            "src/unilab/base/backend/mujoco/backend.py": (
                allowed_import + "from ..base import SimBackend\n"
            ),
        },
    )

    assert audit_backend_isolation(root).ok


def test_physics_layer_undocumented_unilab_import_fails_closed(tmp_path: Path) -> None:
    root = _fixture_repo(
        tmp_path,
        {
            "src/unilab/base/backend/mujoco/backend.py": (
                "from unilab.utils.rotation import quat_apply\n"
            ),
        },
    )

    assert "physics-layer-undocumented-import" in _codes(root)


def test_physics_layer_type_checking_only_import_at_runtime_fails_closed(
    tmp_path: Path,
) -> None:
    root = _fixture_repo(
        tmp_path,
        {
            "src/unilab/base/backend/__init__.py": (
                "from unilab.base.base import EnvCfg\n"
            ),
        },
    )

    report = audit_backend_isolation(root)

    assert "physics-layer-forbidden-import" in {violation.code for violation in report.violations}
    assert any(
        violation.path == "src/unilab/base/backend/__init__.py"
        for violation in report.violations
    )


def test_physics_layer_audit_covers_shared_modules(tmp_path: Path) -> None:
    root = _fixture_repo(
        tmp_path,
        {"src/unilab/base/backend/playback_common.py": "import hydra\n"},
    )

    assert "physics-layer-forbidden-import" in _codes(root)


def test_scripts_private_probe_fails_closed(tmp_path: Path) -> None:
    root = _fixture_repo(
        tmp_path,
        {
            "scripts/tool.py": (
                "def diag(env):\n"
                "    backend = getattr(env, '_backend', None)\n"
                "    return getattr(backend, '_n_threads', -1)\n"
            ),
        },
    )

    report = audit_backend_isolation(root)

    assert "dynamic-backend-probe" in {violation.code for violation in report.violations}
    assert any(violation.path == "scripts/tool.py" for violation in report.violations)


def test_scripts_public_probe_and_contract_calls_pass(tmp_path: Path) -> None:
    root = _fixture_repo(
        tmp_path,
        {
            "scripts/tool.py": (
                "def diag(env):\n"
                "    backend = getattr(env, '_backend', None)\n"
                "    if not hasattr(backend, 'allowed'):\n"
                "        return None\n"
                "    optional = getattr(backend, 'scene_visual_model_file', None)\n"
                "    backend.allowed()\n"
                "    return optional\n"
            ),
        },
    )

    assert audit_backend_isolation(root).ok


def test_runtime_layers_keep_strict_probe_rules(tmp_path: Path) -> None:
    root = _fixture_repo(
        tmp_path,
        {
            "src/unilab/manager/scene.py": (
                "def diag(env):\n"
                "    backend = getattr(env, '_backend', None)\n"
                "    return getattr(backend, 'scene_visual_model_file', None)\n"
            ),
            "src/unilab/training/run.py": (
                "def diag(env):\n"
                "    return hasattr(env._backend, 'allowed')\n"
            ),
        },
    )

    report = audit_backend_isolation(root)

    assert "dynamic-backend-probe" in {violation.code for violation in report.violations}
    assert {violation.path for violation in report.violations} == {
        "src/unilab/manager/scene.py",
        "src/unilab/training/run.py",
    }


def test_scripts_private_backend_member_via_getattr_alias_fails_closed(tmp_path: Path) -> None:
    root = _fixture_repo(
        tmp_path,
        {
            "scripts/tool.py": (
                "def diag(env):\n"
                "    backend = getattr(env, '_backend', None)\n"
                "    return backend._model\n"
            ),
        },
    )

    assert "private-backend-member" in _codes(root)
