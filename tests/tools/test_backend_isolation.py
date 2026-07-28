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


def test_clean_runtime_and_cold_sibling_adapter_pass(tmp_path: Path) -> None:
    root = _fixture_repo(
        tmp_path,
        {
            "src/unilab/base/backend/mujoco/playback.py": (
                "from unilab.base.backend.motrix.backend import MotrixBackend\n"
            ),
        },
    )

    report = audit_backend_isolation(root)

    assert report.ok
    assert report.backend_packages == ("motrix", "mujoco")
    assert report.runtime_modules == (
        "unilab.base.backend.motrix.backend",
        "unilab.base.backend.mujoco.backend",
    )


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
