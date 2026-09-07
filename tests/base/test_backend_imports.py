from __future__ import annotations

import ast
import subprocess
import sys
import textwrap
from importlib.metadata import distribution
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MATERIALIZER_CONSUMERS = (
    "src/unilab/base/config_adapter.py",
    "src/unilab/scripts/train_rsl_rl.py",
    "scripts/train_hora_distill.py",
)


def test_unisim_dependency_is_installed_from_package_index() -> None:
    direct_url = distribution("unisim-core").read_text("direct_url.json")

    assert direct_url is None, (
        "UniLab tests must consume the indexed unisim-core release, not a local, editable, "
        "or VCS checkout"
    )


def test_materializer_consumers_use_unisim_owner_module() -> None:
    offenders: list[str] = []
    for relative_path in _MATERIALIZER_CONSUMERS:
        path = _REPO_ROOT / relative_path
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imports = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            and any(alias.name == "materialize_scene_visual_override" for alias in node.names)
        ]
        if len(imports) != 1 or imports[0].module != "unisim.backend.mujoco.xml":
            modules = [node.module for node in imports]
            offenders.append(f"{relative_path}: {modules}")

    assert offenders == []


def test_mujoco_backend_import_path_does_not_eagerly_import_motrix() -> None:
    code = textwrap.dedent(
        """
        import importlib.util
        import sys

        from unilab.base.backend_factory import create_backend
        from unisim.backend.mujoco.xml import (
            create_discardvisual_xml,
            materialize_scene_visual_override,
        )

        assert create_backend is not None
        assert materialize_scene_visual_override is not None
        assert create_discardvisual_xml is not None

        print("mujoco_runtime", "mujoco" in sys.modules)
        print("mujoco_backend", "unisim.backend.mujoco.backend" in sys.modules)

        if importlib.util.find_spec("mujoco") is not None:
            import unisim.backend.mujoco.backend
            print("mujoco_backend imported")
        else:
            print("mujoco_backend skipped")

        print("motrix_backend", "unisim.backend.motrix.backend" in sys.modules)
        print("motrixsim", "motrixsim" in sys.modules)
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Motphys profiler initialized" not in result.stdout + result.stderr
    lines = result.stdout.splitlines()
    assert lines[:2] == ["mujoco_runtime False", "mujoco_backend False"]
    assert lines[2] in {"mujoco_backend imported", "mujoco_backend skipped"}
    assert lines[3:] == ["motrix_backend False", "motrixsim False"]


def test_motrix_backend_import_path_does_not_eagerly_import_mujoco() -> None:
    code = textwrap.dedent(
        """
        import importlib.util
        import sys

        from unilab.base.backend_factory import create_backend
        from unisim.backend.motrix.scene import (
            materialize_motrix_hfield_attached_scene,
            materialize_motrix_scene,
        )

        assert create_backend is not None
        assert materialize_motrix_scene is not None
        assert materialize_motrix_hfield_attached_scene is not None

        if importlib.util.find_spec("motrixsim") is not None:
            import unisim.backend.motrix.backend
            print("motrix_backend imported")
        else:
            print("motrix_backend skipped")

        print("mujoco_backend", "unisim.backend.mujoco.backend" in sys.modules)
        print("mujoco", "mujoco" in sys.modules)
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )

    lines = result.stdout.splitlines()
    assert lines[0] in {"motrix_backend imported", "motrix_backend skipped"}
    assert lines[1:] == ["mujoco_backend False", "mujoco False"]
