from __future__ import annotations

import ast
import subprocess
import sys
import textwrap
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MATERIALIZER_CONSUMERS = (
    "src/unilab/training/backend_adapter.py",
    "scripts/train_rsl_rl.py",
    "scripts/train_him_ppo.py",
    "scripts/train_hora_distill.py",
    "scripts/play_interactive.py",
    "scripts/manip_loco/benchmark_site_jacobian.py",
)


def test_materializer_consumers_use_backend_facade() -> None:
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
        if len(imports) != 1 or imports[0].module != "unilab.base.backend":
            modules = [node.module for node in imports]
            offenders.append(f"{relative_path}: {modules}")

    assert offenders == []


def test_site_jacobian_benchmark_imports_with_mujoco_stub() -> None:
    code = textwrap.dedent(
        """
        import importlib.util
        import sys
        import types
        from pathlib import Path

        sys.modules["mujoco"] = types.ModuleType("mujoco")
        path = Path(sys.argv[1])
        spec = importlib.util.spec_from_file_location("benchmark_site_jacobian", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)

        print(module.materialize_scene_visual_override.__module__)
        print("mujoco_backend", "unilab.base.backend.mujoco.backend" in sys.modules)
        """
    )
    script = _REPO_ROOT / "scripts" / "manip_loco" / "benchmark_site_jacobian.py"
    result = subprocess.run(
        [sys.executable, "-c", code, str(script)],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.splitlines() == [
        "unilab.base.backend.mujoco.xml",
        "mujoco_backend False",
    ]


def test_mujoco_backend_import_path_does_not_eagerly_import_motrix() -> None:
    code = textwrap.dedent(
        """
        import importlib.util
        import sys

        from unilab.base.backend import create_backend, materialize_scene_visual_override
        from unilab.base.backend.mujoco.xml import create_discardvisual_xml

        assert create_backend is not None
        assert materialize_scene_visual_override is not None
        assert create_discardvisual_xml is not None

        print("mujoco_runtime", "mujoco" in sys.modules)
        print("mujoco_backend", "unilab.base.backend.mujoco.backend" in sys.modules)

        if importlib.util.find_spec("mujoco") is not None:
            import unilab.base.backend.mujoco.backend
            print("mujoco_backend imported")
        else:
            print("mujoco_backend skipped")

        print("motrix_backend", "unilab.base.backend.motrix.backend" in sys.modules)
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

        from unilab.base.backend import (
            create_backend,
            materialize_motrix_hfield_attached_scene,
            materialize_motrix_scene,
        )

        assert create_backend is not None
        assert materialize_motrix_scene is not None
        assert materialize_motrix_hfield_attached_scene is not None

        if importlib.util.find_spec("motrixsim") is not None:
            import unilab.base.backend.motrix.backend
            print("motrix_backend imported")
        else:
            print("motrix_backend skipped")

        print("mujoco_backend", "unilab.base.backend.mujoco.backend" in sys.modules)
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
