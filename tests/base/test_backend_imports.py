from __future__ import annotations

import subprocess
import sys
import textwrap


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
    assert lines[0] in {"mujoco_backend imported", "mujoco_backend skipped"}
    assert lines[1:] == ["motrix_backend False", "motrixsim False"]
