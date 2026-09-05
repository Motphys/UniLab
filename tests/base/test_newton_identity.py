"""Newton optional-runtime identity and routing boundaries."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from importlib.machinery import ModuleSpec
from pathlib import Path

import pytest


def test_newton_import_path_does_not_eagerly_import_engine_modules() -> None:
    pytest.importorskip("unisim.backend.newton")
    code = textwrap.dedent(
        """
        import sys

        from unilab.base.backend_factory import create_backend
        from unisim.backend.newton import NewtonBackend

        assert create_backend is not None
        assert NewtonBackend is not None
        print("newton", "newton" in sys.modules)
        print("mujoco_warp", "mujoco_warp" in sys.modules)
        print("warp", "warp" in sys.modules)
        print("mujoco", "mujoco" in sys.modules)
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.splitlines() == [
        "newton False",
        "mujoco_warp False",
        "warp False",
        "mujoco False",
    ]


def test_newton_owner_routes_through_cli_without_importing_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from unilab import cli

    (tmp_path / "scripts").mkdir(parents=True)
    (tmp_path / "scripts" / "train_rsl_rl.py").write_text("", encoding="utf-8")
    owner = tmp_path / "conf" / "ppo" / "task" / "g1_walk_flat"
    owner.mkdir(parents=True)
    (owner / "newton.yaml").write_text(
        "training:\n  sim_backend: newton\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        cli,
        "find_spec",
        lambda name: (
            ModuleSpec(name, loader=None)
            if name in {"newton", "mujoco_warp", "mujoco", "warp"}
            else None
        ),
    )

    command = cli.build_command(
        mode="train",
        algo="ppo",
        task="g1_walk_flat",
        sim="newton",
        overrides=[],
        root=tmp_path,
    )
    assert command[1:] == [
        str(tmp_path / "scripts" / "train_rsl_rl.py"),
        "task=g1_walk_flat/newton",
    ]
