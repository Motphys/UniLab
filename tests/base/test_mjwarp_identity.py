"""Identity and optional-import contract for the independent ``mjwarp`` backend."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from importlib.machinery import ModuleSpec

import numpy as np
import pytest
from unisim.backend.mjwarp import dependencies


def test_mjwarp_import_path_does_not_eagerly_import_warp_or_mujoco() -> None:
    code = textwrap.dedent(
        """
        import sys

        from unilab.base.backend_factory import create_backend
        from unisim.backend.mjwarp import MjwarpBackend

        assert create_backend is not None
        assert MjwarpBackend is not None
        print("mujoco", "mujoco" in sys.modules)
        print("mujoco_warp", "mujoco_warp" in sys.modules)
        print("warp", "warp" in sys.modules)
        print("mujoco_backend", "unisim.backend.mujoco.backend" in sys.modules)
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.splitlines() == [
        "mujoco False",
        "mujoco_warp False",
        "warp False",
        "mujoco_backend False",
    ]


def test_mjwarp_missing_dependency_reports_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    original_import = dependencies.importlib.import_module

    def import_without_mjwarp(name: str):
        if name == "mujoco_warp":
            raise ModuleNotFoundError(name=name)
        return original_import(name)

    monkeypatch.setattr(dependencies.importlib, "import_module", import_without_mjwarp)

    with pytest.raises(dependencies.MjwarpDependencyError, match="uv sync --extra mjwarp"):
        dependencies.load_mjwarp_dependencies()


def test_mjwarp_dependency_version_mismatch_fails_before_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    imported: list[str] = []
    monkeypatch.setattr(dependencies.metadata, "version", lambda _name: "3.10.0.4")
    monkeypatch.setattr(
        dependencies.importlib,
        "import_module",
        lambda name: imported.append(name),
    )

    with pytest.raises(
        dependencies.MjwarpDependencyError,
        match=r"requires exact mujoco-warp version 3\.10\.0\.3, found 3\.10\.0\.4",
    ):
        dependencies.load_mjwarp_dependencies()

    assert imported == []


def test_mjwarp_identity_is_independent_from_mujoco(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from unilab import cli

    assert "mjwarp" in cli.SUPPORTED_SIMS
    (tmp_path / "scripts").mkdir(parents=True)
    (tmp_path / "scripts" / "train_rsl_rl.py").write_text("", encoding="utf-8")
    owner = tmp_path / "conf" / "ppo" / "task" / "g1_walk_flat"
    owner.mkdir(parents=True)
    (owner / "mjwarp.yaml").write_text(
        "training:\n  sim_backend: mjwarp\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        cli,
        "find_spec",
        lambda name: ModuleSpec(name, loader=None) if name in {"mujoco_warp", "warp"} else None,
    )
    command = cli.build_command(
        mode="train",
        algo="ppo",
        task="g1_walk_flat",
        sim="mjwarp",
        overrides=[],
        root=tmp_path,
    )
    assert command[1:] == [
        str(tmp_path / "scripts" / "train_rsl_rl.py"),
        "task=g1_walk_flat/mjwarp",
    ]


def test_mjwarp_actuation_metadata_uses_cpu_model_only() -> None:
    import mujoco
    from unisim.backend.mjwarp.backend import MjwarpBackend

    from unilab.assets import ASSETS_ROOT_PATH

    model = mujoco.MjModel.from_xml_path(
        str(ASSETS_ROOT_PATH / "robots" / "go2" / "scene_flat.xml")
    )
    backend = object.__new__(MjwarpBackend)
    backend._mujoco = mujoco
    backend._cpu_model = model
    backend._root_qpos_dim = 7
    backend._actuator_names = tuple(
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, index) or f"#{index}"
        for index in range(model.nu)
    )

    assert len(backend.get_actuator_joint_names()) == model.nu
    assert backend.get_actuator_joint_names()[0] == "FR_hip_joint"
    default = backend.get_default_dof_pos()
    np.testing.assert_array_equal(default, model.qpos0[7:])
    default[:] = np.nan
    assert np.isfinite(backend.get_default_dof_pos()).all()
