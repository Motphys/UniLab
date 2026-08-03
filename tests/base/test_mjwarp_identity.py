"""Identity and optional-import contract for the independent ``mjwarp`` backend."""

from __future__ import annotations

import ast
import subprocess
import sys
import textwrap
from importlib.machinery import ModuleSpec
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from unilab.base import registry
from unilab.base.backend.mjwarp import dependencies
from unilab.base.registry import ensure_registries


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_mjwarp_import_path_does_not_eagerly_import_warp_or_mujoco() -> None:
    code = textwrap.dedent(
        """
        import sys

        from unilab.base.backend import create_backend
        from unilab.base.backend.mjwarp import MjwarpBackend

        assert create_backend is not None
        assert MjwarpBackend is not None
        print("mujoco", "mujoco" in sys.modules)
        print("mujoco_warp", "mujoco_warp" in sys.modules)
        print("warp", "warp" in sys.modules)
        print("mujoco_backend", "unilab.base.backend.mujoco.backend" in sys.modules)
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
    ensure_registries()
    assert "mjwarp" in registry.list_registered_envs()["G1WalkFlat"]["available_backends"]

    conf_dir = _repo_root() / "conf" / "ppo"
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(conf_dir), version_base="1.3"):
        cfg = compose("config", overrides=["task=g1_walk_flat/mjwarp"])
    assert cfg.training.sim_backend == "mjwarp"
    assert cfg.env.domain_rand.randomize_kp is False
    assert cfg.env.domain_rand.randomize_kd is False

    from unilab import cli

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


def test_mjwarp_getters_do_not_materialize_warp_arrays() -> None:
    backend_path = _repo_root() / "src" / "unilab" / "base" / "backend" / "mjwarp" / "backend.py"
    tree = ast.parse(backend_path.read_text(encoding="utf-8"))
    backend = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "MjwarpBackend"
    )
    getter_nodes = [
        node
        for node in backend.body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("get_")
    ]
    assert getter_nodes
    offenders: list[str] = []
    for getter in getter_nodes:
        for node in ast.walk(getter):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "numpy"
            ):
                offenders.append(getter.name)
    assert offenders == []
