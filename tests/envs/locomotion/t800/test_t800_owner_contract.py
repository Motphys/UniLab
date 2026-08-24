"""Registry and cold-path owner contracts for the T800 walk-flat task."""

from __future__ import annotations

import importlib
import importlib.util

from unilab.base import registry
from unilab.envs import ManagerBasedRlEnvCfg
from unilab.tasks import __unilab_registry_modules__


def test_t800_registry_metadata_is_manager_based_and_bootstrapped_explicitly() -> None:
    """The production bootstrap exposes only the Manager-Based MuJoCo owner."""
    registry.ensure_registries()

    assert "unilab.tasks.locomotion.t800" in __unilab_registry_modules__
    metadata = registry.list_registered_envs()
    assert metadata["T800WalkFlat"] == {
        "config_factory": "ManagerBasedRlEnvCfg",
        "available_backends": ["mujoco"],
    }

    t800 = importlib.import_module("unilab.tasks.locomotion.t800")
    factory = registry._envs["T800WalkFlat"].env_factory_dict["mujoco"]
    assert factory is t800.make_t800_walk_env
    assert callable(factory)
    assert factory.__name__ == "make_t800_walk_env"
    assert factory.__module__ == "unilab.tasks.locomotion.t800"
    assert registry._envs["T800WalkFlat"].env_cfg_factory is ManagerBasedRlEnvCfg

    try:
        legacy_spec = importlib.util.find_spec("unilab.envs.locomotion.t800")
    except ModuleNotFoundError:
        legacy_spec = None
    assert legacy_spec is None


def test_t800_factory_resolves_assets_before_generic_runtime(monkeypatch) -> None:
    """Asset markers resolve before the exact generic runtime invocation."""
    t800 = importlib.import_module("unilab.tasks.locomotion.t800")
    calls: list[tuple[str, object, object]] = []
    sentinel = object()

    def fake_resolver(directory: str, *, marker: str):
        calls.append(("resolve", directory, marker))
        return sentinel

    def fake_builder(cfg, *, num_envs: int, backend_type: str):
        calls.append(("build", num_envs, backend_type))
        assert cfg is config
        return sentinel

    config = ManagerBasedRlEnvCfg()
    monkeypatch.setattr(t800, "resolve_robot_asset_dir", fake_resolver)
    monkeypatch.setattr(t800, "make_manager_based_rl_env", fake_builder)

    result = t800.make_t800_walk_env(config, num_envs=4, backend_type="mujoco")

    assert result is sentinel
    assert calls == [
        ("resolve", "robots/t800/assets", "LINK_BASE.obj"),
        ("resolve", "robots/t800/textures", "LINK_BASE.png"),
        ("build", 4, "mujoco"),
    ]
