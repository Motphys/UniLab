"""UniLab resolves assets and passes only public SuperDex adapter options."""

from typing import Any

import pytest

from unilab.base import backend_factory
from unilab.base.base import EnvCfg
from unilab.base.scene import SceneCfg


def test_superdex_native_factory_resolves_assets_without_mutating_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, Any] = {}

    def resolve(model_file: str, *, assets_root: str | None) -> str:
        calls["asset"] = (model_file, assets_root)
        return "/registered/assets/fr3.superdex_bot"

    def create(name: str, scene: SceneCfg, n: int, dt: float, **kwargs: Any) -> object:
        calls["backend"] = (name, scene, n, dt, kwargs)
        return object()

    monkeypatch.setattr(backend_factory, "resolve_superdex_robot_asset", resolve)
    monkeypatch.setattr(backend_factory, "ensure_robot_assets_for_paths", lambda *_: None)
    monkeypatch.setattr(backend_factory.unisim, "create_backend", create)
    cfg = EnvCfg(superdex_assets_root="/registered/assets", superdex_effort_limits=[20.0])
    scene = SceneCfg(model_file="bots/arms/fr3_v2/fr3_v2.superdex_bot")
    backend_factory.create_backend(
        "superdex",
        scene,
        2,
        0.002,
        body_state_required=True,
        **backend_factory.env_backend_kwargs(cfg),
    )
    name, resolved_scene, n, dt, kwargs = calls["backend"]
    assert (name, n, dt) == ("superdex", 2, 0.002)
    assert resolved_scene.model_file == "/registered/assets/fr3.superdex_bot"
    assert scene.model_file == "bots/arms/fr3_v2/fr3_v2.superdex_bot"
    assert calls["asset"] == (scene.model_file, "/registered/assets")
    assert kwargs["superdex_effort_limits"] == [20.0]
    assert kwargs["superdex_num_workers"] == 0
    assert kwargs["superdex_allow_contact_approximation"] is False
    assert kwargs["body_state_required"] is False
    assert "superdex_assets_root" not in kwargs


def test_superdex_options_do_not_leak_to_other_adapters(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, Any] = {}

    def create(*args: Any, **kwargs: Any) -> object:
        calls.update(kwargs)
        return object()

    monkeypatch.setattr(backend_factory, "ensure_robot_assets_for_paths", lambda *_: None)
    monkeypatch.setattr(backend_factory.unisim, "create_backend", create)
    backend_factory.create_backend(
        "mujoco",
        SceneCfg(model_file="scene.xml"),
        1,
        0.01,
        **backend_factory.env_backend_kwargs(EnvCfg()),
    )
    assert not any(name.startswith("superdex_") for name in calls)


@pytest.mark.parametrize("workers", [0, 1, 16])
def test_large_superdex_batch_delegates_worker_selection_to_unisim(
    workers: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def create(name: str, scene: SceneCfg, num_envs: int, dt: float, **kwargs: Any) -> object:
        captured.update(backend=name, num_envs=num_envs, **kwargs)
        return object()

    monkeypatch.setattr(backend_factory, "ensure_robot_assets_for_paths", lambda *_: None)
    monkeypatch.setattr(backend_factory.unisim, "create_backend", create)
    cfg = EnvCfg(superdex_num_workers=workers)
    cfg.validate()
    backend_factory.create_backend(
        "superdex",
        SceneCfg(model_file="scene.xml"),
        1024,
        0.01,
        **backend_factory.env_backend_kwargs(cfg),
    )
    assert captured["backend"] == "superdex"
    assert captured["num_envs"] == 1024
    assert captured["superdex_num_workers"] == workers


def test_superdex_execution_mode_is_forwarded_only_when_serial(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def create(name: str, scene: SceneCfg, num_envs: int, dt: float, **kwargs: Any) -> object:
        captured.clear()
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(backend_factory, "ensure_robot_assets_for_paths", lambda *_: None)
    monkeypatch.setattr(backend_factory.unisim, "create_backend", create)
    scene = SceneCfg(model_file="scene.xml")
    serial = EnvCfg(superdex_execution_mode="serial")
    serial.validate()
    backend_factory.create_backend(
        "superdex", scene, 1, 0.002, **backend_factory.env_backend_kwargs(serial)
    )
    # Legacy unisim-core releases predate the option; the default stays absent
    # so they still accept the SuperDex kwargs.
    assert captured["superdex_execution_mode"] == "serial"
    backend_factory.create_backend(
        "superdex", scene, 1, 0.002, **backend_factory.env_backend_kwargs(EnvCfg())
    )
    assert "superdex_execution_mode" not in captured


def test_superdex_execution_mode_validates_and_does_not_leak(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, Any] = {}

    def create(*args: Any, **kwargs: Any) -> object:
        calls.update(kwargs)
        return object()

    with pytest.raises(ValueError, match="superdex_execution_mode"):
        EnvCfg(superdex_execution_mode="threaded").validate()
    monkeypatch.setattr(backend_factory, "ensure_robot_assets_for_paths", lambda *_: None)
    monkeypatch.setattr(backend_factory.unisim, "create_backend", create)
    cfg = EnvCfg(superdex_execution_mode="serial")
    backend_factory.create_backend(
        "mujoco",
        SceneCfg(model_file="scene.xml"),
        1,
        0.01,
        **backend_factory.env_backend_kwargs(cfg),
    )
    assert not any(name.startswith("superdex_") for name in calls)
