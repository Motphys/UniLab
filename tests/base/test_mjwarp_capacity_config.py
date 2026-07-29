"""Config-routing contracts for mjwarp cold-path capacity ownership."""

from __future__ import annotations

from typing import Any

import pytest

import unilab.base.backend as backend_module
from unilab.base.backend import create_backend, env_backend_kwargs
from unilab.base.base import EnvCfg
from unilab.base.scene import SceneCfg


def test_env_backend_kwargs_preserves_explicit_mjwarp_capacities() -> None:
    cfg = EnvCfg(mjwarp_nconmax=128, mjwarp_njmax=256)

    assert env_backend_kwargs(cfg)["mjwarp_nconmax"] == 128
    assert env_backend_kwargs(cfg)["mjwarp_njmax"] == 256


@pytest.mark.parametrize(
    "kwargs",
    ({"mjwarp_nconmax": 0}, {"mjwarp_njmax": True}, {"mjwarp_nconmax": 128.0}),
)
def test_envcfg_rejects_invalid_mjwarp_capacities(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="mjwarp_n"):
        EnvCfg(**kwargs).validate()  # type: ignore[arg-type]


def test_mjwarp_factory_routes_only_owner_capacity_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    class FakeMjwarpBackend:
        def __init__(self, scene: SceneCfg, num_envs: int, sim_dt: float, **kwargs: Any) -> None:
            seen.update(
                {
                    "scene": scene,
                    "num_envs": num_envs,
                    "sim_dt": sim_dt,
                    "kwargs": kwargs,
                }
            )

    monkeypatch.setattr(backend_module, "_load_mjwarp_backend", lambda: FakeMjwarpBackend)
    scene = SceneCfg(model_file="owner-scene.xml")

    backend = create_backend(
        "mjwarp",
        scene,
        8,
        0.0025,
        base_name="pelvis",
        mjwarp_nconmax=128,
        mjwarp_njmax=256,
    )

    assert isinstance(backend, FakeMjwarpBackend)
    assert seen == {
        "scene": scene,
        "num_envs": 8,
        "sim_dt": 0.0025,
        "kwargs": {"base_name": "pelvis", "nconmax": 128, "njmax": 256},
    }


def test_mjwarp_capacity_fields_do_not_leak_to_mujoco(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    class FakeMuJoCoBackend:
        def __init__(self, scene: SceneCfg, num_envs: int, sim_dt: float, **kwargs: Any) -> None:
            seen.update(
                {
                    "scene": scene,
                    "num_envs": num_envs,
                    "sim_dt": sim_dt,
                    "kwargs": kwargs,
                }
            )

    monkeypatch.setattr(backend_module, "_load_mujoco_backend", lambda: FakeMuJoCoBackend)
    scene = SceneCfg(model_file="owner-scene.xml")

    backend = create_backend(
        "mujoco",
        scene,
        8,
        0.0025,
        mjwarp_nconmax=128,
        mjwarp_njmax=256,
    )

    assert isinstance(backend, FakeMuJoCoBackend)
    assert seen["scene"] is scene
    assert seen["num_envs"] == 8
    assert seen["sim_dt"] == 0.0025
    assert "mjwarp_nconmax" not in seen["kwargs"]
    assert "mjwarp_njmax" not in seen["kwargs"]
    assert "nconmax" not in seen["kwargs"]
    assert "njmax" not in seen["kwargs"]
