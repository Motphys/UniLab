"""Focused tests for the tensor-native ray observation integration."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from unilab.envs.mdp.ray_observation import RayPatternCfg, RayScan
from unilab.managers.manager_base import ManagerTermBaseCfg

_REPO_ROOT = Path(__file__).resolve().parents[3]


class _Block:
    """Minimal RayTraceTensorResult-shaped caster output."""

    def __init__(self, values: torch.Tensor):
        self.distance = values


@dataclass
class _Capabilities:
    supports_tensor_pose_sync: bool = True
    supports_tensor_rays: bool = True
    supports_device_output: bool = True
    stream_event_ownership: str | None = "caller"
    torch_devices: tuple[str, ...] = ("cpu",)


class _FakeCaster:
    caster_type = "fake-ray"

    def __init__(self, *, values: torch.Tensor | None = None):
        self.values = values
        self.scene: Any = None
        self.materialized = False
        self.layout: tuple[str, ...] | None = None
        self.poses: list[Any] = []
        self.traces = 0
        self.closed = 0

    @property
    def num_envs(self) -> int:
        return 2

    @property
    def num_rays(self) -> int:
        return 4

    def get_ray_capabilities(self) -> _Capabilities:
        return _Capabilities()

    def materialize(self, scene: Any) -> None:
        assert scene is self.scene
        self.materialized = True

    def bind_pose_layout(self, body_names) -> None:
        self.layout = tuple(body_names)

    def update_pose_tensor(self, pose: Any) -> None:
        self.poses.append(pose)

    def trace_tensor(self, origins, directions, max_distance, env_ids=None, outputs=None):
        self.traces += 1
        self.last_inputs = (origins, directions, max_distance)
        if origins.shape != directions.shape:
            raise ValueError(
                f"origin and direction tensor batches must have the same shape: "
                f"{tuple(origins.shape)} != {tuple(directions.shape)}"
            )
        if origins.ndim != 2 or origins.shape[-1] != 3:
            raise ValueError(
                f"shared ray tensors must have shape [num_rays, 3]: {tuple(origins.shape)}"
            )
        if origins.shape[0] != self.num_rays:
            raise ValueError(f"expected {self.num_rays} rays, got {origins.shape[0]}")
        if not origins.is_contiguous() or not directions.is_contiguous():
            raise ValueError("shared ray tensors must be contiguous")
        values = self.values
        if values is None:
            values = torch.arange(8, dtype=torch.float32).reshape(2, 4)
        return _Block(values)

    def close(self) -> None:
        self.closed += 1


class _RayScene:
    num_bodies = 1
    num_geoms = 0
    geom_types = ()
    geom_sizes = np.empty((0, 3), dtype=np.float64)
    geom_local_pos = np.empty((0, 3), dtype=np.float64)
    geom_local_quat = np.empty((0, 4), dtype=np.float64)
    geom_body_ids = np.empty((0,), dtype=np.intp)
    body_names = ("base",)


class _ReadPlan:
    device = torch.device("cpu")
    ready = True

    def __init__(self, pos: torch.Tensor, quat: torch.Tensor):
        self.pos_w = pos
        self.quat_w = quat

    def ray_pose_view(self, entity_name: str, body_names) -> Any:
        self.entity_name = entity_name
        self.body_names = tuple(body_names)
        return SimpleNamespace(body_names=self.body_names, pos_w=self.pos_w, quat_w=self.quat_w)


class _Backend:
    backend_type = "fake-ray-backend"
    num_envs = 2

    def __init__(self, scene: Any | None = None):
        self.scene = scene if scene is not None else _RayScene()

    def get_ray_scene_description(self, body_names=None) -> Any:
        if tuple(body_names or ()) != self.scene.body_names:
            raise KeyError(tuple(body_names))
        return self.scene


class _Env:
    num_envs = 2
    device = torch.device("cpu")
    backend = _Backend()

    def __init__(self, read_plan: Any, caster: _FakeCaster, scene: Any):
        self.scene = SimpleNamespace(read_plan=read_plan)
        self.backend = _Backend(scene)
        self._caster = caster

    @property
    def caster(self) -> _FakeCaster:
        return self._caster


def _bind(monkeypatch, values: torch.Tensor | None = None) -> tuple[_FakeCaster, RayScan]:
    scene = _RayScene()
    caster = _FakeCaster(values=values)
    read_plan = _ReadPlan(
        torch.randn((2, 1, 3), dtype=torch.float32),
        torch.randn((2, 1, 4), dtype=torch.float32).renorm(dim=-1, p=2, maxnorm=1.0),
    )
    env = _Env(read_plan, caster, scene)
    pattern = RayPatternCfg(
        entity_name="robot",
        body_names=("base",),
        num_rays=4,
        pattern_start=0.0,
        pattern_stop=1.5,
        max_distance=3.0,
    )
    term = RayScan(ManagerTermBaseCfg(func=RayScan, params={"ray_pattern": pattern}), env)
    monkeypatch.setattr(
        "unilab.envs.mdp.ray_observation._create_ray_caster",
        lambda caster_name, **kwargs: (setattr(caster, "scene", scene), caster)[1],
    )
    term.bind_read_phase()
    return caster, term


def test_ray_pattern_config_validates_public_names_and_values() -> None:
    assert RayPatternCfg(
        entity_name="robot",
        body_names=("base",),
        num_rays=4,
        pattern_start=0,
        pattern_stop=1,
        max_distance=2,
    ).body_names == ("base",)
    for kwargs in (
        {"entity_name": ""},
        {"body_names": "base"},
        {"body_names": ()},
        {"body_names": ("base", "base")},
        {"num_rays": 0},
        {"max_distance": 0.0},
        {"caster": ""},
    ):
        base = dict(
            entity_name="robot",
            body_names=("base",),
            num_rays=4,
            pattern_start=0.0,
            pattern_stop=1.0,
            max_distance=1.0,
        )
        base.update(kwargs)
        with pytest.raises((TypeError, ValueError)):
            RayPatternCfg(**base)


def test_ray_scan_binds_and_traces_without_importing_uni_ray(monkeypatch) -> None:
    class _Blocker:
        def find_module(self, fullname, path=None):
            if fullname == "uni_ray" or fullname.startswith("uni_ray."):
                return self
            return None

        def load_module(self, fullname):
            raise AssertionError(f"uni_ray was imported: {fullname}")

    sys.meta_path.insert(0, _Blocker())
    try:
        caster, term = _bind(monkeypatch)
        result = term(term._env)

        assert caster.materialized
        assert caster.layout == ("base",)
        assert term.tensor_body_names == ("base",)
        assert term.observation_width == 4
        assert caster.poses[-1].body_names == ("base",)
        assert result.shape == (2, 4)
        assert result.dtype == torch.float32
        assert result.device == torch.device("cpu")
        term.close()
        assert caster.closed == 1
        term.close()
        assert caster.closed == 1
    finally:
        del sys.meta_path[0]


def test_ray_scan_copies_reused_caster_output_into_term_owned_carrier(monkeypatch) -> None:
    source = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    caster, term = _bind(monkeypatch, source)

    first = term(term._env)
    second = term(term._env)

    assert first is second
    torch.testing.assert_close(first, source)
    source[0, 0] = 99.0
    third = term(term._env)
    assert third[0, 0].item() == source[0, 0].item()
    term.close()


def test_ray_scan_publishes_matching_shared_ray_batches(monkeypatch) -> None:
    caster, term = _bind(monkeypatch)

    term(term._env)

    origins, directions, max_distance = caster.last_inputs
    assert origins.shape == directions.shape == (4, 3)
    assert origins.is_contiguous() and directions.is_contiguous()
    assert origins.dtype == directions.dtype == torch.float32
    assert origins.device == directions.device == torch.device("cpu")
    torch.testing.assert_close(
        origins,
        torch.tensor(
            [[0.0, 0.0, term.ray_pattern.pattern_center_height]] * 4,
            dtype=torch.float32,
        ),
    )
    assert max_distance == term.ray_pattern.max_distance
    term.close()


def test_ray_scan_fails_closed_before_binding_and_on_unsupported_capability(monkeypatch) -> None:
    scene = _RayScene()
    caster = _FakeCaster()
    read_plan = _ReadPlan(
        torch.zeros((2, 1, 3), dtype=torch.float32),
        torch.zeros((2, 1, 4), dtype=torch.float32),
    )
    env = _Env(read_plan, caster, scene)
    pattern = RayPatternCfg(
        entity_name="robot",
        body_names=("base",),
        num_rays=2,
        pattern_start=0.0,
        pattern_stop=1.0,
        max_distance=1.0,
    )
    term = RayScan(ManagerTermBaseCfg(func=RayScan, params={"ray_pattern": pattern}), env)
    with pytest.raises(NotImplementedError, match="bind its read phase"):
        term(env)

    caster.get_ray_capabilities = lambda: _Capabilities(supports_tensor_rays=False)
    monkeypatch.setattr(
        "unilab.envs.mdp.ray_observation._create_ray_caster",
        lambda caster_name, **kwargs: caster,
    )
    with pytest.raises(NotImplementedError, match="tensor pose/ray/output profile"):
        term.bind_read_phase()
    assert not caster.materialized
    assert caster.layout is None


def test_ray_scan_uses_only_public_backend_and_factory_boundaries(monkeypatch) -> None:
    caster, term = _bind(monkeypatch)
    env = term._env

    scene = env.backend.get_ray_scene_description(("base",))
    assert scene is env.backend.scene
    assert term.ray_pattern.caster == "uni_ray"
    assert not any(
        path.name.startswith("_") and path.is_file() for path in Path(__file__).parent.rglob("*.py")
    )
    term.close()
