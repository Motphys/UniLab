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
from unisim.errors import UnsupportedCapabilityError
from unisim.ray_query import RayCollisionDescription, RaySceneDescription, RayTraceOutputs

from unilab.envs.mdp.ray_observation import RayPatternCfg, RayScan
from unilab.managers.manager_base import ManagerTermBaseCfg

_REPO_ROOT = Path(__file__).resolve().parents[3]


class _Block:
    """Minimal RayTraceTensorResult-shaped caster output."""

    def __init__(self, values: torch.Tensor):
        self.distance = values
        self.normal = torch.zeros(*values.shape, 3, dtype=torch.float32)


@dataclass
class _Capabilities:
    supports_tensor_pose_sync: bool = True
    supports_tensor_rays: bool = True
    supports_device_output: bool = True
    stream_event_ownership: str | None = "caller"
    torch_devices: tuple[str, ...] = ("cpu",)
    supports_min_distance: bool = True
    supports_static_filter: bool = True
    supports_tensor_normal: bool = True


class _FakeCaster:
    caster_type = "fake-ray"

    def __init__(self, *, values: torch.Tensor | None = None):
        self.values = values
        self.scene: Any = None
        self.materialized = False
        self.layout: tuple[str, ...] | None = None
        self.poses: list[Any] = []
        self.traces = 0
        self.last_query: tuple[float, object, RayTraceOutputs] | None = None
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

    def trace_tensor(
        self,
        origins,
        directions,
        max_distance,
        env_ids=None,
        outputs=None,
        *,
        min_distance=0.0,
        ray_filter=None,
    ):
        self.traces += 1
        self.last_inputs = (origins, directions, max_distance)
        self.last_query = (min_distance, ray_filter, outputs)
        if origins.shape != directions.shape:
            raise ValueError(
                f"origin and direction tensor batches must have the same shape: "
                f"{tuple(origins.shape)} != {tuple(directions.shape)}"
            )
        if origins.ndim not in (2, 3) or origins.shape[-1] != 3:
            raise ValueError(
                "ray tensors must have shape [num_rays, 3] or "
                f"[num_envs, num_rays, 3]: {tuple(origins.shape)}"
            )
        if origins.ndim == 2 and origins.shape[0] != self.num_rays:
            raise ValueError(f"expected {self.num_rays} rays, got {origins.shape[0]}")
        if origins.ndim == 3 and origins.shape[:2] != (self.num_envs, self.num_rays):
            raise ValueError(
                f"expected {self.num_envs} x {self.num_rays} rays, got {tuple(origins.shape)}"
            )
        if not origins.is_contiguous() or not directions.is_contiguous():
            raise ValueError("shared ray tensors must be contiguous")
        values = self.values
        if values is None:
            values = torch.arange(8, dtype=torch.float32).reshape(2, 4)
        return _Block(values)

    def close(self) -> None:
        self.closed += 1


def _RayScene() -> RaySceneDescription:
    return RaySceneDescription(
        num_bodies=2,
        geom_types=("plane", "sphere"),
        geom_sizes=np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=np.float64),
        geom_local_pos=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64),
        geom_local_quat=np.array([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], dtype=np.float64),
        geom_body_ids=np.array([0, 1], dtype=np.intp),
        body_names=("world", "base"),
    )


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

    def __init__(self, scene: Any | None = None, collision: Any | None = None):
        self.scene = scene if scene is not None else _RayScene()
        self.collision = (
            collision if collision is not None else RayCollisionDescription(scene=self.scene)
        )

    def get_ray_scene_description(self, body_names=None) -> Any:
        if tuple(body_names or ()) != self.scene.body_names:
            raise KeyError(tuple(body_names))
        return self.scene

    def get_ray_collision_description(self, body_names=None) -> Any:
        if body_names is not None and tuple(body_names) != self.scene.body_names:
            raise KeyError(tuple(body_names))
        return self.collision

    def get_body_ids(self, body_names) -> np.ndarray:
        names = tuple(self.scene.body_names)
        return np.asarray([names.index(name) for name in body_names], dtype=np.int64)


class _Env:
    num_envs = 2
    device = torch.device("cpu")
    backend = _Backend()

    def __init__(self, read_plan: Any, caster: _FakeCaster, scene: Any, collision: Any):
        self.scene = SimpleNamespace(
            read_plan=read_plan,
            entities={"robot": SimpleNamespace(body_names=("base",))},
        )
        self.backend = _Backend(scene, collision)
        self._caster = caster

    @property
    def caster(self) -> _FakeCaster:
        return self._caster


def _bind(
    monkeypatch,
    values: torch.Tensor | None = None,
    pose_pos: torch.Tensor | None = None,
    pose_quat: torch.Tensor | None = None,
    **pattern_kwargs: Any,
) -> tuple[_FakeCaster, RayScan]:
    scene = _RayScene()
    collision = RayCollisionDescription(
        scene=scene,
        geom_names=("ground", "ball"),
        geom_tags=(("terrain",), ("robot",)),
        body_parent_ids=np.array([-1, 0], dtype=np.intp),
    )
    caster = _FakeCaster(values=values)
    if pose_pos is None:
        pose_pos = torch.zeros((2, 1, 3), dtype=torch.float32)
    if pose_quat is None:
        pose_quat = torch.zeros((2, 1, 4), dtype=torch.float32)
        pose_quat[..., 0] = 1.0
    read_plan = _ReadPlan(pose_pos, pose_quat)
    env = _Env(read_plan, caster, scene, collision)
    pattern = RayPatternCfg(
        entity_name="robot",
        attachment_body_name="base",
        num_rays=4,
        pattern_start=0.0,
        pattern_stop=1.5,
        max_distance=3.0,
        **pattern_kwargs,
    )
    term = RayScan(ManagerTermBaseCfg(func=RayScan, params={"ray_pattern": pattern}), env)
    monkeypatch.setattr(
        "unilab.envs.mdp.ray_observation._create_ray_caster",
        lambda caster_name, **kwargs: (
            setattr(caster, "scene", kwargs["collision"].scene),
            caster,
        )[1],
    )
    term.bind_read_phase()
    return caster, term


def test_ray_pattern_config_validates_public_names_and_values() -> None:
    assert (
        RayPatternCfg(
            entity_name="robot",
            attachment_body_name="base",
            num_rays=4,
            pattern_start=0,
            pattern_stop=1,
            max_distance=2,
        ).attachment_body_name
        == "base"
    )
    for kwargs in (
        {"entity_name": ""},
        {"attachment_body_name": None},
        {"num_rays": 0},
        {"max_distance": 0.0},
        {"min_distance": -0.1},
        {"min_distance": 1.0},
        {"sensor_offset": "0 0 0"},
        {"sensor_offset": (0.0, 0.0)},
        {"attachment_body_name": ""},
        {"caster_options": {"collision": object()}},
        {"caster": ""},
    ):
        base = dict(
            entity_name="robot",
            attachment_body_name="base",
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
        assert caster.layout == ("world", "base")
        assert term.observation_width == 4
        assert caster.poses[-1].body_names == ("world", "base")
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


def test_ray_scan_uses_public_collision_descriptor_and_rich_query(monkeypatch) -> None:
    source = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    caster, term = _bind(
        monkeypatch,
        values=source,
        min_distance=0.2,
        include_normal=True,
        allowed_geom_names=("ball",),
        allowed_tags=("robot",),
    )
    result = term(term._env)
    collision = term._env.backend.collision
    packed = result.reshape(2, -1, 4)

    assert term.observation_width == 16
    assert result.shape == (2, term.observation_width)
    assert caster.scene is collision.scene
    assert caster.last_query is not None
    min_distance, ray_filter, outputs = caster.last_query
    assert min_distance == 0.2
    assert ray_filter is not None
    assert ray_filter.geom_mask(collision).tolist() == [False, True]
    assert outputs.normal
    torch.testing.assert_close(packed[..., 0], source)
    torch.testing.assert_close(packed[..., 1:4], torch.zeros((2, 4, 3)))
    term.close()


def test_ray_scan_excludes_attachment_body_through_public_filter(monkeypatch) -> None:
    caster, term = _bind(monkeypatch, excluded_body_names=("base",))
    term(term._env)
    collision = term._env.backend.collision

    assert caster.last_query is not None
    ray_filter = caster.last_query[1]
    assert ray_filter is not None
    assert ray_filter.geom_mask(collision).tolist() == [True, False]
    term.close()


def test_ray_scan_fails_closed_on_unsupported_rich_capabilities(monkeypatch) -> None:
    caster = _FakeCaster()
    read_plan = _ReadPlan(
        torch.zeros((2, 1, 3), dtype=torch.float32),
        torch.zeros((2, 1, 4), dtype=torch.float32),
    )
    scene = _RayScene()
    collision = RayCollisionDescription(
        scene=scene,
        geom_names=("ground", "ball"),
        geom_tags=(("terrain",), ("robot",)),
        body_parent_ids=np.array([-1, 0], dtype=np.intp),
    )
    env = _Env(read_plan, caster, scene, collision)
    pattern = RayPatternCfg(
        entity_name="robot",
        attachment_body_name="base",
        num_rays=2,
        pattern_start=0.0,
        pattern_stop=1.0,
        max_distance=2.0,
        min_distance=0.1,
    )
    term = RayScan(ManagerTermBaseCfg(func=RayScan, params={"ray_pattern": pattern}), env)
    caster.get_ray_capabilities = lambda: _Capabilities(supports_min_distance=False)
    monkeypatch.setattr(
        "unilab.envs.mdp.ray_observation._create_ray_caster",
        lambda caster_name, **kwargs: caster,
    )
    with pytest.raises(NotImplementedError, match="min_distance"):
        term.bind_read_phase()
    assert not caster.materialized


def test_ray_scan_fails_closed_on_unsupported_filter_and_normal(monkeypatch) -> None:
    def bind(**pattern_kwargs):
        caster = _FakeCaster()
        read_plan = _ReadPlan(
            torch.zeros((2, 1, 3), dtype=torch.float32),
            torch.zeros((2, 1, 4), dtype=torch.float32),
        )
        scene = _RayScene()
        collision = RayCollisionDescription(
            scene=scene,
            geom_names=("ground", "ball"),
            geom_tags=(("terrain",), ("robot",)),
            body_parent_ids=np.array([-1, 0], dtype=np.intp),
        )
        env = _Env(read_plan, caster, scene, collision)
        values = {
            "entity_name": "robot",
            "attachment_body_name": "base",
            "num_rays": 2,
            "pattern_start": 0.0,
            "pattern_stop": 1.0,
            "max_distance": 2.0,
        }
        values.update(pattern_kwargs)
        term = RayScan(
            ManagerTermBaseCfg(func=RayScan, params={"ray_pattern": RayPatternCfg(**values)}),
            env,
        )
        monkeypatch.setattr(
            "unilab.envs.mdp.ray_observation._create_ray_caster",
            lambda caster_name, **kwargs: caster,
        )
        return caster, term

    caster, term = bind(excluded_body_names=("base",))
    caster.get_ray_capabilities = lambda: _Capabilities(supports_static_filter=False)
    with pytest.raises(NotImplementedError, match="static ray filters"):
        term.bind_read_phase()

    caster, term = bind(include_normal=True)
    caster.get_ray_capabilities = lambda: _Capabilities(supports_tensor_normal=False)
    with pytest.raises(NotImplementedError, match="tensor normals"):
        term.bind_read_phase()


def test_ray_scan_maps_ancestor_exclusion_from_public_names(monkeypatch) -> None:
    caster, term = _bind(monkeypatch, excluded_ancestor_body_names=("world",))
    term(term._env)
    collision = term._env.backend.collision

    assert caster.last_query is not None
    ray_filter = caster.last_query[1]
    assert ray_filter is not None
    assert ray_filter.geom_mask(collision).tolist() == [False, False]
    term.close()


def test_ray_scan_fails_closed_on_missing_ancestry(monkeypatch) -> None:
    caster = _FakeCaster()
    read_plan = _ReadPlan(
        torch.zeros((2, 1, 3), dtype=torch.float32),
        torch.zeros((2, 1, 4), dtype=torch.float32),
    )
    scene = _RayScene()
    collision = RayCollisionDescription(
        scene=scene,
        geom_names=("ground", "ball"),
        geom_tags=(("terrain",), ("robot",)),
    )
    env = _Env(read_plan, caster, scene, collision)
    pattern = RayPatternCfg(
        entity_name="robot",
        attachment_body_name="base",
        num_rays=2,
        pattern_start=0.0,
        pattern_stop=1.0,
        max_distance=2.0,
        excluded_ancestor_body_names=("world",),
    )
    term = RayScan(ManagerTermBaseCfg(func=RayScan, params={"ray_pattern": pattern}), env)
    with pytest.raises(UnsupportedCapabilityError, match="ancestor"):
        term.bind_read_phase()


def test_ray_scan_fails_closed_on_uncovered_non_root_body(monkeypatch) -> None:
    scene = RaySceneDescription(
        num_bodies=3,
        geom_types=("plane", "sphere", "box"),
        geom_sizes=np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.2, 0.2, 0.2]], dtype=np.float64),
        geom_local_pos=np.zeros((3, 3), dtype=np.float64),
        geom_local_quat=np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (3, 1)),
        geom_body_ids=np.array([0, 1, 2], dtype=np.intp),
        body_names=("world", "base", "static_object"),
    )
    collision = RayCollisionDescription(
        scene=scene,
        geom_names=("ground", "ball", "object"),
        geom_tags=(("terrain",), ("robot",), ("terrain",)),
        body_parent_ids=np.array([-1, 0, 0], dtype=np.intp),
    )
    caster = _FakeCaster()
    read_plan = _ReadPlan(
        torch.zeros((2, 1, 3), dtype=torch.float32),
        torch.zeros((2, 1, 4), dtype=torch.float32),
    )
    env = _Env(read_plan, caster, scene, collision)
    pattern = RayPatternCfg(
        entity_name="robot",
        attachment_body_name="base",
        num_rays=2,
        pattern_start=0.0,
        pattern_stop=1.0,
        max_distance=2.0,
    )
    term = RayScan(ManagerTermBaseCfg(func=RayScan, params={"ray_pattern": pattern}), env)
    with pytest.raises(NotImplementedError, match="uncovered bodies"):
        term.bind_read_phase()


def test_ray_scan_publishes_matching_attachment_ray_batches(monkeypatch) -> None:
    caster, term = _bind(monkeypatch)

    term(term._env)

    origins, directions, max_distance = caster.last_inputs
    assert origins.shape == directions.shape == (2, 4, 3)
    assert origins.is_contiguous() and directions.is_contiguous()
    assert origins.dtype == directions.dtype == torch.float32
    assert origins.device == directions.device == torch.device("cpu")
    torch.testing.assert_close(
        origins,
        torch.tensor(
            [[[0.0, 0.0, 0.0]] * 4] * 2,
            dtype=torch.float32,
        ),
    )
    angles = torch.linspace(
        term.ray_pattern.pattern_start,
        term.ray_pattern.pattern_stop,
        term.ray_pattern.num_rays,
        dtype=torch.float32,
    )
    expected_directions = torch.stack(
        (torch.cos(angles), torch.sin(angles), torch.zeros_like(angles)), dim=-1
    ).repeat(2, 1, 1)
    torch.testing.assert_close(directions, expected_directions)
    assert max_distance == term.ray_pattern.max_distance
    term.close()


def test_ray_pattern_follows_body_attachment_pose(monkeypatch) -> None:
    pose_pos = torch.tensor([[[0.1, 0.2, 0.3]], [[-0.4, 0.5, -0.2]]], dtype=torch.float32)
    pose_quat = torch.zeros((2, 1, 4), dtype=torch.float32)
    pose_quat[..., 0] = 0.5**0.5
    pose_quat[..., 3] = 0.5**0.5
    caster, term = _bind(
        monkeypatch,
        pose_pos=pose_pos,
        pose_quat=pose_quat,
        sensor_offset=(0.2, 0.0, 0.1),
    )

    term(term._env)
    origins, directions, _ = caster.last_inputs
    pose = caster.poses[-1]
    assert pose.body_names == ("world", "base")
    assert torch.count_nonzero(pose.pos_w[:, 0, :]).item() == 0
    torch.testing.assert_close(pose.pos_w[:, 1, :], pose_pos[:, 0, :])
    rotation = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    expected_origin = pose_pos + torch.tensor([0.0, 0.2, 0.1], dtype=torch.float32)[None, None, :]
    angles = torch.linspace(0.0, 1.5, 4, dtype=torch.float32)
    local_directions = torch.stack(
        (torch.cos(angles), torch.sin(angles), torch.zeros_like(angles)), dim=-1
    )
    expected_directions = local_directions[None, :, :] @ rotation.T
    torch.testing.assert_close(origins, expected_origin.repeat(1, 4, 1))
    torch.testing.assert_close(directions, expected_directions.repeat(2, 1, 1))
    term.close()


def test_ray_scan_fails_closed_before_binding_and_on_unsupported_capability(monkeypatch) -> None:
    scene = _RayScene()
    caster = _FakeCaster()
    read_plan = _ReadPlan(
        torch.zeros((2, 1, 3), dtype=torch.float32),
        torch.zeros((2, 1, 4), dtype=torch.float32),
    )
    collision = RayCollisionDescription(
        scene=scene,
        geom_names=("ground", "ball"),
        geom_tags=(("terrain",), ("robot",)),
        body_parent_ids=np.array([-1, 0], dtype=np.intp),
    )
    env = _Env(read_plan, caster, scene, collision)
    pattern = RayPatternCfg(
        entity_name="robot",
        attachment_body_name="base",
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

    collision = env.backend.get_ray_collision_description()
    assert collision.scene is env.backend.scene
    assert collision is env.backend.collision
    assert term.ray_pattern.caster == "uni_ray"
    source = (_REPO_ROOT / "src/unilab/envs/mdp/ray_observation.py").read_text(encoding="utf-8")
    for banned in (
        "import uni_ray",
        "MjModel",
        "mjwarp.Model",
        "MjBatchCollision",
        "build_collision_description",
    ):
        assert banned not in source, banned
    assert not any(
        path.name.startswith("_") and path.is_file() for path in Path(__file__).parent.rglob("*.py")
    )
    term.close()
