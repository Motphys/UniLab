from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np

from unilab.base.scene import SceneCfg


class _FakeMotrixLink:
    index = 0
    name = "base"

    def get_mass_override(self, data: Any) -> np.ndarray:
        return np.ones((1,), dtype=np.float64)

    def get_center_of_mass_override(self, data: Any) -> np.ndarray:
        return np.zeros((1, 3), dtype=np.float64)


class _FakeMotrixBody:
    floatingbase = None


class _FakeMotrixGeom:
    def __init__(self, *, name: str = "floor", hfield: object | None = None) -> None:
        self.name = name
        self.index = 0
        self.hfield = hfield


class _FakeMotrixModel:
    def __init__(self) -> None:
        self.options = SimpleNamespace(timestep=None, max_iterations=None)
        self.links = [_FakeMotrixLink()]
        self.num_links = 1
        self.geoms = [_FakeMotrixGeom(hfield=object())]
        self.num_geoms = len(self.geoms)
        self.actuators = []
        self.num_actuators = 0
        self.joint_dof_pos_indices = []
        self.joint_dof_vel_indices = []
        self.floating_bases = []

    def get_body(self, name: str) -> _FakeMotrixBody | None:
        return _FakeMotrixBody() if name == "base" else None

    def get_link(self, name: str) -> _FakeMotrixLink | None:
        return _FakeMotrixLink() if name == "base" else None

    def get_link_index(self, name: str) -> int | None:
        return 0 if name == "base" else None

    def get_geom_index(self, name: str) -> int | None:
        return 0 if name == "floor" else None

    def get_geom(self, arg: Any) -> _FakeMotrixGeom | None:
        if isinstance(arg, int):
            return self.geoms[arg] if 0 <= arg < len(self.geoms) else None
        if isinstance(arg, str):
            geom_id = self.get_geom_index(arg)
            return self.get_geom(geom_id) if geom_id is not None else None
        return None

    def forward_kinematic(self, data: Any) -> None:
        return None

    def get_link_poses(self, data: Any) -> np.ndarray:
        return np.asarray([[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]], dtype=np.float64)


class _FakeNativeHFieldGeom(_FakeMotrixGeom):
    def __init__(self) -> None:
        super().__init__(hfield=object())
        self.data: Any | None = None
        self.xy: np.ndarray | None = None

    def sample_height(self, data: Any, xy: np.ndarray) -> np.ndarray:
        self.data = data
        self.xy = np.asarray(xy)
        return self.xy[..., 0] + 2.0 * self.xy[..., 1]


class _FakeTerrainScanner:
    instances: list["_FakeTerrainScanner"] = []

    def __init__(
        self,
        terrain: Any,
        frame: Any,
        offsets: np.ndarray,
        *,
        alignment: str = "yaw",
        output: str = "height",
    ) -> None:
        self.terrain = terrain
        self.frame = frame
        self.offsets = np.asarray(offsets)
        self.alignment = alignment
        self.output = output
        self.scan_calls = 0
        self.out: np.ndarray | None = None
        _FakeTerrainScanner.instances.append(self)

    def scan(self, data: Any, out: np.ndarray | None = None) -> np.ndarray:
        del data
        self.scan_calls += 1
        self.out = out
        num_envs = out.shape[0] if out is not None else 1
        values = self.offsets[:, 0] + 2.0 * self.offsets[:, 1]
        if self.output == "clearance":
            values = 5.0 - values
        result = np.broadcast_to(values, (num_envs, values.shape[0])).astype(np.float32)
        if out is not None:
            out[...] = result
            return out
        return result


def _install_fake_motrix(monkeypatch, tmp_path):
    import unilab.base.backend.motrix.backend as mod
    import unilab.base.backend.motrix.scene as scene_mod

    fake_model = _FakeMotrixModel()
    _FakeTerrainScanner.instances.clear()

    monkeypatch.setattr(mod, "MOTRIX_AVAILABLE", True)
    monkeypatch.setattr(
        mod,
        "mtx",
        SimpleNamespace(
            SceneData=lambda model, batch: SimpleNamespace(),
            TerrainScanner=_FakeTerrainScanner,
            GeomHField=_FakeNativeHFieldGeom,
        ),
        raising=False,
    )
    monkeypatch.setattr(scene_mod, "materialize_motrix_scene", lambda **kwargs: fake_model)
    return mod, fake_model


def test_motrix_backend_defaults_max_iterations_to_three(monkeypatch, tmp_path) -> None:
    mod, fake_model = _install_fake_motrix(monkeypatch, tmp_path)

    mod.MotrixBackend(SceneCfg(model_file="source.xml"), num_envs=1, sim_dt=0.01, base_name="base")

    assert fake_model.options.timestep == 0.01
    assert fake_model.options.max_iterations == 3


def test_motrix_backend_accepts_max_iterations_override(monkeypatch, tmp_path) -> None:
    mod, fake_model = _install_fake_motrix(monkeypatch, tmp_path)

    mod.MotrixBackend(
        SceneCfg(model_file="source.xml"),
        num_envs=1,
        sim_dt=0.01,
        base_name="base",
        max_iterations=7,
    )

    assert fake_model.options.max_iterations == 7


def test_motrix_backend_motion_body_ids_read_scene_model(monkeypatch, tmp_path) -> None:
    mod, _ = _install_fake_motrix(monkeypatch, tmp_path)

    backend = mod.MotrixBackend(
        SceneCfg(model_file="source.xml"), num_envs=1, sim_dt=0.01, base_name="base"
    )

    np.testing.assert_array_equal(
        backend.get_motion_body_ids(["base"]), np.array([1], dtype=np.int32)
    )


def test_motrix_backend_resolves_geom_ids(monkeypatch, tmp_path) -> None:
    mod, _ = _install_fake_motrix(monkeypatch, tmp_path)

    backend = mod.MotrixBackend(
        SceneCfg(model_file="source.xml"), num_envs=1, sim_dt=0.01, base_name="base"
    )

    assert backend.get_geom_id("floor") == 0


def test_motrix_backend_creates_terrain_scanner(monkeypatch, tmp_path) -> None:
    mod, fake_model = _install_fake_motrix(monkeypatch, tmp_path)
    native_geom = _FakeNativeHFieldGeom()
    fake_model.geoms = [native_geom]
    backend = mod.MotrixBackend(
        SceneCfg(model_file="source.xml"), num_envs=2, sim_dt=0.01, base_name="base"
    )
    backend._num_envs = 2
    offsets = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)

    scanner = backend.create_hfield_scanner(
        hfield_geom_id=0,
        offsets=offsets,
        frame_body_id=0,
        alignment="yaw",
        output="height",
    )
    heights = scanner.scan()
    heights_again = scanner.scan()

    assert len(_FakeTerrainScanner.instances) == 1
    native_scanner = _FakeTerrainScanner.instances[0]
    assert native_scanner.terrain is native_geom
    assert native_scanner.frame is fake_model.links[0]
    assert native_scanner.alignment == "yaw"
    assert native_scanner.output == "height"
    assert native_scanner.offsets.dtype == np.float32
    np.testing.assert_allclose(native_scanner.offsets, offsets)
    assert native_scanner.scan_calls == 2
    assert native_scanner.out is not None
    assert native_scanner.out.shape == (2, 2)
    np.testing.assert_allclose(heights, [[1.0, 2.0], [1.0, 2.0]], atol=1e-6)
    np.testing.assert_allclose(heights_again, heights, atol=1e-6)


def test_motrix_backend_hfield_sampling_can_return_clearance(monkeypatch, tmp_path) -> None:
    mod, fake_model = _install_fake_motrix(monkeypatch, tmp_path)
    fake_model.geoms = [_FakeNativeHFieldGeom()]
    backend = mod.MotrixBackend(
        SceneCfg(model_file="source.xml"), num_envs=1, sim_dt=0.01, base_name="base"
    )

    scanner = backend.create_hfield_scanner(
        hfield_geom_id=0,
        offsets=np.asarray([[1.0, 0.0]], dtype=np.float64),
        frame_body_id=0,
        output="clearance",
    )
    clearance = scanner.scan()

    np.testing.assert_allclose(clearance, [[4.0]], atol=1e-12)


def test_motrix_backend_hfield_sampling_uses_terrain_scanner_output_contract(
    monkeypatch, tmp_path
) -> None:
    mod, fake_model = _install_fake_motrix(monkeypatch, tmp_path)
    native_geom = _FakeNativeHFieldGeom()
    fake_model.geoms = [native_geom]
    backend = mod.MotrixBackend(
        SceneCfg(model_file="source.xml"), num_envs=1, sim_dt=0.01, base_name="base"
    )
    backend._link_poses = np.asarray(
        [[[10.0, 20.0, 5.0, 0.0, 0.0, 0.0, 1.0]]],
        dtype=np.float64,
    )

    scanner = backend.create_hfield_scanner(
        hfield_geom_id=0,
        offsets=np.asarray([[1.0, 0.0]], dtype=np.float64),
        frame_body_id=0,
        output="height",
    )
    heights = scanner.scan()

    assert len(_FakeTerrainScanner.instances) == 1
    scanner = _FakeTerrainScanner.instances[0]
    assert scanner.out is not None
    assert heights.dtype == np.float32
    np.testing.assert_allclose(heights, [[1.0]], atol=1e-6)


def test_create_backend_routes_motrix_max_iterations_override(monkeypatch) -> None:
    import unilab.base.backend as backend_factory
    from unilab.base.scene import SceneCfg

    captured: dict[str, Any] = {}

    class FakeMotrixBackend:
        def __init__(
            self,
            scene: SceneCfg,
            num_envs: int,
            sim_dt: float,
            *,
            max_iterations: int = 3,
            **kwargs: Any,
        ) -> None:
            captured["max_iterations"] = max_iterations
            captured["kwargs"] = kwargs

    monkeypatch.setattr(
        backend_factory,
        "_load_motrix_backend",
        lambda: (FakeMotrixBackend, True),
    )

    backend_factory.create_backend(
        "motrix",
        SceneCfg(model_file="model.xml"),
        num_envs=1,
        sim_dt=0.01,
        motrix_max_iterations=9,
    )

    assert captured["max_iterations"] == 9
    assert "motrix_max_iterations" not in captured["kwargs"]


def test_create_backend_does_not_route_motrix_option_to_mujoco(monkeypatch) -> None:
    import unilab.base.backend as backend_factory
    from unilab.base.scene import SceneCfg

    captured: dict[str, Any] = {}

    class FakeMuJoCoBackend:
        def __init__(self, scene: SceneCfg, num_envs: int, sim_dt: float, **kwargs: Any) -> None:
            captured["kwargs"] = kwargs

    monkeypatch.setattr(backend_factory, "_load_mujoco_backend", lambda: FakeMuJoCoBackend)

    backend_factory.create_backend(
        "mujoco",
        SceneCfg(model_file="model.xml"),
        num_envs=1,
        sim_dt=0.01,
        motrix_max_iterations=9,
    )

    assert "motrix_max_iterations" not in captured["kwargs"]
    assert "max_iterations" not in captured["kwargs"]
