from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from unisim.dr.types import (
    INTERVAL_TERM_BODY_FORCE,
    RESET_TERM_BODY_IPOS,
    RESET_TERM_BODY_MASS,
    RESET_TERM_GEOM_FRICTION,
    RESET_TERM_GRAVITY,
    RESET_TERM_KD,
    RESET_TERM_KP,
    IntervalRandomizationPlan,
    IntervalTermOp,
)

from unilab.base.scene import SceneCfg


class _FakeMotrixLink:
    index = 0
    name = "base"
    joint_indices = (0,)

    def __init__(self) -> None:
        self.mass_override: np.ndarray | None = None
        self.com_override: np.ndarray | None = None
        self.external_force_calls: list[tuple[np.ndarray, bool]] = []

    def get_mass_override(self, data: Any) -> np.ndarray:
        return np.ones((int(getattr(data, "num_envs", 1)),), dtype=np.float64)

    def get_center_of_mass_override(self, data: Any) -> np.ndarray:
        return np.zeros((int(getattr(data, "num_envs", 1)), 3), dtype=np.float64)

    def set_mass_override(self, data: Any, value: np.ndarray) -> None:
        del data
        self.mass_override = np.asarray(value)

    def set_center_of_mass_override(self, data: Any, value: np.ndarray) -> None:
        del data
        self.com_override = np.asarray(value)

    def add_external_force(
        self,
        data: Any,
        force: np.ndarray,
        point: np.ndarray | None = None,
        *,
        local: bool = True,
    ) -> None:
        del data, point
        self.external_force_calls.append((np.asarray(force), bool(local)))


class _FakeMotrixBody:
    floatingbase = None


class _FakeMotrixGeom:
    def __init__(
        self,
        *,
        name: str = "floor",
        hfield: object | None = None,
        link: _FakeMotrixLink | None = None,
        collision_group: int = 1,
        collision_affinity: int = 1,
    ) -> None:
        self.name = name
        self.index = 0
        self.hfield = hfield
        self.link = link if link is not None else _FakeMotrixLink()
        self.collision_group = collision_group
        self.collision_affinity = collision_affinity
        self.size = np.asarray([0.2, 0.1, 0.0], dtype=np.float64)
        self.size_override = np.asarray([[0.2, 0.1]], dtype=np.float64)
        self.friction_override = np.asarray([[1.0, 0.001, 0.002]], dtype=np.float64)

    def get_size_override(self, data: Any) -> np.ndarray:
        num_envs = int(getattr(data, "num_envs", self.size_override.shape[0]))
        if self.size_override.shape[0] == num_envs:
            return self.size_override.copy()
        return np.broadcast_to(
            self.size_override[0], (num_envs, self.size_override.shape[1])
        ).copy()

    def set_size_override(self, data: Any, value: np.ndarray) -> None:
        del data
        self.size_override = np.asarray(value)

    def get_friction_override(self, data: Any) -> np.ndarray:
        num_envs = int(getattr(data, "num_envs", self.friction_override.shape[0]))
        if self.friction_override.shape[0] == num_envs:
            return self.friction_override.copy()
        return np.broadcast_to(
            self.friction_override[0], (num_envs, self.friction_override.shape[1])
        ).copy()

    def set_friction_override(self, data: Any, value: np.ndarray) -> None:
        del data
        self.friction_override = np.asarray(value)


class _FakeMotrixModel:
    def __init__(self) -> None:
        self.options = SimpleNamespace(timestep=None, max_iterations=None)
        self.links = [_FakeMotrixLink()]
        self.num_links = 1
        self.geoms = [_FakeMotrixGeom(hfield=object(), link=self.links[0])]
        self.num_geoms = len(self.geoms)
        self.actuators = []
        self.num_actuators = 0
        self.joint_dof_pos_indices = []
        self.joint_dof_vel_indices = []
        self.floating_bases = []
        self.gravity_override: np.ndarray | None = None

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
        num_envs = int(getattr(data, "num_envs", 1))
        pose = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        return np.broadcast_to(pose, (num_envs, self.num_links, 7)).copy()

    def get_link_velocities(self, data: Any) -> np.ndarray:
        num_envs = int(getattr(data, "num_envs", 1))
        return np.zeros((num_envs, self.num_links, 6), dtype=np.float32)

    def get_gravity_override(self, data: Any) -> np.ndarray:
        num_envs = int(getattr(data, "num_envs", 1))
        if self.gravity_override is None:
            return np.broadcast_to(
                np.asarray([0.0, 0.0, -9.81], dtype=np.float32),
                (num_envs, 3),
            ).copy()
        return self.gravity_override.copy()

    def set_gravity_override(self, data: Any, gravity: np.ndarray) -> None:
        del data
        self.gravity_override = np.asarray(gravity)


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


class _FakePositionActuatorWithDampingOverride:
    typ = "position"
    index = 0

    def __init__(self) -> None:
        self.kd_override: np.ndarray | None = None

    def set_damping_override(self, data: Any, value: np.ndarray) -> None:
        del data
        self.kd_override = np.asarray(value)


def _install_fake_motrix(monkeypatch, tmp_path):
    import unisim.backend.motrix.backend as mod
    import unisim.backend.motrix.scene as scene_mod

    fake_model = _FakeMotrixModel()
    _FakeTerrainScanner.instances.clear()

    monkeypatch.setattr(mod, "MOTRIX_AVAILABLE", True)
    monkeypatch.setattr(
        mod,
        "mtx",
        SimpleNamespace(
            SceneData=lambda model, batch: SimpleNamespace(num_envs=int(batch[0])),
            TerrainScanner=_FakeTerrainScanner,
            GeomHField=_FakeNativeHFieldGeom,
        ),
        raising=False,
    )
    monkeypatch.setattr(
        scene_mod,
        "materialize_scene_with_sensor_inventory",
        lambda **kwargs: (fake_model, (), ()),
    )
    # These tests exercise override caches and terrain plumbing with a fake
    # native model; unisim-core 1.7.6 validates the whole-MJCF joint order
    # against the real source file at construction (covered upstream), which
    # has no meaning for the fake model and only reads "source.xml" from disk.
    monkeypatch.setattr(
        mod.MotrixBackend,
        "_validate_non_portable_joint_order",
        lambda self, model_file: None,
        raising=False,
    )
    return mod, fake_model


def test_motrix_backend_kd_override_supports_damping_api() -> None:
    import unisim.backend.motrix.backend as mod

    actuator = _FakePositionActuatorWithDampingOverride()
    backend = object.__new__(mod.MotrixBackend)
    backend._supports_position_actuator_gains = True
    backend._position_actuators = [actuator]

    backend._set_position_actuator_kd_override(
        object(),
        np.asarray([[0.25], [0.5]], dtype=np.float32),
    )

    np.testing.assert_allclose(actuator.kd_override, [0.25, 0.5])


def test_motrix_backend_dr_capabilities_include_pd_gains_when_overrides_available() -> None:
    import unisim.backend.motrix.backend as mod

    backend = object.__new__(mod.MotrixBackend)
    backend._portable_mode = False
    backend._whole_model_variant_mode = False
    backend._supports_position_actuator_gains = True
    backend._supports_geom_friction_override = True
    backend._supports_gravity_override = True
    backend._supports_external_force = True

    caps = backend.get_dr_capabilities()

    assert RESET_TERM_KP in caps.supported_reset_terms
    assert RESET_TERM_KD in caps.supported_reset_terms
    assert RESET_TERM_BODY_MASS in caps.supported_reset_terms
    assert RESET_TERM_BODY_IPOS in caps.supported_reset_terms
    assert RESET_TERM_GEOM_FRICTION in caps.supported_reset_terms
    assert RESET_TERM_GRAVITY in caps.supported_reset_terms
    assert caps.supports_interval_term(INTERVAL_TERM_BODY_FORCE)


def test_motrix_body_sensor_views_use_canonical_world_state() -> None:
    import unisim.backend.motrix.backend as mod

    backend = object.__new__(mod.MotrixBackend)
    backend._closed = False
    backend._portable_mode = False
    backend._whole_model_variant_mode = False
    backend._portable_variant_assignment = None
    backend._num_envs = 2
    backend._np_dtype = np.float32
    backend._model = SimpleNamespace(get_link_index=lambda name: 0 if name == "base" else None)
    backend._host_get_body_state_w = lambda _body_ids: (
        np.asarray([[[1.0, 2.0, 3.0]], [[4.0, 5.0, 6.0]]], dtype=np.float32),
        np.asarray([[[1.0, 0.0, 0.0, 0.0]], [[0.0, 1.0, 0.0, 0.0]]], dtype=np.float32),
        np.asarray([[[7.0, 8.0, 9.0]], [[10.0, 11.0, 12.0]]], dtype=np.float32),
        np.asarray([[[13.0, 14.0, 15.0]], [[16.0, 17.0, 18.0]]], dtype=np.float32),
    )

    views = {
        name: backend.get_sensor_view(name).numpy()
        for name in (
            "track_pos_w_base",
            "track_quat_w_base",
            "track_linvel_w_base",
            "track_angvel_w_base",
        )
    }

    np.testing.assert_allclose(views["track_pos_w_base"], [[1, 2, 3], [4, 5, 6]])
    np.testing.assert_allclose(views["track_quat_w_base"], [[1, 0, 0, 0], [0, 1, 0, 0]])
    np.testing.assert_allclose(views["track_linvel_w_base"], [[7, 8, 9], [10, 11, 12]])
    np.testing.assert_allclose(views["track_angvel_w_base"], [[13, 14, 15], [16, 17, 18]])


def test_motrix_root_layout_gates_on_configured_base_and_returns_fixed_columns() -> None:
    import unisim.backend.motrix.backend as mod

    bodies = {
        "fixed": SimpleNamespace(floatingbase=None),
        "floating": SimpleNamespace(floatingbase=SimpleNamespace()),
    }
    backend = object.__new__(mod.MotrixBackend)
    backend._portable_mode = False
    backend._whole_model_variant_mode = False
    backend._base_name = "floating"
    backend._model = SimpleNamespace(get_body=lambda name: bodies.get(name))

    layout = backend.get_root_state_layout("floating")
    assert layout.qpos_indices == tuple(range(7))
    assert layout.qvel_indices == tuple(range(6))
    with pytest.raises(NotImplementedError, match="configured base"):
        backend.get_root_state_layout("fixed")
    with pytest.raises(NotImplementedError, match="configured base"):
        backend.get_root_state_layout("missing")


def test_motrix_backend_default_override_caches_are_float32(monkeypatch, tmp_path) -> None:
    mod, _ = _install_fake_motrix(monkeypatch, tmp_path)

    backend = mod.MotrixBackend(
        SceneCfg(model_file="source.xml"),
        num_envs=2,
        sim_dt=0.01,
        base_name="base",
    )

    assert backend.get_body_mass().dtype == np.float32
    assert backend.get_body_ipos().dtype == np.float32
    assert backend.get_geom_friction().dtype == np.float32
    kp, kd = backend.get_actuator_gains()
    assert kp.dtype == np.float32
    assert kd.dtype == np.float32


def test_motrix_backend_applies_body_mass_and_ipos_reset_payload() -> None:
    import unisim.backend.motrix.backend as mod

    link0 = _FakeMotrixLink()
    link1 = _FakeMotrixLink()
    link1.index = 1
    link1.name = "object"
    backend = object.__new__(mod.MotrixBackend)
    backend._portable_mode = False
    backend._whole_model_variant_mode = False
    backend.backend_type = "motrix"
    backend._model = SimpleNamespace(num_links=2)
    backend._links_by_id = {0: link0, 1: link1}
    backend._supports_position_actuator_gains = False

    backend._apply_reset_randomization(
        object(),
        np.asarray([0, 1], dtype=np.int32),
        {
            RESET_TERM_BODY_MASS: np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
            RESET_TERM_BODY_IPOS: np.asarray(
                [
                    [[0.0, 0.1, 0.2], [0.3, 0.4, 0.5]],
                    [[0.6, 0.7, 0.8], [0.9, 1.0, 1.1]],
                ],
                dtype=np.float64,
            ),
        },
    )

    np.testing.assert_allclose(link0.mass_override, [1.0, 3.0])
    np.testing.assert_allclose(link1.mass_override, [2.0, 4.0])
    np.testing.assert_allclose(link0.com_override, [[0.0, 0.1, 0.2], [0.6, 0.7, 0.8]])
    np.testing.assert_allclose(link1.com_override, [[0.3, 0.4, 0.5], [0.9, 1.0, 1.1]])
    assert link0.mass_override.dtype == np.float32
    assert link0.com_override.dtype == np.float32


def test_motrix_backend_applies_geom_friction_reset_payload() -> None:
    import unisim.backend.motrix.backend as mod

    geom0 = _FakeMotrixGeom(name="floor")
    geom1 = _FakeMotrixGeom(name="object")
    geom1.index = 1
    backend = object.__new__(mod.MotrixBackend)
    backend._portable_mode = False
    backend._whole_model_variant_mode = False
    backend.backend_type = "motrix"
    backend._model = SimpleNamespace(num_links=1, num_geoms=2)
    backend._geoms_by_id = {0: geom0, 1: geom1}
    backend._supports_geom_friction_override = True
    backend._supports_position_actuator_gains = False

    backend._apply_reset_randomization(
        object(),
        np.asarray([0, 1], dtype=np.int32),
        {
            RESET_TERM_GEOM_FRICTION: np.asarray(
                [
                    [[1.0, 0.1, 0.01], [2.0, 0.2, 0.02]],
                    [[3.0, 0.3, 0.03], [4.0, 0.4, 0.04]],
                ],
                dtype=np.float64,
            ),
        },
    )

    np.testing.assert_allclose(geom0.friction_override, [[1.0, 0.1, 0.01], [3.0, 0.3, 0.03]])
    np.testing.assert_allclose(geom1.friction_override, [[2.0, 0.2, 0.02], [4.0, 0.4, 0.04]])


def test_motrix_backend_skips_visual_geom_friction_override() -> None:
    import unisim.backend.motrix.backend as mod

    collision_geom = _FakeMotrixGeom(name="object")
    visual_geom = _FakeMotrixGeom(
        name="object_visual",
        collision_group=0,
        collision_affinity=0,
    )
    visual_geom.index = 1
    backend = object.__new__(mod.MotrixBackend)
    backend.backend_type = "motrix"
    backend._model = SimpleNamespace(num_geoms=2)
    backend._geoms_by_id = {0: collision_geom, 1: visual_geom}
    backend._geom_friction_override_ids = (0,)
    backend._default_geom_friction = np.asarray(
        [[1.0, 0.1, 0.01], [0.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    backend._supports_geom_friction_override = True

    geom_friction = np.broadcast_to(backend._default_geom_friction, (2, 2, 3)).copy()
    geom_friction[:, 0, :] = np.asarray([[2.0, 0.2, 0.02], [3.0, 0.3, 0.03]])
    backend._set_geom_friction_overrides(object(), geom_friction)

    np.testing.assert_allclose(collision_geom.friction_override, geom_friction[:, 0, :])
    np.testing.assert_allclose(visual_geom.friction_override, [[1.0, 0.001, 0.002]])

    geom_friction[:, 1, 0] = 1.0
    with pytest.raises(ValueError, match="non-collision geom ids"):
        backend._set_geom_friction_overrides(object(), geom_friction)


def test_motrix_backend_applies_gravity_reset_payload() -> None:
    import unisim.backend.motrix.backend as mod

    fake_model = _FakeMotrixModel()
    backend = object.__new__(mod.MotrixBackend)
    backend._portable_mode = False
    backend._whole_model_variant_mode = False
    backend.backend_type = "motrix"
    backend._model = fake_model
    backend._supports_gravity_override = True
    backend._supports_position_actuator_gains = False
    backend._supports_geom_friction_override = False

    backend._apply_reset_randomization(
        object(),
        np.asarray([0, 1], dtype=np.int32),
        {
            RESET_TERM_GRAVITY: np.asarray([[0.0, 0.0, -8.0], [1.0, 2.0, -3.0]], dtype=np.float64),
        },
    )

    assert fake_model.gravity_override is not None
    assert fake_model.gravity_override.dtype == np.float32
    np.testing.assert_allclose(fake_model.gravity_override, [[0.0, 0.0, -8.0], [1.0, 2.0, -3.0]])


def test_motrix_backend_interval_body_force_uses_link_external_force_delta() -> None:
    import unisim.backend.motrix.backend as mod

    link = _FakeMotrixLink()
    backend = object.__new__(mod.MotrixBackend)
    backend._portable_mode = False
    backend._whole_model_variant_mode = False
    backend._num_envs = 2
    backend._data = object()
    backend._links_by_id = {0: link}
    backend._supports_external_force = True
    backend._applied_body_forces = {}

    backend.apply_interval_randomization(
        IntervalRandomizationPlan(
            ops=(
                IntervalTermOp(
                    INTERVAL_TERM_BODY_FORCE,
                    torch.as_tensor([[[1.0, 2.0, 3.0]], [[4.0, 5.0, 6.0]]], dtype=torch.float64),
                    body_ids=torch.as_tensor([0], dtype=torch.int32),
                ),
            )
        )
    )
    backend.apply_body_force(
        np.asarray([0], dtype=np.int32),
        np.asarray([[[2.0, 3.0, 4.0]], [[1.0, 1.0, 1.0]]], dtype=np.float64),
    )

    assert len(link.external_force_calls) == 2
    first_force, first_local = link.external_force_calls[0]
    second_force, second_local = link.external_force_calls[1]
    assert first_local is False
    assert second_local is False
    np.testing.assert_allclose(first_force, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    np.testing.assert_allclose(second_force, [[1.0, 1.0, 1.0], [-3.0, -4.0, -5.0]])


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

    scanner = backend.bind_height_scanner(
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

    scanner = backend.bind_height_scanner(
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

    scanner = backend.bind_height_scanner(
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
