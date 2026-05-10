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


class _FakeMotrixModel:
    def __init__(self) -> None:
        self.options = SimpleNamespace(timestep=None, max_iterations=None)
        self.links = [_FakeMotrixLink()]
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

    def forward_kinematic(self, data: Any) -> None:
        return None

    def get_link_poses(self, data: Any) -> np.ndarray:
        return np.zeros((1, 1, 7), dtype=np.float64)


def _install_fake_motrix(monkeypatch, tmp_path):
    import unilab.base.backend.motrix_backend as mod
    import unilab.base.backend.motrix_scene as scene_mod

    fake_model = _FakeMotrixModel()

    monkeypatch.setattr(mod, "MOTRIX_AVAILABLE", True)
    monkeypatch.setattr(
        mod,
        "mtx",
        SimpleNamespace(
            SceneData=lambda model, batch: SimpleNamespace(),
        ),
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
