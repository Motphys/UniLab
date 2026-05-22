from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from unilab.base.backend.base import SimBackend


def test_pre_step_control_default_noop() -> None:
    backend = SimpleNamespace(_pre_step_control_fn=None)
    ctrl = np.zeros((2, 3), dtype=np.float32)

    out = SimBackend._apply_pre_step_control(backend, ctrl)  # type: ignore[arg-type]

    assert out is ctrl


def test_pre_step_control_applies_registered_converter() -> None:
    backend = SimpleNamespace(_pre_step_control_fn=None)
    ctrl = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)

    SimBackend.set_pre_step_control(  # type: ignore[arg-type]
        backend,
        lambda current_backend, owner_ctrl: (
            owner_ctrl * (2.0 if current_backend is backend else 0.0)
        ),
    )

    out = SimBackend._apply_pre_step_control(backend, ctrl)  # type: ignore[arg-type]

    np.testing.assert_allclose(out, ctrl * 2.0)
    assert out.dtype == ctrl.dtype


def test_pre_step_control_rejects_shape_mismatch() -> None:
    backend = SimpleNamespace(_pre_step_control_fn=lambda current_backend, ctrl: ctrl[:, :1])
    ctrl = np.zeros((2, 3), dtype=np.float32)

    with pytest.raises(ValueError, match="pre-step control must return shape"):
        SimBackend._apply_pre_step_control(backend, ctrl)  # type: ignore[arg-type]


class _FakeMuJoCoPool:
    def __init__(self) -> None:
        self.step_calls: list[dict] = []
        self.forward_calls: list[np.ndarray] = []

    def step(
        self,
        state,
        *,
        nstep,
        control,
        control_spec,
        return_sensor=False,
        post_step_forward_sensor=False,
    ):
        self.step_calls.append(
            {
                "nstep": nstep,
                "control": np.array(control, copy=True),
                "control_spec": control_spec,
                "return_sensor": return_sensor,
                "post_step_forward_sensor": post_step_forward_sensor,
            }
        )
        state_out = np.asarray(state) + 1.0
        if return_sensor:
            return state_out, state_out[:, :1]
        return state_out

    def forward(self, state):
        state_np = np.asarray(state)
        self.forward_calls.append(state_np.copy())
        return state_np[:, :1]


def _fake_mujoco_backend(pre_step_control_fn=None):
    try:
        from unilab.base.backend.mujoco.backend import MuJoCoBackend
    except Exception as exc:
        pytest.skip(f"MuJoCo backend import unavailable: {exc}")

    backend = object.__new__(MuJoCoBackend)
    backend._pre_step_control_fn = pre_step_control_fn
    backend._num_envs = 1
    backend._np_dtype = np.float32
    backend._physics_state = np.zeros((1, 1), dtype=np.float32)
    backend._sensor_data = np.zeros((1, 1), dtype=np.float32)
    backend._pending_xfrc_applied = np.zeros((1, 0), dtype=np.float64)
    backend._post_step_forward_sensor = False
    backend._pool = _FakeMuJoCoPool()
    return backend


def test_mujoco_step_without_pre_step_control_keeps_batched_nsteps() -> None:
    backend = _fake_mujoco_backend()
    ctrl = np.array([[0.5, -0.5]], dtype=np.float32)

    backend.step(ctrl, nsteps=3)

    assert len(backend._pool.step_calls) == 1
    assert backend._pool.step_calls[0]["nstep"] == 3
    # The fused return_sensor/post_step_forward_sensor path is mujoco-uni 3.8-only.
    # Sharpa currently simulates incorrectly on 3.8, so the backend stays on the
    # 3.7 step()+forward() interface.
    assert backend._pool.step_calls[0]["return_sensor"] is False
    assert backend._pool.step_calls[0]["post_step_forward_sensor"] is False
    assert len(backend._pool.forward_calls) == 1
    expected_control = np.broadcast_to(ctrl[:, None, :], (1, 3, ctrl.shape[-1]))
    np.testing.assert_allclose(backend._pool.step_calls[0]["control"], expected_control)
    np.testing.assert_allclose(backend._physics_state, [[1.0]])
    np.testing.assert_allclose(backend._sensor_data, [[1.0]])


def test_mujoco_step_with_pre_step_control_recomputes_each_physics_step() -> None:
    seen_sensors: list[np.ndarray] = []

    backend = _fake_mujoco_backend()

    def hook(current_backend, owner_ctrl: np.ndarray) -> np.ndarray:
        seen_sensors.append(current_backend._sensor_data.copy())
        return owner_ctrl + len(seen_sensors)

    backend.set_pre_step_control(hook)
    ctrl = np.array([[0.5, -0.5]], dtype=np.float32)

    backend.step(ctrl, nsteps=3)

    assert len(backend._pool.step_calls) == 3
    assert [call["nstep"] for call in backend._pool.step_calls] == [1, 1, 1]
    assert all(call["return_sensor"] is False for call in backend._pool.step_calls)
    assert all(call["post_step_forward_sensor"] is False for call in backend._pool.step_calls)
    assert len(backend._pool.forward_calls) == 3
    np.testing.assert_allclose(seen_sensors, [[[0.0]], [[1.0]], [[2.0]]])
    np.testing.assert_allclose(backend._pool.step_calls[0]["control"], (ctrl + 1)[:, None, :])
    np.testing.assert_allclose(backend._pool.step_calls[1]["control"], (ctrl + 2)[:, None, :])
    np.testing.assert_allclose(backend._pool.step_calls[2]["control"], (ctrl + 3)[:, None, :])
    np.testing.assert_allclose(backend._physics_state, [[3.0]])
    np.testing.assert_allclose(backend._sensor_data, [[3.0]])


class _FakeMotrixModel:
    def __init__(self) -> None:
        self.step_calls = 0
        self.step_n_calls: list[int] = []

    def step(self, data) -> None:
        self.step_calls += 1
        data.sensor_value += 1.0

    def step_n(self, data, nsteps: int) -> None:
        self.step_n_calls.append(nsteps)
        data.sensor_value += float(nsteps)


def _fake_motrix_backend(pre_step_control_fn=None):
    from unilab.base.backend.motrix.backend import MotrixBackend

    backend = object.__new__(MotrixBackend)
    backend._pre_step_control_fn = pre_step_control_fn
    backend._model = _FakeMotrixModel()
    backend._data = SimpleNamespace(
        actuator_ctrls=np.zeros((1, 2), dtype=np.float32),
        sensor_value=0.0,
    )
    backend._refresh_link_pose_cache = lambda: None
    return backend


def test_motrix_step_with_pre_step_control_uses_single_step_loop() -> None:
    seen_sensors: list[float] = []
    backend = _fake_motrix_backend()

    def hook(current_backend, owner_ctrl: np.ndarray) -> np.ndarray:
        seen_sensors.append(float(current_backend._data.sensor_value))
        return owner_ctrl + len(seen_sensors)

    backend.set_pre_step_control(hook)
    ctrl = np.array([[1.0, 2.0]], dtype=np.float32)

    backend.step(ctrl, nsteps=3)

    assert backend._model.step_calls == 3
    assert backend._model.step_n_calls == []
    assert seen_sensors == [0.0, 1.0, 2.0]
    np.testing.assert_allclose(backend._data.actuator_ctrls, ctrl + 3)


def test_motrix_native_video_capture_uses_headless_system_camera(monkeypatch) -> None:
    import unilab.base.backend.motrix.backend as mod

    captured: dict[str, object] = {}

    class FakeCameras:
        def set_system_render_target(self, target, width, height):
            captured["render_target"] = (target, width, height)

    class FakeModel:
        cameras = FakeCameras()

    class FakeSettings:
        enable_shadow = False

    class FakeRenderSettings:
        @staticmethod
        def performance():
            captured["settings"] = True
            return FakeSettings()

    class FakeImage:
        pixels = np.arange(2 * 3 * 4, dtype=np.uint8).reshape(2, 3, 4)

    class FakeTask:
        def take_image(self):
            captured["take_image"] = True
            return FakeImage()

    class FakeSystemCamera:
        def set_view(self, lookat, distance, elevation, azimuth):
            captured["set_view"] = {
                "lookat": lookat,
                "distance": distance,
                "elevation": elevation,
                "azimuth": azimuth,
            }

        def capture(self):
            captured["capture"] = True
            return FakeTask()

    class FakeRenderApp:
        def __init__(self, **kwargs):
            captured["render_app_kwargs"] = kwargs
            self.system_camera = FakeSystemCamera()

        def launch(self, model, *, batch, render_offset, render_settings):
            captured["launch"] = {
                "model": model,
                "batch": batch,
                "render_offset": render_offset,
                "render_settings": render_settings,
            }

        def sync(self, *, data, wait=False):
            captured["sync"] = {"data": data, "wait": wait}

    monkeypatch.setattr(mod, "RenderApp", FakeRenderApp, raising=False)
    monkeypatch.setattr(mod, "RenderSettings", FakeRenderSettings, raising=False)

    backend = object.__new__(mod.MotrixBackend)
    backend._model = FakeModel()
    backend._data = object()
    backend._num_envs = 3
    backend._render_app = None
    backend._render_headless = None
    backend._render_capture_enabled = False

    backend.init_renderer(
        spacing=1.5,
        headless=True,
        capture=True,
        width=3,
        height=2,
        camera_kwargs={
            "cam_lookat": [1.0, 2.0, 3.0],
            "cam_distance": 4.0,
            "cam_elevation": -30.0,
            "cam_azimuth": 45.0,
        },
    )
    frame = backend.capture_video_frame()

    assert captured["render_target"] == ("image", 3, 2)
    assert captured["render_app_kwargs"] == {"headless": True}
    assert captured["launch"]["model"] is backend._model
    assert captured["launch"]["batch"] == 3
    assert captured["launch"]["render_offset"] == [
        [0.0, 0.0, 0.0],
        [1.5, 0.0, 0.0],
        [0.0, 1.5, 0.0],
    ]
    assert captured["set_view"] == {
        "lookat": [1.0, 2.0, 3.0],
        "distance": 4.0,
        "elevation": -30.0,
        "azimuth": 45.0,
    }
    assert captured["sync"] == {"data": backend._data, "wait": True}
    assert captured["capture"] is True
    assert captured["take_image"] is True
    assert frame.shape == (2, 3, 3)
    np.testing.assert_array_equal(frame, FakeImage.pixels[..., :3])


def test_motrix_native_video_capture_defaults_camera_lookat_to_grid_center(monkeypatch) -> None:
    import unilab.base.backend.motrix.backend as mod

    captured: dict[str, object] = {}

    class FakeCameras:
        def set_system_render_target(self, target, width, height):
            del target, width, height

    class FakeModel:
        cameras = FakeCameras()

    class FakeSettings:
        enable_shadow = False

    class FakeRenderSettings:
        @staticmethod
        def performance():
            return FakeSettings()

    class FakeSystemCamera:
        def set_view(self, lookat, distance, elevation, azimuth):
            captured["set_view"] = {
                "lookat": lookat,
                "distance": distance,
                "elevation": elevation,
                "azimuth": azimuth,
            }

    class FakeRenderApp:
        def __init__(self, **kwargs):
            del kwargs
            self.system_camera = FakeSystemCamera()

        def launch(self, model, *, batch, render_offset, render_settings):
            del model, batch, render_offset, render_settings

    monkeypatch.setattr(mod, "RenderApp", FakeRenderApp, raising=False)
    monkeypatch.setattr(mod, "RenderSettings", FakeRenderSettings, raising=False)

    backend = object.__new__(mod.MotrixBackend)
    backend._model = FakeModel()
    backend._num_envs = 4
    backend._render_app = None
    backend._render_headless = None
    backend._render_capture_enabled = False

    backend.init_renderer(
        spacing=2.0,
        headless=True,
        capture=True,
        width=3,
        height=2,
        camera_kwargs=None,
    )

    assert captured["set_view"] == {
        "lookat": [1.0, 1.0, 0.75],
        "distance": 2.0,
        "elevation": -20.0,
        "azimuth": 90.0,
    }


def test_motrix_native_video_capture_tracks_primary_env_base(monkeypatch) -> None:
    import unilab.base.backend.motrix.backend as mod

    captured: dict[str, object] = {"set_views": []}
    base_positions = np.array(
        [
            [0.0, 0.0, 0.5],
            [2.0, 3.0, 0.75],
            [4.0, 5.0, 1.0],
        ],
        dtype=np.float64,
    )

    class FakeImage:
        pixels = np.zeros((2, 3, 3), dtype=np.uint8)

    class FakeCaptureTask:
        def take_image(self):
            return FakeImage()

    class FakeCameras:
        def set_system_render_target(self, target, width, height):
            captured["render_target"] = (target, width, height)

    class FakeModel:
        cameras = FakeCameras()

    class FakeSettings:
        enable_shadow = False

    class FakeRenderSettings:
        @staticmethod
        def performance():
            return FakeSettings()

    class FakeSystemCamera:
        def set_view(self, lookat, distance, elevation, azimuth):
            captured["set_views"].append(
                {
                    "lookat": lookat,
                    "distance": distance,
                    "elevation": elevation,
                    "azimuth": azimuth,
                }
            )

        def capture(self):
            captured["capture"] = True
            return FakeCaptureTask()

    class FakeRenderApp:
        def __init__(self, **kwargs):
            captured["render_app_kwargs"] = kwargs
            self.system_camera = FakeSystemCamera()

        def launch(self, model, *, batch, render_offset, render_settings):
            captured["launch"] = {
                "model": model,
                "batch": batch,
                "render_offset": render_offset,
                "render_settings": render_settings,
            }

        def sync(self, *, data, wait=False):
            captured["sync"] = {"data": data, "wait": wait}

    monkeypatch.setattr(mod, "RenderApp", FakeRenderApp, raising=False)
    monkeypatch.setattr(mod, "RenderSettings", FakeRenderSettings, raising=False)

    backend = object.__new__(mod.MotrixBackend)
    backend._model = FakeModel()
    backend._data = object()
    backend._num_envs = 3
    backend._render_app = None
    backend._render_headless = None
    backend._render_capture_enabled = False
    backend.get_base_pos = lambda: base_positions

    backend.init_renderer(
        spacing=1.5,
        headless=True,
        capture=True,
        width=3,
        height=2,
        camera_kwargs={
            "cam_tracking": True,
            "cam_tracking_env_idx": 1,
            "cam_distance": 6.0,
            "cam_elevation": -30.0,
            "cam_azimuth": 45.0,
        },
    )

    assert captured["set_views"] == [
        {
            "lookat": [3.5, 3.0, 0.75],
            "distance": 6.0,
            "elevation": -30.0,
            "azimuth": 45.0,
        }
    ]

    base_positions[1] = [10.0, 20.0, 1.25]
    frame = backend.capture_video_frame()

    assert captured["set_views"][-1] == {
        "lookat": [11.5, 20.0, 1.25],
        "distance": 6.0,
        "elevation": -30.0,
        "azimuth": 45.0,
    }
    assert captured["sync"] == {"data": backend._data, "wait": True}
    assert captured["capture"] is True
    assert frame.shape == (2, 3, 3)


def test_motrix_interactive_renderer_applies_camera_kwargs(monkeypatch) -> None:
    import unilab.base.backend.motrix.backend as mod

    captured: dict[str, object] = {}

    class FakeModel:
        pass

    class FakeSettings:
        enable_shadow = False

    class FakeRenderSettings:
        @staticmethod
        def performance():
            return FakeSettings()

    class FakeSystemCamera:
        def set_view(self, lookat, distance, elevation, azimuth):
            captured["set_view"] = {
                "lookat": lookat,
                "distance": distance,
                "elevation": elevation,
                "azimuth": azimuth,
            }

    class FakeRenderApp:
        def __init__(self, **kwargs):
            captured["render_app_kwargs"] = kwargs
            self.system_camera = FakeSystemCamera()

        def launch(self, model, *, batch, render_offset, render_settings):
            captured["launch"] = {
                "model": model,
                "batch": batch,
                "render_offset": render_offset,
                "render_settings": render_settings,
            }

        def set_main_camera(self, camera):
            captured["main_camera"] = camera

    monkeypatch.setattr(mod, "RenderApp", FakeRenderApp, raising=False)
    monkeypatch.setattr(mod, "RenderSettings", FakeRenderSettings, raising=False)

    backend = object.__new__(mod.MotrixBackend)
    backend._model = FakeModel()
    backend._num_envs = 1
    backend._render_app = None
    backend._render_headless = None
    backend._render_capture_enabled = False

    backend.init_renderer(
        spacing=1.0,
        camera_kwargs={
            "cam_lookat": [10.0, 20.0, 0.5],
            "cam_distance": 4.0,
            "cam_elevation": -25.0,
            "cam_azimuth": 135.0,
        },
    )

    assert captured["render_app_kwargs"] == {"headless": False}
    assert captured["launch"]["batch"] == 1
    assert captured["set_view"] == {
        "lookat": [10.0, 20.0, 0.5],
        "distance": 4.0,
        "elevation": -25.0,
        "azimuth": 135.0,
    }
    assert captured["main_camera"] is None


def test_motrix_renderer_zero_offset_mode(monkeypatch) -> None:
    import unilab.base.backend.motrix.backend as mod

    captured: dict[str, object] = {}

    class FakeModel:
        pass

    class FakeSettings:
        enable_shadow = False

    class FakeRenderSettings:
        @staticmethod
        def performance():
            return FakeSettings()

    class FakeRenderApp:
        def __init__(self, **kwargs):
            del kwargs

        def launch(self, model, *, batch, render_offset, render_settings):
            del model, batch, render_settings
            captured["render_offset"] = render_offset

    monkeypatch.setattr(mod, "RenderApp", FakeRenderApp, raising=False)
    monkeypatch.setattr(mod, "RenderSettings", FakeRenderSettings, raising=False)

    backend = object.__new__(mod.MotrixBackend)
    backend._model = FakeModel()
    backend._num_envs = 4
    backend._render_app = None
    backend._render_headless = None
    backend._render_capture_enabled = False

    backend.init_renderer(spacing=2.0, offset_mode="zero")

    assert captured["render_offset"] == [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ]
