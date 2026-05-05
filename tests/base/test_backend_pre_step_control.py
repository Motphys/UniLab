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

    def step(self, state, *, nstep, control, control_spec):
        self.step_calls.append(
            {
                "nstep": nstep,
                "control": np.array(control, copy=True),
                "control_spec": control_spec,
            }
        )
        return np.asarray(state) + 1.0

    def forward(self, state):
        state_np = np.asarray(state)
        self.forward_calls.append(state_np.copy())
        return state_np[:, :1]


def _fake_mujoco_backend(pre_step_control_fn=None):
    try:
        from unilab.base.backend.mujoco_backend import MuJoCoBackend
    except Exception as exc:
        pytest.skip(f"MuJoCo backend import unavailable: {exc}")

    backend = object.__new__(MuJoCoBackend)
    backend._pre_step_control_fn = pre_step_control_fn
    backend._num_envs = 1
    backend._np_dtype = np.float32
    backend._physics_state = np.zeros((1, 1), dtype=np.float32)
    backend._sensor_data = np.zeros((1, 1), dtype=np.float32)
    backend._pending_xfrc_applied = np.zeros((1, 0), dtype=np.float64)
    backend._pool = _FakeMuJoCoPool()
    return backend


def test_mujoco_step_without_pre_step_control_keeps_batched_nsteps() -> None:
    backend = _fake_mujoco_backend()
    ctrl = np.array([[0.5, -0.5]], dtype=np.float32)

    backend.step(ctrl, nsteps=3)

    assert len(backend._pool.step_calls) == 1
    assert backend._pool.step_calls[0]["nstep"] == 3
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
    from unilab.base.backend.motrix_backend import MotrixBackend

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
