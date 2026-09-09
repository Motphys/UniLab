"""Tests for the NpEnv playback/overlay contract passthrough.

Covers the debug-overlay getter migration (unilabsim/wuji_unilab#21),
``on_frame`` fail-closed behavior, ``render(mode="rgb_array")`` gating, and
camera config forwarding — all with a stub backend, no physics required.
"""

from __future__ import annotations

from typing import Any, Tuple
from unittest.mock import MagicMock

import gymnasium as gym
import numpy as np
import pytest
from unisim.backend.base import (
    BackendPlayCapabilities,
    BackendPlayRenderPlan,
    CameraCfg,
    DebugPrimitive,
)

from unilab.base.base import EnvCfg, EnvPlayCapabilities
from unilab.base.np_env import NpEnv, NpEnvState


class _PlaybackStubEnv(NpEnv):
    """Concrete NpEnv with a configurable stub backend."""

    def __init__(self, num_envs: int = 2, *, capabilities: BackendPlayCapabilities | None = None):
        backend = MagicMock()
        backend.backend_type = "mujoco"
        backend.step.return_value = None
        backend.get_scene_model_file.return_value = None
        backend.get_play_capabilities.return_value = capabilities or BackendPlayCapabilities()
        super().__init__(EnvCfg(), backend, num_envs)

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        return {"obs": 2}

    @property
    def action_space(self) -> gym.Space:
        return gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

    def apply_action(self, actions: np.ndarray, state: NpEnvState) -> np.ndarray:
        return actions

    def update_state(self, state: NpEnvState) -> NpEnvState:
        return state

    def reset(self, env_indices: np.ndarray) -> Tuple[dict[str, np.ndarray], dict]:
        return {"obs": np.zeros((len(env_indices), 2), dtype=np.float32)}, {}


def _playback_kwargs(**overrides: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "initialize": lambda: None,
        "step": lambda obs: obs,
        "num_steps": 1,
    }
    kwargs.update(overrides)
    return kwargs


def test_run_playback_forwards_debug_overlay_getter_and_camera() -> None:
    env = _PlaybackStubEnv()
    overlay = [DebugPrimitive(kind="sphere", pos=(0.0, 0.0, 1.0), size=(0.05,))]

    def _getter():
        return [overlay, None]

    camera = CameraCfg(cam_distance=3.0)
    env.run_playback(
        **_playback_kwargs(debug_overlay_getter=_getter, camera_kwargs=camera),
    )
    call = env._backend.run_playback.call_args
    assert call.kwargs["debug_overlay_getter"] is _getter
    assert call.kwargs["camera_kwargs"] is camera
    assert "extra_data_getter" not in call.kwargs


def test_run_playback_rejects_on_frame_until_unisim_declares_it() -> None:
    env = _PlaybackStubEnv()
    with pytest.raises(NotImplementedError, match="_PlaybackStubEnv"):
        env.run_playback(**_playback_kwargs(on_frame=lambda i, frame: frame))
    env._backend.run_playback.assert_not_called()


def test_run_playback_mode_forwards_overlay_and_on_plan() -> None:
    env = _PlaybackStubEnv()
    env._backend.resolve_play_render_plan.return_value = BackendPlayRenderPlan(
        mode="record",
        headless=True,
        record_video=True,
        num_steps=3,
        output_video="out.mp4",
    )
    plans = []
    env.run_playback_mode(
        play_render_mode="record",
        play_steps=3,
        output_video="out.mp4",
        initialize=lambda: None,
        step=lambda obs: obs,
        debug_overlay_getter=lambda: None,
        on_plan=plans.append,
    )
    assert len(plans) == 1
    call = env._backend.run_playback.call_args
    assert call.kwargs["num_steps"] == 3
    assert call.kwargs["debug_overlay_getter"] is not None


def test_run_playback_mode_rejects_on_frame() -> None:
    env = _PlaybackStubEnv()
    env._backend.resolve_play_render_plan.return_value = BackendPlayRenderPlan(
        mode="record",
        headless=True,
        record_video=True,
        num_steps=1,
        output_video="out.mp4",
    )
    with pytest.raises(NotImplementedError, match="on_frame"):
        env.run_playback_mode(
            play_render_mode="record",
            play_steps=1,
            output_video="out.mp4",
            initialize=lambda: None,
            step=lambda obs: obs,
            on_frame=lambda i, frame: frame,
        )


def test_play_capabilities_forward_supports_debug_overlay() -> None:
    env = _PlaybackStubEnv(capabilities=BackendPlayCapabilities(supports_debug_overlay=True))
    assert env.play_capabilities.supports_debug_overlay is True
    assert EnvPlayCapabilities().supports_debug_overlay is False


def test_render_rgb_array_gated_on_native_video_capture() -> None:
    env = _PlaybackStubEnv()
    with pytest.raises(NotImplementedError, match="MagicMock"):
        env.render(mode="rgb_array")


def test_render_rejects_unknown_mode_with_class_name() -> None:
    env = _PlaybackStubEnv(capabilities=BackendPlayCapabilities(supports_native_video_capture=True))
    with pytest.raises(NotImplementedError, match="_PlaybackStubEnv.*human"):
        env.render(mode="human")


def test_render_rgb_array_initializes_capture_renderer_once() -> None:
    env = _PlaybackStubEnv(capabilities=BackendPlayCapabilities(supports_native_video_capture=True))
    env._backend.capture_video_frame.return_value = np.zeros((4, 6, 3), dtype=np.uint8)

    frame = env.render(mode="rgb_array")
    assert frame.shape == (4, 6, 3)
    assert frame.dtype == np.uint8
    init_call = env._backend.init_renderer.call_args
    assert init_call.kwargs["headless"] is True
    assert init_call.kwargs["capture"] is True

    env.render(mode="rgb_array")
    assert env._backend.init_renderer.call_count == 1
