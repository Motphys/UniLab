from __future__ import annotations

from typing import Any

import numpy as np
import pytest

pytest.importorskip("mujoco", reason="mujoco not installed")

try:
    from mujoco.batch_env import BatchEnvPool
except Exception:
    pytest.skip(
        "mujoco.batch_env not available (platform/libstdc++ issue)",
        allow_module_level=True,
    )

if not hasattr(BatchEnvPool, "sample_hfield_height"):
    # This is a mujoco-uni 3.8 API. UniLab currently pins mujoco-uni 3.7.0rc0
    # because Sharpa simulates incorrectly on 3.8, so keep this coverage skipped.
    pytest.skip(
        "BatchEnvPool.sample_hfield_height requires mujoco-uni>=3.8.0rc2",
        allow_module_level=True,
    )

from unilab.envs.locomotion.go2w.rough import Go2WJoystickRoughEnv


def test_go2w_rough_height_scan_uses_backend_native_sampling() -> None:
    class FakeHeightScanner:
        def __init__(self, heights: np.ndarray) -> None:
            self.calls = 0
            self.heights = heights

        def scan(self) -> np.ndarray:
            self.calls += 1
            return self.heights

    class FakeBackend:
        def __init__(self) -> None:
            self.scanner_calls: list[dict[str, Any]] = []
            self.base_pos = np.asarray([[0.0, 0.0, 0.6], [1.0, 0.0, 0.7]], dtype=np.float32)
            self.heights = np.asarray([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
            self.scanner = FakeHeightScanner(self.heights)

        def get_base_pos(self) -> np.ndarray:
            return self.base_pos

        def create_hfield_scanner(self, **kwargs: Any) -> FakeHeightScanner:
            self.scanner_calls.append(kwargs)
            return self.scanner

    env = object.__new__(Go2WJoystickRoughEnv)
    fake_backend = FakeBackend()
    env._backend = fake_backend
    env._height_scan_dim = 2
    env._height_scan_hfield_geom_id = 7
    env._height_scan_frame_body_id = 3
    env._height_scan_offsets = np.asarray([[0.0, 0.0], [0.1, -0.1]], dtype=np.float64)
    env._height_scan_sensor = fake_backend.create_hfield_scanner(
        hfield_geom_id=env._height_scan_hfield_geom_id,
        offsets=env._height_scan_offsets,
        frame_body_id=env._height_scan_frame_body_id,
        alignment="yaw",
        output="height",
    )

    raw_heights, base_pos = env._raw_height_scan_obs(num_obs=2)

    np.testing.assert_array_equal(raw_heights, fake_backend.heights)
    np.testing.assert_array_equal(base_pos, fake_backend.base_pos)
    assert fake_backend.scanner.calls == 1
    assert len(fake_backend.scanner_calls) == 1
    call = fake_backend.scanner_calls[0]
    assert call["hfield_geom_id"] == 7
    assert call["frame_body_id"] == 3
    assert call["alignment"] == "yaw"
    assert call["output"] == "height"
    np.testing.assert_array_equal(call["offsets"], env._height_scan_offsets)
