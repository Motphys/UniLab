from __future__ import annotations

from typing import Any

import numpy as np

from unilab.envs.locomotion.go2.rough import Go2JoystickRoughEnv, _height_scan_offsets


def test_go2_rough_height_scan_uses_backend_native_sampling() -> None:
    class FakeBackend:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []
            self.base_pos = np.asarray([[0.0, 0.0, 0.6], [1.0, 0.0, 0.7]], dtype=np.float32)
            self.heights = np.asarray([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)

        def get_base_pos(self) -> np.ndarray:
            return self.base_pos

        def sample_hfield_height(self, **kwargs: Any) -> np.ndarray:
            self.calls.append(kwargs)
            return self.heights

    env = object.__new__(Go2JoystickRoughEnv)
    fake_backend = FakeBackend()
    env._backend = fake_backend
    env._height_scan_dim = 2
    env._height_scan_hfield_geom_id = 7
    env._height_scan_frame_body_id = 3
    env._height_scan_offsets = np.asarray([[0.0, 0.0], [0.1, -0.1]], dtype=np.float64)

    raw_heights, base_pos = env._raw_height_scan_obs(num_obs=2)

    np.testing.assert_array_equal(raw_heights, fake_backend.heights)
    np.testing.assert_array_equal(base_pos, fake_backend.base_pos)
    assert len(fake_backend.calls) == 1
    call = fake_backend.calls[0]
    assert call["hfield_geom_id"] == 7
    assert call["frame_body_id"] == 3
    assert call["alignment"] == "yaw"
    assert call["output"] == "height"
    np.testing.assert_array_equal(call["offsets"], env._height_scan_offsets)


def test_go2_rough_height_scan_offsets_are_grid_ordered() -> None:
    offsets = _height_scan_offsets(points_x=[-0.1, 0.2], points_y=[-0.3, 0.4])

    expected = np.asarray(
        [
            [-0.1, -0.3],
            [-0.1, 0.4],
            [0.2, -0.3],
            [0.2, 0.4],
        ],
        dtype=np.float64,
    )
    np.testing.assert_array_equal(offsets, expected)
    assert offsets.flags.c_contiguous
