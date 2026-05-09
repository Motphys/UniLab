from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator

import numpy as np
import pytest

pytest.importorskip("mujoco", reason="mujoco not installed")

try:
    import mujoco
    from mujoco.batch_env import BatchEnvPool
except Exception:
    pytest.skip(
        "mujoco.batch_env not available (platform/libstdc++ issue)",
        allow_module_level=True,
    )

if not hasattr(BatchEnvPool, "sample_hfield_height"):
    pytest.skip(
        "BatchEnvPool.sample_hfield_height requires mujoco-uni>=3.8.0rc2",
        allow_module_level=True,
    )

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base.backend.mujoco_backend import MuJoCoBackend
from unilab.base.scene import SceneCfg
from unilab.envs.common.rotation import np_yaw_to_quat
from unilab.envs.locomotion.go2w.rough import (
    Go2WJoystickRoughTilesEnv,
    TerrainScanConfig,
    _height_scan_offsets,
)

mj: Any = mujoco


@dataclass(frozen=True)
class HeightScanGrid:
    points_x: tuple[float, ...]
    points_y: tuple[float, ...]

    def local_xy(self, dtype: np.dtype | type = np.float64) -> np.ndarray:
        x_grid, y_grid = np.meshgrid(
            np.asarray(self.points_x, dtype=dtype),
            np.asarray(self.points_y, dtype=dtype),
            indexing="ij",
        )
        return np.stack([x_grid.reshape(-1), y_grid.reshape(-1)], axis=1)


@dataclass(frozen=True)
class HeightFieldCache:
    height_data: np.ndarray
    x_min: float
    x_max: float
    y_min: float
    y_max: float

    @classmethod
    def from_mujoco_model(
        cls,
        model: Any,
        *,
        hfield_name: str,
        geom_name: str,
    ) -> "HeightFieldCache":
        hfield_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_HFIELD, hfield_name)
        if hfield_id < 0:
            raise ValueError(f"Height field '{hfield_name}' not found in MuJoCo model")

        geom_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, geom_name)
        if geom_id < 0:
            raise ValueError(f"Geom '{geom_name}' not found in MuJoCo model")
        if int(model.geom_dataid[geom_id]) != int(hfield_id):
            raise ValueError(f"Geom '{geom_name}' is not backed by height field '{hfield_name}'")

        nrow = int(model.hfield_nrow[hfield_id])
        ncol = int(model.hfield_ncol[hfield_id])
        adr = int(model.hfield_adr[hfield_id])
        size = np.asarray(model.hfield_size[hfield_id], dtype=np.float64)
        geom_pos = np.asarray(model.geom_pos[geom_id], dtype=np.float64)
        raw = np.asarray(model.hfield_data[adr : adr + nrow * ncol], dtype=np.float64).reshape(
            nrow, ncol
        )
        return cls(
            height_data=raw * float(size[2]) + float(geom_pos[2]),
            x_min=float(geom_pos[0] - size[0]),
            x_max=float(geom_pos[0] + size[0]),
            y_min=float(geom_pos[1] - size[1]),
            y_max=float(geom_pos[1] + size[1]),
        )


@dataclass
class HeightScanner:
    cache: HeightFieldCache
    grid: HeightScanGrid
    dtype: np.dtype | type = np.float64
    _local_xy: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._local_xy = self.grid.local_xy(dtype=self.dtype)

    def scan(self, base_xy: np.ndarray, base_quat: np.ndarray) -> np.ndarray:
        base_xy = np.asarray(base_xy, dtype=self.dtype)
        base_quat = np.asarray(base_quat, dtype=self.dtype)
        yaw = _yaw_from_quat(base_quat)
        cos_yaw = np.cos(yaw)[:, None]
        sin_yaw = np.sin(yaw)[:, None]
        local_x = self._local_xy[:, 0][None, :]
        local_y = self._local_xy[:, 1][None, :]
        world_x = base_xy[:, 0:1] + cos_yaw * local_x - sin_yaw * local_y
        world_y = base_xy[:, 1:2] + sin_yaw * local_x + cos_yaw * local_y
        return self.sample_world(world_x, world_y)

    def sample_world(self, world_x: np.ndarray, world_y: np.ndarray) -> np.ndarray:
        data = np.asarray(self.cache.height_data, dtype=self.dtype)
        rows, cols = data.shape
        x = (world_x - self.cache.x_min) / (self.cache.x_max - self.cache.x_min) * (cols - 1)
        y = (world_y - self.cache.y_min) / (self.cache.y_max - self.cache.y_min) * (rows - 1)
        x = np.clip(x, 0.0, cols - 1.0)
        y = np.clip(y, 0.0, rows - 1.0)

        x0 = np.floor(x).astype(np.int32)
        y0 = np.floor(y).astype(np.int32)
        x1 = np.minimum(x0 + 1, cols - 1)
        y1 = np.minimum(y0 + 1, rows - 1)
        wx = x - x0
        wy = y - y0

        h00 = data[y0, x0]
        h10 = data[y0, x1]
        h01 = data[y1, x0]
        h11 = data[y1, x1]
        h0 = h00 * (1.0 - wx) + h10 * wx
        h1 = h01 * (1.0 - wx) + h11 * wx
        return np.asarray(h0 * (1.0 - wy) + h1 * wy, dtype=self.dtype)


def _yaw_from_quat(quat: np.ndarray) -> np.ndarray:
    w = quat[:, 0]
    x = quat[:, 1]
    y = quat[:, 2]
    z = quat[:, 3]
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


@pytest.fixture
def go2w_rough_backend() -> Iterator[MuJoCoBackend]:
    model_file = str(ASSETS_ROOT_PATH / "robots" / "go2w" / "scene_rough_tiles.xml")
    backend = MuJoCoBackend(
        SceneCfg(model_file=model_file), num_envs=3, sim_dt=0.01, base_name="base_link"
    )
    backend.materialize()
    try:
        yield backend
    finally:
        pool = getattr(backend, "_pool", None)
        if pool is not None:
            pool.close()


def test_mujoco_backend_hfield_sampling_matches_python_reference(
    go2w_rough_backend: MuJoCoBackend,
) -> None:
    backend = go2w_rough_backend
    qpos = np.tile(backend.get_default_qpos(), (backend.num_envs, 1))
    qvel = np.zeros((backend.num_envs, backend.nv), dtype=np.float64)
    qpos[:, 0:3] = np.asarray(
        [
            [0.0, 0.0, 0.6],
            [2.2, -1.5, 0.6],
            [-3.5, 4.0, 0.6],
        ],
        dtype=np.float64,
    )
    qpos[:, 3:7] = np_yaw_to_quat(np.asarray([0.0, 0.7, -1.2], dtype=np.float64))
    backend.set_state(np.arange(backend.num_envs, dtype=np.int32), qpos, qvel)

    scan_cfg = TerrainScanConfig(
        measured_points_x=[-0.8, -0.3, 0.0, 0.25, 0.8],
        measured_points_y=[-0.45, -0.1, 0.2, 0.5],
    )
    grid = HeightScanGrid(
        points_x=tuple(scan_cfg.measured_points_x),
        points_y=tuple(scan_cfg.measured_points_y),
    )
    reference = HeightScanner(
        cache=HeightFieldCache.from_mujoco_model(
            backend.model,
            hfield_name=scan_cfg.hfield_name,
            geom_name=scan_cfg.geom_name,
        ),
        grid=grid,
    ).scan(backend.get_base_pos()[:, :2], backend.get_base_quat())

    native = backend.sample_hfield_height(
        hfield_geom_id=backend.get_geom_id(scan_cfg.geom_name),
        offsets=_height_scan_offsets(scan_cfg.measured_points_x, scan_cfg.measured_points_y),
        frame_body_id=backend.get_body_id("base_link"),
        alignment="yaw",
        output="height",
    )

    assert native.shape == reference.shape
    np.testing.assert_allclose(native, reference, atol=1e-6, rtol=0)


def test_go2w_rough_height_scan_uses_backend_native_sampling() -> None:
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

    env = object.__new__(Go2WJoystickRoughTilesEnv)
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
