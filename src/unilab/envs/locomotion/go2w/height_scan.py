from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from unilab.dtype_config import get_global_dtype

DEFAULT_SCAN_POINTS_X: tuple[float, ...] = (
    -0.8,
    -0.7,
    -0.6,
    -0.5,
    -0.4,
    -0.3,
    -0.2,
    -0.1,
    0.0,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
)
DEFAULT_SCAN_POINTS_Y: tuple[float, ...] = (
    -0.5,
    -0.4,
    -0.3,
    -0.2,
    -0.1,
    0.0,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
)


@dataclass(frozen=True)
class HeightScanGrid:
    points_x: tuple[float, ...] = DEFAULT_SCAN_POINTS_X
    points_y: tuple[float, ...] = DEFAULT_SCAN_POINTS_Y

    @property
    def num_points(self) -> int:
        return len(self.points_x) * len(self.points_y)

    def local_xy(self, dtype: np.dtype | type | None = None) -> np.ndarray:
        x_grid, y_grid = np.meshgrid(
            np.asarray(self.points_x, dtype=dtype or get_global_dtype()),
            np.asarray(self.points_y, dtype=dtype or get_global_dtype()),
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
        model,
        *,
        hfield_name: str,
        geom_name: str,
        dtype: np.dtype | type | None = None,
    ) -> "HeightFieldCache":
        import mujoco

        hfield_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_HFIELD, hfield_name)
        if hfield_id < 0:
            raise ValueError(f"Height field '{hfield_name}' not found in MuJoCo model")

        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
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
        height_data = raw * float(size[2]) + float(geom_pos[2])
        out_dtype = dtype or get_global_dtype()
        return cls(
            height_data=np.asarray(height_data, dtype=out_dtype),
            x_min=float(geom_pos[0] - size[0]),
            x_max=float(geom_pos[0] + size[0]),
            y_min=float(geom_pos[1] - size[1]),
            y_max=float(geom_pos[1] + size[1]),
        )

    def __post_init__(self) -> None:
        data = np.asarray(self.height_data)
        if data.ndim != 2:
            raise ValueError(f"height_data must be 2D, got shape {data.shape}")
        if data.shape[0] < 2 or data.shape[1] < 2:
            raise ValueError(f"height_data must be at least 2x2, got shape {data.shape}")
        if not self.x_max > self.x_min:
            raise ValueError("x_max must be greater than x_min")
        if not self.y_max > self.y_min:
            raise ValueError("y_max must be greater than y_min")


@dataclass
class HeightScanner:
    cache: HeightFieldCache
    grid: HeightScanGrid = field(default_factory=HeightScanGrid)
    dtype: np.dtype | type | None = None

    def __post_init__(self) -> None:
        out_dtype = self.dtype or get_global_dtype()
        self._dtype = out_dtype
        self._local_xy = self.grid.local_xy(dtype=out_dtype)

    @property
    def num_points(self) -> int:
        return self.grid.num_points

    def scan(self, base_xy: np.ndarray, base_quat: np.ndarray) -> np.ndarray:
        base_xy = np.asarray(base_xy, dtype=self._dtype)
        base_quat = np.asarray(base_quat, dtype=self._dtype)
        if base_xy.ndim != 2 or base_xy.shape[1] != 2:
            raise ValueError(f"base_xy must have shape (N, 2), got {base_xy.shape}")
        if base_quat.ndim != 2 or base_quat.shape != (base_xy.shape[0], 4):
            raise ValueError(
                f"base_quat must have shape ({base_xy.shape[0]}, 4), got {base_quat.shape}"
            )

        yaw = _yaw_from_quat(base_quat)
        cos_yaw = np.cos(yaw)[:, None]
        sin_yaw = np.sin(yaw)[:, None]
        local_x = self._local_xy[:, 0][None, :]
        local_y = self._local_xy[:, 1][None, :]
        world_x = base_xy[:, 0:1] + cos_yaw * local_x - sin_yaw * local_y
        world_y = base_xy[:, 1:2] + sin_yaw * local_x + cos_yaw * local_y
        return self.sample_world(world_x, world_y)

    def sample_world(self, world_x: np.ndarray, world_y: np.ndarray) -> np.ndarray:
        data = np.asarray(self.cache.height_data, dtype=self._dtype)
        world_x = np.asarray(world_x, dtype=self._dtype)
        world_y = np.asarray(world_y, dtype=self._dtype)
        if world_x.shape != world_y.shape:
            raise ValueError(f"world_x/world_y shape mismatch: {world_x.shape} != {world_y.shape}")

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
        return np.asarray(h0 * (1.0 - wy) + h1 * wy, dtype=self._dtype)


def _yaw_from_quat(quat: np.ndarray) -> np.ndarray:
    w = quat[:, 0]
    x = quat[:, 1]
    y = quat[:, 2]
    z = quat[:, 3]
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
