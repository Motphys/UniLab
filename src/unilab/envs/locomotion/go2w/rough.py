from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base import registry
from unilab.base.scene import SceneCfg
from unilab.dtype_config import get_global_dtype
from unilab.envs.locomotion.go2w.joystick import Go2WJoystickCfg, Go2WJoystickEnv

GO2W_HEIGHT_SCAN_SCALE = 5.0
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


@dataclass
class TerrainScanConfig:
    enabled: bool = True
    hfield_name: str = "go2w_tile_stairs_5x5"
    geom_name: str = "terrain_scan_probe"
    measured_points_x: list[float] = field(default_factory=lambda: list(DEFAULT_SCAN_POINTS_X))
    measured_points_y: list[float] = field(default_factory=lambda: list(DEFAULT_SCAN_POINTS_Y))


@registry.envcfg("Go2WJoystickRoughTiles")
@dataclass
class Go2WJoystickRoughTilesCfg(Go2WJoystickCfg):
    """5x5 tiled stair terrain for MuJoCo PPO Go2W training."""

    scene: SceneCfg = field(
        default_factory=lambda: SceneCfg(
            model_file=str(ASSETS_ROOT_PATH / "robots" / "go2w" / "scene_rough_tiles.xml")
        )
    )
    terrain_scan: TerrainScanConfig = field(default_factory=TerrainScanConfig)


@registry.env("Go2WJoystickRoughTiles", sim_backend="mujoco")
class Go2WJoystickRoughTilesEnv(Go2WJoystickEnv):
    _cfg: Go2WJoystickRoughTilesCfg

    def __init__(self, cfg: Go2WJoystickRoughTilesCfg, num_envs=1, backend_type="mujoco"):
        super().__init__(cfg, num_envs=num_envs, backend_type=backend_type)
        self._init_height_scan_sensor()

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        base_spec = super().obs_groups_spec
        height_dim = getattr(self, "_height_scan_dim", self._configured_height_scan_dim())
        return {"obs": base_spec["obs"], "critic": base_spec["critic"] + height_dim}

    def _configured_height_scan_dim(self) -> int:
        scan_cfg = self._cfg.terrain_scan
        return len(scan_cfg.measured_points_x) * len(scan_cfg.measured_points_y)

    def _init_height_scan_sensor(self) -> None:
        scan_cfg = self._cfg.terrain_scan
        self._height_scan_dim = self._configured_height_scan_dim()
        if self._height_scan_dim <= 0:
            raise ValueError("terrain_scan measured points must be non-empty")

        self._height_scan_hfield_geom_id: int | None = None
        self._height_scan_frame_body_id: int | None = None
        self._height_scan_offsets: np.ndarray | None = None
        self._height_scan_sensor: object | None = None
        if not scan_cfg.enabled:
            return

        self._height_scan_hfield_geom_id = self._backend.get_geom_id(scan_cfg.geom_name)
        self._height_scan_frame_body_id = self._backend.get_body_id(self._cfg.asset.base_name)
        self._height_scan_offsets = _height_scan_offsets(
            scan_cfg.measured_points_x,
            scan_cfg.measured_points_y,
        )
        self._height_scan_sensor = self._backend.create_hfield_scanner(
            hfield_geom_id=self._height_scan_hfield_geom_id,
            offsets=self._height_scan_offsets,
            frame_body_id=self._height_scan_frame_body_id,
            alignment="yaw",
            output="height",
        )

    def _compute_obs(
        self,
        info: dict,
        linvel: np.ndarray,
        gyro: np.ndarray,
        gravity: np.ndarray,
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
    ) -> dict[str, np.ndarray]:
        obs = super()._compute_obs(info, linvel, gyro, gravity, dof_pos, dof_vel)
        num_obs = obs["critic"].shape[0]
        obs["critic"] = np.concatenate(
            [obs["critic"], self._height_scan_obs(num_obs)],
            axis=1,
            dtype=get_global_dtype(),
        )
        return obs

    def _height_scan_obs(self, num_obs: int) -> np.ndarray:
        raw_heights, base_pos = self._raw_height_scan_obs(num_obs)
        if raw_heights is None or base_pos is None:
            return np.zeros((num_obs, self._height_scan_dim), dtype=get_global_dtype())
        heights = np.clip(base_pos[:, 2:3] - 0.5 - raw_heights, -1.0, 1.0)
        return np.asarray(heights * GO2W_HEIGHT_SCAN_SCALE, dtype=get_global_dtype())

    def _reward_base_height_values(self, num_obs: int) -> np.ndarray:
        raw_heights, base_pos = self._raw_height_scan_obs(num_obs)
        if raw_heights is None or base_pos is None:
            return super()._reward_base_height_values(num_obs)
        return np.asarray(np.mean(base_pos[:, 2:3] - raw_heights, axis=1), dtype=get_global_dtype())

    def _raw_height_scan_obs(self, num_obs: int) -> tuple[np.ndarray | None, np.ndarray | None]:
        height_scan_sensor = getattr(self, "_height_scan_sensor", None)
        if (
            self._height_scan_hfield_geom_id is None
            or self._height_scan_frame_body_id is None
            or self._height_scan_offsets is None
            or height_scan_sensor is None
        ):
            return None, None

        base_pos = np.asarray(self._backend.get_base_pos(), dtype=get_global_dtype())
        if base_pos.shape[0] != num_obs:
            return None, None

        scan = getattr(height_scan_sensor, "scan", None)
        if not callable(scan):
            return None, None
        raw_heights = scan()
        if raw_heights.shape != (num_obs, self._height_scan_dim):
            return None, None
        return np.asarray(raw_heights, dtype=get_global_dtype()), base_pos


registry.register_env("Go2WJoystickRoughTiles", Go2WJoystickRoughTilesEnv, sim_backend="motrix")


def _height_scan_offsets(points_x: Sequence[float], points_y: Sequence[float]) -> np.ndarray:
    x_grid, y_grid = np.meshgrid(
        np.asarray(points_x, dtype=np.float64),
        np.asarray(points_y, dtype=np.float64),
        indexing="ij",
    )
    offsets = np.stack([x_grid.reshape(-1), y_grid.reshape(-1)], axis=1)
    return np.ascontiguousarray(offsets, dtype=np.float64)
