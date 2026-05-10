from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base import registry
from unilab.base.scene import SceneCfg, TerrainSceneCfg
from unilab.dtype_config import get_global_dtype
from unilab.envs.locomotion.go2.joystick import (
    Go2JoystickCfg,
    Go2WalkTask,
    RewardConfig,
)
from unilab.terrains import (
    SubTerrainCfg,
    TerrainGeneratorCfg,
    flat,
    hf_pyramid_slope,
    hf_pyramid_slope_inv,
    pyramid_stairs,
    pyramid_stairs_inv,
    random_rough,
    wave_terrain,
)

# pyright: reportIncompatibleVariableOverride=false, reportAttributeAccessIssue=false, reportCallIssue=false

GO2_HEIGHT_SCAN_SCALE = 5.0
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
    geom_name: str = "floor"
    measured_points_x: list[float] = field(default_factory=lambda: list(DEFAULT_SCAN_POINTS_X))
    measured_points_y: list[float] = field(default_factory=lambda: list(DEFAULT_SCAN_POINTS_Y))
    vertical_offset: float = 0.5
    scale: float = GO2_HEIGHT_SCAN_SCALE


@dataclass(kw_only=True)
class Go2RoughTerrainCfg(TerrainGeneratorCfg):
    size: tuple[float, float] = (8.0, 8.0)
    num_rows: int = 1
    num_cols: int = 1
    border_width: float = 1.0
    add_lights: bool = False

    sub_terrains: dict[str, SubTerrainCfg] = field(
        default_factory=lambda: {
            "flat": flat(proportion=0.0),
            "pyramid_stairs": pyramid_stairs(
                proportion=0.0,
                step_height_range=(0.025, 0.10),
                step_width=0.3,
                platform_width=3.0,
                border_width=1.0,
            ),
            "pyramid_stairs_inv": pyramid_stairs_inv(
                proportion=0.0,
                step_height_range=(0.025, 0.10),
                step_width=0.3,
                platform_width=3.0,
                border_width=1.0,
            ),
            "hf_pyramid_slope": hf_pyramid_slope(
                proportion=0.0,
                slope_range=(0.0, 0.7),
                platform_width=2.0,
                border_width=0.25,
            ),
            "hf_pyramid_slope_inv": hf_pyramid_slope_inv(
                proportion=0.0,
                slope_range=(0.0, 0.7),
                platform_width=2.0,
                border_width=0.25,
            ),
            "random_rough": random_rough(
                proportion=0.2,
                noise_range=(0.01, 0.06),
                noise_step=0.01,
                border_width=0.25,
            ),
            "wave_terrain": wave_terrain(
                proportion=0.0,
                amplitude_range=(0.0, 0.12),
                num_waves=4,
                border_width=0.25,
            ),
        }
    )


@registry.envcfg("Go2JoystickRough")
@dataclass
class Go2JoystickRoughCfg(Go2JoystickCfg):
    scene: SceneCfg = field(
        default_factory=lambda: SceneCfg(
            model_file=str(ASSETS_ROOT_PATH / "robots" / "go2" / "go2.xml"),
            fragment_files=[
                str(ASSETS_ROOT_PATH / "robots" / "go2" / "locomotion_task.xml"),
            ],
            terrain=TerrainSceneCfg(
                generator=Go2RoughTerrainCfg(),
                hfield_name="terrain_hfield",
                geom_name="floor",
            ),
        )
    )
    terrain_scan: TerrainScanConfig = field(default_factory=TerrainScanConfig)
    reward_config: RewardConfig | None = None


@registry.env("Go2JoystickRough", sim_backend="mujoco")
class Go2JoystickRoughEnv(Go2WalkTask):
    _cfg: Go2JoystickRoughCfg
    _reward_cfg: RewardConfig

    def __init__(self, cfg: Go2JoystickRoughCfg, num_envs=1, backend_type="mujoco"):
        self._height_scan_dim = len(cfg.terrain_scan.measured_points_x) * len(
            cfg.terrain_scan.measured_points_y
        )
        super().__init__(cfg, num_envs=num_envs, backend_type=backend_type)
        self._init_height_scan_sensor()

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        flat_spec = super().obs_groups_spec
        return {"obs": flat_spec["obs"], "critic": flat_spec["critic"] + self._height_scan_dim}

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
        if not scan_cfg.enabled:
            return

        self._height_scan_hfield_geom_id = self._backend.get_geom_id(scan_cfg.geom_name)
        self._height_scan_frame_body_id = self._backend.get_body_id(self._cfg.asset.base_name)
        self._height_scan_offsets = _height_scan_offsets(
            scan_cfg.measured_points_x,
            scan_cfg.measured_points_y,
        )

    def _compute_obs(
        self,
        info: dict,
        linvel: np.ndarray,
        gyro: np.ndarray,
        gravity: np.ndarray,
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
        feet_phase: np.ndarray,
    ) -> dict[str, np.ndarray]:
        obs = super()._compute_obs(info, linvel, gyro, gravity, dof_pos, dof_vel, feet_phase)
        critic = np.concatenate(
            [obs["critic"], self._height_scan_obs(obs["critic"].shape[0])],
            axis=1,
            dtype=get_global_dtype(),
        )
        obs["critic"] = critic
        return obs

    def _height_scan_obs(self, num_obs: int) -> np.ndarray:
        raw_heights, base_pos = self._raw_height_scan_obs(num_obs)
        if raw_heights is None or base_pos is None:
            return np.zeros((num_obs, self._height_scan_dim), dtype=get_global_dtype())
        scan_cfg = self._cfg.terrain_scan
        heights = np.clip(base_pos[:, 2:3] - scan_cfg.vertical_offset - raw_heights, -1.0, 1.0)
        return np.asarray(heights * scan_cfg.scale, dtype=get_global_dtype())

    def _raw_height_scan_obs(self, num_obs: int) -> tuple[np.ndarray | None, np.ndarray | None]:
        if (
            self._height_scan_hfield_geom_id is None
            or self._height_scan_frame_body_id is None
            or self._height_scan_offsets is None
        ):
            return None, None

        base_pos = np.asarray(self._backend.get_base_pos(), dtype=get_global_dtype())
        if base_pos.shape[0] != num_obs:
            return None, None

        raw_heights = self._backend.sample_hfield_height(
            hfield_geom_id=self._height_scan_hfield_geom_id,
            offsets=self._height_scan_offsets,
            frame_body_id=self._height_scan_frame_body_id,
            alignment="yaw",
            output="height",
        )
        if raw_heights.shape != (num_obs, self._height_scan_dim):
            return None, None
        return np.asarray(raw_heights, dtype=get_global_dtype()), base_pos


def _height_scan_offsets(points_x: Sequence[float], points_y: Sequence[float]) -> np.ndarray:
    x_grid, y_grid = np.meshgrid(
        np.asarray(points_x, dtype=np.float64),
        np.asarray(points_y, dtype=np.float64),
        indexing="ij",
    )
    offsets = np.stack([x_grid.reshape(-1), y_grid.reshape(-1)], axis=1)
    return np.ascontiguousarray(offsets, dtype=np.float64)


registry.register_env("Go2JoystickRough", Go2WalkTask, sim_backend="motrix")
