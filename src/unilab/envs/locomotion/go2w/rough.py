from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base import registry
from unilab.base.np_env import NpEnvState
from unilab.base.scene import SceneCfg, TerrainSceneCfg
from unilab.dtype_config import get_global_dtype
from unilab.envs.locomotion.common.height_scan import (
    HeightScanConfig,
    base_height_from_scan,
    height_scan_obs,
    init_height_scan_sensor,
    raw_height_scan_obs,
    terrain_out_of_bounds,
)
from unilab.envs.locomotion.go2.rough import Go2RoughTerrainCfg, RoughTerminationConfig
from unilab.envs.locomotion.go2w.joystick import Go2WJoystickCfg, Go2WJoystickEnv

# pyright: reportIncompatibleVariableOverride=false, reportAttributeAccessIssue=false, reportCallIssue=false


GO2W_HEIGHT_SCAN_SCALE = 5.0


@dataclass
class TerrainScanConfig(HeightScanConfig):
    """Backward-compatible alias used by Go2W rough yaml configs."""

    hfield_name: str = "terrain_hfield"
    geom_name: str = "floor"
    scale: float = GO2W_HEIGHT_SCAN_SCALE


@registry.envcfg("Go2WJoystickRough")
@dataclass
class Go2WJoystickRoughCfg(Go2WJoystickCfg):
    """Go2W rough terrain task with procedurally generated sub-terrains."""

    scene: SceneCfg = field(
        default_factory=lambda: SceneCfg(
            model_file=str(ASSETS_ROOT_PATH / "robots" / "go2w" / "go2w.xml"),
            fragment_files=[
                str(ASSETS_ROOT_PATH / "robots" / "go2w" / "locomotion_task.xml"),
            ],
            terrain=TerrainSceneCfg(
                generator=Go2RoughTerrainCfg(),
                hfield_name="terrain_hfield",
                geom_name="floor",
            ),
        )
    )
    terrain_scan: TerrainScanConfig = field(default_factory=TerrainScanConfig)
    termination_config: RoughTerminationConfig = field(default_factory=RoughTerminationConfig)


@registry.env("Go2WJoystickRough", sim_backend="mujoco")
class Go2WJoystickRoughEnv(Go2WJoystickEnv):
    _cfg: Go2WJoystickRoughCfg

    def __init__(self, cfg: Go2WJoystickRoughCfg, num_envs=1, backend_type="mujoco"):
        super().__init__(cfg, num_envs=num_envs, backend_type=backend_type)
        init_height_scan_sensor(self, cfg.terrain_scan, cfg.asset.base_name)

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        base_spec = super().obs_groups_spec
        return {"obs": base_spec["obs"], "critic": base_spec["critic"] + self._height_scan_dim}

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
            [obs["critic"], height_scan_obs(self, self._cfg.terrain_scan, num_obs)],
            axis=1,
            dtype=get_global_dtype(),
        )
        return obs

    def _reward_base_height_values(self, num_obs: int) -> np.ndarray:
        height = base_height_from_scan(self, num_obs)
        if height.shape[0] != num_obs:
            return super()._reward_base_height_values(num_obs)
        return height

    def _raw_height_scan_obs(self, num_obs: int) -> tuple[np.ndarray | None, np.ndarray | None]:
        return raw_height_scan_obs(self, num_obs)

    def _compute_truncated(self, state: NpEnvState) -> np.ndarray:
        truncated = super()._compute_truncated(state)
        if self._cfg.termination_config.terrain_out_of_bounds:
            terrain_scene = self._cfg.scene.terrain
            terrain_cfg = terrain_scene.generator if terrain_scene is not None else None
            np.logical_or(
                truncated,
                terrain_out_of_bounds(
                    self,
                    terrain_cfg,
                    float(self._cfg.termination_config.terrain_distance_buffer),
                ),
                out=truncated,
            )
        return truncated


registry.register_env("Go2WJoystickRough", Go2WJoystickRoughEnv, sim_backend="motrix")
