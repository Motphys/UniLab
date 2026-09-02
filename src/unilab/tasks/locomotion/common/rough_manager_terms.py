"""Manager-Based terrain terms shared by the production rough quadrupeds."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from numbers import Real
from typing import TYPE_CHECKING, Any, cast
from weakref import WeakKeyDictionary

import numpy as np
from unisim.backend.base import BackendTerrainSpawnData
from unisim.terrain.generator import SubTerrainCfg, TerrainGeneratorCfg

from unilab.base.entity import EntityCfg
from unilab.dtype_config import get_global_dtype
from unilab.envs.mdp.actions.actions import JointPositionAction, JointPositionActionCfg
from unilab.envs.mdp.commands.velocity_command import (
    UniformVelocityCommand,
    UniformVelocityCommandCfg,
)
from unilab.managers.manager_base import ManagerTermBase, ManagerTermBaseCfg
from unilab.managers.scene_entity_config import SceneEntityCfg
from unilab.tasks.locomotion.common.height_scan import (
    DEFAULT_SCAN_POINTS_X,
    DEFAULT_SCAN_POINTS_Y,
    height_scan_offsets,
)
from unilab.tasks.locomotion.common.terrain_spawn import (
    TerrainCurriculumCfg,
    TerrainSpawnManager,
)
from unilab.terrains import (
    flat,
    hf_pyramid_slope,
    hf_pyramid_slope_inv,
    pyramid_stairs,
    pyramid_stairs_inv,
    random_rough,
    wave_terrain,
)
from unilab.utils.rotation import np_quat_from_euler_xyz, np_quat_mul

if TYPE_CHECKING:
    from unisim.backend.base import BackendHeightScanner

    from unilab.base.entity import Entity
    from unilab.envs.manager_based_rl_env import ManagerBasedRlEnv as RoughManagerBasedRlEnv
    from unilab.managers._types import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")
_POSE_AXES = ("x", "y", "z", "roll", "pitch", "yaw")
_VELOCITY_AXES = ("x", "y", "z", "roll", "pitch", "yaw")


def _rough_sub_terrains() -> dict[str, SubTerrainCfg]:
    return {
        "flat": flat(proportion=0.0),
        "pyramid_stairs": pyramid_stairs(
            proportion=0.1,
            step_height_range=(0.025, 0.10),
            step_width=0.4,
            platform_width=3.0,
            border_width=0.2,
        ),
        "pyramid_stairs_inv": pyramid_stairs_inv(
            proportion=0.1,
            step_height_range=(0.025, 0.10),
            step_width=0.4,
            platform_width=3.0,
            border_width=0.2,
        ),
        "hf_pyramid_slope": hf_pyramid_slope(
            proportion=0.2,
            slope_range=(0.0, 0.3),
            platform_width=2.0,
            border_width=0.2,
        ),
        "hf_pyramid_slope_inv": hf_pyramid_slope_inv(
            proportion=0.2,
            slope_range=(0.0, 0.3),
            platform_width=2.0,
            border_width=0.2,
        ),
        "random_rough": random_rough(
            proportion=0.3,
            noise_range=(0.01, 0.06),
            noise_step=0.01,
            border_width=0.2,
        ),
        "wave_terrain": wave_terrain(
            proportion=0.3,
            amplitude_range=(0.0, 0.12),
            num_waves=4,
            border_width=0.2,
        ),
    }


@dataclass(kw_only=True)
class QuadrupedRoughTerrainCfg(TerrainGeneratorCfg):
    """Shared seven-terrain production generator for Go1, Go2, and Go2W."""

    seed: int | None = 42
    curriculum: bool = False
    size: tuple[float, float] = (8.0, 8.0)
    horizontal_scale: float = 0.2
    vertical_scale: float = 0.005
    border_width: float = 20.0
    num_rows: int = 6
    num_cols: int = 6
    add_lights: bool = True
    sub_terrains: dict[str, SubTerrainCfg] = field(default_factory=_rough_sub_terrains)


def _real(
    value: Any,
    *,
    label: str,
    minimum: float | None = None,
    strict_minimum: bool = False,
) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{label} must be a real number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{label} must be finite")
    if minimum is not None and (result <= minimum if strict_minimum else result < minimum):
        relation = "greater than" if strict_minimum else "at least"
        raise ValueError(f"{label} must be {relation} {minimum}")
    return result


def _pair(value: Any, *, label: str) -> tuple[float, float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (tuple, list)):
        raise TypeError(f"{label} must be a two-value range")
    if len(value) != 2:
        raise ValueError(f"{label} must contain two values")
    lower = _real(value[0], label=f"{label}[0]")
    upper = _real(value[1], label=f"{label}[1]")
    if lower > upper:
        raise ValueError(f"{label} lower bound {lower} exceeds upper bound {upper}")
    return lower, upper


def _ranges(value: Any, axes: Sequence[str], *, label: str) -> dict[str, tuple[float, float]]:
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a mapping")
    if set(value) != set(axes):
        raise ValueError(f"{label} must declare exactly {list(axes)}, got {sorted(value)}")
    return {axis: _pair(value[axis], label=f"{label}.{axis}") for axis in axes}


def _env_ids(env: ManagerBasedRlEnv, env_ids: np.ndarray | slice | None) -> np.ndarray:
    if env_ids is None:
        return np.arange(env.num_envs, dtype=np.int32)
    if isinstance(env_ids, slice):
        return np.arange(env.num_envs, dtype=np.int32)[env_ids]
    raw = np.asarray(env_ids)
    if (
        raw.ndim != 1
        or not np.issubdtype(raw.dtype, np.integer)
        or np.issubdtype(raw.dtype, np.bool_)
    ):
        raise TypeError(f"rough terrain env_ids must be a 1-D integer array, got {raw.dtype}")
    ids = np.asarray(raw, dtype=np.int32)
    if np.any(ids < 0) or np.any(ids >= env.num_envs):
        raise IndexError(f"rough terrain env_ids out of range: {ids.tolist()}")
    if np.unique(ids).size != ids.size:
        raise ValueError(f"rough terrain env_ids contain duplicates: {ids.tolist()}")
    return ids


def _terrain_generator(env: RoughManagerBasedRlEnv) -> TerrainGeneratorCfg:
    scene = env._cfg.scene
    terrain = None if scene is None else scene.terrain
    generator = None if terrain is None else terrain.generator
    if not isinstance(generator, TerrainGeneratorCfg):
        raise TypeError("rough manager terms require SceneCfg.terrain.generator")
    return generator


def _strict_height_sampler(
    sample_height: Callable[[np.ndarray], np.ndarray],
) -> Callable[[np.ndarray], np.ndarray]:
    def sample(xy: np.ndarray) -> np.ndarray:
        points = np.asarray(xy, dtype=np.float64)
        if points.ndim < 1 or points.shape[-1] != 2:
            raise ValueError(f"terrain sample xy must have shape (..., 2), got {points.shape}")
        if not np.isfinite(points).all():
            raise ValueError("terrain sample xy contains NaN or Inf")
        heights = np.asarray(sample_height(points), dtype=np.float64)
        expected = points.shape[:-1]
        if heights.shape != expected:
            raise ValueError(
                f"terrain sample_height returned shape {heights.shape}, expected {expected}"
            )
        if not np.isfinite(heights).all():
            raise ValueError("terrain sample_height returned NaN or Inf")
        return heights

    return sample


@dataclass(frozen=True)
class _RoughTerrainContext:
    spawn_manager: TerrainSpawnManager
    generator: TerrainGeneratorCfg


_TERRAIN_CONTEXTS: WeakKeyDictionary[Any, _RoughTerrainContext] = WeakKeyDictionary()


def _materialize_terrain_context(
    env: RoughManagerBasedRlEnv,
    *,
    promote_frac: float,
    demote_frac: float,
    cycle_top_frac: float,
    spawn_height_margin: float,
) -> _RoughTerrainContext:
    existing = _TERRAIN_CONTEXTS.get(env)
    if existing is not None:
        return existing
    spawn_data = env._backend.get_terrain_spawn_data()
    if not isinstance(spawn_data, BackendTerrainSpawnData):
        raise NotImplementedError(
            "rough terrain reset requires SimBackend.get_terrain_spawn_data()"
        )
    if spawn_data.sample_height is None:
        raise NotImplementedError("rough terrain reset requires terrain sample_height")
    generator = _terrain_generator(env)
    curriculum_cfg = TerrainCurriculumCfg(
        enabled=bool(generator.curriculum),
        promote_frac=promote_frac,
        demote_frac=demote_frac,
        cycle_top_frac=cycle_top_frac,
        spawn_height_margin=spawn_height_margin,
        seed=env._cfg.seed,
    )
    context = _RoughTerrainContext(
        spawn_manager=TerrainSpawnManager(
            env.num_envs,
            spawn_data.terrain_origins,
            cell_size=float(generator.size[0]),
            cfg=curriculum_cfg,
            sample_height=_strict_height_sampler(spawn_data.sample_height),
        ),
        generator=generator,
    )
    _TERRAIN_CONTEXTS[env] = context
    return context


class RoughTerrainReset(ManagerTermBase):
    """Stage a terrain-aware randomized root state in the reset transaction."""

    def __init__(self, cfg: ManagerTermBaseCfg, env: RoughManagerBasedRlEnv):
        super().__init__(env)
        self._asset_cfg = cast(SceneEntityCfg, cfg.params.get("asset_cfg", _DEFAULT_ASSET_CFG))
        self._asset = cast("Entity", env.scene[self._asset_cfg.name])
        self._pose_range = _ranges(
            cfg.params.get("pose_range"), _POSE_AXES, label="RoughTerrainReset pose_range"
        )
        self._velocity_range = _ranges(
            cfg.params.get("velocity_range"),
            _VELOCITY_AXES,
            label="RoughTerrainReset velocity_range",
        )
        promote = _real(
            cfg.params.get("promote_frac", 0.5),
            label="RoughTerrainReset promote_frac",
            minimum=0.0,
        )
        demote = _real(
            cfg.params.get("demote_frac", 0.25),
            label="RoughTerrainReset demote_frac",
            minimum=0.0,
        )
        cycle = _real(
            cfg.params.get("cycle_top_frac", 0.5),
            label="RoughTerrainReset cycle_top_frac",
            minimum=0.0,
        )
        margin = _real(
            cfg.params.get("spawn_height_margin", 0.05),
            label="RoughTerrainReset spawn_height_margin",
            minimum=0.0,
        )
        self._context = _materialize_terrain_context(
            env,
            promote_frac=promote,
            demote_frac=demote,
            cycle_top_frac=cycle,
            spawn_height_margin=margin,
        )
        default = self._asset.data.default_root_state
        expected = (env.num_envs, 13)
        if default.shape != expected or not np.isfinite(default).all():
            raise ValueError(
                f"RoughTerrainReset default root state must be finite {expected}, got {default.shape}"
            )

    @property
    def spawn_manager(self) -> TerrainSpawnManager:
        return self._context.spawn_manager

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        env_ids: np.ndarray | slice | None,
        **params: Any,
    ) -> None:
        del params
        ids = _env_ids(env, env_ids)
        count = len(ids)
        root_state = self._asset.data.default_root_state[ids].copy()
        rng = env.rng
        root_state[:, 0] += rng.uniform(*self._pose_range["x"], size=count)
        root_state[:, 1] += rng.uniform(*self._pose_range["y"], size=count)
        root_state[:, 2] += rng.uniform(*self._pose_range["z"], size=count)
        roll = rng.uniform(*self._pose_range["roll"], size=count)
        pitch = rng.uniform(*self._pose_range["pitch"], size=count)
        yaw = rng.uniform(*self._pose_range["yaw"], size=count)
        root_state[:, 3:7] = np_quat_mul(
            root_state[:, 3:7], np_quat_from_euler_xyz(roll, pitch, yaw)
        )
        for column, axis in enumerate(_VELOCITY_AXES, start=7):
            root_state[:, column] = rng.uniform(*self._velocity_range[axis], size=count)
        root_state[:, :3] = self._context.spawn_manager.apply_spawn(
            ids,
            root_state[:, :3],
            yaw=yaw,
        )
        if not np.isfinite(root_state).all():
            raise ValueError("RoughTerrainReset produced NaN or Inf")
        self._asset.write_root_state_to_sim(root_state, env_ids=ids)
        self._context.spawn_manager.record_episode_start(ids, root_state[:, :3])


class RoughTerrainCurriculum(ManagerTermBase):
    """Settle completed episodes before the following terrain reset selects a cell."""

    def __init__(self, cfg: ManagerTermBaseCfg, env: RoughManagerBasedRlEnv):
        super().__init__(env)
        asset_cfg = cast(SceneEntityCfg, cfg.params.get("asset_cfg", _DEFAULT_ASSET_CFG))
        self._asset = cast("Entity", env.scene[asset_cfg.name])
        context = _TERRAIN_CONTEXTS.get(env)
        if context is None:
            raise RuntimeError("RoughTerrainCurriculum requires RoughTerrainReset")
        self._spawn_manager = context.spawn_manager

    def __call__(
        self,
        env: RoughManagerBasedRlEnv,
        env_ids: np.ndarray | slice | None,
        **params: Any,
    ) -> dict[str, float]:
        del params
        ids = _env_ids(env, env_ids)
        done_ids = ids[env.reset_buf[ids]]
        root_pos = self._asset.data.root_link_pos_w
        expected = (env.num_envs, 3)
        if root_pos.shape != expected or not np.isfinite(root_pos).all():
            raise ValueError(
                f"RoughTerrainCurriculum root position must be finite {expected}, got {root_pos.shape}"
            )
        return self._spawn_manager.update_on_done(done_ids, root_pos[done_ids])


class RoughTerrainOutOfBounds(ManagerTermBase):
    """Cached terrain-footprint truncation term."""

    def __init__(self, cfg: ManagerTermBaseCfg, env: RoughManagerBasedRlEnv):
        super().__init__(env)
        asset_cfg = cast(SceneEntityCfg, cfg.params.get("asset_cfg", _DEFAULT_ASSET_CFG))
        self._asset = cast("Entity", env.scene[asset_cfg.name])
        self._distance_buffer = _real(
            cfg.params.get("distance_buffer", 3.0),
            label="RoughTerrainOutOfBounds distance_buffer",
            minimum=0.0,
        )
        generator = _terrain_generator(env)
        num_cols = len(generator.sub_terrains) if generator.curriculum else generator.num_cols
        self._half_width = 0.5 * (
            generator.num_rows * float(generator.size[0]) + 2.0 * generator.border_width
        )
        self._half_height = 0.5 * (
            num_cols * float(generator.size[1]) + 2.0 * generator.border_width
        )
        if self._distance_buffer >= min(self._half_width, self._half_height):
            raise ValueError("RoughTerrainOutOfBounds distance_buffer consumes the terrain map")

    def __call__(self, env: ManagerBasedRlEnv, **params: Any) -> np.ndarray:
        del params
        root_pos = self._asset.data.root_link_pos_w
        expected = (env.num_envs, 3)
        if root_pos.shape != expected or not np.isfinite(root_pos).all():
            raise ValueError(
                f"RoughTerrainOutOfBounds root position must be finite {expected}, got {root_pos.shape}"
            )
        x_out = np.abs(root_pos[:, 0]) > self._half_width - self._distance_buffer
        y_out = np.abs(root_pos[:, 1]) > self._half_height - self._distance_buffer
        return np.asarray(x_out | y_out, dtype=np.bool_)


class RoughHeightScan(ManagerTermBase):
    """Strict cached yaw-aligned height scan in the legacy critic format."""

    def __init__(self, cfg: ManagerTermBaseCfg, env: RoughManagerBasedRlEnv):
        super().__init__(env)
        asset_cfg = cast(SceneEntityCfg, cfg.params.get("asset_cfg", _DEFAULT_ASSET_CFG))
        self._asset = cast("Entity", env.scene[asset_cfg.name])
        geom_name = cfg.params.get("geom_name", "floor")
        base_body_name = cfg.params.get("base_body_name")
        if not isinstance(geom_name, str) or not geom_name:
            raise ValueError("RoughHeightScan geom_name must be non-empty")
        if base_body_name is None:
            scene = env._cfg.scene
            if scene is None or asset_cfg.name not in scene.entities:
                raise ValueError(
                    f"RoughHeightScan scene entity '{asset_cfg.name}' is not configured"
                )
            entity_cfg = cast(EntityCfg, scene.entities[asset_cfg.name])
            base_body_name = entity_cfg.root_body_name
        if not isinstance(base_body_name, str) or not base_body_name:
            raise ValueError("RoughHeightScan requires base_body_name or an entity root_body_name")
        points_x = cfg.params.get("measured_points_x", DEFAULT_SCAN_POINTS_X)
        points_y = cfg.params.get("measured_points_y", DEFAULT_SCAN_POINTS_Y)
        if isinstance(points_x, (str, bytes)) or not isinstance(points_x, Sequence):
            raise TypeError("RoughHeightScan measured_points_x must be a sequence")
        if isinstance(points_y, (str, bytes)) or not isinstance(points_y, Sequence):
            raise TypeError("RoughHeightScan measured_points_y must be a sequence")
        offsets = height_scan_offsets(points_x, points_y)
        if offsets.shape[0] == 0 or not np.isfinite(offsets).all():
            raise ValueError("RoughHeightScan measured points must be finite and non-empty")
        self._num_points = int(offsets.shape[0])
        self._vertical_offset = _real(
            cfg.params.get("vertical_offset", 0.5), label="RoughHeightScan vertical_offset"
        )
        self._scale = _real(
            cfg.params.get("scale", 5.0),
            label="RoughHeightScan scale",
            minimum=0.0,
        )
        geom_id = env._backend.get_geom_id(geom_name)
        frame_body_id = env._backend.get_body_id(base_body_name)
        self._scanner: BackendHeightScanner = env._backend.create_hfield_scanner(
            hfield_geom_id=geom_id,
            offsets=offsets,
            frame_body_id=frame_body_id,
            alignment="yaw",
            output="height",
        )

    def __call__(self, env: ManagerBasedRlEnv, **params: Any) -> np.ndarray:
        del params
        raw = np.asarray(self._scanner.scan())
        expected = (env.num_envs, self._num_points)
        if raw.shape != expected:
            raise ValueError(
                f"RoughHeightScan scanner returned shape {raw.shape}, expected {expected}"
            )
        if not np.issubdtype(raw.dtype, np.number):
            raise TypeError(f"RoughHeightScan scanner returned non-numeric dtype {raw.dtype}")
        if not np.isfinite(raw).all():
            raise ValueError("RoughHeightScan scanner returned NaN or Inf")
        base_pos = self._asset.data.root_link_pos_w
        if base_pos.shape != (env.num_envs, 3):
            raise ValueError(
                f"RoughHeightScan root position has shape {base_pos.shape}, expected ({env.num_envs}, 3)"
            )
        if not np.isfinite(base_pos).all():
            raise ValueError("RoughHeightScan root position contains NaN or Inf")
        value = np.clip(base_pos[:, 2:3] - self._vertical_offset - raw, -1.0, 1.0)
        return np.asarray(value * self._scale, dtype=get_global_dtype())


@dataclass(kw_only=True)
class RoughJointPositionActionCfg(JointPositionActionCfg):
    """Joint-position action with legacy raw-action clipping."""

    clip_actions: float = 100.0

    def build(self, env: ManagerBasedRlEnv) -> RoughJointPositionAction:
        return RoughJointPositionAction(self, env)


class RoughJointPositionAction(JointPositionAction):
    cfg: RoughJointPositionActionCfg  # pyright: ignore[reportIncompatibleVariableOverride]

    def __init__(self, cfg: RoughJointPositionActionCfg, env: ManagerBasedRlEnv):
        self._clip_actions = _real(
            cfg.clip_actions,
            label="RoughJointPositionActionCfg clip_actions",
            minimum=0.0,
            strict_minimum=True,
        )
        super().__init__(cfg, env)
        self._clipped_input = np.empty_like(self.raw_action)

    def process_actions(self, actions: np.ndarray) -> None:
        if not isinstance(actions, np.ndarray):
            raise TypeError(
                f"RoughJointPositionAction expected np.ndarray, got {type(actions).__name__}"
            )
        if actions.shape != self._clipped_input.shape:
            raise ValueError(
                f"RoughJointPositionAction expected shape {self._clipped_input.shape}, got {actions.shape}"
            )
        np.clip(actions, -self._clip_actions, self._clip_actions, out=self._clipped_input)
        super().process_actions(self._clipped_input)


@dataclass(kw_only=True)
class RoughVelocityCommandCfg(UniformVelocityCommandCfg):
    """Rough-task velocity command with a planar-norm dead zone."""

    planar_dead_zone: float = 0.08

    def build(self, env: ManagerBasedRlEnv) -> RoughVelocityCommand:
        return RoughVelocityCommand(self, env)


class RoughVelocityCommand(UniformVelocityCommand):
    cfg: RoughVelocityCommandCfg  # pyright: ignore[reportIncompatibleVariableOverride]

    def __init__(self, cfg: RoughVelocityCommandCfg, env: ManagerBasedRlEnv):
        self._planar_dead_zone = _real(
            cfg.planar_dead_zone,
            label="RoughVelocityCommandCfg planar_dead_zone",
            minimum=0.0,
        )
        if cfg.heading_command and not np.isclose(cfg.rel_heading_envs, 1.0):
            raise ValueError(
                "RoughVelocityCommandCfg heading_command requires rel_heading_envs=1.0"
            )
        super().__init__(cfg, env)

    def _resample_command(self, env_ids: np.ndarray) -> None:
        super()._resample_command(env_ids)
        planar = self.vel_command_b[env_ids, :2]
        moving = np.linalg.norm(planar, axis=1) > self._planar_dead_zone
        self.vel_command_b[env_ids, :2] = planar * moving[:, None]


def joint_deviation_l2(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> np.ndarray:
    """Squared deviation of selected joints from the keyframe default."""
    asset = cast("Entity", env.scene[asset_cfg.name])
    delta = (
        asset.data.joint_pos[:, asset_cfg.joint_ids]
        - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    )
    if delta.ndim != 2 or delta.shape[0] != env.num_envs or not np.isfinite(delta).all():
        raise ValueError("joint_deviation_l2 received invalid joint state")
    return np.asarray(np.sum(np.square(delta), axis=1), dtype=get_global_dtype())


__all__ = [
    "QuadrupedRoughTerrainCfg",
    "RoughHeightScan",
    "RoughJointPositionAction",
    "RoughJointPositionActionCfg",
    "RoughTerrainCurriculum",
    "RoughTerrainOutOfBounds",
    "RoughTerrainReset",
    "RoughVelocityCommand",
    "RoughVelocityCommandCfg",
    "joint_deviation_l2",
]
