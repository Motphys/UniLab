from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
import torch
from unisim.backend.base import SimBackend, TerrainSpawnData
from unisim.scene import TerrainSceneCfg
from unisim.terrain.generator import TerrainGeneratorCfg

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base.entity import EntityCfg, EntityScene
from unilab.base.reset_state import ResetStateTransaction
from unilab.base.scene import SceneCfg
from unilab.base.terrain import Terrain
from unilab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg, make_manager_based_rl_env, mdp
from unilab.managers import (
    CurriculumManager,
    CurriculumTermCfg,
    EventTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
)
from unilab.managers.rng import ManagerRng
from unilab.terrains import ALL_TERRAIN_PRESETS


class _Backend:
    backend_type = "fake"

    def __init__(self, num_envs: int = 4, spawn_data: TerrainSpawnData | None = None) -> None:
        self.num_envs = num_envs
        self.spawn_data = spawn_data
        self.metadata_calls = 0

    def get_terrain_spawn_data(self) -> TerrainSpawnData | None:
        self.metadata_calls += 1
        return self.spawn_data


def _terrain_cfg(
    *, curriculum: bool = True, rows: int = 4, proportions: tuple[float, ...] = (3.0, 1.0)
) -> TerrainSceneCfg:
    return TerrainSceneCfg(
        generator=TerrainGeneratorCfg(
            size=(2.0, 2.0),
            curriculum=curriculum,
            num_rows=rows,
            num_cols=len(proportions),
            sub_terrains={
                f"terrain_{index}": SimpleNamespace(proportion=proportion)
                for index, proportion in enumerate(proportions)
            },
        )
    )


def _spawn_data(rows: int = 4, cols: int = 2) -> TerrainSpawnData:
    return TerrainSpawnData(np.arange(rows * cols * 3, dtype=np.float32).reshape(rows, cols, 3))


def _terrain(
    *,
    curriculum: bool = True,
    rows: int = 4,
    proportions: tuple[float, ...] = (3.0, 1.0),
    num_envs: int = 4,
) -> tuple[Terrain, _Backend]:
    backend = _Backend(num_envs=num_envs, spawn_data=_spawn_data(rows, len(proportions)))
    cfg = _terrain_cfg(curriculum=curriculum, rows=rows, proportions=proportions)
    return Terrain(cfg, backend, device="cpu"), backend


def _env(*, curriculum: bool = True) -> SimpleNamespace:
    terrain, _ = _terrain(curriculum=curriculum)
    return SimpleNamespace(
        num_envs=terrain.terrain_levels.numel(),
        device=torch.device("cpu"),
        common_step_counter=50,
        rng=ManagerRng.seeded(19),
        scene=SimpleNamespace(terrain=terrain),
    )


def test_flat_scene_owns_stable_zero_terrain_tensors() -> None:
    backend = _Backend()
    scene = EntityScene.from_scene_cfg(
        SceneCfg(model_file="flat.xml"), cast(SimBackend, backend), device="cpu"
    )
    terrain = scene.terrain

    assert terrain.terrain_origins.shape == (1, 1, 3)
    assert terrain.terrain_levels.shape == (4,)
    assert terrain.terrain_types.shape == (4,)
    assert terrain.env_origins.shape == (4, 3)
    assert terrain.terrain_levels.dtype == torch.int64
    assert terrain.terrain_types.dtype == torch.int64
    torch.testing.assert_close(terrain.terrain_origins, torch.zeros_like(terrain.terrain_origins))
    torch.testing.assert_close(terrain.env_origins, torch.zeros_like(terrain.env_origins))
    assert scene.env_origins("cpu") is terrain.env_origins
    assert scene.env_origins("cpu") is terrain.env_origins
    assert backend.metadata_calls == 0


def test_env_origins_other_device_returns_fresh_independent_tensor() -> None:
    scene = EntityScene.from_scene_cfg(
        SceneCfg(model_file="flat.xml"), cast(SimBackend, _Backend()), device="cpu"
    )
    owner = scene.env_origins("cpu")

    first = scene.env_origins("meta")
    second = scene.env_origins("meta")

    assert first.device == torch.device("meta")
    assert second.device == torch.device("meta")
    assert first is not owner
    assert second is not owner
    assert second is not first
    assert first.shape == owner.shape
    assert scene.env_origins("cpu") is owner


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_env_origins_cuda_copy_is_fresh_and_preserves_values() -> None:
    scene = EntityScene.from_scene_cfg(
        SceneCfg(model_file="flat.xml"), cast(SimBackend, _Backend()), device="cpu"
    )
    owner = scene.env_origins("cpu")
    owner.fill_(1.0)
    first = scene.env_origins("cuda")
    owner.fill_(2.0)
    second = scene.env_origins("cuda")

    torch.testing.assert_close(first.cpu(), torch.ones_like(owner))
    torch.testing.assert_close(second.cpu(), torch.full_like(owner, 2.0))
    assert second is not first
    assert scene.env_origins("cpu") is owner


def test_generated_terrain_selects_levels_in_place_only_for_selected_envs() -> None:
    terrain, backend = _terrain()
    levels = terrain.terrain_levels
    types = terrain.terrain_types
    origins = terrain.env_origins
    initial_origins = origins.clone()
    initial_types = types.clone()

    env_ids = torch.tensor([1, 3], dtype=torch.int64)
    selected_levels = torch.tensor([3, 2], dtype=torch.int64)
    terrain.select_levels(env_ids, selected_levels)

    assert backend.metadata_calls == 1
    assert terrain.terrain_levels is levels
    assert terrain.terrain_types is types
    assert terrain.env_origins is origins
    torch.testing.assert_close(types, initial_types)
    torch.testing.assert_close(levels, torch.tensor([0, 3, 0, 2]))
    torch.testing.assert_close(origins[0], initial_origins[0])
    torch.testing.assert_close(origins[2], initial_origins[2])
    torch.testing.assert_close(origins[1], terrain.terrain_origins[3, types[1]])
    torch.testing.assert_close(origins[3], terrain.terrain_origins[2, types[3]])
    torch.testing.assert_close(types, torch.tensor([0, 0, 0, 1]))


@pytest.mark.parametrize(
    ("num_envs", "proportions", "expected"),
    [
        (2, (1.0, 0.0, 1.0), [0, 2]),
        (1, (0.0, 1.0), [1]),
    ],
)
def test_proportional_types_never_allocate_zero_weight_columns(
    num_envs: int, proportions: tuple[float, ...], expected: list[int]
) -> None:
    terrain, _ = _terrain(num_envs=num_envs, proportions=proportions)
    torch.testing.assert_close(terrain.terrain_types, torch.tensor(expected))


def test_generated_terrain_requires_backend_metadata() -> None:
    with pytest.raises(NotImplementedError, match="get_terrain_spawn_data"):
        Terrain(_terrain_cfg(), cast(SimBackend, _Backend()), device="cpu")


def test_generated_terrain_rejects_metadata_shape_mismatch() -> None:
    backend = _Backend(spawn_data=TerrainSpawnData(np.zeros((2, 2, 3), dtype=np.float32)))
    with pytest.raises(ValueError, match=r"expected \(4, 2, 3\)"):
        Terrain(_terrain_cfg(), cast(SimBackend, backend), device="cpu")


def test_non_curriculum_terrain_selection_fails_closed() -> None:
    terrain, _ = _terrain(curriculum=False)
    with pytest.raises(NotImplementedError, match="curriculum=True"):
        terrain.select_levels(torch.tensor([0]), torch.tensor([0]))


def test_nonzero_origins_do_not_block_scene_default_reset() -> None:
    calls: list[torch.Tensor] = []
    reset_state = SimpleNamespace(reset_to_default=lambda mask, *, term_name: calls.append(mask))
    scene = EntityScene(
        {},
        cast(SimBackend, _Backend()),
        reset_state=cast(ResetStateTransaction, reset_state),
    )
    scene.terrain.env_origins.fill_(2.0)
    mask = torch.tensor([True, False, True, False])

    scene.reset_to_default(mask, term_name="reset_scene_to_default")

    assert calls == [mask]


def test_terrain_levels_by_progress_uses_manager_rng_and_selected_rows() -> None:
    env = _env()
    terrain = env.scene.terrain
    types = terrain.terrain_types.clone()
    origins = terrain.env_origins.clone()
    env_ids = torch.tensor([1, 3], dtype=torch.int64)
    expected_levels = ManagerRng.seeded(19).integers(0, 3, 2)
    manager = CurriculumManager(
        {"terrain": CurriculumTermCfg(func=mdp.terrain_levels_by_progress, params={"steps": 100})},
        env,
    )

    manager.compute(env_ids)

    torch.testing.assert_close(
        terrain.terrain_levels, torch.tensor([0, 0, 0, 0]).index_copy_(0, env_ids, expected_levels)
    )
    torch.testing.assert_close(terrain.terrain_types, types)
    torch.testing.assert_close(terrain.env_origins[0], origins[0])
    torch.testing.assert_close(terrain.env_origins[2], origins[2])
    torch.testing.assert_close(
        terrain.env_origins[env_ids], terrain.terrain_origins[expected_levels, types[env_ids]]
    )
    assert manager.reset(env_ids)["Curriculum/terrain/max_level"] == pytest.approx(2.0)
    with pytest.raises(ValueError, match="requires env_ids"):
        manager.compute()


@pytest.mark.parametrize("curriculum", [False, True])
def test_terrain_levels_by_progress_requires_curriculum_terrain(curriculum: bool) -> None:
    env = _env(curriculum=curriculum)
    if curriculum:
        env.scene.terrain = Terrain(None, cast(SimBackend, _Backend()), device="cpu")
    with pytest.raises(NotImplementedError, match="requires generated terrain"):
        mdp.terrain_levels_by_progress(env, torch.tensor([0]), steps=100)


@pytest.mark.parametrize("steps", [0, -1])
def test_terrain_levels_by_progress_rejects_nonpositive_steps(steps: int) -> None:
    env = _env()
    levels = env.scene.terrain.terrain_levels.clone()
    origins = env.scene.terrain.env_origins.clone()

    with pytest.raises(ValueError, match="positive integer"):
        mdp.terrain_levels_by_progress(env, torch.tensor([1]), steps=steps)

    torch.testing.assert_close(env.scene.terrain.terrain_levels, levels)
    torch.testing.assert_close(env.scene.terrain.env_origins, origins)


def test_real_mujoco_reset_applies_selected_terrain_origin(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip(
        "unisim.backend.mujoco.backend",
        reason="unisim-core MuJoCo adapter (mjbatch build) not available",
    )

    def observation(env: ManagerBasedRlEnv) -> torch.Tensor:
        return env.episode_length_buf[:, None].to(torch.float32)

    scene = SceneCfg(
        model_file=str(ASSETS_ROOT_PATH / "robots" / "go2" / "go2_mujoco.xml"),
        terrain=TerrainSceneCfg(
            generator=TerrainGeneratorCfg(
                size=(2.0, 2.0),
                curriculum=True,
                num_rows=2,
                sub_terrains={"flat": ALL_TERRAIN_PRESETS["flat"](proportion=1.0)},
            )
        ),
        entities={"robot": EntityCfg(root_body_name="base")},
    )
    cfg = ManagerBasedRlEnvCfg(
        scene=scene,
        sim_dt=0.01,
        ctrl_dt=0.02,
        max_episode_seconds=1.0,
        seed=19,
        observations={
            "actor": ObservationGroupCfg(terms={"state": ObservationTermCfg(func=observation)})
        },
        actions={},
        events={
            "default": EventTermCfg(func=mdp.reset_scene_to_default, mode="reset"),
            "root": EventTermCfg(
                func=mdp.reset_root_state_uniform, mode="reset", params={"pose_range": {}}
            ),
        },
        curriculum={
            "terrain": CurriculumTermCfg(func=mdp.terrain_levels_by_progress, params={"steps": 100})
        },
        policy_observation_group="actor",
    )
    env = make_manager_based_rl_env(cfg, num_envs=2, backend_type="mujoco")
    try:
        env.init_state()
        terrain = env.scene.terrain
        env_ids = torch.tensor([1], dtype=torch.int64, device=env.device)
        old_level = int(terrain.terrain_levels[1])
        target_level = 1 - old_level

        def controlled_integers(
            low: int,
            high: int,
            size: int | tuple[int, ...],
            *,
            dtype: torch.dtype = torch.int64,
        ) -> torch.Tensor:
            assert (low, high) == (0, 2)
            shape = (size,) if isinstance(size, int) else size
            return torch.full(shape, target_level, dtype=dtype, device=env.rng.device)

        monkeypatch.setattr(env.rng, "integers", controlled_integers)
        root_layout = env.backend.get_root_state_layout("base")
        position_ids = torch.tensor(root_layout.qpos_indices[:3], device=env.device)
        views = env.backend.get_state_views(("qpos", "qvel"), device=env.device)
        perturbed_qpos = views["qpos"].index_select(0, env_ids).clone()
        perturbed_qpos[:, position_ids] += torch.tensor([5.0, -5.0, 1.0], device=env.device)
        env.backend.set_state_tensor(
            env_ids, perturbed_qpos, views["qvel"].index_select(0, env_ids)
        )
        views = env.backend.get_state_views(("qpos", "qvel"), device=env.device)
        before_qpos = views["qpos"].clone()
        before_qvel = views["qvel"].clone()
        before_levels = terrain.terrain_levels.clone()
        before_origins = terrain.env_origins.clone()
        before_types = terrain.terrain_types.clone()

        env.common_step_counter = 100
        _, info = env.reset(env_ids)

        assert env.scene.env_origins(env.device) is terrain.env_origins
        assert info["log"]["Curriculum/terrain/max_level"] == pytest.approx(1.0)
        assert int(terrain.terrain_levels[1]) == target_level
        assert int(terrain.terrain_levels[1]) != old_level
        assert not torch.equal(terrain.env_origins[1], before_origins[1])
        torch.testing.assert_close(
            terrain.env_origins[1], terrain.terrain_origins[target_level, terrain.terrain_types[1]]
        )
        torch.testing.assert_close(terrain.terrain_levels[0], before_levels[0])
        torch.testing.assert_close(terrain.env_origins[0], before_origins[0])
        torch.testing.assert_close(terrain.terrain_types, before_types)
        views = env.backend.get_state_views(("qpos", "qvel"), device=env.device)
        torch.testing.assert_close(views["qpos"][0], before_qpos[0])
        torch.testing.assert_close(views["qvel"][0], before_qvel[0])
        expected = env.scene["robot"].data.default_root_state(env.device)[1, :3]
        torch.testing.assert_close(
            views["qpos"][1, position_ids], expected + terrain.env_origins[1]
        )
    finally:
        env.close()
