"""FR3 owner composition and optional native CPU rollout integration."""

from __future__ import annotations

import multiprocessing as mp
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from unilab import cli
from unilab.base import registry
from unilab.base.base import EnvCfg
from unilab.base.config_adapter import BackendAdapter
from unilab.base.config_materialization import apply_cfg_overrides
from unilab.base.env_factory import registry_env_factory
from unilab.envs import ManagerBasedRlEnvCfg

ROOT = Path(__file__).resolve().parents[2]


def _owner() -> tuple[Any, dict[str, Any]]:
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(ROOT / "src/unilab/conf/ppo"), version_base="1.3"):
        owner = compose("config", overrides=["task=fr3_joint_target/superdex"])
    return owner, BackendAdapter(owner, root_dir=ROOT).build_task_env_cfg_override()


def test_fr3_owner_has_torque_control_without_free_root_terms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner, overrides = _owner()
    cfg = registry.materialize_env_config("FR3JointTarget")
    assert isinstance(cfg, ManagerBasedRlEnvCfg)
    apply_cfg_overrides(cfg, overrides)
    cfg.validate()
    assert cfg.scene is not None
    assert cfg.scene.entities["robot"].root_body_name == "fr3_link0"
    assert list(cfg.events) == ["reset_scene_to_default", "reset_joints"]
    assert list(cfg.observations["policy"].terms) == ["target_error", "joint_vel", "actions"]
    assert cfg.superdex_effort_limits == [20.0] * 4 + [5.0] * 3
    assert cfg.superdex_num_workers == 0
    assert owner.training.play_render_mode == "interactive"
    assert owner.training.play_env_num == 1 and owner.training.device == "cpu"
    assert "superdex" in cli.SUPPORTED_SIMS
    monkeypatch.setattr(cli, "_check_runtime_requirements", lambda *_: None)
    command = cli.build_command(
        mode="train", algo="ppo", task="fr3_joint_target", sim="superdex", overrides=[]
    )
    assert "task=fr3_joint_target/superdex" in command


@pytest.mark.parametrize(
    "kwargs",
    [
        {"superdex_num_workers": True},
        {"superdex_num_workers": -1},
        {"superdex_num_workers": 1.5},
        {"superdex_assets_root": ""},
        {"superdex_effort_limits": [0.0]},
        {"superdex_effort_limits": [float("nan")]},
    ],
)
def test_superdex_owner_options_reject_invalid_values(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="superdex"):
        EnvCfg(**kwargs).validate()


def _require_runtime() -> None:
    pytest.importorskip("superdex.physics")
    pytest.importorskip("superdex.robotics")
    from unilab.assets.hub import resolve_superdex_robot_asset

    try:
        resolve_superdex_robot_asset("bots/arms/fr3_v2/fr3_v2.superdex_bot")
    except Exception as exc:  # noqa: BLE001 - any resolution failure means skip
        pytest.skip(
            "native FR3 integration needs the FR3 assets: allow Hugging Face download "
            f"or set SUPERDEX_ASSETS_PATH to a local checkout ({exc})"
        )


def test_fr3_native_rollout_and_selected_reset() -> None:
    _require_runtime()
    _, overrides = _owner()
    env = registry.make(
        "FR3JointTarget", sim_backend="superdex", num_envs=2, env_cfg_override=overrides
    )
    try:
        state = env.init_state()
        assert env.obs_groups_spec == {"obs": 21}
        assert env.action_space.shape == (7,)
        for _ in range(32):
            state = env.step(np.full((2, 7), 0.05, dtype=np.float32))
            assert np.isfinite(state.obs["obs"]).all()
            assert np.isfinite(state.reward).all()
        before = env.scene["robot"].data.joint_pos.copy()
        counters = state.info["steps"].copy()
        obs, _ = env.reset(env_ids=np.array([0], dtype=np.int32))
        assert obs["obs"].shape == (1, 21)
        np.testing.assert_array_equal(env.scene["robot"].data.joint_pos[1], before[1])
        assert state.info["steps"][1] == counters[1]
        np.testing.assert_allclose(
            obs["obs"][0, :7],
            env.scene["robot"].data.joint_pos[0] - np.array([0.1, -0.7, 0, -2.2, 0, 1.5, 1.57]),
            atol=1e-6,
        )
    finally:
        env.close()
    with pytest.raises(RuntimeError, match="closed"):
        env.step(np.zeros((2, 7), dtype=np.float32))


def _spawn_rollout(overrides: dict[str, Any]) -> tuple[int, ...]:
    factory = registry_env_factory("FR3JointTarget", "superdex")
    env = factory(num_envs=1, env_cfg_override=overrides)
    try:
        obs, _ = env.reset(env_ids=np.array([0], dtype=np.int32))
        env.step(np.zeros((1, 7), dtype=np.float32))
        return obs["obs"].shape
    finally:
        env.close()


def _spawn_parallel_rollout(overrides: dict[str, Any]) -> tuple[int, ...]:
    factory = registry_env_factory("FR3JointTarget", "superdex")
    env = factory(num_envs=2, env_cfg_override={**overrides, "superdex_num_workers": 2})
    try:
        obs, _ = env.reset(env_ids=np.array([0, 1], dtype=np.int32))
        env.step(np.zeros((2, 7), dtype=np.float32))
        return obs["obs"].shape
    finally:
        env.close()


def test_fr3_factory_survives_spawn() -> None:
    _require_runtime()
    _, overrides = _owner()
    with mp.get_context("spawn").Pool(1) as pool:
        result = pool.apply_async(_spawn_rollout, (overrides,))
        assert result.get(timeout=90) == (1, 21)


def test_fr3_parallel_backend_can_run_inside_spawn_collector() -> None:
    _require_runtime()
    _, overrides = _owner()
    with mp.get_context("spawn").Pool(1) as pool:
        result = pool.apply_async(_spawn_parallel_rollout, (overrides,))
        assert result.get(timeout=150) == (2, 21)
