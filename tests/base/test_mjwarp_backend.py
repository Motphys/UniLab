"""Real-CUDA correctness tests for the production ``mjwarp`` host profile.

These are slow by design.  They fail when explicitly invoked without the
``mjwarp`` extra or CUDA on platforms where CUDA is supported; the normal
optional-backend unit lane deselects them instead of treating an unavailable
fake implementation as evidence.  On macOS they skip: the factory's CUDA-only
platform gate fail-closes there, so the lane can never run.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from unisim.backend.mjwarp.dependencies import load_mjwarp_dependencies

from unilab.base import registry
from unilab.base.backend_factory import create_backend
from unilab.base.config_adapter import EnvConfigAdapter
from unilab.base.scene import SceneCfg

pytestmark = pytest.mark.slow
REPO_ROOT = Path(__file__).resolve().parents[2]


def _require_cuda_mjwarp() -> None:
    if sys.platform == "darwin":
        pytest.skip("mjwarp is a CUDA-only backend; macOS has no supported CUDA runtime")
    dependencies = load_mjwarp_dependencies()
    if not bool(dependencies.warp.get_device().is_cuda):
        pytest.fail("mjwarp correctness tests require an active CUDA Warp device")


def _scene() -> SceneCfg:
    from unilab.assets import ASSETS_ROOT_PATH

    return SceneCfg(model_file=str(ASSETS_ROOT_PATH / "robots" / "g1" / "scene_flat.xml"))


def _backend(num_envs: int = 3) -> Any:
    _require_cuda_mjwarp()
    return create_backend("mjwarp", _scene(), num_envs, 0.02 / 3.0, base_name="pelvis")


def _stand_state(backend: Any, count: int) -> tuple[np.ndarray, np.ndarray]:
    qpos = np.tile(backend.get_keyframe_qpos("stand"), (count, 1))
    qvel = np.zeros((count, backend.get_init_qvel().size), dtype=np.float32)
    return qpos.astype(np.float32), qvel


def test_real_cuda_init_reset_step(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = _backend(2)
    assert backend.backend_type == "mjwarp"
    assert backend.num_actuators == 29
    assert backend.num_dof_vel == 29
    caches = (
        (backend._qpos_cache_storage, backend._qpos_cache),
        (backend._qvel_cache_storage, backend._qvel_cache),
        (backend._sensor_cache_storage, backend._sensor_cache),
    )
    for storage, cache in caches:
        assert storage.device.is_cpu
        assert storage.pinned
        assert cache.ctypes.data == storage.ptr
    cache_pointers = tuple(cache.ctypes.data for _, cache in caches)
    root_layout = backend.get_root_state_layout("pelvis")
    assert root_layout.qpos_indices == tuple(range(7))
    assert root_layout.qvel_indices == tuple(range(6))

    graph_launches: list[Any] = []
    original_capture_launch = backend._warp.capture_launch

    def capture_launch(graph: Any) -> None:
        graph_launches.append(graph)
        original_capture_launch(graph)

    monkeypatch.setattr(backend._warp, "capture_launch", capture_launch)

    qpos, qvel = _stand_state(backend, 2)
    backend.set_state(np.asarray([0, 1], dtype=np.int32), qpos, qvel)
    before = backend.get_base_pos().copy()
    result = backend.step(np.tile(qpos[0, -backend.num_actuators :], (2, 1)), nsteps=3)

    assert set(result["timing"]) == {"control_upload_ms", "physics_ms", "host_cache_refresh_ms"}
    if backend._cuda_graph_enabled:
        assert graph_launches == [
            backend._reset_graph,
            backend._forward_graph,
            backend._step_graph,
            backend._step_graph,
            backend._step_graph,
        ]
    else:
        assert graph_launches == []
        assert backend._cuda_graph_disable_reason
    assert np.isfinite(backend.get_base_pos()).all()
    assert np.isfinite(backend.get_dof_pos()).all()
    assert np.isfinite(backend.get_sensor_data("torso_upvector")).all()
    assert not np.array_equal(backend.get_base_pos(), before)
    assert tuple(cache.ctypes.data for _, cache in caches) == cache_pointers


def test_selected_row_reset_isolated() -> None:
    backend = _backend(4)
    qpos, qvel = _stand_state(backend, 4)
    qpos[:, 0] = np.asarray([-0.3, -0.1, 0.1, 0.3], dtype=np.float32)
    qvel[:, 0] = np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    backend.set_state(np.arange(4, dtype=np.int32), qpos, qvel)
    previous_pos = backend.get_base_pos().copy()
    previous_vel = backend.get_base_lin_vel().copy()

    rows = np.asarray([3, 1], dtype=np.int32)
    reset_qpos, reset_qvel = _stand_state(backend, 2)
    reset_qpos[:, 0] = np.asarray([0.45, -0.45], dtype=np.float32)
    reset_qvel[:, 0] = np.asarray([0.8, -0.8], dtype=np.float32)
    backend.set_state(rows, reset_qpos, reset_qvel)

    np.testing.assert_allclose(backend.get_base_pos()[rows, 0], reset_qpos[:, 0])
    np.testing.assert_allclose(backend.get_base_lin_vel()[rows, 0], reset_qvel[:, 0])
    complement = np.asarray([0, 2], dtype=np.int32)
    np.testing.assert_allclose(backend.get_base_pos()[complement], previous_pos[complement])
    np.testing.assert_allclose(backend.get_base_lin_vel()[complement], previous_vel[complement])
    assert np.isfinite(backend.get_sensor_data("pelvis_local_linvel")).all()


def test_large_batch_sparse_reset_matches_full_forward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The bounded reset graph preserves reset sensors and the following step."""
    _require_cuda_mjwarp()
    num_envs = 1024
    backend = create_backend(
        "mjwarp",
        _scene(),
        num_envs,
        0.02 / 3.0,
        base_name="pelvis",
        body_state_required=True,
    )
    capacity = backend._reset_scratch_capacity
    assert capacity == 128
    assert backend._reset_scratch_forward_graph is not None

    rng = np.random.default_rng(7)
    all_rows = np.arange(num_envs, dtype=np.int32)
    all_qpos, all_qvel = _stand_state(backend, num_envs)
    rows = np.sort(rng.choice(num_envs, 32, replace=False)).astype(np.int32)
    reset_qpos, reset_qvel = _stand_state(backend, len(rows))
    reset_qpos[:, 0] += rng.uniform(-0.2, 0.2, len(rows)).astype(np.float32)
    reset_qvel[:] = rng.uniform(-0.2, 0.2, reset_qvel.shape).astype(np.float32)
    ctrl = np.zeros((num_envs, backend.num_actuators), dtype=np.float32)

    def snapshot() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            backend._qpos_cache[rows].copy(),
            backend._qvel_cache[rows].copy(),
            backend._sensor_cache[rows].copy(),
        )

    backend.set_state(all_rows, all_qpos, all_qvel)
    backend._reset_scratch_capacity = 0
    backend.set_state(rows, reset_qpos, reset_qvel)
    full_reset = snapshot()
    backend.step(ctrl, nsteps=3)
    full_step = snapshot()

    backend._reset_scratch_capacity = capacity
    backend.set_state(all_rows, all_qpos, all_qvel)
    graph_launches: list[Any] = []
    original_capture_launch = backend._warp.capture_launch

    def capture_launch(graph: Any) -> None:
        graph_launches.append(graph)
        original_capture_launch(graph)

    monkeypatch.setattr(backend._warp, "capture_launch", capture_launch)
    backend.set_state(rows, reset_qpos, reset_qvel)
    assert graph_launches == [
        backend._reset_graph,
        backend._reset_scratch_reset_graph,
        backend._reset_scratch_forward_graph,
    ]
    sparse_reset = snapshot()
    backend.step(ctrl, nsteps=3)
    sparse_step = snapshot()

    for actual, expected in zip(sparse_reset[:2], full_reset[:2], strict=True):
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_allclose(sparse_reset[2], full_reset[2], rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(sparse_step[0], full_step[0], rtol=1e-3, atol=2e-5)
    np.testing.assert_allclose(sparse_step[1], full_step[1], rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(sparse_step[2], full_step[2], rtol=1e-3, atol=1e-2)


@pytest.mark.parametrize(
    ("config_group", "overrides", "algo_name"),
    [
        pytest.param("ppo", ["task=g1_walk_flat/mjwarp"], "ppo", id="ppo"),
        pytest.param(
            "sac",
            ["task=g1_walk_flat/mjwarp"],
            "sac",
            id="sac",
        ),
    ],
)
def test_g1_walk_flat_owner_one_step(
    config_group: str,
    overrides: list[str],
    algo_name: str,
) -> None:
    _require_cuda_mjwarp()
    GlobalHydra.instance().clear()
    with initialize_config_dir(
        config_dir=str(REPO_ROOT / "src" / "unilab" / "conf" / config_group), version_base="1.3"
    ):
        cfg = compose("config", overrides=overrides)

    env_cfg_override = EnvConfigAdapter(
        cfg,
        root_dir=REPO_ROOT,
    ).build_task_env_cfg_override()
    registry.ensure_registries()
    env = registry.make(
        str(cfg.training.task_name),
        sim_backend=str(cfg.training.sim_backend),
        env_cfg_override=env_cfg_override,
        num_envs=2,
    )

    action_dim = int(env.action_space.shape[-1])
    state = env.step(torch.zeros((2, action_dim), dtype=torch.float32, device=env.device))

    assert set(state.obs) == {"obs", "critic"}
    assert state.obs["obs"].shape == (2, 98)
    assert state.obs["critic"].shape == (2, 101)
    assert torch.isfinite(state.obs["obs"]).all()
    assert torch.isfinite(state.obs["critic"]).all()
    assert torch.isfinite(state.reward).all()


def test_g1_walk_flat_owner_eager_capture_warmup_and_replay_share_semantics() -> None:
    """The full Manager trajectory is invariant across MJWarp execution modes."""
    _require_cuda_mjwarp()
    GlobalHydra.instance().clear()
    with initialize_config_dir(
        config_dir=str(REPO_ROOT / "src" / "unilab" / "conf" / "ppo"),
        version_base="1.3",
    ):
        cfg = compose("config", overrides=["task=g1_walk_flat/mjwarp"])
    env_cfg_override = EnvConfigAdapter(
        cfg,
        root_dir=REPO_ROOT,
    ).build_task_env_cfg_override()
    registry.ensure_registries()

    references: dict[str, list[dict[str, object]]] = {}
    common_seed = 20261013
    controls = (0.1, -0.2, 0.3)
    masks = (
        torch.zeros(2, dtype=torch.bool),
        torch.tensor([False, True], dtype=torch.bool),
        torch.ones(2, dtype=torch.bool),
    )

    for mode in ("eager", "capture-warmup", "graph-replay"):
        env = registry.make(
            str(cfg.training.task_name),
            sim_backend=str(cfg.training.sim_backend),
            env_cfg_override=env_cfg_override,
            num_envs=2,
        )
        env.seed(common_seed)
        try:
            env.init_state()
            env.seed(common_seed)
            if mode == "eager":
                env.backend._disable_cuda_graphs("test requests eager execution")
            else:
                assert env.backend._cuda_graph_enabled

            action_dim = int(env.action_space.shape[-1])
            action = torch.zeros((2, action_dim), dtype=torch.float32, device=env.device)
            # This first public step is capture warmup for the graph backend;
            # forcing eager makes it the same semantic warmup boundary.
            env.step(action)

            trace: list[dict[str, object]] = []
            for control_value, reset_mask in zip(controls, masks, strict=True):
                action.fill_(control_value)
                state = env.step(action)
                reset_obs, _reset_info = env.reset(
                    env_ids=torch.nonzero(
                        reset_mask.to(device=env.device), as_tuple=False
                    ).flatten()
                )
                after_reset = env.step(torch.zeros_like(action))
                completion = after_reset.info.get("completion")
                assert completion is not None
                assert completion.operation == "step"
                assert completion.producer_id is not None
                trace.append(
                    {
                        "obs": {name: value.clone() for name, value in state.obs.items()},
                        "reward": state.reward.clone(),
                        "terminated": state.terminated.clone(),
                        "truncated": state.truncated.clone(),
                        "steps": state.info["steps"].clone(),
                        "reset_obs": {name: value.clone() for name, value in reset_obs.items()},
                        "after_obs": {
                            name: value.clone() for name, value in after_reset.obs.items()
                        },
                        "after_reward": after_reset.reward.clone(),
                        "rng_state": env.rng.get_state().clone(),
                    }
                )
            references[mode] = trace
        finally:
            env.close()

    eager = references["eager"]
    for graph_mode in ("capture-warmup", "graph-replay"):
        candidate = references[graph_mode]
        for expected, actual in zip(eager, candidate, strict=True):
            expected_row = cast(dict[str, object], expected)
            actual_row = cast(dict[str, object], actual)
            expected_obs = cast(dict[str, torch.Tensor], expected_row["obs"])
            actual_obs = cast(dict[str, torch.Tensor], actual_row["obs"])
            for name in expected_obs:
                torch.testing.assert_close(expected_obs[name], actual_obs[name], atol=5e-5, rtol=0)
            for field in ("reward", "terminated", "truncated", "steps"):
                torch.testing.assert_close(
                    cast(torch.Tensor, expected_row[field]), cast(torch.Tensor, actual_row[field])
                )
            for field in ("reset_obs", "after_obs"):
                expected_group = cast(dict[str, torch.Tensor], expected_row[field])
                actual_group = cast(dict[str, torch.Tensor], actual_row[field])
                for name in expected_group:
                    torch.testing.assert_close(
                        expected_group[name], actual_group[name], atol=5e-5, rtol=0
                    )
            torch.testing.assert_close(
                cast(torch.Tensor, expected_row["after_reward"]),
                cast(torch.Tensor, actual_row["after_reward"]),
            )
            torch.testing.assert_close(
                cast(torch.Tensor, expected_row["rng_state"]),
                cast(torch.Tensor, actual_row["rng_state"]),
            )


def test_body_state_matches_mujoco_backend() -> None:
    """Tracked body kinematics on mjwarp match the MuJoCo backend on identical state."""
    import mujoco
    from unisim.backend.mujoco.backend import MuJoCoBackend

    _require_cuda_mjwarp()
    num_envs = 2
    sim_dt = 0.02 / 3.0
    scene = _scene()
    mujoco_backend = MuJoCoBackend(
        scene,
        num_envs,
        sim_dt,
        base_name="pelvis",
        add_body_sensors=True,
    )
    mujoco_backend.materialize()
    mjwarp_backend = create_backend(
        "mjwarp",
        scene,
        num_envs,
        sim_dt,
        base_name="pelvis",
        body_state_required=True,
    )

    rng = np.random.default_rng(0)
    qpos = np.tile(mjwarp_backend.get_keyframe_qpos("stand"), (num_envs, 1))
    qvel = rng.uniform(-0.5, 0.5, size=(num_envs, int(mujoco_backend.model.nv))).astype(np.float32)
    rows = np.arange(num_envs, dtype=np.int32)
    mujoco_backend.set_state(rows, qpos, qvel)
    mjwarp_backend.set_state(rows, qpos, qvel)

    body_names = ["pelvis", "left_hip_pitch_link", "left_knee_link"]
    body_ids = np.asarray(
        [
            mujoco.mj_name2id(mujoco_backend.model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in body_names
        ],
        dtype=np.int32,
    )
    assert (body_ids > 0).all()

    atol = 2e-3
    np.testing.assert_allclose(
        mjwarp_backend.get_body_pos_w(body_ids),
        mujoco_backend.get_body_pos_w(body_ids),
        atol=atol,
    )
    np.testing.assert_allclose(
        mjwarp_backend.get_body_quat_w(body_ids),
        mujoco_backend.get_body_quat_w(body_ids),
        atol=atol,
    )
    np.testing.assert_allclose(
        mjwarp_backend.get_body_lin_vel_w(body_ids),
        mujoco_backend.get_body_lin_vel_w(body_ids),
        atol=atol,
    )
    np.testing.assert_allclose(
        mjwarp_backend.get_body_ang_vel_w(body_ids),
        mujoco_backend.get_body_ang_vel_w(body_ids),
        atol=atol,
    )
    expected_state = mjwarp_backend.get_body_state_w(body_ids)
    row_ids = np.asarray([1, 0], dtype=np.int32)
    mjwarp_pose_rows = mjwarp_backend.get_body_pose_w_rows(row_ids, body_ids)
    for actual, expected in zip(mjwarp_pose_rows, expected_state[:2], strict=True):
        np.testing.assert_allclose(actual, expected[row_ids], atol=atol)
    np.testing.assert_allclose(
        mjwarp_backend.get_body_lin_vel_w_rows(row_ids, body_ids),
        expected_state[2][row_ids],
        atol=atol,
    )
    np.testing.assert_allclose(
        mjwarp_backend.get_body_ang_vel_w_rows(row_ids, body_ids),
        expected_state[3][row_ids],
        atol=atol,
    )
    outputs = tuple(np.empty_like(value) for value in expected_state)
    result = mjwarp_backend.copy_body_state_w(body_ids, *outputs)
    assert result == outputs
    for actual, expected in zip(outputs, expected_state, strict=True):
        np.testing.assert_allclose(actual, expected, atol=atol)
    np.testing.assert_allclose(
        mjwarp_backend.get_body_lin_vel_b(body_ids),
        mujoco_backend.get_body_lin_vel_b(body_ids),
        atol=atol,
    )
    np.testing.assert_allclose(
        mjwarp_backend.get_body_ang_vel_b(body_ids),
        mujoco_backend.get_body_ang_vel_b(body_ids),
        atol=atol,
    )


def test_set_state_returns_schema_conformant_timing() -> None:
    """Issue #1295: mjwarp set_state reports the shared keyset plus its granular
    reset_upload / reset_forward / host_cache_refresh sub-timings."""
    from unilab.base.backend_timing import BACKEND_SET_STATE_DETAIL_TIMING_KEYS

    backend = _backend(2)
    qpos, qvel = _stand_state(backend, 2)

    result = backend.set_state(np.asarray([0, 1], dtype=np.int32), qpos, qvel)
    timing = result["timing"]
    missing = [key for key in BACKEND_SET_STATE_DETAIL_TIMING_KEYS if key not in timing]
    assert not missing, f"missing keys in set_state timing: {missing}"
    for key in BACKEND_SET_STATE_DETAIL_TIMING_KEYS:
        value = timing[key]
        assert isinstance(value, float), f"{key} must be float, got {type(value)!r}"
        if not key.endswith("internal_gap_ms"):
            assert value >= 0.0, f"{key} must be non-negative, got {value}"
    assert abs(timing["set_state_internal_gap_ms"]) < 5.0
    # mjwarp populates its own sub-keys; other backends' sub-keys stay 0.0.
    assert timing["set_state_reset_upload_ms"] > 0.0
    assert timing["set_state_reset_forward_ms"] > 0.0
    assert timing["set_state_host_cache_refresh_ms"] > 0.0
    assert timing["set_state_mask_ms"] == 0.0
    assert timing["set_state_pool_reset_ms"] == 0.0

    empty = backend.set_state(
        np.asarray([], dtype=np.int32),
        np.zeros((0, qpos.shape[1]), dtype=np.float32),
        np.zeros((0, qvel.shape[1]), dtype=np.float32),
    )
    empty_timing = empty["timing"]
    missing = [key for key in BACKEND_SET_STATE_DETAIL_TIMING_KEYS if key not in empty_timing]
    assert not missing, f"missing keys in empty set_state timing: {missing}"
    assert all(value == 0.0 for value in empty_timing.values())


def test_body_state_untracked_body_ids_fail_closed() -> None:
    """Body ids outside the injected tracking set raise instead of wrapping around."""
    _require_cuda_mjwarp()
    backend = create_backend(
        "mjwarp",
        _scene(),
        1,
        0.02 / 3.0,
        base_name="pelvis",
        body_state_required=True,
    )
    with pytest.raises(ValueError, match="without injected tracking sensors"):
        backend.get_body_pos_w(np.asarray([0], dtype=np.int32))
