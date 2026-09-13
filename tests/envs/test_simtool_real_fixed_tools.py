"""Representative SimToolReal fixed-tool Manager-Based integration."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import pytest
from uni_rl.algos.rsl_rl import RslRlVecEnvWrapper

from unilab.envs import make_manager_based_rl_env
from unilab.tasks.manipulation.simtool_real import (
    build_representative_simtool_real_env_cfg,
    write_representative_simtool_real_sources,
)


def _object_names(
    mujoco: Any,
    model: Any,
    object_type: int,
    count: int,
) -> tuple[str, ...]:
    return tuple(mujoco.mj_id2name(model, object_type, index) or "" for index in range(count))


def test_generated_sources_preserve_public_layout_and_vary_model_fields(
    tmp_path: Path,
) -> None:
    mujoco = pytest.importorskip("mujoco", reason="SimToolReal layout audit requires MuJoCo")
    sources = write_representative_simtool_real_sources(tmp_path, variant_count=4)
    models = [mujoco.MjModel.from_xml_path(str(path)) for path in sources.model_files]

    for model in models:
        for name in ("nq", "nv", "nu", "nbody", "njnt", "ngeom", "nsensor"):
            assert getattr(model, name) == getattr(models[0], name)
        for object_type, count_name in (
            (mujoco.mjtObj.mjOBJ_BODY, "nbody"),
            (mujoco.mjtObj.mjOBJ_JOINT, "njnt"),
            (mujoco.mjtObj.mjOBJ_GEOM, "ngeom"),
            (mujoco.mjtObj.mjOBJ_MESH, "nmesh"),
            (mujoco.mjtObj.mjOBJ_ACTUATOR, "nu"),
        ):
            named = _object_names(mujoco, model, object_type, int(getattr(model, count_name)))
            reference = _object_names(
                mujoco, models[0], object_type, int(getattr(models[0], count_name))
            )
            assert named == reference

    np.testing.assert_allclose(
        [model.body_mass[1] for model in models],
        [0.4 + 0.25 * index for index in range(4)],
        rtol=0.0,
        atol=0.0,
    )
    assert len({model.body_inertia[1].tobytes() for model in models}) == len(models)
    assert len({model.geom_size[1].tobytes() for model in models}) == len(models)


def test_cpu_manager_rollout_uses_immutable_variant_identity_and_reset_dr(
    tmp_path: Path,
) -> None:
    pytest.importorskip(
        "unisim.backend.mujoco.backend",
        reason="SimToolReal CPU rollout requires the MuJoCo adapter",
    )
    sources = write_representative_simtool_real_sources(tmp_path)
    cfg = build_representative_simtool_real_env_cfg(sources)
    env = make_manager_based_rl_env(cfg, num_envs=6, backend_type="mujoco")
    try:
        plan = cfg.scene.fixed_variant_plan
        assert plan is not None
        np.testing.assert_array_equal(plan.assignment, np.tile(np.arange(3, dtype=np.int32), 2))

        backend = env._backend
        default_mass = backend.get_reset_term_default("body_mass")
        assert default_mass.shape == (6, backend.model.nbody)
        np.testing.assert_allclose(
            default_mass[:, 1],
            [0.4 + 0.25 * index for index in range(3)] * 2,
        )

        state = env.init_state()
        assert state.obs["obs"].shape == (6, 2)
        assert np.isfinite(state.obs["obs"]).all()
        state = env.step(np.zeros((6, 1), dtype=np.float32))
        assert np.isfinite(state.obs["obs"]).all()
        assert np.isfinite(state.reward).all()

        playback_mass = [float(env.get_playback_model(index).body_mass[1]) for index in range(3)]
        np.testing.assert_allclose(playback_mass, [0.4 + 0.25 * index for index in range(3)])

        env.reset()
        assert np.isfinite(env.state.obs["obs"]).all()
    finally:
        env.close()


def test_cpu_and_mjwarp_representative_rollouts_match_on_cuda(
    tmp_path: Path,
) -> None:
    pytest.importorskip(
        "unisim.backend.mujoco.backend",
        reason="SimToolReal cross-backend comparison requires the MuJoCo adapter",
    )
    pytest.importorskip(
        "unisim.backend.mjwarp.backend",
        reason="SimToolReal MJWarp comparison requires mujoco-warp",
    )
    warp = pytest.importorskip("warp", reason="SimToolReal MJWarp test requires Warp")
    warp.init()
    if not bool(warp.get_device().is_cuda):
        pytest.skip("SimToolReal MJWarp comparison requires CUDA")

    mjwarp_cfg = build_representative_simtool_real_env_cfg(
        write_representative_simtool_real_sources(tmp_path / "mjwarp")
    )
    cpu_cfg = build_representative_simtool_real_env_cfg(
        write_representative_simtool_real_sources(tmp_path / "mujoco")
    )
    mjwarp_env = make_manager_based_rl_env(mjwarp_cfg, num_envs=3, backend_type="mjwarp")
    cpu_env = make_manager_based_rl_env(cpu_cfg, num_envs=3, backend_type="mujoco")
    try:
        plan = mjwarp_cfg.scene.fixed_variant_plan
        assert plan is not None
        assert [Path(mjwarp_env.get_playback_model(index)).resolve() for index in range(3)] == [
            Path(variant.model_file).resolve() for variant in plan.variants
        ]

        actions = np.zeros((3, 1), dtype=np.float32)
        cpu_env.init_state()
        mjwarp_state = mjwarp_env.step(actions)
        cpu_state = cpu_env.step(actions)
        assert np.isfinite(mjwarp_state.obs["obs"]).all()
        assert np.isfinite(cpu_state.obs["obs"]).all()
        np.testing.assert_allclose(
            mjwarp_env.scene["tool"].data.joint_pos,
            cpu_env.scene["tool"].data.joint_pos,
            rtol=2e-4,
            atol=2e-6,
        )
        np.testing.assert_allclose(
            mjwarp_env.scene["tool"].data.joint_vel,
            cpu_env.scene["tool"].data.joint_vel,
            rtol=2e-4,
            atol=2e-5,
        )
    finally:
        mjwarp_env.close()
        cpu_env.close()


@pytest.mark.slow
def test_cpu_representative_fixed_tools_complete_one_ppo_iteration(
    tmp_path: Path,
) -> None:
    pytest.importorskip(
        "unisim.backend.mujoco.backend",
        reason="SimToolReal PPO smoke requires the MuJoCo adapter",
    )
    pytest.importorskip("rsl_rl", reason="SimToolReal PPO smoke requires rsl_rl")
    from rsl_rl.runners import OnPolicyRunner
    from uni_rl.algos.rsl_rl import normalize_ppo_train_cfg

    from unilab.structured_configs import PPOConfig

    sources = write_representative_simtool_real_sources(tmp_path)
    cfg = build_representative_simtool_real_env_cfg(sources)
    env = make_manager_based_rl_env(cfg, num_envs=12, backend_type="mujoco")
    wrapped = RslRlVecEnvWrapper(env)
    train_cfg = PPOConfig().to_dict()
    train_cfg.update(
        {
            "runner": {"logger": "none"},
            "num_steps_per_env": 2,
            "empirical_normalization": False,
            "policy": {
                "actor_hidden_dims": [16],
                "critic_hidden_dims": [16],
                "activation": "elu",
                "init_noise_std": 1.0,
            },
        }
    )
    train_cfg["algorithm"]["num_learning_epochs"] = 1
    train_cfg["algorithm"]["num_mini_batches"] = 1
    train_cfg = normalize_ppo_train_cfg(train_cfg)

    try:
        with TemporaryDirectory() as log_dir:
            runner = OnPolicyRunner(wrapped, train_cfg, log_dir=log_dir, device="cpu")
            runner.learn(num_learning_iterations=1, init_at_random_ep_len=True)
            parameters = [
                parameter.detach().cpu().numpy()
                for parameter in (
                    *runner.alg.actor.parameters(),
                    *runner.alg.critic.parameters(),
                )
            ]
            optimizer_steps = [
                int(state["step"].item())
                for state in runner.alg.optimizer.state.values()
                if "step" in state
            ]
    finally:
        env.close()
    assert parameters
    assert all(np.isfinite(parameter).all() for parameter in parameters)
    assert sum(optimizer_steps) > 0
