from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from unilab.envs.manipulation.sharpa_inhand.rotation import SharpaInhandRotationDRProvider
from unilab.utils.algo_utils import ensure_registries

_CONF_DIR = Path(__file__).resolve().parents[2] / "conf"


def _require_mujoco_runtime() -> None:
    """Require the MuJoCo batch runtime used by Sharpa reset randomization tests.

    Args:
        None.

    Returns:
        None. The helper skips the test when MuJoCo batch runtime is unavailable.
    """
    pytest.importorskip("mujoco", reason="mujoco not installed")
    try:
        from mujoco.batch_env import BatchEnvPool as _  # noqa: F401
    except Exception:
        pytest.skip(
            "mujoco.batch_env not available (platform/libstdc++ issue)",
            allow_module_level=False,
        )


def _compose_sharpa_mujoco_owner_cfg(num_envs: int) -> tuple[Any, dict[str, Any]]:
    """Compose the Sharpa MuJoCo owner config used by the real training path.

    Args:
        num_envs: Number of vectorized environments for the test env.

    Returns:
        Tuple of the composed Hydra config and the env_cfg_override dict for registry.make().
    """
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(_CONF_DIR / "ppo"), version_base="1.3"):
        cfg = compose(
            "config",
            overrides=[
                "task=sharpa_inhand/mujoco",
                f"algo.num_envs={num_envs}",
            ],
        )

    env_cfg_override = OmegaConf.to_container(cfg.env, resolve=True)
    assert isinstance(env_cfg_override, dict)
    env_cfg_override["reward_config"] = OmegaConf.to_container(cfg.reward, resolve=True)
    return cfg, env_cfg_override


def test_sharpa_provider_builds_mujoco_init_geom_scale_plan() -> None:
    env = SimpleNamespace(
        _backend=SimpleNamespace(backend_type="mujoco"),
        _object_geom_base_size=np.array([0.02, 0.016, 0.0], dtype=np.float64),
        scale_values=np.array([0.5, 0.8], dtype=np.float64),
        scale_ids=np.array([0, 0, 1, 1], dtype=np.int32),
        cfg=SimpleNamespace(object_geom_name="object"),
    )

    plan = SharpaInhandRotationDRProvider().build_init_randomization_plan(env)

    assert plan is not None
    np.testing.assert_array_equal(
        plan.model_assignments,
        np.array([0, 0, 1, 1], dtype=np.int32),
    )
    assert len(plan.model_variants) == 2
    assert plan.model_variants[0].geom_size_overrides[0].geom_name == "object"
    np.testing.assert_allclose(
        plan.model_variants[0].geom_size_overrides[0].size,
        [0.01, 0.008, 0.0],
    )
    np.testing.assert_allclose(
        plan.model_variants[1].geom_size_overrides[0].size,
        [0.016, 0.0128, 0.0],
    )


def test_sharpa_provider_skips_non_mujoco_init_geom_scale_plan() -> None:
    env = SimpleNamespace(
        _backend=SimpleNamespace(backend_type="motrix"),
        _object_geom_base_size=np.array([0.02, 0.016, 0.0], dtype=np.float64),
        scale_values=np.array([0.5], dtype=np.float64),
        scale_ids=np.array([0], dtype=np.int32),
        cfg=SimpleNamespace(object_geom_name="object"),
    )

    plan = SharpaInhandRotationDRProvider().build_init_randomization_plan(env)

    assert plan is None


@pytest.mark.slow
def test_sharpa_mujoco_reset_applies_friction_randomization() -> None:
    _require_mujoco_runtime()
    ensure_registries()

    from unilab.base import registry

    num_envs = 4
    cfg, env_cfg_override = _compose_sharpa_mujoco_owner_cfg(num_envs)
    with TemporaryDirectory() as tmp_dir:
        cache_prefix = Path(tmp_dir) / "sharpa_grasp"
        env_cfg_override["grasp_cache_path"] = str(cache_prefix)
        for scale_value in env_cfg_override["scale_list"]:
            cache_file = cache_prefix.parent / f"{cache_prefix.name}_{float(scale_value):g}.npy"
            np.save(cache_file, np.zeros((8, 29), dtype=np.float32))

        env = registry.make(
            "SharpaInhandRotation",
            num_envs=num_envs,
            sim_backend="mujoco",
            env_cfg_override=env_cfg_override,
        )
        env_obj: Any = env
        try:
            env_ids = np.arange(num_envs, dtype=np.int32)
            _, info = env_obj.reset(env_ids)

            backend: Any = env_obj._backend
            pool = backend._pool
            geom_friction = np.stack(
                [pool.get_field(i, "geom_friction") for i in range(num_envs)],
                axis=0,
            ).reshape(num_envs, backend.model.ngeom, 3)
            friction_scale = np.asarray(info["critic_info"][:, 3], dtype=np.float64)

            assert np.unique(np.round(friction_scale, 6)).size > 1
            assert np.all(friction_scale >= cfg.env.randomize_friction_scale_lower)
            assert np.all(friction_scale <= cfg.env.randomize_friction_scale_upper)

            for env_idx in range(num_envs):
                scale = friction_scale[env_idx]
                for material, base_friction in (
                    ("object", cfg.env.object_base_friction),
                    ("metal", cfg.env.metal_base_friction),
                    ("elastomer", cfg.env.elastomer_base_friction),
                ):
                    actual = geom_friction[env_idx, env_obj._friction_geom_ids[material]]
                    expected = env_obj._friction_profile(material, base_friction) * scale
                    np.testing.assert_allclose(actual, np.broadcast_to(expected, actual.shape))
        finally:
            pool = getattr(getattr(env_obj, "_backend", None), "_pool", None)
            if pool is not None:
                pool.close()
            env_obj.close()


@pytest.mark.slow
def test_sharpa_mujoco_reset_randomizes_pd_gains_from_xml_defaults() -> None:
    """Verify Sharpa reset PD gains scale MuJoCo XML actuator defaults per DOF.

    Args:
        None.

    Returns:
        None. The assertions validate info buffers and backend reset payloads.
    """
    _require_mujoco_runtime()
    ensure_registries()

    from unilab.base import registry

    num_envs = 4
    cfg, env_cfg_override = _compose_sharpa_mujoco_owner_cfg(num_envs)
    with TemporaryDirectory() as tmp_dir:
        cache_prefix = Path(tmp_dir) / "sharpa_grasp"
        env_cfg_override["grasp_cache_path"] = str(cache_prefix)
        for scale_value in env_cfg_override["scale_list"]:
            cache_file = cache_prefix.parent / f"{cache_prefix.name}_{float(scale_value):g}.npy"
            np.save(cache_file, np.zeros((8, 29), dtype=np.float32))

        env = registry.make(
            "SharpaInhandRotation",
            num_envs=num_envs,
            sim_backend="mujoco",
            env_cfg_override=env_cfg_override,
        )
        env_obj: Any = env
        try:
            env_ids = np.arange(num_envs, dtype=np.int32)
            _, info = env_obj.reset(env_ids)

            backend: Any = env_obj._backend
            pool = backend._pool
            default_kp, default_kd = backend.get_actuator_gains()
            default_kp = np.asarray(default_kp[: env_obj._num_action], dtype=np.float64)
            default_kd = np.asarray(default_kd[: env_obj._num_action], dtype=np.float64)
            info_kp = np.asarray(info["p_gain"], dtype=np.float64)
            info_kd = np.asarray(info["d_gain"], dtype=np.float64)
            pool_kp = np.stack([pool.get_field(i, "kp") for i in range(num_envs)], axis=0)
            pool_kd = np.stack([pool.get_field(i, "kd") for i in range(num_envs)], axis=0)

            assert np.all(default_kp > 0.0)
            assert np.all(default_kd > 0.0)
            assert info_kp.shape == (num_envs, env_obj._num_action)
            assert info_kd.shape == (num_envs, env_obj._num_action)
            np.testing.assert_allclose(pool_kp[:, : env_obj._num_action], info_kp)
            np.testing.assert_allclose(pool_kd[:, : env_obj._num_action], info_kd)

            kp_scale = info_kp / default_kp[None, :]
            kd_scale = info_kd / default_kd[None, :]
            assert np.unique(np.round(kp_scale.reshape(-1), 6)).size > 1
            assert np.unique(np.round(kd_scale.reshape(-1), 6)).size > 1
            assert np.all(kp_scale >= cfg.env.randomize_p_gain_scale_lower)
            assert np.all(kp_scale <= cfg.env.randomize_p_gain_scale_upper)
            assert np.all(kd_scale >= cfg.env.randomize_d_gain_scale_lower)
            assert np.all(kd_scale <= cfg.env.randomize_d_gain_scale_upper)
        finally:
            pool = getattr(getattr(env_obj, "_backend", None), "_pool", None)
            if pool is not None:
                pool.close()
            env_obj.close()


@pytest.mark.slow
def test_sharpa_mujoco_interval_force_disturbs_object_velocity() -> None:
    """Verify Sharpa force randomization perturbs object linear velocity through DR.

    Args:
        None.

    Returns:
        None. The assertions validate the interval-randomization force path.
    """
    _require_mujoco_runtime()
    ensure_registries()

    from unilab.base import registry

    num_envs = 4
    _, env_cfg_override = _compose_sharpa_mujoco_owner_cfg(num_envs)
    env_cfg_override["force_scale"] = 2.0
    env_cfg_override["random_force_prob_scalar"] = 1.0
    with TemporaryDirectory() as tmp_dir:
        cache_prefix = Path(tmp_dir) / "sharpa_grasp"
        env_cfg_override["grasp_cache_path"] = str(cache_prefix)
        for scale_value in env_cfg_override["scale_list"]:
            cache_file = cache_prefix.parent / f"{cache_prefix.name}_{float(scale_value):g}.npy"
            np.save(cache_file, np.zeros((8, 29), dtype=np.float32))

        env = registry.make(
            "SharpaInhandRotation",
            num_envs=num_envs,
            sim_backend="mujoco",
            env_cfg_override=env_cfg_override,
        )
        env_obj: Any = env
        try:
            env.init_state()
            assert env_obj._backend.get_dr_capabilities().supports_interval_body_velocity_delta
            before = env_obj._backend._physics_state.copy()
            env_obj._dr_manager.apply_interval_randomization_if_due(env_obj.step_counter)
            after = env_obj._backend._physics_state

            body_id = int(env_obj._object_body_id)
            joint_adr = int(env_obj._backend.model.body_jntadr[body_id])
            dof_adr = int(env_obj._backend.model.jnt_dofadr[joint_adr])
            qvel_slice = slice(
                env_obj._backend._idx_qvel + dof_adr,
                env_obj._backend._idx_qvel + dof_adr + 3,
            )
            velocity_delta = after[:, qvel_slice] - before[:, qvel_slice]

            assert np.any(np.linalg.norm(env_obj._random_object_force, axis=1) > 0.0)
            assert np.any(np.linalg.norm(velocity_delta, axis=1) > 0.0)
            np.testing.assert_allclose(
                velocity_delta,
                env_obj._random_object_force
                / env_obj._resolve_current_object_mass()[:, None]
                * env_obj.cfg.ctrl_dt,
            )
        finally:
            pool = getattr(getattr(env_obj, "_backend", None), "_pool", None)
            if pool is not None:
                pool.close()
            env_obj.close()


@pytest.mark.slow
def test_sharpa_mujoco_reset_applies_object_mass_and_com_randomization() -> None:
    _require_mujoco_runtime()
    ensure_registries()

    from unilab.base import registry

    num_envs = 4
    cfg, env_cfg_override = _compose_sharpa_mujoco_owner_cfg(num_envs)
    with TemporaryDirectory() as tmp_dir:
        cache_prefix = Path(tmp_dir) / "sharpa_grasp"
        env_cfg_override["grasp_cache_path"] = str(cache_prefix)
        for scale_value in env_cfg_override["scale_list"]:
            cache_file = cache_prefix.parent / f"{cache_prefix.name}_{float(scale_value):g}.npy"
            np.save(cache_file, np.zeros((8, 29), dtype=np.float32))

        env = registry.make(
            "SharpaInhandRotation",
            num_envs=num_envs,
            sim_backend="mujoco",
            env_cfg_override=env_cfg_override,
        )
        env_obj: Any = env
        try:
            env_ids = np.arange(num_envs, dtype=np.int32)
            _, info = env_obj.reset(env_ids)

            backend: Any = env_obj._backend
            pool = backend._pool
            body_mass = np.stack([pool.get_field(i, "body_mass") for i in range(num_envs)], axis=0)
            body_ipos = np.stack([pool.get_field(i, "body_ipos") for i in range(num_envs)], axis=0)
            body_ipos = body_ipos.reshape(num_envs, backend.model.nbody, 3)

            object_body_id = int(env_obj._object_body_id)
            randomized_mass = np.asarray(info["critic_info"][:, 4], dtype=np.float64)
            randomized_com = np.asarray(info["critic_info"][:, 5:8], dtype=np.float64)

            assert np.unique(np.round(randomized_mass, 6)).size > 1
            assert np.unique(np.round(randomized_com.reshape(-1), 6)).size > 1
            assert np.all(randomized_mass >= cfg.env.randomize_mass_lower)
            assert np.all(randomized_mass <= cfg.env.randomize_mass_upper)
            assert np.all(randomized_com >= cfg.env.randomize_com_lower)
            assert np.all(randomized_com <= cfg.env.randomize_com_upper)

            np.testing.assert_allclose(body_mass[:, object_body_id], randomized_mass)
            np.testing.assert_allclose(
                body_ipos[:, object_body_id, :],
                env_obj._base_body_ipos[object_body_id][None, :] + randomized_com,
            )
        finally:
            pool = getattr(getattr(env_obj, "_backend", None), "_pool", None)
            if pool is not None:
                pool.close()
            env_obj.close()
