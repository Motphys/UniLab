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

from unilab.base.registry import ensure_registries
from unilab.envs.manipulation.sharpa_inhand.base import SharpaInhandBaseEnv
from unilab.envs.manipulation.sharpa_inhand.rotation import SharpaInhandRotationDRProvider

_CONF_DIR = Path(__file__).resolve().parents[2] / "conf"
_SRC_DIR = Path(__file__).resolve().parents[2] / "src"


def test_sharpa_env_uses_backend_contract_for_mujoco_metadata() -> None:
    """Sharpa env code should not read MuJoCo model internals directly."""
    source = "\n".join(
        (_SRC_DIR / "unilab" / "envs" / "manipulation" / "sharpa_inhand" / path).read_text(
            encoding="utf-8"
        )
        for path in ("base.py", "rotation.py")
    )

    assert "import mujoco" not in source
    assert "self._backend.model" not in source
    assert "_backend.model" not in source


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


def _build_fake_tactile_env(
    sensor_data: dict[str, np.ndarray],
    *,
    enable_tactile: bool = True,
    binary_contact: bool = False,
    disable_tactile_ids: list[int] | None = None,
    contact_smooth: float = 0.5,
    contact_threshold: float = 0.05,
    contact_latency: float = 0.0,
    contact_sensor_noise: float = 0.01,
    last_contacts: np.ndarray | None = None,
    prev_tactile_force: np.ndarray | None = None,
) -> Any:
    """Build a minimal fake Sharpa env for tactile-observation unit tests.

    Args:
        sensor_data: Mapping from tactile sensor name to backend sensor array.
        enable_tactile: Whether tactile observation is enabled.
        binary_contact: Whether binary-contact mode is enabled.
        disable_tactile_ids: Optional tactile ids to zero out.
        contact_smooth: Smoothing weight applied to the latest raw force.
        contact_threshold: Threshold used by binary-contact mode.
        contact_latency: Bernoulli probability of keeping the previous tactile output.
        contact_sensor_noise: Binary-contact sensor dropout probability.
        last_contacts: Optional previous tactile output buffer.
        prev_tactile_force: Optional previous raw tactile-force buffer.

    Returns:
        Simple fake env exposing the fields used by ``_compute_tactile_observation``.
    """
    tactile_names = [
        "contact_right_thumb_elastomer_force",
        "contact_right_index_elastomer_force",
        "contact_right_middle_elastomer_force",
        "contact_right_ring_elastomer_force",
        "contact_right_pinky_elastomer_force",
    ]
    num_envs = next(iter(sensor_data.values())).shape[0]
    env = SimpleNamespace(
        _num_envs=num_envs,
        _num_tactile=len(tactile_names),
        _np_dtype=np.float64,
        _backend=SimpleNamespace(get_sensor_data=lambda name: sensor_data[name]),
        _cfg=SimpleNamespace(
            obs=SimpleNamespace(
                enable_tactile=enable_tactile,
                binary_contact=binary_contact,
                contact_smooth=contact_smooth,
                contact_threshold=contact_threshold,
                tactile_force_clip_max=5.0,
            ),
            domain_rand=SimpleNamespace(
                contact_latency=contact_latency,
                contact_sensor_noise=contact_sensor_noise,
            ),
            disable_tactile_ids=list(disable_tactile_ids or []),
            sensor=SimpleNamespace(tactile_force_sensor_names=tactile_names),
        ),
        last_contacts=np.zeros((num_envs, len(tactile_names)), dtype=np.float64)
        if last_contacts is None
        else np.asarray(last_contacts, dtype=np.float64).copy(),
        _prev_tactile_force=np.zeros((num_envs, len(tactile_names)), dtype=np.float64)
        if prev_tactile_force is None
        else np.asarray(prev_tactile_force, dtype=np.float64).copy(),
    )
    env._extract_sensor_scalar = lambda sensor_name: SharpaInhandBaseEnv._extract_sensor_scalar(
        env, sensor_name
    )
    env._read_tactile_force = lambda: SharpaInhandBaseEnv._read_tactile_force(env)
    env._clear_tactile_history = lambda env_ids=None: SharpaInhandBaseEnv._clear_tactile_history(
        env, env_ids
    )
    return env


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


def test_sharpa_tactile_force_matches_reference_smoothing_and_order() -> None:
    """Verify tactile force uses reference sensor order and raw-force smoothing.

    Args:
        None.

    Returns:
        None. The assertions validate thumb→pinky ordering and 2-step smoothing.
    """
    sensor_data = {
        "contact_right_thumb_elastomer_force": np.array([[1.0, 0.0, 0.0]], dtype=np.float64),
        "contact_right_index_elastomer_force": np.array([[2.0, 0.0, 0.0]], dtype=np.float64),
        "contact_right_middle_elastomer_force": np.array([[3.0, 0.0, 0.0]], dtype=np.float64),
        "contact_right_ring_elastomer_force": np.array([[4.0, 0.0, 0.0]], dtype=np.float64),
        "contact_right_pinky_elastomer_force": np.array([[5.0, 0.0, 0.0]], dtype=np.float64),
    }
    env = _build_fake_tactile_env(
        sensor_data,
        contact_smooth=0.25,
        prev_tactile_force=np.array([[10.0, 20.0, 30.0, 40.0, 50.0]], dtype=np.float64),
    )

    tactile = SharpaInhandBaseEnv._compute_tactile_observation(env)

    expected = np.array([[7.75, 15.5, 23.25, 31.0, 38.75]], dtype=np.float64)
    np.testing.assert_allclose(tactile, expected)
    np.testing.assert_allclose(env.last_contacts, expected)
    np.testing.assert_allclose(
        env._prev_tactile_force,
        np.array([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=np.float64),
    )


def test_sharpa_tactile_binary_mode_matches_reference_latency_noise_and_disable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify binary tactile mode matches reference latency/noise semantics.

    Args:
        monkeypatch: Pytest helper used to make the random masks deterministic.

    Returns:
        None. The assertions validate binary thresholding, latency, and dropout.
    """
    sensor_data = {
        "contact_right_thumb_elastomer_force": np.array([[0.2, 0.0, 0.0]], dtype=np.float64),
        "contact_right_index_elastomer_force": np.array([[0.01, 0.0, 0.0]], dtype=np.float64),
        "contact_right_middle_elastomer_force": np.array([[0.5, 0.0, 0.0]], dtype=np.float64),
        "contact_right_ring_elastomer_force": np.array([[0.3, 0.0, 0.0]], dtype=np.float64),
        "contact_right_pinky_elastomer_force": np.array([[0.9, 0.0, 0.0]], dtype=np.float64),
    }
    env = _build_fake_tactile_env(
        sensor_data,
        binary_contact=True,
        disable_tactile_ids=[4],
        contact_smooth=1.0,
        contact_threshold=0.05,
        contact_latency=0.5,
        contact_sensor_noise=0.01,
        last_contacts=np.array([[0.4, 0.7, 0.0, 1.0, 0.2]], dtype=np.float64),
    )

    sampled_masks = iter(
        [
            np.array([[0.9, 0.1, 0.9, 0.9, 0.1]], dtype=np.float64),
            np.array([[0.2, 0.2, 0.005, 0.9, 0.2]], dtype=np.float64),
        ]
    )

    monkeypatch.setattr(np.random, "rand", lambda *shape: next(sampled_masks).copy())

    tactile = SharpaInhandBaseEnv._compute_tactile_observation(env)

    expected_last = np.array([[1.0, 0.7, 1.0, 1.0, 0.2]], dtype=np.float64)
    expected_tactile = np.array([[1.0, 0.7, 0.0, 1.0, 0.2]], dtype=np.float64)
    np.testing.assert_allclose(env.last_contacts, expected_last)
    np.testing.assert_allclose(tactile, expected_tactile)
    np.testing.assert_allclose(
        env._prev_tactile_force,
        np.array([[0.2, 0.01, 0.5, 0.3, 0.9]], dtype=np.float64),
    )


def test_sharpa_tactile_disabled_clears_history() -> None:
    """Verify disabled tactile observation clears cached tactile history.

    Args:
        None.

    Returns:
        None. The assertions validate the disabled-tactile contract.
    """
    sensor_data = {
        "contact_right_thumb_elastomer_force": np.array([[1.0, 0.0, 0.0]], dtype=np.float64),
        "contact_right_index_elastomer_force": np.array([[2.0, 0.0, 0.0]], dtype=np.float64),
        "contact_right_middle_elastomer_force": np.array([[3.0, 0.0, 0.0]], dtype=np.float64),
        "contact_right_ring_elastomer_force": np.array([[4.0, 0.0, 0.0]], dtype=np.float64),
        "contact_right_pinky_elastomer_force": np.array([[5.0, 0.0, 0.0]], dtype=np.float64),
    }
    env = _build_fake_tactile_env(
        sensor_data,
        enable_tactile=False,
        last_contacts=np.full((1, 5), 3.0, dtype=np.float64),
        prev_tactile_force=np.full((1, 5), 4.0, dtype=np.float64),
    )

    tactile = SharpaInhandBaseEnv._compute_tactile_observation(env)

    np.testing.assert_allclose(tactile, np.zeros((1, 5), dtype=np.float64))
    np.testing.assert_allclose(env.last_contacts, np.zeros((1, 5), dtype=np.float64))
    np.testing.assert_allclose(env._prev_tactile_force, np.zeros((1, 5), dtype=np.float64))


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
        for scale_value in env_cfg_override["domain_rand"]["scale_list"]:
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
            assert np.all(friction_scale >= cfg.env.domain_rand.randomize_friction_scale_lower)
            assert np.all(friction_scale <= cfg.env.domain_rand.randomize_friction_scale_upper)

            for env_idx in range(num_envs):
                scale = friction_scale[env_idx]
                for material, base_friction in (
                    ("object", cfg.env.domain_rand.object_base_friction),
                    ("metal", cfg.env.domain_rand.metal_base_friction),
                    ("elastomer", cfg.env.domain_rand.elastomer_base_friction),
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
        for scale_value in env_cfg_override["domain_rand"]["scale_list"]:
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
            assert np.all(kp_scale >= cfg.env.domain_rand.randomize_p_gain_scale_lower)
            assert np.all(kp_scale <= cfg.env.domain_rand.randomize_p_gain_scale_upper)
            assert np.all(kd_scale >= cfg.env.domain_rand.randomize_d_gain_scale_lower)
            assert np.all(kd_scale <= cfg.env.domain_rand.randomize_d_gain_scale_upper)
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
    env_cfg_override["domain_rand"]["force_scale"] = 2.0
    env_cfg_override["domain_rand"]["random_force_prob_scalar"] = 1.0
    with TemporaryDirectory() as tmp_dir:
        cache_prefix = Path(tmp_dir) / "sharpa_grasp"
        env_cfg_override["grasp_cache_path"] = str(cache_prefix)
        for scale_value in env_cfg_override["domain_rand"]["scale_list"]:
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
            assert env_obj._backend.get_dr_capabilities().supports_interval_body_force
            before = env_obj._backend._physics_state.copy()
            env_obj._dr_manager.apply_interval_randomization_if_due(env_obj.step_counter)

            body_id = int(env_obj._object_body_id)
            force_slice = slice(6 * body_id, 6 * body_id + 3)
            joint_adr = int(env_obj._backend.model.body_jntadr[body_id])
            dof_adr = int(env_obj._backend.model.jnt_dofadr[joint_adr])
            qvel_slice = slice(
                env_obj._backend._idx_qvel + dof_adr,
                env_obj._backend._idx_qvel + dof_adr + 3,
            )

            assert np.any(np.linalg.norm(env_obj._random_object_force, axis=1) > 0.0)
            np.testing.assert_allclose(
                env_obj._backend._pending_xfrc_applied[:, force_slice],
                env_obj._random_object_force,
            )
            env_obj._backend.step(
                np.zeros((num_envs, env_obj._num_action), dtype=env_obj._np_dtype),
                nsteps=1,
            )
            after = env_obj._backend._physics_state
            velocity_delta = after[:, qvel_slice] - before[:, qvel_slice]
            assert np.any(np.linalg.norm(velocity_delta, axis=1) > 0.0)
        finally:
            pool = getattr(getattr(env_obj, "_backend", None), "_pool", None)
            if pool is not None:
                pool.close()
            env_obj.close()


@pytest.mark.slow
def test_sharpa_mujoco_interval_force_plan_matches_decay_and_mass_scaled_resample(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify Sharpa force DR matches decay plus mass-scaled Gaussian resampling.

    Args:
        monkeypatch: Pytest helper used to make numpy random sampling deterministic.

    Returns:
        None. The assertions validate the exact interval body-force payload.
    """
    _require_mujoco_runtime()
    ensure_registries()

    from unilab.base import registry

    num_envs = 4
    _, env_cfg_override = _compose_sharpa_mujoco_owner_cfg(num_envs)
    env_cfg_override["domain_rand"]["force_scale"] = 2.0
    env_cfg_override["domain_rand"]["random_force_prob_scalar"] = 0.5
    env_cfg_override["domain_rand"]["randomize_mass"] = False
    with TemporaryDirectory() as tmp_dir:
        cache_prefix = Path(tmp_dir) / "sharpa_grasp"
        env_cfg_override["grasp_cache_path"] = str(cache_prefix)
        for scale_value in env_cfg_override["domain_rand"]["scale_list"]:
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
            env_obj._random_object_force[:] = np.array(
                [
                    [0.4, -0.2, 0.1],
                    [-0.3, 0.5, -0.7],
                    [0.8, -0.6, 0.2],
                    [0.1, 0.3, -0.4],
                ],
                dtype=np.float64,
            )
            previous_force = env_obj._random_object_force.copy()
            sampled_uniform = np.array([0.1, 0.9, 0.2, 0.8], dtype=np.float64)
            sampled_gaussian = np.array(
                [
                    [1.0, -2.0, 0.5],
                    [-1.5, 0.25, 2.0],
                ],
                dtype=np.float64,
            )

            def _fake_rand(*shape: int) -> np.ndarray:
                assert shape == (num_envs,)
                return sampled_uniform.copy()

            def _fake_randn(*shape: int) -> np.ndarray:
                assert shape == (2, 3)
                return sampled_gaussian.copy()

            monkeypatch.setattr(np.random, "rand", _fake_rand)
            monkeypatch.setattr(np.random, "randn", _fake_randn)

            decay = float(
                np.power(
                    env_obj.cfg.domain_rand.force_decay,
                    env_obj.cfg.ctrl_dt / max(env_obj.cfg.domain_rand.force_decay_interval, 1.0e-8),
                )
            )
            object_mass = env_obj._resolve_current_object_mass()
            resample_mask = sampled_uniform < float(
                env_obj.cfg.domain_rand.random_force_prob_scalar
            )

            plan = SharpaInhandRotationDRProvider().build_interval_randomization_plan(
                env_obj,
                step_counter=0,
            )

            assert plan is not None
            assert plan.body_ids is not None
            assert plan.body_force is not None
            np.testing.assert_array_equal(
                plan.body_ids,
                np.asarray([env_obj._object_body_id], dtype=np.int32),
            )

            # Envs below the Bernoulli threshold get a fresh Gaussian force sample;
            # the remaining envs keep the decayed previous force.
            expected_force = previous_force * decay
            expected_force[resample_mask] = (
                sampled_gaussian
                * object_mass[resample_mask, None]
                * float(env_obj.cfg.domain_rand.force_scale)
            )
            np.testing.assert_allclose(env_obj._random_object_force, expected_force)
            np.testing.assert_allclose(plan.body_force, expected_force[:, None, :])
            assert plan.body_linear_velocity_delta is None
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
        for scale_value in env_cfg_override["domain_rand"]["scale_list"]:
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
            assert np.all(randomized_mass >= cfg.env.domain_rand.randomize_mass_lower)
            assert np.all(randomized_mass <= cfg.env.domain_rand.randomize_mass_upper)
            assert np.all(randomized_com >= cfg.env.domain_rand.randomize_com_lower)
            assert np.all(randomized_com <= cfg.env.domain_rand.randomize_com_upper)

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
