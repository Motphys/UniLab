"""Config system verification tests.

These tests enforce that:
1. Base Hydra configs compose without legacy config groups.
2. Every supported runtime variant resolves through exactly one task owner file.
3. Final reward/env/algo sections are present on the composed config, not mounted by Python glue.
4. Backend-specific hyperparameters preserve the intended pre-refactor behavior.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

# CPU-bound on the single-core CI runner; kept in the slow lane (make test-slow).
pytestmark = pytest.mark.slow

CONF_DIR = Path(__file__).parent.parent.parent / "src" / "unilab" / "conf"
_BACKENDS = ("mujoco", "mjwarp")


def _expected_backend_from_variant(name: str) -> str | None:
    for backend in _BACKENDS:
        if name == backend or name.startswith(f"{backend}_"):
            return backend
    return None


def _compose(algo_dir: str, config_name: str = "config", overrides: list[str] | None = None):
    normalized_overrides = _normalize_overrides(algo_dir, overrides)

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_DIR / algo_dir), version_base="1.3"):
        return compose(config_name, overrides=normalized_overrides)


def _normalize_overrides(algo_dir: str, overrides: list[str] | None) -> list[str]:
    normalized: list[str] = []
    task_selected = False

    for override in overrides or []:
        if override.startswith("task="):
            task_selected = True
            normalized.append(override)
            continue
        normalized.append(override)

    if not task_selected:
        if algo_dir in ("sac", "flashsac"):
            normalized.append("task=g1_walk_flat/mujoco")
        else:
            normalized.append("task=go2_joystick_flat/mujoco")

    return normalized


def _assert_reward_populated(cfg, label: str):
    assert hasattr(cfg, "reward"), f"{label} missing cfg.reward"
    reward_dict = OmegaConf.to_container(cfg.reward, resolve=True)
    assert isinstance(reward_dict, dict), f"{label} reward must resolve to mapping"
    if "scales" in reward_dict:
        assert len(reward_dict["scales"]) > 0, f"{label} reward.scales must be non-empty"
        return

    active_terms = {name: term for name, term in reward_dict.items() if term is not None}
    assert active_terms, f"{label} Manager-Based reward terms must be non-empty"
    for term_name, term in active_terms.items():
        assert isinstance(term, dict), f"{label} reward.{term_name} must be a mapping"
        assert "func" in term, f"{label} reward.{term_name} must declare func"
        assert "weight" in term, f"{label} reward.{term_name} must declare weight"


def _supported_task_cases() -> list[tuple[str, str, str, str, str, list[str]]]:
    cases: list[tuple[str, str, str, str, str, list[str]]] = []

    for algo_dir in ["ppo", "appo"]:
        root = CONF_DIR / algo_dir / "task"
        for task_dir in sorted(path for path in root.iterdir() if path.is_dir()):
            for backend_file in sorted(task_dir.glob("*.yaml")):
                expected_backend = _expected_backend_from_variant(backend_file.stem)
                if expected_backend is None:
                    continue
                cases.append(
                    (
                        algo_dir,
                        "config",
                        task_dir.name,
                        expected_backend,
                        str(backend_file.relative_to(CONF_DIR)),
                        [f"task={task_dir.name}/{backend_file.stem}"],
                    )
                )

    for algo_dir in ["sac", "flashsac"]:
        root = CONF_DIR / algo_dir / "task"
        for task_dir in sorted(path for path in root.iterdir() if path.is_dir()):
            for backend_file in sorted(task_dir.glob("*.yaml")):
                expected_backend = _expected_backend_from_variant(backend_file.stem)
                if expected_backend is None:
                    continue
                cases.append(
                    (
                        algo_dir,
                        "config",
                        task_dir.name,
                        expected_backend,
                        str(backend_file.relative_to(CONF_DIR)),
                        [f"task={task_dir.name}/{backend_file.stem}"],
                    )
                )

    return cases


@pytest.mark.parametrize(
    "algo_dir,config_name",
    [
        ("sac", "config"),
        ("flashsac", "config"),
        ("appo", "config"),
        ("ppo", "config"),
    ],
)
def test_algo_config_composes(algo_dir: str, config_name: str):
    cfg = _compose(algo_dir, config_name)
    assert cfg.training.task_name
    assert cfg.training.sim_backend == "mujoco"
    assert int(cfg.training.log_interval) >= 1


def test_backend_task_files_keep_full_identity():
    for path in sorted(CONF_DIR.glob("*/task/**/*.yaml")):
        cfg = OmegaConf.load(path)
        cfg_dict_raw = OmegaConf.to_container(cfg, resolve=True) or {}
        assert isinstance(cfg_dict_raw, dict)
        if path.stem not in _BACKENDS:
            continue
        training_raw = cfg_dict_raw.get("training", {})
        assert isinstance(training_raw, dict)
        assert "task_name" in training_raw, f"task missing task_name: {path}"
        assert "sim_backend" in training_raw, f"task missing sim_backend: {path}"


@pytest.mark.parametrize(
    "algo_dir,config_name,task,backend,task_file,overrides",
    _supported_task_cases(),
)
def test_supported_task_composes(
    algo_dir: str,
    config_name: str,
    task: str,
    backend: str,
    task_file: str,
    overrides: list[str],
):
    cfg = _compose(algo_dir, config_name, overrides=overrides)

    assert cfg.training.task_name, f"{task_file} should resolve task_name"
    assert cfg.training.sim_backend == backend, f"{task_file} should set backend"
    _assert_reward_populated(cfg, task_file)


def test_offpolicy_g1_walk_flat_mjwarp_owner_preserves_sac_contract():
    mujoco_cfg = _compose("sac", overrides=["task=g1_walk_flat/mujoco"])
    mjwarp_cfg = _compose("sac", overrides=["task=g1_walk_flat/mjwarp"])

    assert mjwarp_cfg.training.sim_backend == "mjwarp"
    assert mjwarp_cfg.training.no_play is False
    assert mjwarp_cfg.training.play_render_mode == "record"
    assert mjwarp_cfg.algo.num_envs == mujoco_cfg.algo.num_envs
    assert mjwarp_cfg.env.actions.joint_pos.scale == pytest.approx(
        mujoco_cfg.env.actions.joint_pos.scale
    )
    assert mjwarp_cfg.env.mjwarp_nconmax == 128
    assert mjwarp_cfg.env.mjwarp_njmax == 256
    assert mjwarp_cfg.env.render_spacing == pytest.approx(2.0)
    assert mjwarp_cfg.env.events.pd_gains is None
    assert OmegaConf.to_container(mjwarp_cfg.reward, resolve=True) == OmegaConf.to_container(
        mujoco_cfg.reward, resolve=True
    )


def test_ppo_g1_mjwarp_inherits_enabled_playback_default():
    cfg = _compose("ppo", overrides=["task=g1_walk_flat/mjwarp"])

    assert cfg.training.no_play is False
    assert cfg.training.play_render_mode == "record"


def test_offpolicy_flashsac_g1_walk_flat_tensor_owners_carry_walk_reset_owner():
    """Motrix/mjwarp leaves own the tensor walk command reset like MuJoCo."""
    mujoco_cfg = _compose("flashsac", overrides=["task=g1_walk_flat/mujoco"])
    expected = mujoco_cfg.env.reset_owners.walk.func

    for backend in ("motrix", "mjwarp"):
        cfg = _compose("flashsac", overrides=[f"task=g1_walk_flat/{backend}"])
        assert cfg.env.reset_owners.walk.func == expected
        assert cfg.env.reset_owners.walk.command_name == "twist"


def test_appo_adaptive_lr_factors_are_overridden_only_by_dex_hand_owners():
    cfg = _compose("appo", overrides=["task=g1_walk_flat/mujoco"])

    assert cfg.algo.algorithm.adaptive_kl_factor == pytest.approx(1.2)
    assert cfg.algo.algorithm.adaptive_lr_factor == pytest.approx(1.1)


def test_offpolicy_flashsac_go2_joystick_mujoco_enables_full_dr_stack():
    mujoco_cfg = _compose(
        "flashsac",
        overrides=["task=go2_joystick_flat/mujoco"],
    )

    assert mujoco_cfg.training.task_name == "Go2JoystickFlat"
    assert mujoco_cfg.training.sim_backend == "mujoco"

    assert mujoco_cfg.env.events.pd_gains.func == "unilab.envs.mdp.pd_gains"
    assert (
        mujoco_cfg.env.events.randomize_rigid_body_mass.func
        == "unilab.envs.mdp.randomize_rigid_body_mass"
    )
    assert (
        mujoco_cfg.env.events.randomize_rigid_body_com.func
        == "unilab.envs.mdp.randomize_rigid_body_com"
    )
    assert (
        mujoco_cfg.env.events.randomize_physics_scene_gravity.func
        == "unilab.envs.mdp.randomize_physics_scene_gravity"
    )
    assert (
        mujoco_cfg.env.events.push_by_setting_velocity.func
        == "unilab.envs.mdp.push_by_setting_velocity"
    )
    assert mujoco_cfg.env.events.push_by_setting_velocity.mode == "interval"
    assert mujoco_cfg.env.events.push_by_setting_velocity.is_global_time is True
    assert mujoco_cfg.env.observations.policy.enable_corruption is True
    assert mujoco_cfg.env.observations.policy.terms.joint_pos.noise.n_min == pytest.approx(-0.01)
    assert mujoco_cfg.env.observations.policy.terms.joint_vel.noise.n_min == pytest.approx(-0.1)


def test_model_field_dr_ranges_use_array_bounds_in_both_lifecycles():
    """Keep startup and reset model-field DR declarations visually identical."""
    yaml_path = CONF_DIR / "flashsac" / "task" / "go2_joystick_flat" / "mujoco.yaml"
    yaml_text = yaml_path.read_text(encoding="utf-8")

    assert "mass_distribution_params: [-1.5, 1.5]" in yaml_text
    assert "mass_distribution_params:\n" not in yaml_text
    assert "          x: [-0.05, 0.05]\n" in yaml_text
    assert "          x:\n" not in yaml_text


def test_cli_override_beats_task_defaults():
    cfg = _compose(
        "ppo",
        overrides=["task=g1_walk_flat/mujoco", "algo.max_iterations=1"],
    )

    assert cfg.algo.max_iterations == 1


def test_task_backend_leaves_inherit_only_same_directory_base():
    """ADR-0023: every sim owner leaf composes from its same-dir base.yaml only.

    Backend leaves must never inherit another backend owner; the leaf file plus
    base.yaml must fully determine the composed contract.
    """
    task_roots = sorted(CONF_DIR.glob("*/task/*"))
    for task_dir in task_roots:
        if not task_dir.is_dir():
            continue
        yaml_files = sorted(task_dir.glob("*.yaml"))
        leaves = [path for path in yaml_files if path.stem != "base"]
        if not leaves:
            continue
        base_file = task_dir / "base.yaml"
        assert base_file.is_file(), f"{task_dir} has backend leaves but no base.yaml"
        for leaf in leaves:
            raw = OmegaConf.load(leaf)
            defaults = raw.get("defaults")
            assert defaults is not None, f"{leaf} must declare defaults"
            assert list(defaults) == [
                f"/task/{task_dir.name}/base",
                "_self_",
            ], f"{leaf} must inherit only its same-directory base (ADR-0023)"
