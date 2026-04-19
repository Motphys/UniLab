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

CONF_DIR = Path(__file__).parent.parent.parent / "conf"
_PPO_MLX_TASKS = {"go1_joystick_flat", "go2_joystick_flat", "g1_joystick_flat"}


def _compose(algo_dir: str, config_name: str = "config", overrides: list[str] | None = None):
    normalized_overrides = _normalize_overrides(algo_dir, overrides)

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_DIR / algo_dir), version_base="1.3"):
        return compose(config_name, overrides=normalized_overrides)


def _normalize_overrides(algo_dir: str, overrides: list[str] | None) -> list[str]:
    algo = "sac"
    normalized: list[str] = []
    task_selected = False

    for override in overrides or []:
        if override.startswith("algo="):
            algo = override.split("=", 1)[1]
            normalized.append(override)
            continue
        if override.startswith("task="):
            task_selected = True
            normalized.append(override)
            continue
        normalized.append(override)

    if not task_selected:
        if algo_dir == "offpolicy":
            normalized.append(f"task={algo}/g1_walk_flat/mujoco")
        else:
            normalized.append("task=go1_joystick_flat/mujoco")

    return normalized


def _assert_reward_populated(cfg, label: str):
    assert hasattr(cfg, "reward"), f"{label} missing cfg.reward"
    reward_dict = OmegaConf.to_container(cfg.reward, resolve=True)
    assert isinstance(reward_dict, dict), f"{label} reward must resolve to mapping"
    assert "scales" in reward_dict, f"{label} reward must contain scales"
    assert len(reward_dict["scales"]) > 0, f"{label} reward.scales must be non-empty"


def _supported_task_cases() -> list[tuple[str, str, str, str, str, list[str]]]:
    cases: list[tuple[str, str, str, str, str, list[str]]] = []

    for algo_dir in ["ppo", "appo"]:
        root = CONF_DIR / algo_dir / "task"
        for task_dir in sorted(path for path in root.iterdir() if path.is_dir()):
            for backend_file in sorted(task_dir.glob("*.yaml")):
                cases.append(
                    (
                        algo_dir,
                        "config",
                        task_dir.name,
                        backend_file.stem,
                        str(backend_file.relative_to(CONF_DIR)),
                        [f"task={task_dir.name}/{backend_file.stem}"],
                    )
                )
                if algo_dir == "ppo" and task_dir.name in _PPO_MLX_TASKS:
                    cases.append(
                        (
                            algo_dir,
                            "config_mlx",
                            task_dir.name,
                            backend_file.stem,
                            str(backend_file.relative_to(CONF_DIR)),
                            [f"task={task_dir.name}/{backend_file.stem}"],
                        )
                    )

    offpolicy_root = CONF_DIR / "offpolicy" / "task"
    for algo_root in sorted(path for path in offpolicy_root.iterdir() if path.is_dir()):
        for task_dir in sorted(path for path in algo_root.iterdir() if path.is_dir()):
            for backend_file in sorted(task_dir.glob("*.yaml")):
                cases.append(
                    (
                        "offpolicy",
                        "config",
                        task_dir.name,
                        backend_file.stem,
                        str(backend_file.relative_to(CONF_DIR)),
                        [
                            f"algo={algo_root.name}",
                            f"task={algo_root.name}/{task_dir.name}/{backend_file.stem}",
                        ],
                    )
                )

    return cases


@pytest.mark.parametrize(
    "algo_dir,config_name",
    [
        ("offpolicy", "config"),
        ("appo", "config"),
        ("ppo", "config"),
        ("ppo", "config_mlx"),
    ],
)
def test_algo_config_composes(algo_dir: str, config_name: str):
    cfg = _compose(algo_dir, config_name)
    assert cfg.training.task_name
    assert cfg.training.sim_backend == "mujoco"


def test_legacy_config_groups_removed():
    for path in [
        CONF_DIR / "ppo" / "reward",
        CONF_DIR / "ppo" / "backend_task_preset",
        CONF_DIR / "ppo" / "algo_preset",
        CONF_DIR / "ppo" / "sim_backend",
        CONF_DIR / "appo" / "reward",
        CONF_DIR / "appo" / "backend_task_preset",
        CONF_DIR / "appo" / "sim_backend",
        CONF_DIR / "offpolicy" / "reward",
        CONF_DIR / "offpolicy" / "backend_task_preset",
        CONF_DIR / "offpolicy" / "algo_preset",
        CONF_DIR / "offpolicy" / "sim_backend",
    ]:
        assert not path.exists(), f"legacy config group should be removed: {path}"


def test_task_files_keep_full_identity_without_hidden_backend_marker():
    for path in sorted(CONF_DIR.glob("*/task/**/*.yaml")):
        cfg = OmegaConf.load(path)
        cfg_dict_raw = OmegaConf.to_container(cfg, resolve=True) or {}
        assert isinstance(cfg_dict_raw, dict)
        assert "_selected_sim_backend" not in cfg_dict_raw, (
            f"task has hidden backend marker: {path}"
        )
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


def test_offpolicy_g1_walk_flat_motrix_sac_preserves_backend_overrides():
    cfg = _compose("offpolicy", overrides=["algo=sac", "task=sac/g1_walk_flat/motrix"])

    assert cfg.algo.num_envs == 2048
    assert cfg.algo.max_iterations == 5000
    assert cfg.reward.scales.tracking_lin_vel == pytest.approx(2.2)
    assert cfg.env.domain_rand.randomize_kp is False
    assert cfg.env.domain_rand.randomize_kd is False


def test_offpolicy_g1_walk_flat_mujoco_td3_uses_td3_task_owner():
    cfg = _compose("offpolicy", overrides=["algo=td3", "task=td3/g1_walk_flat/mujoco"])

    assert cfg.training.task_name == "G1WalkFlat"
    assert cfg.training.sim_backend == "mujoco"
    assert cfg.algo.max_iterations == 100000
    assert cfg.reward.scales.tracking_lin_vel == pytest.approx(2.0)
    assert cfg.env.control_config.action_scale == pytest.approx(1.0)


def test_offpolicy_g1_walk_flat_motrix_preserves_backend_specific_algo_value():
    mujoco_cfg = _compose("offpolicy", overrides=["algo=sac", "task=sac/g1_walk_flat/mujoco"])
    motrix_cfg = _compose("offpolicy", overrides=["algo=sac", "task=sac/g1_walk_flat/motrix"])

    assert mujoco_cfg.algo.use_symmetry is True
    assert motrix_cfg.algo.use_symmetry is False


def test_ppo_g1_backend_specific_hyperparams_remain_separate():
    mujoco_cfg = _compose("ppo", overrides=["task=g1_joystick_flat/mujoco"])
    motrix_cfg = _compose("ppo", overrides=["task=g1_joystick_flat/motrix"])

    assert mujoco_cfg.algo.max_iterations == 220
    assert mujoco_cfg.algo.empirical_normalization is False
    assert mujoco_cfg.algo.obs_groups.actor == ["actor"]

    assert motrix_cfg.algo.max_iterations == 151
    assert motrix_cfg.algo.empirical_normalization is True
    assert motrix_cfg.algo.obs_groups.actor == ["policy"]
    assert motrix_cfg.env.iterations == 3
    assert motrix_cfg.env.control_config.action_scale == pytest.approx(0.5)


def test_ppo_go1_motrix_preserves_reward_and_algo_values():
    cfg = _compose("ppo", overrides=["task=go1_joystick_flat/motrix"])

    assert cfg.algo.max_iterations == 151
    assert cfg.algo.empirical_normalization is True
    assert cfg.algo.policy.init_noise_std == pytest.approx(0.5)
    assert cfg.algo.algorithm.learning_rate == pytest.approx(3.0e-4)
    assert cfg.reward.scales.tracking_lin_vel == pytest.approx(1.0)
    assert cfg.env.commands.vel_limit == [[0.5, 0.0, 0.0], [0.5, 0.0, 0.0]]


def test_ppo_go2_motrix_preserves_backend_env_overrides():
    cfg = _compose("ppo", overrides=["task=go2_joystick_flat/motrix"])

    assert cfg.algo.num_envs == 1024
    assert cfg.algo.empirical_normalization is True
    assert cfg.env.domain_rand.randomize_kp is False
    assert cfg.env.domain_rand.randomize_kd is False


def test_offpolicy_g1_walk_flat_motrix_preserves_backend_env_overrides():
    cfg = _compose("offpolicy", overrides=["algo=sac", "task=sac/g1_walk_flat/motrix"])

    assert cfg.training.sim_backend == "motrix"
    assert cfg.algo.num_envs == 2048
    assert cfg.algo.max_iterations == 5000
    assert cfg.env.domain_rand.randomize_kp is False
    assert cfg.env.domain_rand.randomize_kd is False


def test_cli_override_beats_task_defaults():
    cfg = _compose(
        "ppo",
        overrides=["task=g1_joystick_flat/motrix", "algo.max_iterations=1"],
    )

    assert cfg.algo.max_iterations == 1
    assert cfg.algo.empirical_normalization is True
