"""Comprehensive config system verification tests.

These tests enforce that:
1. Every Hydra YAML config composes without errors
2. Every non-default task has a non-empty reward config that reaches cfg.reward
3. Every reward YAML that exists is reachable via some (algo, task, reward) combination
4. The reward config injection chain (Hydra → env_cfg_override → registry.make) works end-to-end
5. sim_backend is present in every algo's training config
"""

from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

CONF_DIR = Path(__file__).parent.parent.parent / "conf"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _compose(algo_dir: str, config_name: str = "config", overrides: list[str] | None = None):
    """Compose a Hydra config from the given algo directory."""
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_DIR / algo_dir), version_base="1.3"):
        return compose(config_name, overrides=overrides or [])


def _assert_reward_populated(cfg, algo: str, task: str):
    """Assert that cfg.reward exists and contains non-empty scales."""
    assert hasattr(cfg, "reward"), (
        f"[{algo}/{task}] cfg.reward missing — check defaults list has 'reward: default'"
    )
    reward = cfg.reward
    assert reward, f"[{algo}/{task}] cfg.reward is empty — task file should override /reward"
    reward_dict = OmegaConf.to_container(reward, resolve=True)
    assert "scales" in reward_dict, f"[{algo}/{task}] cfg.reward has no 'scales' key"
    assert len(reward_dict["scales"]) > 0, f"[{algo}/{task}] cfg.reward.scales is empty"


def _assert_reward_section_loadable(section, label: str):
    """Assert that a composed reward-like section has the expected structure."""
    reward_dict = OmegaConf.to_container(section, resolve=True)
    assert isinstance(reward_dict, dict), f"{label} must resolve to a mapping"
    assert "scales" in reward_dict, f"{label} must have 'scales' key"


def _assert_sim_backend_configurable(cfg, algo: str):
    """Assert that training.sim_backend exists and defaults to mujoco."""
    assert hasattr(cfg.training, "sim_backend"), (
        f"[{algo}] training.sim_backend missing from config"
    )
    assert cfg.training.sim_backend == "mujoco", (
        f"[{algo}] training.sim_backend should default to 'mujoco', got '{cfg.training.sim_backend}'"
    )


# ---------------------------------------------------------------------------
# 1. Every algo config composes at all (smoke test)
# ---------------------------------------------------------------------------


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
    """Each base algorithm config should compose without errors."""
    cfg = _compose(algo_dir, config_name)
    assert cfg.training.task_name, f"[{algo_dir}/{config_name}] task_name is empty"


# ---------------------------------------------------------------------------
# 2. Every algo config has sim_backend in training section
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "algo_dir,config_name",
    [
        ("offpolicy", "config"),
        ("appo", "config"),
        ("ppo", "config"),
        ("ppo", "config_mlx"),
    ],
)
def test_sim_backend_in_config(algo_dir: str, config_name: str):
    """Every algo config must have training.sim_backend."""
    cfg = _compose(algo_dir, config_name)
    _assert_sim_backend_configurable(cfg, algo_dir)


# ---------------------------------------------------------------------------
# 3. Every algo config has 'reward' in defaults
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "algo_dir,config_name",
    [
        ("offpolicy", "config"),
        ("appo", "config"),
        ("ppo", "config"),
        ("ppo", "config_mlx"),
    ],
)
def test_reward_key_in_config(algo_dir: str, config_name: str):
    """Every algo config must declare the reward config group so overrides work.

    Note: default.yaml is intentionally empty, so cfg.reward won't exist with defaults.
    The test verifies that a non-default reward override can be applied without error.
    """
    # Just loading with default is fine (no cfg.reward expected)
    cfg = _compose(algo_dir, config_name)
    assert cfg.training.task_name, "base config should load"

    # But we must be able to override reward explicitly
    # (this would fail if 'reward: default' is missing from the defaults list)
    reward_dir = CONF_DIR / algo_dir / "reward"
    if reward_dir.exists():
        non_default = sorted(f.stem for f in reward_dir.glob("*.yaml") if f.stem != "default")
        if non_default:
            direct_reward = None
            for reward_name in non_default:
                cfg2 = _compose(algo_dir, config_name, overrides=[f"reward={reward_name}"])
                if hasattr(cfg2, "reward"):
                    direct_reward = reward_name
                    break

            assert direct_reward is not None, (
                f"[{algo_dir}/{config_name}] no direct reward override produced cfg.reward "
                f"— add 'reward: default' to defaults list"
            )


# ---------------------------------------------------------------------------
# 4. Every (algo × task) combination composes correctly
# ---------------------------------------------------------------------------


# offpolicy tasks
@pytest.mark.parametrize(
    "task,expected_env",
    [
        ("go1_joystick", "Go1JoystickFlatTerrain"),
        ("go2_joystick", "Go2JoystickFlatTerrain"),
        ("g1_sac", "G1WalkTaskMjSAC"),
        ("allegro_sac", "AllegroInhandRotationSac"),
    ],
)
def test_offpolicy_task_composes(task: str, expected_env: str):
    cfg = _compose("offpolicy", overrides=[f"task={task}"])
    assert cfg.training.task_name == expected_env


# appo tasks
@pytest.mark.parametrize(
    "task,expected_env",
    [
        ("go1_joystick", "Go1JoystickFlatTerrain"),
        ("go2_joystick", "Go2JoystickFlatTerrain"),
        ("g1_joystick", "G1JoystickFlatTerrain"),
    ],
)
def test_appo_task_composes(task: str, expected_env: str):
    cfg = _compose("appo", overrides=[f"task={task}"])
    assert cfg.training.task_name == expected_env


# ppo tasks (rsl_rl + mlx)
@pytest.mark.parametrize(
    "task,expected_env",
    [
        ("go1_joystick", "Go1JoystickFlatTerrain"),
        ("go2_joystick", "Go2JoystickFlatTerrain"),
        ("g1_joystick", "G1JoystickFlatTerrain"),
        ("g1_motion_tracking", "G1MotionTracking"),
        ("allegro_inhand", "AllegroInhandRotation"),
    ],
)
def test_ppo_task_composes(task: str, expected_env: str):
    cfg = _compose("ppo", overrides=[f"task={task}"])
    assert cfg.training.task_name == expected_env


@pytest.mark.parametrize(
    "task,expected_env",
    [
        ("go1_joystick", "Go1JoystickFlatTerrain"),
        ("go2_joystick", "Go2JoystickFlatTerrain"),
        ("g1_joystick", "G1JoystickFlatTerrain"),
    ],
)
def test_mlx_ppo_task_composes(task: str, expected_env: str):
    cfg = _compose("ppo", "config_mlx", overrides=[f"task={task}"])
    assert cfg.training.task_name == expected_env


# ---------------------------------------------------------------------------
# 5. Tasks with non-default reward must populate cfg.reward.scales
#    (g1_motion_tracking uses env defaults, excluded here)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "task",
    ["go1_joystick", "g1_sac"],
)
def test_offpolicy_locomotion_reward_populated(task: str):
    """Offpolicy locomotion tasks with reward override must have non-empty scales."""
    cfg = _compose("offpolicy", overrides=[f"task={task}"])
    _assert_reward_populated(cfg, "offpolicy", task)


@pytest.mark.parametrize(
    "task",
    ["go1_joystick", "go2_joystick", "g1_joystick"],
)
def test_ppo_locomotion_reward_populated(task: str):
    """PPO locomotion tasks must have non-empty reward scales."""
    cfg = _compose("ppo", overrides=[f"task={task}"])
    _assert_reward_populated(cfg, "ppo", task)


@pytest.mark.parametrize(
    "task",
    ["go1_joystick", "go2_joystick", "g1_joystick"],
)
def test_mlx_ppo_locomotion_reward_populated(task: str):
    """MLX PPO locomotion tasks must have non-empty reward scales (shared with PPO)."""
    cfg = _compose("ppo", "config_mlx", overrides=[f"task={task}"])
    _assert_reward_populated(cfg, "mlx_ppo", task)


# ---------------------------------------------------------------------------
# 6. Every reward YAML is loadable (no typos, valid structure)
# ---------------------------------------------------------------------------


def _all_reward_yamls():
    """Discover all reward YAML files across all algo directories."""
    results = []
    for algo_dir in ["offpolicy", "appo", "ppo"]:
        reward_dir = CONF_DIR / algo_dir / "reward"
        if reward_dir.exists():
            for f in sorted(reward_dir.glob("*.yaml")):
                results.append((algo_dir, f.stem))
    return results


@pytest.mark.parametrize("algo_dir,reward_name", _all_reward_yamls())
def test_reward_yaml_loadable(algo_dir: str, reward_name: str):
    """Every reward YAML file must compose through its supported override path."""
    config_name = "config"
    label = f"[{algo_dir}/reward/{reward_name}]"
    cfg = _compose(algo_dir, config_name, overrides=[f"reward={reward_name}"])
    if reward_name == "default":
        # default.yaml is intentionally empty — cfg.reward won't exist
        return
    if hasattr(cfg, "reward"):
        _assert_reward_section_loadable(cfg.reward, label)
        return

    # G1 motion-tracking motrix rewards are sidecar backend-specific configs that
    # are selected via `reward@reward_motrix=...` on the task config.
    if algo_dir in {"appo", "ppo"} and reward_name == "g1_motion_tracking_motrix":
        cfg = _compose(
            algo_dir,
            config_name,
            overrides=[
                "task=g1_motion_tracking",
                f"reward@reward_motrix={reward_name}",
            ],
        )
        assert hasattr(cfg, "reward_motrix"), f"{label} cfg.reward_motrix missing after compose"
        _assert_reward_section_loadable(cfg.reward_motrix, label)
        return

    pytest.fail(f"{label} cfg.reward missing after compose")


# ---------------------------------------------------------------------------
# 7. Motrix reward override composes correctly via CLI
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "algo_dir,task,reward",
    [
        ("offpolicy", "go1_joystick", "go1_sac_motrix"),
        ("offpolicy", "g1_sac", "g1_sac_motrix"),
        ("ppo", "go1_joystick", "go1_ppo_motrix"),
        ("ppo", "go2_joystick", "go2_ppo_motrix"),
        ("ppo", "g1_joystick", "g1_ppo_motrix"),
    ],
)
def test_motrix_reward_override(algo_dir: str, task: str, reward: str):
    """Motrix reward configs must compose with sim_backend=motrix."""
    cfg = _compose(
        algo_dir,
        overrides=[f"task={task}", f"reward={reward}", "training.sim_backend=motrix"],
    )
    assert cfg.training.sim_backend == "motrix"
    _assert_reward_populated(cfg, algo_dir, task)


# ---------------------------------------------------------------------------
# 8. Reward config injection end-to-end: Hydra → env_cfg_override → env
# ---------------------------------------------------------------------------


def test_reward_injection_reaches_env_go1():
    """Verify reward config actually reaches the environment via registry.make."""
    from unilab.base import registry
    from unilab.utils.algo_utils import ensure_registries

    ensure_registries()

    cfg = _compose("ppo", overrides=["task=go1_joystick"])
    reward_dict = OmegaConf.to_container(cfg.reward, resolve=True)
    env_cfg_override = {"reward_config": reward_dict}

    env = registry.make(
        "Go1JoystickFlatTerrain",
        num_envs=1,
        sim_backend="mujoco",
        env_cfg_override=env_cfg_override,
    )
    # Verify the injected config matches what we specified
    actual_scales = env._cfg.reward_config.scales
    expected_scales = reward_dict["scales"]
    for key, expected_val in expected_scales.items():
        assert key in actual_scales, f"Missing reward scale key: {key}"
        assert actual_scales[key] == pytest.approx(expected_val), (
            f"Reward scale '{key}' mismatch: {actual_scales[key]} != {expected_val}"
        )
    env.close()


def test_reward_injection_reaches_env_go2():
    """Verify reward config reaches Go2 env."""
    from unilab.base import registry
    from unilab.utils.algo_utils import ensure_registries

    ensure_registries()

    cfg = _compose("ppo", overrides=["task=go2_joystick"])
    reward_dict = OmegaConf.to_container(cfg.reward, resolve=True)
    env_cfg_override = {"reward_config": reward_dict}

    env = registry.make(
        "Go2JoystickFlatTerrain",
        num_envs=1,
        sim_backend="mujoco",
        env_cfg_override=env_cfg_override,
    )
    actual_scales = env._cfg.reward_config.scales
    assert actual_scales["tracking_lin_vel"] == pytest.approx(1.0)
    env.close()


def test_reward_injection_reaches_env_g1():
    """Verify reward config reaches G1 PPO env."""
    from unilab.base import registry
    from unilab.utils.algo_utils import ensure_registries

    ensure_registries()

    cfg = _compose("ppo", overrides=["task=g1_joystick"])
    reward_dict = OmegaConf.to_container(cfg.reward, resolve=True)
    env_cfg_override = {"reward_config": reward_dict}

    env = registry.make(
        "G1JoystickFlatTerrain",
        num_envs=1,
        sim_backend="mujoco",
        env_cfg_override=env_cfg_override,
    )
    actual_scales = env._cfg.reward_config.scales
    assert "feet_phase" in actual_scales, "G1 PPO should have feet_phase scale"
    assert actual_scales["tracking_lin_vel"] == pytest.approx(2.0)
    env.close()


def test_reward_injection_changes_values():
    """Verify that injecting a different reward config actually changes env behavior."""
    from unilab.base import registry
    from unilab.utils.algo_utils import ensure_registries

    ensure_registries()

    # Create env with default reward config
    default_reward = {
        "scales": {
            "tracking_lin_vel": 1.0,
            "tracking_ang_vel": 0.2,
            "lin_vel_z": -5.0,
            "ang_vel_xy": -0.1,
            "base_height": -100.0,
            "action_rate": -0.005,
            "similar_to_default": -0.1,
            "contact": 0.24,
        },
        "tracking_sigma": 0.25,
        "base_height_target": 0.3,
    }
    env_default = registry.make(
        "Go1JoystickFlatTerrain",
        num_envs=1,
        sim_backend="mujoco",
        env_cfg_override={"reward_config": default_reward},
    )
    default_scales = dict(env_default._cfg.reward_config.scales)
    env_default.close()

    # Create env with custom reward (double tracking_lin_vel)
    custom_override = {
        "reward_config": {
            "scales": {**default_scales, "tracking_lin_vel": 999.0},
            "tracking_sigma": 0.25,
            "base_height_target": 0.3,
        }
    }
    env_custom = registry.make(
        "Go1JoystickFlatTerrain",
        num_envs=1,
        sim_backend="mujoco",
        env_cfg_override=custom_override,
    )
    assert env_custom._cfg.reward_config.scales["tracking_lin_vel"] == pytest.approx(999.0)
    env_custom.close()


# ---------------------------------------------------------------------------
# 9. Default reward (empty) does not break env creation
# ---------------------------------------------------------------------------


def test_default_reward_creates_env_with_hardcoded_defaults():
    """Reward config must be provided - env creation should fail without it."""
    from unilab.base import registry
    from unilab.utils.algo_utils import ensure_registries

    ensure_registries()

    # Should fail without reward_config
    with pytest.raises(ValueError, match="reward_config must be provided"):
        registry.make("Go1JoystickFlatTerrain", num_envs=1, sim_backend="mujoco")


# ---------------------------------------------------------------------------
# 10. Reward YAML values match env dataclass defaults (consistency check)
# ---------------------------------------------------------------------------


def test_go1_ppo_mujoco_matches_env_defaults():
    """RewardConfig no longer has defaults - this test is obsolete."""
    pytest.skip("RewardConfig defaults removed - all configs must come from Hydra YAML")


def test_go2_ppo_mujoco_matches_env_defaults():
    """RewardConfig no longer has defaults - this test is obsolete."""
    pytest.skip("RewardConfig defaults removed - all configs must come from Hydra YAML")


def test_g1_ppo_mujoco_matches_env_defaults():
    """RewardConfigPPO no longer has defaults - this test is obsolete."""
    pytest.skip("RewardConfigPPO defaults removed - all configs must come from Hydra YAML")


# ---------------------------------------------------------------------------
# 11. Verify training scripts extract reward config correctly (unit test)
# ---------------------------------------------------------------------------


def test_offpolicy_reward_extraction_pattern():
    """Verify the reward extraction logic used in train_offpolicy.py."""
    cfg = _compose("offpolicy", overrides=["task=go1_joystick"])
    # Simulate train_offpolicy.py build_runner logic
    env_cfg_override = None
    if hasattr(cfg, "reward") and cfg.reward:
        reward_dict = OmegaConf.to_container(cfg.reward, resolve=True)
        env_cfg_override = {"reward_config": reward_dict}
    assert env_cfg_override is not None, "env_cfg_override should not be None for go1_joystick"
    assert "scales" in env_cfg_override["reward_config"]
    assert env_cfg_override["reward_config"]["scales"]["tracking_lin_vel"] == pytest.approx(1.0)


def test_ppo_reward_extraction_pattern():
    """Verify the reward extraction logic used in train_rsl_rl.py."""
    cfg = _compose("ppo", overrides=["task=g1_joystick"])
    env_cfg_override: dict = {}
    if hasattr(cfg, "reward") and cfg.reward:
        reward_dict = OmegaConf.to_container(cfg.reward, resolve=True)
        env_cfg_override["reward_config"] = reward_dict
    assert "reward_config" in env_cfg_override, "PPO g1_joystick should extract reward"
    assert env_cfg_override["reward_config"]["scales"]["feet_phase"] == pytest.approx(1.0)


def test_motion_tracking_reward_extraction_pattern():
    """Verify g1_motion_tracking now extracts explicit motion-tracking reward config."""
    cfg = _compose("ppo", overrides=["task=g1_motion_tracking"])
    env_cfg_override: dict = {}
    if hasattr(cfg, "reward") and cfg.reward:
        reward_dict = OmegaConf.to_container(cfg.reward, resolve=True)
        env_cfg_override["reward_config"] = reward_dict
    assert "reward_config" in env_cfg_override
    assert env_cfg_override["reward_config"]["scales"]["motion_body_pos"] == pytest.approx(1.0)


def test_appo_reward_extraction_pattern():
    """Verify APPO go1 task extracts its default reward override."""
    cfg = _compose("appo", overrides=["task=go1_joystick"])
    env_cfg_override: dict = {}
    if hasattr(cfg, "reward") and cfg.reward:
        reward_dict = OmegaConf.to_container(cfg.reward, resolve=True)
        env_cfg_override["reward_config"] = reward_dict
    assert "reward_config" in env_cfg_override
    assert env_cfg_override["reward_config"]["scales"]["tracking_lin_vel"] == pytest.approx(1.0)


def test_appo_custom_reward_extraction():
    """Verify APPO with explicit reward override extracts correctly."""
    cfg = _compose("appo", overrides=["task=go1_joystick", "reward=go1_appo_mujoco"])
    env_cfg_override: dict = {}
    if hasattr(cfg, "reward") and cfg.reward:
        reward_dict = OmegaConf.to_container(cfg.reward, resolve=True)
        env_cfg_override["reward_config"] = reward_dict
    assert "reward_config" in env_cfg_override
    assert env_cfg_override["reward_config"]["scales"]["tracking_lin_vel"] == pytest.approx(1.0)
