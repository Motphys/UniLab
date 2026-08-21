"""Tests for script entry-point utilities (pure functions and Hydra config defaults).

Coverage targets:
  - train_offpolicy.py: Hydra defaults, default_device(), resolve_checkpoint_path()
  - play_interactive.py: resolve_checkpoint()                       (skipped if mujoco absent)
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from unilab.base.backend.motrix.playback import run_motrix_playback

_SCRIPTS_DIR = Path(__file__).parent.parent.parent / "scripts"
_CONF_DIR = Path(__file__).parent.parent.parent / "conf"
_SRC_DIR = Path(__file__).parent.parent.parent / "src"


def _normalize_overrides(overrides: list[str] | None, *, offpolicy: bool = False) -> list[str]:
    normalized: list[str] = []
    task_selected = False

    for override in overrides or []:
        if override.startswith("task="):
            task_selected = True
        normalized.append(override)

    if not task_selected:
        if offpolicy:
            normalized.append("task=g1_walk_flat/mujoco")
        else:
            normalized.append("task=go1_joystick_flat/mujoco")
    return normalized


def _load_script(name: str) -> Any:
    """Load a scripts/<name>.py as a fresh module (no __init__ required)."""
    path = _SCRIPTS_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def test_analyze_offpolicy_trace_reports_training_e2e(tmp_path, capsys):
    trace_path = tmp_path / "trace.json"
    trace_path.write_text(
        json.dumps(
            {
                "traceEvents": [
                    {"name": "learner/wait_for_data", "ph": "X", "ts": 0.0, "dur": 10.0},
                    {"name": "learner/wait_for_data", "ph": "X", "ts": 1000.0, "dur": 10.0},
                    {"name": "learner/training_e2e", "ph": "X", "ts": 0.0, "dur": 2500.0},
                ]
            }
        ),
        encoding="utf-8",
    )
    mod = _load_script("analyze_offpolicy_trace")

    mod.analyze_training_e2e(trace_path)

    out = capsys.readouterr().out
    assert "training_e2e: n=1 mean=2.500ms" in out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


try:
    import mujoco  # noqa: F401

    _HAS_MUJOCO = True
except ImportError:
    _HAS_MUJOCO = False


# ---------------------------------------------------------------------------
# train_sac.py / train_td3.py / train_flashsac.py — Hydra config defaults
# (composed from the per-algo trees conf/sac, conf/td3, conf/flashsac)
# ---------------------------------------------------------------------------


def _offpolicy_cfg(overrides=None, *, algo: str = "sac"):
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(_CONF_DIR / algo), version_base="1.3"):
        return compose(
            "config",
            overrides=_normalize_overrides(overrides, offpolicy=True),
            return_hydra_config=True,
        )


def _ppo_cfg(overrides=None):
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(_CONF_DIR / "ppo"), version_base="1.3"):
        return compose("config", overrides=_normalize_overrides(overrides))


def _appo_cfg(overrides=None):
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(_CONF_DIR / "appo"), version_base="1.3"):
        return compose("config", overrides=_normalize_overrides(overrides))


def _hora_distill_cfg(overrides=None):
    """Compose the HORA distillation Hydra config.

    Args:
        overrides: Optional Hydra override strings to apply during composition.

    Returns:
        The composed HORA distillation config.
    """
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(_CONF_DIR / "hora_distill"), version_base="1.3"):
        return compose("config", overrides=overrides or [])


def _train_rsl_rl(monkeypatch: pytest.MonkeyPatch):
    import types

    for module_name in list(sys.modules):
        if module_name == "unilab" or module_name.startswith("unilab."):
            monkeypatch.delitem(sys.modules, module_name, raising=False)

    runners_mod = cast(Any, types.ModuleType("rsl_rl.runners"))
    runners_mod.OnPolicyRunner = object
    rsl_pkg = cast(Any, types.ModuleType("rsl_rl"))
    rsl_pkg.runners = runners_mod
    monkeypatch.setitem(sys.modules, "rsl_rl", rsl_pkg)
    monkeypatch.setitem(sys.modules, "rsl_rl.runners", runners_mod)
    return _load_script("train_rsl_rl")


def _train_appo():
    return _load_script("train_appo")


def _train_hora_distill():
    """Load the HORA distillation entrypoint module.

    Args:
        None.

    Returns:
        The loaded ``scripts/train_hora_distill.py`` module.
    """
    return _load_script("train_hora_distill")


def test_offpolicy_hydra_default_algo():
    cfg = _offpolicy_cfg()
    assert cfg.algo.algo == "sac"


def test_appo_runner_kwargs_forward_algorithm_seed():
    mod = _train_appo()
    cfg = _appo_cfg(["algo.seed=37"])
    rl_cfg = OmegaConf.to_container(cfg.algo, resolve=True)

    kwargs = mod.build_appo_runner_kwargs(
        cfg,
        env_cfg_override={"reward_config": {}},
        collector_device="cpu",
        rl_cfg=cast(dict[str, Any], rl_cfg),
    )

    assert kwargs["seed"] == 37


def test_appo_runner_kwargs_default_load_run_does_not_resume(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = _train_appo()
    cfg = _appo_cfg(["task=allegro_inhand/mujoco", "algo.load_run=-1"])

    def fail_resolve(*args, **kwargs):
        del args, kwargs
        raise AssertionError("training default load_run=-1 must not request resume")

    monkeypatch.setattr(mod, "resolve_appo_checkpoint_path", fail_resolve)

    kwargs = mod.build_appo_runner_kwargs(
        cfg,
        env_cfg_override={"reward_config": {}},
        collector_device="cpu",
    )

    assert "resume_path" not in kwargs


def test_appo_runner_kwargs_explicit_load_run_sets_resume_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    mod = _train_appo()
    cfg = _appo_cfg(["task=allegro_inhand/mujoco", "algo.load_run=run1"])
    log_root = tmp_path / "logs" / "appo"
    run_dir = log_root / cfg.training.task_name / "run1"
    run_dir.mkdir(parents=True)
    (run_dir / "model_3.pt").write_bytes(b"")
    (run_dir / "model_9.pt").write_bytes(b"")
    monkeypatch.setattr(mod, "_get_log_root", lambda _cfg: str(log_root))

    kwargs = mod.build_appo_runner_kwargs(
        cfg,
        env_cfg_override={"reward_config": {}},
        collector_device="cpu",
    )

    assert kwargs["resume_path"] == str(run_dir / "model_9.pt")


def test_appo_runner_kwargs_missing_explicit_load_run_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    mod = _train_appo()
    cfg = _appo_cfg(["task=allegro_inhand/mujoco", "algo.load_run=missing_run"])
    monkeypatch.setattr(mod, "_get_log_root", lambda _cfg: str(tmp_path / "logs" / "appo"))

    with pytest.raises(FileNotFoundError, match="missing_run"):
        mod.build_appo_runner_kwargs(
            cfg,
            env_cfg_override={"reward_config": {}},
            collector_device="cpu",
        )


def test_offpolicy_hydra_default_task():
    cfg = _offpolicy_cfg()
    assert cfg.training.task_name == "G1WalkFlat"


def test_offpolicy_hydra_default_logger():
    cfg = _offpolicy_cfg()
    assert cfg.training.logger == "tensorboard"


def test_offpolicy_hydra_default_wandb_fields():
    cfg = _offpolicy_cfg()
    assert cfg.training.wandb_project == "unilab"
    assert cfg.training.wandb_entity is None
    assert cfg.training.wandb_group is None
    assert cfg.training.wandb_job_type is None
    assert cfg.training.wandb_name is None
    assert cfg.training.wandb_tags == []
    assert cfg.training.wandb_notes is None
    assert cfg.training.wandb_mode is None


def test_offpolicy_hydra_default_sim_backend():
    cfg = _offpolicy_cfg()
    assert cfg.training.sim_backend == "mujoco"


def test_ppo_hydra_default_wandb_fields():
    cfg = _ppo_cfg()
    assert cfg.training.wandb_project == "unilab"
    assert cfg.training.wandb_entity is None
    assert cfg.training.wandb_group is None
    assert cfg.training.wandb_job_type is None
    assert cfg.training.wandb_name is None
    assert cfg.training.wandb_tags == []
    assert cfg.training.wandb_notes is None
    assert cfg.training.wandb_mode is None


def test_offpolicy_hydra_default_play_flags():
    cfg = _offpolicy_cfg()
    assert cfg.training.play_only is False
    assert cfg.training.no_play is False
    assert cfg.training.export_onnx is True
    assert cfg.algo.load_run == "-1"


def test_offpolicy_hydra_default_trace_flags():
    cfg = _offpolicy_cfg()
    assert cfg.training.trace_enabled is False
    assert cfg.training.trace_output_dir is None
    assert cfg.training.trace_thread_time is False
    assert cfg.training.trace_cuda_events is True
    assert cfg.training.nvtx_profile_ranges is False
    assert "verbose_metrics" not in cfg.training
    assert "replay_pipeline" not in cfg.training
    assert "replay_h2d_submitter" not in cfg.training


def test_offpolicy_hydra_default_torch_thread_budget():
    cfg = _offpolicy_cfg()
    assert cfg.training.torch_threads.enabled is True
    assert cfg.training.torch_threads.learner_num_threads == "auto"
    assert cfg.training.torch_threads.collector_num_threads == "auto"
    assert cfg.training.torch_threads.learner_num_interop_threads == 1
    assert cfg.training.torch_threads.collector_num_interop_threads == 1
    assert cfg.training.torch_threads.compile_threads == "auto"
    assert cfg.training.torch_threads.set_env_vars is True


def test_offpolicy_hydra_algo_td3():
    cfg = _offpolicy_cfg(algo="td3")
    assert cfg.algo.algo == "td3"


def test_hora_distill_run_config_records_hardware(tmp_path, monkeypatch):
    mod = _train_hora_distill()
    hardware = {
        "platform": "test-platform",
        "chip": "test-cpu",
        "cpu_total_cores": "8",
        "gpu_name": "test-gpu",
        "memory": "32 GB",
    }
    monkeypatch.setattr(mod, "get_device_info_dict", lambda: hardware)
    cfg = OmegaConf.create({"training": {"task_name": "Task", "sim_backend": "mujoco"}})

    mod._write_distill_run_config(
        tmp_path,
        cfg=cfg,
        teacher_metadata={"checkpoint_path": "teacher.pt"},
    )

    payload = json.loads((tmp_path / "distill_run_config.json").read_text(encoding="utf-8"))
    assert payload["run"]["hardware"] == hardware


def test_hora_distill_task_owner_overrides_root_config_defaults():
    mod = _train_hora_distill()
    root_cfg = OmegaConf.load(_CONF_DIR / "hora_distill" / "config.yaml")
    cfg = mod._apply_teacher_defaults(_hora_distill_cfg(["task=sharpa_inhand/mujoco"]))

    assert root_cfg.algo.num_envs == 4096
    assert root_cfg.algo.save_interval_steps == 100000000
    assert cfg.algo.num_envs == 16384
    assert cfg.algo.save_interval_steps == 10000000


def test_hora_distill_sharpa_appo_student_owner_selects_nodr_demo_profile():
    mod = _train_hora_distill()
    cfg = mod._apply_teacher_defaults(_hora_distill_cfg(["task=sharpa_inhand/mujoco_nodr"]))

    assert cfg.teacher.algo_family == "appo"
    assert cfg.teacher.task == "sharpa_inhand/mujoco_hora"
    assert cfg.training.task_name == "SharpaInhandRotation"
    assert cfg.training.sim_backend == "mujoco"
    assert cfg.interactive.action_mode == "policy"
    assert cfg.interactive.policy_obs_mode == "actor"
    assert cfg.env.post_step_forward_sensor is True
    assert cfg.env.domain_rand.scale_list == [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
    assert cfg.env.domain_rand.randomize_gravity is False
    assert cfg.env.domain_rand.randomize_gravity_direction is False
    assert cfg.env.domain_rand.randomize_pd_gains is False
    assert cfg.env.domain_rand.randomize_friction is False
    assert cfg.env.domain_rand.randomize_com is False
    assert cfg.env.domain_rand.randomize_mass is False
    assert cfg.env.domain_rand.force_scale == pytest.approx(0.0)
    assert cfg.env.domain_rand.random_force_prob_scalar == pytest.approx(0.0)
    assert cfg.env.domain_rand.joint_noise_scale == pytest.approx(0.0)
    assert cfg.env.domain_rand.contact_latency == pytest.approx(0.0)
    assert cfg.env.domain_rand.contact_sensor_noise == pytest.approx(0.0)
    assert cfg.algo.model.priv_info_embed_dim == 9
    assert cfg.algo.model.priv_mlp_hidden_dims == [256, 128, 9]


def test_hora_distill_runtime_checkpoint_records_model_only():
    mod = _train_hora_distill()
    cfg = OmegaConf.create(
        {
            "training": {
                "task_name": "OwnerTask",
                "sim_backend": "mujoco",
                "cam_distance": 1.5,
            },
            "env": {
                "post_step_forward_sensor": True,
                "domain_rand": {"force_scale": 1.2},
            },
            "reward": {"scales": {"rotate": 2.5}},
            "algo": {"model": {"hidden_dims": [512, 256, 128]}},
        }
    )

    runtime = OmegaConf.to_container(mod._resolved_distill_runtime_cfg(cfg), resolve=True)

    assert runtime == {"algo": {"model": {"hidden_dims": [512, 256, 128]}}}


def test_hora_distill_checkpoint_runtime_only_restores_model_structure():
    from unilab.algos.hora.distill import cfg_with_checkpoint_runtime

    cfg = _hora_distill_cfg(["task=sharpa_inhand/mujoco_nodr"])
    checkpoint = {
        "distill_runtime_cfg": {
            "training": {
                "task_name": "CheckpointTask",
                "sim_backend": "motrix",
                "render_spacing": 99.0,
            },
            "reward": {"scales": {"rotate": 999.0}},
            "env": {
                "post_step_forward_sensor": False,
                "domain_rand": {
                    "scale_list": [9.9],
                    "randomize_mass": True,
                    "force_scale": 99.0,
                },
            },
            "algo": {
                "model": {
                    "hidden_dims": [32, 16],
                    "priv_info_embed_dim": 7,
                    "priv_mlp_hidden_dims": [11, 7],
                }
            },
        }
    }

    restored = cfg_with_checkpoint_runtime(cfg, checkpoint)

    assert restored.training.task_name == "SharpaInhandRotation"
    assert restored.training.sim_backend == "mujoco"
    assert restored.training.render_spacing == pytest.approx(0.5)
    assert restored.reward.scales.rotate != pytest.approx(999.0)
    assert restored.env.post_step_forward_sensor is True
    assert restored.env.domain_rand.scale_list == [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
    assert restored.env.domain_rand.randomize_mass is False
    assert restored.env.domain_rand.force_scale == pytest.approx(0.0)
    assert restored.algo.model.hidden_dims == [32, 16]
    assert restored.algo.model.priv_info_embed_dim == 7
    assert restored.algo.model.priv_mlp_hidden_dims == [11, 7]


@pytest.mark.parametrize(
    ("teacher_algo_family", "checkpoint_model"),
    [
        ("ppo", {"hidden_dims": [512, 256, 128], "activation": "elu"}),
        ("appo", {"hidden_dims": [512, 256, 128], "activation": "elu"}),
        (
            "sac",
            {
                "teacher_arch": "hora_sac",
                "actor_hidden_dim": 512,
                "use_layer_norm": True,
            },
        ),
    ],
)
def test_hora_distill_checkpoint_runtime_only_overrides_model_side(
    monkeypatch: pytest.MonkeyPatch,
    teacher_algo_family: str,
    checkpoint_model: dict[str, Any],
):
    from unilab.algos.hora import distill_config
    from unilab.algos.hora.distill import cfg_with_checkpoint_runtime

    owner_cfg = OmegaConf.create(
        {
            "teacher": {"algo_family": teacher_algo_family},
            "training": {
                "task_name": "OwnerTask",
                "sim_backend": "mujoco",
                "cam_distance": 1.5,
            },
            "env": {
                "post_step_forward_sensor": False,
                "domain_rand": {"force_scale": 1.2, "randomize_mass": False},
            },
            "reward": {"scales": {"rotate": 2.5}},
            "algo": {"model": {"owner_model": True}},
        }
    )
    checkpoint = {
        "teacher_algo_family": teacher_algo_family,
        "distill_runtime_cfg": {
            "training": {
                "task_name": "CheckpointTask",
                "sim_backend": "mujoco",
                "cam_distance": 9.0,
            },
            "env": {
                "post_step_forward_sensor": True,
                "domain_rand": {"force_scale": 9.0, "randomize_mass": True},
            },
            "reward": {"scales": {"rotate": 99.0}},
            "algo": {"model": checkpoint_model},
        },
    }

    monkeypatch.setattr(distill_config, "apply_teacher_defaults", lambda cfg: owner_cfg)

    effective_cfg = cfg_with_checkpoint_runtime(OmegaConf.create({}), checkpoint)

    assert effective_cfg.training.task_name == "OwnerTask"
    assert effective_cfg.training.cam_distance == pytest.approx(1.5)
    assert effective_cfg.env.post_step_forward_sensor is False
    assert effective_cfg.env.domain_rand.force_scale == pytest.approx(1.2)
    assert effective_cfg.env.domain_rand.randomize_mass is False
    assert effective_cfg.reward.scales.rotate == pytest.approx(2.5)
    assert OmegaConf.to_container(effective_cfg.algo.model, resolve=True) == checkpoint_model


def test_hora_distill_script_delegates_teacher_owner_resolution():
    source = (_SCRIPTS_DIR / "train_hora_distill.py").read_text(encoding="utf-8")

    assert "OmegaConf.load" not in source
    assert "HoraActorModel" not in source
    assert 'conf" / str(algo_family)' not in source


@pytest.mark.parametrize("teacher_algo_family", ["ppo", "appo", "sac"])
def test_hora_distill_teacher_owner_defaults_support_ppo_appo_and_sac(
    teacher_algo_family: str,
):
    mod = _train_hora_distill()
    teacher_task = "sharpa_inhand/mujoco_hora"
    cfg = mod._apply_teacher_defaults(
        _hora_distill_cfg(
            [
                "task=sharpa_inhand/mujoco",
                f"teacher.algo_family={teacher_algo_family}",
                f"teacher.task={teacher_task}",
            ]
        )
    )

    assert cfg.training.task_name == "SharpaInhandRotation"
    assert cfg.training.sim_backend == "mujoco"
    assert cfg.algo.model.priv_info_embed_dim == 9
    assert cfg.algo.model.priv_mlp_hidden_dims == [256, 128, 9]
    if teacher_algo_family == "sac":
        assert cfg.algo.model.teacher_arch
        assert cfg.algo.model.actor_hidden_dim is not None


def test_hora_distill_sac_teacher_requires_hora_sac_runtime():
    mod = _train_hora_distill()

    with pytest.raises(ValueError, match="runtime_impl='hora_sac'"):
        mod._apply_teacher_defaults(
            _hora_distill_cfg(
                [
                    "task=sharpa_inhand/mujoco",
                    "teacher.algo_family=sac",
                    "teacher.task=g1_walk_flat/mujoco",
                ]
            )
        )


@pytest.mark.parametrize("teacher_algo_family", ["ppo", "appo"])
def test_hora_distill_teacher_run_slug_omits_teacher_run_name(teacher_algo_family: str):
    mod = _train_hora_distill()
    cfg = OmegaConf.create({"teacher": {"task": "sharpa_inhand/mujoco"}})
    teacher_checkpoint = Path("/tmp") / "2026-04-22_13-26-45_mujoco" / "model_10000.pt"

    metadata = mod._teacher_run_metadata(
        cfg,
        teacher_algo_family=teacher_algo_family,
        teacher_checkpoint=teacher_checkpoint,
    )

    assert metadata["run_name"] == "2026-04-22_13-26-45_mujoco"
    assert metadata["run_slug"] == f"teacher-{teacher_algo_family}"


def test_offpolicy_go1_motrix_task_is_not_configured():
    """SAC has no Go1 Motrix owner config; use PPO for Go1 joystick tasks."""
    from hydra.errors import MissingConfigException

    with pytest.raises(MissingConfigException, match="task/go1_joystick_flat/motrix"):
        _offpolicy_cfg(["task=go1_joystick_flat/motrix"])


def test_offpolicy_g1_walk_flat_motrix_resolved_algo_matches_task_owner():
    """Motrix SAC G1 walk flat composes backend-owned algo hyperparameters."""
    cfg = _offpolicy_cfg(["task=g1_walk_flat/motrix"])

    assert cfg.algo.num_envs == 2048
    assert cfg.algo.max_iterations == 5000
    assert cfg.algo.use_symmetry is False


def test_offpolicy_g1_walk_flat_env_cfg_override_has_rewards_and_events():
    cfg = _offpolicy_cfg(["task=g1_walk_flat/motrix"])

    env_cfg_override = _offpolicy().build_offpolicy_env_cfg_override("sac", cfg)

    assert env_cfg_override["rewards"]["tracking_lin_vel"]["weight"] == pytest.approx(2.2)
    assert env_cfg_override["events"]["pd_gains"] is None


def test_offpolicy_g1_walk_flat_backend_scoped_use_symmetry():
    mujoco_cfg = _offpolicy_cfg(["task=g1_walk_flat/mujoco"])
    motrix_cfg = _offpolicy_cfg(["task=g1_walk_flat/motrix"])

    assert mujoco_cfg.algo.use_symmetry is True
    assert motrix_cfg.algo.use_symmetry is False


def test_ppo_go1_resolved_algo_matches_old_motrix_behavior():
    """Equivalence: PPO Go1 algo hyperparams match pre-refactor motrix values."""
    cfg = _ppo_cfg(["task=go1_joystick_flat/motrix"])

    assert cfg.algo.max_iterations == 151
    assert cfg.algo.empirical_normalization is True
    assert cfg.algo.policy.init_noise_std == pytest.approx(0.5)
    assert cfg.algo.algorithm.learning_rate == pytest.approx(3.0e-4)
    assert cfg.algo.algorithm.entropy_coef == pytest.approx(1.0e-3)


def test_ppo_g1_resolved_algo_matches_motrix_owner():
    """Equivalence: PPO G1 algo hyperparams match the Motrix owner values.

    For this migration we align with the final UniLab1 Motrix runtime.
    """
    cfg = _ppo_cfg(["task=g1_walk_flat/motrix"])

    assert cfg.algo.max_iterations == 2200
    assert cfg.algo.empirical_normalization is True
    assert cfg.algo.obs_groups.actor == ["policy"]
    assert cfg.algo.policy.init_noise_std == pytest.approx(0.5)
    assert cfg.algo.algorithm.learning_rate == pytest.approx(3.0e-4)
    assert cfg.algo.algorithm.entropy_coef == pytest.approx(5.0e-3)


def test_ppo_g1_mujoco_base_hyperparams_remain_separate():
    cfg = _ppo_cfg(["task=g1_walk_flat/mujoco"])

    assert cfg.algo.max_iterations == 2200
    assert cfg.algo.empirical_normalization is False
    assert cfg.algo.obs_groups.actor == ["actor"]


def test_ppo_g1_env_preset_has_env_overrides():
    cfg = _ppo_cfg(["task=g1_walk_flat/motrix"])

    assert OmegaConf.select(cfg, "env.motrix_max_iterations") is None
    assert cfg.env.actions.joint_pos.scale == pytest.approx(0.5)
    assert cfg.env.commands.twist.ranges.lin_vel_x == [0.4, 0.7]
    assert cfg.env.observations.policy.terms.gait_phase.params.init_mode == "offset_phase"
    assert cfg.env.events.reset_root_state_uniform.params.velocity_range.x == [-0.05, 0.05]
    assert cfg.reward.feet_phase_contrast.weight == pytest.approx(1.5)
    assert cfg.reward.feet_phase_contact.weight == pytest.approx(1.0)
    assert cfg.reward.feet_double_stance.weight == pytest.approx(-1.0)
    assert cfg.reward.feet_phase.params.min_forward_speed == pytest.approx(0.05)


def test_ppo_task_go2_aligns_mujoco_with_motrix_defaults():
    cfg = _ppo_cfg(["task=go2_joystick_flat/mujoco"])

    assert cfg.algo.num_envs == 1024
    assert cfg.reward.tracking_lin_vel.weight == pytest.approx(1.0)
    assert cfg.reward.tracking_ang_vel.weight == pytest.approx(0.2)
    assert cfg.reward.lin_vel_z.weight == pytest.approx(-5.0)
    assert cfg.reward.ang_vel_xy.weight == pytest.approx(-0.1)
    assert cfg.algo.empirical_normalization is True
    assert cfg.algo.obs_groups.actor == ["actor"]
    assert cfg.algo.obs_groups.critic == ["critic"]
    assert cfg.algo.policy.init_noise_std == pytest.approx(0.5)
    assert cfg.algo.algorithm.learning_rate == pytest.approx(3.0e-4)
    assert cfg.algo.algorithm.entropy_coef == pytest.approx(1.0e-3)


def test_ppo_go1_drake_batch_config_matches_current_contact_support():
    cfg = _ppo_cfg(["task=go1_joystick_flat/drake"])

    assert cfg.env.drake_backend_mode == "batch"
    assert cfg.env.drake_nthread == 0
    assert cfg.reward.contact is None
    assert cfg.env.events.base_mass is None
    assert cfg.env.events.base_com is None
    assert cfg.env.events.pd_gains is None
    assert cfg.env.events.push_robot is None


def test_ppo_go2_drake_batch_config_matches_go2_training_defaults():
    cfg = _ppo_cfg(["task=go2_joystick_flat/drake"])

    assert cfg.training.task_name == "Go2JoystickFlat"
    assert cfg.training.sim_backend == "drake"
    assert cfg.algo.num_envs == 1024
    assert cfg.algo.max_iterations == 151
    assert cfg.algo.empirical_normalization is True
    assert cfg.algo.policy.init_noise_std == pytest.approx(0.5)
    assert cfg.algo.algorithm.learning_rate == pytest.approx(3.0e-4)
    assert cfg.algo.algorithm.entropy_coef == pytest.approx(1.0e-3)
    assert cfg.env.drake_backend_mode == "batch"
    assert cfg.env.drake_nthread == 0
    assert cfg.env.scene.model_file == "src/unilab/assets/robots/go2/scene_flat.xml"
    assert cfg.env.events.pd_gains is None
    assert cfg.reward.contact.weight == pytest.approx(0.24)


def test_build_ppo_env_cfg_override_go1_motrix(
    monkeypatch: pytest.MonkeyPatch,
):
    mod = _train_rsl_rl(monkeypatch)
    cfg = _ppo_cfg(["task=go1_joystick_flat/motrix"])

    env_cfg_override = mod.build_ppo_env_cfg_override(cfg)

    assert env_cfg_override["rewards"]["tracking_lin_vel"]["weight"] == pytest.approx(1.0)
    assert env_cfg_override["rewards"]["contact"] is None
    assert env_cfg_override["commands"]["twist"]["ranges"] == {
        "lin_vel_x": [0.5, 0.5],
        "lin_vel_y": [0.0, 0.0],
        "ang_vel_z": [0.0, 0.0],
    }
    assert env_cfg_override["events"]["push_robot"] is None


def test_build_ppo_env_cfg_override_g1_motrix(
    monkeypatch: pytest.MonkeyPatch,
):
    mod = _train_rsl_rl(monkeypatch)
    cfg = _ppo_cfg(["task=g1_walk_flat/motrix"])

    env_cfg_override = mod.build_ppo_env_cfg_override(cfg)

    # env_cfg_override has reward + env preset fields (flat, matching env cfg structure)
    assert env_cfg_override["rewards"]["upper_body_pose"]["weight"] == pytest.approx(-0.05)
    assert env_cfg_override["rewards"]["penalty_feet_ori"]["weight"] == pytest.approx(0.0)
    assert env_cfg_override["rewards"]["feet_phase_contrast"]["weight"] == pytest.approx(1.5)
    assert env_cfg_override["rewards"]["feet_phase_contact"]["weight"] == pytest.approx(1.0)
    assert env_cfg_override["rewards"]["feet_double_stance"]["weight"] == pytest.approx(-1.0)
    assert env_cfg_override["rewards"]["feet_phase"]["params"][
        "min_forward_speed"
    ] == pytest.approx(0.05)
    assert "motrix_max_iterations" not in env_cfg_override
    assert env_cfg_override["actions"]["joint_pos"]["scale"] == pytest.approx(0.5)
    assert env_cfg_override["commands"]["twist"]["ranges"]["lin_vel_x"] == [0.4, 0.7]
    assert env_cfg_override["events"]["pd_gains"] is None
    assert env_cfg_override["events"]["reset_root_state_uniform"]["params"]["velocity_range"][
        "x"
    ] == [-0.05, 0.05]


def test_build_ppo_env_cfg_override_carries_motrix_max_iterations_override(
    monkeypatch: pytest.MonkeyPatch,
):
    mod = _train_rsl_rl(monkeypatch)
    cfg = _ppo_cfg(["task=g1_walk_flat/motrix", "+env.motrix_max_iterations=9"])

    env_cfg_override = mod.build_ppo_env_cfg_override(cfg)

    assert env_cfg_override["motrix_max_iterations"] == 9


def test_build_ppo_env_cfg_override_carries_post_step_forward_sensor_override(
    monkeypatch: pytest.MonkeyPatch,
):
    mod = _train_rsl_rl(monkeypatch)
    for value in (True, False):
        cfg = _ppo_cfg(["task=g1_walk_flat/mujoco", f"env.post_step_forward_sensor={value}"])

        env_cfg_override = mod.build_ppo_env_cfg_override(cfg)

        assert env_cfg_override["post_step_forward_sensor"] is value


def test_offpolicy_g1_walk_flat_motrix_env_cfg_override_disables_pd_gains():
    cfg = _offpolicy_cfg(["task=g1_walk_flat/motrix"])

    env_cfg_override = _offpolicy().build_offpolicy_env_cfg_override("sac", cfg)

    assert env_cfg_override["events"]["pd_gains"] is None
    assert env_cfg_override["rewards"]["tracking_lin_vel"]["weight"] == pytest.approx(2.2)


def test_build_ppo_env_cfg_override_applies_go2_motrix_reward(
    monkeypatch: pytest.MonkeyPatch,
):
    mod = _train_rsl_rl(monkeypatch)
    cfg = _ppo_cfg(["task=go2_joystick_flat/motrix"])

    env_cfg_override = mod.build_ppo_env_cfg_override(cfg)

    assert cfg.reward.tracking_lin_vel.weight == pytest.approx(1.0)
    assert cfg.algo.num_envs == 1024
    assert env_cfg_override["events"]["pd_gains"] is None
    assert env_cfg_override["rewards"]["tracking_lin_vel"]["weight"] == pytest.approx(1.0)
    assert env_cfg_override["rewards"]["tracking_ang_vel"]["weight"] == pytest.approx(0.2)


def test_build_ppo_env_cfg_override_allegro_mujoco(
    monkeypatch: pytest.MonkeyPatch,
):
    mod = _train_rsl_rl(monkeypatch)
    cfg = _ppo_cfg(["task=allegro_inhand/mujoco"])
    ppo_motrix_cfg = _ppo_cfg(["task=allegro_inhand/motrix"])
    ppo_drake_cfg = _ppo_cfg(["task=allegro_inhand/drake"])
    appo_cfg = _appo_cfg(["task=allegro_inhand/mujoco"])
    appo_motrix_cfg = _appo_cfg(["task=allegro_inhand/motrix"])
    appo_drake_cfg = _appo_cfg(["task=allegro_inhand/drake"])

    env_cfg_override = mod.build_ppo_env_cfg_override(cfg)

    assert cfg.training.task_name == "AllegroInhandRotation"
    assert cfg.algo.empirical_normalization is False
    assert cfg.algo.actor.obs_normalization is True
    assert cfg.algo.critic.obs_normalization is True
    assert env_cfg_override["rewards"]["rotate"]["weight"] == pytest.approx(1.25)
    assert env_cfg_override["terminations"]["dropped"]["params"][
        "minimum_ball_height"
    ] == pytest.approx(0.125)
    assert env_cfg_override["max_episode_seconds"] == pytest.approx(20.0)
    reset_params = env_cfg_override["events"]["reset_hand_ball"]["params"]
    assert reset_params["grasp_cache_path"] is None
    assert reset_params["joint_noise"] == pytest.approx(0.0)
    assert reset_params["ball_velocity_noise"] == pytest.approx(0.0)
    assert reset_params["ball_z_offset"] == pytest.approx(0.0)
    assert env_cfg_override["observations"]["policy"]["history_length"] == 3
    assert env_cfg_override["actions"]["hand"]["action_scale"] == pytest.approx(1.0 / 24.0)
    assert "reward_config" not in env_cfg_override
    assert "domain_rand" not in env_cfg_override
    assert appo_cfg.algo.steps_per_env == cfg.algo.num_steps_per_env
    assert list(appo_cfg.algo.actor.hidden_dims) == list(cfg.algo.actor.hidden_dims)
    assert appo_cfg.algo.actor.activation == cfg.algo.actor.activation
    assert appo_cfg.algo.actor.obs_normalization is True
    assert list(appo_cfg.algo.critic.hidden_dims) == list(cfg.algo.critic.hidden_dims)
    assert appo_cfg.algo.critic.activation == cfg.algo.critic.activation
    assert appo_cfg.algo.critic.obs_normalization is True
    assert appo_cfg.algo.algorithm.value_loss_coef == pytest.approx(
        cfg.algo.algorithm.value_loss_coef
    )
    assert appo_cfg.algo.algorithm.entropy_coef == pytest.approx(cfg.algo.algorithm.entropy_coef)
    assert appo_cfg.algo.algorithm.num_learning_epochs == cfg.algo.algorithm.num_learning_epochs
    assert appo_cfg.algo.algorithm.num_mini_batches == cfg.algo.algorithm.num_mini_batches
    assert appo_cfg.algo.algorithm.clip_param == pytest.approx(cfg.algo.algorithm.clip_param)
    assert appo_cfg.algo.algorithm.gamma == pytest.approx(cfg.algo.algorithm.gamma)
    assert appo_cfg.algo.algorithm.lam == pytest.approx(cfg.algo.algorithm.lam)
    assert appo_cfg.algo.algorithm.max_grad_norm == pytest.approx(cfg.algo.algorithm.max_grad_norm)
    assert (
        appo_cfg.algo.algorithm.use_clipped_value_loss is cfg.algo.algorithm.use_clipped_value_loss
    )
    assert appo_cfg.algo.algorithm.schedule == cfg.algo.algorithm.schedule
    assert appo_motrix_cfg.training.task_name == appo_cfg.training.task_name
    assert appo_motrix_cfg.training.sim_backend == ppo_motrix_cfg.training.sim_backend
    assert appo_motrix_cfg.algo.actor.obs_normalization is True
    assert appo_motrix_cfg.algo.critic.obs_normalization is True
    assert appo_motrix_cfg.reward.rotate.weight == pytest.approx(
        ppo_motrix_cfg.reward.rotate.weight
    )
    assert appo_motrix_cfg.env.events.pd_gains is None
    assert ppo_motrix_cfg.env.events.pd_gains is None
    assert ppo_drake_cfg.training.sim_backend == "drake"
    assert appo_drake_cfg.training.sim_backend == "drake"
    assert ppo_drake_cfg.env.events.pd_gains is None
    assert appo_drake_cfg.env.events.pd_gains is None


def test_build_ppo_env_cfg_override_allegro_grasp_mujoco(
    monkeypatch: pytest.MonkeyPatch,
):
    mod = _train_rsl_rl(monkeypatch)
    cfg = _ppo_cfg(["task=allegro_inhand_grasp/mujoco"])

    env_cfg_override = mod.build_ppo_env_cfg_override(cfg)

    assert cfg.training.task_name == "AllegroInhandRotationGrasp"
    assert cfg.algo.empirical_normalization is False
    assert cfg.algo.actor.obs_normalization is True
    assert cfg.algo.critic.obs_normalization is True
    assert env_cfg_override["rewards"]["rotate"]["weight"] == pytest.approx(0.0)
    assert env_cfg_override["actions"]["hand"]["action_scale"] == pytest.approx(0.0)
    reset = env_cfg_override["events"]["reset_hand_ball"]["params"]
    assert reset["grasp_cache_path"] is None
    assert reset["ball_velocity_noise"] == pytest.approx(0.0)
    assert reset["joint_noise"] == pytest.approx(0.25)
    quality = env_cfg_override["terminations"]["invalid_grasp"]["params"]
    assert quality["enabled"] is True
    assert quality["minimum_contacts"] == 2
    recorder = env_cfg_override["recorders"]["grasp_cache"]["params"]
    assert recorder["collection_target"] == 50000
    assert recorder["auto_save"] is True
    assert "reward_config" not in env_cfg_override
    assert "domain_rand" not in env_cfg_override


def test_build_ppo_env_cfg_override_allegro_grasp_cli_override_wins(
    monkeypatch: pytest.MonkeyPatch,
):
    mod = _train_rsl_rl(monkeypatch)
    cfg = _ppo_cfg(
        [
            "task=allegro_inhand_grasp/mujoco",
            "algo.max_iterations=1",
            "env.recorders.grasp_cache.params.collection_target=128",
            "reward.rotate.weight=0.3",
        ]
    )

    env_cfg_override = mod.build_ppo_env_cfg_override(cfg)

    assert cfg.algo.max_iterations == 1
    assert env_cfg_override["recorders"]["grasp_cache"]["params"]["collection_target"] == 128
    assert env_cfg_override["rewards"]["rotate"]["weight"] == pytest.approx(0.3)


def test_build_ppo_env_cfg_override_sharpa_grasp_cli_override_wins(
    monkeypatch: pytest.MonkeyPatch,
):
    mod = _train_rsl_rl(monkeypatch)
    cfg = _ppo_cfg(
        [
            "task=sharpa_inhand_grasp/mujoco",
            "algo.max_iterations=1",
            "env.grasp_collection_target=128",
            "reward.scales.rotate=0.3",
        ]
    )

    env_cfg_override = mod.build_ppo_env_cfg_override(cfg)

    assert cfg.algo.max_iterations == 1
    assert env_cfg_override["grasp_collection_target"] == 128
    assert env_cfg_override["reward_config"]["scales"]["rotate"] == pytest.approx(0.3)


def test_build_ppo_env_cfg_override_sharpa_grasp_motrix_owner(
    monkeypatch: pytest.MonkeyPatch,
):
    mod = _train_rsl_rl(monkeypatch)
    cfg = _ppo_cfg(
        [
            "task=sharpa_inhand_grasp/motrix",
            "algo.max_iterations=1",
            "env.grasp_collection_target=128",
        ]
    )

    env_cfg_override = mod.build_ppo_env_cfg_override(cfg)

    assert cfg.training.task_name == "SharpaInhandRotationGrasp"
    assert cfg.training.sim_backend == "motrix"
    assert env_cfg_override["grasp_collection_target"] == 128
    assert env_cfg_override["domain_rand"]["scale_list"] == [0.8]


@pytest.mark.parametrize("std_type", ["scalar", "vector"])
def test_rsl_action_std_logging_patch_delegates_with_detached_clone(std_type: str):
    import torch

    from unilab.training.experiment import patch_rsl_rl_action_std_logging

    captured: dict[str, Any] = {}
    expected = torch.tensor([0.25, 0.5], requires_grad=True)
    distribution = types.SimpleNamespace(
        std_type=std_type,
        std_param=expected,
        log_std_param=torch.log(expected),
    )

    class Logger:
        def log(self, *args, **kwargs):
            captured["call"] = (args, kwargs)
            return "logged"

    runner = types.SimpleNamespace(
        logger=Logger(),
        alg=types.SimpleNamespace(
            get_policy=lambda: types.SimpleNamespace(distribution=distribution)
        ),
    )

    patch_rsl_rl_action_std_logging(runner)
    result = runner.logger.log("payload", iteration=3)

    assert result == "logged"
    args, kwargs = captured["call"]
    assert args == ("payload",)
    assert kwargs["iteration"] == 3
    action_std = kwargs["action_std"]
    torch.testing.assert_close(action_std, expected)
    assert action_std.requires_grad is False
    assert action_std.data_ptr() != expected.data_ptr()
    source = (_SCRIPTS_DIR / "train_rsl_rl.py").read_text(encoding="utf-8")
    assert "def _patch_runner_action_std_logging" not in source


def _build_rsl_lifecycle_case(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    learn_exception: Exception | None,
    *,
    run_complete: bool = False,
    close_exception: Exception | None = None,
    play_only: bool = False,
) -> tuple[Any, Any, dict[str, Any], Exception | None]:
    mod = _train_rsl_rl(monkeypatch)
    if run_complete:
        assert learn_exception is None
        learn_exception = mod.RunComplete(
            reason="grasp_collection_target_reached",
            summary={"collected_grasps": 12, "status": "payload_must_not_override"},
        )
    cfg = _ppo_cfg(
        [
            "task=sharpa_inhand_grasp/mujoco",
            f"training.log_dir={tmp_path}",
            "training.logger=none",
            "training.nan_guard.enabled=false",
            "training.no_play=false",
            "training.play_render_mode=record",
            f"training.play_only={str(play_only).lower()}",
        ]
    )
    captured: dict[str, Any] = {
        "env_create": 0,
        "env_close": 0,
        "distributed": [],
        "summaries": [],
        "tracker_finish": 0,
        "playback": 0,
    }

    class FakeEnv:
        num_envs = 1
        play_capabilities = types.SimpleNamespace(supports_physics_state_playback=False)

        def close(self) -> None:
            captured["env_close"] += 1
            if close_exception is not None:
                raise close_exception

    class FakeWrapper:
        def __init__(self, env: Any, device: str) -> None:
            del env, device

    class FakeRunner:
        logger = types.SimpleNamespace(
            tot_timesteps=0,
            tot_time=0.0,
            rewbuffer=[],
            lenbuffer=[],
        )
        current_learning_iteration = 3

        def __init__(self, wrapped_env: Any, train_cfg: dict[str, Any], log_dir: str, device: str):
            del wrapped_env, train_cfg, log_dir, device

        def learn(self, **kwargs: Any) -> None:
            del kwargs
            if learn_exception is not None:
                raise learn_exception
            self.logger.tot_timesteps = 8
            self.logger.tot_time = 2.0

    class FakeTracker:
        def __init__(self, **kwargs: Any) -> None:
            del kwargs

        def start(self) -> None:
            captured["tracker_start"] = True

        def update_summary(self, summary: dict[str, Any]) -> None:
            captured["summaries"].append(summary)

        def finish(self) -> None:
            captured["tracker_finish"] += 1

        def log_video(self, path: str | None) -> None:
            captured["video"] = path

    monkeypatch.setattr(mod, "resolve_dp_topology", lambda _devices: None)
    monkeypatch.setattr(mod, "current_torch_distributed_rank", lambda: 0)
    monkeypatch.setattr(mod, "current_torch_distributed_local_rank", lambda: 0)
    monkeypatch.setattr(mod, "current_torch_distributed_world_size", lambda: 1)
    monkeypatch.setattr(mod, "ensure_registries", lambda: None)
    monkeypatch.setattr(mod, "apply_rsl_rl_rank_seed", lambda _cfg, _rank: 0)
    monkeypatch.setattr(mod, "apply_configured_training_seed", lambda *args, **kwargs: {})
    monkeypatch.setattr(mod, "build_ppo_env_cfg_override", lambda _cfg: {})
    monkeypatch.setattr(mod, "get_default_device", lambda: "cpu")
    monkeypatch.setattr(mod, "resolve_rsl_rl_device", lambda **kwargs: "cpu")
    monkeypatch.setattr(mod, "resolve_ppo_log_dir", lambda _cfg, world_size: str(tmp_path))
    monkeypatch.setattr(mod, "ExperimentTracker", FakeTracker)

    def create_env(*args: Any, **kwargs: Any) -> FakeEnv:
        del args, kwargs
        captured["env_create"] += 1
        return FakeEnv()

    monkeypatch.setattr(mod, "create_env", create_env)
    monkeypatch.setattr(mod, "algo_config_dict", lambda _cfg: {})
    monkeypatch.setattr(mod, "_resolve_ppo_wrapper_cls", lambda _rl_cfg: FakeWrapper)
    monkeypatch.setattr(mod, "normalize_ppo_train_cfg", lambda _rl_cfg: {})
    monkeypatch.setattr(mod, "patch_rsl_rl_resume_state", lambda: None)
    monkeypatch.setattr(mod, "OnPolicyRunner", FakeRunner)
    monkeypatch.setattr(mod, "patch_rsl_rl_action_std_logging", lambda _runner: None)
    monkeypatch.setattr(
        mod,
        "finish_rsl_rl_distributed",
        lambda *, training_succeeded: captured["distributed"].append(training_succeeded),
    )

    def playback(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        captured["playback"] += 1

    monkeypatch.setattr(mod, "play_rsl_rl", playback)
    return mod, cfg, captured, learn_exception


def test_train_rsl_rl_run_complete_closes_resources_and_skips_playback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    mod, cfg, captured, _ = _build_rsl_lifecycle_case(
        monkeypatch,
        tmp_path,
        None,
        run_complete=True,
    )

    assert mod.main.__wrapped__(cfg) is None
    assert captured["env_close"] == 1
    assert captured["distributed"] == [True]
    assert captured["tracker_finish"] == 1
    assert captured["playback"] == 0
    assert captured["summaries"] == [
        {
            "collected_grasps": 12,
            "status": "collection_completed",
            "completion_reason": "grasp_collection_target_reached",
        }
    ]


def test_train_rsl_rl_success_keeps_playback_and_single_cleanup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    mod, cfg, captured, raised = _build_rsl_lifecycle_case(monkeypatch, tmp_path, None)

    assert raised is None
    assert mod.main.__wrapped__(cfg) is None
    assert captured["env_close"] == 1
    assert captured["distributed"] == [True]
    assert captured["tracker_finish"] == 1
    assert captured["playback"] == 1
    assert captured["summaries"][0]["status"] == "completed"
    assert captured["summaries"][0]["run_env_steps"] == 8


def test_train_rsl_rl_runtime_error_propagates_after_single_cleanup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    sentinel = RuntimeError("runner failed")
    mod, cfg, captured, raised = _build_rsl_lifecycle_case(monkeypatch, tmp_path, sentinel)

    with pytest.raises(RuntimeError) as caught:
        mod.main.__wrapped__(cfg)

    assert caught.value is raised is sentinel
    assert captured["env_close"] == 1
    assert captured["distributed"] == [False]
    assert captured["tracker_finish"] == 1
    assert captured["playback"] == 0
    assert captured["summaries"] == []


def test_train_rsl_rl_run_complete_close_error_is_not_misclassified(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    sentinel = RuntimeError("close failed")
    mod, cfg, captured, _ = _build_rsl_lifecycle_case(
        monkeypatch,
        tmp_path,
        None,
        run_complete=True,
        close_exception=sentinel,
    )

    with pytest.raises(RuntimeError) as caught:
        mod.main.__wrapped__(cfg)

    assert caught.value is sentinel
    assert captured["env_close"] == 1
    assert captured["distributed"] == [False]
    assert captured["tracker_finish"] == 1
    assert captured["playback"] == 0
    assert captured["summaries"] == []


def test_train_rsl_rl_play_only_keeps_single_cleanup_and_runs_playback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    mod, cfg, captured, raised = _build_rsl_lifecycle_case(
        monkeypatch,
        tmp_path,
        None,
        play_only=True,
    )

    assert raised is None
    assert mod.main.__wrapped__(cfg) is None
    assert captured["env_create"] == 0
    assert captured["env_close"] == 0
    assert captured["distributed"] == [False]
    assert captured["tracker_finish"] == 0
    assert captured["playback"] == 1
    assert captured["summaries"] == []


@pytest.mark.parametrize(
    ("devices", "world_size"),
    [
        ((0, 1), 1),
        (None, 2),
    ],
)
def test_train_rsl_rl_grasp_collection_rejects_multi_rank_before_launch(
    monkeypatch: pytest.MonkeyPatch,
    devices: tuple[int, ...] | None,
    world_size: int,
) -> None:
    mod = _train_rsl_rl(monkeypatch)
    cfg = _ppo_cfg(["task=sharpa_inhand_grasp/mujoco"])
    monkeypatch.setattr(mod, "resolve_dp_topology", lambda _devices: devices)
    monkeypatch.setattr(mod, "current_torch_distributed_rank", lambda: 0)
    monkeypatch.setattr(mod, "current_torch_distributed_local_rank", lambda: 0)
    monkeypatch.setattr(mod, "current_torch_distributed_world_size", lambda: world_size)
    monkeypatch.setattr(
        mod,
        "launch_torchrun_workers",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must fail before launch")),
    )
    monkeypatch.setattr(
        mod,
        "ensure_registries",
        lambda: (_ for _ in ()).throw(AssertionError("must fail before registry bootstrap")),
    )

    with pytest.raises(ValueError, match="requires one process"):
        mod.main.__wrapped__(cfg)


def test_ppo_cli_algo_override_wins_over_base(
    monkeypatch: pytest.MonkeyPatch,
):
    """CLI override takes precedence over base task algo values via Hydra compose."""
    cfg = _ppo_cfg(["task=g1_walk_flat/motrix", "algo.max_iterations=1"])

    assert cfg.algo.max_iterations == 1
    # Other base values remain intact
    assert cfg.algo.empirical_normalization is True


def test_g1_motion_tracking_ppo_motrix_prefers_backend_specific_reward(
    monkeypatch: pytest.MonkeyPatch,
):
    mod = _train_rsl_rl(monkeypatch)
    cfg = _ppo_cfg(["task=g1_motion_tracking/motrix"])

    assert cfg.reward.motion_body_pos.weight == pytest.approx(1.0)
    cfg.reward.motion_body_pos.weight = 1.25

    env_cfg_override = mod.build_ppo_env_cfg_override(cfg)

    assert env_cfg_override["rewards"]["motion_body_pos"]["weight"] == pytest.approx(1.25)


def test_build_ppo_play_env_cfg_override_applies_g1_motion_tracking_play_profile(
    monkeypatch: pytest.MonkeyPatch,
):
    mod = _train_rsl_rl(monkeypatch)
    cfg = _ppo_cfg(["task=g1_motion_tracking/motrix", "training.play_only=true"])
    assert cfg.training.play_env_num == 16

    monkeypatch.setattr(
        mod,
        "materialize_scene_visual_override",
        lambda *_args, **_kwargs: pytest.fail("manager scene must not be replaced"),
    )

    env_cfg_override = mod.build_ppo_play_env_cfg_override(cfg)

    assert cfg.training.play_env_num == 16
    assert env_cfg_override["render_spacing"] == pytest.approx(2.5)
    assert env_cfg_override["scene"]["model_file"].endswith("robots/g1/scene_flat.xml")
    assert "robot" in env_cfg_override["scene"]["entities"]
    assert env_cfg_override["rewards"]["motion_body_pos"]["weight"] == pytest.approx(1.0)


def test_build_ppo_play_env_cfg_override_respects_cli_play_env_override(
    monkeypatch: pytest.MonkeyPatch,
):
    mod = _train_rsl_rl(monkeypatch)
    cfg = _ppo_cfg(
        [
            "task=g1_motion_tracking/motrix",
            "training.play_only=true",
            "training.play_env_num=32",
        ]
    )
    assert cfg.training.play_env_num == 32
    monkeypatch.setattr(
        mod,
        "materialize_scene_visual_override",
        lambda source_model_file, **kwargs: "/tmp/g1_motion_tracking_play_scene.xml",
    )

    env_cfg_override = mod.build_ppo_play_env_cfg_override(cfg)

    assert cfg.training.play_env_num == 32
    assert env_cfg_override["render_spacing"] == pytest.approx(2.5)


def test_build_ppo_play_env_cfg_override_keeps_task_owned_manager_scene(
    monkeypatch: pytest.MonkeyPatch,
):
    mod = _train_rsl_rl(monkeypatch)
    cfg = _ppo_cfg(["task=g1_motion_tracking/motrix", "training.play_only=true"])
    monkeypatch.setattr(
        mod,
        "materialize_scene_visual_override",
        lambda *_args, **_kwargs: pytest.fail("manager scene must not be replaced"),
    )

    env_cfg_override = mod.build_ppo_play_env_cfg_override(cfg)

    assert env_cfg_override["scene"]["entities"]["robot"]["root_body_name"] == "pelvis"
    assert env_cfg_override["scene"]["entities"]["robot"]["joint_names"]


def test_go2_arm_manip_loco_motrix_eval_uses_visual_floor(
    monkeypatch: pytest.MonkeyPatch,
):
    mod = _train_rsl_rl(monkeypatch)
    cfg = _ppo_cfg(["task=go2_arm_manip_loco/motrix", "training.play_only=true"])

    captured = {}

    def _fake_materialize(source_model_file, **kwargs):
        captured["source_model_file"] = source_model_file
        captured.update(kwargs)
        return "/tmp/go2_arm_manip_loco_play_scene.xml"

    monkeypatch.setattr(mod, "materialize_scene_visual_override", _fake_materialize)

    env_cfg_override = mod.build_ppo_play_env_cfg_override(cfg)

    assert captured["source_model_file"] == str(
        mod.ROOT_DIR / "src/unilab/assets/robots/go2_arm/scene_flat.xml"
    )
    assert captured["ground_texture_file"] == str(
        mod.ROOT_DIR / "src/unilab/assets/robots/g1/textures/floor.png"
    )
    assert captured["skybox_rgb1"] == [0.90, 0.90, 0.91]
    assert captured["skybox_rgb2"] == [0.68, 0.68, 0.70]
    assert captured["ground_texrepeat"] == [0.25, 0.25]
    assert env_cfg_override["scene"].model_file == "/tmp/go2_arm_manip_loco_play_scene.xml"


def test_run_motrix_rsl_play_loop_uses_render_spacing_and_offset_mode(
    monkeypatch: pytest.MonkeyPatch,
):
    import numpy as np
    import torch
    from tensordict import TensorDict

    mod = _train_rsl_rl(monkeypatch)

    class FakePolicy:
        def __call__(self, obs):
            batch = obs.batch_size[0]
            return torch.zeros((batch, 3), dtype=torch.float32)

    class FakeBackend:
        def __init__(self):
            self.init_renderer_calls = []
            self.render_calls = 0

        def init_renderer(self, spacing=1.0, offset_mode="grid"):
            self.init_renderer_calls.append((spacing, offset_mode))

        def render(self):
            self.render_calls += 1

    class FakeEnv:
        def __init__(self):
            self._renderer = FakeBackend()
            self.cfg = type("Cfg", (), {"render_spacing": 2.5, "render_offset_mode": "zero"})()

        def init_play_renderer(self, render_spacing=None, render_offset_mode=None):
            offset_mode = "grid" if render_offset_mode is None else render_offset_mode
            if render_spacing is None:
                self._renderer.init_renderer(offset_mode=offset_mode)
            else:
                self._renderer.init_renderer(render_spacing, offset_mode=offset_mode)

        def render_play_frame(self):
            self._renderer.render()

        def run_playback(self, **kwargs):
            kwargs.pop("frame_state_getter", None)
            kwargs.setdefault("output_video", None)
            kwargs.setdefault("camera_kwargs", None)
            return run_motrix_playback(
                backend=self._renderer,
                env=self,
                headless=False if kwargs.get("headless") is None else bool(kwargs["headless"]),
                record_video=(
                    bool(kwargs["record_video"])
                    if kwargs.get("record_video") is not None
                    else kwargs.get("output_video") is not None
                ),
                **{k: v for k, v in kwargs.items() if k not in {"headless", "record_video"}},
            )

    class FakeWrapper:
        def __init__(self):
            self.env = FakeEnv()
            self.reset_calls = 0
            self.step_calls = 0

        def reset(self):
            self.reset_calls += 1
            return TensorDict({"policy": torch.ones((2, 5), dtype=torch.float32)}, batch_size=2), {}

        def step(self, actions):
            self.step_calls += 1
            return (
                TensorDict({"policy": torch.ones((2, 5), dtype=torch.float32)}, batch_size=2),
                torch.zeros((2,), dtype=torch.float32),
                torch.zeros((2,), dtype=torch.bool),
                {},
            )

    wrapped_env = FakeWrapper()

    mod.run_motrix_rsl_play_loop(
        wrapped_env=wrapped_env,
        policy=FakePolicy(),
        render_spacing=2.5,
        render_offset_mode="zero",
        num_steps=3,
    )

    assert wrapped_env.reset_calls == 1
    assert wrapped_env.step_calls == 3
    assert wrapped_env.env._renderer.init_renderer_calls == [(2.5, "zero")]
    assert wrapped_env.env._renderer.render_calls == 3


def test_g1_motion_tracking_appo_reward_extraction_prefers_backend_specific_reward():
    from unilab.training import BackendAdapter

    cfg = _appo_cfg(["task=g1_motion_tracking/motrix"])

    assert cfg.reward.motion_body_pos.weight == pytest.approx(1.0)
    cfg.reward.motion_body_pos.weight = 1.5

    env_cfg_override = BackendAdapter(cfg, root_dir=_SRC_DIR.parent).build_task_env_cfg_override()

    assert env_cfg_override["rewards"]["motion_body_pos"]["weight"] == pytest.approx(1.5)


def test_g1_motion_tracking_ppo_task_exposes_final_reward():
    cfg = _ppo_cfg(["task=g1_motion_tracking/motrix"])

    assert cfg.reward.motion_body_pos.weight == pytest.approx(1.0)


def test_g1_motion_tracking_appo_task_exposes_final_reward():
    cfg = _appo_cfg(["task=g1_motion_tracking/motrix"])

    assert cfg.reward.motion_body_pos.weight == pytest.approx(1.0)


def test_sharpa_appo_motrix_owner_uses_backend_specific_overrides():
    cfg = _appo_cfg(["task=sharpa_inhand/motrix"])

    assert cfg.training.task_name == "SharpaInhandRotation"
    assert cfg.training.sim_backend == "motrix"
    assert cfg.algo.num_envs == 2048
    assert cfg.env.sim_dt == pytest.approx(0.01)
    assert cfg.env.domain_rand.randomize_gravity is True
    assert cfg.env.domain_rand.randomize_gravity_direction is False
    assert cfg.env.domain_rand.randomize_pd_gains is True


# ---------------------------------------------------------------------------
# train_appo.py — motrix runner / play helpers
# ---------------------------------------------------------------------------


def test_build_appo_runner_kwargs_forwards_sim_backend():
    mod = _train_appo()
    cfg = _appo_cfg(["task=g1_motion_tracking/motrix"])

    runner_kwargs = mod.build_appo_runner_kwargs(
        cfg,
        env_cfg_override={"rewards": {}},
        collector_device="cpu",
    )

    assert runner_kwargs["env_name"] == "G1MotionTracking"
    assert runner_kwargs["sim_backend"] == "motrix"
    assert runner_kwargs["collector_device"] == "cpu"
    assert runner_kwargs["num_envs"] == cfg.algo.num_envs
    assert runner_kwargs["steps_per_env"] == cfg.algo.steps_per_env
    assert "num_workers" not in runner_kwargs
    assert "num_collectors" not in runner_kwargs
    assert runner_kwargs["env_cfg_overrides"]["rewards"] == {}


def test_run_motrix_play_loop_runs_without_physics_state():
    import numpy as np
    import torch

    mod = _train_appo()

    class FakeActor:
        def __call__(self, td):
            batch = td.batch_size[0]
            return torch.zeros((batch, 3), dtype=torch.float32)

    class FakeBackend:
        def __init__(self):
            self.init_renderer_calls = 0
            self.render_calls = 0

        def init_renderer(self, spacing=1.0, offset_mode="grid", **kwargs):
            del spacing, offset_mode, kwargs
            self.init_renderer_calls += 1

        def render(self):
            self.render_calls += 1

    class FakeState:
        def __init__(self):
            self.obs = {"obs": np.ones((2, 5), dtype=np.float32)}

    class FakeEnv:
        def __init__(self):
            self.state = None
            self._renderer = FakeBackend()
            self.init_state_calls = 0
            self.reset_calls = 0
            self.step_calls = 0

        def init_state(self):
            self.init_state_calls += 1
            self.state = object()

        def reset(self, env_indices):
            self.reset_calls += 1
            assert env_indices.shape == (2,)
            return {"obs": np.ones((2, 5), dtype=np.float32)}, {}

        def step(self, actions):
            self.step_calls += 1
            assert actions.shape == (2, 3)
            return FakeState()

        def init_play_renderer(self, render_spacing=None, render_offset_mode=None):
            del render_spacing, render_offset_mode
            self._renderer.init_renderer()

        def render_play_frame(self):
            self._renderer.render()

        def run_playback(self, **kwargs):
            kwargs.pop("frame_state_getter", None)
            kwargs.setdefault("output_video", None)
            kwargs.setdefault("render_spacing", None)
            kwargs.setdefault("render_offset_mode", None)
            kwargs.setdefault("camera_kwargs", None)
            return run_motrix_playback(
                backend=self._renderer,
                env=self,
                headless=False if kwargs.get("headless") is None else bool(kwargs["headless"]),
                record_video=(
                    bool(kwargs["record_video"])
                    if kwargs.get("record_video") is not None
                    else kwargs.get("output_video") is not None
                ),
                **{k: v for k, v in kwargs.items() if k not in {"headless", "record_video"}},
            )

    env = FakeEnv()

    mod.run_motrix_play_loop(
        env=env,
        actor=FakeActor(),
        device="cpu",
        play_env_num=2,
        num_steps=3,
    )

    assert env.init_state_calls == 1
    assert env.reset_calls == 1
    assert env.step_calls == 3
    assert env._renderer.init_renderer_calls == 1
    assert env._renderer.render_calls == 3


def test_resolve_appo_checkpoint_path_prefers_latest_model_in_explicit_dir(tmp_path):
    mod = _train_appo()
    run_dir = tmp_path / "logs" / "appo" / "MyTask" / "run1"
    run_dir.mkdir(parents=True)
    (run_dir / "model_1.pt").write_bytes(b"")
    (run_dir / "model_7.pt").write_bytes(b"")

    checkpoint_path, checkpoint_dir = mod.resolve_appo_checkpoint_path(
        base_log_dir=tmp_path / "logs" / "appo" / "MyTask",
        load_run=str(run_dir),
    )

    assert checkpoint_path is not None
    assert checkpoint_path.endswith("model_7.pt")
    assert checkpoint_dir == str(run_dir)


# ---------------------------------------------------------------------------
# train_offpolicy.py — default_device()
# ---------------------------------------------------------------------------


def _offpolicy():
    return _load_script("train_offpolicy")


def test_offpolicy_default_device_preferred_cpu():
    mock_torch = MagicMock()
    assert _offpolicy().default_device(mock_torch, preferred="cpu") == "cpu"


def test_offpolicy_default_device_preferred_cuda():
    mock_torch = MagicMock()
    assert _offpolicy().default_device(mock_torch, preferred="cuda") == "cuda"


def test_offpolicy_default_device_cuda_available():
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = True
    assert _offpolicy().default_device(mock_torch) == "cuda"


def test_offpolicy_default_device_mps_fallback():
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False
    mock_torch.xpu.is_available.return_value = False
    mock_torch.backends.mps.is_available.return_value = True
    assert _offpolicy().default_device(mock_torch) == "mps"


def test_offpolicy_default_device_xpu_before_mps():
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False
    mock_torch.xpu.is_available.return_value = True
    mock_torch.backends.mps.is_available.return_value = True
    assert _offpolicy().default_device(mock_torch) == "xpu"


def test_offpolicy_default_device_cpu_fallback():
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False
    mock_torch.xpu.is_available.return_value = False
    mock_torch.backends.mps.is_available.return_value = False
    assert _offpolicy().default_device(mock_torch) == "cpu"


def test_offpolicy_enable_faulthandler_respects_disable_env(monkeypatch: pytest.MonkeyPatch):
    mod = _offpolicy()
    fake_faulthandler = types.SimpleNamespace(
        is_enabled=lambda: False,
        enable=MagicMock(),
    )
    monkeypatch.setitem(sys.modules, "faulthandler", fake_faulthandler)
    monkeypatch.setenv("UNILAB_FAULTHANDLER", "0")

    mod.enable_faulthandler()

    fake_faulthandler.enable.assert_not_called()


def test_offpolicy_enable_faulthandler_default_enables(monkeypatch: pytest.MonkeyPatch):
    mod = _offpolicy()
    fake_faulthandler = types.SimpleNamespace(
        is_enabled=lambda: False,
        enable=MagicMock(),
    )
    monkeypatch.setitem(sys.modules, "faulthandler", fake_faulthandler)
    monkeypatch.delenv("UNILAB_FAULTHANDLER", raising=False)

    mod.enable_faulthandler()

    fake_faulthandler.enable.assert_called_once_with(all_threads=True)


def test_offpolicy_build_failure_summary_preserves_failed_status():
    mod = _offpolicy()
    exc = RuntimeError("collector died")

    summary = mod.build_failure_summary(exc, {"status": "collector_died", "total_env_steps": 12})

    assert summary["status"] == "collector_died"
    assert summary["total_env_steps"] == 12
    assert summary["error_type"] == "RuntimeError"
    assert summary["error"] == "collector died"


def test_offpolicy_build_run_dir_name_uses_timestamp_and_backend():
    mod = _offpolicy()

    assert mod.build_run_dir_name("2026-06-22_22-31-24", "mujoco") == ("2026-06-22_22-31-24_mujoco")
    assert (
        mod.build_run_dir_name("2026-06-22_22-31-24", "mujoco", world_size=2)
        == "2026-06-22_22-31-24_mujoco_gpux2"
    )


def test_offpolicy_main_failure_summary_and_skips_playback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    mod = _offpolicy()
    cfg = _offpolicy_cfg(
        [
            f"training.log_dir={tmp_path}",
            "training.no_play=false",
            "training.play_render_mode=record",
        ]
    )
    captured: dict[str, Any] = {"summaries": []}

    class FakeTracker:
        def __init__(self, **kwargs):
            captured["tracker_kwargs"] = kwargs

        def start(self):
            captured["tracker_started"] = True

        def update_summary(self, summary):
            captured["summaries"].append(summary)

        def log_video(self, path):
            captured["video"] = path

        def finish(self):
            captured["tracker_finished"] = True

    class FakeRunner:
        last_run_summary = {"status": "collector_died", "total_env_steps": 12}

        def learn(self, **kwargs):
            del kwargs
            raise RuntimeError("collector died")

        def close(self):
            captured["runner_closed"] = True

    monkeypatch.setattr(mod, "enable_faulthandler", lambda: None)
    monkeypatch.setattr(mod, "ensure_registries", lambda: None)
    monkeypatch.setattr(mod, "apply_configured_training_seed", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        mod, "assert_offpolicy_task_choice_matches_algo", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(mod, "ExperimentTracker", FakeTracker)
    monkeypatch.setattr(mod, "build_runner", lambda algo_name, cfg, log_dir=None: FakeRunner())
    monkeypatch.setattr(
        mod,
        "play_offpolicy",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("playback must not run after training failure")
        ),
    )

    with pytest.raises(RuntimeError, match="collector died"):
        mod.main(cfg)

    assert captured["tracker_started"] is True
    assert captured["tracker_finished"] is True
    assert captured["runner_closed"] is True
    assert len(captured["summaries"]) == 1
    failure_summary = captured["summaries"][0]
    assert failure_summary["status"] == "collector_died"
    assert failure_summary["total_env_steps"] == 12
    assert failure_summary["error_type"] == "RuntimeError"
    assert failure_summary["error"] == "collector died"
    assert "video" not in captured


# ---------------------------------------------------------------------------
# train_offpolicy.py — resolve_checkpoint_path()
# ---------------------------------------------------------------------------


def test_resolve_checkpoint_no_base_dir(tmp_path):
    """load_run='-1' with no log directory → (None, None)."""
    path, path_dir = _offpolicy().resolve_checkpoint_path(tmp_path, "sac", "MyTask", "-1")
    assert path is None
    assert path_dir is None


def test_resolve_checkpoint_explicit_existing_file(tmp_path):
    """load_run = absolute path to existing .pt → returns that path."""
    model_file = tmp_path / "model_100.pt"
    model_file.write_bytes(b"")
    path, path_dir = _offpolicy().resolve_checkpoint_path(
        tmp_path, "sac", "MyTask", str(model_file)
    )
    assert path == str(model_file)
    assert path_dir == str(tmp_path)


def test_resolve_checkpoint_latest_picks_highest_iter(tmp_path):
    """load_run='-1' picks model with numerically highest iteration."""
    task_dir = tmp_path / "logs" / "sac" / "MyTask" / "run1"
    task_dir.mkdir(parents=True)
    (task_dir / "model_10.pt").write_bytes(b"")
    (task_dir / "model_50.pt").write_bytes(b"")
    (task_dir / "model_100.pt").write_bytes(b"")

    path, path_dir = _offpolicy().resolve_checkpoint_path(tmp_path, "sac", "MyTask", "-1")
    assert path is not None
    assert "model_100.pt" in path


def test_resolve_checkpoint_accepts_integer_latest_run(tmp_path):
    """load_run=-1 from Hydra CLI picks the latest model."""
    task_dir = tmp_path / "logs" / "sac" / "MyTask" / "run1"
    task_dir.mkdir(parents=True)
    (task_dir / "model_10.pt").write_bytes(b"")
    (task_dir / "model_50.pt").write_bytes(b"")

    path, path_dir = _offpolicy().resolve_checkpoint_path(tmp_path, "sac", "MyTask", -1)

    assert path is not None
    assert "model_50.pt" in path
    assert path_dir == str(task_dir)


def test_resolve_checkpoint_explicit_run_name(tmp_path):
    """load_run = run-directory name under the log root."""
    task_dir = tmp_path / "logs" / "sac" / "MyTask" / "myrun"
    task_dir.mkdir(parents=True)
    (task_dir / "model_5.pt").write_bytes(b"")

    path, path_dir = _offpolicy().resolve_checkpoint_path(tmp_path, "sac", "MyTask", "myrun")
    assert path is not None
    assert "model_5.pt" in path
    assert path_dir == str(task_dir)


def test_resolve_checkpoint_nonexistent_explicit_path(tmp_path):
    """load_run points to a path that doesn't exist → (None, None)."""
    path, path_dir = _offpolicy().resolve_checkpoint_path(
        tmp_path, "sac", "MyTask", "/nonexistent/model.pt"
    )
    assert path is None
    assert path_dir is None


def test_resolve_checkpoint_empty_run_dir(tmp_path):
    """Run directory exists but has no model_*.pt → (None, None)."""
    task_dir = tmp_path / "logs" / "sac" / "MyTask" / "run1"
    task_dir.mkdir(parents=True)

    path, path_dir = _offpolicy().resolve_checkpoint_path(tmp_path, "sac", "MyTask", "-1")
    assert path is None


def test_offpolicy_extract_reset_obs_handles_two_tuple():
    obs = {"obs": "value"}

    result = _offpolicy().extract_reset_obs((obs, {"info": 1}))

    assert result is obs


def test_offpolicy_extract_reset_obs_rejects_three_tuple():
    obs = {"obs": "value"}

    with pytest.raises(ValueError, match="Unexpected env.reset return format"):
        _offpolicy().extract_reset_obs(("ignored", obs, {"info": 1}))


def test_offpolicy_resolve_play_obs_dim_ignores_critic():
    obs_dim = _offpolicy().resolve_play_obs_dim({"obs": 98, "critic": 101})

    assert obs_dim == 98


def test_offpolicy_extract_play_obs_uses_obs_group_only():
    import numpy as np

    obs = {
        "obs": np.ones((2, 98), dtype=np.float32),
        "critic": np.full((2, 101), 2.0, dtype=np.float32),
    }

    play_obs = _offpolicy().extract_play_obs(obs)

    assert play_obs.shape == (2, 98)
    assert np.allclose(play_obs, 1.0)


def test_offpolicy_play_actor_spec_uses_hora_sac_runtime():
    from unilab.training.offpolicy import resolve_play_actor_spec

    cfg = _offpolicy_cfg(
        [
            "task=sharpa_inhand/mujoco_hora",
        ]
    )

    actor_algo_type, actor_kwargs = resolve_play_actor_spec(
        "sac",
        cfg,
        obs_dim=4,
        critic_obs_dim=6,
    )

    assert actor_algo_type == "hora_sac"
    assert actor_kwargs["priv_info_dim"] == 2


def test_offpolicy_play_actor_spec_keeps_standard_sac_and_flashsac():
    from unilab.training.offpolicy import resolve_play_actor_spec

    sac_cfg = _offpolicy_cfg(["task=g1_walk_flat/mujoco"])
    flashsac_cfg = _offpolicy_cfg(["task=g1_walk_flat/mujoco"], algo="flashsac")

    sac_algo_type, sac_kwargs = resolve_play_actor_spec(
        "sac",
        sac_cfg,
        obs_dim=98,
        critic_obs_dim=101,
    )
    flash_algo_type, flash_kwargs = resolve_play_actor_spec(
        "flashsac",
        flashsac_cfg,
        obs_dim=98,
        critic_obs_dim=101,
    )

    assert (sac_algo_type, sac_kwargs) == ("sac", {})
    assert (flash_algo_type, flash_kwargs) == ("flashsac", {})


def test_offpolicy_build_play_actor_preserves_flashsac_model_kwargs(
    monkeypatch: pytest.MonkeyPatch,
):
    import unilab.algos.common.actor_factory as actor_factory
    import unilab.algos.common.normalization as normalization
    from unilab.training.offpolicy import build_play_actor

    captured: dict[str, Any] = {}

    class FakeActor:
        def eval(self):
            captured["actor_eval"] = True

    class FakeNormalizer:
        def __init__(self, *args, **kwargs):
            captured["normalizer_init"] = (args, kwargs)

    def fake_build_actor(*args, **kwargs):
        captured["actor_init"] = (args, kwargs)
        return FakeActor()

    monkeypatch.setattr(actor_factory, "build_actor", fake_build_actor)
    monkeypatch.setattr(normalization, "EmpiricalNormalization", FakeNormalizer)
    cfg = _offpolicy_cfg(["algo.obs_normalization=true"], algo="flashsac")

    actor, normalizer, actor_algo_type, actor_kwargs = build_play_actor(
        "flashsac",
        cfg,
        obs_dim=98,
        critic_obs_dim=101,
        action_dim=12,
        device="cpu",
    )

    assert isinstance(actor, FakeActor)
    assert isinstance(normalizer, FakeNormalizer)
    assert (actor_algo_type, actor_kwargs) == ("flashsac", {})
    args, kwargs = captured["actor_init"]
    assert args == (
        "flashsac",
        98,
        12,
        cfg.algo.actor_hidden_dim,
        cfg.algo.use_layer_norm,
        "cpu",
    )
    assert kwargs == {
        "actor_num_blocks": cfg.algo.algo_params.actor_num_blocks,
        "actor_noise_zeta_mu": cfg.algo.algo_params.actor_noise_zeta_mu,
        "actor_noise_zeta_max": cfg.algo.algo_params.actor_noise_zeta_max,
    }
    assert captured["normalizer_init"] == ((), {"shape": 98, "device": "cpu"})
    assert captured["actor_eval"] is True


def test_offpolicy_build_play_actor_restores_td3_state_and_normalizer(
    monkeypatch: pytest.MonkeyPatch,
):
    import torch

    import unilab.algos.fast_td3.learner as learner_module
    from unilab.training.offpolicy import build_play_actor, load_play_actor

    captured: dict[str, Any] = {}

    class FakeActor:
        def __init__(self, *args, **kwargs):
            captured["actor_init"] = (args, kwargs)

        def eval(self):
            captured["actor_eval"] = True

        def load_state_dict(self, state_dict, strict=True):
            captured["actor_load"] = (state_dict, strict)

    class FakeNormalizer:
        def __init__(self, *args, **kwargs):
            captured["normalizer_init"] = (args, kwargs)

        def load_state_dict(self, state_dict):
            captured["normalizer_load"] = state_dict

        def eval(self):
            captured["normalizer_eval"] = True

    monkeypatch.setattr(learner_module, "TD3Actor", FakeActor)
    monkeypatch.setattr(learner_module, "EmpiricalNormalization", FakeNormalizer)
    cfg = _offpolicy_cfg(algo="td3")
    actor_state = {"weight": torch.ones(1), "noise_scales": torch.zeros(1)}
    normalizer_state = {"mean": torch.ones(1)}

    actor, normalizer, actor_algo_type, actor_kwargs = build_play_actor(
        "td3",
        cfg,
        obs_dim=4,
        critic_obs_dim=6,
        action_dim=2,
        device="cpu",
    )
    load_play_actor(
        "td3",
        actor,
        normalizer,
        {"actor": actor_state, "obs_normalizer": normalizer_state},
    )

    assert isinstance(actor, FakeActor)
    assert isinstance(normalizer, FakeNormalizer)
    assert (actor_algo_type, actor_kwargs) == ("td3", {})
    assert captured["actor_load"] == ({"weight": actor_state["weight"]}, False)
    assert captured["actor_eval"] is True
    assert captured["normalizer_load"] == normalizer_state
    assert captured["normalizer_eval"] is True


@pytest.mark.parametrize("algo_name", ["sac", "flashsac"])
def test_offpolicy_load_play_actor_keeps_sac_state_dict_strict(algo_name: str):
    from unilab.training.offpolicy import load_play_actor

    captured: dict[str, Any] = {}

    class FakeActor:
        def load_state_dict(self, state_dict):
            captured["state_dict"] = state_dict

    actor_state = {"weight": object()}
    load_play_actor(algo_name, FakeActor(), None, {"actor": actor_state})

    assert captured["state_dict"] is actor_state


def test_play_offpolicy_can_skip_onnx_export_and_still_record_video(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    import torch

    mod = _offpolicy()
    cfg = _offpolicy_cfg(
        [
            "task=g1_walk_flat/mujoco",
            "training.play_only=true",
            "training.play_render_mode=record",
            "training.export_onnx=false",
        ]
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    checkpoint = run_dir / "model_5000.pt"
    torch.save({"actor": {}}, checkpoint)

    captured: dict[str, Any] = {}

    class FakeActor:
        def eval(self):
            return self

        def load_state_dict(self, state_dict):
            captured["loaded_state_dict"] = state_dict

        def as_export_module(self):
            raise AssertionError("ONNX export should be skipped when training.export_onnx=false")

        def explore(self, obs, deterministic=True):
            captured["deterministic"] = deterministic
            return torch.zeros((obs.shape[0], 2), dtype=obs.dtype, device=obs.device)

    class FakeEnv:
        def __init__(self):
            self.obs_groups_spec = {"obs": 4}
            self.action_space = type("ActionSpace", (), {"shape": (2,)})()
            self.state = None

        def init_state(self):
            self.state = type(
                "State",
                (),
                {"obs": {"obs": np.zeros((cfg.training.play_env_num, 4), dtype=np.float32)}},
            )()

        def reset(self, env_ids):
            batch = len(env_ids)
            return ({"obs": np.zeros((batch, 4), dtype=np.float32)}, {})

        def step(self, actions):
            batch = actions.shape[0]
            self.state = type(
                "State",
                (),
                {"obs": {"obs": np.ones((batch, 4), dtype=np.float32)}},
            )()
            captured["actions_shape"] = actions.shape
            return self.state

        def run_playback_mode(self, **kwargs):
            captured["play_render_mode"] = kwargs["play_render_mode"]
            captured["output_video"] = kwargs["output_video"]
            init_obs = kwargs["initialize"]()
            captured["init_obs_shape"] = init_obs.shape
            next_obs = kwargs["step"](init_obs)
            captured["next_obs_shape"] = next_obs.shape
            return str(kwargs["output_video"])

    monkeypatch.setattr(mod, "build_offpolicy_env_cfg_override", lambda algo_name, cfg: {})
    monkeypatch.setattr(mod, "default_device", lambda torch_module, preferred=None: "cpu")
    monkeypatch.setattr(mod, "create_env", lambda *args, **kwargs: FakeEnv())
    monkeypatch.setattr(mod, "resolve_play_obs_dim", lambda obs_groups_spec: 4)
    monkeypatch.setattr(mod, "extract_play_obs", lambda obs_dict: obs_dict["obs"])
    monkeypatch.setattr(
        mod,
        "resolve_checkpoint_path",
        lambda *args, **kwargs: (str(checkpoint), str(run_dir)),
    )
    monkeypatch.setattr(
        torch.onnx,
        "export",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("torch.onnx.export should not be called when training.export_onnx=false")
        ),
    )

    import unilab.algos.common.actor_factory as actor_factory

    monkeypatch.setattr(actor_factory, "build_actor", lambda *args, **kwargs: FakeActor())

    result = mod.play_offpolicy("sac", cfg)
    out = capsys.readouterr().out

    assert result == str(run_dir / "play_video.mp4")
    assert captured["loaded_state_dict"] == {}
    assert captured["play_render_mode"] == "record"
    assert captured["actions_shape"] == (cfg.training.play_env_num, 2)
    assert captured["init_obs_shape"] == (cfg.training.play_env_num, 4)
    assert captured["next_obs_shape"] == (cfg.training.play_env_num, 4)
    assert captured["deterministic"] is True
    assert "Skipping ONNX export because training.export_onnx=false." in out
    assert not (run_dir / "policy.onnx").exists()


def test_play_offpolicy_uses_hora_sac_actor_and_priv_info(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    import torch

    mod = _offpolicy()
    cfg = _offpolicy_cfg(
        [
            "task=sharpa_inhand/mujoco_hora",
            "training.play_only=true",
            "training.play_render_mode=record",
            "training.export_onnx=false",
            "training.play_env_num=2",
        ]
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    checkpoint = run_dir / "model_5000.pt"
    torch.save({"actor": {}}, checkpoint)

    captured: dict[str, Any] = {}
    reset_priv = np.array([[4.0, 5.0], [6.0, 7.0]], dtype=np.float32)
    step_priv = np.array([[8.0, 9.0], [10.0, 11.0]], dtype=np.float32)

    class FakeHoraActor:
        def eval(self):
            return self

        def load_state_dict(self, state_dict):
            captured["loaded_state_dict"] = state_dict

        def explore(self, obs, priv_info, deterministic=True):
            captured["obs_shape"] = tuple(obs.shape)
            captured["priv_info"] = priv_info.detach().cpu().numpy()
            captured["deterministic"] = deterministic
            return torch.zeros((obs.shape[0], 2), dtype=obs.dtype, device=obs.device)

    class FakeEnv:
        def __init__(self):
            self.obs_groups_spec = {"obs": 3, "critic": 5}
            self.action_space = type("ActionSpace", (), {"shape": (2,)})()
            self.state = None

        def init_state(self):
            self.state = type(
                "State",
                (),
                {
                    "obs": {
                        "obs": np.zeros((cfg.training.play_env_num, 3), dtype=np.float32),
                        "critic": np.zeros((cfg.training.play_env_num, 5), dtype=np.float32),
                    },
                    "info": {"critic_info": reset_priv},
                },
            )()

        def reset(self, env_ids):
            batch = len(env_ids)
            return (
                {
                    "obs": np.zeros((batch, 3), dtype=np.float32),
                    "critic": np.concatenate(
                        [np.zeros((batch, 3), dtype=np.float32), reset_priv],
                        axis=1,
                    ),
                },
                {"critic_info": reset_priv},
            )

        def step(self, actions):
            batch = actions.shape[0]
            captured["actions_shape"] = actions.shape
            self.state = type(
                "State",
                (),
                {
                    "obs": {
                        "obs": np.ones((batch, 3), dtype=np.float32),
                        "critic": np.concatenate(
                            [np.ones((batch, 3), dtype=np.float32), step_priv],
                            axis=1,
                        ),
                    },
                    "info": {"critic_info": step_priv},
                },
            )()
            return self.state

        def run_playback_mode(self, **kwargs):
            init_obs = kwargs["initialize"]()
            captured["init_obs_shape"] = init_obs.shape
            next_obs = kwargs["step"](init_obs)
            captured["next_obs_shape"] = next_obs.shape
            return str(kwargs["output_video"])

    monkeypatch.setattr(mod, "build_offpolicy_env_cfg_override", lambda algo_name, cfg: {})
    monkeypatch.setattr(mod, "default_device", lambda torch_module, preferred=None: "cpu")
    monkeypatch.setattr(mod, "create_env", lambda *args, **kwargs: FakeEnv())
    monkeypatch.setattr(
        mod,
        "resolve_checkpoint_path",
        lambda *args, **kwargs: (str(checkpoint), str(run_dir)),
    )

    import unilab.algos.common.actor_factory as actor_factory

    def fake_build_actor(algo_type, obs_dim, action_dim, hidden_dim, use_layer_norm, device, **kw):
        captured["build_actor"] = (algo_type, obs_dim, action_dim, kw)
        return FakeHoraActor()

    monkeypatch.setattr(actor_factory, "build_actor", fake_build_actor)

    result = mod.play_offpolicy("sac", cfg)

    assert result == str(run_dir / "play_video.mp4")
    assert captured["build_actor"][0] == "hora_sac"
    assert captured["build_actor"][1:3] == (3, 2)
    assert captured["build_actor"][3]["priv_info_dim"] == 2
    assert captured["loaded_state_dict"] == {}
    assert captured["actions_shape"] == (cfg.training.play_env_num, 2)
    assert captured["init_obs_shape"] == (cfg.training.play_env_num, 3)
    assert captured["next_obs_shape"] == (cfg.training.play_env_num, 3)
    assert captured["obs_shape"] == (cfg.training.play_env_num, 3)
    np.testing.assert_allclose(captured["priv_info"], reset_priv)
    assert captured["deterministic"] is True
    assert not (run_dir / "policy.onnx").exists()


# ---------------------------------------------------------------------------
# play_interactive.py — resolve_checkpoint()
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_MUJOCO, reason="mujoco not installed")
def test_play_resolve_checkpoint_nonexistent_run(tmp_path):
    """Passing a non-existent explicit path returns None."""
    mod = _load_script("play_interactive")
    result = mod.resolve_checkpoint("MyTask", str(tmp_path / "no_run"))
    assert result is None


@pytest.mark.skipif(not _HAS_MUJOCO, reason="mujoco not installed")
def test_play_resolve_checkpoint_dir_with_model(tmp_path):
    """Directory path containing model_*.pt files resolves to the latest."""
    mod = _load_script("play_interactive")
    run_dir = tmp_path / "2024-01-01_mujoco"
    run_dir.mkdir()
    (run_dir / "model_10.pt").write_bytes(b"")
    (run_dir / "model_50.pt").write_bytes(b"")

    result = mod.resolve_checkpoint("MyTask", str(run_dir))
    assert result is not None
    assert "model_50.pt" in result


@pytest.mark.skipif(not _HAS_MUJOCO, reason="mujoco not installed")
def test_play_resolve_checkpoint_explicit_file(tmp_path):
    """Absolute path to existing .pt file returns that path unchanged."""
    mod = _load_script("play_interactive")
    model_file = tmp_path / "model_99.pt"
    model_file.write_bytes(b"")
    result = mod.resolve_checkpoint("MyTask", str(model_file))
    assert result == str(model_file)


@pytest.mark.skipif(not _HAS_MUJOCO, reason="mujoco not installed")
def test_play_resolve_checkpoint_empty_dir(tmp_path):
    """Directory with no model_*.pt files returns None."""
    mod = _load_script("play_interactive")
    run_dir = tmp_path / "run1"
    run_dir.mkdir()
    result = mod.resolve_checkpoint("MyTask", str(run_dir))
    assert result is None


@pytest.mark.skipif(not _HAS_MUJOCO, reason="mujoco not installed")
def test_play_resolve_checkpoint_delegates_to_shared_helper(
    monkeypatch: pytest.MonkeyPatch, tmp_path
):
    mod = _load_script("play_interactive")
    model_path = tmp_path / "resolved" / "model_12.pt"
    model_path.parent.mkdir(parents=True)
    model_path.write_bytes(b"")
    captured: dict[str, object] = {}

    def _fake_resolver(root_dir, **kwargs):
        captured["root_dir"] = root_dir
        captured.update(kwargs)
        return model_path, model_path.parent

    monkeypatch.setattr(mod, "resolve_task_checkpoint_path", _fake_resolver)

    result = mod.resolve_checkpoint("MyTask", "-1", checkpoint="12", algo_log_name="custom_ppo")

    assert result == str(model_path)
    assert captured["root_dir"] == mod.ROOT_DIR
    assert captured["task_name"] == "MyTask"
    assert captured["load_run"] == "-1"
    assert captured["algo_log_name"] == "custom_ppo"
    assert captured["checkpoint"] == "12"


# ---------------------------------------------------------------------------
# play_interactive.py — RslRlVecEnvWrapper contract behavior
# ---------------------------------------------------------------------------


def _play_interactive():
    """Load play_interactive.py as a module."""
    return _load_script("play_interactive")


def test_play_wrapper_imports_shared_implementation():
    """Verify play_interactive.py uses shared RslRlVecEnvWrapper."""
    from unilab.training.rsl_rl import RslRlVecEnvWrapper as SharedWrapper

    mod = _play_interactive()
    # The wrapper class in play_interactive should be the shared one
    assert mod.RslRlVecEnvWrapper is SharedWrapper


def test_play_wrapper_uses_current_reset_contract():
    """Verify wrapper reset() uses current (obs, info) contract, not old (_, obs, _)."""
    import numpy as np
    from tensordict import TensorDict

    from unilab.training.rsl_rl import RslRlVecEnvWrapper

    # Create a fake environment that returns (obs, info) tuple
    class FakeEnv:
        def __init__(self):
            self.num_envs = 2
            self.state = type("State", (), {"obs": {"obs": np.ones((2, 5), dtype=np.float32)}})()
            self.cfg = type("Cfg", (), {"max_episode_seconds": 10.0, "ctrl_dt": 0.02})()
            self.observation_space = type("Space", (), {"shape": (5,)})()
            self.action_space = type("Space", (), {"shape": (3,)})()
            self.obs_groups_spec = {"obs": 5}

        def init_state(self):
            pass

        def reset(self, env_indices):
            # Returns current contract: (obs, info)
            return {"obs": np.ones((2, 5), dtype=np.float32)}, {}

    env = FakeEnv()
    wrapper = RslRlVecEnvWrapper(env, device="cpu", policy_obs_mode="flat")

    # Reset should work with current contract
    obs_td, info = wrapper.reset()

    assert isinstance(obs_td, TensorDict)
    assert "policy" in obs_td
    assert "actor" in obs_td
    assert obs_td.batch_size == (2,)


def test_play_wrapper_policy_obs_mode_actor():
    """Verify wrapper supports policy_obs_mode='actor'."""
    import numpy as np

    from unilab.training.rsl_rl import RslRlVecEnvWrapper

    class FakeEnv:
        def __init__(self):
            self.num_envs = 1
            self.state = type("State", (), {"obs": {"obs": np.ones((1, 3), dtype=np.float32)}})()
            self.cfg = type("Cfg", (), {"max_episode_seconds": 10.0, "ctrl_dt": 0.02})()
            self.observation_space = type("Space", (), {"shape": (3,)})()
            self.action_space = type("Space", (), {"shape": (2,)})()
            self.obs_groups_spec = {"obs": 3, "critic": 5}

        def init_state(self):
            pass

        def reset(self, env_indices):
            return {
                "obs": np.ones((1, 3), dtype=np.float32),
                "critic": np.zeros((1, 5), dtype=np.float32),
            }, {}

    env = FakeEnv()

    # Test actor mode - num_obs should match actor obs dim only
    wrapper_actor = RslRlVecEnvWrapper(env, device="cpu", policy_obs_mode="actor")
    assert wrapper_actor.num_obs == 3  # Only "obs" group
    assert wrapper_actor._actor_obs_dim == 3
    assert wrapper_actor._flat_obs_dim == 3

    obs_td, _ = wrapper_actor.reset()
    # In actor mode, policy obs should equal actor obs
    assert obs_td["policy"].shape == (1, 3)
    assert obs_td["actor"].shape == (1, 3)
    assert obs_td["critic"].shape == (1, 5)


def test_play_wrapper_flat_policy_excludes_critic_only_group():
    import numpy as np

    from unilab.training.rsl_rl import RslRlVecEnvWrapper

    class FakeEnv:
        def __init__(self):
            self.num_envs = 1
            self.state = type(
                "State",
                (),
                {
                    "obs": {
                        "obs": np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
                        "critic": np.array([[9.0, 9.0, 9.0, 9.0]], dtype=np.float32),
                    }
                },
            )()
            self.cfg = type("Cfg", (), {"max_episode_seconds": 10.0, "ctrl_dt": 0.02})()
            self.observation_space = type("Space", (), {"shape": (7,)})()
            self.action_space = type("Space", (), {"shape": (2,)})()
            self.obs_groups_spec = {"obs": 3, "critic": 4}

        def init_state(self):
            pass

        def reset(self, env_indices):
            return cast(dict[str, np.ndarray], getattr(self.state, "obs")), {}

    wrapper = RslRlVecEnvWrapper(FakeEnv(), device="cpu", policy_obs_mode="flat")
    obs_td, _ = wrapper.reset()

    np.testing.assert_allclose(
        obs_td["policy"].cpu().numpy(),
        np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
    )
    np.testing.assert_allclose(
        obs_td["critic"].cpu().numpy(),
        np.array([[9.0, 9.0, 9.0, 9.0]], dtype=np.float32),
    )
    assert wrapper.num_obs == 3
    assert wrapper.num_privileged_obs == 4


def test_play_wrapper_preserves_hora_priv_info_and_proprio_history():
    import numpy as np

    from unilab.algos.hora.rsl_rl import HoraRslRlVecEnvWrapper

    class FakeEnv:
        def __init__(self):
            self.num_envs = 1
            self.state = type(
                "State",
                (),
                {
                    "obs": {
                        "obs": np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
                        "proprio": np.array([[6.0, 7.0]], dtype=np.float32),
                        "critic": np.array([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=np.float32),
                    },
                    "info": {
                        "critic_info": np.array([[4.0, 5.0]], dtype=np.float32),
                        "proprio_hist": np.array(
                            [[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]],
                            dtype=np.float32,
                        ),
                    },
                },
            )()
            self.cfg = type("Cfg", (), {"max_episode_seconds": 10.0, "ctrl_dt": 0.02})()
            self.observation_space = type("Space", (), {"shape": (5,)})()
            self.action_space = type("Space", (), {"shape": (2,)})()
            self.obs_groups_spec = {"obs": 3, "proprio": 2, "critic": 5}

        def init_state(self):
            pass

        def reset(self, env_indices):
            del env_indices
            return (
                cast(dict[str, np.ndarray], getattr(self.state, "obs")),
                cast(dict[str, np.ndarray], getattr(self.state, "info")),
            )

    assert "reset" not in HoraRslRlVecEnvWrapper.__dict__
    assert "get_observations" not in HoraRslRlVecEnvWrapper.__dict__

    wrapper = HoraRslRlVecEnvWrapper(FakeEnv(), device="cpu", policy_obs_mode="flat")
    reset_obs_td, reset_info = wrapper.reset()
    current_obs_td = wrapper.get_observations()

    assert reset_info is wrapper.env.state.info
    for obs_td in (reset_obs_td, current_obs_td):
        np.testing.assert_allclose(
            obs_td["policy"].cpu().numpy(),
            np.array([[1.0, 2.0, 3.0, 6.0, 7.0]], dtype=np.float32),
        )
        np.testing.assert_allclose(
            obs_td["priv_info"].cpu().numpy(),
            np.array([[4.0, 5.0]], dtype=np.float32),
        )
        np.testing.assert_allclose(
            obs_td["proprio_hist"].cpu().numpy(),
            np.array([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]], dtype=np.float32),
        )


def test_play_wrapper_step_exports_timeout_bootstrap_obs():
    import torch

    from unilab.training.rsl_rl import RslRlVecEnvWrapper

    class FakeEnv:
        def __init__(self):
            self.num_envs = 1
            self.cfg = type("Cfg", (), {"max_episode_seconds": 10.0, "ctrl_dt": 0.02})()
            self.observation_space = type("Space", (), {"shape": (3,)})()
            self.action_space = type("Space", (), {"shape": (2,)})()
            self.obs_groups_spec = {"obs": 3, "critic": 2}
            self.state = type("State", (), {"obs": {"obs": np.zeros((1, 3), dtype=np.float32)}})()

        def init_state(self):
            pass

        def reset(self, env_indices):
            return {"obs": np.zeros((1, 3), dtype=np.float32)}, {}

        def step(self, actions):
            return type(
                "StepState",
                (),
                {
                    "obs": {"obs": np.array([[1.0, 2.0, 3.0]], dtype=np.float32)},
                    "reward": np.array([1.0], dtype=np.float32),
                    "terminated": np.array([False]),
                    "truncated": np.array([True]),
                    "final_observation": {
                        "obs": np.array([[7.0, 8.0, 9.0]], dtype=np.float32),
                        "critic": np.array([[4.0, 5.0]], dtype=np.float32),
                    },
                    "info": {
                        "final_observation": {
                            "obs": np.array([[7.0, 8.0, 9.0]], dtype=np.float32),
                            "critic": np.array([[4.0, 5.0]], dtype=np.float32),
                        }
                    },
                },
            )()

    wrapper = RslRlVecEnvWrapper(FakeEnv(), device="cpu", policy_obs_mode="flat")

    _, _, _, infos = wrapper.step(torch.zeros((1, 2)))

    assert torch.equal(infos["time_outs"], torch.tensor([True]))
    np.testing.assert_allclose(
        infos["time_out_bootstrap_obs"]["policy"].cpu().numpy(),
        np.array([[7.0, 8.0, 9.0]], dtype=np.float32),
    )
    np.testing.assert_allclose(
        infos["time_out_bootstrap_obs"]["critic"].cpu().numpy(),
        np.array([[4.0, 5.0]], dtype=np.float32),
    )


def test_play_wrapper_timeout_bootstrap_preserves_hora_priv_info():
    import torch

    from unilab.algos.hora.rsl_rl import HoraRslRlVecEnvWrapper

    class FakeEnv:
        def __init__(self):
            self.num_envs = 1
            self.cfg = type("Cfg", (), {"max_episode_seconds": 10.0, "ctrl_dt": 0.02})()
            self.observation_space = type("Space", (), {"shape": (5,)})()
            self.action_space = type("Space", (), {"shape": (2,)})()
            self.obs_groups_spec = {"obs": 3, "critic": 5}
            self.state = type(
                "State",
                (),
                {
                    "obs": {
                        "obs": np.zeros((1, 3), dtype=np.float32),
                        "critic": np.zeros((1, 5), dtype=np.float32),
                    },
                    "info": {
                        "critic_info": np.zeros((1, 2), dtype=np.float32),
                        "proprio_hist": np.zeros((1, 2, 3), dtype=np.float32),
                    },
                },
            )()

        def init_state(self):
            pass

        def reset(self, env_indices):
            del env_indices
            return cast(dict[str, np.ndarray], getattr(self.state, "obs")), cast(
                dict[str, np.ndarray], getattr(self.state, "info")
            )

        def step(self, actions):
            del actions
            return type(
                "StepState",
                (),
                {
                    "obs": {"obs": np.array([[1.0, 2.0, 3.0]], dtype=np.float32)},
                    "reward": np.array([1.0], dtype=np.float32),
                    "terminated": np.array([True]),
                    "truncated": np.array([True]),
                    "final_observation": {
                        "obs": np.array([[7.0, 8.0, 9.0]], dtype=np.float32),
                        "critic": np.array([[7.0, 8.0, 9.0, 4.0, 5.0]], dtype=np.float32),
                    },
                    "info": {
                        "final_observation": {
                            "obs": np.array([[7.0, 8.0, 9.0]], dtype=np.float32),
                            "critic": np.array([[7.0, 8.0, 9.0, 4.0, 5.0]], dtype=np.float32),
                        },
                        "critic_info": np.array([[0.0, 0.0]], dtype=np.float32),
                        "proprio_hist": np.zeros((1, 2, 3), dtype=np.float32),
                    },
                },
            )()

    wrapper = HoraRslRlVecEnvWrapper(FakeEnv(), device="cpu", policy_obs_mode="flat")

    _, _, _, infos = wrapper.step(torch.zeros((1, 2)))

    np.testing.assert_allclose(
        infos["time_out_bootstrap_obs"]["priv_info"].cpu().numpy(),
        np.array([[4.0, 5.0]], dtype=np.float32),
    )


# ---------------------------------------------------------------------------
# Issue #168: Unified log directory and load_run resolution
# ---------------------------------------------------------------------------


def test_ppo_hydra_default_algo_log_name():
    """Verify PPO config has algo_log_name in algo section."""
    cfg = _ppo_cfg()
    assert cfg.algo.algo_log_name == "rsl_rl_ppo"


def test_ppo_hydra_load_run_in_algo_not_training():
    """Verify load_run is in algo section, not training section (issue #168)."""
    from omegaconf import OmegaConf

    cfg = _ppo_cfg()
    assert cfg.algo.load_run == "-1"
    # training section should NOT have load_run anymore
    assert "load_run" not in cfg.training or OmegaConf.is_missing(cfg.training, "load_run")


def test_appo_hydra_default_algo_log_name():
    """Verify APPO config has algo_log_name in algo section."""
    cfg = _appo_cfg()
    assert cfg.algo.algo_log_name == "appo"
    assert cfg.algo.load_run == "-1"


def test_offpolicy_sac_hydra_default_algo_log_name():
    """Verify SAC config has algo_log_name in algo section."""
    cfg = _offpolicy_cfg()
    assert cfg.algo.algo_log_name == "fast_sac"
    assert cfg.algo.load_run == "-1"


def test_offpolicy_td3_hydra_default_algo_log_name():
    """Verify TD3 config has algo_log_name in algo section."""
    cfg = _offpolicy_cfg(algo="td3")
    assert cfg.algo.algo_log_name == "fast_td3"
    assert cfg.algo.load_run == "-1"


def test_offpolicy_flashsac_hydra_algo_log_name():
    cfg = _offpolicy_cfg(["task=g1_walk_flat/mujoco"], algo="flashsac")
    assert cfg.algo.algo_log_name == "flash_sac"
    assert cfg.algo.load_run == "-1"


def test_offpolicy_flashsac_g1_walk_flat_task_composes() -> None:
    cfg = _offpolicy_cfg(["task=g1_walk_flat/mujoco"], algo="flashsac")
    assert cfg.training.task_name == "G1WalkFlat"
    assert cfg.training.sim_backend == "mujoco"


def test_offpolicy_g1_rough_terrain_task_composes() -> None:
    cfg = _offpolicy_cfg(["task=g1_walk_rough/mujoco"])

    assert cfg.training.task_name == "G1WalkRough"
    assert cfg.training.sim_backend == "mujoco"


def test_offpolicy_rejects_algo_argument_mismatch():
    """build_runner must reject an algo argument inconsistent with cfg.algo.algo."""
    cfg = _offpolicy_cfg(["task=g1_walk_flat/mujoco"])

    with pytest.raises(ValueError, match="inconsistent with cfg.algo.algo"):
        _offpolicy().build_runner("flashsac", cfg)


def test_train_rsl_rl_get_log_root_uses_algo_log_name(monkeypatch: pytest.MonkeyPatch):
    """Verify _get_log_root uses algo.algo_log_name (issue #168)."""
    monkeypatch.delenv("UNILAB_TEST_LOG_ROOT", raising=False)
    mod = _train_rsl_rl(monkeypatch)
    cfg = _ppo_cfg()

    # Override algo_log_name to test
    cfg.algo.algo_log_name = "test_rsl_rl_ppo"

    log_root = mod._get_log_root(cfg)
    assert "logs/test_rsl_rl_ppo" in log_root.replace("\\", "/")


def test_train_rsl_rl_play_missing_checkpoint_skips_env_creation_and_prints_context(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
):
    monkeypatch.delenv("UNILAB_TEST_LOG_ROOT", raising=False)
    mod = _train_rsl_rl(monkeypatch)
    cfg = _ppo_cfg(["task=go1_joystick_flat/mujoco", "training.play_only=true"])
    cfg.algo.algo_log_name = "custom_ppo"

    original_root = mod.ROOT_DIR
    mod.ROOT_DIR = tmp_path
    try:
        monkeypatch.setattr(
            mod,
            "create_env",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("play_rsl_rl should not create an env before checkpoint resolution")
            ),
        )

        result = mod.play_rsl_rl(cfg, device="cpu")
    finally:
        mod.ROOT_DIR = original_root

    captured = capsys.readouterr().out
    expected_task_log_root = tmp_path / "logs" / "custom_ppo" / cfg.training.task_name

    assert result is None
    assert "Could not resolve a checkpoint for play mode." in captured
    assert "Task log root does not exist." in captured
    assert f"task_log_root={expected_task_log_root}" in captured
    assert "algo.load_run='-1'" in captured


def test_train_rsl_rl_play_reports_missing_requested_checkpoint_in_resolved_run(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
):
    monkeypatch.delenv("UNILAB_TEST_LOG_ROOT", raising=False)
    mod = _train_rsl_rl(monkeypatch)
    cfg = _ppo_cfg(["task=go1_joystick_flat/mujoco", "training.play_only=true"])
    cfg.algo.algo_log_name = "custom_ppo"
    cfg.algo.checkpoint = 12

    run_dir = (
        tmp_path / "logs" / "custom_ppo" / cfg.training.task_name / "2024-01-01_00-00-00_mujoco"
    )
    run_dir.mkdir(parents=True)
    (run_dir / "model_9.pt").write_bytes(b"")

    original_root = mod.ROOT_DIR
    mod.ROOT_DIR = tmp_path
    try:
        monkeypatch.setattr(
            mod,
            "create_env",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("play_rsl_rl should not create an env before checkpoint resolution")
            ),
        )

        result = mod.play_rsl_rl(cfg, device="cpu")
    finally:
        mod.ROOT_DIR = original_root

    captured = capsys.readouterr().out

    assert result is None
    assert "Could not resolve a checkpoint for play mode." in captured
    assert f"resolved_run={run_dir}" in captured
    assert "algo.checkpoint=12" in captured


def test_train_rsl_rl_motrix_auto_play_is_interactive(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    mod = _train_rsl_rl(monkeypatch)
    cfg = _ppo_cfg(
        [
            "task=go2_joystick_rough/motrix",
            "training.play_only=true",
            "training.play_steps=37",
            "training.render_spacing=2.5",
        ]
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    checkpoint = run_dir / "model_37.pt"
    mod.torch.save({"actor_state_dict": {}}, checkpoint)

    class FakeEnv:
        def __init__(self):
            self.cfg = type("Cfg", (), {"render_spacing": 2.5, "render_offset_mode": "zero"})()

        def run_playback_mode(self, **kwargs):
            assert kwargs["play_render_mode"] == "auto"
            assert kwargs["play_steps"] == 37
            plan = type(
                "Plan",
                (),
                {
                    "mode": "interactive",
                    "headless": False,
                    "record_video": False,
                    "num_steps": None,
                    "output_video": None,
                },
            )()
            kwargs["on_plan"](plan)
            captured["env"] = self
            captured.update({key: value for key, value in kwargs.items() if key != "on_plan"})
            captured["headless"] = plan.headless
            captured["record_video"] = plan.record_video
            captured["num_steps"] = plan.num_steps
            captured["output_video"] = plan.output_video
            return None

    class FakeWrapper:
        def __init__(self, env, device):
            self.env = env
            self.device = device

        def reset(self):
            return 0, {}

        def step(self, actions):
            return 0, 0, False, {}

    class FakeRunner:
        def __init__(self, wrapped_env, train_cfg, log_dir, device):
            self.wrapped_env = wrapped_env
            self.train_cfg = train_cfg
            self.log_dir = log_dir
            self.device = device

        def load(self, path, **kwargs):
            self.loaded_path = path
            self.load_kwargs = kwargs

        def get_inference_policy(self, device):
            return lambda obs: obs

    captured: dict[str, Any] = {}

    monkeypatch.setattr(mod, "EXPORT_POLICY", False, raising=False)
    monkeypatch.setattr(mod, "parse_checkpoint_path", lambda *args, **kwargs: (checkpoint, run_dir))
    monkeypatch.setattr(mod, "build_ppo_play_env_cfg_override", lambda cfg: {})
    monkeypatch.setattr(mod, "create_env", lambda *args, **kwargs: FakeEnv())
    monkeypatch.setattr(mod, "_resolve_ppo_wrapper_cls", lambda rl_cfg: FakeWrapper)
    monkeypatch.setattr(mod, "normalize_ppo_train_cfg", lambda rl_cfg: {})
    monkeypatch.setattr(mod, "OnPolicyRunner", FakeRunner)

    result = mod.play_rsl_rl(cfg, device="cpu")

    assert result is None
    assert captured["headless"] is False
    assert captured["record_video"] is False
    assert captured["num_steps"] is None
    assert captured["output_video"] is None
    assert captured["render_spacing"] == pytest.approx(2.5)
    assert captured["render_offset_mode"] == "zero"


def test_train_rsl_rl_record_play_uses_backend_plan(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    mod = _train_rsl_rl(monkeypatch)
    cfg = _ppo_cfg(
        [
            "task=go2_joystick_rough/motrix",
            "training.play_only=true",
            "training.play_render_mode=record",
            "training.play_steps=37",
        ]
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    checkpoint = run_dir / "model_37.pt"
    mod.torch.save({"actor_state_dict": {}}, checkpoint)

    class FakeEnv:
        def __init__(self):
            self.cfg = type("Cfg", (), {"render_spacing": 1.0, "render_offset_mode": "grid"})()

        def run_playback_mode(self, **kwargs):
            assert kwargs["play_render_mode"] == "record"
            assert kwargs["play_steps"] == 37
            plan = type(
                "Plan",
                (),
                {
                    "mode": "record",
                    "headless": True,
                    "record_video": True,
                    "num_steps": 37,
                    "output_video": kwargs["output_video"],
                },
            )()
            kwargs["on_plan"](plan)
            captured["env"] = self
            captured.update({key: value for key, value in kwargs.items() if key != "on_plan"})
            captured["headless"] = plan.headless
            captured["record_video"] = plan.record_video
            captured["num_steps"] = plan.num_steps
            captured["output_video"] = plan.output_video
            return str(plan.output_video)

    class FakeWrapper:
        def __init__(self, env, device):
            self.env = env
            self.device = device

        def reset(self):
            return 0, {}

        def step(self, actions):
            return 0, 0, False, {}

    class FakeRunner:
        def __init__(self, wrapped_env, train_cfg, log_dir, device):
            self.wrapped_env = wrapped_env
            self.train_cfg = train_cfg
            self.log_dir = log_dir
            self.device = device

        def load(self, path, map_location=None):
            self.loaded_path = path
            self.map_location = map_location

        def get_inference_policy(self, device):
            return lambda obs: obs

    captured: dict[str, Any] = {}

    monkeypatch.setattr(mod, "EXPORT_POLICY", False, raising=False)
    monkeypatch.setattr(mod, "parse_checkpoint_path", lambda *args, **kwargs: (checkpoint, run_dir))
    monkeypatch.setattr(mod, "build_ppo_play_env_cfg_override", lambda cfg: {})
    monkeypatch.setattr(mod, "create_env", lambda *args, **kwargs: FakeEnv())
    monkeypatch.setattr(mod, "_resolve_ppo_wrapper_cls", lambda rl_cfg: FakeWrapper)
    monkeypatch.setattr(mod, "normalize_ppo_train_cfg", lambda rl_cfg: {})
    monkeypatch.setattr(mod, "OnPolicyRunner", FakeRunner)

    result = mod.play_rsl_rl(cfg, device="cpu")

    assert result == str(run_dir / "play_video.mp4")
    assert captured["headless"] is True
    assert captured["record_video"] is True
    assert captured["num_steps"] == 37
    assert captured["output_video"] == run_dir / "play_video.mp4"


def test_train_appo_get_log_root_uses_algo_log_name(monkeypatch: pytest.MonkeyPatch):
    """Verify APPO _get_log_root uses algo.algo_log_name (issue #168)."""
    monkeypatch.delenv("UNILAB_TEST_LOG_ROOT", raising=False)
    mod = _train_appo()
    cfg = _appo_cfg()

    cfg.algo.algo_log_name = "test_appo"

    log_root = mod._get_log_root(cfg)
    assert "logs/test_appo" in log_root.replace("\\", "/")


def test_play_resolve_checkpoint_uses_algo_log_name(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """Verify play_interactive.resolve_checkpoint uses algo_log_name (issue #168)."""
    monkeypatch.delenv("UNILAB_TEST_LOG_ROOT", raising=False)
    mod = _play_interactive()

    # Create test directory structure with custom algo_log_name
    run_dir = tmp_path / "logs" / "custom_ppo" / "MyTask" / "2024-01-01_mujoco"
    run_dir.mkdir(parents=True)
    (run_dir / "model_50.pt").write_bytes(b"")

    # Temporarily override ROOT_DIR to use tmp_path
    original_root = mod.ROOT_DIR
    try:
        mod.ROOT_DIR = tmp_path
        result = mod.resolve_checkpoint("MyTask", "-1", algo_log_name="custom_ppo")
        assert result is not None
        assert "model_50.pt" in result
    finally:
        mod.ROOT_DIR = original_root


def test_ppo_interactive_config_includes_playback_controls():
    cfg = _ppo_cfg()

    assert cfg.interactive.speed == pytest.approx(1.0)
    assert cfg.interactive.start_paused is False


def test_play_interactive_respects_training_device_override():
    mod = _play_interactive()
    cfg = OmegaConf.create({"training": {"device": "cpu"}})

    assert mod._select_playback_device(cfg) == "cpu"


def test_play_interactive_parses_explicit_cli():
    mod = _play_interactive()

    parsed = mod._parse_interactive_cli(
        ["--algo", "hora_distill", "--task", "sharpa_inhand", "--sim", "mujoco_nodr"]
    )

    assert parsed.algo == "hora_distill"
    assert parsed.task == "sharpa_inhand"
    assert parsed.sim == "mujoco_nodr"
    assert parsed.overrides == ["task=sharpa_inhand/mujoco_nodr"]


@pytest.mark.parametrize("algo", ["appo", "sac", "hora_distill"])
def test_play_interactive_parses_feature_algo_flags(algo: str):
    mod = _play_interactive()

    parsed = mod._parse_interactive_cli(
        [f"--algo={algo}", "--task", "sharpa_inhand", "--sim", "mujoco_hora"]
    )

    assert parsed.algo == algo
    assert parsed.overrides == ["task=sharpa_inhand/mujoco_hora"]


def test_play_interactive_cli_respects_owner_action_mode_and_user_override():
    mod = _play_interactive()

    default_parsed = mod._parse_interactive_cli(
        ["--algo", "ppo", "--task", "go2_joystick_rough", "--sim", "mujoco"]
    )
    default_cfg = mod._compose_interactive_config(default_parsed.algo, default_parsed.overrides)

    assert default_cfg.interactive.action_mode == "policy"

    parsed = mod._parse_interactive_cli(
        [
            "--algo",
            "ppo",
            "--task",
            "go2_joystick_rough",
            "--sim",
            "mujoco",
            "interactive.action_mode=random",
        ]
    )
    cfg = mod._compose_interactive_config(parsed.algo, parsed.overrides)

    assert parsed.overrides == [
        "task=go2_joystick_rough/mujoco",
        "interactive.action_mode=random",
    ]
    assert cfg.interactive.action_mode == "random"


def test_play_interactive_rejects_unknown_algo_flag():
    mod = _play_interactive()

    with pytest.raises(SystemExit):
        mod._parse_interactive_cli(
            ["--algo=unknown", "--task", "go1_joystick_flat", "--sim", "mujoco"]
        )


def test_play_interactive_dynamic_compose_supports_algo_roots():
    mod = _play_interactive()

    ppo_cfg = mod._compose_interactive_config("ppo", ["task=go1_joystick_flat/mujoco"])
    appo_cfg = mod._compose_interactive_config("appo", ["task=sharpa_inhand/mujoco_hora"])
    sac_cfg = mod._compose_interactive_config("sac", ["task=sharpa_inhand/mujoco_hora"])
    distill_cfg = mod._compose_interactive_config("hora_distill", ["task=sharpa_inhand/mujoco"])

    assert ppo_cfg.algo.algo == "ppo"
    assert appo_cfg.algo.runtime_impl == "hora_appo"
    assert appo_cfg.interactive.action_mode == "policy"
    assert sac_cfg.algo.algo == "sac"
    assert sac_cfg.algo.runtime_impl == "hora_sac"
    assert sac_cfg.interactive.policy_obs_mode == "actor"
    assert distill_cfg.algo.algo_log_name == "hora_distill"
    assert distill_cfg.interactive.action_mode == "policy"


def test_play_interactive_sac_overrides_pass_through():
    mod = _play_interactive()

    overrides = mod._normalize_interactive_overrides(
        "sac",
        ["task=sharpa_inhand/mujoco_hora", "algo.load_run=my_run"],
    )

    assert overrides == [
        "task=sharpa_inhand/mujoco_hora",
        "algo.load_run=my_run",
    ]


def test_play_interactive_runner_log_dir_uses_algo_log_name(monkeypatch: pytest.MonkeyPatch):
    import types

    mod = _play_interactive()
    captured: dict[str, object] = {}

    class FakeWrapper:
        def __init__(self, env, device, policy_obs_mode):
            self.env = env
            captured["policy_obs_mode"] = policy_obs_mode

        def reset(self):
            return None, {}

    class FakeRunner:
        def __init__(self, wrapped_env, train_cfg, log_dir, device):
            del wrapped_env, train_cfg, device
            captured["log_dir"] = log_dir

        def load(self, ckpt, load_cfg):
            captured["ckpt"] = ckpt
            captured["load_cfg"] = load_cfg

        def get_inference_policy(self, device):
            del device
            return object()

    class FakeViewer:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def is_running(self):
            return False

        def sync(self):
            pass

        user_scn = type("Scene", (), {"ngeom": 0})()

    fake_env = types.SimpleNamespace(
        obs_groups_spec={"obs": 5},
        action_space=types.SimpleNamespace(shape=(3,), low=np.full((3,), -1.0), high=np.ones((3,))),
        cfg=types.SimpleNamespace(ctrl_dt=0.02),
        get_playback_model=lambda: object(),
        get_scene_visual_model_file=lambda: None,
        get_physics_state_snapshot=lambda: np.zeros((1, 8), dtype=np.float32),
    )

    monkeypatch.setattr(mod.registry, "make", lambda *args, **kwargs: fake_env)
    monkeypatch.setattr(mod, "resolve_checkpoint", lambda *args, **kwargs: "/tmp/model_10.pt")
    monkeypatch.setattr(
        mod,
        "get_entrypoint_log_root",
        lambda root_dir, *, algo_log_name, log_root=None: Path("/tmp") / algo_log_name,
    )
    monkeypatch.setattr(mod, "RslRlVecEnvWrapper", FakeWrapper)
    monkeypatch.setattr(mod, "OnPolicyRunner", FakeRunner)
    monkeypatch.setattr(mod, "PPOConfig", lambda: types.SimpleNamespace(to_dict=lambda: {}))
    monkeypatch.setattr(mod.mujoco, "MjData", lambda model: object())
    monkeypatch.setattr(mod.mujoco, "mj_setState", lambda *args, **kwargs: None)
    monkeypatch.setattr(mod.mujoco, "mj_forward", lambda *args, **kwargs: None)
    monkeypatch.setattr(mod.mujoco, "mjtState", types.SimpleNamespace(mjSTATE_FULLPHYSICS=0))
    monkeypatch.setattr(mod.mujoco.viewer, "launch_passive", lambda *args, **kwargs: FakeViewer())

    args = types.SimpleNamespace(
        task="MyTask",
        load_run="-1",
        checkpoint=None,
        action_mode="policy",
        policy_obs_mode="flat",
        algo_log_name="custom_ppo",
        show_target_bodies=False,
        show_reward_debug=False,
        target_body_names="",
        target_max_bodies=0,
        target_marker_radius=0.02,
        target_axis_length=0.08,
        target_marker_alpha=0.75,
        target_show_axes=False,
        reward_debug_show_velocity=False,
        reward_debug_lin_vel_scale=0.08,
        reward_debug_ang_vel_scale=0.05,
        reward_debug_show_connectors=False,
        reward_debug_show_global_anchor=False,
        speed=1.0,
        start_paused=False,
    )

    mod.play_interactive(args)

    assert captured["ckpt"] == "/tmp/model_10.pt"
    assert captured["log_dir"].replace("\\", "/") == "/tmp/custom_ppo/MyTask/play_temp"


def test_play_interactive_import_does_not_swallow_registry_bootstrap_errors(
    monkeypatch: pytest.MonkeyPatch,
):
    import types

    play_interactive_path = _SCRIPTS_DIR / "play_interactive.py"
    training_mod = cast(Any, types.ModuleType("unilab.training"))

    def _fail_bootstrap() -> None:
        raise RuntimeError("bootstrap failed")

    training_mod.ensure_registries = _fail_bootstrap
    training_mod.algo_config_dict = lambda cfg: {}
    training_mod.get_entrypoint_log_root = lambda *args, **kwargs: Path("/tmp")
    training_mod.resolve_task_checkpoint_path = lambda *args, **kwargs: (None, None)
    monkeypatch.setitem(sys.modules, "unilab.training", training_mod)

    mujoco_mod = cast(Any, types.ModuleType("mujoco"))
    mujoco_mod.viewer = cast(Any, types.ModuleType("mujoco.viewer"))
    monkeypatch.setitem(sys.modules, "mujoco", mujoco_mod)
    monkeypatch.setitem(sys.modules, "mujoco.viewer", mujoco_mod.viewer)

    spec = importlib.util.spec_from_file_location(
        "play_interactive_test_module", play_interactive_path
    )
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]

    with pytest.raises(RuntimeError, match="bootstrap failed"):
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
