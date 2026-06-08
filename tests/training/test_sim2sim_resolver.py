"""Unit tests for the cross-backend sim2sim contract resolver.

Pure and fast: no environment, registry, torch, or backend creation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from unilab.training.sim2sim import (
    ALLOWLIST,
    DENYLIST,
    ENV_STRUCTURAL_DENYLIST,
    WARNING_LIST,
    CrossBackendIncompatibleError,
    Sim2SimConfigResolver,
    extract_contract_snapshot,
    policy_load_dim_guard,
    resolve_sim2sim_config,
)


def _write_sidecar(run_dir: Path, snapshot: dict[str, Any] | None) -> Path:
    payload: dict[str, Any] = {"run": {}, "config": {}}
    if snapshot is not None:
        payload["contract_snapshot"] = snapshot
    (run_dir / "run_config.json").write_text(json.dumps(payload), encoding="utf-8")
    return run_dir


def _mujoco_cfg() -> Any:
    return OmegaConf.create(
        {
            "training": {"sim_backend": "mujoco"},
            "algo": {
                "obs_groups": {"actor": ["actor"]},
                "empirical_normalization": False,
                "policy": {
                    "actor_hidden_dims": [512, 256, 128],
                    "critic_hidden_dims": [512, 256, 128],
                },
            },
            "env": {
                "control_config": {"action_scale": 0.25, "simulate_action_latency": False},
                "ctrl_dt": 0.01,
            },
            "reward": {"scales": {"tracking_lin_vel": 2.0}, "max_tilt_deg": 25.0},
        }
    )


def test_field_lists_are_disjoint():
    deny, warn, allow = set(DENYLIST), set(WARNING_LIST), set(ALLOWLIST)
    assert deny.isdisjoint(warn)
    assert deny.isdisjoint(allow)
    assert warn.isdisjoint(allow)


def test_extract_snapshot_includes_only_present_contract_fields():
    snapshot = extract_contract_snapshot(_mujoco_cfg())
    # Present DENY/WARN fields are captured...
    assert snapshot["env.control_config.action_scale"] == 0.25
    assert snapshot["algo.obs_groups"] == {"actor": ["actor"]}
    assert snapshot["algo.empirical_normalization"] is False
    assert snapshot["reward.scales"] == {"tracking_lin_vel": 2.0}
    # ...ALLOWLIST fields are excluded...
    assert "training.sim_backend" not in snapshot
    # ...and absent fields are omitted (never stored as None).
    assert "algo.obs_normalization" not in snapshot
    assert "env.sampling_mode" not in snapshot
    assert "reward.base_height_target" not in snapshot


def test_snapshot_json_round_trips():
    snapshot = extract_contract_snapshot(_mujoco_cfg())
    assert json.loads(json.dumps(snapshot)) == snapshot


def test_matching_contract_returns_same_cfg(tmp_path):
    _write_sidecar(tmp_path, extract_contract_snapshot(_mujoco_cfg()))
    target = _mujoco_cfg()
    assert resolve_sim2sim_config(tmp_path, target) is target


def test_denylist_mismatch_raises_with_field_in_message(tmp_path):
    _write_sidecar(tmp_path, extract_contract_snapshot(_mujoco_cfg()))
    target = _mujoco_cfg()
    target.env.control_config.action_scale = 0.5
    with pytest.raises(CrossBackendIncompatibleError) as excinfo:
        resolve_sim2sim_config(tmp_path, target)
    msg = str(excinfo.value)
    assert "action_scale" in msg
    assert "0.25" in msg
    assert "0.5" in msg


def test_denylist_nested_dict_mismatch_raises(tmp_path):
    _write_sidecar(tmp_path, extract_contract_snapshot(_mujoco_cfg()))
    target = _mujoco_cfg()
    target.algo.obs_groups = {"actor": ["policy"], "critic": ["critic"]}
    with pytest.raises(CrossBackendIncompatibleError):
        resolve_sim2sim_config(tmp_path, target)


def test_warning_mismatch_does_not_raise(tmp_path, capsys):
    _write_sidecar(tmp_path, extract_contract_snapshot(_mujoco_cfg()))
    target = _mujoco_cfg()
    target.env.ctrl_dt = 0.02
    assert resolve_sim2sim_config(tmp_path, target) is target
    assert "[sim2sim] WARNING override env.ctrl_dt" in capsys.readouterr().out


def test_missing_snapshot_falls_back(tmp_path, capsys):
    _write_sidecar(tmp_path, None)  # run_config.json without contract_snapshot
    target = _mujoco_cfg()
    assert resolve_sim2sim_config(tmp_path, target) is target
    assert "no contract_snapshot" in capsys.readouterr().out


def test_missing_file_falls_back(tmp_path, capsys):
    target = _mujoco_cfg()
    assert resolve_sim2sim_config(tmp_path, target) is target  # no run_config.json
    assert "no contract_snapshot" in capsys.readouterr().out


def test_corrupt_sidecar_falls_back(tmp_path, capsys):
    (tmp_path / "run_config.json").write_text("{ not valid json", encoding="utf-8")
    target = _mujoco_cfg()
    assert resolve_sim2sim_config(tmp_path, target) is target
    assert "no contract_snapshot" in capsys.readouterr().out


def test_none_source_returns_none(capsys):
    assert resolve_sim2sim_config(None, _mujoco_cfg()) is None
    assert "no source run dir" in capsys.readouterr().out


def test_target_missing_path_is_skipped(tmp_path):
    # Snapshot was taken from a PPO run (empirical_normalization); the off-policy
    # target only has obs_normalization, so the snapshot path is simply skipped.
    _write_sidecar(tmp_path, {"algo.empirical_normalization": True})
    target = OmegaConf.create({"algo": {"obs_normalization": True}, "env": {}})
    assert resolve_sim2sim_config(tmp_path, target) is target


def test_non_strict_downgrades_denial_to_warning(tmp_path, capsys):
    _write_sidecar(tmp_path, extract_contract_snapshot(_mujoco_cfg()))
    target = _mujoco_cfg()
    target.env.control_config.action_scale = 0.5
    assert resolve_sim2sim_config(tmp_path, target, strict=False) is target
    assert "action_scale" in capsys.readouterr().out


def test_empirical_normalization_on_vs_off_raises(tmp_path):
    # The real case: trained with obs normalization ON, played with it OFF.
    _write_sidecar(tmp_path, {"algo.empirical_normalization": True})
    target = OmegaConf.create({"algo": {"empirical_normalization": False}})
    with pytest.raises(CrossBackendIncompatibleError):
        resolve_sim2sim_config(tmp_path, target)


def test_int_and_float_compare_equal(tmp_path):
    _write_sidecar(tmp_path, {"env.ctrl_dt": 0})
    target = OmegaConf.create({"env": {"ctrl_dt": 0.0}})
    assert resolve_sim2sim_config(tmp_path, target) is target


def test_action_scale_list_form(tmp_path):
    _write_sidecar(tmp_path, {"env.control_config.action_scale": [0.5, 0.5]})
    ok = OmegaConf.create({"env": {"control_config": {"action_scale": [0.5, 0.5]}}})
    assert resolve_sim2sim_config(tmp_path, ok) is ok
    bad = OmegaConf.create({"env": {"control_config": {"action_scale": 0.25}}})
    with pytest.raises(CrossBackendIncompatibleError):
        resolve_sim2sim_config(tmp_path, bad)


# --- issue #579 P2: env-structural asymmetric-presence fail-closed -----------------


def test_env_structural_denylist_is_the_env_subset():
    assert ENV_STRUCTURAL_DENYLIST == ["env.control_config.action_scale", "env.sampling_mode"]
    assert set(ENV_STRUCTURAL_DENYLIST) <= set(DENYLIST)


def test_env_field_present_in_source_absent_in_target_raises(tmp_path):
    # Forward asymmetry: source (mujoco) sets action_scale, target (motrix) omits it
    # and would fall back to a differing env default. Fail closed instead of skipping.
    _write_sidecar(tmp_path, {"env.control_config.action_scale": [0.5, 0.5, 0.5]})
    target = OmegaConf.create({"env": {"control_config": {}}})  # no action_scale set
    with pytest.raises(CrossBackendIncompatibleError) as excinfo:
        resolve_sim2sim_config(tmp_path, target)
    msg = str(excinfo.value)
    assert "action_scale" in msg
    assert "target=<absent>" in msg


def test_env_field_present_in_target_absent_in_source_raises(tmp_path):
    # Reverse asymmetry: the trained run omitted sampling_mode (used the env default),
    # the target sets it explicitly. Still unverifiable -> fail closed.
    _write_sidecar(tmp_path, {"algo.empirical_normalization": False})
    target = OmegaConf.create(
        {"algo": {"empirical_normalization": False}, "env": {"sampling_mode": "adaptive"}}
    )
    with pytest.raises(CrossBackendIncompatibleError) as excinfo:
        resolve_sim2sim_config(tmp_path, target)
    msg = str(excinfo.value)
    assert "sampling_mode" in msg
    assert "source=<absent>" in msg


def test_env_field_symmetric_absence_does_not_raise(tmp_path):
    # Neither side sets the env structural field -> both use the same env default -> ok.
    _write_sidecar(tmp_path, {"algo.empirical_normalization": False})
    target = OmegaConf.create({"algo": {"empirical_normalization": False}, "env": {}})
    assert resolve_sim2sim_config(tmp_path, target) is target


def test_algo_field_absent_in_target_still_skipped_not_fail_closed(tmp_path):
    # Regression: algo-specific fields keep the cross-algo skip and must NOT fail closed
    # the way env structural fields now do.
    _write_sidecar(tmp_path, {"algo.empirical_normalization": True})
    target = OmegaConf.create({"algo": {"obs_normalization": True}, "env": {}})
    assert resolve_sim2sim_config(tmp_path, target) is target


def test_env_field_fail_closed_even_if_default_might_match(tmp_path):
    # Fail-closed semantics: the guard cannot resolve the omitted side's runtime
    # default, so it raises even when that default could happen to equal the explicit
    # value. The fix is a one-line explicit declaration in the target YAML.
    _write_sidecar(tmp_path, {"env.control_config.action_scale": 0.25})
    target = OmegaConf.create({"env": {"control_config": {}}})
    with pytest.raises(CrossBackendIncompatibleError):
        resolve_sim2sim_config(tmp_path, target)


# --- issue #579: play-time runtime dimension guard (policy_load_dim_guard) ----------


def test_dim_guard_passes_through_on_success():
    # No error inside the block -> the guard is a no-op.
    ran = False
    with policy_load_dim_guard(env_obs_dim=10, env_action_dim=3, algo_name="ppo"):
        ran = True
    assert ran


def test_dim_guard_translates_torch_size_mismatch():
    # torch raises RuntimeError("size mismatch ...") on a shape-incompatible load.
    err = "Error(s) in loading state_dict for ActorCritic:\n\tsize mismatch for actor.0.weight"
    with pytest.raises(CrossBackendIncompatibleError) as excinfo:
        with policy_load_dim_guard(env_obs_dim=42, env_action_dim=12, algo_name="ppo"):
            raise RuntimeError(err)
    msg = str(excinfo.value)
    assert "42" in msg and "12" in msg  # env dims surfaced
    assert "audit_sim2sim_contracts" in msg  # actionable pointer
    assert isinstance(excinfo.value.__cause__, RuntimeError)  # original chained


def test_dim_guard_translates_mlx_shape_valueerror():
    # mlx load_weights(strict=True) raises ValueError mentioning the expected shape.
    with pytest.raises(CrossBackendIncompatibleError):
        with policy_load_dim_guard(env_obs_dim=8, env_action_dim=2, algo_name="ppo"):
            raise ValueError("Expected shape (256, 8) but received (256, 11)")


def test_dim_guard_reraises_unrelated_errors_unchanged():
    # A non-dimension load failure must propagate as-is (not masked as a sim2sim error).
    with pytest.raises(RuntimeError) as excinfo:
        with policy_load_dim_guard(env_obs_dim=10, env_action_dim=3):
            raise RuntimeError("CUDA out of memory")
    assert not isinstance(excinfo.value, CrossBackendIncompatibleError)


def test_dim_guard_does_not_swallow_keyerror():
    # A missing checkpoint key is not a dim mismatch; it must surface unchanged.
    with pytest.raises(KeyError):
        with policy_load_dim_guard(env_obs_dim=10, env_action_dim=3):
            raise KeyError("actor")


# --- issue #579: Sim2SimConfigResolver class facade + user-level bypass -------------


def test_resolver_class_exposes_field_lists():
    # The RFC names this class; it re-exposes the module constants (single source).
    assert Sim2SimConfigResolver.DENYLIST is DENYLIST
    assert Sim2SimConfigResolver.WARNING_LIST is WARNING_LIST
    assert Sim2SimConfigResolver.ALLOWLIST is ALLOWLIST
    assert Sim2SimConfigResolver.ENV_STRUCTURAL_DENYLIST is ENV_STRUCTURAL_DENYLIST


def test_resolver_class_resolve_delegates_and_raises(tmp_path):
    _write_sidecar(tmp_path, extract_contract_snapshot(_mujoco_cfg()))
    target = _mujoco_cfg()
    target.env.control_config.action_scale = 0.5
    with pytest.raises(CrossBackendIncompatibleError):
        Sim2SimConfigResolver.resolve(tmp_path, target)


def test_resolver_class_strict_false_is_user_bypass(tmp_path, capsys):
    # training.sim2sim_strict=false maps to strict=False: a DENYLIST denial becomes a
    # warning and play proceeds with the target cfg (the load-time dim guard still bites).
    _write_sidecar(tmp_path, extract_contract_snapshot(_mujoco_cfg()))
    target = _mujoco_cfg()
    target.env.control_config.action_scale = 0.5
    assert Sim2SimConfigResolver.resolve(tmp_path, target, strict=False) is target
    assert "action_scale" in capsys.readouterr().out


def test_resolver_class_extract_and_dim_guard_delegate():
    snap = Sim2SimConfigResolver.extract_snapshot(_mujoco_cfg())
    assert snap["env.control_config.action_scale"] == 0.25
    with pytest.raises(CrossBackendIncompatibleError):
        with Sim2SimConfigResolver.load_dim_guard(env_obs_dim=5, env_action_dim=2):
            raise RuntimeError("size mismatch for actor.0.weight")


def _compose_task(task: str) -> Any:
    conf_dir = str(Path(__file__).resolve().parents[2] / "conf" / "ppo")
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=conf_dir, version_base="1.3"):
        return compose("config", overrides=[f"task={task}"])


def test_g1_walk_flat_mujoco_inherits_base_contract():
    # The MuJoCo owner must inherit the shared contract from base.yaml verbatim.
    mujoco = _compose_task("g1_walk_flat/mujoco")
    assert OmegaConf.select(mujoco, "env.control_config.action_scale") == 0.25
    assert OmegaConf.select(mujoco, "algo.empirical_normalization") is False
    assert OmegaConf.select(mujoco, "algo.obs_groups.actor") == ["actor"]


def test_g1_walk_flat_cross_backend_play_is_guarded(tmp_path):
    # Motrix intentionally overrides contract fields for backend-specific tuning,
    # so a MuJoCo-trained policy is not sim2sim-transferable to Motrix: the guard
    # must surface that on the real composed configs (issue #579 by-design).
    snapshot = extract_contract_snapshot(_compose_task("g1_walk_flat/mujoco"))
    (tmp_path / "run_config.json").write_text(
        json.dumps({"contract_snapshot": snapshot}), encoding="utf-8"
    )
    motrix = _compose_task("g1_walk_flat/motrix")
    with pytest.raises(CrossBackendIncompatibleError):
        resolve_sim2sim_config(tmp_path, motrix)
