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

from unilab.utils.sim2sim import (
    ALLOWLIST,
    DENYLIST,
    ENV_STRUCTURAL_DENYLIST,
    WARNING_LIST,
    CrossBackendIncompatibleError,
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
                "actor": {
                    "hidden_dims": [512, 256, 128],
                    "obs_normalization": False,
                },
                "critic": {
                    "hidden_dims": [512, 256, 128],
                },
            },
            "env": {
                "control_config": {"action_scale": 0.25, "simulate_action_latency": False},
                "ctrl_dt": 0.01,
            },
            "reward": {"scales": {"tracking_lin_vel": 2.0}, "max_tilt_deg": 25.0},
        }
    )


def _manager_cfg() -> Any:
    return OmegaConf.create(
        {
            "training": {"sim_backend": "mujoco"},
            "algo": {"obs_groups": {"actor": ["actor"]}},
            "env": {
                "observations": {
                    "policy": {
                        "_target_": "unilab.managers.ObservationGroupCfg",
                        "terms": {
                            "joint_pos": {
                                "_target_": "unilab.managers.ObservationTermCfg",
                                "func": "unilab.envs.mdp.joint_pos_rel",
                            }
                        },
                    },
                    "critic": {
                        "_target_": "unilab.managers.ObservationGroupCfg",
                        "terms": {
                            "joint_pos": {
                                "_target_": "unilab.managers.ObservationTermCfg",
                                "func": "unilab.envs.mdp.joint_pos_rel",
                            }
                        },
                    },
                },
                "actions": {
                    "joint_pos": {
                        "_target_": "unilab.envs.mdp.JointPositionActionCfg",
                        "entity_name": "robot",
                        "actuator_names": [".*"],
                        "scale": 0.25,
                    }
                },
                "policy_observation_group": "policy",
                "critic_observation_group": "critic",
            },
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
    assert snapshot["algo.actor.obs_normalization"] is False
    assert snapshot["reward.scales"] == {"tracking_lin_vel": 2.0}
    # ...ALLOWLIST fields are excluded...
    assert "training.sim_backend" not in snapshot
    # ...and absent fields are omitted (never stored as None).
    assert "algo.obs_normalization" not in snapshot
    assert "env.commands.motion.params.sampling_mode" not in snapshot
    assert "reward.base_height_target" not in snapshot


def test_snapshot_json_round_trips():
    snapshot = extract_contract_snapshot(_mujoco_cfg())
    assert json.loads(json.dumps(snapshot)) == snapshot


def test_extract_snapshot_reads_only_canonical_paths() -> None:
    retired_source = OmegaConf.create(
        {
            "algo": {
                "policy": {
                    "actor_hidden_dims": [512, 256, 128],
                    "critic_hidden_dims": [256, 128],
                }
            }
        }
    )
    canonical_source = OmegaConf.create(
        {
            "algo": {
                "actor": {"hidden_dims": [512, 256, 128]},
                "critic": {"hidden_dims": [256, 128]},
            }
        }
    )
    expected = {
        "algo.actor.hidden_dims": [512, 256, 128],
        "algo.critic.hidden_dims": [256, 128],
    }

    # Retired pre-canonical config paths are no longer canonicalized on
    # extraction; only canonical keys are ever read or written.
    assert extract_contract_snapshot(retired_source) == {}
    assert extract_contract_snapshot(canonical_source) == expected


def test_extract_snapshot_captures_manager_policy_io_contract() -> None:
    cfg = _manager_cfg()

    snapshot = extract_contract_snapshot(cfg)

    assert snapshot["env.actions"]["joint_pos"]["scale"] == pytest.approx(0.25)
    assert snapshot["env.observations"]["policy"]["terms"]["joint_pos"]["func"] == (
        "unilab.envs.mdp.joint_pos_rel"
    )
    assert snapshot["env.policy_observation_group"] == "policy"
    assert snapshot["env.critic_observation_group"] == "critic"
    assert "env.control_config.action_scale" not in snapshot


def test_matching_manager_policy_io_contract_passes(tmp_path: Path) -> None:
    _write_sidecar(tmp_path, extract_contract_snapshot(_manager_cfg()))
    target = _manager_cfg()

    assert resolve_sim2sim_config(tmp_path, target) is target


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("env.actions.joint_pos.scale", 0.5),
        ("env.observations.policy.terms.joint_pos.func", "unilab.envs.mdp.joint_vel_rel"),
        ("env.policy_observation_group", "critic"),
        ("env.critic_observation_group", "policy"),
    ],
)
def test_manager_policy_io_mismatch_fails_closed(tmp_path: Path, path: str, value: object) -> None:
    source = _manager_cfg()
    _write_sidecar(tmp_path, extract_contract_snapshot(source))
    target = _manager_cfg()
    OmegaConf.update(target, path, value, merge=False)

    with pytest.raises(CrossBackendIncompatibleError, match="env\\."):
        resolve_sim2sim_config(tmp_path, target)


def test_matching_contract_returns_same_cfg(tmp_path):
    _write_sidecar(tmp_path, extract_contract_snapshot(_mujoco_cfg()))
    target = _mujoco_cfg()
    assert resolve_sim2sim_config(tmp_path, target) is target


def test_visualization_only_action_fields_are_exempt(tmp_path):
    # log_goal/show_rays only drive playback debug overlays; toggling them for
    # interactive play is not a policy-defining change and must not fail the
    # env.actions DENYLIST comparison.
    source = _manager_cfg()
    source.env.actions.joint_pos.log_goal = False
    source.env.actions.joint_pos.show_rays = False
    _write_sidecar(tmp_path, extract_contract_snapshot(source))

    target = _manager_cfg()
    target.env.actions.joint_pos.log_goal = True
    target.env.actions.joint_pos.show_rays = True

    assert resolve_sim2sim_config(tmp_path, target) is target


def test_visualization_only_fields_do_not_mask_policy_mismatches(tmp_path):
    # Stripping visualization fields must not hide a real policy-I/O difference
    # in the same action mapping.
    source = _manager_cfg()
    source.env.actions.joint_pos.log_goal = False
    _write_sidecar(tmp_path, extract_contract_snapshot(source))
    target = _manager_cfg()
    target.env.actions.joint_pos.log_goal = True
    target.env.actions.joint_pos.scale = 0.5

    with pytest.raises(CrossBackendIncompatibleError, match="env\\.actions"):
        resolve_sim2sim_config(tmp_path, target)


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


def test_ctrl_dt_mismatch_is_denied(tmp_path):
    _write_sidecar(tmp_path, extract_contract_snapshot(_mujoco_cfg()))
    target = _mujoco_cfg()
    target.env.ctrl_dt = 0.02
    with pytest.raises(CrossBackendIncompatibleError) as excinfo:
        resolve_sim2sim_config(tmp_path, target)
    assert "env.ctrl_dt" in str(excinfo.value)


def test_command_vel_limit_mismatch_is_denied(tmp_path):
    _write_sidecar(tmp_path, {"env.commands.vel_limit": 1.5})
    target = OmegaConf.create({"env": {"commands": {"vel_limit": 2.0}}})

    with pytest.raises(CrossBackendIncompatibleError) as excinfo:
        resolve_sim2sim_config(tmp_path, target)
    assert "env.commands.vel_limit" in str(excinfo.value)


def test_missing_snapshot_fails_closed(tmp_path):
    _write_sidecar(tmp_path, None)  # run_config.json without contract_snapshot
    target = _mujoco_cfg()
    with pytest.raises(CrossBackendIncompatibleError) as excinfo:
        resolve_sim2sim_config(tmp_path, target)
    msg = str(excinfo.value)
    assert "contract_snapshot" in msg
    assert "Minimum supported format" in msg


def test_missing_file_fails_closed(tmp_path):
    target = _mujoco_cfg()
    with pytest.raises(CrossBackendIncompatibleError) as excinfo:
        resolve_sim2sim_config(tmp_path, target)  # no run_config.json
    msg = str(excinfo.value)
    assert "run_config.json" in msg
    assert "Minimum supported format" in msg


def test_missing_snapshot_fails_closed_even_non_strict(tmp_path):
    # Snapshot-integrity failures are not DENYLIST mismatches: strict=False does
    # not downgrade them.
    _write_sidecar(tmp_path, None)
    with pytest.raises(CrossBackendIncompatibleError):
        resolve_sim2sim_config(tmp_path, _mujoco_cfg(), strict=False)


def test_corrupt_sidecar_fails_closed(tmp_path):
    (tmp_path / "run_config.json").write_text("{ not valid json", encoding="utf-8")
    target = _mujoco_cfg()
    with pytest.raises(CrossBackendIncompatibleError) as excinfo:
        resolve_sim2sim_config(tmp_path, target)
    assert "not valid JSON" in str(excinfo.value)


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param("[]", id="non-mapping-top-level"),
        pytest.param(json.dumps({"contract_snapshot": None}), id="null-snapshot"),
        pytest.param(json.dumps({"contract_snapshot": [512, 256]}), id="non-mapping-snapshot"),
    ],
)
def test_malformed_snapshot_fails_closed(tmp_path, payload):
    (tmp_path / "run_config.json").write_text(payload, encoding="utf-8")
    target = _mujoco_cfg()

    with pytest.raises(CrossBackendIncompatibleError, match="contract_snapshot|JSON object"):
        resolve_sim2sim_config(tmp_path, target)


def test_none_source_returns_none(capsys):
    assert resolve_sim2sim_config(None, _mujoco_cfg()) is None
    assert "no source run dir" in capsys.readouterr().out


def test_target_missing_path_is_skipped(tmp_path):
    # Snapshot was taken from a PPO run (actor obs normalization); the off-policy
    # target only has obs_normalization, so the snapshot path is simply skipped.
    _write_sidecar(tmp_path, {"algo.actor.obs_normalization": True})
    target = OmegaConf.create({"algo": {"obs_normalization": True}, "env": {}})
    assert resolve_sim2sim_config(tmp_path, target) is target


def test_non_strict_downgrades_denial_to_warning(tmp_path, capsys):
    _write_sidecar(tmp_path, extract_contract_snapshot(_mujoco_cfg()))
    target = _mujoco_cfg()
    target.env.control_config.action_scale = 0.5
    assert resolve_sim2sim_config(tmp_path, target, strict=False) is target
    assert "action_scale" in capsys.readouterr().out


def test_actor_obs_normalization_on_vs_off_raises(tmp_path):
    # The real case: trained with obs normalization ON, played with it OFF.
    _write_sidecar(tmp_path, {"algo.actor.obs_normalization": True})
    target = OmegaConf.create({"algo": {"actor": {"obs_normalization": False}}})
    with pytest.raises(CrossBackendIncompatibleError):
        resolve_sim2sim_config(tmp_path, target)


@pytest.mark.parametrize("target_value", [True, False])
def test_retired_empirical_normalization_key_fails_closed(tmp_path, target_value):
    # Retired pre-canonical snapshot keys are unsupported input: they fail closed
    # before any comparison, whether or not the target would have matched.
    _write_sidecar(tmp_path, {"algo.empirical_normalization": True})
    target = OmegaConf.create({"algo": {"actor": {"obs_normalization": target_value}}, "env": {}})
    with pytest.raises(CrossBackendIncompatibleError) as excinfo:
        resolve_sim2sim_config(tmp_path, target)
    assert "algo.empirical_normalization" in str(excinfo.value)


@pytest.mark.parametrize(
    "retired_key",
    [
        "algo.policy.actor_hidden_dims",
        "algo.policy.critic_hidden_dims",
    ],
)
def test_retired_hidden_dims_keys_fail_closed(tmp_path, retired_key):
    _write_sidecar(tmp_path, {retired_key: [512, 256, 128]})
    target = OmegaConf.create({"algo": {}, "env": {}})

    with pytest.raises(CrossBackendIncompatibleError) as excinfo:
        resolve_sim2sim_config(tmp_path, target)
    assert retired_key in str(excinfo.value)


def test_unknown_snapshot_key_fails_closed(tmp_path):
    # Keys outside the current contract schema (newer/older writers) are reported
    # as an unsupported format instead of being silently skipped.
    _write_sidecar(tmp_path, {"algo.future_field": 1})
    target = OmegaConf.create({"algo": {}, "env": {}})

    with pytest.raises(CrossBackendIncompatibleError) as excinfo:
        resolve_sim2sim_config(tmp_path, target)
    msg = str(excinfo.value)
    assert "algo.future_field" in msg
    assert "unsupported" in msg.lower() or "outside the current contract schema" in msg


def test_retired_alias_tables_stay_removed():
    # Regression guard for the retired-key removal: the resolver must not
    # reintroduce legacy alias/fallback tables.
    import unilab.utils.sim2sim as sim2sim_module

    assert not hasattr(sim2sim_module, "LEGACY_PATH_ALIASES")
    assert not hasattr(sim2sim_module, "_CANONICAL_PATH_FALLBACKS")


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


# --- env-structural asymmetric-presence fail-closed --------------------------------


def test_env_structural_denylist_is_the_env_subset():
    assert ENV_STRUCTURAL_DENYLIST == [
        "env.control_config.action_scale",
        "env.observations",
        "env.actions",
        "env.policy_observation_group",
        "env.critic_observation_group",
        "env.ctrl_dt",
        "env.commands.vel_limit",
        "env.commands.motion.params.sampling_mode",
    ]
    assert set(ENV_STRUCTURAL_DENYLIST) <= set(DENYLIST)


def test_env_field_present_in_source_absent_in_target_raises(tmp_path):
    # Forward asymmetry: the source sets action_scale and the target omits it,
    # falling back to a differing env default. Fail closed instead of skipping.
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
    _write_sidecar(tmp_path, {"algo.actor.obs_normalization": False})
    target = OmegaConf.create(
        {
            "algo": {"actor": {"obs_normalization": False}},
            "env": {"commands": {"motion": {"params": {"sampling_mode": "adaptive"}}}},
        }
    )
    with pytest.raises(CrossBackendIncompatibleError) as excinfo:
        resolve_sim2sim_config(tmp_path, target)
    msg = str(excinfo.value)
    assert "sampling_mode" in msg
    assert "source=<absent>" in msg


def test_retired_sampling_mode_key_fails_closed(tmp_path) -> None:
    _write_sidecar(tmp_path, {"env.sampling_mode": "adaptive"})
    target = OmegaConf.create(
        {"env": {"commands": {"motion": {"params": {"sampling_mode": "adaptive"}}}}}
    )

    with pytest.raises(CrossBackendIncompatibleError) as excinfo:
        resolve_sim2sim_config(tmp_path, target)
    assert "env.sampling_mode" in str(excinfo.value)


def test_retired_config_sampling_mode_path_is_not_snapshotted() -> None:
    # The retired pre-canonical config path is no longer read during extraction;
    # only the canonical env.commands.motion.params.sampling_mode is captured.
    cfg = OmegaConf.create({"env": {"sampling_mode": "adaptive"}})

    assert extract_contract_snapshot(cfg) == {}


def test_env_field_symmetric_absence_does_not_raise(tmp_path):
    # Neither side sets the env structural field -> both use the same env default -> ok.
    _write_sidecar(tmp_path, {"algo.actor.obs_normalization": False})
    target = OmegaConf.create({"algo": {"actor": {"obs_normalization": False}}, "env": {}})
    assert resolve_sim2sim_config(tmp_path, target) is target


def test_algo_field_absent_in_target_still_skipped_not_fail_closed(tmp_path):
    # Regression: algo-specific fields keep the cross-algo skip and must NOT fail closed
    # the way env structural fields now do.
    _write_sidecar(tmp_path, {"algo.actor.obs_normalization": True})
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


# --- play-time runtime dimension guard ---------------------------------------------


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


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("CUDA out of memory"),
        RuntimeError("Expected all tensors to be on the same device, but found cuda:0 and cpu"),
        RuntimeError("expected scalar type Float but found Half"),
        RuntimeError("shape '[12, 4]' is invalid for input of size 36"),
        RuntimeError("no kernel image is available for execution on the device"),
        RuntimeError("copying a param failed: expected all tensors to use the same dtype"),
        ValueError("Expected input batch_size (32) to match target batch_size (16)"),
    ],
)
def test_dim_guard_reraises_unrelated_errors_unchanged(error):
    with pytest.raises(type(error)) as excinfo:
        with policy_load_dim_guard(env_obs_dim=10, env_action_dim=3):
            raise error
    assert excinfo.value is error


def test_dim_guard_does_not_swallow_keyerror():
    # A missing checkpoint key is not a dim mismatch; it must surface unchanged.
    with pytest.raises(KeyError):
        with policy_load_dim_guard(env_obs_dim=10, env_action_dim=3):
            raise KeyError("actor")


def _compose_task(task: str) -> Any:
    conf_dir = str(Path(__file__).resolve().parents[2] / "src" / "unilab" / "conf" / "ppo")
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=conf_dir, version_base="1.3"):
        return compose("config", overrides=[f"task={task}"])


def test_g1_walk_flat_mujoco_inherits_base_contract():
    # The MuJoCo owner inherits the full contract from the shared base owner.
    mujoco = _compose_task("g1_walk_flat/mujoco")
    assert OmegaConf.select(mujoco, "env.actions.joint_pos.scale") == 0.25
    assert OmegaConf.select(mujoco, "algo.actor.obs_normalization") is False
    assert OmegaConf.select(mujoco, "algo.obs_groups.actor") == ["actor"]


def test_g1_walk_flat_mujoco_to_genesis_play_passes_guard(tmp_path):
    # The genesis owner keeps DENYLIST parity with the MuJoCo owner, so a
    # MuJoCo-trained checkpoint passes the contract guard for genesis play.
    snapshot = extract_contract_snapshot(_compose_task("g1_walk_flat/mujoco"))
    (tmp_path / "run_config.json").write_text(
        json.dumps({"contract_snapshot": snapshot}), encoding="utf-8"
    )
    genesis = _compose_task("g1_walk_flat/genesis")
    assert OmegaConf.select(genesis, "training.sim_backend") == "genesis"
    assert resolve_sim2sim_config(tmp_path, genesis) is genesis
