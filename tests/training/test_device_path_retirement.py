"""Retired device-resident mjwarp requests must fail with explicit diagnostics.

Issue #886 (Phase 0 scope reset): stale owner YAMLs, composed configs,
checkpoints, resume, and playback requests that reference the removed
production device path (``mjwarp_device_v1`` / ``device_resident`` /
``unilab.training.rsl_rl_device``) must raise ``RetiredDevicePathError``
instead of an obscure import/attribute/shape error.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from unilab.training.retirement import (
    RetiredDevicePathError,
    check_retired_checkpoint,
    check_retired_config,
    check_retired_task_overrides,
    check_retired_task_owner,
)
from unilab.training.sim2sim import resolve_sim2sim_config


def _write_run_config(run_dir: Path, payload: dict) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    run_config = run_dir / "run_config.json"
    run_config.write_text(json.dumps(payload), encoding="utf-8")
    (run_dir / "model_0.pt").touch()
    return run_config


# ---------------------------------------------------------------------------
# Task owner
# ---------------------------------------------------------------------------


def test_retired_task_owner_hydra_path_form_raises() -> None:
    with pytest.raises(RetiredDevicePathError, match="issue #886"):
        check_retired_task_owner("g1_walk_flat/mjwarp")


def test_retired_task_owner_composed_name_form_raises() -> None:
    # train scripts see the composed training.task_name/sim_backend pair.
    with pytest.raises(RetiredDevicePathError, match="g1_walk_flat/mujoco"):
        check_retired_task_owner("G1WalkFlat/mjwarp")


@pytest.mark.parametrize(
    "task",
    ["g1_walk_flat/mujoco", "G1WalkFlat/mujoco", "go2_joystick_flat/motrix"],
)
def test_active_task_owners_pass(task: str) -> None:
    check_retired_task_owner(task)


def test_retired_task_override_scan_raises() -> None:
    with pytest.raises(RetiredDevicePathError):
        check_retired_task_overrides(["task=g1_walk_flat/mjwarp", "algo.num_envs=8"])
    check_retired_task_overrides(["task=g1_walk_flat/mujoco", "algo.num_envs=8"])


# ---------------------------------------------------------------------------
# Composed config markers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("training.execution_profile", "device_resident"),
        ("algo.runtime_impl", "mjwarp_device_v1"),
        (
            "algo.runtime_resolver",
            "unilab.training.rsl_rl_device:resolve_mjwarp_device_ppo_runtime",
        ),
    ],
)
def test_retired_config_marker_raises(path: str, value: str) -> None:
    keys = path.split(".")
    cfg = OmegaConf.create({keys[0]: {keys[1]: value}})
    with pytest.raises(RetiredDevicePathError, match="issue #886"):
        check_retired_config(cfg)


def test_retired_config_entrypoints_block_raises() -> None:
    cfg = OmegaConf.create({"training": {"sim_backend": "mujoco"}, "entrypoints": {"routes": {}}})
    with pytest.raises(RetiredDevicePathError, match="entrypoints"):
        check_retired_config(cfg)


def test_clean_config_passes() -> None:
    cfg = OmegaConf.create(
        {
            "training": {"task_name": "G1WalkFlat", "sim_backend": "mujoco"},
            "algo": {"obs_groups": {"actor": ["actor"]}},
        }
    )
    check_retired_config(cfg)


def test_plain_mapping_config_is_supported() -> None:
    with pytest.raises(RetiredDevicePathError):
        check_retired_config({"algo": {"runtime_impl": "mjwarp_device_v1"}})
    check_retired_config({"training": {"sim_backend": "mujoco"}})


# ---------------------------------------------------------------------------
# Checkpoint / run_config markers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "payload",
    [
        # runtime_impl recorded in the algo config subtree
        {"config": {"algo": {"runtime_impl": "mjwarp_device_v1"}}},
        # execution_profile recorded in the training subtree
        {"config": {"training": {"execution_profile": "device_resident"}}},
        # resolver import path recorded in the algo subtree
        {
            "config": {
                "algo": {
                    "runtime_resolver": (
                        "unilab.training.rsl_rl_device:resolve_mjwarp_device_ppo_runtime"
                    )
                }
            }
        },
        # device ABI snapshot recorded in the contract snapshot
        {
            "contract_snapshot": {
                "manager.policy_abi": {
                    "executor_key": "device.cuda.0.v1",
                    "execution_profile": "device_resident",
                }
            }
        },
    ],
)
def test_checkpoint_from_retired_run_raises(tmp_path: Path, payload: dict) -> None:
    _write_run_config(tmp_path, payload)
    with pytest.raises(RetiredDevicePathError, match="issue #886"):
        check_retired_checkpoint(tmp_path / "model_0.pt")


def test_checkpoint_without_run_config_passes_silently(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model_0.pt"
    checkpoint.touch()
    check_retired_checkpoint(checkpoint)


def test_checkpoint_from_clean_run_passes(tmp_path: Path) -> None:
    _write_run_config(
        tmp_path,
        {
            "config": {"training": {"sim_backend": "mujoco"}},
            "contract_snapshot": {"algo.obs_groups": {"actor": ["actor"]}},
        },
    )
    check_retired_checkpoint(tmp_path / "model_0.pt")


def test_run_config_found_in_checkpoint_parent_directory(tmp_path: Path) -> None:
    run_dir = tmp_path / "logs" / "run"
    _write_run_config(run_dir, {"config": {"algo": {"runtime_impl": "mjwarp_device_v1"}}})
    nested = run_dir / "nested" / "model_0.pt"
    nested.parent.mkdir(parents=True)
    nested.touch()
    # run_config.json lives two levels up; still within the ancestor search.
    with pytest.raises(RetiredDevicePathError):
        check_retired_checkpoint(nested)


# ---------------------------------------------------------------------------
# sim2sim resolver path
# ---------------------------------------------------------------------------


def test_sim2sim_resolver_raises_retired_error_for_device_run(tmp_path: Path) -> None:
    _write_run_config(
        tmp_path,
        {
            "contract_snapshot": {
                "manager.policy_abi": {
                    "executor_key": "device.cuda.0.v1",
                    "execution_profile": "device_resident",
                }
            }
        },
    )
    target_cfg = OmegaConf.create({"training": {"sim_backend": "mujoco"}})
    with pytest.raises(RetiredDevicePathError, match="retired"):
        resolve_sim2sim_config(tmp_path, target_cfg, algo_name="ppo")


def test_sim2sim_resolver_passes_for_clean_run(tmp_path: Path) -> None:
    _write_run_config(
        tmp_path,
        {"contract_snapshot": {"algo.obs_groups": {"actor": ["actor"]}}},
    )
    target_cfg = OmegaConf.create({"algo": {"obs_groups": {"actor": ["actor"]}}})
    assert resolve_sim2sim_config(tmp_path, target_cfg, algo_name="ppo") is not None
