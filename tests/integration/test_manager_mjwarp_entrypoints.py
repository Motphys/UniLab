"""Production entrypoint matrix for the promoted managed MuJoCo/MJWarp rollout owner."""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import torch

from unilab.base.backend.mjwarp.dependencies import load_mjwarp_dependencies

ROOT_DIR = Path(__file__).resolve().parents[2]
_NUM_ENVS = 128
_STEPS_PER_ENV = 2
_ENTRYPOINT_FINGERPRINT = "entrypoints-v1"
_POLICY_ABI_FINGERPRINT = (
    "managed-policy-abi-v1:931aaecc6db37967f05db6439ec509a071bea1dce6ee53fec3f8069a2c030ae9"
)
_MJWARP_IDENTITY = {
    "task_name": "G1WalkFlat",
    "backend": "mjwarp",
    "execution_profile": "device_resident",
    "runtime_impl": "mjwarp_device_v1",
    "runtime_resolver": "unilab.training.rsl_rl_device:resolve_mjwarp_device_ppo_runtime",
}

pytestmark = pytest.mark.slow


def _require_cuda_mjwarp() -> None:
    try:
        dependencies = load_mjwarp_dependencies()
        is_cuda = bool(dependencies.warp.get_device().is_cuda)
    except Exception as exc:  # noqa: BLE001 - mandatory lane failures must not become skips
        pytest.fail(f"entrypoint matrix could not initialize mjwarp: {type(exc).__name__}: {exc}")
    if not is_cuda:
        pytest.fail(
            "managed MuJoCo/MJWarp rollout entrypoint matrix requires an active CUDA Warp device"
        )


def _run(
    script: str,
    *arguments: str,
    timeout: int = 300,
    without_display: bool = False,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["HYDRA_FULL_ERROR"] = "1"
    if without_display:
        env.pop("DISPLAY", None)
    return subprocess.run(
        [sys.executable, script, *arguments],
        cwd=ROOT_DIR,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _output(result: subprocess.CompletedProcess[str]) -> str:
    return f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"


def _assert_success(result: subprocess.CompletedProcess[str], *, operation: str) -> None:
    assert result.returncode == 0, f"{operation} failed:\n{_output(result)}"


def _assert_failure(
    result: subprocess.CompletedProcess[str],
    *fragments: str,
    before_materialization: bool = False,
) -> str:
    assert result.returncode != 0, f"operation unexpectedly succeeded:\n{_output(result)}"
    combined = _output(result)
    for fragment in fragments:
        assert fragment in combined, combined
    if before_materialization:
        assert "Warp " not in combined, combined
        assert "Resolved observation sets" not in combined, combined
    return combined


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _single_run(log_root: Path) -> Path:
    runs = sorted((log_root / "G1WalkFlat").glob("*_mjwarp"))
    assert len(runs) == 1, runs
    return runs[0]


def _assert_route_identity(contract: dict[str, Any], route: str) -> None:
    assert contract["fingerprint"] == _ENTRYPOINT_FINGERPRINT
    assert contract["route"] == route
    assert contract["disposition"] == "native"
    assert contract["identity"] == _MJWARP_IDENTITY
    assert contract["renderer_backend"] is None
    assert contract["adapter_backend"] is None


def _assert_device_run_evidence(
    run_config: dict[str, Any],
    run_summary: dict[str, Any],
) -> None:
    config = run_config["config"]
    assert run_config["run"]["sim_backend"] == "mjwarp"
    assert str(run_config["run"]["device"]).startswith("cuda")
    assert config["training"]["task_name"] == "G1WalkFlat"
    assert config["training"]["sim_backend"] == "mjwarp"
    assert config["training"]["execution_profile"] == "device_resident"
    assert config["algo"]["runtime_impl"] == "mjwarp_device_v1"
    assert config["entrypoints"]["fingerprint"] == _ENTRYPOINT_FINGERPRINT
    assert config["entrypoints"]["identity"] == _MJWARP_IDENTITY

    abi = run_config["contract_snapshot"]["manager.policy_abi"]
    assert abi["policy_abi_fingerprint"] == _POLICY_ABI_FINGERPRINT
    assert abi["executor_key"] == "device.torch.g1-walk-flat.v1"
    assert abi["observation_groups"][0]["width"] == 98
    assert abi["action"]["dim"] == 29

    assert run_summary["status"] == "completed"
    assert run_summary["sim_backend"] == "mjwarp"
    assert run_summary["configured_seed"] == 0
    assert run_summary["effective_seed"] == 0
    rss = run_summary["peak_process_rss_bytes"]
    assert isinstance(rss, (int, float)) and math.isfinite(float(rss)) and rss > 0

    traffic = run_summary["runtime_traffic_diagnostics"]
    assert traffic["policy_steps"] == _STEPS_PER_ENV
    assert traffic["step_barriers"] == _STEPS_PER_ENV
    assert traffic["reset_barriers"] > 0
    assert traffic["state_materializations"] == (
        traffic["step_barriers"] + traffic["reset_barriers"]
    )
    for field in (
        "host_to_device_transfers",
        "device_to_host_transfers",
        "host_to_device_bytes",
        "device_to_host_bytes",
        "global_synchronizations",
        "backend_allocations",
        "dynamic_getter_calls",
        "selector_resolutions",
        "asset_metadata_reads",
        "registry_lookups",
    ):
        assert traffic[field] == 0, (field, traffic[field])
    assert traffic["instrumentation_complete"] is True

    performance = run_summary["runtime_performance_diagnostics"]
    assert performance["backend_type"] == "mjwarp"
    lifecycle = performance["lifecycle"]
    assert lifecycle["step_graph_launches"] == _STEPS_PER_ENV
    assert lifecycle["reset_graph_launches"] == traffic["reset_barriers"]
    assert lifecycle["runtime_barriers"] == (
        lifecycle["step_graph_launches"] + lifecycle["reset_graph_launches"]
    )
    assert lifecycle["forward_graph_launches"] > 0
    graph = performance["graph"]
    assert graph["execution_mode"] == "cuda_graph"
    assert graph["launch_count"] > 0
    assert graph["eager_fallback_count"] == 0
    assert graph["instrumentation_complete"] is True

    wrapper = run_summary["wrapper_traffic_diagnostics"]
    assert wrapper["action_publications"] == _STEPS_PER_ENV
    assert wrapper["observation_snapshots"] == _STEPS_PER_ENV + 1
    stability = run_summary["runtime_stability_diagnostics"]
    assert stability["warm_numeric_allocations"] == 0
    assert stability["address_churn"] == 0
    assert stability["instrumentation_complete"] is True


def _clone_policy_source(source_run: Path, destination: Path) -> Path:
    destination.mkdir(parents=True)
    shutil.copy2(source_run / "run_config.json", destination / "run_config.json")
    shutil.copy2(source_run / "model_0.pt", destination / "model_0.pt")
    return destination


def _export_arguments(source_run: Path, log_root: Path, *extra: str) -> tuple[str, ...]:
    return (
        "task=g1_walk_flat/mjwarp",
        "training.operation=export",
        f"training.play_env_num={_NUM_ENVS}",
        f"algo.load_run={source_run}",
        "algo.checkpoint=0",
        f"training.log_root={log_root}",
        *extra,
    )


def _assert_no_export_artifacts(run_dir: Path) -> None:
    for name in ("policy.onnx", "policy.pt", "entrypoint_export_receipt.json"):
        assert not (run_dir / name).exists(), name


def _exercise_matrix_once(root: Path) -> None:
    train_log_root = root / "train-logs"
    train = _run(
        "scripts/train_rsl_rl.py",
        "task=g1_walk_flat/mjwarp",
        "training.operation=train",
        "algo.seed=0",
        f"algo.num_envs={_NUM_ENVS}",
        f"algo.num_steps_per_env={_STEPS_PER_ENV}",
        "algo.max_iterations=1",
        "algo.save_interval=1",
        "algo.capture_performance_diagnostics=true",
        "training.no_play=true",
        "training.logger=tensorboard",
        f"training.log_root={train_log_root}",
    )
    _assert_success(train, operation="mjwarp train/checkpoint-save")
    assert "Using device: cuda" in train.stdout
    assert "Learning iteration 0/1" in train.stdout

    source_run = _single_run(train_log_root)
    source_checkpoint = source_run / "model_0.pt"
    assert source_checkpoint.is_file() and source_checkpoint.stat().st_size > 0
    source_config = _load_json(source_run / "run_config.json")
    source_summary = _load_json(source_run / "run_summary.json")
    _assert_device_run_evidence(source_config, source_summary)
    assert source_summary["total_env_steps"] == _NUM_ENVS * _STEPS_PER_ENV
    train_receipt = source_summary["entrypoint_receipt"]
    _assert_route_identity(train_receipt["contract"], "train")
    _assert_route_identity(train_receipt["checkpoint_save_contract"], "checkpoint_save")
    assert train_receipt["checkpoint_load_contract"] is None
    assert train_receipt["checkpoint"] is None
    assert train_receipt["outputs"] == [str(source_checkpoint)]
    assert train_receipt["managed_policy_abi_fingerprint"] == _POLICY_ABI_FINGERPRINT
    assert train_receipt["observation_dim"] == 98
    assert train_receipt["action_dim"] == 29

    resume_log_root = root / "resume-logs"
    resume = _run(
        "scripts/train_rsl_rl.py",
        "task=g1_walk_flat/mjwarp",
        "training.operation=train",
        "algo.seed=0",
        f"algo.num_envs={_NUM_ENVS}",
        f"algo.num_steps_per_env={_STEPS_PER_ENV}",
        "algo.max_iterations=1",
        "algo.save_interval=1",
        "algo.capture_performance_diagnostics=true",
        f"algo.load_run={source_run}",
        "algo.checkpoint=0",
        "training.no_play=true",
        "training.logger=tensorboard",
        f"training.log_root={resume_log_root}",
    )
    _assert_success(resume, operation="mjwarp resume/checkpoint-load")
    assert f"Resuming from {source_checkpoint}" in resume.stdout
    resumed_run = _single_run(resume_log_root)
    resumed_checkpoint = resumed_run / "model_0.pt"
    resumed_config = _load_json(resumed_run / "run_config.json")
    resumed_summary = _load_json(resumed_run / "run_summary.json")
    _assert_device_run_evidence(resumed_config, resumed_summary)
    resume_receipt = resumed_summary["entrypoint_receipt"]
    _assert_route_identity(resume_receipt["contract"], "resume")
    _assert_route_identity(resume_receipt["checkpoint_save_contract"], "checkpoint_save")
    _assert_route_identity(resume_receipt["checkpoint_load_contract"], "checkpoint_load")
    assert resume_receipt["checkpoint"] == str(source_checkpoint)
    assert resume_receipt["outputs"] == [str(resumed_checkpoint)]
    assert resumed_checkpoint.is_file() and resumed_checkpoint.stat().st_size > 0

    export = _run(
        "scripts/train_rsl_rl.py",
        *_export_arguments(source_run, root / "export-logs"),
    )
    _assert_success(export, operation="mjwarp export")
    export_receipt = _load_json(source_run / "entrypoint_export_receipt.json")
    _assert_route_identity(export_receipt["contract"], "export")
    _assert_route_identity(export_receipt["checkpoint_load_contract"], "checkpoint_load")
    assert export_receipt["contract"]["export_formats"] == ["onnx", "jit"]
    assert export_receipt["checkpoint"] == str(source_checkpoint)
    assert export_receipt["managed_policy_abi_fingerprint"] == _POLICY_ABI_FINGERPRINT
    assert export_receipt["observation_dim"] == 98
    assert export_receipt["action_dim"] == 29
    expected_exports = [source_run / "policy.onnx", source_run / "policy.pt"]
    assert export_receipt["outputs"] == [str(path) for path in expected_exports]
    for path in expected_exports:
        assert path.is_file() and path.stat().st_size > 0

    unsupported_play = _run(
        "scripts/train_rsl_rl.py",
        "task=g1_walk_flat/mjwarp",
        "training.operation=play",
    )
    _assert_failure(
        unsupported_play,
        "Entrypoint route 'play' is unsupported",
        "no native playback model or renderer",
        "task=g1_walk_flat/mujoco",
        before_materialization=True,
    )
    unsupported_interactive = _run(
        "scripts/play_interactive.py",
        "--algo",
        "ppo",
        "--task",
        "g1_walk_flat",
        "--sim",
        "mjwarp",
        "interactive.action_mode=zero",
        without_display=True,
    )
    _assert_failure(
        unsupported_interactive,
        "Entrypoint route 'visualize' is unsupported",
        "no mjwarp renderer or explicit playback adapter",
        before_materialization=True,
    )
    unsupported_viser = _run(
        "scripts/play_viser.py",
        "task=g1_walk_flat/mjwarp",
        "interactive.action_mode=zero",
        without_display=True,
    )
    viser_output = _assert_failure(
        unsupported_viser,
        "Entrypoint route 'visualize' is unsupported",
        "no mjwarp renderer or explicit playback adapter",
        before_materialization=True,
    )
    assert "viser is not installed" not in viser_output

    config_mismatch = _clone_policy_source(source_run, root / "config-mismatch")
    mismatch = _run(
        "scripts/train_rsl_rl.py",
        *_export_arguments(
            config_mismatch,
            root / "config-mismatch-logs",
            "env.control_config.action_scale=0.5",
        ),
    )
    _assert_failure(
        mismatch,
        "Cross-backend sim2sim contract mismatch",
        "env.control_config.action_scale: source=0.25 target=0.5",
        before_materialization=True,
    )
    _assert_no_export_artifacts(config_mismatch)

    missing_metadata = root / "missing-metadata"
    missing_metadata.mkdir()
    shutil.copy2(source_checkpoint, missing_metadata / "model_0.pt")
    missing = _run(
        "scripts/train_rsl_rl.py",
        *_export_arguments(missing_metadata, root / "missing-metadata-logs"),
    )
    _assert_failure(
        missing,
        "Policy source metadata is missing",
        "policy loads cannot bypass config or managed ABI guards",
        before_materialization=True,
    )
    _assert_no_export_artifacts(missing_metadata)

    malformed_source = _clone_policy_source(source_run, root / "malformed-checkpoint")
    (malformed_source / "model_0.pt").write_bytes(b"not a torch checkpoint")
    malformed = _run(
        "scripts/train_rsl_rl.py",
        *_export_arguments(malformed_source, root / "malformed-logs"),
    )
    _assert_failure(
        malformed,
        "could not be parsed as an rsl-rl checkpoint",
        before_materialization=True,
    )
    _assert_no_export_artifacts(malformed_source)

    for override, field in (
        ("training.sim_backend=mujoco", "backend"),
        ("training.execution_profile=host_numpy", "execution_profile"),
        ("algo.runtime_impl=rsl_rl_default", "runtime_impl"),
    ):
        identity = _run(
            "scripts/train_rsl_rl.py",
            "task=g1_walk_flat/mjwarp",
            "training.operation=train",
            override,
        )
        _assert_failure(
            identity,
            f"owner-declared entrypoint identity mismatch for {field}",
            "do not override backend, execution profile, or runtime identity fields",
            before_materialization=True,
        )

    abi_source = _clone_policy_source(source_run, root / "abi-mismatch")
    abi_config_path = abi_source / "run_config.json"
    abi_config = _load_json(abi_config_path)
    abi_config["contract_snapshot"]["manager.policy_abi"]["executor_key"] = "device.torch.tampered"
    abi_config_path.write_text(json.dumps(abi_config), encoding="utf-8")
    abi_mismatch = _run(
        "scripts/train_rsl_rl.py",
        *_export_arguments(abi_source, root / "abi-mismatch-logs"),
    )
    abi_output = _assert_failure(
        abi_mismatch,
        "Cross-backend sim2sim contract mismatch",
        "manager.policy_abi",
        "device.torch.tampered",
    )
    assert "Warp " in abi_output
    _assert_no_export_artifacts(abi_source)

    dimension_source = _clone_policy_source(source_run, root / "dimension-mismatch")
    dimension_checkpoint = dimension_source / "model_0.pt"
    checkpoint_payload = torch.load(dimension_checkpoint, map_location="cpu", weights_only=True)
    actor_weight = checkpoint_payload["actor_state_dict"]["mlp.0.weight"]
    assert tuple(actor_weight.shape) == (512, 98)
    checkpoint_payload["actor_state_dict"]["mlp.0.weight"] = actor_weight[:, :97].clone()
    torch.save(checkpoint_payload, dimension_checkpoint)
    dimension_mismatch = _run(
        "scripts/train_rsl_rl.py",
        *_export_arguments(dimension_source, root / "dimension-mismatch-logs"),
    )
    dimension_output = _assert_failure(
        dimension_mismatch,
        "Trained policy checkpoint does not fit this play environment",
        "env policy obs dim: 98",
        f"managed policy ABI: {_POLICY_ABI_FINGERPRINT}",
        "size mismatch",
    )
    assert "Warp " in dimension_output
    _assert_no_export_artifacts(dimension_source)

    missing_policy_root = root / "missing-mujoco-policy"
    missing_policy_root.mkdir()
    missing_policy = _run(
        "scripts/play_interactive.py",
        "--algo",
        "ppo",
        "--task",
        "g1_walk_flat",
        "--sim",
        "mujoco",
        "interactive.action_mode=policy",
        "training.device=cpu",
        f"training.log_root={missing_policy_root}",
        "algo.load_run=missing",
        without_display=True,
    )
    missing_policy_output = _assert_failure(
        missing_policy,
        "policy action mode requires a checkpoint",
        "select interactive.action_mode=zero explicitly",
    )
    assert "Policy obs mode:" not in missing_policy_output

    zero_action = _run(
        "scripts/play_interactive.py",
        "--algo",
        "ppo",
        "--task",
        "g1_walk_flat",
        "--sim",
        "mujoco",
        "interactive.action_mode=zero",
        "training.device=cpu",
        without_display=True,
    )
    _assert_success(zero_action, operation="explicit MuJoCo zero-action playback")
    assert "[play_interactive] Action mode: zero" in zero_action.stdout
    assert "GLFW viewer initialization failed (no usable display)" in zero_action.stdout


def test_supported_train_play_visualize_export_matrix(tmp_path: Path) -> None:
    """Execute the frozen public route/guard matrix in two isolated subprocess sets."""

    _require_cuda_mjwarp()
    for repetition in range(2):
        _exercise_matrix_once(tmp_path / f"repetition-{repetition}")
