from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from scripts.benchmark.torch_env import g1_flashsac_backend as g1_backend
from scripts.benchmark.torch_env.g1_flashsac_backend import (
    _build_cfg,
    _clear_workload_state_aliases,
    _materialize_and_negotiate,
    _parse_args,
    _release_cuda_view_aliases,
    _reset_stride,
)
from scripts.benchmark.torch_env.motion_tracking import MotionTrackingWorkload
from scripts.benchmark.torch_env.xp import TorchBackend, TorchRng


def test_isaacsim_fixture_loader_is_rejected_after_productionization() -> None:
    with pytest.raises(ValueError, match="productionized in #1771"):
        _build_cfg("mujoco", num_envs=2, isaacsim_test_fixture=True)


def test_isaacsim_fixture_cli_flag_is_obsolete(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        _parse_args(["--backends", "isaacsim,mujoco"])
    assert "shelved benchmark backend(s)" in capsys.readouterr().err

    args, backends = _parse_args(["--backends", "mujoco,mjwarp"])
    assert backends == ["mujoco", "mjwarp"]
    assert args.isaacsim_test_fixture is False


def test_tensor_runtime_diagnostics_are_serialized_for_backend_provenance() -> None:
    backend = SimpleNamespace(
        get_tensor_runtime_diagnostics=lambda: {
            "cuda_graph": SimpleNamespace(
                requested=True,
                enabled=False,
                disable_reason="forced ineligibility",
            )
        }
    )

    assert g1_backend._tensor_runtime_diagnostics(backend) == {
        "cuda_graph": {
            "requested": True,
            "enabled": False,
            "disable_reason": "forced ineligibility",
        }
    }


@pytest.mark.parametrize(
    ("num_envs", "expected_stride"),
    ((1, 1), (2, 2), (15, 15), (16, 16), (2048, 16)),
)
def test_reset_stride_keeps_small_smoke_runs_non_empty(num_envs: int, expected_stride: int) -> None:
    assert _reset_stride(num_envs) == expected_stride


def test_reset_stride_rejects_non_positive_values() -> None:
    with pytest.raises(ValueError, match="num_envs must be a positive integer"):
        _reset_stride(0)


def test_minimal_reset_schedule_selects_one_row_every_control_step() -> None:
    stride = _reset_stride(2)
    row_pattern = torch.arange(2) % stride

    for iteration in range(stride):
        mask = row_pattern == (iteration % stride)
        assert int(mask.sum()) == 1


def test_motion_tracking_reset_done_returns_empty_noop() -> None:
    workload = MotionTrackingWorkload(
        TorchBackend("cpu"), TorchRng("cpu"), seed=7, vectorized_reset_rng=True, num_envs=2
    )

    qpos, qvel, reset_obs, info = workload.reset_done(torch.empty(0, dtype=torch.int64))

    assert qpos.shape == (0, 36)
    assert qvel.shape == (0, 35)
    assert reset_obs["obs"].shape == (0, 160)
    assert reset_obs["critic"].shape == (0, 289)
    assert info["current_actions"].shape == (0, 29)
    assert info["last_actions"].shape == (0, 29)


def test_cuda_view_alias_release_drops_all_owned_views() -> None:
    released: list[str] = []

    class View:
        def __init__(self, name: str):
            self.name = name

        def __del__(self):
            released.append(self.name)

    views = {"state": View("state"), "sensors": View("sensors")}

    _release_cuda_view_aliases(views)

    assert views == {}
    assert sorted(released) == ["sensors", "state"]


def test_workload_alias_cleanup_drops_state_and_sensor_views() -> None:
    workload = SimpleNamespace(
        dof_pos=object(),
        dof_vel=object(),
        linvel=object(),
        gyro=object(),
        body_pos=object(),
        body_quat=object(),
        body_lin_vel=object(),
        body_ang_vel=object(),
    )

    _clear_workload_state_aliases(workload)

    assert workload.__dict__ == {
        "dof_pos": None,
        "dof_vel": None,
        "linvel": None,
        "gyro": None,
        "body_pos": None,
        "body_quat": None,
        "body_lin_vel": None,
        "body_ang_vel": None,
    }


def test_acceptance_mode_rejects_enabled_worker_profiler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in g1_backend._PROFILER_ENV_KEYS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(
        g1_backend,
        "_nvidia_smi_snapshot",
        lambda: {"gpus": [{"index": "0"}], "compute_apps": []},
    )

    g1_backend._validate_acceptance_environment()

    monkeypatch.setenv(g1_backend._PROFILER_ENV_KEYS[0], "/tmp/trace.json")
    with pytest.raises(RuntimeError, match="profiling to be disabled"):
        g1_backend._validate_acceptance_environment()
    monkeypatch.setenv(g1_backend._PROFILER_ENV_KEYS[0], "")
    with pytest.raises(RuntimeError, match="profiling to be disabled"):
        g1_backend._validate_acceptance_environment()


def test_acceptance_mode_rejects_gpu_contention(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in g1_backend._PROFILER_ENV_KEYS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(
        g1_backend,
        "_nvidia_smi_snapshot",
        lambda: {"gpus": [{"index": "0"}], "compute_apps": [{"pid": "3100340"}]},
    )

    with pytest.raises(RuntimeError, match="idle GPU before-run; active PIDs: 3100340"):
        g1_backend._validate_acceptance_environment()


def test_acceptance_mode_rejects_after_run_gpu_contention(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before = {"gpus": [{"index": "0"}], "compute_apps": []}
    after = {
        "gpus": [{"index": "0"}],
        "compute_apps": [{"pid": "3100340"}],
    }
    monkeypatch.setattr(g1_backend, "_nvidia_smi_snapshot", lambda: after)
    monkeypatch.setattr(g1_backend, "_AFTER_RUN_GPU_QUIESCE_TIMEOUT_S", 0.0)
    result: dict[str, object] = {}

    with pytest.raises(RuntimeError, match="idle GPU after-run.*3100340"):
        g1_backend._record_gpu_snapshots(result, before)

    assert result["gpu_snapshots"] == {"before": before, "after": after}


def test_acceptance_mode_allows_bounded_isaac_worker_cuda_teardown_quiesce(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before = {"gpus": [{"index": "0"}], "compute_apps": []}
    teardown = {
        "gpus": [{"index": "0"}],
        "compute_apps": [{"pid": "4172249"}],
    }
    idle = {"gpus": [{"index": "0"}], "compute_apps": []}
    snapshots = iter([teardown, idle])
    monkeypatch.setattr(g1_backend, "_nvidia_smi_snapshot", lambda: next(snapshots))
    monkeypatch.setattr(g1_backend, "_AFTER_RUN_GPU_QUIESCE_POLL_S", 0.0)
    result: dict[str, object] = {}

    g1_backend._record_gpu_snapshots(result, before)

    assert result["gpu_snapshots"] == {"before": before, "after": idle}
    assert isinstance(result["after_run_gpu_quiesce_wait_s"], float)
    assert result["after_run_gpu_quiesce_wait_s"] >= 0.0


def test_acceptance_mode_fails_closed_without_gpu_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in g1_backend._PROFILER_ENV_KEYS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(
        g1_backend, "_nvidia_smi_snapshot", lambda: {"gpus": [], "compute_apps": []}
    )

    with pytest.raises(RuntimeError, match="cannot verify before-run nvidia-smi"):
        g1_backend._validate_acceptance_environment()


def test_acceptance_mode_fails_closed_when_gpu_evidence_query_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = {
        "gpus": [{"index": "0"}],
        "compute_apps": [],
        "errors": ["compute_apps: nvidia-smi exited 1: query unsupported"],
    }

    with pytest.raises(
        RuntimeError,
        match="cannot verify before-run nvidia-smi.*query unsupported",
    ):
        g1_backend._validate_acceptance_gpu_snapshot(snapshot, "before-run")


def test_acceptance_mode_fails_closed_after_run_when_snapshot_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before = {"gpus": [{"index": "0"}], "compute_apps": []}
    after = {
        "gpus": [{"index": "0"}],
        "compute_apps": [],
        "errors": ["gpus: nvidia-smi timed out after 2.0 seconds"],
    }
    monkeypatch.setattr(g1_backend, "_nvidia_smi_snapshot", lambda: after)
    result: dict[str, object] = {}

    with pytest.raises(RuntimeError, match="cannot verify after-run nvidia-smi.*timed out"):
        g1_backend._record_gpu_snapshots(result, before)

    assert result["gpu_snapshots"] == {"before": before, "after": after}


def test_nvidia_smi_snapshot_records_query_failures_instead_of_empty_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(*args: object, **kwargs: object) -> object:
        return SimpleNamespace(returncode=1, stdout="", stderr="driver initialization failed")

    monkeypatch.setattr(g1_backend.subprocess, "run", fake_run)

    snapshot = g1_backend._nvidia_smi_snapshot()

    assert snapshot["gpus"] == []
    assert snapshot["compute_apps"] == []
    assert snapshot["errors"] == [
        "gpus: nvidia-smi exited 1: driver initialization failed",
        "compute_apps: nvidia-smi exited 1: driver initialization failed",
    ]


def test_benchmark_environment_records_runtime_and_profiler_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("UNILAB_LOCAL_UNISIM", "/unisim")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    environment = g1_backend._benchmark_environment()

    assert environment["UNILAB_LOCAL_UNISIM"] == "/unisim"
    assert environment["CUDA_VISIBLE_DEVICES"] is None
    assert set(g1_backend._PROFILER_ENV_KEYS) <= set(environment)


def test_benchmark_shape_rejects_empty_or_negative_measurement() -> None:
    g1_backend._validate_benchmark_shape(num_envs=2, warmup=0, iters=1)

    with pytest.raises(ValueError, match="num_envs must be a positive integer"):
        g1_backend._validate_benchmark_shape(num_envs=0, warmup=1, iters=1)
    with pytest.raises(ValueError, match="warmup must be nonnegative"):
        g1_backend._validate_benchmark_shape(num_envs=2, warmup=-1, iters=1)
    with pytest.raises(ValueError, match="iters must be positive"):
        g1_backend._validate_benchmark_shape(num_envs=2, warmup=1, iters=0)


def test_worker_version_probe_reports_missing_packages() -> None:
    versions = g1_backend._worker_package_versions(
        Path(sys.executable), ("torch", "unisim-not-a-real-package")
    )

    assert versions["torch"] == torch.__version__
    assert versions["unisim-not-a-real-package"] == "not-installed"


def test_tensor_benchmark_negotiates_after_subprocess_materialization() -> None:
    calls: list[str] = []

    class Backend:
        def materialize(self) -> None:
            calls.append("materialize")

        def tensor_execution(self) -> object:
            calls.append("tensor_execution")
            return type("Execution", (), {"value": "device_resident"})()

        def get_tensor_capabilities(self) -> object:
            calls.append("capabilities")
            return object()

    mode, capabilities = _materialize_and_negotiate("mjwarp", Backend())

    assert mode == "device_resident"
    assert capabilities is not None
    assert calls == ["materialize", "tensor_execution", "capabilities"]


def test_tensor_benchmark_closes_backend_when_lifecycle_setup_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Backend:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    backend = Backend()
    scene = SimpleNamespace(entities={"robot": SimpleNamespace(body_names=("pelvis",))})
    monkeypatch.setattr(
        g1_backend.torch.cuda,
        "is_available",
        lambda: True,
        raising=False,
    )
    monkeypatch.setattr(
        g1_backend,
        "_build_cfg",
        lambda backend, num_envs, isaacsim_test_fixture=False: SimpleNamespace(scene=scene),
    )
    monkeypatch.setattr(
        g1_backend,
        "_build_backend",
        lambda backend_name, num_envs, isaacsim_test_fixture=False: backend,
    )

    def fail_setup(*args: object, **kwargs: object) -> dict[str, object]:
        raise RuntimeError("sensor view negotiation failed")

    monkeypatch.setattr(g1_backend, "_run_with_backend", fail_setup)

    with pytest.raises(RuntimeError, match="sensor view negotiation failed"):
        g1_backend._run("mjwarp", 2, warmup=0, iters=1)

    assert backend.closed is True


@pytest.mark.parametrize("backend", ("drake", "isaacgym", "isaacsim", "superdex"))
def test_shelved_benchmark_backends_fail_closed(
    backend: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit):
        _parse_args(["--backends", backend])

    error = capsys.readouterr().err
    assert backend in error
    assert "issue #1811" in error
