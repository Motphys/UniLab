"""Tests for the off-policy multi-GPU data-parallel rank topology."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

import unilab.ipc.dp_launcher as dp_launcher
from unilab.ipc.dp_launcher import (
    UNILAB_DP_DEVICES,
    UNILAB_DP_LOG_DIR,
    UNILAB_DP_RANK,
    UNILAB_DP_WORLD_SIZE,
    DpRankSupervisor,
    apply_dp_rank_config,
    current_dp_rank,
    current_dp_world_size,
    current_torch_distributed_local_rank,
    current_torch_distributed_rank,
    current_torch_distributed_world_size,
    launch_torchrun_workers,
    resolve_collector_cpu_ids,
    resolve_cuda_visible_devices,
    resolve_dp_rank_device,
    resolve_dp_topology,
    validate_dp_launchable,
)

_ROOT = Path(__file__).parent.parent.parent
_CONF_DIR = _ROOT / "conf"


def _offpolicy_cfg(overrides: list[str] | None = None):
    GlobalHydra.instance().clear()
    normalized = list(overrides or [])
    if not any(override.startswith("task=") for override in normalized):
        normalized.append("task=sac/g1_walk_flat/mujoco")
    with initialize_config_dir(config_dir=str(_CONF_DIR / "offpolicy"), version_base="1.3"):
        return compose("config", overrides=normalized)


class _FakePopen:
    """Minimal subprocess.Popen stand-in with a scriptable exit code."""

    instances: list["_FakePopen"] = []
    # When True, wait() on a still-running child makes it exit cleanly
    # (simulates a rank that finishes during the normal-exit grace window).
    wait_completes: bool = False
    interrupt_exits: bool = True

    def __init__(self, argv, env=None, **kwargs):
        self.argv = list(argv)
        self.env = dict(env or {})
        self.start_new_session = bool(kwargs.pop("start_new_session", False))
        assert not kwargs
        self.pid = 10_000 + len(_FakePopen.instances)
        self.returncode: int | None = None
        self.terminated = False
        self.killed = False
        self.signals: list[int] = []
        _FakePopen.instances.append(self)

    def poll(self):
        return self.returncode

    def terminate(self):
        self.send_signal(signal.SIGTERM)

    def kill(self):
        self.send_signal(signal.SIGKILL)

    def send_signal(self, signum):
        self.signals.append(signum)
        if signum == signal.SIGINT and self.interrupt_exits:
            self.returncode = -signal.SIGINT
        elif signum == signal.SIGTERM:
            self.terminated = True
            self.returncode = -signal.SIGTERM
        elif signum == signal.SIGKILL:
            self.killed = True
            self.returncode = -signal.SIGKILL

    def wait(self, timeout=None):
        if self.returncode is None:
            if _FakePopen.wait_completes:
                self.returncode = 0
                return 0
            raise subprocess.TimeoutExpired(cmd=self.argv, timeout=timeout)
        return self.returncode


@pytest.fixture()
def fake_popen(monkeypatch: pytest.MonkeyPatch):
    _FakePopen.instances = []
    _FakePopen.wait_completes = False
    _FakePopen.interrupt_exits = True
    monkeypatch.setattr(dp_launcher.subprocess, "Popen", _FakePopen)
    monkeypatch.setattr(
        dp_launcher, "_signal_process_group", lambda child, signum: child.send_signal(signum)
    )
    monkeypatch.setattr(
        dp_launcher, "_process_group_exists", lambda child: child.returncode is None
    )
    monkeypatch.setattr(dp_launcher, "_COOPERATIVE_EXIT_GRACE_S", 0.0)
    monkeypatch.setattr(dp_launcher, "_TERMINATE_TIMEOUT_S", 0.0)
    monkeypatch.setattr(dp_launcher, "_NORMAL_EXIT_GRACE_S", 0.01)
    return _FakePopen


# ---------------------------------------------------------------------------
# resolve_dp_topology
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("devices_cfg", [None, []])
def test_resolve_dp_topology_single_device_default(devices_cfg):
    assert resolve_dp_topology(devices_cfg) is None


def test_resolve_dp_topology_preserves_user_order():
    assert resolve_dp_topology([0, 1]) == (0, 1)
    assert resolve_dp_topology([2, 0]) == (2, 0)


@pytest.mark.parametrize(
    "devices_cfg",
    [[0, 0], [-1], [0, "1"], [True], [0.5]],
)
def test_resolve_dp_topology_rejects_invalid_entries(devices_cfg):
    with pytest.raises(ValueError, match="training.devices"):
        resolve_dp_topology(devices_cfg)


# ---------------------------------------------------------------------------
# rank / world-size environment
# ---------------------------------------------------------------------------


def test_current_dp_rank_defaults(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(UNILAB_DP_RANK, raising=False)
    monkeypatch.delenv(UNILAB_DP_WORLD_SIZE, raising=False)
    assert current_dp_rank() == 0
    assert current_dp_world_size() == 1


def test_current_dp_rank_reads_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(UNILAB_DP_RANK, "2")
    monkeypatch.setenv(UNILAB_DP_WORLD_SIZE, "4")
    assert current_dp_rank() == 2
    assert current_dp_world_size() == 4


def test_current_torch_distributed_rank_defaults(monkeypatch: pytest.MonkeyPatch):
    for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE"):
        monkeypatch.delenv(name, raising=False)
    assert current_torch_distributed_rank() == 0
    assert current_torch_distributed_local_rank() == 0
    assert current_torch_distributed_world_size() == 1


def test_current_torch_distributed_rank_reads_torchrun_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("RANK", "3")
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "4")
    assert current_torch_distributed_rank() == 3
    assert current_torch_distributed_local_rank() == 1
    assert current_torch_distributed_world_size() == 4


def test_resolve_cuda_visible_devices_preserves_parent_mapping():
    assert resolve_cuda_visible_devices((2, 0)) == "2,0"
    assert (
        resolve_cuda_visible_devices(
            (1, 0),
            current_visible_devices="GPU-parent-0,GPU-parent-1",
        )
        == "GPU-parent-1,GPU-parent-0"
    )


def test_resolve_cuda_visible_devices_rejects_hidden_index():
    with pytest.raises(ValueError, match="CUDA_VISIBLE_DEVICES"):
        resolve_cuda_visible_devices((0, 2), current_visible_devices="4,7")


def test_launch_torchrun_workers_builds_standard_single_node_command(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    captured: dict[str, object] = {}

    def fake_run(command, *, env, check):
        captured.update(command=list(command), env=dict(env), check=check)
        return type("Result", (), {"returncode": 0})()

    monkeypatch.setattr(dp_launcher, "validate_dp_launchable", lambda devices: None)
    monkeypatch.setattr(dp_launcher.subprocess, "run", fake_run)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-a,GPU-b,GPU-c")
    monkeypatch.delenv("NCCL_P2P_DISABLE", raising=False)
    monkeypatch.delenv("NCCL_SHM_DISABLE", raising=False)
    script = tmp_path / "train_rsl_rl.py"

    launch_torchrun_workers(
        (2, 0),
        script_path=script,
        argv=["task=go2_joystick_flat/mujoco", "training.devices=[2,0]"],
        log_dir="/tmp/ppo_gpux2",
    )

    command = captured["command"]
    assert isinstance(command, list)
    assert command[:3] == [sys.executable, "-m", "torch.distributed.run"]
    assert "--standalone" in command
    assert "--nnodes=1" in command
    assert "--nproc_per_node=2" in command
    assert command[-2:] == ["task=go2_joystick_flat/mujoco", "training.devices=[2,0]"]
    launch_env = captured["env"]
    assert isinstance(launch_env, dict)
    assert launch_env["CUDA_VISIBLE_DEVICES"] == "GPU-c,GPU-a"
    assert launch_env["NCCL_P2P_DISABLE"] == "1"
    assert launch_env["NCCL_SHM_DISABLE"] == "1"
    assert launch_env[UNILAB_DP_LOG_DIR] == "/tmp/ppo_gpux2"
    assert captured["check"] is False


def test_launch_torchrun_workers_propagates_failure(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(dp_launcher, "validate_dp_launchable", lambda devices: None)
    monkeypatch.setattr(
        dp_launcher.subprocess,
        "run",
        lambda *args, **kwargs: type("Result", (), {"returncode": 7})(),
    )

    with pytest.raises(RuntimeError, match="exit code 7"):
        launch_torchrun_workers(
            (0, 1),
            script_path="scripts/train_rsl_rl.py",
            argv=[],
            log_dir="/tmp/ppo_gpux2",
        )


def test_launch_torchrun_workers_preserves_explicit_nccl_transport(
    monkeypatch: pytest.MonkeyPatch,
):
    captured: dict[str, str] = {}

    def fake_run(*args, env, **kwargs):
        del args, kwargs
        captured.update(env)
        return type("Result", (), {"returncode": 0})()

    monkeypatch.setattr(dp_launcher, "validate_dp_launchable", lambda devices: None)
    monkeypatch.setattr(dp_launcher.subprocess, "run", fake_run)
    monkeypatch.setenv("NCCL_P2P_DISABLE", "0")
    monkeypatch.setenv("NCCL_SHM_DISABLE", "0")

    launch_torchrun_workers(
        (0, 1),
        script_path="scripts/train_rsl_rl.py",
        argv=[],
        log_dir="/tmp/ppo_gpux2",
    )

    assert captured["NCCL_P2P_DISABLE"] == "0"
    assert captured["NCCL_SHM_DISABLE"] == "0"


# ---------------------------------------------------------------------------
# Hydra compose surface
# ---------------------------------------------------------------------------


def test_offpolicy_config_devices_defaults_to_null():
    cfg = _offpolicy_cfg()
    assert cfg.training.devices is None
    assert resolve_dp_topology(cfg.training.devices) is None


def test_offpolicy_config_devices_compose():
    cfg = _offpolicy_cfg(["training.devices=[0,1]"])
    assert resolve_dp_topology(cfg.training.devices) == (0, 1)


def test_offpolicy_config_has_no_redundant_singular_device():
    cfg = _offpolicy_cfg()
    assert "device" not in cfg.training


# ---------------------------------------------------------------------------
# apply_dp_rank_config / N=1 equivalence
# ---------------------------------------------------------------------------


def test_apply_dp_rank_config_maps_rank_to_device_and_seed():
    cfg = _offpolicy_cfg(["training.devices=[0,1]", "algo.seed=42"])
    base_seed = int(cfg.algo.seed)
    device = apply_dp_rank_config(cfg, (0, 1), rank=1)
    assert device == "cuda:1"
    assert "device" not in cfg.training
    assert int(cfg.algo.seed) == base_seed + 1


def test_apply_dp_rank_config_rank_zero_keeps_seed():
    cfg = _offpolicy_cfg(["training.devices=[0,1]", "algo.seed=42"])
    assert apply_dp_rank_config(cfg, (0, 1), rank=0) == "cuda:0"
    assert int(cfg.algo.seed) == 42


def test_resolve_dp_rank_device_uses_auto_selection_without_devices():
    assert resolve_dp_rank_device(None, rank=0) is None
    assert resolve_dp_rank_device((2, 0), rank=1) == "cuda:0"
    with pytest.raises(ValueError, match="out of range"):
        resolve_dp_rank_device((0, 1), rank=2)


def test_single_device_topology_spawns_no_children(fake_popen, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(UNILAB_DP_RANK, raising=False)
    cfg = _offpolicy_cfg(["training.devices=[0]"])
    devices = resolve_dp_topology(cfg.training.devices)
    assert devices == (0,)
    assert apply_dp_rank_config(cfg, devices, rank=0) == "cuda:0"
    with DpRankSupervisor(devices, log_dir="/tmp/dp_test_log"):
        assert fake_popen.instances == []
    assert os.environ.get(UNILAB_DP_RANK) is None


# ---------------------------------------------------------------------------
# DpRankSupervisor
# ---------------------------------------------------------------------------


def test_supervisor_spawn_argv_and_env(fake_popen, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(UNILAB_DP_RANK, raising=False)
    monkeypatch.setattr(sys, "argv", ["train_offpolicy.py", "algo=sac", "training.devices=[0,1,2]"])
    with DpRankSupervisor((0, 1, 2), log_dir="/tmp/dp_test_log"):
        assert len(fake_popen.instances) == 2
        for rank, child in enumerate(fake_popen.instances, start=1):
            assert child.argv[0] == sys.executable
            assert child.argv[1].endswith("scripts/train_offpolicy.py")
            assert child.argv[2:] == ["algo=sac", "training.devices=[0,1,2]"]
            assert child.start_new_session is (os.name == "posix")
            assert child.env[UNILAB_DP_RANK] == str(rank)
            assert child.env[UNILAB_DP_WORLD_SIZE] == "3"
            assert child.env[UNILAB_DP_DEVICES] == "0,1,2"
            assert child.env[UNILAB_DP_LOG_DIR] == "/tmp/dp_test_log"
        # Rank 0's own environment stays untouched.
        assert os.environ.get(UNILAB_DP_RANK) is None
        for child in fake_popen.instances:
            child.returncode = 0
    for child in fake_popen.instances:
        assert not child.terminated


def test_supervisor_normal_exit_waits_for_children(fake_popen):
    _FakePopen.wait_completes = True
    with DpRankSupervisor((0, 1), log_dir="/tmp/dp_test_log"):
        pass
    child = fake_popen.instances[0]
    assert child.returncode == 0
    assert not child.terminated


def test_supervisor_grace_timeout_is_a_failure(fake_popen):
    with pytest.raises(RuntimeError, match="rank 1 exit code timeout"):
        with DpRankSupervisor((0, 1), log_dir="/tmp/dp_test_log"):
            pass
    child = fake_popen.instances[0]
    assert child.terminated
    assert child.returncode == -signal.SIGTERM


def test_supervisor_clean_child_exit_is_not_a_failure(fake_popen):
    with DpRankSupervisor((0, 1), log_dir="/tmp/dp_test_log"):
        fake_popen.instances[0].returncode = 0


def test_supervisor_failed_child_makes_rank_zero_fail(fake_popen, monkeypatch: pytest.MonkeyPatch):
    # Keep the watchdog from polling so __exit__ observes the exit code first.
    monkeypatch.setattr(dp_launcher, "_WATCHDOG_INTERVAL_S", 60.0)
    supervisor = DpRankSupervisor((0, 1), log_dir="/tmp/dp_test_log")
    supervisor.__enter__()
    fake_popen.instances[0].returncode = 3
    with pytest.raises(RuntimeError, match="rank 1 exit code 3"):
        supervisor.__exit__(None, None, None)


def test_supervisor_error_exit_terminates_live_children(fake_popen):
    fake_popen.interrupt_exits = False
    with pytest.raises(ValueError, match="boom"):
        with DpRankSupervisor((0, 1, 2), log_dir="/tmp/dp_test_log"):
            raise ValueError("boom")
    assert all(child.terminated for child in fake_popen.instances)
    assert all(child.returncode == -signal.SIGTERM for child in fake_popen.instances)
    assert all(child.signals == [signal.SIGINT, signal.SIGTERM] for child in fake_popen.instances)


def test_supervisor_ctrl_c_forwards_sigint_for_cooperative_rank_cleanup(fake_popen):
    previous_sigint = signal.getsignal(signal.SIGINT)

    with pytest.raises(KeyboardInterrupt):
        with DpRankSupervisor((0, 1, 2), log_dir="/tmp/dp_test_log") as supervisor:
            installed = signal.getsignal(signal.SIGINT)
            assert getattr(installed, "__self__", None) is supervisor
            installed(signal.SIGINT, None)

    assert all(child.signals == [signal.SIGINT] for child in fake_popen.instances)
    assert all(child.returncode == -signal.SIGINT for child in fake_popen.instances)
    assert signal.getsignal(signal.SIGINT) == previous_sigint


def test_supervisor_watchdog_sigterms_rank_zero_on_child_death(
    fake_popen, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(dp_launcher, "_WATCHDOG_INTERVAL_S", 0.01)
    killed: list[int] = []
    monkeypatch.setattr(dp_launcher.os, "kill", lambda pid, sig: killed.append(sig))
    with pytest.raises(RuntimeError, match="exit code 1"):
        with DpRankSupervisor((0, 1), log_dir="/tmp/dp_test_log"):
            fake_popen.instances[0].returncode = 1
            deadline = time.monotonic() + 5.0
            while not killed and time.monotonic() < deadline:
                time.sleep(0.01)
    assert killed == [signal.SIGTERM]


def test_supervisor_restores_signal_handlers(fake_popen):
    previous_sigint = signal.getsignal(signal.SIGINT)
    previous_sigterm = signal.getsignal(signal.SIGTERM)
    with DpRankSupervisor((0, 1), log_dir="/tmp/dp_test_log") as supervisor:
        assert getattr(signal.getsignal(signal.SIGINT), "__self__", None) is supervisor
        assert getattr(signal.getsignal(signal.SIGTERM), "__self__", None) is supervisor
        fake_popen.instances[0].returncode = 0
    assert signal.getsignal(signal.SIGINT) == previous_sigint
    assert signal.getsignal(signal.SIGTERM) == previous_sigterm


def test_supervisor_restores_signal_handlers_when_group_reaping_fails(
    fake_popen, monkeypatch: pytest.MonkeyPatch
):
    previous_sigint = signal.getsignal(signal.SIGINT)
    previous_sigterm = signal.getsignal(signal.SIGTERM)
    supervisor = DpRankSupervisor((0, 1), log_dir="/tmp/dp_test_log")
    supervisor.__enter__()
    fake_popen.instances[0].returncode = 0

    def fail_reaping() -> None:
        raise RuntimeError("group reaping failed")

    monkeypatch.setattr(supervisor, "_terminate_children", fail_reaping)

    with pytest.raises(RuntimeError, match="group reaping failed"):
        supervisor.__exit__(None, None, None)

    assert signal.getsignal(signal.SIGINT) == previous_sigint
    assert signal.getsignal(signal.SIGTERM) == previous_sigterm


# ---------------------------------------------------------------------------
# resolve_collector_cpu_ids
# ---------------------------------------------------------------------------


def test_resolve_collector_cpu_ids_single_rank_returns_none():
    assert resolve_collector_cpu_ids(1, 0, 128) is None
    assert resolve_collector_cpu_ids(0, 0, 128) is None


def test_resolve_collector_cpu_ids_even_partition():
    assert resolve_collector_cpu_ids(2, 0, 128) == list(range(0, 64))
    assert resolve_collector_cpu_ids(2, 1, 128) == list(range(64, 128))


def test_resolve_collector_cpu_ids_remainder_stays_unassigned():
    # 129 CPUs / 2 ranks -> 64+64; CPU 128 keeps default OS scheduling.
    assert resolve_collector_cpu_ids(2, 0, 129) == list(range(0, 64))
    assert resolve_collector_cpu_ids(2, 1, 129) == list(range(64, 128))


def test_resolve_collector_cpu_ids_requires_one_cpu_per_rank():
    with pytest.raises(ValueError, match="cpu_count"):
        resolve_collector_cpu_ids(4, 0, 3)


def test_resolve_collector_cpu_ids_rank_out_of_range():
    with pytest.raises(ValueError, match="out of range"):
        resolve_collector_cpu_ids(2, 2, 128)


def test_resolve_collector_cpu_ids_explicit_valid():
    assert resolve_collector_cpu_ids(2, 1, 128, explicit=[[0, 1], [2, 3]]) == [2, 3]


def test_resolve_collector_cpu_ids_empty_explicit_falls_back_to_auto():
    assert resolve_collector_cpu_ids(2, 1, 128, explicit=[]) == list(range(64, 128))


def test_resolve_collector_cpu_ids_explicit_rejects_segment_count_mismatch():
    with pytest.raises(ValueError, match="world_size"):
        resolve_collector_cpu_ids(2, 0, 128, explicit=[[0], [1], [2]])


def test_resolve_collector_cpu_ids_explicit_rejects_overlapping_segments():
    with pytest.raises(ValueError, match="overlap"):
        resolve_collector_cpu_ids(2, 0, 128, explicit=[[0, 1], [1, 2]])


def test_resolve_collector_cpu_ids_explicit_rejects_empty_segment():
    with pytest.raises(ValueError, match="non-empty"):
        resolve_collector_cpu_ids(2, 0, 128, explicit=[[0, 1], []])


@pytest.mark.parametrize("bad_id", [-1, True, "2"])
def test_resolve_collector_cpu_ids_explicit_rejects_invalid_ids(bad_id):
    with pytest.raises(ValueError, match="non-negative integers"):
        resolve_collector_cpu_ids(2, 0, 128, explicit=[[0, bad_id], [2, 3]])


def test_offpolicy_config_dp_collector_cpu_ids_defaults_to_null():
    cfg = _offpolicy_cfg()
    assert cfg.training.dp_collector_cpu_ids is None


def test_offpolicy_config_dp_collector_cpu_ids_compose():
    cfg = _offpolicy_cfg(["training.dp_collector_cpu_ids=[[0,1],[2,3]]"])
    explicit = OmegaConf.to_container(cfg.training.dp_collector_cpu_ids, resolve=True)
    assert explicit == [[0, 1], [2, 3]]
    assert resolve_collector_cpu_ids(2, 1, 128, explicit=explicit) == [2, 3]


# ---------------------------------------------------------------------------
# Hardware-gated smoke
# ---------------------------------------------------------------------------


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires >=2 CUDA devices")
@pytest.mark.slow
def test_dp_topology_validates_on_two_gpu_host():
    devices = resolve_dp_topology([0, 1])
    assert devices == (0, 1)
    validate_dp_launchable(devices)
