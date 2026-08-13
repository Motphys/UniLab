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

    def __init__(self, argv, env=None, **kwargs):
        del kwargs
        self.argv = list(argv)
        self.env = dict(env or {})
        self.pid = 10_000 + len(_FakePopen.instances)
        self.returncode: int | None = None
        self.terminated = False
        self.killed = False
        _FakePopen.instances.append(self)

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        if self.returncode is None:
            self.returncode = -signal.SIGTERM

    def kill(self):
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
    monkeypatch.setattr(dp_launcher.subprocess, "Popen", _FakePopen)
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


# ---------------------------------------------------------------------------
# Hydra compose surface
# ---------------------------------------------------------------------------


def test_offpolicy_config_devices_defaults_to_null():
    cfg = _offpolicy_cfg()
    assert cfg.training.devices is None
    assert resolve_dp_topology(cfg.training.devices) is None


def test_offpolicy_config_devices_compose():
    cfg = _offpolicy_cfg(["training.devices=[0,1]", "training.device=null"])
    assert resolve_dp_topology(cfg.training.devices) == (0, 1)


def test_training_device_and_devices_are_mutually_exclusive():
    cfg = _offpolicy_cfg(["training.devices=[0,1]", "training.device=cuda"])
    with pytest.raises(ValueError, match="mutually exclusive"):
        apply_dp_rank_config(cfg, resolve_dp_topology(cfg.training.devices), rank=0)


# ---------------------------------------------------------------------------
# apply_dp_rank_config / N=1 equivalence
# ---------------------------------------------------------------------------


def test_apply_dp_rank_config_maps_rank_to_device_and_seed():
    cfg = _offpolicy_cfg(["training.devices=[0,1]", "training.device=null", "algo.seed=42"])
    base_seed = int(cfg.algo.seed)
    apply_dp_rank_config(cfg, (0, 1), rank=1)
    assert cfg.training.device == "cuda:1"
    assert int(cfg.algo.seed) == base_seed + 1


def test_apply_dp_rank_config_rank_zero_keeps_seed():
    cfg = _offpolicy_cfg(["training.devices=[0,1]", "training.device=null", "algo.seed=42"])
    apply_dp_rank_config(cfg, (0, 1), rank=0)
    assert cfg.training.device == "cuda:0"
    assert int(cfg.algo.seed) == 42


def test_single_device_topology_spawns_no_children(fake_popen, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(UNILAB_DP_RANK, raising=False)
    cfg = _offpolicy_cfg(["training.devices=[0]", "training.device=null"])
    devices = resolve_dp_topology(cfg.training.devices)
    assert devices == (0,)
    apply_dp_rank_config(cfg, devices, rank=0)
    assert cfg.training.device == "cuda:0"
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
    with pytest.raises(ValueError, match="boom"):
        with DpRankSupervisor((0, 1, 2), log_dir="/tmp/dp_test_log"):
            raise ValueError("boom")
    assert all(child.terminated for child in fake_popen.instances)
    assert all(child.returncode == -signal.SIGTERM for child in fake_popen.instances)


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


def test_supervisor_restores_sigterm_handler(fake_popen):
    previous = signal.getsignal(signal.SIGTERM)
    with DpRankSupervisor((0, 1), log_dir="/tmp/dp_test_log"):
        assert signal.getsignal(signal.SIGTERM) is dp_launcher._sigterm_system_exit
        fake_popen.instances[0].returncode = 0
    assert signal.getsignal(signal.SIGTERM) == previous


# ---------------------------------------------------------------------------
# Hardware-gated smoke
# ---------------------------------------------------------------------------


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires >=2 CUDA devices")
@pytest.mark.slow
def test_dp_topology_validates_on_two_gpu_host():
    devices = resolve_dp_topology([0, 1])
    assert devices == (0, 1)
    validate_dp_launchable(devices)
