"""Tests for env-owned CPU block process confinement (``apply_env_cpu_runtime``).

Multi-rank off-policy collectors pin their MuJoCo pool workers to a per-rank
CPU block via ``EnvCfg.cpu_ids``; ``TorchEnv.__init__`` additionally confines the
owning process and its existing host-side threads to the same block, so they
cannot drift onto sibling ranks' CPUs. One subprocess test validates the real
placement contract.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from unittest.mock import MagicMock

import gymnasium as gym
import numpy as np
import pytest
import torch

import unilab.base.cpu_runtime as cpu_runtime
from unilab.base.base import EnvCfg
from unilab.base.cpu_runtime import apply_env_cpu_runtime
from unilab.base.torch_env import TorchEnv, TorchEnvState


def _record_affinity(monkeypatch: pytest.MonkeyPatch, available: set[int]) -> list[tuple]:
    calls: list[tuple] = []
    # raising=False: sched_*affinity is Linux-only, so the attribute may not
    # exist on the host running the tests (e.g. macOS dev machines).
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: set(available), raising=False)
    monkeypatch.setattr(
        os, "sched_setaffinity", lambda pid, ids: calls.append((pid, set(ids))), raising=False
    )
    return calls


def _record_confine(monkeypatch: pytest.MonkeyPatch) -> list[set[int]]:
    calls: list[set[int]] = []
    monkeypatch.setattr(
        cpu_runtime, "_confine_existing_threads", lambda ids: calls.append(set(ids))
    )
    return calls


def test_none_is_noop(monkeypatch: pytest.MonkeyPatch):
    affinity_calls = _record_affinity(monkeypatch, {0, 1, 2, 3})
    confine_calls = _record_confine(monkeypatch)

    apply_env_cpu_runtime(None)

    assert affinity_calls == []
    assert confine_calls == []


def test_applies_affinity_and_confines_existing_threads(monkeypatch: pytest.MonkeyPatch):
    affinity_calls = _record_affinity(monkeypatch, {0, 1, 2, 3})
    confine_calls = _record_confine(monkeypatch)

    apply_env_cpu_runtime([1, 2])

    assert affinity_calls == [(0, {1, 2})]
    assert confine_calls == [{1, 2}]


def test_unavailable_cpu_ids_fail_closed(monkeypatch: pytest.MonkeyPatch):
    affinity_calls = _record_affinity(monkeypatch, {0, 1})
    confine_calls = _record_confine(monkeypatch)

    with pytest.raises(ValueError, match="not available"):
        apply_env_cpu_runtime([1, 2])

    assert affinity_calls == []
    assert confine_calls == []


def test_platform_without_affinity_warns(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delattr(os, "sched_setaffinity", raising=False)
    monkeypatch.delattr(os, "sched_getaffinity", raising=False)
    confine_calls = _record_confine(monkeypatch)

    with pytest.warns(UserWarning, match="sched_setaffinity"):
        apply_env_cpu_runtime([0, 1])

    assert confine_calls == []


def test_confine_existing_threads_pins_tasks_and_skips_failures(
    monkeypatch: pytest.MonkeyPatch,
):
    calls: list[tuple] = []

    def fake_setaffinity(pid, ids):
        if pid == 456:
            raise ProcessLookupError
        calls.append((pid, set(ids)))

    monkeypatch.setattr(os, "sched_setaffinity", fake_setaffinity, raising=False)
    monkeypatch.setattr(os.path, "isdir", lambda path: path == cpu_runtime._PROC_TASK_DIR)
    monkeypatch.setattr(os, "listdir", lambda path: ["123", "456", "789"])

    cpu_runtime._confine_existing_threads({1, 2})

    assert calls == [(123, {1, 2}), (789, {1, 2})]


def test_confine_existing_threads_without_proc_is_noop(monkeypatch: pytest.MonkeyPatch):
    calls: list[tuple] = []
    monkeypatch.setattr(
        os, "sched_setaffinity", lambda pid, ids: calls.append((pid, set(ids))), raising=False
    )
    monkeypatch.setattr(os.path, "isdir", lambda path: False)

    cpu_runtime._confine_existing_threads({0})

    assert calls == []


# ---------------------------------------------------------------------------
# TorchEnv wiring
# ---------------------------------------------------------------------------


@dataclass
class _StubCfg(EnvCfg):
    max_episode_seconds: float | None = 1.0


class _StubTorchEnv(TorchEnv):
    def __init__(self, cfg: EnvCfg):
        backend = MagicMock()
        backend.get_scene_model_file.return_value = None
        super().__init__(cfg, backend, 1)

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        return {"obs": 1}

    @property
    def action_space(self) -> gym.Space:
        return gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    def apply_action(self, actions: torch.Tensor, state: TorchEnvState) -> torch.Tensor:
        return actions

    def update_state(self, state: TorchEnvState) -> TorchEnvState:
        return state


@pytest.mark.parametrize("cpu_ids", (None, [2, 3]))
def test_torch_env_init_applies_env_cpu_runtime(monkeypatch: pytest.MonkeyPatch, cpu_ids):
    import unilab.base.torch_env as torch_env_module

    calls: list[list[int] | None] = []
    monkeypatch.setattr(
        torch_env_module,
        "apply_env_cpu_runtime",
        lambda value: calls.append(None if value is None else list(value)),
    )

    _StubTorchEnv(EnvCfg(cpu_ids=cpu_ids))

    assert calls == [cpu_ids]


# ---------------------------------------------------------------------------
# Real placement contract in a fresh process
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not (hasattr(os, "sched_setaffinity") and os.path.isdir("/proc/self/task")),
    reason="requires Linux sched affinity and /proc",
)
def test_existing_threads_inherit_confined_block_in_fresh_process():
    script = r"""
import json
import os

# Production collectors import NumPy before env construction, so the OpenBLAS
# pool predates the env hook and must be confined retroactively.
import numpy as np

from unilab.base.cpu_runtime import apply_env_cpu_runtime

block = sorted(os.sched_getaffinity(0))[:2]
apply_env_cpu_runtime(block)
np.zeros(256)  # keep the host BLAS import observable


def _expand(mask):
    cpus = set()
    for part in mask.split(","):
        if "-" in part:
            lo, hi = part.split("-", 1)
            cpus.update(range(int(lo), int(hi) + 1))
        elif part:
            cpus.add(int(part))
    return sorted(cpus)


masks = []
for tid in os.listdir("/proc/self/task"):
    with open(f"/proc/self/task/{tid}/status") as fh:
        for line in fh:
            if line.startswith("Cpus_allowed_list"):
                masks.append(_expand(line.split(":", 1)[1].strip()))

print(
    "RESULT:"
    + json.dumps({"block": block, "affinity": sorted(os.sched_getaffinity(0)), "masks": masks})
)
"""
    env = dict(os.environ)
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=300,
        env=env,
    )
    assert proc.returncode == 0, proc.stderr
    result_lines = [line for line in proc.stdout.splitlines() if line.startswith("RESULT:")]
    assert len(result_lines) == 1, proc.stdout
    payload = json.loads(result_lines[0].removeprefix("RESULT:"))
    assert payload["affinity"] == payload["block"]
    # Every already-running thread (including the OpenBLAS pool spawned at
    # import) must stay inside the env-owned block.
    assert payload["masks"]
    for mask in payload["masks"]:
        assert set(mask) <= set(payload["block"])
