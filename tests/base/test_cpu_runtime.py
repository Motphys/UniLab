"""Tests for env-owned CPU block process confinement (``apply_env_cpu_runtime``).

Multi-rank off-policy collectors pin their MuJoCo pool workers to a per-rank
CPU block via ``EnvCfg.cpu_ids``; ``NpEnv.__init__`` additionally confines the
owning process to the same block and sizes Numba's parallel pool to it, so
host-side kernels cannot drift onto sibling ranks' CPUs. Unit tests mock the
OS/Numba seams; one subprocess test validates the real placement contract.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from unittest.mock import MagicMock

import gymnasium as gym
import numba
import numpy as np
import pytest

import unilab.base.cpu_runtime as cpu_runtime
from unilab.base.base import EnvCfg
from unilab.base.cpu_runtime import apply_env_cpu_runtime
from unilab.base.np_env import NpEnv, NpEnvState


def _record_affinity(monkeypatch: pytest.MonkeyPatch, available: set[int]) -> list[tuple]:
    calls: list[tuple] = []
    # raising=False: sched_*affinity is Linux-only, so the attribute may not
    # exist on the host running the tests (e.g. macOS dev machines).
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: set(available), raising=False)
    monkeypatch.setattr(
        os, "sched_setaffinity", lambda pid, ids: calls.append((pid, set(ids))), raising=False
    )
    return calls


def _record_numba(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    calls: list[int] = []
    monkeypatch.setattr(numba, "set_num_threads", lambda n: calls.append(int(n)))
    return calls


def _record_confine(monkeypatch: pytest.MonkeyPatch) -> list[set[int]]:
    calls: list[set[int]] = []
    monkeypatch.setattr(
        cpu_runtime, "_confine_existing_threads", lambda ids: calls.append(set(ids))
    )
    return calls


def test_none_is_noop(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("NUMBA_NUM_THREADS", raising=False)
    affinity_calls = _record_affinity(monkeypatch, {0, 1, 2, 3})
    confine_calls = _record_confine(monkeypatch)
    numba_calls = _record_numba(monkeypatch)

    apply_env_cpu_runtime(None)

    assert affinity_calls == []
    assert confine_calls == []
    assert numba_calls == []


def test_applies_affinity_and_numba_cap(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("NUMBA_NUM_THREADS", raising=False)
    affinity_calls = _record_affinity(monkeypatch, {0, 1, 2, 3})
    confine_calls = _record_confine(monkeypatch)
    numba_calls = _record_numba(monkeypatch)

    apply_env_cpu_runtime([1, 2])

    assert affinity_calls == [(0, {1, 2})]
    assert confine_calls == [{1, 2}]
    assert numba_calls == [2]


def test_respects_explicit_numba_num_threads(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("NUMBA_NUM_THREADS", "4")
    affinity_calls = _record_affinity(monkeypatch, {0, 1, 2, 3})
    confine_calls = _record_confine(monkeypatch)
    numba_calls = _record_numba(monkeypatch)

    apply_env_cpu_runtime([1, 2])

    assert affinity_calls == [(0, {1, 2})]
    assert confine_calls == [{1, 2}]
    assert numba_calls == []


def test_unavailable_cpu_ids_fail_closed(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("NUMBA_NUM_THREADS", raising=False)
    affinity_calls = _record_affinity(monkeypatch, {0, 1})
    confine_calls = _record_confine(monkeypatch)
    numba_calls = _record_numba(monkeypatch)

    with pytest.raises(ValueError, match="not available"):
        apply_env_cpu_runtime([1, 2])

    assert affinity_calls == []
    assert confine_calls == []
    assert numba_calls == []


def test_platform_without_affinity_warns_and_caps_numba(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("NUMBA_NUM_THREADS", raising=False)
    monkeypatch.delattr(os, "sched_setaffinity", raising=False)
    monkeypatch.delattr(os, "sched_getaffinity", raising=False)
    confine_calls = _record_confine(monkeypatch)
    numba_calls = _record_numba(monkeypatch)

    with pytest.warns(UserWarning, match="sched_setaffinity"):
        apply_env_cpu_runtime([0, 1])

    assert confine_calls == []
    assert numba_calls == [2]


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
# NpEnv wiring
# ---------------------------------------------------------------------------


@dataclass
class _StubCfg(EnvCfg):
    max_episode_seconds: float | None = 1.0


class _StubNpEnv(NpEnv):
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

    def apply_action(self, actions: np.ndarray, state: NpEnvState) -> np.ndarray:
        return actions

    def update_state(self, state: NpEnvState) -> NpEnvState:
        return state


@pytest.mark.parametrize("cpu_ids", (None, [2, 3]))
def test_np_env_init_applies_env_cpu_runtime(monkeypatch: pytest.MonkeyPatch, cpu_ids):
    import unilab.base.np_env as np_env_module

    calls: list[list[int] | None] = []
    monkeypatch.setattr(
        np_env_module,
        "apply_env_cpu_runtime",
        lambda value: calls.append(None if value is None else list(value)),
    )

    _StubNpEnv(EnvCfg(cpu_ids=cpu_ids))

    assert calls == [cpu_ids]


# ---------------------------------------------------------------------------
# Real placement contract in a fresh process
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not (hasattr(os, "sched_setaffinity") and os.path.isdir("/proc/self/task")),
    reason="requires Linux sched affinity and /proc",
)
def test_numba_threads_inherit_confined_block_in_fresh_process():
    script = r"""
import json
import os

# Production collectors import numpy/numba (through the backend modules)
# before env construction, so mirror that ordering here: the OpenBLAS pool
# spawned at `import numpy` predates the env hook and must be confined
# retroactively, while Numba's pool launches after it and inherits the mask.
import numba  # noqa: F401
import numpy as np

from unilab.base.cpu_runtime import apply_env_cpu_runtime

block = sorted(os.sched_getaffinity(0))[:2]
apply_env_cpu_runtime(block)

from numba import get_num_threads, njit, prange


@njit(parallel=True)
def _probe(out):
    for i in prange(out.shape[0]):
        out[i] = i * 2.0


_probe(np.zeros(256))


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
    + json.dumps(
        {
            "block": block,
            "affinity": sorted(os.sched_getaffinity(0)),
            "numba_threads": get_num_threads(),
            "masks": masks,
        }
    )
)
"""
    env = dict(os.environ)
    env.pop("NUMBA_NUM_THREADS", None)
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
    assert payload["numba_threads"] == len(payload["block"])
    # Every thread in the process must stay inside the block: the main thread,
    # the OpenBLAS pool spawned at import (confined retroactively), and Numba's
    # pool (inherits the confined mask at its lazy launch).
    assert payload["masks"]
    for mask in payload["masks"]:
        assert set(mask) <= set(payload["block"])
