"""Repro #2 (issue #594): NanGuardCfg pickles across spawn ctx and the
guard can be constructed and attached inside the child process.

The async/double-buffer runner ships NanGuardCfg via collector_kwargs to a
spawn-context child. This test pins the prerequisite: the cfg pickles
through the spawn launcher, and a NanGuard built from it inside the child
attaches to a real NpEnv and dumps once on a NaN reward.
"""

from __future__ import annotations

import multiprocessing as mp
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple
from unittest.mock import MagicMock

import gymnasium as gym
import numpy as np

from unilab.base.base import EnvCfg
from unilab.base.np_env import NpEnv, NpEnvState
from unilab.utils.nan_guard import NanGuard, NanGuardCfg


@dataclass
class _StubCfgForSpawn(EnvCfg):
    max_episode_seconds: float | None = 1.0
    ctrl_dt: float = 0.1
    sim_dt: float = 0.01


class _StubNpEnvForSpawn(NpEnv):
    """Module-level stub so spawn-imported child can reconstruct it."""

    OBS_SPEC = {"obs": 5, "critic": 7}

    def __init__(self, num_envs: int = 4, bad_rewards: np.ndarray | None = None):
        cfg = _StubCfgForSpawn()
        backend = MagicMock()
        backend.backend_type = "mujoco"
        backend.step = MagicMock()
        super().__init__(cfg, backend, num_envs)
        self._bad_rewards = bad_rewards

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        return self.OBS_SPEC

    @property
    def action_space(self) -> gym.Space:
        return gym.spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

    def apply_action(self, actions: np.ndarray, state: NpEnvState) -> np.ndarray:
        return actions

    def update_state(self, state: NpEnvState) -> NpEnvState:
        obs = {
            "obs": np.ones((self._num_envs, 5), dtype=np.float32),
            "critic": np.full((self._num_envs, 7), 0.5, dtype=np.float32),
        }
        reward = (
            self._bad_rewards.copy()
            if self._bad_rewards is not None
            else np.ones((self._num_envs,), dtype=np.float32)
        )
        return state.replace(
            obs=obs,
            reward=reward,
            terminated=np.zeros((self._num_envs,), dtype=bool),
            truncated=np.zeros((self._num_envs,), dtype=bool),
        )

    def reset(self, env_indices: np.ndarray) -> Tuple[dict[str, np.ndarray], dict]:
        n = len(env_indices)
        obs = {
            "obs": np.zeros((n, 5), dtype=np.float32),
            "critic": np.zeros((n, 7), dtype=np.float32),
        }
        return obs, {}


def _child_attach_nan_guard(nan_guard_cfg: NanGuardCfg, result_queue, tmp_path_str: str) -> None:
    """Run inside the spawn child: rebuild guard, attach, and trigger dump."""
    try:
        # cfg pickled across the spawn boundary; build guard here
        guard = NanGuard(nan_guard_cfg, num_envs=4, supports_state_playback=False)

        bad_rewards = np.array([0.0, np.nan, 0.0, 0.0], dtype=np.float32)
        env = _StubNpEnvForSpawn(num_envs=4, bad_rewards=bad_rewards)
        env.init_state()
        env.set_nan_guard(guard)

        attached = env._nan_guard is guard

        env.step(np.zeros((4, 3), dtype=np.float64))

        dump_dir = Path(tmp_path_str) / "child_dumps"
        dump_files = list(dump_dir.glob("nan_dump_*.npz")) if dump_dir.exists() else []
        # Filter the latest symlink the same way the parent test does.
        dump_files = [p for p in dump_files if not p.is_symlink()]

        result_queue.put(
            {
                "attached": attached,
                "type_name": type(guard).__name__,
                "cfg_enabled": nan_guard_cfg.enabled,
                "dumped": len(dump_files) == 1,
            }
        )
        sys.exit(0)
    except Exception as exc:  # pragma: no cover - debugging aid
        try:
            result_queue.put({"error": f"{type(exc).__name__}: {exc}"})
        finally:
            sys.exit(1)


def test_nan_guard_cfg_pickles_across_spawn_and_attaches_in_child_process(tmp_path):
    cfg = NanGuardCfg(enabled=True, output_dir=str(tmp_path / "child_dumps"))

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    proc = ctx.Process(
        target=_child_attach_nan_guard,
        args=(cfg, q, str(tmp_path)),
    )
    proc.start()
    try:
        result = q.get(timeout=15)
    finally:
        proc.join(timeout=5)
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=2)

    assert proc.exitcode == 0, f"child exited with {proc.exitcode}; result={result}"
    assert "error" not in result, f"child raised: {result.get('error')}"
    assert result["attached"] is True
    assert result["type_name"] == "NanGuard"
    assert result["cfg_enabled"] is True
    assert result["dumped"] is True
