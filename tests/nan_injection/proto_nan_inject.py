"""
Prototype script for stage 2: validate that NaN injection at three different
points (obs / reward / ctrl) actually triggers nan_guard.dump on a real np_env
instance.

This script uses the existing _StubNpEnv pattern from tests/base/test_np_env.py
to keep things minimal — no full algo runner needed for the injection mechanic
itself. Once validated here, the same patch helpers will be lifted into
tests/algos/test_nan_inject_rsl_rl.py with a real PPO/HIM-PPO runner.

Run:
    .venv/bin/python tests/nan_injection/proto_nan_inject.py
"""

from __future__ import annotations

import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock

import gymnasium as gym
import numpy as np

# Resolve repo root from this file's location
ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR / "src"))

from unilab.base.base import EnvCfg
from unilab.base.np_env import NpEnv, NpEnvState
from unilab.utils.nan_guard import NanGuard, NanGuardCfg


@dataclass
class _StubCfg(EnvCfg):
    max_episode_seconds: float | None = 1.0
    ctrl_dt: float = 0.1
    sim_dt: float = 0.01


class _StubEnv(NpEnv):
    OBS_SPEC = {"obs": 5, "critic": 7}

    def __init__(self, num_envs: int = 4):
        cfg = _StubCfg()
        backend = MagicMock()
        backend.backend_type = "mujoco"
        backend.step = MagicMock(return_value=None)
        super().__init__(cfg, backend, num_envs)

    @property
    def obs_groups_spec(self):
        return self.OBS_SPEC

    @property
    def action_space(self):
        return gym.spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

    def apply_action(self, actions: np.ndarray, state):
        return actions

    def update_state(self, state):
        obs = {
            "obs": np.ones((self._num_envs, 5), dtype=np.float32),
            "critic": np.full((self._num_envs, 7), 0.5, dtype=np.float32),
        }
        return state.replace(
            obs=obs,
            reward=np.ones((self._num_envs,), dtype=np.float32),
            terminated=np.zeros((self._num_envs,), dtype=bool),
            truncated=np.zeros((self._num_envs,), dtype=bool),
        )

    def reset(self, env_indices):
        n = len(env_indices)
        obs = {
            "obs": np.zeros((n, 5), dtype=np.float32),
            "critic": np.zeros((n, 7), dtype=np.float32),
        }
        return obs, {}


def _attach_guard(env: _StubEnv, output_dir: Path) -> NanGuard:
    cfg = NanGuardCfg(enabled=True, output_dir=str(output_dir))
    guard = NanGuard(cfg, num_envs=env._num_envs, supports_state_playback=False)
    env.set_nan_guard(guard)
    return guard


def _run_steps(env: _StubEnv, n: int):
    env.init_state()
    actions = np.zeros((env._num_envs, 3), dtype=np.float32)
    for _ in range(n):
        env.step(actions)


def _list_dumps(output_dir: Path):
    """Return only timestamped dump files, excluding the 'nan_dump_latest.npz' pointer."""
    return [p for p in output_dir.glob("nan_dump_*.npz") if "latest" not in p.name]


def proto_obs_nan():
    """Patch env.update_state to write NaN into obs at step K."""
    print("=" * 60)
    print("[1] OBS NaN injection via update_state patch")
    print("=" * 60)
    with tempfile.TemporaryDirectory() as td:
        output_dir = Path(td)
        env = _StubEnv(num_envs=4)
        guard = _attach_guard(env, output_dir)

        orig_update = env.update_state
        K = 2
        calls = [0]

        def update_with_nan(state):
            new_state = orig_update(state)
            calls[0] += 1
            if calls[0] == K:
                new_state.obs["obs"][0, 0] = np.nan
            return new_state

        env.update_state = update_with_nan  # type: ignore

        _run_steps(env, n=4)

        dumped = bool(guard._dumped)
        files = _list_dumps(output_dir)
        print(f"  guard._dumped: {dumped}")
        print(f"  dump files: {[f.name for f in files]}")
        assert dumped, "obs NaN should have triggered dump"
        assert len(files) == 1, f"expected 1 dump, got {len(files)}"
        data = np.load(files[0], allow_pickle=True)
        print(f"  npz keys: {list(data.files)}")
        print("  PASS")


def proto_reward_nan():
    """Patch env.update_state to write NaN into reward at step K."""
    print("=" * 60)
    print("[2] REWARD NaN injection via update_state patch")
    print("=" * 60)
    with tempfile.TemporaryDirectory() as td:
        output_dir = Path(td)
        env = _StubEnv(num_envs=4)
        guard = _attach_guard(env, output_dir)

        orig_update = env.update_state
        K = 2
        calls = [0]

        def update_with_nan(state):
            new_state = orig_update(state)
            calls[0] += 1
            if calls[0] == K:
                new_state.reward[0] = np.nan
            return new_state

        env.update_state = update_with_nan  # type: ignore

        _run_steps(env, n=4)

        dumped = bool(guard._dumped)
        files = _list_dumps(output_dir)
        print(f"  guard._dumped: {dumped}")
        print(f"  dump files: {[f.name for f in files]}")
        assert dumped, "reward NaN should have triggered dump"
        assert len(files) == 1, f"expected 1 dump, got {len(files)}"
        # Verify reward was nan_to_num'd after dump (sanitize should still run)
        # Note: dump only fires once, so post-dump steps continue cleanly.
        print("  PASS")


def proto_ctrl_nan():
    """Patch env.apply_action to return NaN ctrl at step K (validates check_ctrl)."""
    print("=" * 60)
    print("[3] CTRL NaN injection via apply_action patch")
    print("=" * 60)
    with tempfile.TemporaryDirectory() as td:
        output_dir = Path(td)
        env = _StubEnv(num_envs=4)
        guard = _attach_guard(env, output_dir)

        orig_apply = env.apply_action
        K = 2
        calls = [0]

        def apply_with_nan(actions, state):
            ctrl = orig_apply(actions, state)
            calls[0] += 1
            if calls[0] == K:
                ctrl = ctrl.copy()
                ctrl[0, 0] = np.nan
            return ctrl

        env.apply_action = apply_with_nan  # type: ignore

        _run_steps(env, n=4)

        dumped = bool(guard._dumped)
        files = _list_dumps(output_dir)
        print(f"  guard._dumped: {dumped}")
        print(f"  dump files: {[f.name for f in files]}")
        assert dumped, "ctrl NaN should have triggered dump (check_ctrl path)"
        assert len(files) == 1, f"expected 1 dump, got {len(files)}"
        print("  PASS")


def proto_clean_no_dump():
    """Negative control: clean run should NOT produce any dump."""
    print("=" * 60)
    print("[4] CLEAN run — no NaN, no dump expected")
    print("=" * 60)
    with tempfile.TemporaryDirectory() as td:
        output_dir = Path(td)
        env = _StubEnv(num_envs=4)
        guard = _attach_guard(env, output_dir)

        _run_steps(env, n=4)

        dumped = bool(guard._dumped)
        files = _list_dumps(output_dir)
        print(f"  guard._dumped: {dumped}")
        print(f"  dump files: {[f.name for f in files]}")
        assert not dumped, "clean run should NOT trigger dump"
        assert len(files) == 0, f"expected 0 dumps, got {len(files)}"
        print("  PASS")


if __name__ == "__main__":
    proto_obs_nan()
    proto_reward_nan()
    proto_ctrl_nan()
    proto_clean_no_dump()
    print()
    print("All 4 prototypes PASSED. Injection mechanics validated.")
