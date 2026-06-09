"""Stage 3: NaN guard wiring verification (mock-based assertions).

Validates that nan_guard_cfg actually flows through the collector_kwargs
of the production runners (DoubleBuffer / APPO / OffPolicy), not just
present in source as a substring.

This script constructs each runner with a NanGuardCfg, mocks dependencies,
triggers the collector startup path (via learn(max_iterations=0)), and asserts
that the captured collector kwargs contain the expected nan_guard_cfg.

Run:
    .venv/bin/python tests/nan_injection/stage3_nan_inject.py

Exit: 0 if all wirings OK, 1 if any failure.
"""

from __future__ import annotations

import queue
import sys
import tempfile
import textwrap
import traceback
from pathlib import Path
from unittest.mock import patch

# Resolve repo root from this file's location: <repo>/tests/nan_injection/<this>
ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR / "src"))

import torch

from unilab.utils.nan_guard import NanGuardCfg

# ---------------------------------------------------------------------------
# Fake classes (copied from tests/algos/*.py to keep stage3 self-contained)
# ---------------------------------------------------------------------------


class _FakeActor:
    def __init__(self):
        self._state = {"weight": torch.zeros(1)}

    def state_dict(self):
        return {key: value.clone() for key, value in self._state.items()}

    def parameters(self):
        return [torch.nn.Parameter(torch.zeros(1))]


class _FakeLearner:
    def __init__(self, *args, **kwargs):
        del args, kwargs
        self.actor = _FakeActor()
        self.critic = _FakeActor()  # APPO reads learner.critic.state_dict()
        self.update_count = 0
        self.num_learning_epochs = 1  # APPO uses this in log_status

    def get_state_dict(self):
        return {"update_count": self.update_count}


class _FakeReplayBuffer:
    def __init__(self, **kwargs):
        del kwargs
        self.size = torch.zeros(1, dtype=torch.int64)
        self.ptr = torch.zeros(1, dtype=torch.int64)
        self._storage = torch.zeros(16, 16)

    def close(self):
        pass


class _FakeWeightSync:
    def __init__(self):
        self.name = "fake-ws"
        self._lock = None

    @classmethod
    def from_state_dict(cls, state_dict, create=True):
        del state_dict, create
        return cls()

    def close(self):
        pass

    def cleanup(self):
        pass


class _FakeLogger:
    def __init__(self, **kwargs):
        del kwargs
        self._total_steps = 0
        self._buffer_size = 0
        self._mean_ep_length = 0.0

    def set_collection_sync(self, enabled, env_steps_per_sync):
        del enabled, env_steps_per_sync

    def start(self, **kwargs):
        del kwargs

    def log_status(self, status):
        del status

    def log_save(self, ckpt_path):
        del ckpt_path

    def log_collector(self, total_steps, buffer_size, mean_reward=0.0):
        del total_steps, buffer_size, mean_reward

    def log_buffer_fill(self, current, target):
        del current, target

    def update_buffer_utilization(self, utilization):
        del utilization

    def update_ep_length(self, mean_ep_length):
        del mean_ep_length

    def update_collector_timing(self, timing_ms):
        del timing_ms

    def update_done_rates(self, timeout_rate, terminated_rate):
        del timeout_rate, terminated_rate

    def update_replay_queue(self, current_len, max_size):
        del current_len, max_size

    def update_staging_pool(self, current_len, max_size):
        del current_len, max_size

    def log_step(self, **kwargs):
        del kwargs

    def finish(self, *args, **kwargs):
        del args, kwargs

    def close(self):
        pass


class _FakePipeline:
    """Mock pipeline for DoubleBuffer test."""

    def __init__(self, *args, **kwargs):
        del args, kwargs

    def close(self):
        pass


class _FakeRolloutRingBuffer:
    """Mock rollout storage for APPO test."""

    def __init__(self, **kwargs):
        del kwargs
        self.name = "fake-storage"
        self._write_ptr = object()
        self._read_ptr = object()

    @property
    def slot_shapes(self):
        return {
            "obs": (2, 4, 4),
            "critic": (2, 4, 4),
            "actions": (2, 4, 2),
            "log_probs": (2, 4),
            "rewards": (2, 4),
            "dones": (2, 4),
            "truncated": (2, 4),
            "last_obs": (2, 4),
            "last_critic": (2, 4),
        }

    def cleanup(self):
        pass


# ---------------------------------------------------------------------------
# Wiring checks
# ---------------------------------------------------------------------------


def _check_double_buffer_runner_wires_nan_guard():
    """Verify DoubleBufferOffPolicyRunner passes nan_guard_cfg to collector."""
    import unilab.algos.torch.offpolicy.double_buffer_runner as db_mod
    import unilab.algos.torch.offpolicy.runner as runner_mod

    with (
        patch.object(db_mod, "ReplayBuffer", _FakeReplayBuffer),
        patch.object(db_mod, "SharedWeightSync", _FakeWeightSync),
        patch.object(db_mod, "CPUPinnedDoubleBufferReplayPipeline", _FakePipeline),
        patch.object(runner_mod, "get_env_dims", return_value=(4, 2, 0)),
        patch.object(db_mod.torch, "save", lambda *args, **kwargs: None),
        patch.object(db_mod._SPAWN_CTX, "Queue", lambda maxsize=0: queue.Queue()),
        patch.object(db_mod.time, "sleep", lambda seconds: None),
    ):
        learner = _FakeLearner()
        nan_guard_cfg = NanGuardCfg(enabled=True)
        runner = db_mod.DoubleBufferOffPolicyRunner(
            learner=learner,
            env_name="DummyEnv",
            algo_type="sac",
            num_envs=2,
            replay_buffer_n=8,
            batch_size=8,
            learning_starts=6,
            updates_per_step=1,
            policy_frequency=1,
            sync_collection=False,
            env_steps_per_sync=1,
            device="cpu",
            nan_guard_cfg=nan_guard_cfg,
        )

        captured = {}

        def capture_start_collector(*, target_fn, kwargs):
            del target_fn
            captured.update(kwargs)

        with patch.object(runner, "_start_collector", capture_start_collector):
            with tempfile.TemporaryDirectory() as tmp_dir:
                runner.learn(max_iterations=0, save_interval=0, log_dir=tmp_dir)

        if "nan_guard_cfg" not in captured:
            return False
        if captured["nan_guard_cfg"] is not nan_guard_cfg:
            return False
        if captured["nan_guard_cfg"].enabled is not True:
            return False
        return True


def _check_appo_runner_wires_nan_guard():
    """Verify APPORunner passes nan_guard_cfg to collector."""
    import unilab.algos.torch.appo.runner as appo_mod
    from unilab.algos.torch.appo.runner import APPORunner

    def fake_detect_dims(self):
        self.critic_dim = 4
        self.critic_input_dim = 4
        return (4, 2)

    with (
        patch.object(APPORunner, "_detect_dims", fake_detect_dims),
        patch.object(APPORunner, "_build_learner", lambda self: _FakeLearner()),
        patch.object(appo_mod, "RolloutRingBuffer", _FakeRolloutRingBuffer),
        patch.object(appo_mod, "SharedWeightSync", _FakeWeightSync),
        patch.object(appo_mod, "OffPolicyLogger", _FakeLogger),
        patch.object(appo_mod.torch, "save", lambda *args, **kwargs: None),
    ):
        nan_guard_cfg = NanGuardCfg(enabled=True)
        runner = APPORunner(
            env_name="DummyEnv",
            env_cfg_overrides={},
            rl_cfg={"actor": {}, "critic": {}, "algorithm": {}},
            device="cpu",
            collector_device="cpu",
            num_envs=2,
            steps_per_env=4,
            nan_guard_cfg=nan_guard_cfg,
        )

        captured = {}

        def capture_start_collector(*, target_fn, kwargs):
            del target_fn
            captured.update(kwargs)

        with patch.object(runner, "_start_collector", capture_start_collector):
            with tempfile.TemporaryDirectory() as tmp_dir:
                runner.learn(max_iterations=0, save_interval=0, log_dir=tmp_dir)

        if "nan_guard_cfg" not in captured:
            return False
        if captured["nan_guard_cfg"] is not nan_guard_cfg:
            return False
        if captured["nan_guard_cfg"].enabled is not True:
            return False
        return True


def _check_offpolicy_runner_wires_nan_guard():
    """Verify OffPolicyRunner passes nan_guard_cfg to collector (defensive check)."""
    import unilab.algos.torch.offpolicy.runner as runner_mod

    with (
        patch.object(runner_mod, "ReplayBuffer", _FakeReplayBuffer),
        patch.object(runner_mod, "SharedWeightSync", _FakeWeightSync),
        patch.object(runner_mod, "OffPolicyLogger", _FakeLogger),
        patch.object(runner_mod, "get_env_dims", return_value=(4, 2, 0)),
        patch.object(runner_mod.torch, "save", lambda *args, **kwargs: None),
        patch.object(runner_mod._SPAWN_CTX, "Queue", lambda maxsize=0: queue.Queue()),
        patch.object(runner_mod.time, "sleep", lambda seconds: None),
    ):
        learner = _FakeLearner()
        nan_guard_cfg = NanGuardCfg(enabled=True)
        runner = runner_mod.OffPolicyRunner(
            learner=learner,
            env_name="DummyEnv",
            algo_type="sac",
            num_envs=2,
            replay_buffer_n=8,
            batch_size=8,
            learning_starts=6,
            updates_per_step=1,
            policy_frequency=1,
            sync_collection=False,
            env_steps_per_sync=1,
            device="cpu",
            nan_guard_cfg=nan_guard_cfg,
        )

        captured = {}

        def capture_start_collector(*, target_fn, kwargs):
            del target_fn
            captured.update(kwargs)

        with patch.object(runner, "_start_collector", capture_start_collector):
            with tempfile.TemporaryDirectory() as tmp_dir:
                runner.learn(max_iterations=0, save_interval=0, log_dir=tmp_dir)

        if "nan_guard_cfg" not in captured:
            return False
        if captured["nan_guard_cfg"] is not nan_guard_cfg:
            return False
        if captured["nan_guard_cfg"].enabled is not True:
            return False
        return True


# ---------------------------------------------------------------------------
# Manual end-to-end test recipe
# ---------------------------------------------------------------------------


def _print_manual_recipe():
    """Print the manual smoke test recipe (preserved from original stage3)."""
    print("=" * 78)
    print("Manual end-to-end test recipe (optional)")
    print("=" * 78)
    print(
        textwrap.dedent("""
        To exercise NaN detection end-to-end inside a collector subprocess:

        1. Pick a task env file, e.g.
           src/unilab/envs/locomotion/go1/go1_joystick.py
        2. Inside the env's update_state or apply_action, add a temporary
           one-shot NaN raise guarded by an env-counter, for example:

               if not getattr(self, "_nan_done", False) and self.step_counter >= 5:
                   state.reward[0] = float("nan")
                   self._nan_done = True

        3. Run a short training, e.g. (APPO):
               python scripts/train_appo.py task=go1_joystick_flat/mujoco \\
                 algo.num_envs=8 algo.num_steps_per_env=4 \\
                 algo.train_for_env_steps=64 \\
                 training.nan_guard.enabled=true \\
                 training.nan_guard.output_dir=/tmp/unilab/nan_dumps

        4. Verify a dump file appears under the output_dir.
        5. Revert the env edit.

        Repeat with scripts/train_offpolicy.py to exercise SAC/TD3/FlashSAC
        (set algo=sac / algo=td3 / algo=flashsac via Hydra override).
    """).strip()
    )
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("Stage 3: NaN guard wiring assertions")
    print("=" * 78)
    print()
    print("Approach: mock-based capture of collector_kwargs to verify")
    print("  that nan_guard_cfg flows from runner.__init__ to _start_collector.")
    print()

    checks = [
        (
            "DoubleBufferOffPolicyRunner (SAC/TD3/FlashSAC prod path)",
            _check_double_buffer_runner_wires_nan_guard,
        ),
        ("APPORunner (APPO prod path)", _check_appo_runner_wires_nan_guard),
        ("OffPolicyRunner (defensive)", _check_offpolicy_runner_wires_nan_guard),
    ]

    all_ok = True
    for label, fn in checks:
        try:
            ok = fn()
        except Exception as exc:
            ok = False
            print(f"  ✗ {label}")
            print(f"    Exception: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            all_ok = False
            continue

        marker = "✓" if ok else "✗"
        print(f"  {marker} {label}")
        if not ok:
            all_ok = False

    print()
    print("=" * 78)
    print("Summary:", "PASS" if all_ok else "FAIL")
    print("=" * 78)
    print()

    _print_manual_recipe()

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
