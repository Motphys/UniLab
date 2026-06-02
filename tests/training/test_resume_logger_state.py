"""Regression tests for issue #441 — TensorBoard time-axis resets after resume.

The bug surface is ``rsl_rl.utils.logger.Logger.tot_time`` /
``tot_timesteps`` being used as the step for ``Train/mean_reward/time`` and
``Train/mean_episode_length/time`` while ``OnPolicyRunner.load`` does not
restore them. ``patch_rsl_rl_resume_state`` round-trips both counters via the
saved checkpoint's ``unilab_logger_state`` key.

These tests bypass ``OnPolicyRunner.__init__`` (which needs a full VecEnv and
PPO algorithm) by stubbing the few attributes ``save`` / ``load`` actually
touch. That keeps the patch surface — not the integration — under test, and
avoids pulling MuJoCo or simulator dependencies into a unit test.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

pytest.importorskip("rsl_rl")

from rsl_rl.runners.on_policy_runner import OnPolicyRunner  # noqa: E402

from unilab.training.experiment import patch_rsl_rl_resume_state


class _AlgStub:
    def __init__(self, returns_iter: bool = True) -> None:
        self._returns_iter = returns_iter

    def save(self) -> dict:
        return {
            "actor_state_dict": {},
            "critic_state_dict": {},
            "optimizer_state_dict": {},
        }

    def load(self, _loaded_dict: dict, _load_cfg: dict | None, _strict: bool) -> bool:
        return self._returns_iter


class _LoggerStub:
    def __init__(self) -> None:
        self.tot_time = 0.0
        self.tot_timesteps = 0
        self.saved_models: list[tuple[str, int]] = []

    def save_model(self, path: str, it: int) -> None:
        self.saved_models.append((path, it))


def _make_stub_runner(
    *,
    iter: int = 0,
    tot_time: float = 0.0,
    tot_timesteps: int = 0,
    alg_returns_iter: bool = True,
) -> Any:
    runner = OnPolicyRunner.__new__(OnPolicyRunner)
    runner.alg = _AlgStub(returns_iter=alg_returns_iter)  # type: ignore[assignment]
    runner.logger = _LoggerStub()  # type: ignore[assignment]
    runner.logger.tot_time = tot_time
    runner.logger.tot_timesteps = tot_timesteps
    runner.current_learning_iteration = iter
    return runner


def test_patch_is_idempotent() -> None:
    patch_rsl_rl_resume_state()
    save_after_first = OnPolicyRunner.save
    load_after_first = OnPolicyRunner.load
    patch_rsl_rl_resume_state()
    assert OnPolicyRunner.save is save_after_first
    assert OnPolicyRunner.load is load_after_first
    assert getattr(OnPolicyRunner, "_UNILAB_RESUME_PATCHED", False) is True


def test_save_round_trip_restores_logger_counters(tmp_path: Path) -> None:
    patch_rsl_rl_resume_state()

    saver = _make_stub_runner(iter=100, tot_time=987.5, tot_timesteps=4096 * 100)
    ckpt = tmp_path / "model_100.pt"
    saver.save(str(ckpt))

    # The unilab key must land in the file so external readers (e.g. analysis
    # scripts) can recover wall-clock progress without rerunning the trainer.
    raw = torch.load(str(ckpt), weights_only=False)
    assert "unilab_logger_state" in raw
    assert raw["unilab_logger_state"]["tot_time"] == pytest.approx(987.5)
    assert raw["unilab_logger_state"]["tot_timesteps"] == 4096 * 100
    assert raw["iter"] == 100
    assert saver.logger.saved_models == [(str(ckpt), 100)]

    loader = _make_stub_runner(iter=0, tot_time=0.0, tot_timesteps=0)
    loader.load(str(ckpt))

    assert loader.current_learning_iteration == 100
    assert loader.logger.tot_time == pytest.approx(987.5)
    assert loader.logger.tot_timesteps == 4096 * 100


def test_load_legacy_checkpoint_without_unilab_state(tmp_path: Path) -> None:
    """Pre-patch checkpoints must still load; logger counters keep their defaults."""
    patch_rsl_rl_resume_state()

    legacy = {
        "actor_state_dict": {},
        "critic_state_dict": {},
        "optimizer_state_dict": {},
        "iter": 50,
        "infos": None,
    }
    ckpt = tmp_path / "model_50_legacy.pt"
    torch.save(legacy, str(ckpt))

    loader = _make_stub_runner(iter=0, tot_time=0.0, tot_timesteps=0)
    loader.load(str(ckpt))

    assert loader.current_learning_iteration == 50
    assert loader.logger.tot_time == 0.0
    assert loader.logger.tot_timesteps == 0


def test_load_does_not_overwrite_iter_when_alg_skips_iteration(tmp_path: Path) -> None:
    """If ``alg.load`` returns False the runner's iter must not be clobbered.

    Mirrors the upstream behaviour at ``OnPolicyRunner.load`` so the patch
    does not change the contract for partial loads (e.g. policy-only).
    """
    patch_rsl_rl_resume_state()

    saver = _make_stub_runner(iter=200, tot_time=12.0, tot_timesteps=10)
    ckpt = tmp_path / "model_200.pt"
    saver.save(str(ckpt))

    loader = _make_stub_runner(iter=7, alg_returns_iter=False)
    loader.load(str(ckpt))

    assert loader.current_learning_iteration == 7  # untouched
    # Logger state still restored — wall-clock is independent of alg load_cfg.
    assert loader.logger.tot_time == pytest.approx(12.0)
    assert loader.logger.tot_timesteps == 10
