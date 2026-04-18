from __future__ import annotations

from typing import Any

import gymnasium as gym
import pytest


class _FakeSymmetryAugmentation:
    batch_multiplier = 2

    def augment_obs_and_actions(self, obs, actions, *, obs_group: str = "obs"):
        return obs, actions

    def mirror_obs(self, obs, *, obs_group: str = "obs"):
        return obs


class _ForbiddenBackend:
    @property
    def model(self):
        raise AssertionError("FastSAC runner should not read env._backend.model")


class _FakeEnv:
    def __init__(self, augmentation: Any | None):
        self.obs_groups_spec = {"obs": 4, "critic": 6}
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,))
        self._backend = _ForbiddenBackend()
        self._augmentation = augmentation
        self.closed = False
        self.last_device: str | None = None

    def get_obs_structure(self):
        raise AssertionError("FastSAC runner should not call env.get_obs_structure()")

    def build_symmetry_augmentation(self, *, device: str):
        self.last_device = device
        return self._augmentation

    def close(self):
        self.closed = True


def test_fast_sac_runner_uses_env_owned_symmetry_contract(monkeypatch: pytest.MonkeyPatch):
    from unilab.algos.torch.fast_sac.runner import FastSACRunner
    from unilab.base import registry
    from unilab.utils import algo_utils

    augmentation = _FakeSymmetryAugmentation()
    fake_env = _FakeEnv(augmentation)

    monkeypatch.setattr(algo_utils, "ensure_registries", lambda: None)
    monkeypatch.setattr(registry, "make", lambda *args, **kwargs: fake_env)

    runner = FastSACRunner(
        env_name="FakeEnv",
        device="cpu",
        num_envs=1,
        replay_buffer_n=8,
        batch_size=8,
        learning_starts=0,
        updates_per_step=1,
        policy_frequency=1,
        use_symmetry=True,
        obs_normalization=False,
    )

    assert fake_env.closed is True
    assert fake_env.last_device == "cpu"
    assert runner.batch_size == 4
    assert runner.learner.symmetry is augmentation


def test_fast_sac_runner_skips_symmetry_builder_when_disabled(monkeypatch: pytest.MonkeyPatch):
    from unilab.algos.torch.fast_sac.runner import FastSACRunner
    from unilab.base import registry
    from unilab.utils import algo_utils

    fake_env = _FakeEnv(_FakeSymmetryAugmentation())

    def _unexpected_builder(*args, **kwargs):
        raise AssertionError("Symmetry builder should not be called when use_symmetry is false")

    fake_env.build_symmetry_augmentation = _unexpected_builder  # type: ignore[method-assign]

    monkeypatch.setattr(algo_utils, "ensure_registries", lambda: None)
    monkeypatch.setattr(registry, "make", lambda *args, **kwargs: fake_env)

    runner = FastSACRunner(
        env_name="FakeEnv",
        device="cpu",
        num_envs=1,
        replay_buffer_n=8,
        batch_size=8,
        learning_starts=0,
        updates_per_step=1,
        policy_frequency=1,
        use_symmetry=False,
        obs_normalization=False,
    )

    assert fake_env.closed is True
    assert runner.batch_size == 8
    assert runner.learner.symmetry is None
