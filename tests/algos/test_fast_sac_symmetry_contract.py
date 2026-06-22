from __future__ import annotations

from typing import Any

import gymnasium as gym
import pytest
import torch

import unilab.algos.torch.fast_sac.learner as learner_module


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

    augmentation = _FakeSymmetryAugmentation()
    fake_env = _FakeEnv(augmentation)

    monkeypatch.setattr(registry, "ensure_registries", lambda: None)
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

    fake_env = _FakeEnv(_FakeSymmetryAugmentation())

    def _unexpected_builder(*args, **kwargs):
        raise AssertionError("Symmetry builder should not be called when use_symmetry is false")

    fake_env.build_symmetry_augmentation = _unexpected_builder  # type: ignore[method-assign]

    monkeypatch.setattr(registry, "ensure_registries", lambda: None)
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


def test_fast_sac_learner_rejects_symmetry_without_augmentation():
    from unilab.algos.torch.fast_sac.learner import FastSACLearner

    with pytest.raises(
        ValueError,
        match="FastSACLearner use_symmetry=True requires a symmetry_augmentation contract",
    ):
        FastSACLearner(
            obs_dim=4,
            action_dim=2,
            critic_obs_dim=4,
            device="cpu",
            use_symmetry=True,
        )


def test_fast_sac_learner_exposes_multi_gpu_initial_sync_contract():
    from unilab.algos.torch.fast_sac.learner import FastSACLearner

    learner = FastSACLearner(
        obs_dim=4,
        action_dim=2,
        critic_obs_dim=4,
        device="cpu",
        world_size=1,
    )

    assert callable(getattr(learner, "sync_initial_parameters", None))
    learner.sync_initial_parameters(src=0)


def test_multi_gpu_offpolicy_runner_rejects_sac_symmetry_capability():
    from unilab.algos.torch.offpolicy.multi_gpu_runner import MultiGPUOffPolicyRunner

    with pytest.raises(
        ValueError,
        match="Off-policy symmetry augmentation does not support training.num_gpus > 1",
    ):
        MultiGPUOffPolicyRunner.validate_capabilities(
            algo_type="sac",
            learner_kwargs={"use_symmetry": True},
            num_gpus=2,
        )


@pytest.mark.parametrize(
    ("algo_type", "learner_kwargs", "num_gpus"),
    [
        ("sac", {"use_symmetry": False}, 2),
        ("sac", {"use_symmetry": True}, 1),
        ("td3", {"use_symmetry": True}, 2),
    ],
)
def test_multi_gpu_offpolicy_runner_allows_supported_capabilities(
    algo_type: str,
    learner_kwargs: dict[str, bool],
    num_gpus: int,
):
    from unilab.algos.torch.offpolicy.multi_gpu_runner import MultiGPUOffPolicyRunner

    MultiGPUOffPolicyRunner.validate_capabilities(
        algo_type=algo_type,
        learner_kwargs=learner_kwargs,
        num_gpus=num_gpus,
    )


def test_fast_sac_local_sgd_skips_per_update_gradient_all_reduce(
    monkeypatch: pytest.MonkeyPatch,
):
    from unilab.algos.torch.fast_sac.learner import FastSACLearner

    learner = FastSACLearner(
        obs_dim=4,
        critic_obs_dim=5,
        action_dim=2,
        device="cpu",
        world_size=2,
        distributed_sync_mode="local_sgd",
    )
    calls = 0

    def fake_all_reduce(tensor, op=None):
        del tensor, op
        nonlocal calls
        calls += 1

    monkeypatch.setattr(learner_module.dist, "all_reduce", fake_all_reduce)
    for param in learner.qnet.parameters():
        param.grad = torch.ones_like(param)

    learner._reduce_gradients(learner.qnet)

    assert calls == 0


def test_fast_sac_local_sgd_parameter_average_uses_single_flat_all_reduce(
    monkeypatch: pytest.MonkeyPatch,
):
    from unilab.algos.torch.fast_sac.learner import FastSACLearner

    learner = FastSACLearner(
        obs_dim=4,
        critic_obs_dim=5,
        action_dim=2,
        device="cpu",
        world_size=2,
        distributed_sync_mode="local_sgd",
    )
    seen_sizes: list[int] = []

    def fake_all_reduce(tensor, op=None):
        del op
        seen_sizes.append(tensor.numel())
        tensor.mul_(2.0)

    monkeypatch.setattr(learner_module.dist, "all_reduce", fake_all_reduce)
    before = learner.log_alpha.detach().clone()

    learner.average_distributed_parameters()

    assert len(seen_sizes) == 1
    assert seen_sizes[0] > 0
    assert torch.allclose(learner.log_alpha, before)
