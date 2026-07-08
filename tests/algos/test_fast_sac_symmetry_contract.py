from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import gymnasium as gym
import pytest
import torch

import unilab.algos.torch.fast_sac.learner as learner_module


class _FakeSymmetryAugmentation:
    batch_multiplier = 2

    def __init__(self):
        self.augment_obs_calls: list[str] = []
        self.augment_obs_and_actions_calls: list[str] = []
        self.mirror_obs_calls: list[str] = []

    def augment_obs(self, obs, *, obs_group: str = "obs"):
        self.augment_obs_calls.append(obs_group)
        return torch.cat([obs, obs], dim=0)

    def augment_obs_and_actions(self, obs, actions, *, obs_group: str = "obs"):
        self.augment_obs_and_actions_calls.append(obs_group)
        return torch.cat([obs, obs], dim=0), torch.cat([actions, actions], dim=0)

    def mirror_obs(self, obs, *, obs_group: str = "obs"):
        self.mirror_obs_calls.append(obs_group)
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


def test_fast_sac_obs_normalization_uses_global_distributed_moments(
    monkeypatch: pytest.MonkeyPatch,
):
    from unilab.algos.torch.fast_sac.learner import FastSACLearner

    learner = FastSACLearner(
        obs_dim=2,
        action_dim=1,
        critic_obs_dim=3,
        device="cpu",
        actor_hidden_dim=8,
        critic_hidden_dim=8,
        num_atoms=3,
        num_q_networks=2,
        use_layer_norm=False,
        use_autotune=False,
        obs_normalization=True,
        world_size=2,
    )

    monkeypatch.setattr(learner_module.dist, "is_available", lambda: True)
    monkeypatch.setattr(learner_module.dist, "is_initialized", lambda: True)

    def fake_all_reduce(tensor, op=None):
        del op
        remote_moments = torch.tensor([12.0, 16.0, 74.0, 130.0, 2.0])
        tensor.add_(remote_moments)

    monkeypatch.setattr(learner_module.dist, "all_reduce", fake_all_reduce)

    learner.normalize_obs(torch.tensor([[1.0, 3.0], [3.0, 5.0]]), update=True)

    torch.testing.assert_close(learner.obs_normalizer.mean, torch.tensor([4.0, 6.0]))
    torch.testing.assert_close(
        learner.obs_normalizer.std,
        torch.full((2,), 5.0**0.5),
    )
    assert int(learner.obs_normalizer.count.item()) == 4

    restored = FastSACLearner(
        obs_dim=2,
        action_dim=1,
        critic_obs_dim=3,
        device="cpu",
        actor_hidden_dim=8,
        critic_hidden_dim=8,
        num_atoms=3,
        num_q_networks=2,
        use_layer_norm=False,
        use_autotune=False,
        obs_normalization=True,
    )
    restored.load_state_dict(learner.get_state_dict())
    torch.testing.assert_close(restored.obs_normalizer.mean, learner.obs_normalizer.mean)


def test_multi_gpu_offpolicy_runner_rejects_sac_symmetry_capability():
    from unilab.algos.torch.fast_sac.learner import FastSACLearner
    from unilab.algos.torch.offpolicy.multi_gpu_runner import MultiGPUOffPolicyRunner

    with pytest.raises(
        ValueError,
        match="Off-policy symmetry augmentation does not support training.num_gpus > 1",
    ):
        MultiGPUOffPolicyRunner.validate_capabilities(
            algo_type="sac",
            learner_cls=FastSACLearner,
            learner_kwargs={"use_symmetry": True},
            num_gpus=2,
        )


@pytest.mark.parametrize(
    ("learner_kwargs", "num_gpus"),
    [
        ({"use_symmetry": False}, 2),
        ({"use_symmetry": True}, 1),
    ],
)
def test_multi_gpu_offpolicy_runner_allows_supported_capabilities(
    learner_kwargs: dict[str, bool],
    num_gpus: int,
):
    from unilab.algos.torch.fast_sac.learner import FastSACLearner
    from unilab.algos.torch.offpolicy.multi_gpu_runner import MultiGPUOffPolicyRunner

    MultiGPUOffPolicyRunner.validate_capabilities(
        algo_type="sac",
        learner_cls=FastSACLearner,
        learner_kwargs=learner_kwargs,
        num_gpus=num_gpus,
    )


def test_multi_gpu_offpolicy_runner_rejects_unsupported_learner_capability():
    from unilab.algos.torch.fast_td3.learner import FastTD3Learner
    from unilab.algos.torch.offpolicy.multi_gpu_runner import MultiGPUOffPolicyRunner

    with pytest.raises(ValueError, match="FastTD3Learner.*does not support training.num_gpus"):
        MultiGPUOffPolicyRunner.validate_capabilities(
            algo_type="td3",
            learner_cls=FastTD3Learner,
            learner_kwargs={},
            num_gpus=2,
        )


def test_multi_gpu_offpolicy_runner_rejects_unsupported_sync_mode():
    from unilab.algos.torch.fast_sac.learner import FastSACLearner
    from unilab.algos.torch.offpolicy.multi_gpu_runner import MultiGPUOffPolicyRunner

    with pytest.raises(ValueError, match="training.multi_gpu_sync_mode must be one of"):
        MultiGPUOffPolicyRunner.validate_capabilities(
            algo_type="sac",
            learner_cls=FastSACLearner,
            learner_kwargs={"use_symmetry": False},
            num_gpus=2,
            sync_mode="bogus",
        )


def test_multi_gpu_offpolicy_runner_normalizes_sync_mode_before_capability_check():
    from unilab.algos.torch.fast_sac.learner import FastSACLearner
    from unilab.algos.torch.offpolicy.multi_gpu_runner import MultiGPUOffPolicyRunner

    MultiGPUOffPolicyRunner.validate_capabilities(
        algo_type="sac",
        learner_cls=FastSACLearner,
        learner_kwargs={"use_symmetry": False},
        num_gpus=2,
        sync_mode="LOCAL_SGD",
    )


def test_multi_gpu_offpolicy_runner_requires_direct_learner_opt_in():
    from unilab.algos.torch.fast_sac.learner import FastSACLearner
    from unilab.algos.torch.offpolicy.multi_gpu_runner import MultiGPUOffPolicyRunner

    class CustomSAC(FastSACLearner):
        pass

    with pytest.raises(ValueError, match="CustomSAC.*does not support training.num_gpus"):
        MultiGPUOffPolicyRunner.validate_capabilities(
            algo_type="custom_sac",
            learner_cls=CustomSAC,
            learner_kwargs={"use_symmetry": False},
            num_gpus=2,
        )


def test_multi_gpu_offpolicy_runner_rejects_missing_distributed_hooks_before_spawn():
    from unilab.algos.torch.offpolicy.multi_gpu_runner import MultiGPUOffPolicyRunner

    class IncompleteLearner:
        supports_multi_gpu = True
        supports_multi_gpu_symmetry = False
        supported_multi_gpu_sync_modes = frozenset({"local_sgd"})

    with pytest.raises(ValueError, match="IncompleteLearner.*sync_initial_parameters"):
        MultiGPUOffPolicyRunner.validate_capabilities(
            algo_type="incomplete",
            learner_cls=IncompleteLearner,
            learner_kwargs={},
            num_gpus=2,
            sync_mode="local_sgd",
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


def test_fast_sac_symmetry_augmentation_emits_fine_grained_nvtx_ranges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from unilab.algos.torch.fast_sac.learner import FastSACLearner

    seen_ranges: list[str] = []

    @contextmanager
    def record_range(name: str, enabled: bool):
        if enabled:
            seen_ranges.append(name)
        yield

    monkeypatch.setattr(learner_module, "_cuda_nvtx_range", record_range)

    learner = FastSACLearner(
        obs_dim=4,
        action_dim=2,
        critic_obs_dim=5,
        device="cpu",
        actor_hidden_dim=8,
        critic_hidden_dim=8,
        num_atoms=3,
        num_q_networks=2,
        use_layer_norm=False,
        use_autotune=False,
        use_symmetry=True,
        symmetry_augmentation=_FakeSymmetryAugmentation(),
    )
    learner.nvtx_profile_ranges = True

    def fake_critic_loss_tensors(*args, **kwargs):
        del args, kwargs
        return (
            torch.tensor(0.0, requires_grad=True),
            torch.tensor(0.0),
            torch.tensor(0.0),
            torch.zeros(4),
        )

    def fake_actor_loss_tensors(*args, **kwargs):
        del args, kwargs
        return (
            torch.tensor(0.0, requires_grad=True),
            torch.tensor(0.0),
            torch.tensor(0.0),
        )

    monkeypatch.setattr(learner, "_critic_loss_tensors", fake_critic_loss_tensors)
    monkeypatch.setattr(learner, "_actor_loss_tensors", fake_actor_loss_tensors)
    batch = {
        "obs": torch.randn(4, 4),
        "critic": torch.randn(4, 5),
        "actions": torch.randn(4, 2),
        "rewards": torch.randn(4),
        "next_obs": torch.randn(4, 4),
        "next_critic": torch.randn(4, 5),
        "dones": torch.zeros(4),
        "truncated": torch.zeros(4),
    }

    learner.update_critic(batch)
    learner.update_actor(batch)

    for expected in {
        "critic/symmetry_obs_actions",
        "critic/symmetry_next_obs",
        "critic/symmetry_critic_obs",
        "critic/symmetry_critic_next_obs",
        "critic/symmetry_aux_repeat",
        "actor/symmetry_obs",
        "actor/symmetry_critic_obs",
    }:
        assert expected in seen_ranges


def test_fast_sac_symmetry_uses_obs_only_augmentation_for_obs_only_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from unilab.algos.torch.fast_sac.learner import FastSACLearner

    symmetry = _FakeSymmetryAugmentation()
    learner = FastSACLearner(
        obs_dim=4,
        action_dim=2,
        critic_obs_dim=5,
        device="cpu",
        actor_hidden_dim=8,
        critic_hidden_dim=8,
        num_atoms=3,
        num_q_networks=2,
        use_layer_norm=False,
        use_autotune=False,
        use_symmetry=True,
        symmetry_augmentation=symmetry,
    )

    def fake_critic_loss_tensors(critic_obs, actions, rewards, next_obs, critic_next_obs, *args):
        assert critic_obs.shape[0] == 8
        assert actions.shape[0] == 8
        assert rewards.shape[0] == 8
        assert next_obs.shape[0] == 8
        assert critic_next_obs.shape[0] == 8
        return (
            torch.tensor(0.0, requires_grad=True),
            torch.tensor(0.0),
            torch.tensor(0.0),
            torch.zeros(8),
        )

    def fake_actor_loss_tensors(obs, critic_obs):
        assert obs.shape[0] == 8
        assert critic_obs.shape[0] == 8
        return (
            torch.tensor(0.0, requires_grad=True),
            torch.tensor(0.0),
            torch.tensor(0.0),
        )

    monkeypatch.setattr(learner, "_critic_loss_tensors", fake_critic_loss_tensors)
    monkeypatch.setattr(learner, "_actor_loss_tensors", fake_actor_loss_tensors)
    batch = {
        "obs": torch.randn(4, 4),
        "critic": torch.randn(4, 5),
        "actions": torch.randn(4, 2),
        "rewards": torch.randn(4),
        "next_obs": torch.randn(4, 4),
        "next_critic": torch.randn(4, 5),
        "dones": torch.zeros(4),
        "truncated": torch.zeros(4),
    }

    learner.update_critic(batch)
    learner.update_actor(batch)

    assert symmetry.augment_obs_and_actions_calls == ["obs"]
    assert symmetry.augment_obs_calls == ["obs", "critic", "critic", "obs", "critic"]
    assert symmetry.mirror_obs_calls == []
