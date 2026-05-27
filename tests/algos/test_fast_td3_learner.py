"""Unit tests for FastTD3 learner and TD3Actor."""

from __future__ import annotations

import pytest
import torch

from unilab.algos.torch.fast_td3.learner import FastTD3Learner, TD3Actor

# ---------------------------------------------------------------------------
# TD3Actor
# ---------------------------------------------------------------------------


class TestTD3Actor:
    @pytest.fixture()
    def actor(self):
        return TD3Actor(
            obs_dim=48,
            n_act=12,
            num_envs=4,
            init_scale=0.01,
            hidden_dim=64,
            log_std_min=-3.0,
            log_std_max=0.0,
            device=torch.device("cpu"),
        )

    def test_forward_shape(self, actor: TD3Actor):
        obs = torch.randn(4, 48)
        actions = actor(obs)
        assert actions.shape == (4, 12)

    def test_forward_tanh_bounded(self, actor: TD3Actor):
        obs = torch.randn(4, 48) * 10
        actions = actor(obs)
        assert actions.min() >= -1.0
        assert actions.max() <= 1.0

    def test_explore_adds_noise(self, actor: TD3Actor):
        obs = torch.randn(4, 48)
        torch.manual_seed(42)
        det = actor.explore(obs, deterministic=True)
        torch.manual_seed(42)
        noisy = actor.explore(obs, deterministic=False)
        assert not torch.allclose(det, noisy, atol=1e-6)

    def test_explore_deterministic_matches_forward(self, actor: TD3Actor):
        obs = torch.randn(4, 48)
        with torch.no_grad():
            fwd = actor(obs)
            det = actor.explore(obs, deterministic=True)
        assert torch.allclose(fwd, det)

    def test_explore_clamped(self, actor: TD3Actor):
        obs = torch.randn(4, 48) * 100
        actions = actor.explore(obs, deterministic=False)
        assert actions.min() >= -1.0
        assert actions.max() <= 1.0

    def test_noise_resampled_on_done(self, actor: TD3Actor):
        scales_before = actor.noise_scales.clone()
        dones = torch.tensor([1.0, 0.0, 1.0, 0.0])
        obs = torch.randn(4, 48)
        actor.explore(obs, dones=dones)
        # Envs 0 and 2 should have new noise scales, 1 and 3 unchanged
        assert not torch.equal(scales_before[0], actor.noise_scales[0])
        assert torch.equal(scales_before[1], actor.noise_scales[1])
        assert not torch.equal(scales_before[2], actor.noise_scales[2])
        assert torch.equal(scales_before[3], actor.noise_scales[3])

    def test_noise_scales_in_range(self, actor: TD3Actor):
        import math

        std_min = math.exp(-3.0)
        std_max = math.exp(0.0)
        assert (actor.noise_scales >= std_min).all()
        assert (actor.noise_scales <= std_max).all()


# ---------------------------------------------------------------------------
# FastTD3Learner
# ---------------------------------------------------------------------------


class TestFastTD3Learner:
    @pytest.fixture()
    def learner(self):
        return FastTD3Learner(
            obs_dim=48,
            action_dim=12,
            critic_obs_dim=48,
            num_envs=4,
            device="cpu",
            gamma=0.97,
            tau=0.01,
            actor_lr=3e-4,
            critic_lr=3e-4,
            actor_hidden_dim=64,
            critic_hidden_dim=128,
            num_atoms=51,
            v_min=-10.0,
            v_max=10.0,
            init_scale=0.01,
            log_std_min=-3.0,
            log_std_max=0.0,
            weight_decay=0.001,
            use_cdq=True,
            policy_noise=0.1,
            noise_clip=0.2,
            policy_frequency=2,
            obs_normalization=True,
        )

    def _make_batch(self, batch_size=16, obs_dim=48, action_dim=12):
        return {
            "obs": torch.randn(batch_size, obs_dim),
            "actions": torch.randn(batch_size, action_dim).clamp(-1, 1),
            "rewards": torch.randn(batch_size),
            "next_obs": torch.randn(batch_size, obs_dim),
            "dones": torch.zeros(batch_size),
            "truncated": torch.zeros(batch_size),
            "critic": torch.randn(batch_size, obs_dim),
            "next_critic": torch.randn(batch_size, obs_dim),
        }

    def test_update_critic_returns_metrics(self, learner: FastTD3Learner):
        batch = self._make_batch()
        metrics = learner.update_critic(batch)
        assert "qf_loss" in metrics
        assert "qf_max" in metrics
        assert "qf_min" in metrics

    def test_update_actor_returns_metrics(self, learner: FastTD3Learner):
        batch = self._make_batch()
        metrics = learner.update_actor(batch)
        assert "actor_loss" in metrics

    def test_soft_update_target_only_updates_critic(self, learner: FastTD3Learner):
        """soft_update_target updates qnet_target only, not actor_target."""
        actor_target_before = [p.clone() for p in learner.actor_target.parameters()]
        qnet_target_before = [p.clone() for p in learner.qnet_target.parameters()]

        with torch.no_grad():
            for p in learner.qnet.parameters():
                p.add_(torch.randn_like(p))

        learner.soft_update_target()

        # qnet_target should have moved
        for before, after in zip(qnet_target_before, learner.qnet_target.parameters()):
            assert not torch.allclose(before, after.data)

        # actor_target should be unchanged
        for before, after in zip(actor_target_before, learner.actor_target.parameters()):
            assert torch.allclose(before, after.data)

    def test_obs_normalization(self, learner: FastTD3Learner):
        obs = torch.randn(4, 48) * 10 + 5
        normed = learner.normalize_obs(obs, update=True)
        assert normed.shape == obs.shape

    def test_get_and_load_state_dict(self, learner: FastTD3Learner):
        state = learner.get_state_dict()
        assert "actor" in state
        assert "actor_target" in state
        assert "qnet" in state
        assert "qnet_target" in state
        learner.load_state_dict(state)

    def test_training_loop_updates(self, learner: FastTD3Learner):
        batch = self._make_batch()
        for step in range(4):
            learner.update_critic(batch)
            if step % learner.policy_frequency == 0:
                learner.update_actor(batch)
            learner.soft_update_target()

    def test_cdq_disabled(self):
        learner = FastTD3Learner(
            obs_dim=16,
            action_dim=4,
            critic_obs_dim=16,
            num_envs=2,
            device="cpu",
            actor_hidden_dim=32,
            critic_hidden_dim=64,
            num_atoms=11,
            use_cdq=False,
        )
        batch = {
            "obs": torch.randn(8, 16),
            "actions": torch.randn(8, 4).clamp(-1, 1),
            "rewards": torch.randn(8),
            "next_obs": torch.randn(8, 16),
            "dones": torch.zeros(8),
            "truncated": torch.zeros(8),
            "critic": torch.randn(8, 16),
            "next_critic": torch.randn(8, 16),
        }
        assert "qf_loss" in learner.update_critic(batch)
        assert "actor_loss" in learner.update_actor(batch)


# ---------------------------------------------------------------------------
# CLI routing
# ---------------------------------------------------------------------------


class TestTD3CLIRouting:
    def test_build_route_td3_go2_joystick_flat_motrix(self):
        from unilab.cli import build_route

        route = build_route("td3", "go2_joystick_flat", "motrix")
        assert route.script_name == "train_offpolicy.py"
        assert route.config_group == "offpolicy"
        assert route.owner_task == "td3/go2_joystick_flat/motrix.yaml"
        assert "algo=td3" in route.generated_overrides
        assert "task=td3/go2_joystick_flat/motrix" in route.generated_overrides

    def test_build_route_td3_go1_joystick_flat_motrix(self):
        from unilab.cli import build_route

        route = build_route("td3", "go1_joystick_flat", "motrix")
        assert route.script_name == "train_offpolicy.py"
        assert route.owner_task == "td3/go1_joystick_flat/motrix.yaml"

    def test_td3_in_offpolicy_algos(self):
        from unilab.cli import OFFPOLICY_ALGOS, SUPPORTED_ALGOS

        assert "td3" in SUPPORTED_ALGOS
        assert "td3" in OFFPOLICY_ALGOS
