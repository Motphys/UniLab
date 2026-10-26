"""Tests for the FlashSAC learner-side custom actor contract (issue #30).

Covers construction via ``actor_config``, auxiliary-loss gradients and
diagnostics, checkpoint actor-semantics persistence and fail-closed
validation, custom action transform / log-probability consistency, spawn-safe
downstream module loading, CUDA graph rejection, and the default-actor
regression path.
"""

from __future__ import annotations

import json
import math
import multiprocessing
from typing import Any

import pytest
import torch
import torch.nn.functional as F

from tests.rl.algos import flash_sac_actor_fixtures as fixtures
from tests.rl.algos.flash_sac_actor_fixtures import ScaledTanhToyActor
from unilab.rl.algos.flash_sac.actor_spec import (
    FlashSACActorSpec,
    validate_actor_semantics,
)
from unilab.rl.algos.flash_sac.learner import FlashSACLearner
from unilab.rl.logging.metric_schema import normalize_metric_map

_BUILDER = "tests.rl.algos.flash_sac_actor_fixtures:build_scaled_tanh_actor"

_OBS_DIM = 4
_ACTION_DIM = 2
_CRITIC_OBS_DIM = 6


def _actor_config(**kwargs: Any) -> dict[str, Any]:
    return {"builder": _BUILDER, "kwargs": kwargs}


def _make_learner(**kwargs: Any) -> FlashSACLearner:
    defaults: dict[str, Any] = {
        "obs_dim": _OBS_DIM,
        "action_dim": _ACTION_DIM,
        "critic_obs_dim": _CRITIC_OBS_DIM,
        "actor_hidden_dim": 8,
        "critic_hidden_dim": 8,
        "actor_num_blocks": 1,
        "critic_num_blocks": 1,
        "num_atoms": 5,
        "device": "cpu",
        "use_compile": False,
        "normalize_reward": False,
    }
    defaults.update(kwargs)
    return FlashSACLearner(**defaults)


def _make_batch(batch_size: int = 8) -> dict[str, torch.Tensor]:
    return {
        "obs": torch.randn(batch_size, _OBS_DIM),
        "critic": torch.randn(batch_size, _CRITIC_OBS_DIM),
        "actions": torch.tanh(torch.randn(batch_size, _ACTION_DIM)),
        "rewards": torch.randn(batch_size),
        "next_obs": torch.randn(batch_size, _OBS_DIM),
        "next_critic": torch.randn(batch_size, _CRITIC_OBS_DIM),
        "dones": torch.zeros(batch_size),
        "truncated": torch.zeros(batch_size),
    }


class TestCustomActorConstruction:
    def test_builds_custom_actor_and_records_semantics(self) -> None:
        learner = _make_learner(
            actor_config=_actor_config(hidden_dim=8, action_scale=2.0, aux_weight=0.5)
        )

        assert isinstance(learner.actor, ScaledTanhToyActor)
        semantics = learner.get_state_dict()["actor_semantics"]
        assert semantics == {
            "schema_version": 1,
            "kind": "custom",
            "builder": _BUILDER,
            "kwargs": {"action_scale": 2.0, "aux_weight": 0.5, "hidden_dim": 8},
            "optimizer_param_groups": {},
            "obs_dim": _OBS_DIM,
            "action_dim": _ACTION_DIM,
            "action_transform": "scaled_tanh_test_v1",
        }

    def test_actor_factory_builds_custom_actor(self) -> None:
        from unilab.rl.algos.common.actor_factory import build_actor

        actor = build_actor(
            "flashsac",
            obs_dim=_OBS_DIM,
            action_dim=_ACTION_DIM,
            actor_hidden_dim=8,
            use_layer_norm=False,
            device="cpu",
            actor_config=_actor_config(hidden_dim=8),
        )

        assert isinstance(actor, ScaledTanhToyActor)

    def test_rejects_missing_builder(self) -> None:
        with pytest.raises(ValueError, match="builder"):
            FlashSACActorSpec.from_config({"kwargs": {}})

    def test_rejects_non_mapping_config(self) -> None:
        with pytest.raises(TypeError, match="mapping"):
            FlashSACActorSpec.from_config("not-a-mapping")

    def test_rejects_non_json_kwargs(self) -> None:
        with pytest.raises(ValueError, match="JSON-serializable"):
            FlashSACActorSpec.from_config({"builder": _BUILDER, "kwargs": {"bad": object()}})

    def test_rejects_unimportable_builder(self) -> None:
        with pytest.raises(ImportError, match="no_such_package_xyz"):
            _make_learner(actor_config={"builder": "no_such_package_xyz.actors:build"})

    def test_rejects_non_module_actor(self) -> None:
        with pytest.raises(TypeError, match="nn.Module"):
            _make_learner(
                actor_config={
                    "builder": "tests.rl.algos.flash_sac_actor_fixtures:build_non_module_actor"
                }
            )

    def test_rejects_actor_missing_contract_methods(self) -> None:
        with pytest.raises(TypeError, match="missing required contract methods"):
            _make_learner(
                actor_config={
                    "builder": "tests.rl.algos.flash_sac_actor_fixtures:build_broken_actor"
                }
            )

    def test_rejects_non_scalar_auxiliary_loss(self) -> None:
        with pytest.raises(TypeError, match="0-dim tensor"):
            _make_learner(
                actor_config={
                    "builder": "tests.rl.algos.flash_sac_actor_fixtures:build_bad_aux_actor"
                }
            )

    def test_rejects_whole_cycle_cuda_graph(self) -> None:
        with pytest.raises(ValueError, match="whole-cycle CUDA graph"):
            _make_learner(
                actor_config=_actor_config(),
                use_whole_cycle_cuda_graph=True,
            )

    def test_rejects_actor_normalize_parameters_opt_out(self) -> None:
        with pytest.raises(ValueError, match="actor_normalize_parameters"):
            _make_learner(
                actor_config=_actor_config(),
                actor_normalize_parameters=False,
            )


class TestOptimizerParamGroups:
    @staticmethod
    def _config() -> dict[str, Any]:
        return {
            "builder": _BUILDER,
            "kwargs": {"hidden_dim": 8},
            "optimizer_param_groups": {
                "residual_mean": {
                    "parameters": ["mean_head.weight", "mean_head.bias"],
                    "lr_multiplier": 0.1,
                }
            },
        }

    def test_multiplier_scales_adam_update_not_gradient(self) -> None:
        learner = _make_learner(actor_config=self._config())
        optimizer = learner.actor_optimizer
        default = learner.actor.trunk[0].weight
        scaled = learner.actor.mean_head.weight
        for parameter in learner.actor.parameters():
            parameter.grad = torch.ones_like(parameter)
        default_before = default.detach().clone()
        scaled_before = scaled.detach().clone()

        optimizer.step()

        default_delta = (default - default_before).abs().mean()
        scaled_delta = (scaled - scaled_before).abs().mean()
        torch.testing.assert_close(scaled_delta, default_delta * 0.1, rtol=1e-5, atol=1e-8)
        torch.testing.assert_close(scaled.grad, torch.ones_like(scaled))
        group = next(group for group in optimizer.param_groups if group["name"] == "residual_mean")
        assert group["lr"] == pytest.approx(optimizer.param_groups[0]["lr"] * 0.1)

        learner.actor_scheduler.step()
        assert optimizer.param_groups[0]["lr"] > 0
        assert group["lr"] == pytest.approx(optimizer.param_groups[0]["lr"] * 0.1)

    def test_optimizer_group_checkpoint_round_trip(self) -> None:
        learner = _make_learner(actor_config=self._config())
        learner.update_actor(_make_batch())
        state = learner.get_state_dict()

        restored = _make_learner(actor_config=self._config())
        restored.load_state_dict(state)

        assert [group["name"] for group in restored.actor_optimizer.param_groups] == [
            "default",
            "residual_mean",
        ]
        assert (
            restored.get_state_dict()["actor_semantics"]["optimizer_param_groups"]["residual_mean"][
                "lr_multiplier"
            ]
            == 0.1
        )

    def test_unknown_parameter_fails_closed(self) -> None:
        config = self._config()
        config["optimizer_param_groups"]["residual_mean"]["parameters"] = ["missing.weight"]
        with pytest.raises(ValueError, match="unknown actor parameters"):
            _make_learner(actor_config=config)


class TestAuxiliaryLosses:
    def test_aux_loss_gradients_reach_actor_parameters(self) -> None:
        torch.manual_seed(7)
        learner = _make_learner(actor_config=_actor_config(hidden_dim=8, aux_weight=0.5))

        metrics = learner.update_actor(_make_batch())

        assert "Loss/aux_token_recon" in metrics
        assert math.isfinite(metrics["Loss/aux_token_recon"])
        assert metrics["Loss/aux_token_recon"] > 0
        normalize_metric_map(metrics)
        aux_grad = learner.actor.aux_head.weight.grad
        assert aux_grad is not None
        assert aux_grad.abs().sum() > 0

    def test_aux_loss_is_added_to_actor_objective(self) -> None:
        torch.manual_seed(11)
        with_aux = _make_learner(actor_config=_actor_config(hidden_dim=8, aux_weight=0.5))
        torch.manual_seed(11)
        without_aux = _make_learner(actor_config=_actor_config(hidden_dim=8, aux_weight=0.5))
        without_aux.actor.emit_aux = False
        batch = _make_batch()

        # Reseed so both updates draw identical reparameterization noise.
        torch.manual_seed(123)
        with_metrics = with_aux.update_actor(batch)
        torch.manual_seed(123)
        without_metrics = without_aux.update_actor(batch)

        assert "Loss/aux_token_recon" in with_metrics
        assert "Loss/aux_token_recon" not in without_metrics
        expected = without_metrics["Loss/actor"] + with_metrics["Loss/aux_token_recon"]
        assert with_metrics["Loss/actor"] == pytest.approx(expected, rel=1e-5, abs=1e-6)

    def test_aux_loss_names_must_be_stable(self) -> None:
        learner = _make_learner(actor_config=_actor_config(hidden_dim=8))
        learner.update_actor(_make_batch())

        learner.actor.emit_aux = False
        with pytest.raises(RuntimeError, match="auxiliary loss names changed"):
            learner.update_actor(_make_batch())


class TestActionTransform:
    def test_custom_action_range_and_deterministic_consistency(self) -> None:
        learner = _make_learner(actor_config=_actor_config(hidden_dim=8, action_scale=2.0))
        actor = learner.actor
        obs = torch.randn(16, _OBS_DIM)

        sampled = actor.explore(obs, deterministic=False)
        deterministic = actor.explore(obs, deterministic=True)
        trained_actions, info = actor(obs, training=True)
        exported = actor.as_export_module()(obs)

        assert sampled.abs().max() <= 2.0
        assert trained_actions.abs().max() <= 2.0
        mean, _std = actor.get_mean_and_std(obs, training=False)
        torch.testing.assert_close(deterministic, 2.0 * torch.tanh(mean))
        torch.testing.assert_close(exported, deterministic)
        assert info["log_prob"].shape == (16,)

    def test_log_prob_matches_change_of_variables(self) -> None:
        learner = _make_learner(actor_config=_actor_config(hidden_dim=8, action_scale=2.0))
        actor = learner.actor
        obs = torch.randn(4, _OBS_DIM)
        mean, std = actor.get_mean_and_std(obs, training=False)
        raw = torch.distributions.Normal(mean, std).sample()

        dist = torch.distributions.Normal(mean, std)
        log_det = math.log(2.0) + 2.0 * (math.log(2.0) - raw - F.softplus(-2.0 * raw))
        expected = (dist.log_prob(raw) - log_det).sum(dim=-1)

        torch.testing.assert_close(actor.log_prob_from_raw(mean, std, raw), expected)


class TestCheckpointSemantics:
    def test_custom_actor_checkpoint_round_trip(self) -> None:
        torch.manual_seed(3)
        learner = _make_learner(actor_config=_actor_config(hidden_dim=8, action_scale=2.0))
        learner.update_actor(_make_batch())
        state = learner.get_state_dict()

        restored = _make_learner(actor_config=_actor_config(hidden_dim=8, action_scale=2.0))
        restored.load_state_dict(state)

        torch.testing.assert_close(restored.actor.state_dict(), learner.actor.state_dict())

    def test_mismatched_builder_kwargs_fail_closed(self) -> None:
        learner = _make_learner(actor_config=_actor_config(hidden_dim=8, action_scale=2.0))
        state = learner.get_state_dict()

        other = _make_learner(actor_config=_actor_config(hidden_dim=8, action_scale=3.0))
        with pytest.raises(ValueError, match="actor semantics"):
            other.load_state_dict(state)

    def test_custom_checkpoint_rejected_by_builtin_config(self) -> None:
        learner = _make_learner(actor_config=_actor_config(hidden_dim=8))
        state = learner.get_state_dict()

        builtin = _make_learner()
        with pytest.raises(ValueError, match="actor semantics"):
            builtin.load_state_dict(state)

    def test_custom_config_rejects_checkpoint_without_semantics(self) -> None:
        builtin = _make_learner()
        state = builtin.get_state_dict()
        state.pop("actor_semantics")

        custom = _make_learner(actor_config=_actor_config(hidden_dim=8))
        with pytest.raises(ValueError, match="does not record actor semantics"):
            custom.load_state_dict(state)

    def test_unknown_semantics_schema_version_fails_closed(self) -> None:
        with pytest.raises(ValueError, match="schema version"):
            validate_actor_semantics(
                {"schema_version": 999},
                {"schema_version": 1, "kind": "builtin"},
            )

    def test_non_mapping_semantics_fails_closed(self) -> None:
        with pytest.raises(TypeError, match="must be a mapping"):
            validate_actor_semantics(
                "legacy-string",
                {"schema_version": 1, "kind": "builtin"},
            )

    def test_null_semantics_fails_closed_for_builtin_actor(self) -> None:
        learner = _make_learner()
        state = learner.get_state_dict()
        state["actor_semantics"] = None

        with pytest.raises(ValueError, match="does not record actor semantics"):
            _make_learner().load_state_dict(state)

    # #2416: pre-contract checkpoints (no actor_semantics entry) are rejected
    # for every actor kind, including the built-in actor; retrain to migrate.
    def test_pre_contract_builtin_checkpoint_fails_closed(self) -> None:
        learner = _make_learner()
        canonical = learner.get_state_dict()
        pre_contract = learner.get_state_dict()
        pre_contract.pop("actor_semantics")

        assert "actor_semantics" in canonical
        assert "actor_semantics" not in pre_contract
        with pytest.raises(ValueError, match="pre-contract FlashSAC checkpoints"):
            _make_learner().load_state_dict(pre_contract)


class TestDefaultActorRegression:
    def test_default_actor_unchanged(self) -> None:
        learner = _make_learner()

        metrics = learner.update_actor(_make_batch())

        assert set(metrics) == {
            "Loss/actor",
            "Loss/entropy",
            "Policy/temperature",
            "Loss/temperature",
        }
        normalize_metric_map(metrics)
        semantics = learner.get_state_dict()["actor_semantics"]
        assert semantics["kind"] == "builtin"
        assert semantics["action_transform"] == "tanh"
        obs = torch.randn(8, _OBS_DIM)
        assert learner.actor.explore(obs, deterministic=False).abs().max() <= 1.0

    def test_default_actor_round_trip(self) -> None:
        learner = _make_learner()
        learner.update_actor(_make_batch())
        state = learner.get_state_dict()

        restored = _make_learner()
        restored.load_state_dict(state)

        torch.testing.assert_close(restored.actor.state_dict(), learner.actor.state_dict())


class TestSpawnSafety:
    def test_spawn_subprocess_builds_custom_actor(self) -> None:
        ctx = multiprocessing.get_context("spawn")
        result_queue = ctx.Queue()
        process = ctx.Process(
            target=fixtures.spawn_build_child,
            args=(
                _BUILDER,
                json.dumps({"hidden_dim": 8, "action_scale": 2.0}),
                _OBS_DIM,
                _ACTION_DIM,
                result_queue,
            ),
        )
        process.start()
        process.join(timeout=120)

        assert process.exitcode == 0
        result = result_queue.get(timeout=10)
        assert result["transform_id"] == "scaled_tanh_test_v1"
        assert result["max_abs"] <= 2.0
        assert result["export_matches"]


class TestPlayValidation:
    def test_play_load_validates_actor_semantics(self) -> None:
        from unilab.rl.algos.common.actor_factory import build_actor
        from unilab.rl.algos.flash_sac.actor_spec import build_custom_actor_semantics
        from unilab.visualization.interactive_playback import load_play_actor

        actor = build_actor(
            "flashsac",
            obs_dim=_OBS_DIM,
            action_dim=_ACTION_DIM,
            actor_hidden_dim=8,
            use_layer_norm=False,
            device="cpu",
            actor_config=_actor_config(hidden_dim=8, action_scale=2.0),
        )
        spec = FlashSACActorSpec.from_config(_actor_config(hidden_dim=8, action_scale=2.0))
        semantics = build_custom_actor_semantics(
            spec,
            obs_dim=_OBS_DIM,
            action_dim=_ACTION_DIM,
            action_transform_id=actor.action_transform_id,
        )
        checkpoint = {"actor": actor.state_dict(), "actor_semantics": semantics}

        fresh = build_actor(
            "flashsac",
            obs_dim=_OBS_DIM,
            action_dim=_ACTION_DIM,
            actor_hidden_dim=8,
            use_layer_norm=False,
            device="cpu",
            actor_config=_actor_config(hidden_dim=8, action_scale=2.0),
        )
        load_play_actor("flashsac", fresh, None, checkpoint, expected_actor_semantics=semantics)
        torch.testing.assert_close(fresh.state_dict(), actor.state_dict())

        mismatched = dict(semantics)
        mismatched["kwargs"] = {"hidden_dim": 8, "action_scale": 9.0}
        with pytest.raises(ValueError, match="actor semantics"):
            load_play_actor(
                "flashsac", fresh, None, checkpoint, expected_actor_semantics=mismatched
            )

    def test_play_load_rejects_custom_checkpoint_without_semantics(self) -> None:
        from unilab.rl.algos.common.actor_factory import build_actor
        from unilab.rl.algos.flash_sac.actor_spec import (
            FlashSACActorSpec,
            build_custom_actor_semantics,
        )
        from unilab.visualization.interactive_playback import load_play_actor

        actor = build_actor(
            "flashsac",
            obs_dim=_OBS_DIM,
            action_dim=_ACTION_DIM,
            actor_hidden_dim=8,
            use_layer_norm=False,
            device="cpu",
            actor_config=_actor_config(hidden_dim=8),
        )
        semantics = build_custom_actor_semantics(
            FlashSACActorSpec.from_config(_actor_config(hidden_dim=8)),
            obs_dim=_OBS_DIM,
            action_dim=_ACTION_DIM,
            action_transform_id=actor.action_transform_id,
        )
        legacy_checkpoint = {"actor": actor.state_dict()}

        with pytest.raises(ValueError, match="does not record actor semantics"):
            load_play_actor(
                "flashsac",
                actor,
                None,
                legacy_checkpoint,
                expected_actor_semantics=semantics,
            )


def test_validate_custom_actor_requires_parameters() -> None:
    from unilab.rl.algos.flash_sac.actor_spec import validate_custom_actor

    class _Paramless(torch.nn.Module):
        action_transform_id = "noop"

        def forward(self, observations, training):
            actions = torch.zeros(observations.shape[0], _ACTION_DIM)
            return actions, {"log_prob": torch.zeros(observations.shape[0])}

        def get_mean_and_std(self, observations, training):
            zeros = torch.zeros(observations.shape[0], _ACTION_DIM)
            return zeros, torch.ones_like(zeros)

        def explore(self, obs, dones=None, deterministic=False):
            return torch.zeros(obs.shape[0], _ACTION_DIM)

        def normalize_parameters(self) -> None:
            return None

        def as_export_module(self):
            return torch.nn.Identity()

        def inference_startup_memory_categories(self, batch_size):
            return {"persistent_exploration_scratch": 0}

    with pytest.raises(ValueError, match="no parameters"):
        validate_custom_actor(_Paramless(), obs_dim=_OBS_DIM, action_dim=_ACTION_DIM)
