from __future__ import annotations

import ast
import copy
import inspect
import logging
import textwrap
from typing import Any, cast

import pytest
import torch
from tensordict import TensorDict


def test_hora_ppo_logs_when_symmetry_is_logging_only(
    caplog: pytest.LogCaptureFixture,
) -> None:
    from unilab.algos.hora.ppo import HoraPPO

    actor = torch.nn.Linear(2, 2)
    critic = torch.nn.Linear(2, 1)
    actor.is_recurrent = False  # type: ignore[attr-defined]
    critic.is_recurrent = False  # type: ignore[attr-defined]
    symmetry_cfg = {
        "use_data_augmentation": False,
        "use_mirror_loss": False,
        "data_augmentation_func": lambda *_args: None,
    }

    with caplog.at_level(logging.WARNING, logger="unilab.algos.hora.ppo"):
        HoraPPO(
            cast(Any, actor),
            cast(Any, critic),
            cast(Any, object()),
            symmetry_cfg=symmetry_cfg,
        )

    assert "Symmetry not used for learning. We will use it for logging instead." in caplog.text


def test_hora_sac_actor_shapes_and_stable_module_names() -> None:
    from unilab.algos.hora.sac_models import HoraSACActor

    actor = HoraSACActor(
        obs_dim=5,
        priv_info_dim=3,
        action_dim=2,
        hidden_dim=16,
        priv_info_embed_dim=4,
        priv_mlp_hidden_dims=(8, 4),
        use_layer_norm=False,
    )

    obs = torch.zeros(6, 5)
    priv_info = torch.zeros(6, 3)
    actions, log_probs, log_std = actor.get_actions_and_log_probs(obs, priv_info)

    assert hasattr(actor, "priv_encoder")
    assert hasattr(actor, "actor_trunk")
    assert hasattr(actor, "action_mean_head")
    assert hasattr(actor, "action_logstd_head")
    assert actions.shape == (6, 2)
    assert log_probs.shape == (6,)
    assert log_std.shape == (6, 2)


def test_hora_sac_learner_derives_priv_info_from_critic_contract() -> None:
    from unilab.algos.hora.sac_learner import derive_priv_info_from_critic_obs

    actor_obs = torch.zeros((4, 5), dtype=torch.float32)
    priv_info = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    critic_obs = torch.cat([actor_obs, priv_info], dim=-1)

    torch.testing.assert_close(
        derive_priv_info_from_critic_obs(actor_obs, critic_obs, context="test"),
        priv_info,
    )

    with pytest.raises(ValueError, match="privileged tail"):
        derive_priv_info_from_critic_obs(actor_obs, actor_obs, context="test")


def test_hora_sac_learner_updates_with_privileged_tail() -> None:
    from unilab.algos.hora.sac_learner import HoraSACLearner

    torch.manual_seed(23)
    learner = HoraSACLearner(
        obs_dim=5,
        critic_obs_dim=8,
        priv_info_dim=3,
        action_dim=2,
        device="cpu",
        actor_hidden_dim=16,
        critic_hidden_dim=16,
        priv_info_embed_dim=4,
        priv_mlp_hidden_dims=(8, 4),
        num_atoms=11,
        use_layer_norm=False,
        actor_lr=1e-3,
        critic_lr=1e-3,
        alpha_lr=1e-3,
    )
    obs = torch.randn(7, 5)
    next_obs = torch.randn(7, 5)
    priv_info = torch.randn(7, 3)
    next_priv_info = torch.randn(7, 3)
    batch = {
        "obs": obs,
        "critic": torch.cat([obs, priv_info], dim=-1),
        "actions": torch.randn(7, 2).clamp(-0.5, 0.5),
        "rewards": torch.randn(7),
        "next_obs": next_obs,
        "next_critic": torch.cat([next_obs, next_priv_info], dim=-1),
        "dones": torch.zeros(7),
        "truncated": torch.zeros(7),
    }

    critic_metrics = learner.update_critic(batch)
    actor_metrics = learner.update_actor(batch)

    assert torch.isfinite(torch.tensor(list(critic_metrics.values()))).all()
    assert torch.isfinite(torch.tensor(list(actor_metrics.values()))).all()


def test_hora_sac_disables_cuda_graph_critic_path() -> None:
    from unilab.algos.hora.sac_learner import HoraSACLearner

    learner = HoraSACLearner(
        obs_dim=5,
        critic_obs_dim=8,
        priv_info_dim=3,
        action_dim=2,
        device="cpu",
        actor_hidden_dim=16,
        critic_hidden_dim=16,
        num_atoms=11,
        use_layer_norm=False,
        use_cuda_graph_critic=True,
        use_cuda_graph_actor=True,
    )

    assert not learner.use_cuda_graph_critic
    assert not learner.use_cuda_graph_actor


def test_hora_sac_distilled_student_forward_does_not_require_priv_info() -> None:
    from unilab.algos.hora.distill import HoraSACDistillActor, HoraSACDistillShared

    shared = HoraSACDistillShared(
        obs_dim=12,
        action_dim=4,
        priv_info_dim=3,
        hidden_dim=32,
        priv_info_embed_dim=3,
        priv_mlp_hidden_dims=(8, 3),
        use_layer_norm=False,
        proprio_hist_len=30,
        proprio_frame_dim=2,
        device="cpu",
    )
    actor = HoraSACDistillActor(shared)
    student_obs = TensorDict(
        {
            "actor": torch.zeros((5, 12), dtype=torch.float32),
            "proprio_hist": torch.zeros((5, 30, 2), dtype=torch.float32),
        },
        batch_size=[5],
    )

    actions = actor(student_obs)

    assert actions.shape == (5, 4)
    with pytest.raises(ValueError, match="priv_info is required"):
        shared.policy_mean(student_obs, prefer_student=True)


def test_hora_sac_distill_loads_teacher_actor_weights(tmp_path) -> None:
    from unilab.algos.hora.distill import (
        HoraSACDistillActor,
        HoraSACDistillShared,
        load_teacher_actor_weights,
    )
    from unilab.algos.hora.sac_models import HoraSACActor

    teacher = HoraSACActor(
        obs_dim=12,
        priv_info_dim=3,
        action_dim=4,
        hidden_dim=32,
        priv_info_embed_dim=3,
        priv_mlp_hidden_dims=(8, 3),
        use_layer_norm=False,
    )
    shared = HoraSACDistillShared(
        obs_dim=12,
        action_dim=4,
        priv_info_dim=3,
        hidden_dim=32,
        priv_info_embed_dim=3,
        priv_mlp_hidden_dims=(8, 3),
        use_layer_norm=False,
        proprio_hist_len=30,
        proprio_frame_dim=2,
        device="cpu",
    )
    actor = HoraSACDistillActor(shared)
    checkpoint = tmp_path / "model.pt"
    torch.save({"actor": teacher.state_dict()}, checkpoint)

    load_teacher_actor_weights(
        actor,
        checkpoint,
        teacher_algo_family="sac",
        device=torch.device("cpu"),
    )

    torch.testing.assert_close(
        actor.shared.action_mean_head.weight,
        teacher.action_mean_head.weight,
    )
    torch.testing.assert_close(
        actor.shared.encode_privileged_info(torch.zeros(2, 3)),
        teacher.encode_privileged_info(torch.zeros(2, 3)),
    )


def test_hora_rsl_wrapper_uses_explicit_np_env_state_contract() -> None:
    """HORA wrapper must not probe required NpEnvState fields dynamically."""
    from unilab.algos.hora.rsl_rl import HoraRslRlVecEnvWrapper

    source = textwrap.dedent(inspect.getsource(HoraRslRlVecEnvWrapper.step))
    tree = ast.parse(source)
    forbidden_calls: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        if node.func.id not in {"getattr", "hasattr"}:
            continue
        if not node.args or not isinstance(node.args[0], ast.Name):
            continue
        if node.args[0].id == "state":
            forbidden_calls.append(node.func.id)

    assert forbidden_calls == []


def test_hora_appo_learner_derives_priv_info_from_critic_contract() -> None:
    from unilab.algos.hora.appo_learner import _derive_priv_info_from_critic

    actor_obs = torch.zeros((2, 3, 4), dtype=torch.float32)
    priv_info = torch.arange(12, dtype=torch.float32).reshape(2, 3, 2)
    critic_obs = torch.cat([actor_obs, priv_info], dim=-1)

    torch.testing.assert_close(
        _derive_priv_info_from_critic(actor_obs, critic_obs, context="test"),
        priv_info,
    )

    with pytest.raises(ValueError, match="privileged tail"):
        _derive_priv_info_from_critic(actor_obs, actor_obs, context="test")


def _make_hora_appo_learner(**algorithm_overrides):
    from unilab.algos.hora.appo_learner import HoraAPPOLearner
    from unilab.algos.hora.models import (
        HoraActorModel,
        HoraCriticModel,
        HoraSharedActorCritic,
    )

    obs = TensorDict(
        {
            "actor": torch.zeros(4, 5),
            "priv_info": torch.zeros(4, 2),
        },
        batch_size=4,
    )
    shared = HoraSharedActorCritic(
        obs_dim=5,
        action_dim=3,
        priv_info_dim=2,
        priv_info_embed_dim=2,
        actor_hidden_dims=(8,),
        priv_mlp_hidden_dims=(4, 2),
    )
    actor = HoraActorModel(obs, {}, "actor", 3, shared_model=shared)
    critic = HoraCriticModel(obs, {}, "critic", 1, shared_model=shared)
    kwargs = {
        "actor": actor,
        "critic": critic,
        "num_learning_epochs": 1,
        "num_mini_batches": 1,
        "device": "cpu",
    }
    kwargs.update(algorithm_overrides)
    return HoraAPPOLearner(**kwargs)


def test_hora_appo_learner_uses_one_shared_actor_critic_core() -> None:
    learner = _make_hora_appo_learner()

    assert learner.actor.shared is learner.critic.shared


def test_hora_appo_minibatch_tensor_path_matches_tensordict_forward() -> None:
    torch.manual_seed(29)
    learner = _make_hora_appo_learner()
    actor_obs = torch.randn(6, 5)
    priv_info = torch.randn(6, 2)
    critic_obs = torch.cat([actor_obs, priv_info], dim=-1)
    obs_td = TensorDict(
        {"actor": actor_obs, "priv_info": priv_info},
        batch_size=actor_obs.shape[0],
    )

    with torch.inference_mode():
        learner.actor(obs_td, stochastic_output=True)
        expected_mean = learner.actor.output_mean.clone()
        expected_std = learner.actor.output_std.clone()
        expected_value = learner.critic(obs_td).squeeze(-1)
        mean, std, value = learner._minibatch_policy_value(actor_obs, critic_obs)

    torch.testing.assert_close(mean, expected_mean)
    torch.testing.assert_close(std, expected_std)
    torch.testing.assert_close(value, expected_value)


def test_hora_appo_runner_builds_shared_actor_critic_core() -> None:
    from unilab.algos.hora.appo_runner import HoraAPPORunner

    runner = HoraAPPORunner.__new__(HoraAPPORunner)
    runner.num_envs = 4
    runner.obs_dim = 5
    runner.action_dim = 3
    runner.priv_info_dim = 2
    runner.device = "cpu"
    runner.seed = None
    runner.rl_cfg = {
        "obs_groups": {
            "actor": {"actor": 5, "priv_info": 2},
            "critic": {"actor": 5, "priv_info": 2},
        },
        "actor": {
            "class_name": "unilab.algos.hora:HoraActorModel",
            "hidden_dims": [8],
            "priv_info_embed_dim": 2,
            "priv_mlp_hidden_dims": [4, 2],
        },
        "critic": {
            "class_name": "unilab.algos.hora:HoraCriticModel",
            "priv_info_embed_dim": 2,
            "priv_mlp_hidden_dims": [4, 2],
        },
        "algorithm": {
            "num_learning_epochs": 1,
            "num_mini_batches": 1,
        },
    }

    learner = runner._build_learner()

    assert learner.actor.shared is learner.critic.shared


def test_hora_appo_worker_builds_shared_actor_critic_core() -> None:
    from unilab.algos.hora.appo_worker import hora_appo_collector_fn

    source = textwrap.dedent(inspect.getsource(hora_appo_collector_fn))
    tree = ast.parse(source)
    shared_model_keywords = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.keyword)
        and node.arg == "shared_model"
        and isinstance(node.value, ast.Name)
        and node.value.id == "shared_model"
    ]

    assert "build_hora_shared_actor_critic" in source
    assert len(shared_model_keywords) >= 2


def test_hora_appo_play_builds_explicit_shared_actor_core() -> None:
    from unilab.algos.hora.appo import play_hora_appo

    source = textwrap.dedent(inspect.getsource(play_hora_appo))
    tree = ast.parse(source)
    shared_model_keywords = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.keyword)
        and node.arg == "shared_model"
        and isinstance(node.value, ast.Name)
        and node.value.id == "shared_model"
    ]

    assert "build_hora_shared_actor_critic" in source
    assert len(shared_model_keywords) >= 1


def _patch_hora_appo_play_fakes(monkeypatch: pytest.MonkeyPatch, *, actor_cls: type) -> None:
    """Patch env/actor construction fakes for ``play_hora_appo`` contract tests."""
    from types import SimpleNamespace

    import numpy as np
    import rsl_rl.utils as rsl_rl_utils

    import unilab.algos.hora.appo as hora_appo

    fake_env = SimpleNamespace(
        obs_groups_spec={"obs": 3, "critic": 5},
        action_space=SimpleNamespace(shape=(2,)),
        state=SimpleNamespace(
            obs={"obs": np.zeros((1, 3), dtype=np.float32)},
            info={},
        ),
        cfg=SimpleNamespace(render_spacing=1.0),
        run_playback_mode=lambda **kwargs: None,
    )

    class FakeBackendAdapter:
        def __init__(self, *args, **kwargs):
            pass

        def build_task_env_cfg_override(self):
            return {}

    monkeypatch.setattr(hora_appo, "BackendAdapter", FakeBackendAdapter)
    monkeypatch.setattr(hora_appo, "create_env", lambda *args, **kwargs: fake_env)
    monkeypatch.setattr(
        hora_appo,
        "split_hora_obs_with_priv_info",
        lambda obs, info: (
            np.zeros((1, 3), dtype=np.float32),
            None,
            np.zeros((1, 2), dtype=np.float32),
        ),
    )
    monkeypatch.setattr(hora_appo, "is_rsl_rl_v5", lambda: True)
    monkeypatch.setattr(
        hora_appo,
        "build_hora_shared_actor_critic",
        lambda **kwargs: torch.nn.Identity(),
    )
    monkeypatch.setattr(rsl_rl_utils, "resolve_callable", lambda path: actor_cls)


def _hora_appo_play_cfg():
    from omegaconf import OmegaConf

    return OmegaConf.create(
        {
            "training": {
                "task_name": "Task",
                "device": "cpu",
                "play_env_num": 1,
                "cam_distance": 3.0,
                "cam_elevation": -20.0,
                "cam_azimuth": 45.0,
            },
        }
    )


def test_hora_appo_play_runs_sim2sim_preflight_before_checkpoint_load(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    import unilab.algos.hora.appo as hora_appo

    checkpoint = tmp_path / "model_10.pt"
    torch.save({"actor": {"weight": torch.tensor(1.0)}}, checkpoint)
    events: list[str] = []
    preflight_calls: list[tuple] = []

    class FakeActor(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

        def load_state_dict(self, state_dict, strict=True):
            events.append("load")
            return None

    _patch_hora_appo_play_fakes(monkeypatch, actor_cls=FakeActor)

    def fake_preflight(source_run_dir, target_cfg, *, algo_name, strict):
        events.append("preflight")
        preflight_calls.append((source_run_dir, algo_name, strict))
        return target_cfg

    monkeypatch.setattr(hora_appo, "resolve_sim2sim_config", fake_preflight)

    hora_appo.play_hora_appo(
        _hora_appo_play_cfg(),
        {"actor": {"class_name": "fake.Actor"}, "critic": {}},
        root_dir=tmp_path,
        resolve_checkpoint_path=lambda current_cfg: (str(checkpoint), str(tmp_path)),
    )

    assert preflight_calls == [(str(tmp_path), "appo", True)]
    assert events == ["preflight", "load"]


def test_hora_appo_play_dim_mismatch_reraises_explicit_sim2sim_diagnostic(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    capsys,
) -> None:
    import unilab.algos.hora.appo as hora_appo
    from unilab.training.sim2sim import CrossBackendIncompatibleError

    checkpoint = tmp_path / "model_10.pt"
    torch.save({"actor": {"weight": torch.tensor(1.0)}}, checkpoint)

    class MismatchActor(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

        def load_state_dict(self, state_dict, strict=True):
            raise RuntimeError("size mismatch for actor.mlp.0.weight: copying a param ...")

    _patch_hora_appo_play_fakes(monkeypatch, actor_cls=MismatchActor)

    with pytest.raises(CrossBackendIncompatibleError, match="does not fit this play environment"):
        hora_appo.play_hora_appo(
            _hora_appo_play_cfg(),
            {"actor": {"class_name": "fake.Actor"}, "critic": {}},
            root_dir=tmp_path,
            resolve_checkpoint_path=lambda current_cfg: (str(checkpoint), str(tmp_path)),
        )

    # Old runs without a contract snapshot keep the fallback + warning semantics.
    assert "no contract_snapshot" in capsys.readouterr().out


def test_hora_appo_resume_rejects_inconsistent_shared_checkpoint() -> None:
    from unilab.algos.hora.appo_runner import _validate_hora_shared_checkpoint

    learner = _make_hora_appo_learner()
    joint_checkpoint = {
        "actor": copy.deepcopy(learner.actor.state_dict()),
        "critic": copy.deepcopy(learner.critic.state_dict()),
    }

    _validate_hora_shared_checkpoint(joint_checkpoint)

    bad_checkpoint = copy.deepcopy(joint_checkpoint)
    shared_key = next(
        key
        for key, value in bad_checkpoint["critic"].items()
        if key.startswith("shared.") and torch.is_floating_point(value)
    )
    bad_checkpoint["critic"][shared_key] = bad_checkpoint["critic"][shared_key] + 1.0

    with pytest.raises(ValueError, match="Invalid HORA APPO checkpoint"):
        _validate_hora_shared_checkpoint(bad_checkpoint)


def test_hora_appo_combined_optimizer_has_unique_parameters() -> None:
    learner = _make_hora_appo_learner()

    combined_ids = [id(param) for param in learner._combined_params]
    optimizer_ids = [
        id(param) for group in learner.optimizer.param_groups for param in group["params"]
    ]

    assert len(combined_ids) == len(set(combined_ids))
    assert optimizer_ids == combined_ids
    assert len(optimizer_ids) == len(set(optimizer_ids))


def test_hora_appo_update_uses_joint_shared_optimizer() -> None:
    torch.manual_seed(13)
    learner = _make_hora_appo_learner(learning_rate=1e-3)
    observations = torch.randn(2, 3, 5)
    priv_info = torch.randn(2, 3, 2)
    last_obs = torch.randn(3, 5)
    last_priv_info = torch.randn(3, 2)
    batch = {
        "observations": observations,
        "critic": torch.cat([observations, priv_info], dim=-1),
        "actions": torch.randn(2, 3, 3),
        "actions_log_prob": torch.zeros(2, 3),
        "rewards": torch.randn(2, 3),
        "dones": torch.zeros(2, 3),
        "last_obs": last_obs,
        "last_critic": torch.cat([last_obs, last_priv_info], dim=-1),
    }

    trunk_before = [param.detach().clone() for param in learner.actor.shared.trunk.parameters()]

    learner.process_batch(batch)
    metrics = learner.update(batch)

    trunk_after = list(learner.actor.shared.trunk.parameters())

    assert metrics["appo/updates_executed"] == pytest.approx(1.0)
    assert any(
        not torch.allclose(before, after)
        for before, after in zip(trunk_before, trunk_after, strict=True)
    )
