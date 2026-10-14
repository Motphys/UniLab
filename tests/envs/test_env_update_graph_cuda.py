"""Focused real-CUDA tests for the state-update-only graph runtime."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from unisim.backend.mjwarp.dependencies import load_mjwarp_dependencies

from unilab.base import registry
from unilab.base.config_adapter import BackendAdapter
from unilab.base.config_materialization import apply_cfg_overrides
from unilab.base.env_update_graph import validate_env_update_graph_manifest
from unilab.envs import ManagerBasedRlEnv

pytestmark = pytest.mark.slow
ROOT_DIR = Path(__file__).resolve().parents[2]
CONF_DIR = ROOT_DIR / "src" / "unilab" / "conf"


def _require_cuda_mjwarp() -> None:
    if sys.platform == "darwin":
        pytest.skip("mjwarp is a CUDA-only backend; macOS has no supported CUDA runtime")
    dependencies = load_mjwarp_dependencies()
    if not bool(dependencies.warp.get_device().is_cuda):
        pytest.fail("env update graph CUDA tests require an active CUDA Warp device")


def _materialize_env() -> Any:
    _require_cuda_mjwarp()
    registry.ensure_registries()
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_DIR / "ppo"), version_base="1.3"):
        hydra_cfg = compose("config", overrides=["task=g1_walk_flat/mjwarp"])
    override = BackendAdapter(hydra_cfg, root_dir=ROOT_DIR).build_task_env_cfg_override()
    env_cfg = cast(Any, registry.materialize_env_config("G1WalkFlat"))
    apply_cfg_overrides(env_cfg, override)
    env_cfg.validate()
    assert env_cfg.scene is not None
    env_cfg.scene.default_keyframe_name = "stand"
    env_cfg.commands = {}
    env_cfg.observations["policy"].enable_corruption = False
    env_cfg.observations["critic"].enable_corruption = False
    from unilab.managers import ObservationTermCfg

    policy_obs_carrier = torch.zeros((2, 1), dtype=torch.float32, device="cuda")

    def policy_obs(_env):
        return policy_obs_carrier

    critic_obs_carrier = torch.zeros((2, 2), dtype=torch.float32, device="cuda")

    def critic_obs(_env):
        return critic_obs_carrier

    setattr(policy_obs, "returns_transient_tensor", True)
    setattr(critic_obs, "returns_transient_tensor", True)
    env_cfg.observations["policy"].terms = {"constant": ObservationTermCfg(func=policy_obs)}
    env_cfg.observations["critic"].terms = {"constant": ObservationTermCfg(func=critic_obs)}
    for group in env_cfg.observations.values():
        for term in group.terms.values():
            term.noise = None
    # Start with an allocation-safe scalar reward and add graph terms after the
    # first production candidate proves the fixed carrier invariant.
    from unilab.managers import RewardTermCfg

    def reward(env):
        return env.action_manager.action[:, 0]

    setattr(reward, "returns_transient_tensor", True)
    env_cfg.rewards = {"constant": RewardTermCfg(func=reward, weight=1.0)}
    env_cfg.terminations = {
        "time_out": env_cfg.terminations["time_out"],
    }
    # Revalidate after the focused first-candidate terms remove configured noise.
    assert env_cfg.max_episode_seconds == 20.0
    env_cfg.validate()
    del override
    from unilab.tasks.locomotion.g1.manager_terms import make_g1_walk_env

    return make_g1_walk_env(env_cfg, num_envs=2, backend_type="mjwarp")


def test_real_cuda_env_update_graph_capture_replay_and_allocator_invariant() -> None:
    env = _materialize_env()
    try:
        assert isinstance(env, ManagerBasedRlEnv)
        assert env.device.type == "cuda"
        env.set_deferred_validation_boundaries(True)
        env.reset(seed=7)
        # This allocator-focused candidate starts from a deterministic non-terminal
        # boundary; semantic reset parity belongs to the later Gate U suite.
        env._state.info["steps"].zero_()
        env.episode_length_buf.zero_()
        env.configure_env_update_graph("graph")

        manifest = env.get_env_update_graph_manifest()
        assert manifest["enabled"] is True
        assert manifest["state"] == "ready"
        assert manifest["replay_count"] == 1

        actions = torch.zeros(
            (env.num_envs, env.action_manager.total_action_dim),
            dtype=torch.float32,
            device=env.device,
        )
        for _ in range(4):
            env.step(actions)
            manifest = env.get_env_update_graph_manifest()
            assert manifest["enabled"] is True
            assert manifest["state"] == "ready"
            assert manifest["replay_count"] > 1
    finally:
        env.close()


def test_real_cuda_env_update_graph_semantic_parity_and_fallback() -> None:
    env = _materialize_env()
    try:
        assert isinstance(env, ManagerBasedRlEnv)
        assert env.device.type == "cuda"
        env.set_deferred_validation_boundaries(True)
        env.reset(seed=7)
        # Force deterministic timeout sequences from a common pre-transition
        # boundary: no-done -> empty-done -> partial-done -> full-done.
        env._state.info["steps"].zero_()
        env.episode_length_buf.zero_()

        def actions(values: list[float]) -> torch.Tensor:
            row = torch.tensor(values, dtype=torch.float32, device=env.device)
            return (
                row[:, None].expand(env.num_envs, env.action_manager.total_action_dim).contiguous()
            )

        # Establish graph mode after the deterministic common boundary.
        env.configure_env_update_graph("graph")
        capture_id = env.get_env_update_graph_manifest()["capture_id"]
        assert capture_id is not None
        owner_scratch = env._reset_state._owner_tensor_write_mask
        assert owner_scratch is not None
        owner_scratch_pointer = owner_scratch.data_ptr()

        sequence = (
            ("no_done", actions([0.0, 0.0])),
            ("empty_done", actions([0.0, 0.0])),
            ("partial_done", actions([0.0, 0.0])),
            ("full_done", actions([0.0, 0.0])),
        )
        # Empty-done is represented by a subsequent transition whose done mask
        # is all false; a host branch would incorrectly publish reset evidence.
        # timeout occurs at episode step 2; prepare exact rows for each case.
        # The first case remains below timeout. For the latter cases, steps are
        # manipulated before the transition through public state.info carrier.
        state = env._state
        assert state is not None
        state.info["steps"].fill_(0)
        env.episode_length_buf.zero_()
        bootstrap_completion = env.last_reset_completion
        assert bootstrap_completion is not None
        last_reset_sequence = bootstrap_completion.sequence

        for index, (name, action) in enumerate(sequence):
            if name == "partial_done":
                state.info["steps"][0] = env.max_episode_length
                state.info["steps"][1] = 0
                env.episode_length_buf.copy_(state.info["steps"])
            elif name == "full_done":
                state.info["steps"].fill_(env.max_episode_length)
                env.episode_length_buf.fill_(env.max_episode_length)
            before = env.get_env_update_graph_manifest()
            result = env.step(action)
            after = env.get_env_update_graph_manifest()
            assert after["enabled"] is True
            assert after["capture_id"] == capture_id
            assert after["replay_count"] == before["replay_count"] + 1
            assert after["backend_step_receipt"]["operation"] == "step"
            done = result.terminated | result.truncated
            expected_done = {
                "no_done": [False, False],
                "empty_done": [False, False],
                "partial_done": [True, False],
                "full_done": [True, True],
            }[name]
            assert done.detach().cpu().tolist() == expected_done, name
            reset_completion = result.info["graph_reset_completion"]
            if any(expected_done):
                assert reset_completion.operation == "mask_reset"
                if last_reset_sequence is not None:
                    assert reset_completion.sequence > last_reset_sequence
                last_reset_sequence = reset_completion.sequence
            else:
                assert last_reset_sequence is not None
            reset_state = env._reset_state
            assert reset_state.active is False
            assert reset_state.last_commit_had_writes is any(expected_done)
            owner_mask = reset_state._owner_tensor_write_mask
            assert owner_mask is not None
            assert owner_mask.data_ptr() == owner_scratch_pointer
            assert not bool(owner_mask.any())
            if any(expected_done):
                # The reset transaction owns the selected-row evidence; the
                # command owner scratch stays clear after its graph commit.
                assert env.last_reset_completion is not None
                assert env.last_reset_completion.operation == "mask_reset"
            else:
                assert env.last_reset_completion.operation == "mask_reset"
            torch.testing.assert_close(
                result.obs["obs"].detach().cpu(), torch.zeros_like(result.obs["obs"].cpu())
            )
            del index

        # A repeated replay must preserve allocator and capture identity.
        repeated_manifest = env.get_env_update_graph_manifest()
        result = env.step(actions([0.0, 0.0]))
        del result
        final_manifest = env.get_env_update_graph_manifest()
        assert final_manifest["replay_count"] == repeated_manifest["replay_count"] + 1
        assert final_manifest["capture_id"] == capture_id

        # Explicit eager fallback resets graph counters/manifest state without
        # contaminating the prior backend transition sequence.
        env.configure_env_update_graph("eager")
        fallback = env.get_env_update_graph_manifest()
        validate_env_update_graph_manifest(fallback)
        assert fallback["mode"] == "eager"
        assert fallback["fallback_count"] == 0
        assert fallback["replay_count"] == 0
        assert fallback["capture_id"] is None
        fallback_state = env.step(actions([0.0, 0.0]))
        assert fallback_state.terminated.device == env.device
    finally:
        env.close()
