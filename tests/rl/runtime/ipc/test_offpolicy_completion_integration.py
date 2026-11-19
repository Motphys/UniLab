"""Cross-package completion integration through the off-policy replay ingress."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from unilab.base import registry
from unilab.base.config_adapter import EnvConfigAdapter
from unilab.rl.runtime.ipc.replay_buffer import ReplayBuffer
from unilab.rl.runtime.offpolicy.worker import timeout_bootstrap_mask

ROOT_DIR = Path(__file__).parents[4]
CONF_DIR = ROOT_DIR / "src" / "unilab" / "conf"


def test_eager_mujoco_replay_rows_own_their_tick_across_autoreset() -> None:
    """The no-receipt eager path must publish replay rows that own their tick.

    Synchronous host-bridge backends emit no completion receipt; ingress
    ordering is owned by the replay buffer itself. Each replay row's obs[t]
    and next_obs[t] must belong to transition tick t, including across an
    autoreset boundary where a done row's next observation is the terminal
    pre-reset observation and the next tick starts from the post-reset obs.
    """
    pytest.importorskip(
        "unisim.backend.mujoco.backend",
        reason="unisim-core MuJoCo adapter (mjbatch build) not available",
    )

    registry.ensure_registries()
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_DIR / "ppo"), version_base="1.3"):
        cfg = compose("config", overrides=["task=go2_joystick_flat/mujoco"])
    override = EnvConfigAdapter(cfg, root_dir=ROOT_DIR).build_task_env_cfg_override()
    # The replay integration drives the eager selected-row lifecycle on CPU.
    override["autoreset_cuda_graph"] = False
    env = registry.make(
        str(cfg.training.task_name),
        sim_backend="mujoco",
        env_cfg_override=override,
        num_envs=2,
    )
    replay = ReplayBuffer(
        capacity=16,
        obs_dim=env.obs_groups_spec["obs"],
        action_dim=env.action_space.shape[0],
        device="cpu",
        ingress_slot_rows=2,
        critic_dim=env.obs_groups_spec["critic"],
    )

    try:
        initial_obs, _ = env.reset(seed=2359)
        # The mujoco host bridge is a synchronous producer: it declares the
        # no-receipt semantic, so CUDA replay ingress must not demand one.
        assert env.produces_completion_receipts is False
        # Normalize the randomized bootstrap counters so the forced time-out at
        # the autoreset tick is the only episode boundary in this sequence.
        env.set_episode_length_buf(torch.zeros(env.num_envs, dtype=torch.int64, device=env.device))
        autoreset_tick = 2
        autoreset_terminal: dict[str, torch.Tensor] | None = None
        current_obs = initial_obs
        states = []
        packed_rows: list[torch.Tensor] = []
        source_snapshots: list[dict[str, torch.Tensor]] = []
        next_snapshots: list[dict[str, torch.Tensor]] = []
        expected_replay_next: list[dict[str, torch.Tensor]] = []

        for tick in range(5):
            if tick == autoreset_tick:
                env.set_episode_length_buf(
                    torch.full(
                        (env.num_envs,),
                        env.max_episode_length - 1,
                        dtype=torch.int64,
                        device=env.device,
                    )
                )
            source_snapshots.append({name: value.clone() for name, value in current_obs.items()})
            actions = torch.full(
                (2, env.action_space.shape[0]),
                (tick + 1) * 0.02,
                dtype=torch.float32,
                device=env.device,
            )
            state = env.step(actions)
            states.append(state)
            next_snapshots.append({name: value.clone() for name, value in state.obs.items()})

            # Eager synchronous host-bridge steps emit no completion receipt.
            completion = state.info.get("completion")
            assert completion is None

            done = state.terminated | state.truncated
            terminal_final = state.final_observation
            terminal_obs = terminal_final.get("obs") if terminal_final is not None else None
            terminal_critic = terminal_final.get("critic") if terminal_final is not None else None
            if tick == autoreset_tick:
                # The forced time-out must drive a full-batch autoreset here.
                assert bool(done.all())
                assert terminal_obs is not None and terminal_critic is not None
                # The live state carrier is reused by later steps; snapshot the
                # terminal observation now for the end-of-sequence assertions.
                autoreset_terminal = {
                    "obs": terminal_obs.clone(),
                    "critic": terminal_critic.clone(),
                }

            assert replay.add_batch(
                current_obs["obs"],
                actions,
                state.reward.to(dtype=torch.float32).ravel(),
                state.obs["obs"],
                done.to(dtype=torch.float32),
                # Match the collector write boundary: the replay `truncated`
                # column carries the timeout-bootstrap eligibility mask.
                timeout_bootstrap_mask(state.terminated, state.truncated).ravel(),
                terminal_mask=done,
                terminal_next_obs=terminal_obs,
                critic=current_obs["critic"],
                next_critic=state.obs["critic"],
                terminal_next_critic=terminal_critic,
                completion=completion,
            )
            # Host ingress has no device publication event to export.
            assert replay.last_completion is None

            ingress = replay.take_published_ingress()
            assert ingress is not None
            slot, start, count, packed = ingress
            replay.commit_ingress(slot=slot, start=start, count=count)
            assert (start, count) == (tick * 2, 2)
            packed_rows.append(packed.clone())
            expected_replay_next.append(
                {
                    group: torch.where(done[:, None], terminal, state.obs[group])
                    if terminal is not None
                    else state.obs[group].clone()
                    for group, terminal in (("obs", terminal_obs), ("critic", terminal_critic))
                }
            )
            current_obs = state.obs

        # A public observation remains valid after later eager steps even when
        # the consumer and replay publication retain it without an extra clone,
        # and every replay row keeps the obs/next_obs of its own transition
        # tick — for done rows next_obs is the terminal pre-reset observation.
        for tick, (state, packed) in enumerate(zip(states, packed_rows, strict=True)):
            for group, field in (("obs", "obs"), ("critic", "critic")):
                torch.testing.assert_close(
                    replay.field_view(packed, field),
                    source_snapshots[tick][group],
                )
            for group, field in (("obs", "next_obs"), ("critic", "next_critic")):
                torch.testing.assert_close(state.obs[group], next_snapshots[tick][group])
                torch.testing.assert_close(
                    replay.field_view(packed, field),
                    expected_replay_next[tick][group],
                )
        # The autoreset tick must prove the terminal patch really engaged.
        assert autoreset_terminal is not None
        torch.testing.assert_close(
            replay.field_view(packed_rows[autoreset_tick], "next_obs"),
            autoreset_terminal["obs"],
        )
        torch.testing.assert_close(
            replay.field_view(packed_rows[autoreset_tick], "next_critic"),
            autoreset_terminal["critic"],
        )
        # The tick after autoreset starts from the post-reset observation.
        torch.testing.assert_close(
            replay.field_view(packed_rows[autoreset_tick + 1], "obs"),
            next_snapshots[autoreset_tick]["obs"],
        )
    finally:
        replay.close()
        env.close()
