"""MicroDuck training-infra alignment (issue #1456, child 4/5).

Covers the two runtime-facing pieces of the infra alignment:

- env-level RNG seeding: the three ``microduck_*/mjwarp`` owners set
  ``env.seed: 42``; the Hydra -> BackendAdapter -> registry -> ManagerBasedRlEnvCfg
  chain must thread it through so command/noise/DR sampling is reproducible
  across runs (default ``None`` behavior stays unchanged).
- ``init_at_random_ep_len``: RSL-RL's OnPolicyRunner assigns a fresh tensor to
  ``env.episode_length_buf`` at learn() start; the RslRlVecEnvWrapper setter
  must propagate it into the env's real episode counters so initial episode
  timeouts actually stagger (matching upstream mjlab, where the setter writes
  the env's buffer directly).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from uni_rl.algos.rsl_rl import RslRlVecEnvWrapper

from unilab.base import registry
from unilab.base.config_adapter import BackendAdapter
from unilab.base.config_materialization import apply_cfg_overrides
from unilab.envs import ManagerBasedRlEnvCfg

ROOT_DIR = Path(__file__).parents[4]
CONF_DIR = ROOT_DIR / "src" / "unilab" / "conf" / "ppo"

MICRODUCK_TASKS = (
    "microduck_velocity_flat",
    "microduck_sitstand_flat",
    "microduck_ground_pick_flat",
)


def _compose(task_owner: str) -> Any:
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_DIR), version_base="1.3"):
        return compose("config", overrides=[f"task={task_owner}"])


def _materialize_env_cfg(task_owner: str) -> ManagerBasedRlEnvCfg:
    cfg = _compose(task_owner)
    registry.ensure_registries()
    env_cfg = registry.materialize_env_config(str(cfg.training.task_name))
    assert isinstance(env_cfg, ManagerBasedRlEnvCfg)
    apply_cfg_overrides(
        env_cfg,
        BackendAdapter(cfg, root_dir=ROOT_DIR, algo_name="ppo").build_task_env_cfg_override(),
    )
    env_cfg.validate()
    return env_cfg


@pytest.mark.parametrize("task", MICRODUCK_TASKS)
def test_mjwarp_owner_threads_env_seed_from_hydra(task: str) -> None:
    """env.seed=42 in the mjwarp owner reaches ManagerBasedRlEnvCfg.seed."""
    env_cfg = _materialize_env_cfg(f"{task}/mjwarp")
    assert env_cfg.seed == 42


def test_env_seed_defaults_to_none_when_unset() -> None:
    """Backward compatibility: owners without env.seed keep the None default."""
    env_cfg = _materialize_env_cfg("microduck_velocity_flat/mujoco")
    assert env_cfg.seed is None


def _make_velocity_env(seed: int, num_envs: int = 8):
    cfg = _compose("microduck_velocity_flat/mujoco")
    registry.ensure_registries()
    override = BackendAdapter(cfg, root_dir=ROOT_DIR, algo_name="ppo").build_task_env_cfg_override()
    override["seed"] = seed
    return registry.make(
        str(cfg.training.task_name),
        num_envs=num_envs,
        sim_backend="mujoco",
        env_cfg_override=override,
    )


def _command_trace(env: Any, rounds: int = 3) -> list[np.ndarray]:
    all_ids = np.arange(env.num_envs, dtype=np.int32)
    env.reset(all_ids)
    trace = [env.command_manager.get_command("twist").copy()]
    for _ in range(rounds):
        env.reset(all_ids)
        trace.append(env.command_manager.get_command("twist").copy())
    return trace


def test_env_seed_makes_command_sampling_reproducible() -> None:
    """Same env seed -> identical command sampling sequences across runs."""
    env_a = _make_velocity_env(seed=42)
    env_b = _make_velocity_env(seed=42)
    try:
        trace_a = _command_trace(env_a)
        trace_b = _command_trace(env_b)
    finally:
        env_a.close()
        env_b.close()
    for step, (cmd_a, cmd_b) in enumerate(zip(trace_a, trace_b, strict=True)):
        np.testing.assert_array_equal(cmd_a, cmd_b, err_msg=f"reset round {step}")


def test_env_seed_difference_changes_command_sampling() -> None:
    """Different env seeds -> different command sampling sequences."""
    env_a = _make_velocity_env(seed=42)
    env_b = _make_velocity_env(seed=43)
    try:
        trace_a = _command_trace(env_a)
        trace_b = _command_trace(env_b)
    finally:
        env_a.close()
        env_b.close()
    assert any(
        not np.array_equal(cmd_a, cmd_b) for cmd_a, cmd_b in zip(trace_a, trace_b, strict=True)
    )


def test_random_ep_len_assignment_staggers_env_episode_counters() -> None:
    """The wrapper's episode_length_buf setter propagates into the env.

    Mirrors ``OnPolicyRunner.learn(init_at_random_ep_len=True)``: assigning a
    randomized tensor to ``wrapped.episode_length_buf`` must stagger the env's
    own timeout counters so initial episodes do not all end in lockstep.
    """
    num_envs = 16
    env = _make_velocity_env(seed=42, num_envs=num_envs)
    try:
        wrapped = RslRlVecEnvWrapper(env, device="cpu")

        torch.manual_seed(0)
        wrapped.episode_length_buf = torch.randint_like(
            wrapped.episode_length_buf, high=int(wrapped.max_episode_length)
        )

        expected = wrapped.episode_length_buf.cpu().numpy().astype(np.int64)
        np.testing.assert_array_equal(env.episode_length_buf, expected)

        remaining = env.max_episode_length - env.episode_length_buf
        assert len(np.unique(remaining)) > 1, (
            "initial episode stagger must leave non-uniform remaining lengths"
        )

        # Step until the shortest episode times out: truncation must be
        # staggered (some envs time out, others keep running).
        action_dim = env.action_space.shape[0]
        assert action_dim is not None
        actions = np.zeros((num_envs, action_dim), dtype=np.float32)
        state = None
        for _ in range(int(remaining.min())):
            state = env.step(actions)
        assert state is not None
        assert state.truncated.any()
        assert not state.truncated.all()
    finally:
        env.close()
