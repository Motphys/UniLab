"""
Stage 2 prototype B: validate NaN injection on REAL HIM-PPO runner with real
Go2ArmManipLoco env (mujoco backend). Mirrors the patches from
proto_nan_inject.py but applied to the actual training loop.

This proves the helper functions work on a real algo runner before lifting
them into the formal pytest file.

Run:
    .venv/bin/python tests/nan_injection/proto_him_ppo_inject.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

# Resolve repo root from this file's location
ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "src"))

from unilab.algos.torch.him_ppo.runner import HIMOnPolicyRunner  # noqa: E402
from unilab.base.backend.mujoco.xml import materialize_scene_visual_override  # noqa: E402
from unilab.training import BackendAdapter, create_env, ensure_registries  # noqa: E402
from unilab.training.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from unilab.utils.nan_guard import NanGuard, NanGuardCfg  # noqa: E402


def _list_dumps(output_dir: Path):
    return [p for p in output_dir.glob("nan_dump_*.npz") if "latest" not in p.name]


def _build_him_ppo(num_envs: int, log_dir: str, output_dir: Path):
    """Build a minimal HIM-PPO runner with attached NanGuard, mirroring train_him_ppo.py.

    Placeholder: hand-built cfg approach was abandoned in favor of the hydra-compose
    path in ``build_via_hydra`` below.
    """
    ensure_registries()
    return None  # placeholder, switch to hydra below


def build_via_hydra(num_envs: int, log_dir: str, output_dir: Path):
    """Use hydra compose API to build the real cfg, then mirror train_him_ppo.py wiring."""
    from hydra import compose, initialize_config_dir

    ensure_registries()

    config_dir = str(ROOT_DIR / "conf/ppo_him")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(
            config_name="config",
            overrides=[
                "task=go2_arm_manip_loco/mujoco",
                f"algo.num_envs={num_envs}",
                "algo.num_steps_per_env=4",
                "algo.max_iterations=1",
                "algo.save_interval=100",
                f"training.nan_guard.output_dir={output_dir}",
                "training.no_play=true",
            ],
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[build] device={device}")

    backend_adapter = BackendAdapter(
        cfg,
        root_dir=ROOT_DIR,
        algo_name="ppo_him",
        scene_materializer=materialize_scene_visual_override,
    )
    env_cfg_override = backend_adapter.build_task_env_cfg_override()
    env = create_env(cfg, num_envs=cfg.algo.num_envs, env_cfg_override=env_cfg_override)

    # Attach NanGuard exactly like train_him_ppo.py does.
    nan_guard_cfg = cfg.training.nan_guard
    guard = NanGuard(
        NanGuardCfg(
            enabled=True,
            buffer_size=int(nan_guard_cfg.buffer_size),
            max_envs_to_dump=int(nan_guard_cfg.max_envs_to_dump),
            output_dir=str(output_dir),
        ),
        num_envs=env.num_envs,
        supports_state_playback=env.play_capabilities.supports_physics_state_playback,
    )
    env.set_nan_guard(guard)

    wrapped_env = RslRlVecEnvWrapper(env, device=device)
    rl_cfg = OmegaConf.to_container(cfg.algo, resolve=True)
    runner = HIMOnPolicyRunner(wrapped_env, rl_cfg, log_dir=log_dir, device=device)
    return env, runner, guard


def proto_him_ppo_obs_nan():
    print("=" * 60)
    print("[B1] HIM-PPO REAL runner — OBS NaN injection")
    print("=" * 60)
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        output_dir = td_path / "dumps"
        output_dir.mkdir()
        log_dir = td_path / "log"
        log_dir.mkdir()
        env, runner, guard = build_via_hydra(
            num_envs=8, log_dir=str(log_dir), output_dir=output_dir
        )

        # Patch update_state to inject NaN at step K=2 ONLY (one-shot).
        # Rationale: nan_guard only clears reward via nan_to_num — obs NaN would propagate
        # to actor and break torch distribution sampling. By restoring after one step we
        # confirm dump fires but allow training to complete.
        orig_update = env.update_state
        K = 2
        calls = [0]

        def update_with_nan(state):
            new_state = orig_update(state)
            calls[0] += 1
            if calls[0] == K:
                first_key = next(iter(new_state.obs))
                new_state.obs[first_key][0, 0] = np.nan
                env.update_state = orig_update  # restore — single-shot injection
            return new_state

        env.update_state = update_with_nan  # type: ignore

        try:
            runner.learn(num_learning_iterations=1, init_at_random_ep_len=False)
        except RuntimeError as e:
            # Acceptable: NaN may still propagate one step before being cleaned.
            print(f"  [info] training raised post-injection (expected): {type(e).__name__}")

        files = _list_dumps(output_dir)
        print(f"  guard._dumped: {bool(guard._dumped)}")
        print(f"  dump files: {[f.name for f in files]}")
        assert guard._dumped, "obs NaN should have triggered dump"
        assert len(files) >= 1, f"expected ≥1 dump, got {len(files)}"
        env.close()
        print("  PASS")


def proto_him_ppo_reward_nan():
    print("=" * 60)
    print("[B2] HIM-PPO REAL runner — REWARD NaN injection")
    print("=" * 60)
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        output_dir = td_path / "dumps"
        output_dir.mkdir()
        log_dir = td_path / "log"
        log_dir.mkdir()
        env, runner, guard = build_via_hydra(
            num_envs=8, log_dir=str(log_dir), output_dir=output_dir
        )

        # Reward NaN is sanitized via nan_to_num after dump, so single-shot is safe to
        # leave permanent — but for symmetry with obs/ctrl tests, we still one-shot it.
        orig_update = env.update_state
        K = 2
        calls = [0]

        def update_with_nan(state):
            new_state = orig_update(state)
            calls[0] += 1
            if calls[0] == K:
                new_state.reward[0] = np.nan
                env.update_state = orig_update
            return new_state

        env.update_state = update_with_nan  # type: ignore

        try:
            runner.learn(num_learning_iterations=1, init_at_random_ep_len=False)
        except RuntimeError as e:
            print(f"  [info] training raised post-injection (expected): {type(e).__name__}")

        files = _list_dumps(output_dir)
        print(f"  guard._dumped: {bool(guard._dumped)}")
        print(f"  dump files: {[f.name for f in files]}")
        assert guard._dumped, "reward NaN should have triggered dump"
        assert len(files) >= 1, f"expected ≥1 dump, got {len(files)}"
        env.close()
        print("  PASS")


def proto_him_ppo_ctrl_nan():
    print("=" * 60)
    print("[B3] HIM-PPO REAL runner — CTRL NaN injection (via apply_action)")
    print("=" * 60)
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        output_dir = td_path / "dumps"
        output_dir.mkdir()
        log_dir = td_path / "log"
        log_dir.mkdir()
        env, runner, guard = build_via_hydra(
            num_envs=8, log_dir=str(log_dir), output_dir=output_dir
        )

        # ctrl NaN is intercepted by check_ctrl BEFORE backend.step — never propagates
        # to physics. So one-shot restore not strictly needed, but use it for consistency.
        orig_apply = env.apply_action
        K = 2
        calls = [0]

        def apply_with_nan(actions, state):
            ctrl = orig_apply(actions, state)
            calls[0] += 1
            if calls[0] == K:
                ctrl = ctrl.copy()
                ctrl[0, 0] = np.nan
                env.apply_action = orig_apply
            return ctrl

        env.apply_action = apply_with_nan  # type: ignore

        try:
            runner.learn(num_learning_iterations=1, init_at_random_ep_len=False)
        except RuntimeError as e:
            print(f"  [info] training raised post-injection (expected): {type(e).__name__}")

        files = _list_dumps(output_dir)
        print(f"  guard._dumped: {bool(guard._dumped)}")
        print(f"  dump files: {[f.name for f in files]}")
        assert guard._dumped, "ctrl NaN should have triggered dump (check_ctrl path)"
        assert len(files) >= 1, f"expected ≥1 dump, got {len(files)}"
        env.close()
        print("  PASS")


if __name__ == "__main__":
    proto_him_ppo_obs_nan()
    proto_him_ppo_reward_nan()
    proto_him_ppo_ctrl_nan()
    print()
    print("All 3 HIM-PPO real-runner prototypes PASSED.")
