"""
Stage 2: NaN injection validation for single-process algorithms.

Covers:
  PPO (rsl_rl)  x {obs, reward, ctrl} = 3 cases
  HIM-PPO       x {obs, reward, ctrl} = 3 cases

For each case: build a real runner via Hydra compose, attach NanGuard, patch
env to inject NaN at step K (one-shot), assert nan_guard fires + dump file
lands with metadata pointing at step K.

Run:
    .venv/bin/python tests/nan_injection/stage2_nan_inject.py
"""

from __future__ import annotations

import sys
import tempfile
import traceback
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

# Resolve repo root from this file's location: <repo>/tests/nan_injection/<this>
ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "src"))

from unilab.base.backend.mujoco.xml import materialize_scene_visual_override  # noqa: E402
from unilab.training import BackendAdapter, create_env, ensure_registries  # noqa: E402
from unilab.training.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from unilab.utils.nan_guard import NanGuard, NanGuardCfg  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _list_dumps(output_dir: Path) -> list[Path]:
    """Return only timestamped dump files, excluding the 'latest' pointer."""
    return [p for p in output_dir.glob("nan_dump_*.npz") if "latest" not in p.name]


def _attach_guard(env, output_dir: Path) -> NanGuard:
    cfg = NanGuardCfg(enabled=True, output_dir=str(output_dir))
    guard = NanGuard(
        cfg,
        num_envs=env.num_envs,
        supports_state_playback=env.play_capabilities.supports_physics_state_playback,
    )
    env.set_nan_guard(guard)
    return guard


# ---------------------------------------------------------------------------
# Builders: build real runner + env + attached NanGuard
# ---------------------------------------------------------------------------


def build_him_ppo(num_envs: int, log_dir: Path, output_dir: Path):
    from unilab.algos.torch.him_ppo.runner import HIMOnPolicyRunner

    ensure_registries()
    with initialize_config_dir(config_dir=str(ROOT_DIR / "conf/ppo_him"), version_base=None):
        cfg = compose(
            config_name="config",
            overrides=[
                "task=go2_arm_manip_loco/mujoco",
                f"algo.num_envs={num_envs}",
                "algo.num_steps_per_env=4",
                "algo.max_iterations=1",
                "algo.save_interval=100",
                f"training.nan_guard.output_dir={output_dir}",
            ],
        )

    backend_adapter = BackendAdapter(
        cfg,
        root_dir=ROOT_DIR,
        algo_name="ppo_him",
        scene_materializer=materialize_scene_visual_override,
    )
    env_cfg_override = backend_adapter.build_task_env_cfg_override()
    env = create_env(cfg, num_envs=cfg.algo.num_envs, env_cfg_override=env_cfg_override)
    guard = _attach_guard(env, output_dir)

    device = _device()
    wrapped_env = RslRlVecEnvWrapper(env, device=device)
    rl_cfg = OmegaConf.to_container(cfg.algo, resolve=True)
    runner = HIMOnPolicyRunner(wrapped_env, rl_cfg, log_dir=str(log_dir), device=device)
    return env, runner, guard


def build_ppo_rsl_rl(num_envs: int, log_dir: Path, output_dir: Path):
    from rsl_rl.runners import OnPolicyRunner

    from unilab.training.experiment import patch_rsl_rl_resume_state
    from unilab.training.rsl_rl import normalize_ppo_train_cfg

    # Import apply_ppo_runtime_flags from train_rsl_rl script
    sys.path.insert(0, str(ROOT_DIR / "scripts"))
    from train_rsl_rl import apply_ppo_runtime_flags

    ensure_registries()
    with initialize_config_dir(config_dir=str(ROOT_DIR / "conf/ppo"), version_base=None):
        cfg = compose(
            config_name="config",
            overrides=[
                "task=go1_joystick_flat/mujoco",
                f"algo.num_envs={num_envs}",
                "algo.num_steps_per_env=4",
                "algo.max_iterations=1",
                "algo.save_interval=100",
                "training.nan_guard.enabled=true",
                f"training.nan_guard.output_dir={output_dir}",
            ],
        )

    backend_adapter = BackendAdapter(
        cfg,
        root_dir=ROOT_DIR,
        algo_name="rsl_rl_ppo",
        scene_materializer=materialize_scene_visual_override,
    )
    env_cfg_override = backend_adapter.build_task_env_cfg_override()
    env = create_env(cfg, num_envs=cfg.algo.num_envs, env_cfg_override=env_cfg_override)
    guard = _attach_guard(env, output_dir)

    device = _device()
    rl_cfg = OmegaConf.to_container(cfg.algo, resolve=True)
    train_cfg = normalize_ppo_train_cfg(rl_cfg)
    apply_ppo_runtime_flags(train_cfg, cfg, training_enabled=True)
    if "runner" not in train_cfg:
        train_cfg["runner"] = {}
    train_cfg["runner"]["logger"] = "none"
    train_cfg["logger"] = "none"
    patch_rsl_rl_resume_state()

    wrapped_env = RslRlVecEnvWrapper(env, device=device)
    runner = OnPolicyRunner(wrapped_env, train_cfg, log_dir=None, device=device)
    return env, runner, guard


# ---------------------------------------------------------------------------
# Injection patches: one-shot NaN injection at step K
# ---------------------------------------------------------------------------


def _patch_obs_nan(env, K: int):
    orig_update = env.update_state
    calls = [0]

    def update_with_nan(state):
        new_state = orig_update(state)
        calls[0] += 1
        if calls[0] == K:
            first_key = next(iter(new_state.obs))
            new_state.obs[first_key][0, 0] = np.nan
            env.update_state = orig_update  # one-shot restore
        return new_state

    env.update_state = update_with_nan  # type: ignore


def _patch_reward_nan(env, K: int):
    orig_update = env.update_state
    calls = [0]

    def update_with_nan(state):
        new_state = orig_update(state)
        calls[0] += 1
        if calls[0] == K:
            new_state.reward[0] = np.nan
            env.update_state = orig_update
        return new_state

    env.update_state = update_with_nan  # type: ignore


def _patch_ctrl_nan(env, K: int):
    orig_apply = env.apply_action
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


# ---------------------------------------------------------------------------
# Test runner wrapper
# ---------------------------------------------------------------------------


def _run_case(
    name: str,
    builder: Callable,
    patch_fn: Callable,
    K: int,
    expected_step: int,
    suppress_runtime_err: bool = True,
) -> tuple[bool, str]:
    """Run one injection case, return (passed, message).

    K: which call to the patched method triggers NaN injection (1-indexed).
    expected_step: expected value of meta_detection_step in the dump.
      For obs/reward (checked after step_counter increment): expected_step == K.
      For ctrl (checked before step_counter increment):      expected_step == K - 1.
    """
    try:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            output_dir = td_path / "dumps"
            output_dir.mkdir()
            log_dir = td_path / "log"
            log_dir.mkdir()

            env, runner, guard = builder(num_envs=8, log_dir=log_dir, output_dir=output_dir)
            patch_fn(env, K)

            try:
                runner.learn(num_learning_iterations=1, init_at_random_ep_len=False)
            except (RuntimeError, ValueError):
                if not suppress_runtime_err:
                    raise
                # Expected: NaN may propagate post-dump in obs/reward path
                # rsl_rl's check_nan raises ValueError; torch sampling raises RuntimeError

            files = _list_dumps(output_dir)
            if not guard._dumped:
                return False, "guard._dumped=False (no dump triggered)"
            if len(files) < 1:
                return False, f"expected ≥1 dump file, got {len(files)}"

            data = np.load(files[0], allow_pickle=True)
            if "states" not in data.files:
                return False, "'states' missing from dump npz"
            detected_step = int(data["meta_detection_step"])
            if detected_step != expected_step:
                return False, f"detected_step={detected_step}, expected {expected_step}"

            env.close()
            return True, f"OK (dump at step {detected_step})"

    except Exception as e:
        return False, f"EXCEPTION: {type(e).__name__}: {e}\n{traceback.format_exc()}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("=" * 70)
    print("Stage 2: Single-process NaN injection validation")
    print("=" * 70)
    print()

    cases = [
        # (name, builder, patch_fn, K, expected_step)
        # K = which call (1-indexed) triggers injection
        # expected_step = step recorded in dump (K for obs/reward, K-1 for ctrl)
        ("PPO obs NaN", build_ppo_rsl_rl, _patch_obs_nan, 2, 2),
        ("PPO reward NaN", build_ppo_rsl_rl, _patch_reward_nan, 2, 2),
        ("PPO ctrl NaN", build_ppo_rsl_rl, _patch_ctrl_nan, 3, 2),
        ("HIM-PPO obs NaN", build_him_ppo, _patch_obs_nan, 2, 2),
        ("HIM-PPO reward NaN", build_him_ppo, _patch_reward_nan, 2, 2),
        ("HIM-PPO ctrl NaN", build_him_ppo, _patch_ctrl_nan, 3, 2),
    ]

    results = []
    for idx, (name, builder, patch_fn, K, expected_step) in enumerate(cases, start=1):
        print(f"[{idx}/{len(cases)}] {name:<25}", end="", flush=True)
        passed, msg = _run_case(name, builder, patch_fn, K, expected_step)
        status = "PASS" if passed else "FAIL"
        print(f" {status}")
        if not passed:
            print(f"    {msg}")
        results.append((name, passed))

    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    passed_count = sum(1 for _, p in results if p)
    print(f"{passed_count}/{len(results)} cases passed")
    for name, passed in results:
        status = "✓" if passed else "✗"
        print(f"  {status} {name}")
    print()

    if passed_count == len(results):
        print("All cases PASSED. NaN injection mechanics validated.")
        return 0
    else:
        print(f"{len(results) - passed_count} case(s) FAILED.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
