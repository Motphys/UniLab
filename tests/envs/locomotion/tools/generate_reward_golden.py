"""Generate golden reward fixtures for locomotion regression testing.

These fixtures pin the *current* reward behavior of the locomotion envs before
any Phase 4 refactoring (reward pure-function extraction, joystick base class,
DR provider convergence). Any refactor that changes reward values — even by
1e-7 — fails the golden tests in ``test_reward_golden.py``.

Run from the repo root::

    uv run python tests/envs/locomotion/tools/generate_reward_golden.py

This regenerates every ``*_golden.npz`` under ``tests/envs/locomotion/fixtures/``.
Fixtures are committed alongside the test + this tool in the same PR, so a clean
``make test-all`` checkout has the safety net available.

``run_env_trajectory`` / ``deterministic_actions`` are the single source of truth:
``test_reward_golden.py`` imports them so the test replays the env exactly the way
the fixtures were generated (same seed, same action schedule, same construction).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

# Repo root: tools -> locomotion -> envs -> tests -> <repo>
REPO = Path(__file__).resolve().parents[4]

# (task override, fixture basename). mujoco only — the actual Phase 4 targets.
MUJOCO_TASKS: list[tuple[str, str]] = [
    ("go2_joystick_rough/mujoco", "go2_rough"),
    ("go1_joystick_rough/mujoco", "go1_rough"),
    ("go2w_joystick_rough/mujoco", "go2w_rough"),  # mixed gated+env-bound table
    ("go2_joystick_flat/mujoco", "go2_joystick"),
    ("go1_joystick_flat/mujoco", "go1_joystick"),
]
# motrix dual-backend coverage (M1). Generated best-effort: skipped if the motrix
# backend is unavailable in this environment (logged, not silently dropped).
MOTRIX_TASKS: list[tuple[str, str]] = [
    ("go2_joystick_flat/motrix", "go2_joystick_motrix"),
]


def deterministic_actions(
    num_envs: int, action_dim: int, steps: int, *, seed: int = 0
) -> np.ndarray:
    """Seeded non-zero clipped-random action schedule, shape (steps, num_envs, action_dim).

    Non-zero actions are required to exercise action-dependent rewards
    (``action_rate`` reads ``info["current_actions"]`` / ``info["last_actions"]`` —
    an all-zero schedule leaves it identically 0 and the golden would not protect it).
    """
    rng = np.random.RandomState(seed)
    acts = rng.uniform(-1.0, 1.0, size=(steps, num_envs, action_dim)).astype(np.float32)
    return np.clip(acts, -1.0, 1.0)


def _build_env(task: str, num_envs: int, seed: int = 42):
    """Construct an env via the real training path (Hydra compose + BackendAdapter).

    Mirrors scripts/train_*.py so reward_config injection matches actual training.
    ``seed`` drives env/reset/command/DR sampling (paired with the same seed for the action
    schedule), so the trajectory is reproducible — fixes the seed-API inconsistency where a
    non-default ``seed`` only changed actions, not env construction.
    """
    from hydra import compose, initialize_config_dir

    from unilab.training.backend_adapter import BackendAdapter
    from unilab.training.common import create_env, ensure_registries
    from unilab.training.seed import apply_training_seed

    ensure_registries()
    # Seed before construction — reset/command/DR all sample random numbers.
    apply_training_seed(seed, torch_runtime=True, cuda=False)

    from omegaconf import OmegaConf

    with initialize_config_dir(config_dir=str(REPO / "conf" / "ppo"), version_base="1.3"):
        cfg = compose("config", overrides=[f"task={task}"])

    # Pin terrain spawn determinism on rough (terrain) tasks. terrain_curriculum.seed defaults
    # to None (terrain_spawn.py:59) → np.random.default_rng(None) (terrain_spawn.py:98) draws a
    # RANDOM spawn tile per process. default_rng is a SEPARATE RNG from np.random.seed(), so
    # apply_training_seed() never controls it — the real source of cross-process golden
    # flakiness (NOT "MuJoCo float chaos"; MuJoCo is bit-deterministic for identical inputs).
    # force_add because some rough YAMLs (e.g. go2w) don't materialize env.terrain_curriculum —
    # only the typed cfg default does. Gated on "rough" so flat-task cfg classes that lack the
    # field (e.g. Go1JoystickCfg) stay untouched; flat tasks use BaseSpawnManager (deterministic).
    if "rough" in task:
        OmegaConf.update(cfg, "env.terrain_curriculum.seed", int(seed), force_add=True)

    adapter = BackendAdapter(cfg, root_dir=REPO)
    env_cfg_override = adapter.build_task_env_cfg_override()
    return create_env(cfg, num_envs=num_envs, env_cfg_override=env_cfg_override)


def run_env_trajectory(
    task: str, *, num_envs: int = 4, steps: int = 10, seed: int = 42
) -> tuple[list[np.ndarray], dict[str, Any]]:
    """Run a fixed env trajectory; return (per-step rewards, final reward-component log).

    init_state() rather than reset(env_indices) avoids double-init (reset does not set
    self._state, so the first step would re-init). ``seed`` drives env construction, the
    action schedule, AND the forced terrain_curriculum.seed (see _build_env), so the whole
    trajectory is bit-reproducible across processes — verified: two builds give identical
    spawn levels and 0.0 reward diff over all 10 steps. 10 steps is safe now that the
    spawn-tile randomization is pinned (was the only nondeterminism source).
    """
    env = _build_env(task, num_envs, seed)
    try:
        env.init_state()
        action_dim = int(env.action_space.shape[0])
        actions = deterministic_actions(num_envs, action_dim, steps, seed=seed)
        rewards: list[np.ndarray] = []
        final_log: dict[str, Any] = {}
        for i in range(steps):
            state = env.step(actions[i])
            rewards.append(np.asarray(state.reward, dtype=np.float64).copy())
            # info["log"] carries the most recent logged components forward
            # (dispatch reuses the prior dict on non-log steps), so the last
            # step always exposes the latest reward/* values.
            final_log = dict(state.info.get("log", {}))
    finally:
        env.close()
    return rewards, final_log


def generate(task: str, fixture: str, *, num_envs: int = 4, steps: int = 10) -> int:
    rewards, final_log = run_env_trajectory(task, num_envs=num_envs, steps=steps)
    save: dict[str, np.ndarray] = {f"reward_step{i}": r for i, r in enumerate(rewards)}
    for key, value in final_log.items():
        if key.startswith("reward/"):
            save[f"comp_{key}"] = np.asarray(value, dtype=np.float64)
    out = REPO / "tests" / "envs" / "locomotion" / "fixtures" / f"{fixture}_golden.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **save)
    print(
        f"  saved {out.name}: {len(rewards)} reward steps + "
        f"{sum(k.startswith('comp_') for k in save)} components"
    )
    return len(save)


def main() -> None:
    print("Generating mujoco golden fixtures (Phase 4 refactor targets)...")
    for task, fixture in MUJOCO_TASKS:
        generate(task, fixture)

    # M1 motrix dual-backend golden is deferred to a dedicated lane: motrixsim is an
    # optional extra (default `uv sync` may omit it), so a motrix golden cannot live in
    # the default not-slow `make test-all` net. It needs its own importorskip-guarded
    # test + marker before generation is wired in. MOTRIX_TASKS is kept above for that
    # follow-up; not generated by default to avoid committing an unconsumed fixture.
    print("Skipping motrix golden (deferred to importorskip lane — see MOTRIX_TASKS).")


if __name__ == "__main__":
    main()
