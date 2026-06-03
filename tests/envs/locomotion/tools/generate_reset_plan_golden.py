"""Generate golden ResetPlan fixtures for DR-provider characterization (Phase 4.5).

Reward golden cannot prove DR ``build_reset_plan`` behavior — qpos/qvel spawn pose,
randomization payload (motor gains, friction, mass/COM, gravity, ...), info_updates,
and subset ``env_ids`` handling are invisible to reward values. Phase 4.5 converges the
Go1/Go2/Go2W rough providers' ``build_reset_plan`` (90% clones); these fixtures pin its
*current* output so the convergence cannot silently change reset behavior.

Deterministic: ``_build_env`` seeds (``apply_training_seed(42)``) then constructs the env
the same way every run, so the RNG state at ``build_reset_plan`` time is reproducible.

Run from the repo root::

    uv run python tests/envs/locomotion/tools/generate_reset_plan_golden.py
"""

from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np

# Reuse the single source of truth for env construction (same tools/ dir on sys.path).
from generate_reward_golden import REPO, _build_env  # noqa: E402

# (task, fixture basename, provider module, provider class). The 90%-clone rough providers.
PROVIDERS: list[tuple[str, str, str, str]] = [
    (
        "go2_joystick_rough/mujoco",
        "go2_rough",
        "unilab.envs.locomotion.go2.rough",
        "Go2JoystickRoughDomainRandomizationProvider",
    ),
    (
        "go1_joystick_rough/mujoco",
        "go1_rough",
        "unilab.envs.locomotion.go1.rough",
        "Go1JoystickRoughDomainRandomizationProvider",
    ),
    (
        "go2w_joystick_rough/mujoco",
        "go2w_rough",
        "unilab.envs.locomotion.go2w.rough",
        "Go2WJoystickRoughDomainRandomizationProvider",
    ),
]

# Subset reset — exercises env_ids handling, not just full reset.
ENV_IDS = np.array([0, 2], dtype=np.int64)

# ResetRandomizationPayload fields (dr/types.py:91), all np.ndarray | None.
RAND_FIELDS = [
    "base_mass_delta",
    "base_com_offset",
    "gravity",
    "body_iquat",
    "body_inertia",
    "body_ipos",
    "body_mass",
    "dof_armature",
    "geom_friction",
    "kp",
    "kd",
]


def snapshot_reset_plan(
    task: str, provider_module: str, provider_cls: str
) -> dict[str, np.ndarray]:
    """Build env, call provider.build_reset_plan(env, ENV_IDS), flatten ResetPlan to arrays."""
    provider_type = getattr(importlib.import_module(provider_module), provider_cls)
    env = _build_env(task, 4)
    try:
        env.init_state()
        plan = provider_type().build_reset_plan(env, ENV_IDS)
        save: dict[str, np.ndarray] = {
            "env_ids": np.asarray(plan.env_ids),
            "qpos": np.asarray(plan.qpos, dtype=np.float64),
            "qvel": np.asarray(plan.qvel, dtype=np.float64),
            # info_updates key set (lock against dropped/renamed keys).
            # Unicode str array (not dtype=object) so fixtures load without allow_pickle.
            "info_keys": np.array(sorted(plan.info_updates)),
        }
        for key, value in plan.info_updates.items():
            if isinstance(value, np.ndarray):
                save[f"info_{key}"] = np.asarray(value, dtype=np.float64)
        if plan.randomization is not None:
            # active DR terms (lock which randomizations fire)
            save["rand_terms"] = np.array(sorted(plan.randomization.requested_terms()))
            for field in RAND_FIELDS:
                value = getattr(plan.randomization, field)
                if value is not None:
                    save[f"rand_{field}"] = np.asarray(value, dtype=np.float64)
        # Terrain-curriculum side effect: build_reset_plan mutates env._spawn in-place
        # (record_episode_start -> _episode_start_xyz/_has_started, terrain_spawn.py:185-187),
        # later read by update_on_done for level progression. The returned ResetPlan does NOT
        # show this, so a refactor could drop the call and the qpos/qvel golden would still pass.
        # Snapshot the post-call state for ENV_IDS so the side effect is locked too. Go1/Go2 rough
        # record it (has_started True); Go2W's override does NOT — both truths are pinned per-fixture.
        spawn = getattr(env, "_spawn", None)
        start_xyz = getattr(spawn, "_episode_start_xyz", None)
        has_started = getattr(spawn, "_has_started", None)
        if start_xyz is not None and has_started is not None:
            save["spawn_episode_start_xyz"] = np.asarray(start_xyz, dtype=np.float64)[ENV_IDS]
            save["spawn_has_started"] = np.asarray(has_started, dtype=np.float64)[ENV_IDS]
    finally:
        env.close()
    return save


def generate(task: str, fixture: str, provider_module: str, provider_cls: str) -> None:
    save = snapshot_reset_plan(task, provider_module, provider_cls)
    out = REPO / "tests" / "envs" / "locomotion" / "fixtures" / f"reset_{fixture}_golden.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **save)
    n_rand = sum(k.startswith("rand_") and k != "rand_terms" for k in save)
    print(
        f"  saved {out.name}: qpos{save['qpos'].shape} qvel{save['qvel'].shape} "
        f"{len(save['info_keys'])} info keys + {n_rand} rand fields"
    )


def main() -> None:
    print("Generating DR ResetPlan golden fixtures (Phase 4.5 targets)...")
    for task, fixture, module, cls in PROVIDERS:
        generate(task, fixture, module, cls)


if __name__ == "__main__":
    main()
