"""Golden regression tests for locomotion reward numerical equivalence.

These pin reward values *before* Phase 4 refactoring. Any change to reward values
(even 1e-7) fails here, so reward pure-function extraction / joystick base class /
DR provider convergence cannot silently drift algorithm behavior.

Fixtures live in ``fixtures/*_golden.npz`` and are produced by
``tools/generate_reward_golden.py``. This test imports the generator's
``run_env_trajectory`` so the replay matches fixture generation exactly (same seed,
same non-zero action schedule, same env construction) — single source of truth, no drift.

Not marked ``slow`` — this is the safety net, so it must run inside ``make test-all``
(4 envs x 10 steps is light enough). Fixtures + tool + this test ship in one PR.
"""

from __future__ import annotations

import functools
import sys
from pathlib import Path

import numpy as np
import pytest

# Import the generator's trajectory runner (single source of truth for env build +
# action schedule). Insert the sibling tools/ dir rather than relying on a package chain.
_TOOLS = Path(__file__).resolve().parent / "tools"
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))
from generate_reward_golden import run_env_trajectory  # noqa: E402

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"

# (task override, fixture basename) — mirrors MUJOCO_TASKS in the generator.
TASKS: list[tuple[str, str]] = [
    ("go2_joystick_rough/mujoco", "go2_rough"),
    ("go1_joystick_rough/mujoco", "go1_rough"),
    ("go2w_joystick_rough/mujoco", "go2w_rough"),
    ("go2_joystick_flat/mujoco", "go2_joystick"),
    ("go1_joystick_flat/mujoco", "go1_joystick"),
]


@functools.lru_cache(maxsize=None)
def _trajectory(task: str):
    """Run + cache one env trajectory per task (avoids rebuilding MuJoCo env per test)."""
    return run_env_trajectory(task)


def _fixture_path(fixture: str) -> Path:
    return FIXTURE_DIR / f"{fixture}_golden.npz"


def test_golden_fixtures_present():
    """Hard-fail (not silent skip) if any expected fixture is missing from the repo.

    The golden tests below skip when a fixture is absent so a partial checkout gives a
    clear reason rather than a confusing error; this test makes 'fixtures must be
    committed with the test' explicit so the safety net cannot silently degrade to no-op.
    """
    missing = [f"{name}_golden.npz" for _, name in TASKS if not _fixture_path(name).exists()]
    assert not missing, (
        "Missing golden fixtures: "
        + ", ".join(missing)
        + " — run `uv run python tests/envs/locomotion/tools/generate_reward_golden.py`"
        + " and commit fixtures alongside this test."
    )


@pytest.mark.parametrize("task,fixture", TASKS)
def test_reward_trajectory(task: str, fixture: str):
    """Full step0..9 reward trajectory matches golden (not just the last step)."""
    path = _fixture_path(fixture)
    if not path.exists():
        pytest.skip(f"fixture missing: {path.name} (see test_golden_fixtures_present)")
    golden = np.load(path)
    rewards, _ = _trajectory(task)
    n = sum(1 for k in golden.files if k.startswith("reward_step"))
    assert len(rewards) == n, f"{fixture}: trajectory length {len(rewards)} != golden {n}"
    for i, reward in enumerate(rewards):
        np.testing.assert_allclose(
            reward,
            golden[f"reward_step{i}"],
            rtol=1e-6,
            atol=1e-7,
            err_msg=f"{fixture} reward mismatch at step {i}",
        )


@pytest.mark.parametrize("task,fixture", TASKS)
def test_reward_components(task: str, fixture: str):
    """Per-component reward log: key set unchanged + each component value matches.

    Catches refactor errors that total reward can hide — a dropped reward key, a renamed
    log entry, or one term's scale/gating wrong but offset by another in the sum.
    """
    path = _fixture_path(fixture)
    if not path.exists():
        pytest.skip(f"fixture missing: {path.name} (see test_golden_fixtures_present)")
    golden = np.load(path)
    _, final_log = _trajectory(task)

    expected_keys = {k[len("comp_") :] for k in golden.files if k.startswith("comp_reward/")}
    assert expected_keys, f"{fixture}: no reward components captured in fixture"
    # Exact set (a.md/b.md): catch ADDED reward keys too, not just dropped — a newly logged
    # reward/* component is also a behavior change Phase 4 must not introduce silently.
    actual_keys = {k for k in final_log if k.startswith("reward/")}
    assert actual_keys == expected_keys, (
        f"{fixture}: reward component key set changed: "
        f"added={sorted(actual_keys - expected_keys)} dropped={sorted(expected_keys - actual_keys)}"
    )

    for key in golden.files:
        if not key.startswith("comp_reward/"):
            continue
        name = key[len("comp_") :]
        np.testing.assert_allclose(
            float(final_log[name]),
            float(golden[key]),
            rtol=1e-5,
            atol=1e-6,
            err_msg=f"{fixture} component mismatch: {name}",
        )
