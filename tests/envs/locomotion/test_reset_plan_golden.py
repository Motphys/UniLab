"""Golden characterization for DR-provider ``build_reset_plan`` (Phase 4.5 safety net).

Pins the *current* ResetPlan output of the Go1/Go2/Go2W rough providers — qpos/qvel
spawn pose, randomization payload (motor gains, friction, mass/COM, gravity, ...),
info_updates key set, and active DR terms — before Phase 4.5 converges their 90%-clone
``build_reset_plan`` methods. Reward golden cannot see any of this.

Also pins the terrain-curriculum *side effect*: build_reset_plan records each episode's
start pose into ``env._spawn`` (``spawn_episode_start_xyz``/``spawn_has_started``), which
update_on_done later reads. The returned ResetPlan hides this, so without the snapshot a
refactor could drop the recording and the qpos/qvel golden would still pass. Go1/Go2 record
it; Go2W's override does not — the generic numeric compare locks whichever is current.

Imports ``snapshot_reset_plan`` from the generator (single source of truth: same env
build + same provider call), so replay matches fixtures exactly. Not marked ``slow``.
"""

from __future__ import annotations

import functools
import sys
from pathlib import Path

import numpy as np
import pytest

_TOOLS = Path(__file__).resolve().parent / "tools"
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))
from generate_reset_plan_golden import PROVIDERS, snapshot_reset_plan  # noqa: E402

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"
_META_KEYS = ("info_keys", "rand_terms")  # object string arrays, compared as sets


@functools.lru_cache(maxsize=None)
def _snapshot(task: str, module: str, cls: str):
    return snapshot_reset_plan(task, module, cls)


def _fixture_path(fixture: str) -> Path:
    return FIXTURE_DIR / f"reset_{fixture}_golden.npz"


def test_reset_plan_fixtures_present():
    """Hard-fail (not silent skip) if any DR fixture is missing from the repo."""
    missing = [
        f"reset_{name}_golden.npz"
        for _, name, _, _ in PROVIDERS
        if not _fixture_path(name).exists()
    ]
    assert not missing, (
        "Missing DR ResetPlan fixtures: "
        + ", ".join(missing)
        + " — run `uv run python tests/envs/locomotion/tools/generate_reset_plan_golden.py`"
    )


@pytest.mark.parametrize("task,fixture,module,cls", PROVIDERS)
def test_reset_plan_unchanged(task: str, fixture: str, module: str, cls: str):
    path = _fixture_path(fixture)
    if not path.exists():
        pytest.skip(f"fixture missing: {path.name} (see test_reset_plan_fixtures_present)")
    # Fixtures are produced by the sibling generator and committed in-repo; they contain
    # only ndarrays (numeric + unicode str arrays, no pickled objects), so allow_pickle stays off.
    golden = np.load(path)
    snap = _snapshot(task, module, cls)

    # Whole-shape lock: a dropped/added qpos field, info entry, or DR term changes the key set.
    assert set(snap) == set(golden.files), (
        f"{fixture}: ResetPlan key set changed: {set(snap) ^ set(golden.files)}"
    )

    # info_updates keys + active DR terms — exact set match.
    for meta in _META_KEYS:
        if meta in golden.files:
            assert sorted(snap[meta].tolist()) == sorted(golden[meta].tolist()), (
                f"{fixture}: {meta} changed"
            )

    # All numeric arrays fully deterministic now: _build_env forces terrain_curriculum.seed,
    # so env._spawn picks fixed tiles and absolute spawn (qpos[:,0:3]) is reproducible too
    # (previously excluded — the unseeded default_rng(None) made tile choice random per process).
    # This now locks the full spawn pose (qpos incl. xy/z), qvel, info_updates, and DR payload.
    for key in golden.files:
        if key in _META_KEYS:
            continue
        np.testing.assert_allclose(
            np.asarray(snap[key], dtype=np.float64),
            golden[key],
            rtol=1e-6,
            atol=1e-7,
            err_msg=f"{fixture} ResetPlan mismatch: {key}",
        )
