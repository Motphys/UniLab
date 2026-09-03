"""Drift guard for the MicroDuck × microduck_rl alignment contract (issue #1453).

For each MicroDuck locomotion task (ppo tree, mjwarp owner) the declared
status of every contract entry must agree with its evaluated value:

- ``match`` entries must keep matching the upstream target (drift guard).
- ``gap`` entries must keep NOT matching (the gap set only shrinks when a
  child issue lands an alignment change and flips the entry in
  ``alignment_contract.ENTRIES`` in the same commit — this test is the
  explicit checklist enforcing that).
- ``note`` entries are informational and not judged.
"""

from __future__ import annotations

import pytest

from unilab.tasks.locomotion.microduck.alignment_contract import (
    ENTRIES,
    MICRODUCK_TASKS,
    evaluate_task,
)


def test_contract_statuses_are_well_formed() -> None:
    for entry in ENTRIES:
        assert entry.status in ("match", "gap", "note"), entry.name
        assert entry.category in ("physics", "mdp", "infra"), entry.name
        if entry.status == "note":
            assert entry.target is None or entry.name == "infra.mujoco_warp_version", entry.name
        for task in entry.tasks:
            assert task in MICRODUCK_TASKS, entry.name


@pytest.mark.parametrize("task", MICRODUCK_TASKS)
def test_declared_status_matches_evaluated_value(task: str) -> None:
    results = evaluate_task(task)
    assert results, f"no contract entries apply to {task}"
    stale = [r for r in results if r.matches is not None and r.matches != (r.status == "match")]
    assert not stale, (
        "Contract entries whose declared status is stale; update "
        "src/unilab/tasks/locomotion/microduck/alignment_contract.py: "
        + ", ".join(f"{r.name} (declared={r.status}, matches_target={r.matches})" for r in stale)
    )


@pytest.mark.parametrize("task", MICRODUCK_TASKS)
def test_match_entries_keep_matching(task: str) -> None:
    results = [r for r in evaluate_task(task) if r.status == "match"]
    assert results
    drifted = [r.name for r in results if r.matches is not True]
    assert not drifted, f"aligned entries drifted from the upstream target: {drifted}"
