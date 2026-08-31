from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "audit_sim2sim_contracts.py"


def _load_audit_module():
    spec = importlib.util.spec_from_file_location("audit_sim2sim_contracts", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_discover_preserves_standard_task_layout() -> None:
    audit = _load_audit_module()

    discovered = audit._discover("ppo")

    assert {"mujoco", "motrix", "genesis"}.issubset(discovered["g1_walk_flat"])


def test_discover_offpolicy_trees_group_by_task() -> None:
    audit = _load_audit_module()

    sac = audit._discover("sac")
    flashsac = audit._discover("flashsac")
    td3 = audit._discover("td3")

    assert {"mujoco", "motrix", "mjwarp", "isaacgym", "genesis"}.issubset(sac["g1_walk_flat"])
    assert {"mujoco", "motrix", "mjwarp"}.issubset(flashsac["g1_walk_flat"])
    assert td3["g1_walk_flat"] == ["mujoco"]


@pytest.mark.parametrize(
    ("tree", "task_variant", "expected_algo"),
    [
        ("sac", "g1_walk_flat/mujoco", "sac"),
        ("td3", "g1_walk_flat/mujoco", "td3"),
        ("flashsac", "g1_walk_flat/mujoco", "flashsac"),
    ],
)
def test_compose_resolves_tree_algo(tree: str, task_variant: str, expected_algo: str) -> None:
    audit = _load_audit_module()

    cfg = audit._compose(tree, task_variant)

    assert cfg.algo.algo == expected_algo


def test_audit_tree_compares_one_offpolicy_backend_pair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audit = _load_audit_module()
    monkeypatch.setattr(
        audit,
        "_discover",
        lambda tree: {"g1_walk_flat": ["motrix", "mujoco"]},
    )

    rows = audit.audit_tree("flashsac")

    assert len(rows) == 1
    assert rows[0]["task"] == "g1_walk_flat"
    assert rows[0]["backends"] == ["motrix", "mujoco"]
    assert rows[0]["errors"] == {}


def test_audit_tree_rejects_empty_discovery(monkeypatch: pytest.MonkeyPatch) -> None:
    audit = _load_audit_module()
    monkeypatch.setattr(audit, "_discover", lambda tree: {})

    with pytest.raises(ValueError, match="No task owner configs discovered"):
        audit.audit_tree("sac")
