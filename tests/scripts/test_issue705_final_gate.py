"""Release-level freshness and fault tests for Issue #705."""

from __future__ import annotations

import os
import subprocess
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

from omegaconf import OmegaConf

from unilab.tools.issue705_final_gate import (
    ARTIFACT_PATH,
    PLAN_PATH,
    FinalGatePlanError,
    _phase7_immutable_semantics,
    _source_snapshot,
    command_evidence_errors,
    load_final_gate_evidence,
    load_final_gate_plan,
    validate_final_gate_evidence,
    validate_final_head,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _plan():
    return load_final_gate_plan(REPO_ROOT / PLAN_PATH)


def _valid_command_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for command in _plan().commands:
        for repetition in range(1, command.repetitions + 1):
            rows.append(
                {
                    "name": f"{command.name}#{repetition}",
                    "series": command.name,
                    "claim_id": command.claim_id,
                    "lane": command.lane,
                    "repetition": repetition,
                    "argv": list(command.argv),
                    "required_test_id": command.required_test_id,
                    "exit_code": 0,
                    "duration_sec": 0.1,
                    "pytest": {
                        "passed": 1,
                        "skipped": 0,
                        "xfailed": 0,
                        "xpassed": 0,
                        "deselected": 0,
                    },
                    "stdout": f"{command.required_test_id} PASSED\n1 passed",
                    "stderr": "",
                }
            )
    return rows


def test_final_head_has_fresh_all_phase_evidence() -> None:
    """This exact node is the Phase 7 final-regression acceptance oracle."""

    report = validate_final_head(REPO_ROOT, _plan(), require_promotion=False)

    assert report.ok, report.errors
    assert len(report.components) == 17
    assert all(component.ok for component in report.components)
    claim_component = next(item for item in report.components if item.name == "claim_inventory")
    assert claim_component.details["targets"] == 0


def test_final_plan_is_strict_and_keeps_rss_diagnostic_only(tmp_path: Path) -> None:
    plan = _plan()
    assert plan.rss_mode == "diagnostic_only"
    assert {command.lane for command in plan.commands} == {"A", "B", "C", "D"}
    assert {entry.phase for entry in plan.phase_evidence} == set(range(1, 7))

    raw = OmegaConf.load(REPO_ROOT / PLAN_PATH)
    raw.rss_policy.mode = "absolute_blocker"
    bad_path = tmp_path / "bad_final_plan.yaml"
    OmegaConf.save(raw, bad_path)
    try:
        load_final_gate_plan(bad_path)
    except FinalGatePlanError as exc:
        assert "RSS is diagnostic-only" in str(exc)
    else:
        raise AssertionError("RSS blocker fault was accepted")


def test_final_command_matrix_rejects_skip_deselect_tamper_and_missing_repetition() -> None:
    plan = _plan()
    rows = _valid_command_rows()
    assert command_evidence_errors(rows, plan=plan) == ()

    skipped = deepcopy(rows)
    skipped[0]["pytest"]["skipped"] = 1  # type: ignore[index]
    assert any(
        "skipped: expected 0" in error for error in command_evidence_errors(skipped, plan=plan)
    )

    deselected = deepcopy(rows)
    deselected[0]["pytest"]["deselected"] = 1  # type: ignore[index]
    assert any(
        "deselected: expected 0" in error
        for error in command_evidence_errors(deselected, plan=plan)
    )

    tampered = deepcopy(rows)
    tampered[0]["argv"] = ["uv", "run", "pytest", "not-the-frozen-node"]
    assert any("argv: differs" in error for error in command_evidence_errors(tampered, plan=plan))

    missing = rows[:-1]
    assert any(
        "expected repetitions" in error for error in command_evidence_errors(missing, plan=plan)
    )


def test_malformed_final_artifact_fails_closed(monkeypatch) -> None:
    def _unexpected_head_validation(*args, **kwargs):
        raise AssertionError("malformed artifacts must fail before current-head validation")

    monkeypatch.setattr(
        "unilab.tools.issue705_final_gate.validate_final_head",
        _unexpected_head_validation,
    )
    malformed = {
        "plan": [],
        "source": [],
        "inputs": [],
        "head_validation": [],
        "commands": [None],
        "diagnostics": [],
        "summary": [],
    }

    errors = validate_final_gate_evidence(
        malformed,
        root=REPO_ROOT,
        plan=_plan(),
        require_promotion=False,
    )

    assert errors
    assert "artifact: keys do not exactly match the v1 schema" in errors


def test_committed_final_artifact_is_fresh_and_promoted() -> None:
    artifact = load_final_gate_evidence(REPO_ROOT / ARTIFACT_PATH)
    errors = validate_final_gate_evidence(
        artifact,
        root=REPO_ROOT,
        plan=_plan(),
        require_promotion=True,
    )

    assert errors == ()


def test_final_artifact_rejects_source_and_input_hash_faults() -> None:
    artifact = load_final_gate_evidence(REPO_ROOT / ARTIFACT_PATH)
    plan = _plan()

    stale_source = deepcopy(artifact)
    stale_source["source"]["tree_sha256"] = "sha256:" + "0" * 64
    assert any(
        "source.tree_sha256" in error
        for error in validate_final_gate_evidence(
            stale_source,
            root=REPO_ROOT,
            plan=plan,
            require_promotion=True,
        )
    )

    stale_modes = deepcopy(artifact)
    stale_modes["source"]["tracked_modes_sha256"] = "sha256:" + "0" * 64
    assert any(
        "source.tracked_modes_sha256" in error
        for error in validate_final_gate_evidence(
            stale_modes,
            root=REPO_ROOT,
            plan=plan,
            require_promotion=True,
        )
    )

    stale_input = deepcopy(artifact)
    key = PLAN_PATH.as_posix()
    stale_input["inputs"]["files"][key] = "sha256:" + "0" * 64
    assert any(
        f"inputs.files.{key}" in error
        for error in validate_final_gate_evidence(
            stale_input,
            root=REPO_ROOT,
            plan=plan,
            require_promotion=True,
        )
    )


def test_source_snapshot_rejects_executable_bit_changes(tmp_path: Path) -> None:
    subprocess.run(("git", "init", "-q"), cwd=tmp_path, check=True)
    subprocess.run(("git", "config", "user.name", "Issue 705 Test"), cwd=tmp_path, check=True)
    subprocess.run(
        ("git", "config", "user.email", "issue705@example.invalid"),
        cwd=tmp_path,
        check=True,
    )
    tracked = tmp_path / "tracked.sh"
    tracked.write_text("echo final-gate\n", encoding="utf-8")
    tracked.chmod(0o644)
    subprocess.run(("git", "add", "tracked.sh"), cwd=tmp_path, check=True)
    subprocess.run(("git", "commit", "-q", "-m", "fixture"), cwd=tmp_path, check=True)
    commit = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    plan = replace(_plan(), source_exclusions=())

    committed = _source_snapshot(tmp_path, plan, commit=commit)
    before = _source_snapshot(tmp_path, plan)
    assert before == committed

    os.chmod(tracked, 0o755)
    after = _source_snapshot(tmp_path, plan)
    assert tracked.read_text(encoding="utf-8") == "echo final-gate\n"
    assert after["tracked_paths_sha256"] == before["tracked_paths_sha256"]
    assert after["tracked_modes_sha256"] != before["tracked_modes_sha256"]
    assert after["tree_sha256"] != before["tree_sha256"]


def test_phase7_semantic_snapshot_allows_only_promotion_fields() -> None:
    raw = OmegaConf.to_container(
        OmegaConf.load(REPO_ROOT / "tests/acceptance/issue_705/manifests/phase_7.yaml"),
        resolve=False,
    )
    assert isinstance(raw, dict)
    baseline = _phase7_immutable_semantics(raw)

    promotion = deepcopy(raw)
    promotion["claims"][0]["status"] = "planned"
    promotion["claims"][0]["evidence"] = {
        "result": "PENDING",
        "artifact_refs": [],
    }
    assert _phase7_immutable_semantics(promotion) == baseline

    for field, value in (
        ("owner", "wrong-owner"),
        ("environment", {"hardware": "unreviewed"}),
        ("acceptance", {"thresholds": {"stale_claims": 99}}),
        ("invalidation", {"paths": []}),
    ):
        changed = deepcopy(raw)
        changed["claims"][0][field] = value
        assert _phase7_immutable_semantics(changed) != baseline


def test_final_artifact_rejects_phase7_semantic_snapshot_tamper() -> None:
    artifact = load_final_gate_evidence(REPO_ROOT / ARTIFACT_PATH)
    artifact["phase7_manifest"]["snapshot"]["claims"][0]["owner"] = "wrong-owner"

    errors = validate_final_gate_evidence(
        artifact,
        root=REPO_ROOT,
        plan=_plan(),
        require_promotion=True,
    )

    assert any("phase7_manifest" in error for error in errors)
