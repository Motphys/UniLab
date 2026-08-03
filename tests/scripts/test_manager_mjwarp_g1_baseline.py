from __future__ import annotations

import json
from pathlib import Path

import pytest
from tooling.acceptance.commands.g1_baseline import ARTIFACT_PATH, PLAN_PATH, _payload, main

pytestmark = [pytest.mark.slow, pytest.mark.local_evidence]


def test_real_artifact_reports_history_verification_status() -> None:
    payload = _payload(PLAN_PATH, ARTIFACT_PATH)

    assert payload["ok"] is True
    assert isinstance(payload["git_history_verified"], bool)


def test_validator_reports_missing_artifact_without_traceback(tmp_path: Path) -> None:
    payload = _payload(PLAN_PATH, tmp_path / "missing.json")

    assert payload["ok"] is False
    assert payload["errors"]


def test_validator_reports_schema_errors_for_incomplete_artifact(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.json"
    artifact.write_text("{}", encoding="utf-8")

    payload = _payload(PLAN_PATH, artifact)

    assert payload["ok"] is False
    assert any("missing key" in error for error in payload["errors"])


def test_json_cli_failure_is_machine_readable(tmp_path: Path, capsys) -> None:
    missing = tmp_path / "missing.json"

    assert main(["--artifact", str(missing), "--json"]) == 1

    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["artifact"] == str(missing)
