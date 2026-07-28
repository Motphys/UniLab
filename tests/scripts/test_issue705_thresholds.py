from __future__ import annotations

import json
from pathlib import Path

from omegaconf import OmegaConf
from scripts import validate_issue705_thresholds


def test_default_cli_validates_manifest_and_freeze_receipt(capsys) -> None:
    exit_code = validate_issue705_thresholds.main(["--json"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["result"] == "PASS"
    assert payload["freeze_commit"] == "a2419b342b8663998b2e29cf20a4dce49b3127f5"


def test_manifest_only_cli_passes_and_reports_frozen_matrix(capsys) -> None:
    exit_code = validate_issue705_thresholds.main(["--manifest-only", "--json"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["result"] == "PASS"
    assert payload["freeze_commit"] is None
    assert payload["env_batches"] == [128, 1024, 4096]
    assert payload["dr_densities"] == [0.01, 0.1, 1.0]
    assert payload["ppo_seeds"] == [0, 1, 2, 3, 4]


def test_manifest_only_cli_fails_on_threshold_relaxation(tmp_path: Path, capsys) -> None:
    raw = OmegaConf.load(
        validate_issue705_thresholds.REPO_ROOT
        / validate_issue705_thresholds.THRESHOLD_MANIFEST_PATH
    )
    raw.gates.performance.p50_latency_ratio_max = 1.06
    path = tmp_path / "thresholds.yaml"
    OmegaConf.save(raw, path)

    exit_code = validate_issue705_thresholds.main(["--manifest", str(path), "--manifest-only"])

    assert exit_code == 1
    output = capsys.readouterr().out
    assert "FAIL" in output
    assert "frozen value is 1.05" in output
