from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import pytest
from omegaconf import OmegaConf

from unilab.tools.g1_baseline_provenance import sha256_file
from unilab.tools.issue705_thresholds import (
    AMENDMENT_MANIFEST_PATH,
    FreezeReceipt,
    ThresholdValidationError,
    load_threshold_amendment,
    load_threshold_manifest,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_MANIFEST_PATH = REPO_ROOT / "tests/acceptance/issue_705/g1_threshold_manifest.yaml"
AMENDMENT_PATH = REPO_ROOT / AMENDMENT_MANIFEST_PATH


@pytest.fixture(scope="module")
def base_manifest():
    return load_threshold_manifest(BASE_MANIFEST_PATH, repo_root=REPO_ROOT)


@pytest.fixture(scope="module")
def base_receipt(base_manifest):
    return FreezeReceipt(
        source_path=Path("<memory>"),
        data={
            "threshold_set_id": base_manifest.data["threshold_set_id"],
            "manifest_sha256": sha256_file(BASE_MANIFEST_PATH),
            "freeze_commit": "a2419b342b8663998b2e29cf20a4dce49b3127f5",
        },
    )


def _write_mutated_amendment(tmp_path: Path, mutate: Callable[[dict[str, Any]], None]) -> Path:
    raw = OmegaConf.to_container(OmegaConf.load(AMENDMENT_PATH), resolve=False)
    assert isinstance(raw, dict)
    mutate(raw)
    destination = tmp_path / "amendment.yaml"
    OmegaConf.save(OmegaConf.create(raw), destination)
    return destination


def test_amendment_is_narrow_and_preserves_base(base_manifest, base_receipt) -> None:
    amendment = load_threshold_amendment(
        AMENDMENT_PATH,
        base_manifest=base_manifest,
        base_receipt=base_receipt,
        repo_root=REPO_ROOT,
    )

    assert amendment.amendment_id == "g1-phase5-ppo-rss-ratio-v1"
    assert amendment.host_memory_ratio_max == pytest.approx(1.26)
    assert base_manifest.gates["memory"]["host_preferred_metric_ratio_max"] == pytest.approx(1.25)
    assert amendment.data["scope"]["artifact_kind"] == ("issue705-mjwarp-device-ppo-benchmark-v1")
    assert amendment.data["scope"]["lanes"] == ["throughput", "behavior"]


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda raw: raw.update({"unexpected": True}), "unknown key `unexpected`"),
        (lambda raw: raw["change"].update({"amended_value": 1.27}), "frozen amendment value"),
        (lambda raw: raw["change"].update({"previous_value": 1.24}), "previous_value"),
        (lambda raw: raw["scope"].update({"profile": "host_fused"}), "scope.profile"),
        (lambda raw: raw["scope"].pop("lanes"), "missing key `lanes`"),
        (
            lambda raw: raw["governance"].update({"no_protocol_change": False}),
            "no_protocol_change",
        ),
        (
            lambda raw: raw["base_threshold"].update({"manifest_sha256": "sha256:" + "0" * 64}),
            "base_threshold.manifest_sha256",
        ),
    ],
)
def test_amendment_rejects_scope_value_and_provenance_tampering(
    tmp_path: Path,
    base_manifest,
    base_receipt,
    mutate: Callable[[dict[str, Any]], None],
    message: str,
) -> None:
    path = _write_mutated_amendment(tmp_path, mutate)
    with pytest.raises(ThresholdValidationError, match=message):
        load_threshold_amendment(
            path,
            base_manifest=base_manifest,
            base_receipt=base_receipt,
            repo_root=REPO_ROOT,
        )


def test_amendment_requires_the_declared_adr(tmp_path: Path, base_manifest, base_receipt) -> None:
    path = _write_mutated_amendment(
        tmp_path,
        lambda raw: raw["governance"].update({"adr_path": "docs/missing.md"}),
    )
    with pytest.raises(ThresholdValidationError, match="governance.adr_path"):
        load_threshold_amendment(
            path,
            base_manifest=base_manifest,
            base_receipt=base_receipt,
            repo_root=REPO_ROOT,
        )
