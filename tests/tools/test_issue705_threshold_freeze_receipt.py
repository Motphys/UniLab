from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Callable

import pytest
from omegaconf import OmegaConf

from unilab.tools import issue705_thresholds
from unilab.tools.g1_baseline_provenance import sha256_file
from unilab.tools.issue705_thresholds import (
    ThresholdManifest,
    ThresholdValidationError,
    load_freeze_receipt,
    load_threshold_manifest,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "tests/acceptance/issue_705/g1_threshold_manifest.yaml"
RECEIPT_PATH = REPO_ROOT / "tests/acceptance/issue_705/g1_threshold_freeze_receipt.yaml"


@pytest.fixture(scope="module")
def manifest() -> ThresholdManifest:
    return load_threshold_manifest(MANIFEST_PATH, repo_root=REPO_ROOT)


def _write_mutated_receipt(tmp_path: Path, mutate: Callable[[dict[str, Any]], None]) -> Path:
    raw = OmegaConf.to_container(OmegaConf.load(RECEIPT_PATH), resolve=False)
    assert isinstance(raw, dict)
    mutate(raw)
    destination = tmp_path / "receipt.yaml"
    OmegaConf.save(OmegaConf.create(raw), destination)
    return destination


def test_real_receipt_reports_whether_frozen_history_is_available(
    manifest: ThresholdManifest,
) -> None:
    receipt = load_freeze_receipt(
        RECEIPT_PATH,
        manifest=manifest,
        repo_root=REPO_ROOT,
    )

    assert receipt.freeze_commit == "a2419b342b8663998b2e29cf20a4dce49b3127f5"
    assert receipt.data["manifest_git_blob"] == "f622c724eec1368cf4c28ce9a243a0fcac16d09d"
    assert receipt.data["manifest_sha256"] == sha256_file(MANIFEST_PATH)
    commit_available = (
        subprocess.run(
            ["git", "cat-file", "-e", f"{receipt.freeze_commit}^{{commit}}"],
            cwd=REPO_ROOT,
            check=False,
        ).returncode
        == 0
    )
    assert receipt.git_history_verified is commit_available


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda raw: raw.update({"extra": True}), "unknown key `extra`"),
        (lambda raw: raw.pop("freeze_commit"), "missing key `freeze_commit`"),
        (
            lambda raw: raw.update({"manifest_sha256": "sha256:" + "0" * 64}),
            "expected current",
        ),
        (
            lambda raw: raw.update({"baseline_artifact_sha256": "sha256:" + "0" * 64}),
            "does not match threshold manifest",
        ),
        (lambda raw: raw.update({"freeze_commit": "short"}), "full lowercase commit SHA"),
    ],
)
def test_receipt_rejects_schema_identity_and_hash_tampering(
    tmp_path: Path,
    manifest: ThresholdManifest,
    mutate: Callable[[dict[str, Any]], None],
    message: str,
) -> None:
    path = _write_mutated_receipt(tmp_path, mutate)
    with pytest.raises(ThresholdValidationError, match=message):
        load_freeze_receipt(
            path,
            manifest=manifest,
            repo_root=REPO_ROOT,
            verify_git=False,
        )


def test_receipt_rejects_manifest_changed_after_freeze(
    tmp_path: Path, manifest: ThresholdManifest
) -> None:
    changed_manifest_path = tmp_path / "g1_threshold_manifest.yaml"
    changed_manifest_path.write_bytes(MANIFEST_PATH.read_bytes() + b"\n# post-hoc change\n")
    changed_manifest = ThresholdManifest(
        source_path=changed_manifest_path,
        data=manifest.data,
    )
    receipt_path = _write_mutated_receipt(
        tmp_path,
        lambda raw: raw.update({"manifest_sha256": sha256_file(changed_manifest_path)}),
    )

    with pytest.raises(ThresholdValidationError, match="manifest bytes differ"):
        load_freeze_receipt(
            receipt_path,
            manifest=changed_manifest,
            repo_root=REPO_ROOT,
        )


def test_receipt_rejects_wrong_git_blob(tmp_path: Path, manifest: ThresholdManifest) -> None:
    path = _write_mutated_receipt(
        tmp_path,
        lambda raw: raw.update({"manifest_git_blob": "0" * 40}),
    )

    with pytest.raises(ThresholdValidationError, match="commit contains"):
        load_freeze_receipt(path, manifest=manifest, repo_root=REPO_ROOT)


def test_receipt_rejects_unresolvable_freeze_commit(
    tmp_path: Path, manifest: ThresholdManifest
) -> None:
    path = _write_mutated_receipt(
        tmp_path,
        lambda raw: raw.update({"freeze_commit": "c" * 40}),
    )

    with pytest.raises(ThresholdValidationError, match="cannot verify Git object"):
        load_freeze_receipt(path, manifest=manifest, repo_root=REPO_ROOT)


def test_shallow_checkout_uses_hash_receipt_without_false_ancestry_failure(
    manifest: ThresholdManifest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_git(
        _repo_root: Path,
        args: list[str] | tuple[str, ...],
        *,
        check: bool = True,
    ) -> subprocess.CompletedProcess[bytes]:
        del check
        if args[0] == "cat-file":
            return subprocess.CompletedProcess(args, 1, b"", b"missing")
        if list(args) == ["rev-parse", "--is-shallow-repository"]:
            return subprocess.CompletedProcess(args, 0, b"true\n", b"")
        raise AssertionError(f"unexpected git command: {args}")

    monkeypatch.setattr(issue705_thresholds, "_git", fake_git)

    receipt = load_freeze_receipt(
        RECEIPT_PATH,
        manifest=manifest,
        repo_root=REPO_ROOT,
    )

    assert receipt.git_history_verified is False
