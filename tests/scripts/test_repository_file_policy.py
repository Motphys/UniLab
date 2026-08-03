from __future__ import annotations

from pathlib import Path

from tooling.check_repository_files import repository_file_errors


def test_repository_file_policy_accepts_small_regular_file(tmp_path: Path) -> None:
    (tmp_path / "summary.json").write_text("{}", encoding="utf-8")

    assert repository_file_errors(tmp_path, [Path("summary.json")], max_bytes=32) == ()


def test_repository_file_policy_rejects_oversized_file(tmp_path: Path) -> None:
    (tmp_path / "trace.json").write_bytes(b"x" * 33)

    errors = repository_file_errors(tmp_path, [Path("trace.json")], max_bytes=32)

    assert any("committed-file limit" in error for error in errors)


def test_repository_file_policy_rejects_lfs_pointer_and_rules(tmp_path: Path) -> None:
    (tmp_path / "trace.bin").write_text(
        "version https://git-lfs.github.com/spec/v1\noid sha256:" + "0" * 64 + "\nsize 1\n",
        encoding="utf-8",
    )
    (tmp_path / ".gitattributes").write_text("*.bin filter=lfs diff=lfs\n", encoding="utf-8")

    errors = repository_file_errors(
        tmp_path,
        [Path("trace.bin"), Path(".gitattributes")],
    )

    assert any("pointer files are forbidden" in error for error in errors)
    assert any("rules are forbidden" in error for error in errors)
