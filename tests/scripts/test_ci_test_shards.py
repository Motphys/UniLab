from __future__ import annotations

from pathlib import Path

from scripts.audit_ci_test_shards import CiTestShardAuditResult, audit_ci_test_shards


def _audit_fixture(root: Path) -> CiTestShardAuditResult:
    return audit_ci_test_shards(root, expected_local_evidence_test_ids=frozenset())


def _write_fixture(root: Path, *, paths_a: str, paths_b: str, selection: str) -> None:
    workflow = root / ".github" / "workflows" / "ci.yml"
    workflow.parent.mkdir(parents=True)
    workflow.write_text(
        f"""
jobs:
  benchmark-smoke:
    runs-on: ubuntu-slim
    steps:
      - name: Benchmark entrypoint smoke test
        run: uv run python benchmark/smoke_test.py
  test-shard:
    runs-on: ubuntu-slim
    strategy:
      fail-fast: false
      matrix:
        include:
          - shard: a
            paths: {paths_a}
          - shard: b
            paths: {paths_b}
    steps:
      - uses: actions/checkout@v6.0.2
        with:
          fetch-depth: 0
      - name: Test with coverage
        env:
          COVERAGE_FILE: coverage.${{{{ matrix.shard }}}}
        run: >-
          uv run pytest ${{{{ matrix.paths }}}} -m "{selection}"
          --cov=src/unilab --cov-report= --cov-fail-under=0
      - name: Upload coverage data
        uses: actions/upload-artifact@v4
        with:
          name: coverage-${{{{ matrix.shard }}}}
          path: coverage.${{{{ matrix.shard }}}}
          if-no-files-found: error
  test:
    name: test (ubuntu-slim)
    if: always()
    needs: [benchmark-smoke, test-shard]
    steps:
      - name: Require smoke and all test shards
        env:
          BENCHMARK_SMOKE_RESULT: ${{{{ needs.benchmark-smoke.result }}}}
          TEST_SHARD_RESULT: ${{{{ needs.test-shard.result }}}}
        run: |
          test "$BENCHMARK_SMOKE_RESULT" = success
          test "$TEST_SHARD_RESULT" = success
      - name: Download coverage data
        uses: actions/download-artifact@v4
        with:
          pattern: coverage-*
          path: coverage-data
          merge-multiple: true
      - name: Audit test shard coverage
        run: uv run scripts/audit_ci_test_shards.py
      - name: Combine coverage
        run: coverage combine coverage-data/coverage.* && coverage report --fail-under=25
""",
        encoding="utf-8",
    )
    for relative_path in ("tests/a/test_one.py", "tests/b/test_two.py"):
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("def test_placeholder(): pass\n", encoding="utf-8")


def test_current_ci_shards_cover_every_test_file_once() -> None:
    root = Path(__file__).resolve().parents[2]

    result = audit_ci_test_shards(root)

    assert result.ok, "\n".join(result.errors)
    assert result.shards == ("a", "b", "c", "d", "e", "f", "g", "h")
    assert result.test_files > 0


def test_audit_rejects_uncovered_and_duplicate_routes(tmp_path: Path) -> None:
    _write_fixture(
        tmp_path,
        paths_a="tests/a",
        paths_b="tests/a",
        selection="not slow and not local_evidence",
    )

    result = _audit_fixture(tmp_path)

    assert not result.ok
    assert any("appear more than once" in error for error in result.errors)
    assert any("tests/b/test_two.py must match exactly one" in error for error in result.errors)


def test_audit_requires_local_evidence_filter(tmp_path: Path) -> None:
    _write_fixture(
        tmp_path,
        paths_a="tests/a",
        paths_b="tests/b",
        selection="not slow",
    )

    result = _audit_fixture(tmp_path)

    assert not result.ok
    assert any("not slow and not local_evidence" in error for error in result.errors)


def test_audit_requires_full_git_history_for_evidence_ancestry(tmp_path: Path) -> None:
    _write_fixture(
        tmp_path,
        paths_a="tests/a",
        paths_b="tests/b",
        selection="not slow and not local_evidence",
    )
    workflow = tmp_path / ".github" / "workflows" / "ci.yml"
    content = workflow.read_text(encoding="utf-8")
    workflow.write_text(content.replace("fetch-depth: 0", "fetch-depth: 1"), encoding="utf-8")

    result = _audit_fixture(tmp_path)

    assert not result.ok
    assert any("fetch-depth: 0" in error for error in result.errors)


def test_audit_requires_fail_closed_coverage_aggregation(tmp_path: Path) -> None:
    _write_fixture(
        tmp_path,
        paths_a="tests/a",
        paths_b="tests/b",
        selection="not slow and not local_evidence",
    )
    workflow = tmp_path / ".github" / "workflows" / "ci.yml"
    content = workflow.read_text(encoding="utf-8")
    workflow.write_text(
        content.replace(
            "needs: [benchmark-smoke, test-shard]", "needs: [benchmark-smoke, lint]"
        ).replace("coverage report --fail-under=25", "coverage report"),
        encoding="utf-8",
    )

    result = _audit_fixture(tmp_path)

    assert not result.ok
    assert any("must need benchmark-smoke and test-shard" in error for error in result.errors)
    assert any("enforce --fail-under=25" in error for error in result.errors)


def test_audit_requires_one_independent_smoke_gate(tmp_path: Path) -> None:
    _write_fixture(
        tmp_path,
        paths_a="tests/a",
        paths_b="tests/b",
        selection="not slow and not local_evidence",
    )
    workflow = tmp_path / ".github" / "workflows" / "ci.yml"
    content = workflow.read_text(encoding="utf-8")
    workflow.write_text(
        content.replace("uv run python benchmark/smoke_test.py", "echo skipped-smoke").replace(
            'test "$BENCHMARK_SMOKE_RESULT" = success\n', ""
        ),
        encoding="utf-8",
    )

    result = _audit_fixture(tmp_path)

    assert not result.ok
    assert any("must execute benchmark/smoke_test.py" in error for error in result.errors)
    assert any("explicitly reject failed smoke" in error for error in result.errors)


def test_audit_rejects_mismatched_coverage_artifact_routes(tmp_path: Path) -> None:
    _write_fixture(
        tmp_path,
        paths_a="tests/a",
        paths_b="tests/b",
        selection="not slow and not local_evidence",
    )
    workflow = tmp_path / ".github" / "workflows" / "ci.yml"
    content = workflow.read_text(encoding="utf-8")
    workflow.write_text(
        content.replace(
            "path: coverage.${{ matrix.shard }}",
            "path: coverage.wrong",
        ).replace("pattern: coverage-*", "pattern: coverage-a"),
        encoding="utf-8",
    )

    result = _audit_fixture(tmp_path)

    assert not result.ok
    assert any("path must match COVERAGE_FILE" in error for error in result.errors)
    assert any("select every shard artifact" in error for error in result.errors)


def test_audit_rejects_unregistered_module_wide_local_evidence_marker(tmp_path: Path) -> None:
    _write_fixture(
        tmp_path,
        paths_a="tests/a",
        paths_b="tests/b",
        selection="not slow and not local_evidence",
    )
    marked = tmp_path / "tests/a/test_one.py"
    marked.write_text(
        "import pytest\n\npytestmark = pytest.mark.local_evidence\n\ndef test_placeholder(): pass\n",
        encoding="utf-8",
    )

    result = _audit_fixture(tmp_path)

    assert not result.ok
    assert any("unregistered local_evidence nodes" in error for error in result.errors)
    assert any("tests/a/test_one.py::<module>" in error for error in result.errors)
