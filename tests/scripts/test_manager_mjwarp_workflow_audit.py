from __future__ import annotations

from pathlib import Path

from tooling.acceptance.commands.workflow_triggers import audit_workflow_triggers


def _write_workflow(root: Path, name: str, content: str) -> None:
    workflow_dir = root / ".github" / "workflows"
    workflow_dir.mkdir(parents=True, exist_ok=True)
    (workflow_dir / name).write_text(content, encoding="utf-8")


def test_current_workflows_exclude_manager_mjwarp_integration_branch() -> None:
    root = Path(__file__).resolve().parents[2]

    result = audit_workflow_triggers(root)

    assert result.ok, "\n".join(result.errors)
    assert {(trigger.path, trigger.event, trigger.branches) for trigger in result.triggers} == {
        (".github/workflows/ci.yml", "pull_request", ("main",)),
        (".github/workflows/docs.yml", "pull_request", ("main",)),
        (".github/workflows/docs.yml", "push", ("main",)),
    }


def test_audit_accepts_explicit_main_only_triggers(tmp_path: Path) -> None:
    _write_workflow(
        tmp_path,
        "ci.yml",
        """
name: CI
on:
  pull_request:
    branches: [main]
  push:
    branches:
      - main
jobs: {}
""",
    )

    result = audit_workflow_triggers(tmp_path)

    assert result.ok
    assert {trigger.event for trigger in result.triggers} == {"pull_request", "push"}


def test_audit_rejects_integration_branch_target(tmp_path: Path) -> None:
    _write_workflow(
        tmp_path,
        "ci.yml",
        """
on:
  pull_request:
    branches: [main, feat/manager-mjwarp-manager-mjwarp]
jobs: {}
""",
    )

    result = audit_workflow_triggers(tmp_path)

    assert not result.ok
    assert any("branches must be exactly ['main']" in error for error in result.errors)


def test_audit_rejects_implicit_all_branches(tmp_path: Path) -> None:
    _write_workflow(
        tmp_path,
        "ci.yml",
        """
on: [push, pull_request]
jobs: {}
""",
    )

    result = audit_workflow_triggers(tmp_path)

    assert not result.ok
    assert any("must declare `branches: [main]` explicitly" in error for error in result.errors)


def test_audit_rejects_branches_ignore(tmp_path: Path) -> None:
    _write_workflow(
        tmp_path,
        "ci.yml",
        """
on:
  push:
    branches-ignore: [main]
jobs: {}
""",
    )

    result = audit_workflow_triggers(tmp_path)

    assert not result.ok
    assert any("requires an explicit allowlist" in error for error in result.errors)


def test_audit_rejects_unfilterable_pr_activity_event(tmp_path: Path) -> None:
    _write_workflow(
        tmp_path,
        "review.yml",
        """
on: [pull_request_review]
jobs: {}
""",
    )

    result = audit_workflow_triggers(tmp_path)

    assert not result.ok
    assert any("can run during the intermediate PR lifecycle" in error for error in result.errors)


def test_audit_rejects_branch_create_event(tmp_path: Path) -> None:
    _write_workflow(
        tmp_path,
        "create.yml",
        """
on: [create]
jobs: {}
""",
    )

    result = audit_workflow_triggers(tmp_path)

    assert not result.ok
    assert any(
        "`create` can run during the intermediate PR lifecycle" in error for error in result.errors
    )


def test_audit_rejects_invalid_yaml(tmp_path: Path) -> None:
    _write_workflow(tmp_path, "broken.yml", "on: [pull_request\n")

    result = audit_workflow_triggers(tmp_path)

    assert not result.ok
    assert any("cannot parse workflow YAML" in error for error in result.errors)


def test_audit_rejects_empty_workflow_directory(tmp_path: Path) -> None:
    result = audit_workflow_triggers(tmp_path)

    assert not result.ok
    assert any("no workflow YAML files found" in error for error in result.errors)
