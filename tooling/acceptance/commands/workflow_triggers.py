"""Audit that managed MuJoCo/MJWarp rollout intermediate PRs cannot trigger GitHub Actions.

The integration branch is intentionally excluded from both pull-request and
push workflow triggers. This audit is read-only and fails closed.

    uv run scripts/audit_acceptance.py workflow-triggers
    uv run scripts/audit_acceptance.py workflow-triggers --json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
WORKFLOW_DIR = Path(".github/workflows")
INTEGRATION_BRANCH = "feat/manager-mjwarp-manager-mjwarp"
ALLOWED_BRANCHES = ("main",)

BRANCH_FILTERED_EVENTS = ("pull_request", "pull_request_target", "push")
UNFILTERABLE_INTERMEDIATE_EVENTS = (
    "create",
    "delete",
    "pull_request_review",
    "pull_request_review_comment",
)


@dataclass(frozen=True)
class TriggerRecord:
    path: str
    event: str
    branches: tuple[str, ...]


@dataclass(frozen=True)
class WorkflowAuditResult:
    workflow_count: int
    triggers: tuple[TriggerRecord, ...]
    errors: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.errors


def _workflow_paths(root: Path) -> list[Path]:
    workflow_dir = root / WORKFLOW_DIR
    return sorted({*workflow_dir.glob("*.yml"), *workflow_dir.glob("*.yaml")})


def _load_workflow(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        raw = yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        return None, f"{path}: cannot parse workflow YAML: {type(exc).__name__}: {exc}"
    if not isinstance(raw, dict):
        return None, f"{path}: workflow root must be a mapping"
    return raw, None


def _event_map(workflow: dict[str, Any], path: Path) -> tuple[dict[str, Any], list[str]]:
    raw_on = workflow.get("on")
    if isinstance(raw_on, str):
        return {raw_on: None}, []
    if isinstance(raw_on, list):
        if not all(isinstance(event, str) for event in raw_on):
            return {}, [f"{path}: every event in `on` must be a string"]
        return dict.fromkeys(raw_on), []
    if isinstance(raw_on, dict):
        return raw_on, []
    return {}, [f"{path}: `on` must be a string, list, or mapping"]


def _branches_for_event(config: Any, path: Path, event: str) -> tuple[tuple[str, ...], str | None]:
    if not isinstance(config, dict):
        return (), f"{path}: `{event}` must declare `branches: [main]` explicitly"
    if "branches-ignore" in config:
        return (), f"{path}: `{event}` uses branches-ignore; #705 requires an explicit allowlist"

    raw_branches = config.get("branches")
    if isinstance(raw_branches, str):
        branches = (raw_branches,)
    elif isinstance(raw_branches, list) and all(isinstance(branch, str) for branch in raw_branches):
        branches = tuple(raw_branches)
    else:
        return (), f"{path}: `{event}` must declare `branches: [main]` explicitly"

    if branches != ALLOWED_BRANCHES:
        return (
            branches,
            f"{path}: `{event}` branches must be exactly {list(ALLOWED_BRANCHES)!r}; "
            f"got {list(branches)!r}",
        )
    return branches, None


def audit_workflow_triggers(root: Path = REPO_ROOT) -> WorkflowAuditResult:
    paths = _workflow_paths(root)
    errors: list[str] = []
    triggers: list[TriggerRecord] = []
    if not paths:
        errors.append(f"{root / WORKFLOW_DIR}: no workflow YAML files found")

    for path in paths:
        relative_path = str(path.relative_to(root))
        workflow, load_error = _load_workflow(path)
        if load_error is not None:
            errors.append(load_error.replace(str(root) + "/", ""))
            continue
        assert workflow is not None

        events, event_errors = _event_map(workflow, Path(relative_path))
        errors.extend(event_errors)
        for event in UNFILTERABLE_INTERMEDIATE_EVENTS:
            if event in events:
                errors.append(
                    f"{relative_path}: `{event}` can run during the intermediate PR lifecycle"
                )

        for event in BRANCH_FILTERED_EVENTS:
            if event not in events:
                continue
            branches, branch_error = _branches_for_event(events[event], Path(relative_path), event)
            triggers.append(TriggerRecord(relative_path, event, branches))
            if branch_error is not None:
                errors.append(branch_error)

    return WorkflowAuditResult(
        workflow_count=len(paths),
        triggers=tuple(triggers),
        errors=tuple(errors),
    )


def _print_human(result: WorkflowAuditResult) -> None:
    print(
        f"managed MuJoCo/MJWarp rollout workflow audit: workflows={result.workflow_count} "
        f"branch_triggers={len(result.triggers)}"
    )
    for trigger in result.triggers:
        print(f"  PASS {trigger.path}: {trigger.event} -> {list(trigger.branches)!r}")
    if result.errors:
        for error in result.errors:
            print(f"  FAIL {error}")
        return
    print(f"PASS: no workflow trigger can target `{INTEGRATION_BRANCH}`")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    args = parser.parse_args()

    result = audit_workflow_triggers()
    if args.json:
        print(json.dumps(asdict(result), indent=2))
    else:
        _print_human(result)
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
