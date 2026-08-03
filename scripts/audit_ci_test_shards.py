"""Audit the standard CI test sharding and coverage aggregation contract.

uv run scripts/audit_ci_test_shards.py
uv run scripts/audit_ci_test_shards.py --json

Two-workflow model
------------------
ci.yml     — standard PR gate: fast shards (a, b, d, f, scripts).
ci-full.yml — full CI: evidence / benchmark / acceptance shards (c, e, g, h).

Together the two workflows cover every test file exactly once.  This script
validates:
  1. ci.yml structural contract (fetch-depth, coverage artifacts, aggregate job).
  2. Combined shard coverage: every test file matches exactly one route across
     ci.yml + ci-full.yml.
  3. No overlapping routes between the two workflows.
"""

from __future__ import annotations

import argparse
import ast
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_PATH = Path(".github/workflows/ci.yml")
FULL_WORKFLOW_PATH = Path(".github/workflows/ci-full.yml")
PYPROJECT_PATH = Path("pyproject.toml")
TEST_ROOT = Path("tests")
LOCAL_EVIDENCE_EXPRESSION = "not slow and not local_evidence"
AGGREGATE_INSTALL_COMMAND = "uv sync --only-group dev"
EXPECTED_LOCAL_EVIDENCE_TEST_IDS = frozenset(
    {
        "tests/acceptance/manager_mjwarp/test_g1_baseline_provenance.py::<module>",
        "tests/acceptance/manager_mjwarp/test_phase1_evidence.py::<module>",
        "tests/acceptance/manager_mjwarp/test_phase2_evidence.py::<module>",
        "tests/acceptance/manager_mjwarp/test_phase3_evidence.py::<module>",
        "tests/acceptance/manager_mjwarp/test_phase4_evidence.py::<module>",
        "tests/acceptance/manager_mjwarp/test_phase5_evidence.py::<module>",
        "tests/acceptance/manager_mjwarp/test_phase6_evidence.py::<module>",
        "tests/acceptance/manager_mjwarp/test_threshold_manifest.py::<module>",
        "tests/benchmark/test_manager_mjwarp_g1_baseline_runner.py::"
        "test_subprocess_is_uv_run_and_enforces_cpu_affinity",
        "tests/benchmark/test_manager_mjwarp_training_behavior.py::"
        "test_all_paired_seeds_meet_frozen_behavior_gates",
        "tests/benchmark/test_managed_g1_host_benchmark.py::"
        "test_fused_host_meets_preregistered_gate",
        "tests/benchmark/test_mjwarp_dr_benchmark.py::"
        "test_dr_profiles_meet_preregistered_density_gates",
        "tests/benchmark/test_mjwarp_ppo_benchmark.py::test_device_profile_meets_end_to_end_gate",
        "tests/integration/test_manager_mjwarp_legacy_retirement.py::"
        "test_legacy_removal_requires_full_entrypoint_and_rollback_evidence",
        "tests/scripts/test_manager_mjwarp_final_gate.py::"
        "test_committed_final_artifact_is_fresh_and_promoted",
        "tests/scripts/test_manager_mjwarp_final_gate.py::<module>",
        "tests/scripts/test_manager_mjwarp_g1_baseline.py::<module>",
        "tests/scripts/test_manager_mjwarp_support_audit.py::<module>",
        "tests/scripts/test_manager_mjwarp_task_rollout_audit.py::<module>",
        "tests/scripts/test_manager_mjwarp_thresholds.py::<module>",
        "tests/tools/test_manager_mjwarp_phase4_evidence.py::<module>",
        "tests/tools/test_manager_mjwarp_phase5_evidence.py::<module>",
        "tests/tools/test_manager_mjwarp_phase6_evidence.py::<module>",
        "tests/tools/test_manager_mjwarp_threshold_amendment.py::<module>",
        "tests/tools/test_manager_mjwarp_threshold_contract.py::<module>",
        "tests/tools/test_manager_mjwarp_threshold_freeze_receipt.py::<module>",
        "tests/tools/test_manager_mjwarp_training_behavior.py::<module>",
    }
)


@dataclass(frozen=True)
class CiTestShardAuditResult:
    shards: tuple[str, ...]
    test_files: int
    local_evidence_tests: tuple[str, ...]
    errors: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.errors


def _mapping(value: Any, label: str, errors: list[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        errors.append(f"{label} must be a mapping")
        return {}
    return value


def _steps(job: dict[str, Any], label: str, errors: list[str]) -> list[dict[str, Any]]:
    raw_steps = job.get("steps")
    if not isinstance(raw_steps, list) or not raw_steps:
        errors.append(f"{label}.steps must be a non-empty list")
        return []
    if not all(isinstance(step, dict) for step in raw_steps):
        errors.append(f"{label}.steps must contain only mappings")
        return []
    return raw_steps


def _named_step(steps: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
    return next((step for step in steps if step.get("name") == name), None)


def _matches_test_path(relative_file: Path, route: str, root: Path) -> bool:
    if any(character in route for character in "*?["):
        return relative_file.match(route)
    route_path = Path(route)
    absolute_route = root / route_path
    if absolute_route.is_dir():
        return relative_file == route_path or route_path in relative_file.parents
    return relative_file == route_path


def _test_files(root: Path) -> tuple[Path, ...]:
    tests_root = root / TEST_ROOT
    if not tests_root.is_dir():
        return ()
    return tuple(
        sorted(path.relative_to(root) for path in tests_root.rglob("test*.py") if path.is_file())
    )


def _audit_aggregate_dependencies(root: Path, errors: list[str]) -> None:
    pyproject_path = root / PYPROJECT_PATH
    try:
        pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, tomllib.TOMLDecodeError) as exc:
        errors.append(
            f"{PYPROJECT_PATH}: cannot parse project metadata: {type(exc).__name__}: {exc}"
        )
        return

    pyproject_map = _mapping(pyproject, str(PYPROJECT_PATH), errors)
    dependency_groups = _mapping(
        pyproject_map.get("dependency-groups"),
        f"{PYPROJECT_PATH}.dependency-groups",
        errors,
    )
    dev_dependencies = dependency_groups.get("dev")
    if not isinstance(dev_dependencies, list):
        errors.append(f"{PYPROJECT_PATH}.dependency-groups.dev must be a list")
        return
    if "pyyaml" not in dev_dependencies:
        errors.append(
            f"{PYPROJECT_PATH}.dependency-groups.dev must declare pyyaml for the CI shard audit"
        )


def _is_local_evidence_marker(node: ast.expr) -> bool:
    if isinstance(node, ast.Call):
        node = node.func
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "local_evidence"
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == "mark"
        and isinstance(node.value.value, ast.Name)
        and node.value.value.id == "pytest"
    )


def _contains_local_evidence_marker(node: ast.AST) -> bool:
    return any(
        isinstance(candidate, ast.expr) and _is_local_evidence_marker(candidate)
        for candidate in ast.walk(node)
    )


class _LocalEvidenceVisitor(ast.NodeVisitor):
    def __init__(self, relative_file: Path) -> None:
        self.relative_file = relative_file
        self.class_names: list[str] = []
        self.test_ids: list[str] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.class_names.append(node.name)
        if any(_is_local_evidence_marker(decorator) for decorator in node.decorator_list):
            parts = [self.relative_file.as_posix(), *self.class_names, "<class>"]
            self.test_ids.append("::".join(parts))
        for statement in node.body:
            if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
                continue
            targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
            if any(
                isinstance(target, ast.Name) and target.id == "pytestmark" for target in targets
            ):
                if statement.value is not None and _contains_local_evidence_marker(statement.value):
                    parts = [self.relative_file.as_posix(), *self.class_names, "<pytestmark>"]
                    self.test_ids.append("::".join(parts))
        self.generic_visit(node)
        self.class_names.pop()

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        if any(_is_local_evidence_marker(decorator) for decorator in node.decorator_list):
            parts = [self.relative_file.as_posix(), *self.class_names, node.name]
            self.test_ids.append("::".join(parts))
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # noqa: N802
        self._visit_function(node)


def _local_evidence_test_ids(
    root: Path,
    test_files: tuple[Path, ...],
    errors: list[str],
) -> tuple[str, ...]:
    test_ids: list[str] = []
    for relative_file in test_files:
        path = root / relative_file
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(relative_file))
        except (OSError, UnicodeError, SyntaxError) as exc:
            errors.append(
                f"{relative_file}: cannot parse test marker scope: {type(exc).__name__}: {exc}"
            )
            continue
        for statement in tree.body:
            if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
                continue
            targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
            value = statement.value
            if any(
                isinstance(target, ast.Name) and target.id == "pytestmark" for target in targets
            ):
                if value is not None and _contains_local_evidence_marker(value):
                    test_ids.append(f"{relative_file.as_posix()}::<module>")
        visitor = _LocalEvidenceVisitor(relative_file)
        visitor.visit(tree)
        test_ids.extend(visitor.test_ids)
    return tuple(sorted(test_ids))


def _load_shard_routes(
    workflow: dict[str, Any],
    workflow_label: str,
    errors: list[str],
) -> dict[str, tuple[str, ...]]:
    """Extract {shard_name: (route, ...)} from a parsed workflow YAML."""
    jobs = _mapping(workflow.get("jobs"), f"{workflow_label}.jobs", errors)
    shard_job = _mapping(jobs.get("test-shard"), f"{workflow_label}.jobs.test-shard", errors)
    strategy = _mapping(
        shard_job.get("strategy"), f"{workflow_label}.jobs.test-shard.strategy", errors
    )
    matrix = _mapping(
        strategy.get("matrix"), f"{workflow_label}.jobs.test-shard.strategy.matrix", errors
    )
    raw_includes = matrix.get("include")
    includes: list[dict[str, Any]] = []
    if not isinstance(raw_includes, list) or not raw_includes:
        errors.append(
            f"{workflow_label}.jobs.test-shard.strategy.matrix.include must be a non-empty list"
        )
        return {}
    if not all(isinstance(entry, dict) for entry in raw_includes):
        errors.append(
            f"{workflow_label}.jobs.test-shard.strategy.matrix.include must contain only mappings"
        )
        return {}
    includes = raw_includes

    shard_routes: dict[str, tuple[str, ...]] = {}
    for index, entry in enumerate(includes):
        shard = entry.get("shard")
        paths = entry.get("paths")
        if not isinstance(shard, str) or not shard:
            errors.append(
                f"{workflow_label}.matrix.include[{index}].shard must be a non-empty string"
            )
            continue
        if shard in shard_routes:
            errors.append(f"{workflow_label}: duplicate shard name: {shard}")
            continue
        if not isinstance(paths, str) or not paths.split():
            errors.append(
                f"{workflow_label}.matrix.include[{index}].paths must be a non-empty string"
            )
            continue
        routes = tuple(paths.split())
        invalid = [route for route in routes if not route.startswith("tests/")]
        if invalid:
            errors.append(f"{workflow_label} shard {shard} has routes outside tests/: {invalid}")
        shard_routes[shard] = routes
    return shard_routes


def audit_ci_test_shards(
    root: Path = REPO_ROOT,
    *,
    expected_local_evidence_test_ids: frozenset[str] = EXPECTED_LOCAL_EVIDENCE_TEST_IDS,
) -> CiTestShardAuditResult:
    errors: list[str] = []
    workflow_path = root / WORKFLOW_PATH
    try:
        workflow = yaml.load(workflow_path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        return CiTestShardAuditResult(
            shards=(),
            test_files=0,
            local_evidence_tests=(),
            errors=(f"{WORKFLOW_PATH}: cannot parse workflow: {type(exc).__name__}: {exc}",),
        )

    workflow_map = _mapping(workflow, str(WORKFLOW_PATH), errors)
    _audit_aggregate_dependencies(root, errors)
    jobs = _mapping(workflow_map.get("jobs"), "jobs", errors)
    smoke_job = _mapping(jobs.get("benchmark-smoke"), "jobs.benchmark-smoke", errors)
    shard_job = _mapping(jobs.get("test-shard"), "jobs.test-shard", errors)
    aggregate_job = _mapping(jobs.get("test"), "jobs.test", errors)

    # --- standard shard routes (structural contract applies) ---
    strategy = _mapping(shard_job.get("strategy"), "jobs.test-shard.strategy", errors)
    matrix = _mapping(strategy.get("matrix"), "jobs.test-shard.strategy.matrix", errors)
    raw_includes = matrix.get("include")
    includes: list[dict[str, Any]] = []
    if not isinstance(raw_includes, list) or not raw_includes:
        errors.append("jobs.test-shard.strategy.matrix.include must be a non-empty list")
    elif not all(isinstance(entry, dict) for entry in raw_includes):
        errors.append("jobs.test-shard.strategy.matrix.include must contain only mappings")
    else:
        includes = raw_includes

    shard_routes: dict[str, tuple[str, ...]] = {}
    all_routes: list[str] = []
    for index, entry in enumerate(includes):
        shard = entry.get("shard")
        paths = entry.get("paths")
        if not isinstance(shard, str) or not shard:
            errors.append(f"matrix.include[{index}].shard must be a non-empty string")
            continue
        if shard in shard_routes:
            errors.append(f"duplicate shard name: {shard}")
            continue
        if not isinstance(paths, str) or not paths.split():
            errors.append(f"matrix.include[{index}].paths must be a non-empty string")
            continue
        routes = tuple(paths.split())
        invalid = [route for route in routes if not route.startswith("tests/")]
        if invalid:
            errors.append(f"shard {shard} has routes outside tests/: {invalid}")
        shard_routes[shard] = routes
        all_routes.extend(routes)

    duplicate_routes = sorted({route for route in all_routes if all_routes.count(route) > 1})
    if duplicate_routes:
        errors.append(f"test routes appear more than once: {duplicate_routes}")

    # --- full CI shard routes (loaded from ci-full.yml if present) ---
    full_shard_routes: dict[str, tuple[str, ...]] = {}
    full_workflow_path = root / FULL_WORKFLOW_PATH
    if full_workflow_path.exists():
        try:
            full_workflow = yaml.load(
                full_workflow_path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader
            )
        except (OSError, UnicodeError, yaml.YAMLError) as exc:
            errors.append(
                f"{FULL_WORKFLOW_PATH}: cannot parse workflow: {type(exc).__name__}: {exc}"
            )
            full_workflow = None
        if isinstance(full_workflow, dict):
            full_shard_routes = _load_shard_routes(full_workflow, str(FULL_WORKFLOW_PATH), errors)
            # Check for route overlap between standard and full workflows
            full_all_routes: list[str] = [
                route for routes in full_shard_routes.values() for route in routes
            ]
            overlap = sorted(set(all_routes) & set(full_all_routes))
            if overlap:
                errors.append(
                    f"routes appear in both ci.yml and ci-full.yml (must be non-overlapping): "
                    f"{overlap}"
                )

    # --- combined coverage check across standard + full shards ---
    combined_routes: dict[str, tuple[str, ...]] = {**shard_routes, **full_shard_routes}

    test_files = _test_files(root)
    if not test_files:
        errors.append(f"{TEST_ROOT}: no test*.py files found")
    local_evidence_test_ids = _local_evidence_test_ids(root, test_files, errors)
    actual_local_evidence = set(local_evidence_test_ids)
    missing_local_evidence = sorted(expected_local_evidence_test_ids - actual_local_evidence)
    unexpected_local_evidence = sorted(actual_local_evidence - expected_local_evidence_test_ids)
    if missing_local_evidence:
        errors.append(f"local_evidence nodes are missing markers: {missing_local_evidence}")
    if unexpected_local_evidence:
        errors.append(f"unregistered local_evidence nodes: {unexpected_local_evidence}")

    combined_all_routes = [route for routes in combined_routes.values() for route in routes]
    route_hits = dict.fromkeys(combined_all_routes, 0)
    for test_file in test_files:
        matched_routes = [
            (shard, route)
            for shard, routes in combined_routes.items()
            for route in routes
            if _matches_test_path(test_file, route, root)
        ]
        for _, route in matched_routes:
            route_hits[route] += 1
        if len(matched_routes) != 1:
            errors.append(
                f"{test_file} must match exactly one shard route; got {matched_routes or 'none'}"
            )
    empty_routes = sorted(route for route, hits in route_hits.items() if hits == 0)
    if empty_routes:
        errors.append(f"test routes match no test files: {empty_routes}")

    if shard_job.get("runs-on") != "ubuntu-slim":
        errors.append("jobs.test-shard.runs-on must be ubuntu-slim")
    if strategy.get("fail-fast") != "false":
        errors.append("jobs.test-shard.strategy.fail-fast must be false")
    shard_steps = _steps(shard_job, "jobs.test-shard", errors)
    if _named_step(shard_steps, "Benchmark entrypoint smoke test") is not None:
        errors.append("test-shard must not repeat the benchmark entrypoint smoke test")
    checkout_step = next(
        (step for step in shard_steps if str(step.get("uses", "")).startswith("actions/checkout@")),
        None,
    )
    if checkout_step is None:
        errors.append("test-shard must check out the repository with actions/checkout")
    else:
        checkout_with = _mapping(
            checkout_step.get("with"),
            "jobs.test-shard.steps.actions/checkout.with",
            errors,
        )
        if checkout_with.get("fetch-depth") != "0":
            errors.append("test-shard checkout must set fetch-depth: 0 for evidence ancestry")
    test_step = _named_step(shard_steps, "Test with coverage")
    if test_step is None:
        errors.append("test-shard must contain `Test with coverage`")
    else:
        command = test_step.get("run")
        environment = test_step.get("env")
        if not isinstance(command, str) or "${{ matrix.paths }}" not in command:
            errors.append("test command must execute `${{ matrix.paths }}`")
        if not isinstance(command, str) or LOCAL_EVIDENCE_EXPRESSION not in command:
            errors.append(f"test command must select `{LOCAL_EVIDENCE_EXPRESSION}`")
        if not isinstance(command, str) or "--cov=src/unilab" not in command:
            errors.append("test shards must measure coverage for src/unilab")
        if not isinstance(command, str) or "--cov-report=" not in command:
            errors.append("test shards must defer coverage reporting to aggregation")
        if not isinstance(command, str) or "--cov-fail-under=0" not in command:
            errors.append("test shards must defer the coverage threshold to aggregation")
        if not isinstance(environment, dict) or environment.get("COVERAGE_FILE") != (
            "coverage.${{ matrix.shard }}"
        ):
            errors.append("test shards must write a unique coverage.${{ matrix.shard }} file")
    upload_step = _named_step(shard_steps, "Upload coverage data")
    if upload_step is None or not str(upload_step.get("uses", "")).startswith(
        "actions/upload-artifact@"
    ):
        errors.append("test-shard must upload coverage with actions/upload-artifact")
    else:
        upload_with = _mapping(
            upload_step.get("with"),
            "jobs.test-shard.steps.Upload coverage data.with",
            errors,
        )
        if upload_with.get("name") != "coverage-${{ matrix.shard }}":
            errors.append("coverage artifact name must include matrix.shard")
        if upload_with.get("path") != "coverage.${{ matrix.shard }}":
            errors.append("coverage artifact path must match COVERAGE_FILE")
        if upload_with.get("if-no-files-found") != "error":
            errors.append("coverage upload must fail when its data file is missing")

    if aggregate_job.get("name") != "test (ubuntu-slim)":
        errors.append("aggregate test job must preserve the `test (ubuntu-slim)` check name")
    smoke_steps = _steps(smoke_job, "jobs.benchmark-smoke", errors)
    smoke_step = _named_step(smoke_steps, "Benchmark entrypoint smoke test")
    if smoke_job.get("runs-on") != "ubuntu-slim":
        errors.append("jobs.benchmark-smoke.runs-on must be ubuntu-slim")
    if smoke_step is None or "benchmark/smoke_test.py" not in str(smoke_step.get("run", "")):
        errors.append("benchmark-smoke must execute benchmark/smoke_test.py")
    needs = aggregate_job.get("needs")
    if not isinstance(needs, list) or set(needs) != {"benchmark-smoke", "test-shard"}:
        errors.append("aggregate test job must need benchmark-smoke and test-shard")
    if "always()" not in str(aggregate_job.get("if", "")):
        errors.append("aggregate test job must run with always() and fail closed on shard failure")
    aggregate_steps = _steps(aggregate_job, "jobs.test", errors)
    install_step = _named_step(aggregate_steps, "Install validation dependencies")
    install_command = "" if install_step is None else str(install_step.get("run", "")).strip()
    if install_command != AGGREGATE_INSTALL_COMMAND:
        errors.append(
            "aggregate test job must install its isolated validation environment with "
            f"`{AGGREGATE_INSTALL_COMMAND}`"
        )
    require_step = _named_step(aggregate_steps, "Require smoke and all test shards")
    require_command = "" if require_step is None else str(require_step.get("run", ""))
    require_environment = None if require_step is None else require_step.get("env")
    required_result_checks = (
        'test "$BENCHMARK_SMOKE_RESULT" = success',
        'test "$TEST_SHARD_RESULT" = success',
    )
    if any(check not in require_command for check in required_result_checks):
        errors.append("aggregate test job must explicitly reject failed smoke and shard results")
    elif not isinstance(require_environment, dict) or require_environment != {
        "BENCHMARK_SMOKE_RESULT": "${{ needs.benchmark-smoke.result }}",
        "TEST_SHARD_RESULT": "${{ needs.test-shard.result }}",
    }:
        errors.append("aggregate test job must read smoke and test-shard results from needs")
    download_step = _named_step(aggregate_steps, "Download coverage data")
    if download_step is None or not str(download_step.get("uses", "")).startswith(
        "actions/download-artifact@"
    ):
        errors.append("aggregate test job must download every coverage artifact")
    else:
        download_with = _mapping(
            download_step.get("with"),
            "jobs.test.steps.Download coverage data.with",
            errors,
        )
        if download_with.get("pattern") != "coverage-*":
            errors.append("coverage download must select every shard artifact")
        if download_with.get("path") != "coverage-data":
            errors.append("coverage download path must be coverage-data")
        if download_with.get("merge-multiple") != "true":
            errors.append("coverage download must merge all shard artifacts")
    audit_step = _named_step(aggregate_steps, "Audit test shard coverage")
    if audit_step is None or "scripts/audit_ci_test_shards.py" not in str(
        audit_step.get("run", "")
    ):
        errors.append("aggregate test job must execute the shard audit")
    combine_step = _named_step(aggregate_steps, "Combine coverage")
    combine_command = "" if combine_step is None else str(combine_step.get("run", ""))
    if (
        "coverage combine coverage-data/coverage.*" not in combine_command
        or "--fail-under=25" not in combine_command
    ):
        errors.append("aggregate test job must combine coverage and enforce --fail-under=25")

    return CiTestShardAuditResult(
        shards=tuple(shard_routes),
        test_files=len(test_files),
        local_evidence_tests=local_evidence_test_ids,
        errors=tuple(errors),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = audit_ci_test_shards()
    if args.json:
        print(json.dumps(asdict(result), indent=2))
    elif result.ok:
        print(
            "PASS CI test shards: "
            f"shards={len(result.shards)} test_files={result.test_files} "
            f"local_evidence_tests={len(result.local_evidence_tests)}"
        )
    else:
        print("FAIL CI test shard audit")
        for error in result.errors:
            print(f"- {error}")
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
