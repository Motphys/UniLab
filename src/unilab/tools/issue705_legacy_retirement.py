"""Fail-closed release evidence for Issue #705 legacy-route retirement."""

from __future__ import annotations

import ast
import hashlib
import json
import platform
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

from omegaconf import OmegaConf

from unilab.tools.claim_gap_audit import InventoryTestState, load_claim_gap_inventory

SCHEMA_VERSION = 1
ISSUE = 705
CHILD_ISSUE = 841
CLAIM_ID = "P7-LEGACY-RETIREMENT"
PLAN_FINGERPRINT = "issue705-legacy-retirement-v1"
PLAN_PATH = Path("tests/acceptance/issue_705/legacy_retirement_plan.yaml")
ROLLBACK_PATH = Path("tests/acceptance/issue_705/legacy_retirement_rollback.yaml")
EVIDENCE_PATH = Path("tests/acceptance/issue_705/artifacts/phase_7_legacy_retirement.json")
CLAIM_INVENTORY_PATH = Path("tests/acceptance/issue_705/claim_test_inventory.yaml")
OWNER_PATH = Path("conf/ppo/task/g1_walk_flat/mjwarp.yaml")
OWNER_MODULE_PATH = Path("src/unilab/envs/locomotion/g1/joystick.py")
ENTRYPOINT_TEST_ID = (
    "tests/integration/test_issue705_entrypoints.py::"
    "test_supported_train_play_visualize_export_matrix"
)
LEGACY_TEST_ID = (
    "tests/integration/test_issue705_legacy_retirement.py::"
    "test_legacy_removal_requires_full_entrypoint_and_rollback_evidence"
)

_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_PYTEST_COUNT_RE = re.compile(r"(\d+) (passed|skipped|xfailed|xpassed|deselected)")
_EVIDENCE_KIND = "issue705-legacy-retirement-evidence-v1"
_ROLLBACK_KIND = "issue705-legacy-retirement-rollback-v1"
_INTEGRATION_BASE_COMMIT = "f9920a7f0d86317e36383e1266918755cb8841c2"
_ENTRYPOINT_PR = 840
_ENTRYPOINT_ARGV = (
    "uv",
    "run",
    "pytest",
    "-m",
    "slow",
    ENTRYPOINT_TEST_ID,
    "-vv",
)
_INPUT_PATHS = (
    Path("conf/ppo/config.yaml"),
    OWNER_PATH,
    Path("scripts/audit_issue705_legacy_retirement.py"),
    Path("scripts/capture_issue705_legacy_retirement.py"),
    Path("scripts/train_rsl_rl.py"),
    Path("src/unilab/base/np_env.py"),
    Path("src/unilab/base/registry.py"),
    Path("src/unilab/envs/locomotion/g1/__init__.py"),
    OWNER_MODULE_PATH,
    Path("src/unilab/envs/locomotion/g1/managed_device.py"),
    Path("src/unilab/envs/locomotion/g1/managed_fused.py"),
    Path("src/unilab/envs/locomotion/g1/managed_reference.py"),
    Path("src/unilab/tools/issue705_legacy_retirement.py"),
    Path("src/unilab/training/entrypoints.py"),
    CLAIM_INVENTORY_PATH,
    PLAN_PATH,
    ROLLBACK_PATH,
    Path("tests/benchmark/test_managed_g1_host_benchmark.py"),
    Path("tests/envs/locomotion/g1/test_mjwarp_managed_env.py"),
    Path("tests/integration/test_issue705_entrypoints.py"),
    Path("tests/integration/test_issue705_legacy_retirement.py"),
    Path("tests/manager/test_g1_reference_differential.py"),
    Path("tests/scripts/test_issue705_legacy_retirement_audit.py"),
    Path("uv.lock"),
)
_EXPECTED_IMPLEMENTATION_PATHS = (
    Path("scripts/audit_issue705_legacy_retirement.py"),
    Path("scripts/capture_issue705_legacy_retirement.py"),
    Path("src/unilab/envs/locomotion/g1/__init__.py"),
    OWNER_MODULE_PATH,
    Path("src/unilab/tools/issue705_legacy_retirement.py"),
    CLAIM_INVENTORY_PATH,
    PLAN_PATH,
    ROLLBACK_PATH,
    Path("tests/envs/locomotion/g1/test_mjwarp_managed_env.py"),
    Path("tests/integration/test_issue705_legacy_retirement.py"),
    Path("tests/scripts/test_issue705_legacy_retirement_audit.py"),
)
_EXPECTED_BASELINE_FILES = (
    (
        OWNER_MODULE_PATH,
        "sha256:c9634092e5648135d248b2486209c2bb11fc0793be767e27c6b5562eb1214756",
    ),
    (
        Path("src/unilab/training/entrypoints.py"),
        "sha256:92d72a5e539de00aaf844178a23fca93c1e5585722fc5f307df23edfe7f90bee",
    ),
    (
        Path("tests/integration/test_issue705_entrypoints.py"),
        "sha256:7d5fa82c018d13e0b6869c2761095798fbdc3dda5c68bc6ea4e58df3bd1ac19b",
    ),
    (
        OWNER_PATH,
        "sha256:c886ea57bb5e8733a8b4dffe442161c654fa9ecfe547953b8fac5e5bff32812c",
    ),
)
_EXPECTED_ROUTES = (
    "train",
    "play",
    "visualize",
    "export",
    "checkpoint_save",
    "checkpoint_load",
    "resume",
)
_EXPECTED_PROHIBITED = (
    "init_state",
    "reset",
    "step",
    "apply_action",
    "update_state",
    "play",
)
_EXPECTED_DIAGNOSTICS = (
    "G1WalkFlat + mjwarp is managed-only",
    "retired hand-written NpEnv lifecycle",
    "task=g1_walk_flat/mjwarp",
    "training.operation=train",
    "training.operation=export",
    "task=g1_walk_flat/mujoco",
)
_EXPECTED_RETAINED = (
    ("g1_walk_flat_mujoco_owner", "mujoco", "G1WalkEnv"),
    ("g1_walk_flat_motrix_owner", "motrix", "G1WalkEnv"),
    ("g1_managed_reference_oracle", "mujoco", "G1WalkEnv"),
)


class LegacyRetirementError(ValueError):
    """Raised when frozen retirement evidence is malformed or stale."""


@dataclass(frozen=True)
class IntegrationBase:
    commit: str
    entrypoint_pr: int
    entrypoint_merge_commit: str


@dataclass(frozen=True)
class RetiredRoute:
    env_name: str
    backend: str
    owner_module: Path
    previous_owner_class: str
    replacement_owner_class: str
    replacement_base_class: str
    runtime_factory: str
    prohibited_operations: tuple[str, ...]


@dataclass(frozen=True)
class RetainedRoute:
    route_id: str
    env_name: str
    backend: str
    owner_class: str
    reason: str


@dataclass(frozen=True)
class EntrypointPrerequisite:
    contract_fingerprint: str
    owner_yaml: Path
    required_test_id: str
    repetitions: int
    required_routes: tuple[str, ...]


@dataclass(frozen=True)
class DiagnosticContract:
    exception_class: str
    required_fragments: tuple[str, ...]


@dataclass(frozen=True)
class LegacyRetirementPlan:
    schema_version: int
    issue: int
    child_issue: int
    claim_id: str
    plan_fingerprint: str
    integration_base: IntegrationBase
    retired_route: RetiredRoute
    retained_routes: tuple[RetainedRoute, ...]
    entrypoint_prerequisite: EntrypointPrerequisite
    diagnostic_contract: DiagnosticContract
    rollback_receipt: Path
    evidence_artifact: Path
    implementation_paths: tuple[Path, ...]
    source: Path


@dataclass(frozen=True)
class RollbackReceipt:
    schema_version: int
    artifact_kind: str
    issue: int
    child_issue: int
    plan_fingerprint: str
    baseline_commit: str
    baseline_files: tuple[tuple[Path, str], ...]
    restore_env_name: str
    restore_backend: str
    restore_owner_class: str
    restore_registration: str
    restore_procedure: str
    retained_paths: tuple[str, ...]
    source: Path


@dataclass(frozen=True)
class LegacyRetirementAuditReport:
    changed_paths: int
    entrypoint_repetitions: int
    retained_routes: int
    errors: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.errors


class _Parser:
    def __init__(self, source: Path) -> None:
        self.source = source
        self.errors: list[str] = []

    def mapping(self, value: object, path: str, keys: Sequence[str]) -> dict[str, Any]:
        if not isinstance(value, dict):
            self.errors.append(f"{path}: expected mapping")
            return {}
        expected = set(keys)
        actual = set(value)
        for key in sorted(expected - actual):
            self.errors.append(f"{path}: missing key `{key}`")
        for key in sorted(actual - expected, key=str):
            self.errors.append(f"{path}: unknown key `{key}`")
        return value

    def string(self, value: object, path: str) -> str:
        if not isinstance(value, str) or not value.strip():
            self.errors.append(f"{path}: expected non-empty string")
            return ""
        if "${" in value:
            self.errors.append(f"{path}: interpolation is not allowed")
        return value

    def integer(self, value: object, path: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            self.errors.append(f"{path}: expected integer")
            return -1
        return value

    def strings(self, value: object, path: str) -> tuple[str, ...]:
        if not isinstance(value, list):
            self.errors.append(f"{path}: expected list")
            return ()
        result = tuple(self.string(item, f"{path}[{index}]") for index, item in enumerate(value))
        if len(set(result)) != len(result):
            self.errors.append(f"{path}: duplicate entries are not allowed")
        return result

    def paths(self, value: object, path: str) -> tuple[Path, ...]:
        return tuple(Path(item) for item in self.strings(value, path))

    def finish(self) -> None:
        if self.errors:
            raise LegacyRetirementError(
                f"invalid Issue #705 legacy retirement data {self.source}:\n"
                + "\n".join(f"- {error}" for error in self.errors)
            )


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        raw = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    except Exception as exc:
        raise LegacyRetirementError(f"cannot load {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise LegacyRetirementError(f"{path}: root must be a mapping")
    if not all(isinstance(key, str) for key in raw):
        raise LegacyRetirementError(f"{path}: root keys must be strings")
    return cast(dict[str, Any], raw)


def load_legacy_retirement_plan(path: Path) -> LegacyRetirementPlan:
    raw = _load_yaml(path)
    parser = _Parser(path)
    root = parser.mapping(
        raw,
        "root",
        (
            "schema_version",
            "issue",
            "child_issue",
            "claim_id",
            "plan_fingerprint",
            "integration_base",
            "retired_route",
            "retained_routes",
            "entrypoint_prerequisite",
            "diagnostic_contract",
            "rollback_receipt",
            "evidence_artifact",
            "implementation_paths",
        ),
    )
    base = parser.mapping(
        root.get("integration_base"),
        "integration_base",
        ("commit", "entrypoint_pr", "entrypoint_merge_commit"),
    )
    retired = parser.mapping(
        root.get("retired_route"),
        "retired_route",
        (
            "env_name",
            "backend",
            "owner_module",
            "previous_owner_class",
            "replacement_owner_class",
            "replacement_base_class",
            "runtime_factory",
            "prohibited_operations",
        ),
    )
    retained_raw = root.get("retained_routes")
    if not isinstance(retained_raw, list):
        parser.errors.append("retained_routes: expected list")
        retained_raw = []
    retained_routes: list[RetainedRoute] = []
    for index, item in enumerate(retained_raw):
        value = parser.mapping(
            item,
            f"retained_routes[{index}]",
            ("route_id", "env_name", "backend", "owner_class", "reason"),
        )
        retained_routes.append(
            RetainedRoute(
                route_id=parser.string(value.get("route_id"), f"retained_routes[{index}].route_id"),
                env_name=parser.string(value.get("env_name"), f"retained_routes[{index}].env_name"),
                backend=parser.string(value.get("backend"), f"retained_routes[{index}].backend"),
                owner_class=parser.string(
                    value.get("owner_class"), f"retained_routes[{index}].owner_class"
                ),
                reason=parser.string(value.get("reason"), f"retained_routes[{index}].reason"),
            )
        )
    entrypoint = parser.mapping(
        root.get("entrypoint_prerequisite"),
        "entrypoint_prerequisite",
        (
            "contract_fingerprint",
            "owner_yaml",
            "required_test_id",
            "repetitions",
            "required_routes",
        ),
    )
    diagnostic = parser.mapping(
        root.get("diagnostic_contract"),
        "diagnostic_contract",
        ("exception_class", "required_fragments"),
    )
    plan = LegacyRetirementPlan(
        schema_version=parser.integer(root.get("schema_version"), "schema_version"),
        issue=parser.integer(root.get("issue"), "issue"),
        child_issue=parser.integer(root.get("child_issue"), "child_issue"),
        claim_id=parser.string(root.get("claim_id"), "claim_id"),
        plan_fingerprint=parser.string(root.get("plan_fingerprint"), "plan_fingerprint"),
        integration_base=IntegrationBase(
            commit=parser.string(base.get("commit"), "integration_base.commit"),
            entrypoint_pr=parser.integer(
                base.get("entrypoint_pr"), "integration_base.entrypoint_pr"
            ),
            entrypoint_merge_commit=parser.string(
                base.get("entrypoint_merge_commit"),
                "integration_base.entrypoint_merge_commit",
            ),
        ),
        retired_route=RetiredRoute(
            env_name=parser.string(retired.get("env_name"), "retired_route.env_name"),
            backend=parser.string(retired.get("backend"), "retired_route.backend"),
            owner_module=Path(
                parser.string(retired.get("owner_module"), "retired_route.owner_module")
            ),
            previous_owner_class=parser.string(
                retired.get("previous_owner_class"), "retired_route.previous_owner_class"
            ),
            replacement_owner_class=parser.string(
                retired.get("replacement_owner_class"),
                "retired_route.replacement_owner_class",
            ),
            replacement_base_class=parser.string(
                retired.get("replacement_base_class"),
                "retired_route.replacement_base_class",
            ),
            runtime_factory=parser.string(
                retired.get("runtime_factory"), "retired_route.runtime_factory"
            ),
            prohibited_operations=parser.strings(
                retired.get("prohibited_operations"),
                "retired_route.prohibited_operations",
            ),
        ),
        retained_routes=tuple(retained_routes),
        entrypoint_prerequisite=EntrypointPrerequisite(
            contract_fingerprint=parser.string(
                entrypoint.get("contract_fingerprint"),
                "entrypoint_prerequisite.contract_fingerprint",
            ),
            owner_yaml=Path(
                parser.string(entrypoint.get("owner_yaml"), "entrypoint_prerequisite.owner_yaml")
            ),
            required_test_id=parser.string(
                entrypoint.get("required_test_id"),
                "entrypoint_prerequisite.required_test_id",
            ),
            repetitions=parser.integer(
                entrypoint.get("repetitions"), "entrypoint_prerequisite.repetitions"
            ),
            required_routes=parser.strings(
                entrypoint.get("required_routes"), "entrypoint_prerequisite.required_routes"
            ),
        ),
        diagnostic_contract=DiagnosticContract(
            exception_class=parser.string(
                diagnostic.get("exception_class"), "diagnostic_contract.exception_class"
            ),
            required_fragments=parser.strings(
                diagnostic.get("required_fragments"),
                "diagnostic_contract.required_fragments",
            ),
        ),
        rollback_receipt=Path(parser.string(root.get("rollback_receipt"), "rollback_receipt")),
        evidence_artifact=Path(parser.string(root.get("evidence_artifact"), "evidence_artifact")),
        implementation_paths=parser.paths(root.get("implementation_paths"), "implementation_paths"),
        source=path,
    )
    parser.finish()
    return plan


def load_rollback_receipt(path: Path) -> RollbackReceipt:
    raw = _load_yaml(path)
    parser = _Parser(path)
    root = parser.mapping(
        raw,
        "root",
        (
            "schema_version",
            "artifact_kind",
            "issue",
            "child_issue",
            "plan_fingerprint",
            "baseline",
            "restore",
            "retained_during_retirement",
        ),
    )
    baseline = parser.mapping(root.get("baseline"), "baseline", ("commit", "files"))
    files_raw = baseline.get("files")
    if not isinstance(files_raw, dict):
        parser.errors.append("baseline.files: expected mapping")
        files_raw = {}
    baseline_files = tuple(
        (
            Path(parser.string(key, f"baseline.files key {index}")),
            parser.string(value, f"baseline.files[{key}]"),
        )
        for index, (key, value) in enumerate(files_raw.items())
    )
    restore = parser.mapping(
        root.get("restore"),
        "restore",
        ("env_name", "backend", "owner_class", "registration", "procedure"),
    )
    receipt = RollbackReceipt(
        schema_version=parser.integer(root.get("schema_version"), "schema_version"),
        artifact_kind=parser.string(root.get("artifact_kind"), "artifact_kind"),
        issue=parser.integer(root.get("issue"), "issue"),
        child_issue=parser.integer(root.get("child_issue"), "child_issue"),
        plan_fingerprint=parser.string(root.get("plan_fingerprint"), "plan_fingerprint"),
        baseline_commit=parser.string(baseline.get("commit"), "baseline.commit"),
        baseline_files=baseline_files,
        restore_env_name=parser.string(restore.get("env_name"), "restore.env_name"),
        restore_backend=parser.string(restore.get("backend"), "restore.backend"),
        restore_owner_class=parser.string(restore.get("owner_class"), "restore.owner_class"),
        restore_registration=parser.string(restore.get("registration"), "restore.registration"),
        restore_procedure=parser.string(restore.get("procedure"), "restore.procedure"),
        retained_paths=parser.strings(
            root.get("retained_during_retirement"), "retained_during_retirement"
        ),
        source=path,
    )
    parser.finish()
    return receipt


def sha256_bytes(data: bytes) -> str:
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def _git(root: Path, args: Sequence[str], *, check: bool = True) -> str:
    result = subprocess.run(("git", *args), cwd=root, capture_output=True, text=True, check=False)
    if check and result.returncode != 0:
        raise LegacyRetirementError(
            f"git {' '.join(args)} failed with exit {result.returncode}: {result.stderr.strip()}"
        )
    return result.stdout.strip()


def _is_ancestor(root: Path, ancestor: str, descendant: str) -> bool:
    return (
        subprocess.run(
            ("git", "merge-base", "--is-ancestor", ancestor, descendant),
            cwd=root,
            capture_output=True,
            check=False,
        ).returncode
        == 0
    )


def _git_blob(root: Path, commit: str, path: Path) -> bytes:
    result = subprocess.run(
        ("git", "show", f"{commit}:{path.as_posix()}"),
        cwd=root,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise LegacyRetirementError(
            f"cannot read rollback baseline {commit}:{path}: "
            + result.stderr.decode(errors="replace").strip()
        )
    return result.stdout


def _registry_bindings(source: str) -> dict[tuple[str, str], str]:
    tree = ast.parse(source)
    bindings: dict[tuple[str, str], str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (
            isinstance(func, ast.Attribute)
            and func.attr == "register_env"
            and isinstance(func.value, ast.Name)
            and func.value.id == "registry"
        ):
            continue
        if len(node.args) < 2:
            continue
        env_node, owner_node = node.args[:2]
        backend_node = next(
            (keyword.value for keyword in node.keywords if keyword.arg == "sim_backend"), None
        )
        if not (
            isinstance(env_node, ast.Constant)
            and isinstance(env_node.value, str)
            and isinstance(owner_node, ast.Name)
            and isinstance(backend_node, ast.Constant)
            and isinstance(backend_node.value, str)
        ):
            continue
        key = (env_node.value, backend_node.value)
        if key in bindings:
            raise LegacyRetirementError(f"duplicate registry binding for {key!r}")
        bindings[key] = owner_node.id
    return bindings


def _class_contract_errors(source: str, plan: LegacyRetirementPlan) -> list[str]:
    errors: list[str] = []
    tree = ast.parse(source)
    classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
    if plan.retired_route.previous_owner_class not in classes:
        errors.append("hand-written G1 owner required by retained backends is missing")
    exception = classes.get(plan.diagnostic_contract.exception_class)
    if exception is None:
        errors.append("managed-only diagnostic exception class is missing")
    else:
        exception_bases = tuple(
            base.id if isinstance(base, ast.Name) else ast.unparse(base) for base in exception.bases
        )
        if exception_bases != ("RuntimeError",):
            errors.append("managed-only diagnostic exception must derive from RuntimeError")
    replacement = classes.get(plan.retired_route.replacement_owner_class)
    if replacement is None:
        return ["replacement managed-only env class is missing"]
    bases = tuple(
        base.id if isinstance(base, ast.Name) else ast.unparse(base) for base in replacement.bases
    )
    if bases != (plan.retired_route.replacement_base_class,):
        errors.append(
            "replacement class base must be exactly "
            f"{plan.retired_route.replacement_base_class!r}, got {bases!r}"
        )
    methods = {node.name: node for node in replacement.body if isinstance(node, ast.FunctionDef)}
    rejected_methods = {
        "init_state": "init_state",
        "reset": "reset",
        "step": "step",
        "apply_action": "apply_action",
        "update_state": "update_state",
        "resolve_play_render_plan": "play",
        "run_playback": "play",
    }
    expected_methods = set(rejected_methods)
    missing = sorted(expected_methods - set(methods))
    if missing:
        errors.append(f"replacement class does not override retired operations: {missing!r}")
    if plan.retired_route.runtime_factory not in methods:
        errors.append("replacement class has no managed runtime factory")
    reject_helper = methods.get("_reject_legacy_lifecycle")
    if reject_helper is None or not any(
        isinstance(node, ast.Raise)
        and isinstance(node.exc, ast.Call)
        and isinstance(node.exc.func, ast.Name)
        and node.exc.func.id == plan.diagnostic_contract.exception_class
        for node in ast.walk(reject_helper)
        if reject_helper is not None
    ):
        errors.append("managed-only rejection helper does not raise the typed diagnostic")
    for method_name, operation in rejected_methods.items():
        method = methods.get(method_name)
        if method is None:
            continue
        calls = [node for node in ast.walk(method) if isinstance(node, ast.Call)]
        reject_calls = [
            node
            for node in calls
            if isinstance(node.func, ast.Attribute)
            and node.func.attr == "_reject_legacy_lifecycle"
            and len(node.args) == 1
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == operation
        ]
        if len(reject_calls) != 1 or len(calls) != 1:
            errors.append(
                f"replacement method {method_name!r} must only call the typed "
                f"legacy rejection helper for {operation!r}"
            )
    forbidden_names = {
        node.id
        for node in ast.walk(replacement)
        if isinstance(node, ast.Name) and node.id in {"G1WalkEnv", "G1BaseEnv"}
    }
    if forbidden_names:
        errors.append(
            "replacement class references hand-written task lifecycle owners: "
            f"{sorted(forbidden_names)!r}"
        )
    factory = methods.get(plan.retired_route.runtime_factory)
    if factory is not None and not any(
        isinstance(node, ast.Call)
        and (
            (isinstance(node.func, ast.Name) and node.func.id == "create_g1_managed_device_runtime")
            or (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "create_g1_managed_device_runtime"
            )
        )
        for node in ast.walk(factory)
    ):
        errors.append("managed runtime factory does not call create_g1_managed_device_runtime")
    for fragment in plan.diagnostic_contract.required_fragments:
        if fragment not in source:
            errors.append(f"managed-only diagnostic is missing fragment {fragment!r}")
    return errors


def _plan_errors(plan: LegacyRetirementPlan) -> list[str]:
    errors: list[str] = []
    expected_scalars = {
        "schema_version": SCHEMA_VERSION,
        "issue": ISSUE,
        "child_issue": CHILD_ISSUE,
        "claim_id": CLAIM_ID,
        "plan_fingerprint": PLAN_FINGERPRINT,
    }
    for name, expected in expected_scalars.items():
        if getattr(plan, name) != expected:
            errors.append(f"plan.{name}: expected {expected!r}, got {getattr(plan, name)!r}")
    expected_base = (
        _INTEGRATION_BASE_COMMIT,
        _ENTRYPOINT_PR,
        _INTEGRATION_BASE_COMMIT,
    )
    actual_base = (
        plan.integration_base.commit,
        plan.integration_base.entrypoint_pr,
        plan.integration_base.entrypoint_merge_commit,
    )
    if actual_base != expected_base:
        errors.append("plan.integration_base differs from the frozen entrypoint merge")
    route = plan.retired_route
    expected_route = (
        "G1WalkFlat",
        "mjwarp",
        OWNER_MODULE_PATH,
        "G1WalkEnv",
        "G1MjwarpManagedEnv",
        "NpEnv",
        "create_device_managed_runtime",
        _EXPECTED_PROHIBITED,
    )
    actual_route = (
        route.env_name,
        route.backend,
        route.owner_module,
        route.previous_owner_class,
        route.replacement_owner_class,
        route.replacement_base_class,
        route.runtime_factory,
        route.prohibited_operations,
    )
    if actual_route != expected_route:
        errors.append("plan.retired_route differs from the frozen mjwarp retirement boundary")
    retained = tuple(
        (item.route_id, item.backend, item.owner_class) for item in plan.retained_routes
    )
    if retained != _EXPECTED_RETAINED or any(
        item.env_name != "G1WalkFlat" or not item.reason for item in plan.retained_routes
    ):
        errors.append("plan.retained_routes differs from the frozen owner/oracle set")
    entrypoint = plan.entrypoint_prerequisite
    if (
        entrypoint.contract_fingerprint != "issue705-entrypoints-v1"
        or entrypoint.owner_yaml != OWNER_PATH
        or entrypoint.required_test_id != ENTRYPOINT_TEST_ID
        or entrypoint.repetitions != 2
        or entrypoint.required_routes != _EXPECTED_ROUTES
    ):
        errors.append("plan.entrypoint_prerequisite differs from the frozen full matrix")
    if (
        plan.diagnostic_contract.exception_class != "G1MjwarpManagedOnlyError"
        or plan.diagnostic_contract.required_fragments != _EXPECTED_DIAGNOSTICS
    ):
        errors.append("plan.diagnostic_contract differs from the actionable-error snapshot")
    if plan.rollback_receipt != ROLLBACK_PATH or plan.evidence_artifact != EVIDENCE_PATH:
        errors.append("plan evidence paths are not canonical")
    if plan.implementation_paths != _EXPECTED_IMPLEMENTATION_PATHS:
        errors.append("plan.implementation_paths differs from the frozen retirement scope")
    return errors


def _rollback_errors(
    plan: LegacyRetirementPlan, receipt: RollbackReceipt, *, root: Path
) -> list[str]:
    errors: list[str] = []
    if (
        receipt.schema_version != SCHEMA_VERSION
        or receipt.artifact_kind != _ROLLBACK_KIND
        or receipt.issue != ISSUE
        or receipt.child_issue != CHILD_ISSUE
        or receipt.plan_fingerprint != PLAN_FINGERPRINT
    ):
        errors.append("rollback receipt identity differs from the frozen contract")
    if receipt.baseline_commit != plan.integration_base.commit:
        errors.append("rollback baseline commit differs from the integration base")
    if not _COMMIT_RE.fullmatch(receipt.baseline_commit):
        errors.append("rollback baseline commit is not a full Git SHA")
        return errors
    if receipt.baseline_files != _EXPECTED_BASELINE_FILES:
        errors.append("rollback baseline file/hash set differs from the frozen receipt")
    for path, expected_hash in receipt.baseline_files:
        if not _SHA256_RE.fullmatch(expected_hash):
            errors.append(f"rollback baseline hash is malformed for {path}")
            continue
        try:
            actual_hash = sha256_bytes(_git_blob(root, receipt.baseline_commit, path))
        except LegacyRetirementError as exc:
            errors.append(str(exc))
            continue
        if actual_hash != expected_hash:
            errors.append(f"rollback baseline hash differs for {path}")
    expected_restore = (
        "G1WalkFlat",
        "mjwarp",
        "G1WalkEnv",
        'registry.register_env("G1WalkFlat", G1WalkEnv, sim_backend="mjwarp")',
    )
    if (
        receipt.restore_env_name,
        receipt.restore_backend,
        receipt.restore_owner_class,
        receipt.restore_registration,
    ) != expected_restore:
        errors.append("rollback restore operation differs from the pre-retirement route")
    if "Revert" not in receipt.restore_procedure or "entrypoint matrix" not in (
        receipt.restore_procedure
    ):
        errors.append("rollback procedure must require revert plus full entrypoint validation")
    required_retained = {
        "src/unilab/envs/locomotion/g1/joystick.py::G1WalkEnv",
        "src/unilab/envs/locomotion/g1/managed_reference.py",
        "src/unilab/envs/locomotion/g1/managed_fused.py",
        "tests/manager/test_g1_reference_differential.py",
        "tests/benchmark/test_managed_g1_host_benchmark.py",
    }
    if set(receipt.retained_paths) != required_retained:
        errors.append("rollback retained path set differs from owner and differential obligations")
    for item in receipt.retained_paths:
        path = Path(item.split("::", 1)[0])
        if not (root / path).is_file():
            errors.append(f"retained rollback path is missing: {path}")
    return errors


def _source_errors(
    plan: LegacyRetirementPlan,
    receipt: RollbackReceipt,
    evidence: Mapping[str, Any],
    *,
    root: Path,
) -> tuple[list[str], int]:
    errors: list[str] = []
    source = evidence.get("source")
    if not isinstance(source, Mapping):
        return ["evidence.source: expected mapping"], 0
    source_commit = source.get("commit_sha")
    if not isinstance(source_commit, str) or not _COMMIT_RE.fullmatch(source_commit):
        return ["evidence.source.commit_sha: expected full Git SHA"], 0
    if (
        subprocess.run(
            ("git", "cat-file", "-e", f"{source_commit}^{{commit}}"),
            cwd=root,
            capture_output=True,
            check=False,
        ).returncode
        != 0
    ):
        return ["evidence.source.commit_sha does not identify a Git commit"], 0
    head = _git(root, ("rev-parse", "HEAD"))
    if not _is_ancestor(root, receipt.baseline_commit, source_commit):
        errors.append("integration base is not an ancestor of the evidence source commit")
    if not _is_ancestor(root, source_commit, head):
        errors.append("evidence source commit is not an ancestor of HEAD")
    changed = tuple(
        sorted(
            line
            for line in _git(
                root,
                ("diff", "--name-only", f"{receipt.baseline_commit}..{source_commit}"),
            ).splitlines()
            if line
        )
    )
    expected_changed = tuple(path.as_posix() for path in plan.implementation_paths)
    if changed != expected_changed:
        errors.append(
            "implementation diff paths differ from the frozen retirement scope: "
            f"expected={expected_changed!r}, got={changed!r}"
        )
    try:
        baseline_source = _git_blob(root, receipt.baseline_commit, plan.retired_route.owner_module)
        source_blob = _git_blob(root, source_commit, plan.retired_route.owner_module)
        baseline_bindings = _registry_bindings(baseline_source.decode())
        current_bindings = _registry_bindings(source_blob.decode())
    except (LegacyRetirementError, SyntaxError, UnicodeDecodeError) as exc:
        errors.append(f"cannot inspect registry deletion diff: {exc}")
        return errors, len(changed)
    key = (plan.retired_route.env_name, plan.retired_route.backend)
    if baseline_bindings.get(key) != plan.retired_route.previous_owner_class:
        errors.append("integration baseline does not contain the declared legacy owner route")
    if current_bindings.get(key) != plan.retired_route.replacement_owner_class:
        errors.append("source commit does not bind mjwarp to the managed-only replacement")
    for retained in plan.retained_routes[:2]:
        retained_key = (retained.env_name, retained.backend)
        if current_bindings.get(retained_key) != retained.owner_class:
            errors.append(f"retained owner binding changed for {retained_key!r}")
    errors.extend(_class_contract_errors(source_blob.decode(), plan))
    return errors, len(changed)


def _evidence_errors(
    plan: LegacyRetirementPlan,
    evidence: Mapping[str, Any],
    *,
    root: Path,
) -> tuple[list[str], int]:
    errors: list[str] = []
    expected_root_keys = {
        "schema_version",
        "artifact_kind",
        "issue",
        "child_issue",
        "claim_id",
        "plan_fingerprint",
        "source",
        "inputs",
        "environment",
        "commands",
        "summary",
    }
    if set(evidence) != expected_root_keys:
        errors.append("evidence root keys do not exactly match the registered schema")
    expected_identity = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": _EVIDENCE_KIND,
        "issue": ISSUE,
        "child_issue": CHILD_ISSUE,
        "claim_id": CLAIM_ID,
        "plan_fingerprint": PLAN_FINGERPRINT,
    }
    for key, expected in expected_identity.items():
        if evidence.get(key) != expected:
            errors.append(f"evidence.{key}: expected {expected!r}")
    source = evidence.get("source")
    if not isinstance(source, Mapping):
        errors.append("evidence.source: expected mapping")
        source = {}
    if set(source) != {"commit_sha", "git_status"}:
        errors.append("evidence.source keys do not exactly match the registered schema")
    if source.get("git_status") != "":
        errors.append("evidence.source.git_status: capture source was not clean")
    inputs = evidence.get("inputs")
    if not isinstance(inputs, Mapping):
        errors.append("evidence.inputs: expected mapping")
        inputs = {}
    expected_inputs = {path.as_posix() for path in _INPUT_PATHS}
    if set(inputs) != expected_inputs:
        errors.append("evidence.inputs do not exactly match the registered input set")
    for path in _INPUT_PATHS:
        expected = inputs.get(path.as_posix())
        if not isinstance(expected, str) or not _SHA256_RE.fullmatch(expected):
            errors.append(f"evidence input hash is malformed for {path}")
        elif not (root / path).is_file():
            errors.append(f"evidence input is missing: {path}")
        elif sha256_file(root / path) != expected:
            errors.append(f"evidence input is stale: {path}")
        else:
            source_commit = source.get("commit_sha")
            if isinstance(source_commit, str) and _COMMIT_RE.fullmatch(source_commit):
                try:
                    source_hash = sha256_bytes(_git_blob(root, source_commit, path))
                except LegacyRetirementError as exc:
                    errors.append(str(exc))
                else:
                    if source_hash != expected:
                        errors.append(f"evidence input does not match source commit: {path}")
    environment = evidence.get("environment")
    if not isinstance(environment, Mapping):
        errors.append("evidence.environment: expected mapping")
        environment = {}
    if set(environment) != {"platform", "python"}:
        errors.append("evidence.environment keys do not exactly match the registered schema")
    for key in ("platform", "python"):
        value = environment.get(key)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"evidence.environment.{key}: expected non-empty string")
    commands = evidence.get("commands")
    if not isinstance(commands, list):
        errors.append("evidence.commands: expected list")
        commands = []
    if len(commands) != plan.entrypoint_prerequisite.repetitions:
        errors.append("evidence command count differs from frozen repetitions")
    for index, command in enumerate(commands, start=1):
        label = f"evidence.commands[{index - 1}]"
        if not isinstance(command, Mapping):
            errors.append(f"{label}: expected mapping")
            continue
        if set(command) != {
            "repetition",
            "argv",
            "required_test_id",
            "exit_code",
            "duration_sec",
            "pytest",
            "stdout",
            "stderr",
        }:
            errors.append(f"{label}: keys do not exactly match the registered schema")
        if command.get("repetition") != index:
            errors.append(f"{label}.repetition: expected {index}")
        if command.get("argv") != list(_ENTRYPOINT_ARGV):
            errors.append(f"{label}.argv differs from the frozen entrypoint command")
        if command.get("required_test_id") != ENTRYPOINT_TEST_ID:
            errors.append(f"{label}.required_test_id differs from the full matrix node")
        if command.get("exit_code") != 0:
            errors.append(f"{label}.exit_code: expected 0")
        duration = command.get("duration_sec")
        if not isinstance(duration, (int, float)) or isinstance(duration, bool) or duration <= 0:
            errors.append(f"{label}.duration_sec: expected positive number")
        counts = command.get("pytest")
        if counts != {
            "passed": 1,
            "skipped": 0,
            "xfailed": 0,
            "xpassed": 0,
            "deselected": 0,
        }:
            errors.append(f"{label}.pytest: expected one pass and no non-pass outcome")
        stdout = command.get("stdout")
        stderr = command.get("stderr")
        if not isinstance(stdout, str) or ENTRYPOINT_TEST_ID not in stdout:
            errors.append(f"{label}.stdout does not identify the mandatory entrypoint test")
        if not isinstance(stderr, str):
            errors.append(f"{label}.stderr: expected string")
    summary = evidence.get("summary")
    if summary != {
        "repetitions": 2,
        "passed": 2,
        "skipped": 0,
        "xfailed": 0,
        "xpassed": 0,
        "deselected": 0,
    }:
        errors.append("evidence.summary differs from command receipts")
    return errors, len(commands)


def _repository_contract_errors(plan: LegacyRetirementPlan, *, root: Path) -> list[str]:
    errors: list[str] = []
    owner_raw = OmegaConf.to_container(OmegaConf.load(root / OWNER_PATH), resolve=True)
    if not isinstance(owner_raw, dict):
        return ["mjwarp owner YAML root is not a mapping"]
    entrypoints = owner_raw.get("entrypoints")
    if not isinstance(entrypoints, dict):
        return ["mjwarp owner entrypoints must be a mapping"]
    identity = entrypoints.get("identity")
    routes = entrypoints.get("routes")
    if not isinstance(identity, dict):
        errors.append("mjwarp owner entrypoint identity must be a mapping")
        identity = {}
    if not isinstance(routes, dict):
        errors.append("mjwarp owner entrypoint routes must be a mapping")
        routes = {}
    if identity != {
        "task_name": "G1WalkFlat",
        "backend": "mjwarp",
        "execution_profile": "device_resident",
        "runtime_impl": "mjwarp_device_v1",
        "runtime_resolver": "unilab.training.rsl_rl_device:resolve_mjwarp_device_ppo_runtime",
    }:
        errors.append("mjwarp owner entrypoint identity differs from the managed-only route")
    if tuple(routes) != _EXPECTED_ROUTES:
        errors.append("mjwarp owner route order/set differs from the complete entrypoint matrix")
    if routes.get("train") != "native" or routes.get("export") != "native":
        errors.append("mjwarp owner must retain native train and export routes")
    if routes.get("play") != "unsupported" or routes.get("visualize") != "unsupported":
        errors.append("mjwarp owner must fail closed for native play and visualization")
    inventory = load_claim_gap_inventory(root / CLAIM_INVENTORY_PATH)
    by_test = {entry.test_id: entry for entry in inventory.entries}
    legacy_entry = by_test.get(LEGACY_TEST_ID)
    if legacy_entry is None or legacy_entry.claim_id != CLAIM_ID:
        errors.append("claim inventory does not map the legacy retirement acceptance test")
    elif legacy_entry.state is not InventoryTestState.EXISTING:
        errors.append("legacy retirement claim inventory node is not existing")
    entrypoint_entry = by_test.get(ENTRYPOINT_TEST_ID)
    if entrypoint_entry is None or entrypoint_entry.state is not InventoryTestState.EXISTING:
        errors.append("entrypoint prerequisite claim inventory node is not existing")
    return errors


def load_legacy_retirement_evidence(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise LegacyRetirementError(
            f"cannot load legacy retirement evidence {path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise LegacyRetirementError(f"{path}: evidence root must be an object")
    return payload


def audit_legacy_retirement(
    plan: LegacyRetirementPlan,
    rollback: RollbackReceipt,
    evidence: Mapping[str, Any],
    *,
    root: Path,
) -> LegacyRetirementAuditReport:
    root = root.resolve()
    errors = _plan_errors(plan)
    errors.extend(_rollback_errors(plan, rollback, root=root))
    evidence_errors, repetitions = _evidence_errors(plan, evidence, root=root)
    errors.extend(evidence_errors)
    source_errors, changed_paths = _source_errors(plan, rollback, evidence, root=root)
    errors.extend(source_errors)
    errors.extend(_repository_contract_errors(plan, root=root))
    return LegacyRetirementAuditReport(
        changed_paths=changed_paths,
        entrypoint_repetitions=repetitions,
        retained_routes=len(plan.retained_routes),
        errors=tuple(errors),
    )


def _pytest_counts(output: str) -> dict[str, int]:
    counts = {
        "passed": 0,
        "skipped": 0,
        "xfailed": 0,
        "xpassed": 0,
        "deselected": 0,
    }
    for raw_count, category in _PYTEST_COUNT_RE.findall(output):
        counts[category] += int(raw_count)
    return counts


def capture_legacy_retirement_evidence(root: Path) -> dict[str, Any]:
    root = root.resolve()
    status = _git(root, ("status", "--porcelain"))
    if status:
        raise LegacyRetirementError(
            "legacy retirement evidence must be captured from a clean source tree:\n" + status
        )
    source_commit = _git(root, ("rev-parse", "HEAD"))
    if not _COMMIT_RE.fullmatch(source_commit):
        raise LegacyRetirementError("git did not return a full source commit SHA")
    missing = [path.as_posix() for path in _INPUT_PATHS if not (root / path).is_file()]
    if missing:
        raise LegacyRetirementError(f"legacy retirement evidence inputs are missing: {missing!r}")
    commands: list[dict[str, Any]] = []
    for repetition in range(1, 3):
        started = time.perf_counter()
        result = subprocess.run(
            _ENTRYPOINT_ARGV,
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
            timeout=900,
        )
        duration = time.perf_counter() - started
        combined = f"{result.stdout}\n{result.stderr}"
        counts = _pytest_counts(combined)
        failures: list[str] = []
        if result.returncode != 0:
            failures.append(f"exit={result.returncode}")
        if counts != {
            "passed": 1,
            "skipped": 0,
            "xfailed": 0,
            "xpassed": 0,
            "deselected": 0,
        }:
            failures.append(f"pytest={counts!r}")
        if ENTRYPOINT_TEST_ID not in result.stdout:
            failures.append("mandatory node absent from stdout")
        if failures:
            raise LegacyRetirementError(
                f"entrypoint evidence repetition {repetition} failed ({'; '.join(failures)}):\n"
                f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            )
        commands.append(
            {
                "repetition": repetition,
                "argv": list(_ENTRYPOINT_ARGV),
                "required_test_id": ENTRYPOINT_TEST_ID,
                "exit_code": result.returncode,
                "duration_sec": duration,
                "pytest": counts,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": _EVIDENCE_KIND,
        "issue": ISSUE,
        "child_issue": CHILD_ISSUE,
        "claim_id": CLAIM_ID,
        "plan_fingerprint": PLAN_FINGERPRINT,
        "source": {"commit_sha": source_commit, "git_status": ""},
        "inputs": {path.as_posix(): sha256_file(root / path) for path in _INPUT_PATHS},
        "environment": {
            "platform": platform.platform(),
            "python": sys.version,
        },
        "commands": commands,
        "summary": {
            "repetitions": 2,
            "passed": 2,
            "skipped": 0,
            "xfailed": 0,
            "xpassed": 0,
            "deselected": 0,
        },
    }


def write_legacy_retirement_evidence(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "CLAIM_ID",
    "EVIDENCE_PATH",
    "LegacyRetirementAuditReport",
    "LegacyRetirementError",
    "LegacyRetirementPlan",
    "PLAN_PATH",
    "ROLLBACK_PATH",
    "RollbackReceipt",
    "audit_legacy_retirement",
    "capture_legacy_retirement_evidence",
    "load_legacy_retirement_evidence",
    "load_legacy_retirement_plan",
    "load_rollback_receipt",
    "write_legacy_retirement_evidence",
]
