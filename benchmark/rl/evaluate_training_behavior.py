"""Capture and validate managed MuJoCo/MJWarp rollout's frozen paired-seed training behavior gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tooling.acceptance.training_behavior import (  # noqa: E402
    ARTIFACT_KIND,
    ARTIFACT_PATH,
    BENCHMARK_ID,
    CLAIM_ID,
    FREEZE_RECEIPT_PATH,
    ISSUE,
    PARENT_ISSUE,
    PLAN_PATH,
    SCHEMA_VERSION,
    TrainingBehaviorContractError,
    TrainingBehaviorPlan,
    build_training_behavior_sections,
    load_frozen_training_inputs,
    load_training_behavior_artifact,
    load_training_behavior_freeze_receipt,
    load_training_behavior_plan,
    summarize_training_behavior_raw,
    validate_training_behavior_artifact,
)

from benchmark.mjwarp.process_dr_evidence import (  # noqa: E402
    event_scalars,
    hardware_payload,
    json_safe,
    preflight_payload,
    run_subprocess,
    utc_now,
)
from unilab.tools.g1_baseline_provenance import (  # noqa: E402
    canonical_sha256,
    load_g1_baseline_plan,
    sha256_file,
    source_tree_sha256,
)

DEFAULT_OUTPUT = Path("/tmp/manager_mjwarp_training_behavior.json")


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=ROOT_DIR,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _source_payload(plan: TrainingBehaviorPlan) -> dict[str, Any]:
    if _git("status", "--short"):
        raise TrainingBehaviorContractError(
            PLAN_PATH, ["training behavior capture requires a clean git worktree"]
        )
    commit = _git("rev-parse", "HEAD")
    return {
        "commit": commit,
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": False,
        "tree_sha256": source_tree_sha256(ROOT_DIR, plan.source_inputs),
        "uv_lock_sha256": sha256_file(ROOT_DIR / "uv.lock"),
        "owner_yaml_sha256": sha256_file(ROOT_DIR / cast(str, plan.source_contract["owner_yaml"])),
    }


def _assert_source_unchanged(plan: TrainingBehaviorPlan, source: Mapping[str, Any]) -> None:
    if _git("status", "--short"):
        raise TrainingBehaviorContractError(
            PLAN_PATH, ["training behavior capture modified the candidate worktree"]
        )
    if _git("rev-parse", "HEAD") != source.get("commit"):
        raise TrainingBehaviorContractError(
            PLAN_PATH, ["candidate HEAD changed during training behavior capture"]
        )
    if source_tree_sha256(ROOT_DIR, plan.source_inputs) != source.get("tree_sha256"):
        raise TrainingBehaviorContractError(
            PLAN_PATH, ["registered source inputs changed during behavior capture"]
        )
    if sha256_file(ROOT_DIR / "uv.lock") != source.get("uv_lock_sha256"):
        raise TrainingBehaviorContractError(PLAN_PATH, ["uv.lock changed during capture"])
    if sha256_file(ROOT_DIR / cast(str, plan.source_contract["owner_yaml"])) != source.get(
        "owner_yaml_sha256"
    ):
        raise TrainingBehaviorContractError(PLAN_PATH, ["owner YAML changed during capture"])


def _run_dirs(log_root: Path, env_name: str, backend: str) -> list[Path]:
    return sorted((log_root / env_name).glob(f"*_{backend}"))


def _worker_case(plan: TrainingBehaviorPlan, *, seed: int) -> dict[str, Any]:
    baseline_plan = load_g1_baseline_plan(
        ROOT_DIR / cast(str, plan.source_contract["baseline_plan"])
    )
    measurement = plan.measurement
    case_id = f"behavior-mjwarp_device-seed{seed}"
    base: dict[str, Any] = {
        "case_id": case_id,
        "seed": seed,
        "sequence_index": plan.seeds.index(seed),
        "process_retries": 0,
        "batch_size": measurement["num_envs"],
        "num_steps_per_env": measurement["num_steps_per_env"],
        "iterations": measurement["max_iterations"],
        "mode": "mjwarp_device",
    }
    with tempfile.TemporaryDirectory(prefix=f"manager_mjwarp_p7c_seed_{seed}_") as temp_dir:
        log_root = Path(temp_dir) / "logs"
        command = [
            "uv",
            "run",
            "scripts/train_rsl_rl.py",
            f"task={measurement['task_slug']}/{measurement['candidate_backend']}",
            f"algo.seed={seed}",
            f"algo.num_envs={measurement['num_envs']}",
            f"algo.num_steps_per_env={measurement['num_steps_per_env']}",
            f"algo.max_iterations={measurement['max_iterations']}",
            f"algo.save_interval={measurement['save_interval']}",
            "algo.capture_performance_diagnostics=true",
            "training.no_play=true",
            "training.logger=tensorboard",
            f"training.log_root={log_root}",
            *cast(Sequence[str], measurement["hydra_overrides"]),
        ]
        process, memory_samples, stdout, stderr = run_subprocess(
            command,
            baseline_plan,
            repo_root=ROOT_DIR,
            memory_poll_interval=float(measurement["memory_poll_interval_sec"]),
        )
        base["process"] = process
        if process["return_code"] != 0:
            base["collection_error"] = {
                "kind": "training_process_failed",
                "stdout_tail_sha256": (
                    "sha256:" + hashlib.sha256(stdout[-6000:].encode()).hexdigest()
                ),
                "stderr_tail_sha256": (
                    "sha256:" + hashlib.sha256(stderr[-6000:].encode()).hexdigest()
                ),
            }
            return base
        try:
            run_dirs = _run_dirs(
                log_root,
                cast(str, measurement["env_name"]),
                cast(str, measurement["candidate_backend"]),
            )
            if len(run_dirs) != 1:
                raise ValueError(f"expected one run directory, got {len(run_dirs)}")
            run_dir = run_dirs[0]
            run_config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
            run_summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
            raw = {
                "scalars": event_scalars(
                    run_dir, cast(Sequence[str], measurement["required_scalar_tags"])
                ),
                "memory_samples": memory_samples,
                "run_config": run_config,
                "run_config_sha256": canonical_sha256(run_config),
                "run_summary": run_summary,
            }
            base["raw"] = raw
            base["summary"] = summarize_training_behavior_raw(
                raw, plan, label=f"candidate/seed={seed}"
            )
        except Exception as exc:  # noqa: BLE001 - retain a failed seed receipt in the artifact
            base["collection_error"] = {
                "kind": "run_artifact_collection_failed",
                "error_type": type(exc).__name__,
                "message": str(exc),
            }
    return base


def _execute_worker(plan: TrainingBehaviorPlan, *, seed: int, output: Path) -> int:
    if seed not in plan.seeds:
        raise TrainingBehaviorContractError(PLAN_PATH, [f"seed {seed} is not frozen"])
    case = _worker_case(plan, seed=seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(json_safe(case), sort_keys=True), encoding="utf-8")
    return 0


def _run_seed_worker(plan: TrainingBehaviorPlan, *, seed: int) -> dict[str, Any]:
    baseline_plan = load_g1_baseline_plan(
        ROOT_DIR / cast(str, plan.source_contract["baseline_plan"])
    )
    case_id = f"behavior-mjwarp_device-seed{seed}"
    with tempfile.TemporaryDirectory(prefix=f"manager_mjwarp_p7c_worker_{seed}_") as temp_dir:
        output = Path(temp_dir) / "case.json"
        command = [
            "uv",
            "run",
            "benchmark/rl/evaluate_training_behavior.py",
            "--worker",
            "--seed",
            str(seed),
            "--worker-out",
            str(output),
        ]
        process, _, stdout, stderr = run_subprocess(command, baseline_plan, repo_root=ROOT_DIR)
        if process["return_code"] != 0 or not output.is_file():
            return {
                "case_id": case_id,
                "seed": seed,
                "sequence_index": plan.seeds.index(seed),
                "process_retries": 0,
                "batch_size": plan.measurement["num_envs"],
                "num_steps_per_env": plan.measurement["num_steps_per_env"],
                "iterations": plan.measurement["max_iterations"],
                "mode": "mjwarp_device",
                "worker_process": process,
                "collection_error": {
                    "kind": "worker_process_failed",
                    "stdout_tail_sha256": (
                        "sha256:" + hashlib.sha256(stdout[-6000:].encode()).hexdigest()
                    ),
                    "stderr_tail_sha256": (
                        "sha256:" + hashlib.sha256(stderr[-6000:].encode()).hexdigest()
                    ),
                },
            }
        try:
            case = json.loads(output.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise TrainingBehaviorContractError(
                output, [f"worker output is invalid: {type(exc).__name__}: {exc}"]
            ) from exc
    if not isinstance(case, dict):
        raise TrainingBehaviorContractError(output, ["worker output root must be a mapping"])
    case["worker_process"] = process
    return cast(dict[str, Any], case)


def collect_artifact(*, output: Path) -> dict[str, Any]:
    output = output.resolve()
    if output.is_relative_to(ROOT_DIR):
        raise TrainingBehaviorContractError(
            output, ["raw candidate artifact must be written outside the repository"]
        )
    plan = load_training_behavior_plan(ROOT_DIR / PLAN_PATH, repo_root=ROOT_DIR)
    receipt = load_training_behavior_freeze_receipt(
        ROOT_DIR / FREEZE_RECEIPT_PATH,
        plan=plan,
        repo_root=ROOT_DIR,
    )
    threshold, baseline = load_frozen_training_inputs(plan, repo_root=ROOT_DIR)
    source = _source_payload(plan)
    if source["commit"] == receipt.freeze_commit:
        raise TrainingBehaviorContractError(
            PLAN_PATH, ["candidate commit must differ from the freeze commit"]
        )
    baseline_plan = load_g1_baseline_plan(
        ROOT_DIR / cast(str, plan.source_contract["baseline_plan"])
    )
    preflight_before = preflight_payload(baseline_plan)
    cases: list[dict[str, Any]] = []
    for index, seed in enumerate(plan.seeds, start=1):
        case = _run_seed_worker(plan, seed=seed)
        cases.append(case)
        status = "PASS" if "collection_error" not in case else "FAIL"
        print(f"[{index}/{len(plan.seeds)}] {case['case_id']} {status}", flush=True)
    _assert_source_unchanged(plan, source)
    artifact: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "issue": ISSUE,
        "parent_issue": PARENT_ISSUE,
        "claim_id": CLAIM_ID,
        "kind": ARTIFACT_KIND,
        "generated_at": utc_now(),
        "contract": {
            "plan_path": PLAN_PATH.as_posix(),
            "plan_sha256": sha256_file(ROOT_DIR / PLAN_PATH),
            "freeze_receipt_path": FREEZE_RECEIPT_PATH.as_posix(),
            "freeze_receipt_sha256": sha256_file(ROOT_DIR / FREEZE_RECEIPT_PATH),
            "freeze_commit": receipt.freeze_commit,
            "threshold_manifest_path": cast(str, plan.source_contract["threshold_manifest"]),
            "threshold_manifest_sha256": sha256_file(
                ROOT_DIR / cast(str, plan.source_contract["threshold_manifest"])
            ),
            "baseline_artifact_path": cast(str, plan.source_contract["baseline_artifact"]),
            "baseline_artifact_sha256": threshold["baseline"]["artifact_sha256"],
        },
        "source": source,
        "hardware": hardware_payload(baseline_plan),
        "execution": {
            "process_isolation": True,
            "process_retries": 0,
            "case_order": [case["case_id"] for case in cases],
            "environment_variables": plan.hardware["environment_variables"],
            "preflight_before": preflight_before,
            "preflight_after": preflight_payload(baseline_plan, enforce_cpu_load=False),
        },
        "success_metric": plan.measurement["success_metric"],
        "cases": cases,
        "pairs": [],
        "aggregates": {},
        "gate": {"passed": False, "errors": ["not evaluated"]},
    }
    pairs, aggregates, gate = build_training_behavior_sections(
        artifact,
        plan=plan,
        receipt=receipt,
        threshold=threshold,
        baseline=baseline,
        repo_root=ROOT_DIR,
    )
    artifact["pairs"] = pairs
    artifact["aggregates"] = aggregates
    artifact["gate"] = gate
    report = validate_training_behavior_artifact(
        artifact,
        plan=plan,
        receipt=receipt,
        threshold=threshold,
        baseline=baseline,
        repo_root=ROOT_DIR,
    )
    if tuple(gate["errors"]) != report.errors:
        raise TrainingBehaviorContractError(
            output, ["collector and independent validator errors differ", *report.errors]
        )
    return artifact


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--validate-artifact", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--worker-out", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.worker:
        if args.seed is None or args.worker_out is None:
            raise SystemExit("worker requires --seed and --worker-out")
        plan = load_training_behavior_plan(ROOT_DIR / PLAN_PATH, repo_root=ROOT_DIR)
        return _execute_worker(plan, seed=args.seed, output=args.worker_out)
    if args.execute:
        output = args.out.resolve()
        artifact = collect_artifact(output=output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(json_safe(artifact), indent=2, sort_keys=True), encoding="utf-8"
        )
        if artifact["gate"]["passed"] is not True:
            print(f"FAIL wrote diagnostic artifact to {output}")
            for error in artifact["gate"]["errors"]:
                print(f"- {error}")
            return 2
        print(f"PASS wrote {output}")
        return 0
    artifact_path = (
        args.validate_artifact.resolve()
        if args.validate_artifact is not None
        else ROOT_DIR / ARTIFACT_PATH
    )
    _, report = load_training_behavior_artifact(artifact_path, repo_root=ROOT_DIR)
    if report.ok:
        print(f"PASS {artifact_path}")
        return 0
    print(f"FAIL {artifact_path}")
    for error in report.errors:
        print(f"- {error}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
