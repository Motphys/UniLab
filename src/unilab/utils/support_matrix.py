"""Generate backend support matrix content from registry, configs, and tests."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path

from omegaconf import OmegaConf

from unilab.base import registry
from unilab.base.registry import ensure_registries
from unilab.support.evidence import (
    SUPPORT_EVIDENCE_PATH,
    DeclaredEvidenceLevel,
    SupportEvidenceManifest,
    load_support_evidence,
)

BEGIN_MARKER = "<!-- BEGIN GENERATED SUPPORT MATRIX -->"
END_MARKER = "<!-- END GENERATED SUPPORT MATRIX -->"
BACKENDS: tuple[str, ...] = ("mujoco", "mjwarp", "motrix")

_TASK_ORDER = {
    "go1_joystick_flat": 0,
    "go2_joystick_flat": 1,
    "go2_joystick_rough": 2,
    "g1_walk_flat": 3,
    "g1_walk_rough": 4,
    "g1_motion_tracking": 5,
    "g1_flip_tracking": 6,
    "g1_wall_flip_tracking": 7,
    "x2_wall_flip_tracking": 8,
    "allegro_inhand": 9,
    "allegro_sac": 10,
    "sharpa_inhand": 11,
    "sharpa_inhand_grasp": 12,
}
_TASK_LABELS = {
    "go1_joystick_flat": "Go1 joystick",
    "go2_joystick_flat": "Go2 joystick",
    "go2_joystick_rough": "Go2 joystick rough",
    "g1_walk_flat": "G1 walk flat",
    "g1_walk_rough": "G1 walk rough",
    "g1_motion_tracking": "G1 motion tracking",
    "g1_flip_tracking": "G1 flip tracking",
    "g1_wall_flip_tracking": "G1 wall flip tracking",
    "x2_wall_flip_tracking": "X2 wall flip tracking",
    "allegro_inhand": "Allegro in-hand",
    "allegro_sac": "Allegro SAC in-hand",
    "sharpa_inhand": "Sharpa in-hand",
    "sharpa_inhand_grasp": "Sharpa in-hand grasp",
}


class EvidenceLevel(IntEnum):
    MISSING = 0
    REGISTERED = 1
    CONFIGURED = 2
    TESTED = 3
    BENCHMARKED = 4
    RECOMMENDED = 5

    @property
    def label(self) -> str:
        return {
            EvidenceLevel.MISSING: "-",
            EvidenceLevel.REGISTERED: "Registered",
            EvidenceLevel.CONFIGURED: "Configured",
            EvidenceLevel.TESTED: "Tested",
            EvidenceLevel.BENCHMARKED: "Benchmarked",
            EvidenceLevel.RECOMMENDED: "Recommended",
        }[self]


@dataclass(frozen=True)
class EntrypointSpec:
    entrypoint_id: str
    label: str
    config_dir: str
    task_glob: str
    generic_tested: bool = False


@dataclass(frozen=True)
class SupportCell:
    env_name: str
    level: EvidenceLevel
    execution_profile: str | None = None


@dataclass(frozen=True)
class SupportRow:
    entrypoint_id: str
    entrypoint_label: str
    task_slug: str
    task_label: str
    cells: dict[str, SupportCell]


ENTRYPOINT_SPECS: tuple[EntrypointSpec, ...] = (
    EntrypointSpec(
        entrypoint_id="ppo_torch",
        label="PPO (torch)",
        config_dir="conf/ppo/task",
        task_glob="*/*.yaml",
        generic_tested=True,
    ),
    EntrypointSpec(
        entrypoint_id="ppo_mlx",
        label="PPO (mlx)",
        config_dir="conf/ppo/task",
        task_glob="*/*.yaml",
        generic_tested=False,
    ),
    EntrypointSpec(
        entrypoint_id="appo_torch",
        label="APPO (torch)",
        config_dir="conf/appo/task",
        task_glob="*/*.yaml",
        generic_tested=True,
    ),
    EntrypointSpec(
        entrypoint_id="sac_torch",
        label="SAC (torch)",
        config_dir="conf/offpolicy/task/sac",
        task_glob="*/*.yaml",
        generic_tested=True,
    ),
    EntrypointSpec(
        entrypoint_id="td3_torch",
        label="TD3 (torch)",
        config_dir="conf/offpolicy/task/td3",
        task_glob="*/*.yaml",
        generic_tested=True,
    ),
    EntrypointSpec(
        entrypoint_id="flashsac_torch",
        label="FlashSAC (torch)",
        config_dir="conf/offpolicy/task/flashsac",
        task_glob="*/*.yaml",
        generic_tested=True,
    ),
)


def repo_root(root: Path | None = None) -> Path:
    return root or Path(__file__).resolve().parents[3]


def _task_sort_key(task_slug: str) -> tuple[int, str]:
    return (_TASK_ORDER.get(task_slug, 999), task_slug)


def _task_label(task_slug: str) -> str:
    return _TASK_LABELS.get(task_slug, task_slug.replace("_", " "))


def _load_task_name(task_path: Path) -> str:
    raw = OmegaConf.to_container(OmegaConf.load(task_path), resolve=True) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Expected mapping config in {task_path}")
    training = raw.get("training")
    if not isinstance(training, dict) or "task_name" not in training:
        raise ValueError(f"Missing training.task_name in {task_path}")
    task_name = training["task_name"]
    if not isinstance(task_name, str):
        raise ValueError(f"training.task_name must be a string in {task_path}")
    return task_name


def _load_registry_backends() -> dict[str, set[str]]:
    ensure_registries()
    registered = registry.list_registered_envs()
    return {
        env_name: set(meta["available_backends"])
        for env_name, meta in registered.items()
        if isinstance(meta.get("available_backends"), list)
    }


def _configured_entries(root: Path, spec: EntrypointSpec) -> dict[str, dict[str, str]]:
    task_root = root / spec.config_dir
    entries: dict[str, dict[str, str]] = {}
    for task_path in sorted(task_root.glob(spec.task_glob)):
        task_slug = task_path.parent.name
        backend = task_path.stem
        if backend not in BACKENDS:
            continue
        entries.setdefault(task_slug, {})[backend] = _load_task_name(task_path)
    return entries


def _is_tested(spec: EntrypointSpec, backend: str) -> bool:
    # mjwarp claims are always combination-specific.  Generic compose coverage
    # may establish Configured, but cannot promote an execution profile.
    if backend == "mjwarp":
        return False
    return spec.generic_tested


def _declared_level(level: DeclaredEvidenceLevel) -> EvidenceLevel:
    return {
        DeclaredEvidenceLevel.TESTED: EvidenceLevel.TESTED,
        DeclaredEvidenceLevel.BENCHMARKED: EvidenceLevel.BENCHMARKED,
        DeclaredEvidenceLevel.RECOMMENDED: EvidenceLevel.RECOMMENDED,
    }[level]


def _load_declarations(root: Path) -> SupportEvidenceManifest | None:
    path = root / SUPPORT_EVIDENCE_PATH
    return load_support_evidence(path) if path.is_file() else None


def _cell_level(
    *,
    backend: str,
    env_name: str,
    configured_backends: dict[str, str],
    registry_backends: dict[str, set[str]],
    tested: bool,
    declared: DeclaredEvidenceLevel | None,
) -> EvidenceLevel:
    available_backends = registry_backends.get(env_name, set())
    if backend not in available_backends:
        return EvidenceLevel.MISSING

    level = EvidenceLevel.REGISTERED
    if backend in configured_backends:
        level = EvidenceLevel.CONFIGURED
    if declared is not None and backend in configured_backends:
        return _declared_level(declared)
    if backend in configured_backends and tested:
        level = EvidenceLevel.TESTED
    return level


def build_support_rows(
    root: Path | None = None,
    *,
    support_evidence: SupportEvidenceManifest | None = None,
) -> list[SupportRow]:
    resolved_root = repo_root(root)
    registry_backends = _load_registry_backends()
    evidence = support_evidence or _load_declarations(resolved_root)
    declarations = (
        {combination.key: combination for combination in evidence.combinations}
        if evidence is not None
        else {}
    )
    rows: list[SupportRow] = []

    for spec in ENTRYPOINT_SPECS:
        for task_slug, configured_backends in sorted(
            _configured_entries(resolved_root, spec).items(),
            key=lambda item: _task_sort_key(item[0]),
        ):
            env_name = next(iter(configured_backends.values()))
            cells = {
                backend: SupportCell(
                    env_name=env_name,
                    level=_cell_level(
                        backend=backend,
                        env_name=env_name,
                        configured_backends=configured_backends,
                        registry_backends=registry_backends,
                        tested=_is_tested(spec, backend),
                        declared=(
                            declarations[(spec.entrypoint_id, task_slug, backend)].evidence_level
                            if (spec.entrypoint_id, task_slug, backend) in declarations
                            else None
                        ),
                    ),
                    execution_profile=(
                        declarations[(spec.entrypoint_id, task_slug, backend)].execution_profile
                        if (spec.entrypoint_id, task_slug, backend) in declarations
                        else None
                    ),
                )
                for backend in BACKENDS
            }
            rows.append(
                SupportRow(
                    entrypoint_id=spec.entrypoint_id,
                    entrypoint_label=spec.label,
                    task_slug=task_slug,
                    task_label=_task_label(task_slug),
                    cells=cells,
                )
            )

    return rows


def render_support_matrix(root: Path | None = None) -> str:
    resolved_root = repo_root(root)
    evidence = _load_declarations(resolved_root)
    mlx_tested_tasks = sorted(
        {
            combination.task_slug
            for combination in evidence.combinations
            if combination.entrypoint_id == "ppo_mlx"
            and combination.evidence_level == DeclaredEvidenceLevel.TESTED
        }
        if evidence is not None
        else (),
        key=_task_sort_key,
    )

    lines = [
        "### Evidence Grades",
        "",
        "| 等级 | 仓库事实来源 |",
        "|------|--------------|",
        "| `Registered` | `ensure_registries()` 导入后的 `registry.list_registered_envs()` 中存在该 env/backend。 |",
        "| `Configured` | 存在对应的 owner YAML：`conf/{ppo,appo,offpolicy}/task/...`。 |",
        "| `Tested` | `tests/` 中有自动化覆盖该 entrypoint/task owner/backend 组合。这里的 `Tested` 包含 config compose 与脚本/运行时测试，不等同于默认推荐路径。 |",
        "| `Benchmarked` | 逐组合 metadata 绑定 passing benchmark、fresh phase gate 与 compiled signature。 |",
        "| `Recommended` | 在 `Benchmarked` 基础上还有显式 rollout/support promotion 证据。 |",
        "",
        "`Tested` 只描述仓库中已有自动化覆盖，不代表该组合具备同名 MuJoCo owner 的全部 backend capability；"
        "例如 phase-1 Motrix owner 可能只覆盖训练 smoke 和明确启用的 DR 子集。",
        "",
        "`mjwarp` 不继承 entrypoint 级的通用 `Tested` 标记；其 `Tested` 及以上等级必须来自"
        " `conf/support/evidence.yaml` 中的逐组合声明，并通过仓库验收工具双向审计。",
        "Phase 7 task rollout 完成前，任何 `mjwarp` 组合都不得提升为 `Recommended`。",
        "`Recommended` 只适用于矩阵中声明的 entrypoint、task owner、backend 和 execution profile；"
        "不隐含未声明的原生 play/visualization 能力。当前 `mjwarp` 原生 play/visualization 仍显式 fail closed。",
        "",
        "### Entrypoint x Task Owner",
        "",
        "| Entrypoint | Task owner | MuJoCo | mjwarp | Motrix |",
        "|------------|------------|--------|--------|--------|",
    ]

    for row in build_support_rows(resolved_root):
        lines.append(
            f"| {row.entrypoint_label} | `{row.task_slug}` ({row.task_label}) | "
            f"{row.cells['mujoco'].level.label} | {row.cells['mjwarp'].level.label} | "
            f"{row.cells['motrix'].level.label} |"
        )

    lines.extend(
        [
            "",
            "### Source Index",
            "",
            "- Registry bootstrap: `src/unilab/envs/**` decorators via `unilab.base.registry.ensure_registries()`.",
            "- Owner YAML scan: `conf/ppo/task/**`, `conf/appo/task/**`, `conf/offpolicy/task/**`.",
            "- High-grade mjwarp declaration: `conf/support/evidence.yaml`.",
            "- Bidirectional audit: `uv run scripts/audit_acceptance.py support`.",
            "- Generic compose coverage is audited against the configured owner set.",
            "- MLX-specific declarations in `conf/support/evidence.yaml` upgrade these task owners: "
            + ", ".join(f"`{task}`" for task in mlx_tested_tasks)
            + ".",
            "- Runtime smoke and owner coverage are checked by the repository test lanes.",
        ]
    )
    return "\n".join(lines)


def render_generated_block(root: Path | None = None) -> str:
    return "\n".join([BEGIN_MARKER, render_support_matrix(root), END_MARKER])


def replace_generated_block(content: str, rendered_block: str) -> str:
    pattern = re.compile(
        rf"{re.escape(BEGIN_MARKER)}.*?{re.escape(END_MARKER)}",
        flags=re.DOTALL,
    )
    if pattern.search(content) is None:
        raise ValueError("Generated support matrix markers not found")
    return pattern.sub(rendered_block, content)
