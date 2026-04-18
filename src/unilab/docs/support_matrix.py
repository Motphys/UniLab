"""Generate backend support matrix content from registry, configs, and tests."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path

from omegaconf import OmegaConf

from unilab.base import registry
from unilab.utils.algo_utils import ensure_registries

BEGIN_MARKER = "<!-- BEGIN GENERATED SUPPORT MATRIX -->"
END_MARKER = "<!-- END GENERATED SUPPORT MATRIX -->"
BACKENDS: tuple[str, str] = ("mujoco", "motrix")

_TASK_ORDER = {
    "go1_joystick_flat": 0,
    "go2_joystick_flat": 1,
    "g1_joystick_flat": 2,
    "g1_motion_tracking": 3,
    "g1_flip_tracking": 4,
    "g1_walk_flat": 5,
    "g1_walk_rough": 6,
    "allegro_inhand": 7,
    "allegro_sac": 8,
}
_TASK_LABELS = {
    "go1_joystick_flat": "Go1 joystick",
    "go2_joystick_flat": "Go2 joystick",
    "g1_joystick_flat": "G1 joystick",
    "g1_motion_tracking": "G1 motion tracking",
    "g1_flip_tracking": "G1 flip tracking",
    "g1_walk_flat": "G1 walk flat",
    "g1_walk_rough": "G1 walk rough",
    "allegro_inhand": "Allegro in-hand",
    "allegro_sac": "Allegro SAC in-hand",
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


@dataclass(frozen=True)
class SupportRow:
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


def _mlx_tested_task_slugs(root: Path) -> set[str]:
    config_test_path = root / "tests" / "config" / "test_config_system.py"
    content = config_test_path.read_text(encoding="utf-8")
    match = re.search(r"_PPO_MLX_TASKS\s*=\s*(\{[^\n]+\})", content)
    if match is None:
        return set()
    parsed = ast.literal_eval(match.group(1))
    if not isinstance(parsed, set):
        return set()
    return {item for item in parsed if isinstance(item, str)}


def _has_checked_in_benchmark_manifest(root: Path) -> bool:
    del root
    return False


def _has_recommendation_metadata(root: Path) -> bool:
    del root
    return False


def _configured_entries(root: Path, spec: EntrypointSpec) -> dict[str, dict[str, str]]:
    task_root = root / spec.config_dir
    entries: dict[str, dict[str, str]] = {}
    for task_path in sorted(task_root.glob(spec.task_glob)):
        task_slug = task_path.parent.name
        backend = task_path.stem
        entries.setdefault(task_slug, {})[backend] = _load_task_name(task_path)
    return entries


def _is_tested(spec: EntrypointSpec, task_slug: str, root: Path) -> bool:
    if spec.entrypoint_id == "ppo_mlx":
        return task_slug in _mlx_tested_task_slugs(root)
    return spec.generic_tested


def _cell_level(
    *,
    backend: str,
    env_name: str,
    configured_backends: dict[str, str],
    registry_backends: dict[str, set[str]],
    tested: bool,
    benchmarked: bool,
    recommended: bool,
) -> EvidenceLevel:
    available_backends = registry_backends.get(env_name, set())
    if backend not in available_backends:
        return EvidenceLevel.MISSING

    level = EvidenceLevel.REGISTERED
    if backend in configured_backends:
        level = EvidenceLevel.CONFIGURED
    if backend in configured_backends and tested:
        level = EvidenceLevel.TESTED
    if backend in configured_backends and tested and benchmarked:
        level = EvidenceLevel.BENCHMARKED
    if backend in configured_backends and tested and benchmarked and recommended:
        level = EvidenceLevel.RECOMMENDED
    return level


def build_support_rows(root: Path | None = None) -> list[SupportRow]:
    resolved_root = repo_root(root)
    registry_backends = _load_registry_backends()
    benchmarked = _has_checked_in_benchmark_manifest(resolved_root)
    recommended = _has_recommendation_metadata(resolved_root)
    rows: list[SupportRow] = []

    for spec in ENTRYPOINT_SPECS:
        for task_slug, configured_backends in sorted(
            _configured_entries(resolved_root, spec).items(),
            key=lambda item: _task_sort_key(item[0]),
        ):
            env_name = next(iter(configured_backends.values()))
            tested = _is_tested(spec, task_slug, resolved_root)
            cells = {
                backend: SupportCell(
                    env_name=env_name,
                    level=_cell_level(
                        backend=backend,
                        env_name=env_name,
                        configured_backends=configured_backends,
                        registry_backends=registry_backends,
                        tested=tested,
                        benchmarked=benchmarked,
                        recommended=recommended,
                    ),
                )
                for backend in BACKENDS
            }
            rows.append(
                SupportRow(
                    entrypoint_label=spec.label,
                    task_slug=task_slug,
                    task_label=_task_label(task_slug),
                    cells=cells,
                )
            )

    return rows


def render_support_matrix(root: Path | None = None) -> str:
    resolved_root = repo_root(root)
    mlx_tested_tasks = sorted(_mlx_tested_task_slugs(resolved_root), key=_task_sort_key)
    benchmark_note = (
        "未检测到与这些组合绑定的已提交 benchmark manifest，因此当前不会自动提升到 `Benchmarked`。"
    )
    recommendation_note = (
        "仓库中目前也没有单独的 recommendation 元数据，因此当前不会自动提升到 `Recommended`。"
    )

    lines = [
        "### Evidence Grades",
        "",
        "| 等级 | 仓库事实来源 |",
        "|------|--------------|",
        "| `Registered` | `ensure_registries()` 导入后的 `registry.list_registered_envs()` 中存在该 env/backend。 |",
        "| `Configured` | 存在对应的 owner YAML：`conf/{ppo,appo,offpolicy}/task/...`。 |",
        "| `Tested` | `tests/` 中有自动化覆盖该 entrypoint/task owner/backend 组合。这里的 `Tested` 包含 config compose 与脚本/运行时测试，不等同于默认推荐路径。 |",
        "| `Benchmarked` | 存在与该组合绑定的已提交 benchmark manifest。 |",
        "| `Recommended` | 仓库中存在显式 recommendation 元数据。 |",
        "",
        benchmark_note,
        recommendation_note,
        "",
        "### Entrypoint x Task Owner",
        "",
        "| Entrypoint | Task owner | MuJoCo | Motrix |",
        "|------------|------------|--------|--------|",
    ]

    for row in build_support_rows(resolved_root):
        lines.append(
            f"| {row.entrypoint_label} | `{row.task_slug}` ({row.task_label}) | "
            f"{row.cells['mujoco'].level.label} | {row.cells['motrix'].level.label} |"
        )

    lines.extend(
        [
            "",
            "### Source Index",
            "",
            "- Registry bootstrap: `src/unilab/envs/**` decorators via `unilab.utils.algo_utils.ensure_registries()`.",
            "- Owner YAML scan: `conf/ppo/task/**`, `conf/appo/task/**`, `conf/offpolicy/task/**`.",
            "- Generic compose coverage: `tests/config/test_config_system.py::test_supported_task_composes`.",
            "- MLX-specific compose coverage only upgrades task owners listed in `tests/config/test_config_system.py::_PPO_MLX_TASKS`: "
            + ", ".join(f"`{task}`" for task in mlx_tested_tasks)
            + ".",
            "- MLX runtime smoke: `tests/algos/test_mlx_ppo.py::test_mlx_ppo_one_iteration_real_env` currently exercises `go2_joystick_flat/mujoco`.",
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
