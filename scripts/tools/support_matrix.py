"""Generate backend support matrix content from registry, configs, and tests."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path

from omegaconf import OmegaConf

from unilab.base import registry
from unilab.base.registry import ensure_registries

BEGIN_MARKER = "<!-- BEGIN GENERATED SUPPORT MATRIX -->"
END_MARKER = "<!-- END GENERATED SUPPORT MATRIX -->"
BACKENDS: tuple[str, ...] = (
    "mujoco",
    "mjwarp",
    "motrix",
    "isaacgym",
    "genesis",
    "isaacsim",
    "newton",
)

# Maintainer-confirmed completed training validations. Keep this mapping narrow:
# generic config/contract coverage must not promote an unvalidated entrypoint.
_MAINTAINER_VALIDATED_MJWARP_ENTRYPOINT_TASKS = frozenset(
    {
        ("ppo_torch", "g1_walk_flat"),
        ("sac_torch", "g1_walk_flat"),
    }
)

# Maintainer-confirmed completed training validations for the isaacgym
# subprocess backend (real hardware, external Python 3.8 worker runtime;
# not covered by repo CI).
_MAINTAINER_VALIDATED_ISAACGYM_ENTRYPOINT_TASKS: frozenset[tuple[str, str]] = frozenset(
    {
        ("sac_torch", "g1_walk_flat"),
    }
)

# Maintainer-confirmed completed training validations for the genesis
# in-process backend (real hardware, genesis-world extra + CUDA; not covered
# by repo CI). sac_torch g1_walk_flat: full 5000-iteration training completed
# on 2026-08-31 (RTX 4090, torch 2.8.0+cu128, genesis-world 1.3.3; reward/mean
# 6.5 -> 244.8, episode length -> 987, run 2026-08-31_23-04-01_genesis) plus
# record playback validation on model_5000.pt.
_MAINTAINER_VALIDATED_GENESIS_ENTRYPOINT_TASKS: frozenset[tuple[str, str]] = frozenset(
    {
        ("sac_torch", "g1_walk_flat"),
    }
)

# IsaacSim is a Python 3.11 worker integration with eval-owned Kit viewer and
# RGB camera protocol coverage. No full training or successful real playback
# validation is promoted here: the checked-in evidence remains owner/config,
# protocol tests, and bounded backend smoke coverage.
_MAINTAINER_VALIDATED_ISAACSIM_ENTRYPOINT_TASKS: frozenset[tuple[str, str]] = frozenset()

# Maintainer-confirmed completed training validations for the newton
# in-process backend (real hardware, newton extra + CUDA; not covered by
# repo CI). sac_torch g1_walk_flat: full 5000-iteration training completed
# on 2026-09-06 (RTX 4090, torch 2.8.0+cu128, newton 1.5.1, mujoco-warp
# 3.11; reward/mean 6.68 -> 242.3, episode length -> 983, ~43k steps/s,
# 4m25s wall, run 2026-09-06_01-21-36_newton) plus record playback
# validation on model_5000.pt (800-frame MuJoCo snapshot render, walking
# gait confirmed). PPO remains ``Configured`` without runtime evidence.
_MAINTAINER_VALIDATED_NEWTON_ENTRYPOINT_TASKS: frozenset[tuple[str, str]] = frozenset(
    {
        ("sac_torch", "g1_walk_flat"),
    }
)

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
        config_dir="src/unilab/conf/ppo/task",
        task_glob="*/*.yaml",
        generic_tested=True,
    ),
    EntrypointSpec(
        entrypoint_id="appo_torch",
        label="APPO (torch)",
        config_dir="src/unilab/conf/appo/task",
        task_glob="*/*.yaml",
        generic_tested=True,
    ),
    EntrypointSpec(
        entrypoint_id="sac_torch",
        label="SAC (torch)",
        config_dir="src/unilab/conf/sac/task",
        task_glob="*/*.yaml",
        generic_tested=True,
    ),
    EntrypointSpec(
        entrypoint_id="td3_torch",
        label="TD3 (torch)",
        config_dir="src/unilab/conf/td3/task",
        task_glob="*/*.yaml",
        generic_tested=True,
    ),
    EntrypointSpec(
        entrypoint_id="flashsac_torch",
        label="FlashSAC (torch)",
        config_dir="src/unilab/conf/flashsac/task",
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
        if backend not in BACKENDS:
            continue
        entries.setdefault(task_slug, {})[backend] = _load_task_name(task_path)
    return entries


def _is_tested(spec: EntrypointSpec, task_slug: str, backend: str, root: Path) -> bool:
    if backend == "mjwarp":
        return (
            spec.entrypoint_id,
            task_slug,
        ) in _MAINTAINER_VALIDATED_MJWARP_ENTRYPOINT_TASKS
    if backend == "isaacgym":
        return (
            spec.entrypoint_id,
            task_slug,
        ) in _MAINTAINER_VALIDATED_ISAACGYM_ENTRYPOINT_TASKS
    if backend == "genesis":
        return (
            spec.entrypoint_id,
            task_slug,
        ) in _MAINTAINER_VALIDATED_GENESIS_ENTRYPOINT_TASKS
    if backend == "isaacsim":
        return (
            spec.entrypoint_id,
            task_slug,
        ) in _MAINTAINER_VALIDATED_ISAACSIM_ENTRYPOINT_TASKS
    if backend == "newton":
        return (
            spec.entrypoint_id,
            task_slug,
        ) in _MAINTAINER_VALIDATED_NEWTON_ENTRYPOINT_TASKS
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
            cells = {
                backend: SupportCell(
                    env_name=env_name,
                    level=_cell_level(
                        backend=backend,
                        env_name=env_name,
                        configured_backends=configured_backends,
                        registry_backends=registry_backends,
                        tested=_is_tested(spec, task_slug, backend, resolved_root),
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
        "| `Configured` | 存在对应的 owner YAML：`src/unilab/conf/{ppo,appo,sac,td3,flashsac}/task/...`。 |",
        "| `Tested` | `tests/` 中有自动化覆盖该 entrypoint/task owner/backend 组合，或存在显式 maintainer 完整训练验证并具备近风险自动化测试。这里的 `Tested` 不等同于默认推荐路径。 |",
        "| `Benchmarked` | 存在与该组合绑定的已提交 benchmark manifest。 |",
        "| `Recommended` | 仓库中存在显式 recommendation 元数据。 |",
        "",
        "`Tested` 只描述仓库中已有自动化覆盖或显式 maintainer 训练验证，不代表该组合具备同名 MuJoCo "
        "owner 的全部 backend capability；例如 phase-1 Motrix owner 可能只覆盖训练 smoke 和明确启用的 DR 子集。",
        "",
        "`mjwarp` 完成训练验证的只有 `g1_walk_flat` host adapter：PPO (torch) 与 SAC (torch) owner "
        "已完成训练验证，并有 backend、contract 与 playback 自动化覆盖，因此标记为 `Tested`。"
        "mjwarp playback 默认仅支持显式、有限步数的 `record` 并复用 MuJoCo 离线 renderer；"
        "`uv run eval --sim mjwarp --render-mode interactive` 路由到 MuJoCo 交互 viewer"
        "（mjwarp 跑物理、MuJoCo 渲染 env[0]，强制单 env）；不支持 `auto` 或 native playback。"
        "其他 entrypoint 中出现的 `Registered` 只表示 env/backend registry "
        "identity，不代表对应算法、terrain、完整 DR 或 production training 支持。",
        "",
        "`isaacgym` 是 Python 3.8 子进程后端，当前只接入 `g1_walk_flat`。SAC (torch) owner 已在真机"
        "（external Python 3.8 worker runtime，不在仓库 CI 覆盖）完成训练与 record playback 验证，"
        "标记为 `Tested`；其余 isaacgym cell 最高只到 `Configured`（registry + owner YAML + "
        "compose/contract 覆盖），不代表任何训练或 play 验证。playback 走 IsaacGym 原生渲染"
        "（viewer + camera sensor 离屏录制），有显示器时 `play_render_mode=auto` 打开交互 viewer，"
        "无显示器时自动降级为离屏录制。",
        "",
        "`genesis` 是进程内后端（genesis-world==1.3.3，要求 torch>=2.8 与 CUDA；一进程只允许一次 "
        "`gs.init`），当前只接入 `g1_walk_flat` 的 PPO (torch) 与 SAC (torch) owner。SAC cell 标记 "
        "`Tested`：真机完整训练验证（5000/5000 iterations，reward/mean 6.5 → 244.8，episode length "
        "→ 987/1000，10.26M env steps / 224s wall time；run 2026-08-31_23-04-01_genesis）加 "
        "model_5000.pt 的 record playback 验证；PPO cell 最高只到 `Configured`（registry + owner "
        "YAML + compose/contract 覆盖），不代表训练验证。真机证据另有："
        "env smoke 慢车道测试（`tests/envs/locomotion/g1/test_g1_owner_contract.py`：compose → env 构造 "
        "→ keyframe reset → 12 步有限稳定 → cleanup，覆盖 ppo 与 sac 两棵树）在装有 CUDA 与 genesis "
        "extra 的机器上通过。adapter 的 "
        "`materialize()` 幂等且惰性触发（entity 校验在 env 的 materialize 钩子前读取状态 getter；"
        "isaacgym 后端同模式）。Genesis 在 import 时丢弃 MJCF 全局 "
        "`<option>`，owner YAML 显式重声明 `genesis_integrator=implicitfast`。原生 playback/渲染已接入："
        "`play_render_mode=auto` 在有显示时打开 post-build 挂载的交互 viewer、无显示时降级离屏录制"
        "（`record` 写 `play_video.mp4`；`get_physics_state` 快照不声明）。未支持边界：geom 名称"
        "契约、terrain spawn 与 height scanner、contact "
        'sensor 为 per-link net-force 阈值近似（非 geom 对 `data="found"`）、`get_geom_friction` 类'
        "绝对摩擦 DR fail-closed（geom 摩擦只有 per-env ratio API）。",
        "",
        "`isaacsim` 是 IsaacSim 5.1 / IsaacLab v2.3.0 的独立 Python 3.11 子进程后端，当前只接入"
        " `g1_walk_flat` 的 PPO/SAC owner，矩阵标记为 `Configured`。仓库没有把 bounded headless"
        " physics smoke 和 mock rendering protocol 覆盖提升为训练或 playback 的 `Tested` 证据。"
        "eval 已接入 Kit viewer 与 IsaacLab RGB camera；当前真实主机在 RTX renderer 初始化阶段"
        "崩溃，因此没有成功 playback 证据，也不会生成占位视频。contact-force sensor 和 domain "
        "randomization 仍保持 fail-closed。",
        "",
        benchmark_note,
        recommendation_note,
        "",
        "### Entrypoint x Task Owner",
        "",
        "| Entrypoint | Task owner | MuJoCo | mjwarp | Motrix | IsaacGym | Genesis | IsaacSim | Newton |",
        "|------------|------------|--------|--------|--------|----------|---------|----------|--------|",
    ]

    for row in build_support_rows(resolved_root):
        lines.append(
            f"| {row.entrypoint_label} | `{row.task_slug}` ({row.task_label}) | "
            f"{row.cells['mujoco'].level.label} | {row.cells['mjwarp'].level.label} | "
            f"{row.cells['motrix'].level.label} | {row.cells['isaacgym'].level.label} | "
            f"{row.cells['genesis'].level.label} | {row.cells['isaacsim'].level.label} |"
            f" {row.cells['newton'].level.label} |"
        )

    lines.extend(
        [
            "",
            "### Source Index",
            "",
            "- Registry bootstrap: `src/unilab/envs/**` decorators via `unilab.base.registry.ensure_registries()`.",
            "- Owner YAML scan: `src/unilab/conf/ppo/task/**`, `src/unilab/conf/appo/task/**`, `src/unilab/conf/sac/task/**`, `src/unilab/conf/td3/task/**`, `src/unilab/conf/flashsac/task/**`.",
            "- Generic compose coverage: `tests/config/test_config_system.py::test_supported_task_composes`.",
            "- Validated mjwarp entrypoints are explicitly recorded in `_MAINTAINER_VALIDATED_MJWARP_ENTRYPOINT_TASKS`; near-risk coverage lives in `tests/base/test_mjwarp_backend.py`, `tests/base/test_backend_conformance.py`, `tests/base/test_mjwarp_differential.py`, and `tests/base/test_mjwarp_playback.py`.",
            "- Validated isaacgym entrypoints are explicitly recorded in `_MAINTAINER_VALIDATED_ISAACGYM_ENTRYPOINT_TASKS` (real hardware via the external Python 3.8 worker runtime; not covered by repo CI).",
            "- Validated genesis entrypoints are explicitly recorded in `_MAINTAINER_VALIDATED_GENESIS_ENTRYPOINT_TASKS` (real hardware, genesis-world extra + CUDA; not covered by repo CI); near-risk coverage lives in `tests/base/test_genesis_backend.py` (fake runtime), `tests/base/test_genesis_runtime.py` (real-runtime slow lane), and the genesis env smoke in `tests/envs/locomotion/g1/test_g1_owner_contract.py`.",
            "- IsaacSim owner scope is intentionally not promoted to `Tested`; `_MAINTAINER_VALIDATED_ISAACSIM_ENTRYPOINT_TASKS` is empty until a maintainer records full training evidence. Rendering protocol coverage lives in `tests/base/test_isaacsim_backend.py`; it is not a substitute for successful real playback.",
            "- `newton` is an isolated optional owner backed by Newton 1.5.1 and the MuJoCo-Warp 3.11 / Warp 1.16 line. Validated newton entrypoints are explicitly recorded in `_MAINTAINER_VALIDATED_NEWTON_ENTRYPOINT_TASKS` (real hardware, newton extra + CUDA; not covered by repo CI); remaining cells rely on the G1 PPO/SAC owner configs, compose/contract checks, and fail-closed runtime/import boundaries.",
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
