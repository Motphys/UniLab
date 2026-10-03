"""Generate backend support matrix content from registry, configs, and tests."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path

from omegaconf import OmegaConf
from unisim.support import TensorPlatformProfile, get_tensor_platform_profiles

from unilab.base import registry
from unilab.base.registry import ensure_registries

BEGIN_MARKER = "<!-- BEGIN GENERATED SUPPORT MATRIX -->"
END_MARKER = "<!-- END GENERATED SUPPORT MATRIX -->"
# UniSim owns the reviewed tensor adapter inventory. During issue #1811 the
# production runtime exposes only the scoped tensor Manager backends; shelved
# adapters remain in UniSim but are intentionally absent from this support
# matrix until capability/parity evidence re-enables them.
_ALL_BACKENDS: tuple[str, ...] = tuple(get_tensor_platform_profiles())
BACKENDS: tuple[str, ...] = tuple(
    backend for backend in _ALL_BACKENDS if backend in {"mujoco", "mjwarp", "genesis", "newton"}
)
SHELVED_BACKENDS = frozenset(_ALL_BACKENDS) - frozenset(BACKENDS)

# Issue-gated M9 candidate owners (#1674) are benchmark/test fixtures,
# not support claims.  Exclude them from the generated matrix until those
# issues close;
# otherwise the scanner would mechanically promote a task/backend cell from
# Registered to Configured without acceptance evidence.
_ISSUE_GATED_CANDIDATE_CONFIGS = frozenset(
    {
        "src/unilab/conf/flashsac/task/g1_motion_tracking/isaacgym.yaml",
        "src/unilab/conf/flashsac/task/g1_motion_tracking/isaacsim.yaml",
    }
)

# Maintainer-confirmed completed training validations. Keep this mapping narrow:
# generic config/contract coverage must not promote an unvalidated entrypoint.
_MAINTAINER_VALIDATED_MJWARP_ENTRYPOINT_TASKS = frozenset(
    {
        ("ppo_torch", "g1_walk_flat"),
        ("sac_torch", "g1_walk_flat"),
        ("warpsac_torch", "g1_walk_flat"),
        ("warpsac_torch", "g1_motion_tracking"),
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
# 4m25s wall, run 2026-09-06_01-21-36_newton). Playback validated the same
# day on model_5000.pt: native ViewerGL offscreen record (800-frame
# 1280x720 mp4 via ``newton-viewer-gl``) and an interactive ViewerGL window
# smoke on a live X display; the MuJoCo snapshot record path remains as the
# no-render-deps fallback. PPO remains ``Configured`` without runtime
# evidence.
_MAINTAINER_VALIDATED_NEWTON_ENTRYPOINT_TASKS: frozenset[tuple[str, str]] = frozenset(
    {
        ("sac_torch", "g1_walk_flat"),
    }
)

_TASK_ORDER = {
    "go2_joystick_flat": 0,
    "g1_walk_flat": 1,
    "g1_motion_tracking": 2,
    "g1_flip_tracking": 3,
    "x2_wall_flip_tracking": 4,
    "allegro_inhand": 5,
    "allegro_sac": 6,
}
_TASK_LABELS = {
    "go2_joystick_flat": "Go2 joystick",
    "g1_walk_flat": "G1 walk flat",
    "g1_motion_tracking": "G1 motion tracking",
    "g1_flip_tracking": "G1 flip tracking",
    "x2_wall_flip_tracking": "X2 wall flip tracking",
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
        entrypoint_id="flashsac_torch",
        label="FlashSAC (torch)",
        config_dir="src/unilab/conf/flashsac/task",
        task_glob="*/*.yaml",
        generic_tested=True,
    ),
    EntrypointSpec(
        entrypoint_id="warpsac_torch",
        label="WarpSAC (torch)",
        config_dir="src/unilab/conf/warpsac/task",
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


def _load_task_owner(task_path: Path) -> tuple[str, str, bool]:
    raw = OmegaConf.to_container(OmegaConf.load(task_path), resolve=True) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Expected mapping config in {task_path}")
    if task_path.name == "base.yaml":
        return "", "", False
    training = raw.get("training")
    if not isinstance(training, dict) or "task_name" not in training:
        raise ValueError(f"Missing training.task_name in {task_path}")
    task_name = training["task_name"]
    if not isinstance(task_name, str):
        raise ValueError(f"training.task_name must be a string in {task_path}")
    backend = training.get("sim_backend")
    if not isinstance(backend, str) or not backend.strip():
        raise ValueError(f"training.sim_backend must be a non-empty string in {task_path}")
    env = raw.get("env")
    tensor_runtime = isinstance(env, dict) and bool(env.get("tensor_runtime", False))
    return task_name, backend.strip(), tensor_runtime


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
        relative_path = task_path.relative_to(root).as_posix()
        if relative_path in _ISSUE_GATED_CANDIDATE_CONFIGS:
            continue
        task_slug = task_path.parent.name
        task_name, backend, _tensor_runtime = _load_task_owner(task_path)
        if backend not in BACKENDS:
            continue
        entries.setdefault(task_slug, {})[backend] = task_name
    return entries


def _is_tested(spec: EntrypointSpec, task_slug: str, backend: str, root: Path) -> bool:
    if backend == "superdex":
        # Bounded native smoke/short training does not establish full training support.
        return False
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


def platform_display(profile: TensorPlatformProfile, language: str) -> str:
    execution = {
        "host_bridge": "Host bridge" if language == "en" else "Host bridge",
        "device_resident": "Device-resident" if language == "en" else "Device-resident",
    }[profile.execution.value]
    topology = "in-process" if profile.process_topology.value == "in_process" else "external worker"
    data_plane = profile.data_plane.value.replace("_", " ")
    return f"{execution} / {topology} / {data_plane}"


def _capability_label(value: str, language: str) -> str:
    if value == "exact":
        return "Exact" if language == "en" else "支持"
    if value == "unsupported":
        return "Unsupported" if language == "en" else "不支持"
    return value


def _backend_label(backend: str) -> str:
    return {
        "mujoco": "MuJoCo",
        "mjwarp": "mjwarp",
        "motrix": "Motrix",
        "isaacgym": "IsaacGym",
        "genesis": "Genesis",
        "isaacsim": "IsaacSim",
        "newton": "Newton",
        "superdex": "SuperDex",
        "drake": "Drake",
    }[backend]


def render_support_matrix(root: Path | None = None, language: str = "zh") -> str:
    resolved_root = repo_root(root)
    if language not in {"zh", "en"}:
        raise ValueError(f"Unsupported support-matrix language: {language!r}")
    profiles = get_tensor_platform_profiles()
    if tuple(profiles) != _ALL_BACKENDS:
        raise ValueError(
            "UniSim tensor platform inventory does not match the UniLab backend order: "
            f"{tuple(profiles)} != {_ALL_BACKENDS}"
        )

    if language == "zh":
        lines = [
            "### Evidence Grades",
            "",
            "| 等级 | 仓库事实来源 |",
            "|------|--------------|",
            "| `Registered` | `ensure_registries()` 导入后的 `registry.list_registered_envs()` 中存在该 env/backend。 |",
            "| `Configured` | owner YAML 的 `training.sim_backend` 指向该 backend。 |",
            "| `Tested` | 自动化覆盖或显式 maintainer 完整训练验证；不等同于默认推荐路径。 |",
            "| `Benchmarked` | 存在与该组合绑定的已提交 benchmark manifest。 |",
            "| `Recommended` | 仓库中存在显式 recommendation 元数据。 |",
            "",
            "`Tested` 只描述仓库证据，不表示同名 MuJoCo owner 的全部 DR、渲染或 production 能力。当前没有已提交 benchmark/recommendation 元数据，因此不会自动提升到 `Benchmarked` 或 `Recommended`。",
            "",
            "### Tensor Backend Platform Matrix",
            "",
            "该表来自 UniSim SDK-free 公开静态能力清单；它表示 reviewed tensor lifecycle 边界，不表示可选 SDK 已安装，也不把所有 task owner 自动提升为可用组合。",
            "",
            "| Backend | Execution / process / data plane | Torch devices | CUDA runtime | Linux+CUDA | macOS | ROCm | Worker | Reset randomization | Fixed variants | Host callbacks | Packed bridge |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
    else:
        lines = [
            "### Evidence Grades",
            "",
            "| Grade | Repository evidence |",
            "|---|---|",
            "| `Registered` | The env/backend exists in `registry.list_registered_envs()` after `ensure_registries()`. |",
            "| `Configured` | The owner YAML sets `training.sim_backend` to the backend. |",
            "| `Tested` | Automated coverage or explicit maintainer validation; it is not a default recommendation. |",
            "| `Benchmarked` | A checked-in benchmark manifest is bound to the combination. |",
            "| `Recommended` | Explicit recommendation metadata exists in the repository. |",
            "",
            "`Tested` describes repository evidence only; it does not imply every DR, rendering, or production capability of the MuJoCo owner. No benchmark or recommendation metadata is currently checked in, so rows do not auto-promote to `Benchmarked` or `Recommended`.",
            "",
            "### Tensor Backend Platform Matrix",
            "",
            "This table is derived from UniSim's SDK-free public static inventory. It describes reviewed tensor-lifecycle boundaries, not optional-SDK installation and not automatic task-owner support.",
            "",
            "| Backend | Execution / process / data plane | Torch devices | CUDA runtime | Linux+CUDA | macOS | ROCm | Worker | Reset randomization | Fixed variants | Host callbacks | Packed bridge |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]

    for backend in BACKENDS:
        profile = profiles[backend]
        lines.append(
            f"| `{backend}` | {platform_display(profile, language)} | "
            f"{' / '.join(label.upper() for label in profile.torch_devices)} | "
            f"{profile.cuda_runtime} | {profile.linux_cuda} | {profile.macos_tensor_profile} | "
            f"{profile.rocm_tensor_profile} | {profile.worker_requirement} | "
            f"{_capability_label(profile.reset_randomization, language)} | "
            f"{_capability_label(profile.fixed_variants, language)} | "
            f"{_capability_label(profile.host_pre_step_control, language)} | "
            f"{_capability_label(profile.packed_host_bridge, language)} |"
        )

    lines.extend(
        [
            "",
            "### Entrypoint x Task Owner",
            "",
            f"| Entrypoint | Task owner | {' | '.join(_backend_label(b) for b in BACKENDS)} |",
            f"|------------|------------|{'---|' * len(BACKENDS)}",
        ]
    )

    for row in build_support_rows(resolved_root):
        lines.append(
            f"| {row.entrypoint_label} | `{row.task_slug}` ({row.task_label}) | "
            + " | ".join(row.cells[backend].level.label for backend in BACKENDS)
            + " |"
        )

    lines.extend(
        [
            "",
            "### Source Index",
            "",
            "- Registry bootstrap: `src/unilab/envs/**` decorators via `unilab.base.registry.ensure_registries()`.",
            "- Owner backend identity: `training.sim_backend` in `src/unilab/conf/{ppo,appo,sac,flashsac,warpsac}/task/**`.",
            "- Platform/capability source: `unisim.support.get_tensor_platform_profiles()`.",
            "- Unsupported platform/device requests are guarded before backend construction in `src/unilab/base/backend_factory.py`.",
            "- Generic compose coverage: `tests/config/test_config_system.py::test_supported_task_composes`.",
        ]
    )
    return "\n".join(lines)


def render_generated_block(root: Path | None = None, language: str = "zh") -> str:
    return "\n".join([BEGIN_MARKER, render_support_matrix(root, language), END_MARKER])


def replace_generated_block(content: str, rendered_block: str) -> str:
    pattern = re.compile(
        rf"{re.escape(BEGIN_MARKER)}.*?{re.escape(END_MARKER)}",
        flags=re.DOTALL,
    )
    if pattern.search(content) is None:
        raise ValueError("Generated support matrix markers not found")
    return pattern.sub(rendered_block, content)
