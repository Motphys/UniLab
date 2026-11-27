"""Task bootstrap and package dependency boundary tests."""

from __future__ import annotations

import ast
from pathlib import Path

from unilab.base import registry
from unilab.tasks import __unilab_registry_modules__

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ENV_PACKAGE = _REPO_ROOT / "src" / "unilab" / "envs"
_CONCRETE_TASK_PACKAGES = ("locomotion", "manipulation", "motion_tracking")

_TASK_REGISTRY_MODULES = (
    "unilab.tasks.locomotion.go2",
    "unilab.tasks.locomotion.g1",
    "unilab.tasks.motion_tracking.g1",
)

_RUNTIME_NUMPY_ALLOWLIST = {
    "src/unilab/tasks/motion_tracking/common/motion_loader.py",
}
_PRODUCTION_NUMPY_IMPORT_ALLOWLIST = {
    # Base-owner adapter/reset-host boundary. These modules negotiate CPU
    # backend host formats and legacy play APIs; they are not Manager runtime.
    "src/unilab/base/base.py",
    "src/unilab/base/entity.py",
    "src/unilab/base/reset_state.py",
    "src/unilab/base/scene.py",
    "src/unilab/base/torch_env.py",
    "src/unilab/base/variants.py",
    # Cold asset/materialization leaves.
    "src/unilab/terrains/heightfield_terrains.py",
    "src/unilab/terrains/heightfield_utils.py",
    # Serialization and external interop.
    "src/unilab/training/onnx_export.py",
    "src/unilab/training/tensor_diagnostics.py",
    "src/unilab/utils/nan_viz.py",
    # Playback/rendering/visualization host formats.
    "src/unilab/scripts/play_interactive.py",
    "src/unilab/scripts/train_rsl_rl.py",
    "src/unilab/visualization/debug_primitives.py",
    "src/unilab/visualization/interactive_playback.py",
    "src/unilab/visualization/playback.py",
    "src/unilab/visualization/playback_session.py",
    "src/unilab/visualization/playback_state.py",
    "src/unilab/visualization/viser_playback.py",
    "src/unilab/visualization/viser_scene.py",
    # Cold NPZ motion ingestion.
    *_RUNTIME_NUMPY_ALLOWLIST,
    # RL runtime NumPy leaves merged from unilab-rl (ADR-0016): replay ring,
    # weight sync, NaN diagnostics, RNG application, tensor conversion.
    "src/unilab/rl/runtime/ipc/replay_buffer.py",
    "src/unilab/rl/runtime/ipc/weight_sync.py",
    "src/unilab/rl/runtime/offpolicy/actor_adapter.py",
    "src/unilab/rl/runtime/seed.py",
}


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
        elif isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
    return modules


def test_tasks_is_the_only_default_registry_bootstrap() -> None:
    assert registry._DEFAULT_REGISTRY_PACKAGES == ("unilab.tasks",)
    assert __unilab_registry_modules__ == _TASK_REGISTRY_MODULES


def test_env_runtime_does_not_depend_on_tasks() -> None:
    violations = [
        (path.relative_to(_REPO_ROOT).as_posix(), module)
        for path in sorted(_ENV_PACKAGE.rglob("*.py"))
        for module in sorted(_imports(path))
        if module == "unilab.tasks" or module.startswith("unilab.tasks.")
    ]

    assert violations == [], "unilab.envs must not import concrete unilab.tasks modules"


def test_env_runtime_does_not_own_concrete_task_packages() -> None:
    violations = [
        path.relative_to(_REPO_ROOT).as_posix()
        for package in _CONCRETE_TASK_PACKAGES
        for path in sorted((_ENV_PACKAGE / package).rglob("*.py"))
    ]

    assert violations == [], "concrete task source must be owned by unilab.tasks"


def test_tensor_runtime_switch_is_absent_from_source_and_owner_configs() -> None:
    offenders: list[str] = []
    patterns = (
        "cfg.tensor_runtime",
        "env.tensor_runtime",
        "tensor_runtime_device",
        "tensor_runtime:",
    )
    # The merged RL layer (ADR-0016) owns the tensor-runtime resolution and its
    # legacy `env.tensor_runtime` compatibility reads; the guard stays for the
    # env/task/manager layers and owner configs.
    allowlist = {
        "src/unilab/training/tensor_runtime.py",
        "src/unilab/rl/runtime/placement.py",
        "src/unilab/rl/runtime/offpolicy/worker.py",
        "src/unilab/rl/runtime/offpolicy/double_buffer_runner.py",
    }
    for path in (_REPO_ROOT / "src").rglob("*"):
        if not path.is_file() or path.suffix not in {".py", ".yaml"}:
            continue
        if ".venv" in path.parts or "__pycache__" in path.parts:
            continue
        relative = path.relative_to(_REPO_ROOT).as_posix()
        if relative in allowlist:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if any(pattern in text for pattern in patterns):
            offenders.append(relative)
    assert offenders == [], f"tensor-runtime switch references remain: {offenders}"


def test_production_source_does_not_import_or_configure_numba() -> None:
    source_root = _REPO_ROOT / "src" / "unilab"
    import_violations = [
        (path.relative_to(_REPO_ROOT).as_posix(), module)
        for path in sorted(source_root.rglob("*.py"))
        for module in sorted(_imports(path))
        if module == "numba" or module.startswith("numba.")
    ]
    configuration_violations = [
        (path.relative_to(_REPO_ROOT).as_posix(), line_number, line.strip())
        for path in sorted(source_root.rglob("*.py"))
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        # ``set_num_threads`` only counts as Numba configuration when it targets
        # numba itself; torch thread control (e.g. rl/runtime/offpolicy/
        # thread_budget.py) is a different owner.
        if "NUMBA_" in line or "numba.set_num_threads" in line
    ]

    assert import_violations == [], f"production Numba imports remain: {import_violations}"
    assert configuration_violations == [], (
        f"production Numba configuration remains: {configuration_violations}"
    )


def test_manager_base_curriculum_and_event_core_do_not_import_numpy() -> None:
    manager_core_paths = (
        _REPO_ROOT / "src" / "unilab" / "managers" / "manager_base.py",
        _REPO_ROOT / "src" / "unilab" / "managers" / "curriculum_manager.py",
        _REPO_ROOT / "src" / "unilab" / "managers" / "event_manager.py",
    )
    violations = [
        (path.relative_to(_REPO_ROOT).as_posix(), module)
        for path in manager_core_paths
        for module in sorted(_imports(path))
        if module == "numpy" or module.startswith("numpy.")
    ]

    assert violations == [], f"manager core NumPy imports remain: {violations}"


def test_manager_scheduling_has_no_numpy_rng_fallback() -> None:
    paths = (
        _REPO_ROOT / "src" / "unilab" / "managers" / "command_manager.py",
        _REPO_ROOT / "src" / "unilab" / "managers" / "event_manager.py",
    )
    offenders = [
        (path.relative_to(_REPO_ROOT).as_posix(), line_number, line.strip())
        for path in paths
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if ".rng.uniform" in line
    ]

    assert offenders == [], f"Manager scheduling NumPy RNG fallbacks remain: {offenders}"


def test_generic_event_terms_do_not_use_environment_numpy_rng() -> None:
    path = _REPO_ROOT / "src" / "unilab" / "envs" / "mdp" / "events.py"
    offenders = [
        line_number
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if "env.rng.uniform" in line or "self._env.rng.uniform" in line
    ]

    assert offenders == [], f"generic event NumPy RNG references remain at lines: {offenders}"


def test_manager_noise_and_temporal_buffers_do_not_import_numpy() -> None:
    source_root = _REPO_ROOT / "src" / "unilab" / "managers"
    owner_paths = (
        *sorted((source_root / "_buffers").glob("*.py")),
        *sorted((source_root / "_noise").glob("*.py")),
    )
    violations = [
        (path.relative_to(_REPO_ROOT).as_posix(), module)
        for path in owner_paths
        for module in sorted(_imports(path))
        if module == "numpy" or module.startswith("numpy.")
    ]

    assert violations == [], f"buffer/noise NumPy imports remain: {violations}"


def test_locomotion_runtime_terms_do_not_use_environment_numpy_rng() -> None:
    paths = (_REPO_ROOT / "src" / "unilab" / "tasks" / "locomotion" / "g1" / "manager_terms.py",)
    offenders = [
        (path.relative_to(_REPO_ROOT).as_posix(), line_number)
        for path in paths
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if "env.rng.uniform" in line or "self._env.rng.uniform" in line
    ]

    assert offenders == [], f"locomotion NumPy RNG references remain: {offenders}"


def test_reward_and_termination_managers_do_not_import_numpy() -> None:
    paths = (
        _REPO_ROOT / "src" / "unilab" / "managers" / "reward_manager.py",
        _REPO_ROOT / "src" / "unilab" / "managers" / "termination_manager.py",
    )
    violations = [
        (path.relative_to(_REPO_ROOT).as_posix(), module)
        for path in paths
        for module in sorted(_imports(path))
        if module == "numpy" or module.startswith("numpy.")
    ]

    assert violations == [], f"reward/termination NumPy imports remain: {violations}"


def test_observation_manager_has_no_numpy_result_conversion() -> None:
    path = _REPO_ROOT / "src" / "unilab" / "managers" / "observation_manager.py"
    text = path.read_text(encoding="utf-8")
    offenders = [
        pattern
        for pattern in ("np.ndarray | torch.Tensor", "torch.from_numpy", "expected np.ndarray")
        if pattern in text
    ]

    assert offenders == [], f"observation NumPy result conversion remains: {offenders}"


def test_manager_publication_boundary_has_no_numpy_conversion() -> None:
    path = _REPO_ROOT / "src" / "unilab" / "envs" / "manager_based_rl_env.py"
    offenders = [
        line_number
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if "np.array(values" in line or "torch.from_numpy(host)" in line
    ]

    assert offenders == [], f"Manager publication NumPy conversion remains: {offenders}"


def test_metrics_and_recorder_managers_do_not_import_numpy() -> None:
    paths = (
        _REPO_ROOT / "src" / "unilab" / "managers" / "metrics_manager.py",
        _REPO_ROOT / "src" / "unilab" / "managers" / "recorder_manager.py",
    )
    violations = [
        (path.relative_to(_REPO_ROOT).as_posix(), module)
        for path in paths
        for module in sorted(_imports(path))
        if module == "numpy" or module.startswith("numpy.")
    ]

    assert violations == [], f"metrics/recorder NumPy imports remain: {violations}"


def test_manager_env_protocol_reset_buffer_is_torch() -> None:
    path = _REPO_ROOT / "src" / "unilab" / "managers" / "types.py"
    offenders = [
        line_number
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if "def reset_buf(self) -> np.ndarray" in line
    ]

    assert offenders == [], f"Manager reset buffer remains NumPy-typed: {offenders}"


def test_flashsac_motion_owner_uses_generic_manager_runtime() -> None:
    registry.ensure_registries()
    assert "motrix" in registry._envs["G1MotionTracking"].env_factory_dict


def test_motion_direct_runtime_is_removed() -> None:
    direct_runtime = _REPO_ROOT / "src/unilab/tasks/motion_tracking/g1/torch_flashsac_env.py"
    assert not direct_runtime.exists()


def test_manager_and_env_runtime_do_not_import_numpy() -> None:
    runtime_roots = (
        _REPO_ROOT / "src" / "unilab" / "managers",
        _REPO_ROOT / "src" / "unilab" / "envs",
    )
    violations = [
        (path.relative_to(_REPO_ROOT).as_posix(), module)
        for root in runtime_roots
        for path in sorted(root.rglob("*.py"))
        for module in sorted(_imports(path))
        if module == "numpy" or module.startswith("numpy.")
    ]

    assert violations == [], f"Manager/env NumPy imports remain: {violations}"


def test_task_runtime_numpy_is_limited_to_cold_npz_loader() -> None:
    task_root = _REPO_ROOT / "src" / "unilab" / "tasks"
    violations = [
        (path.relative_to(_REPO_ROOT).as_posix(), module)
        for path in sorted(task_root.rglob("*.py"))
        for module in sorted(_imports(path))
        if (module == "numpy" or module.startswith("numpy."))
        and path.relative_to(_REPO_ROOT).as_posix() not in _RUNTIME_NUMPY_ALLOWLIST
    ]

    assert violations == [], f"unallowlisted task NumPy imports remain: {violations}"


def test_manager_env_and_task_runtime_have_no_numpy_conversion() -> None:
    runtime_roots = (
        _REPO_ROOT / "src" / "unilab" / "managers",
        _REPO_ROOT / "src" / "unilab" / "envs",
        _REPO_ROOT / "src" / "unilab" / "tasks",
    )
    offenders = [
        (path.relative_to(_REPO_ROOT).as_posix(), line_number, line.strip())
        for root in runtime_roots
        for path in sorted(root.rglob("*.py"))
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if (".numpy()" in line or ".cpu().numpy()" in line or "torch.from_numpy(" in line)
        and path.relative_to(_REPO_ROOT).as_posix() not in _RUNTIME_NUMPY_ALLOWLIST
    ]

    assert offenders == [], f"runtime NumPy conversions remain: {offenders}"


def test_production_source_does_not_seed_global_numpy_rng() -> None:
    source_root = _REPO_ROOT / "src" / "unilab"
    # rl/runtime/seed.py deliberately applies the process RNG seed to NumPy's
    # legacy global RandomState for collector host-side env physics/DR
    # (behavior preserved from unilab-rl, ADR-0016).
    allowlist = {"src/unilab/rl/runtime/seed.py"}
    offenders = [
        (path.relative_to(_REPO_ROOT).as_posix(), line_number, line.strip())
        for path in sorted(source_root.rglob("*.py"))
        if path.relative_to(_REPO_ROOT).as_posix() not in allowlist
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if "np.random.seed(" in line or "numpy.random.seed(" in line
    ]

    assert offenders == [], f"global NumPy RNG seeding remains: {offenders}"


def test_production_numpy_imports_are_explicit_leaves() -> None:
    source_root = _REPO_ROOT / "src" / "unilab"
    violations = [
        (path.relative_to(_REPO_ROOT).as_posix(), module)
        for path in sorted(source_root.rglob("*.py"))
        for module in sorted(_imports(path))
        if (module == "numpy" or module.startswith("numpy."))
        and path.relative_to(_REPO_ROOT).as_posix() not in _PRODUCTION_NUMPY_IMPORT_ALLOWLIST
    ]

    assert violations == [], f"unallowlisted production NumPy imports remain: {violations}"


_MANAGER_HOST_FIELD_PREFIXES = ("root_link_", "body_link_")
_MANAGER_HOST_FIELD_NAMES = frozenset(
    {
        "joint_pos",
        "joint_vel",
        "joint_pos_biased",
        "projected_gravity_b",
    }
)


def test_manager_runtime_does_not_read_entity_public_host_carriers() -> None:
    """Generic Manager terms must not resurrect hidden Entity host fallbacks."""

    runtime_roots = (
        _REPO_ROOT / "src" / "unilab" / "managers",
        _REPO_ROOT / "src" / "unilab" / "envs",
        _REPO_ROOT / "src" / "unilab" / "tasks",
    )
    offenders: list[tuple[str, int, str]] = []
    for root in runtime_roots:
        for path in sorted(root.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if not (
                    isinstance(node, ast.Attribute)
                    and node.attr in _MANAGER_HOST_FIELD_NAMES
                    or (
                        isinstance(node, ast.Attribute)
                        and node.attr.startswith(_MANAGER_HOST_FIELD_PREFIXES)
                    )
                ):
                    continue
                value = node.value
                if isinstance(value, ast.Attribute) and value.attr == "data":
                    offenders.append(
                        (
                            path.relative_to(_REPO_ROOT).as_posix(),
                            node.lineno,
                            node.attr,
                        )
                    )

    assert offenders == [], f"Manager Entity host-carrier reads remain: {offenders}"


_ENTITY_PUBLIC_HOST_FIELD_PREFIXES = ("root_link_", "body_link_")
_ENTITY_PUBLIC_HOST_FIELD_NAMES = frozenset(
    {
        "joint_pos",
        "joint_vel",
        "joint_pos_biased",
        "projected_gravity_b",
        "heading_w",
        "root_link_pose_w",
        "root_link_vel_w",
        "body_link_pose_w",
        "body_link_vel_w",
    }
)


def test_entity_does_not_reintroduce_public_manager_host_carriers() -> None:
    path = _REPO_ROOT / "src" / "unilab" / "base" / "entity.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    entity_data = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "EntityData"
    )
    violations: list[tuple[int, str]] = []
    for node in entity_data.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name.startswith("_"):
            continue
        if node.name in _ENTITY_PUBLIC_HOST_FIELD_NAMES or node.name.startswith(
            _ENTITY_PUBLIC_HOST_FIELD_PREFIXES
        ):
            violations.append((node.lineno, node.name))

    assert violations == [], f"public Entity host carriers remain: {violations}"
