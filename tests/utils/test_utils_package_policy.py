import importlib
from pathlib import Path

import unilab.utils

ALLOWED_UTILS_API = {"get_default_device", "to_numpy", "to_torch"}
ALLOWED_UTILS_MODULES = {"__init__", "device", "tensor"}
REMOVED_UTILS_SHIMS = {
    "algo_utils",
    "device_utils",
    "experiment_tracking",
    "final_observation",
    "hardware_monitor",
    "logging_common",
    "math_utils",
    "obs_utils",
    "offpolicy_logger",
    "onpolicy_logger",
    "render_many",
    "reward_utils",
    "rsl_rl_compat",
    "rsl_rl_vec_env_wrapper",
    "run_utils",
    "torch_utils",
    "viser_scene",
    "xml_utils",
}
REMOVED_OWNER_ALIASES = {
    "unilab.algos.torch.offpolicy.logging",
    "unilab.algos.torch.common.tensor",
}


def test_utils_api_is_whitelisted() -> None:
    assert set(unilab.utils.__all__) == ALLOWED_UTILS_API


def test_utils_directory_is_whitelisted() -> None:
    modules = {path.stem for path in Path("src/unilab/utils").glob("*.py")}
    assert modules == ALLOWED_UTILS_MODULES


def test_repo_has_no_package_level_utils_imports() -> None:
    current_file = Path(__file__).resolve()
    for root in (Path("src"), Path("tests"), Path("scripts"), Path("benchmark")):
        for path in root.rglob("*.py"):
            if path.resolve() == current_file:
                continue
            assert "from unilab.utils import" not in path.read_text(encoding="utf-8"), path


def test_removed_utils_shims_are_not_importable() -> None:
    for module_name in sorted(f"unilab.utils.{name}" for name in REMOVED_UTILS_SHIMS):
        assert importlib.util.find_spec(module_name) is None, module_name


def test_removed_owner_aliases_are_not_importable() -> None:
    for module_name in sorted(REMOVED_OWNER_ALIASES):
        assert importlib.util.find_spec(module_name) is None, module_name


def test_algos_torch_common_no_longer_reexports_utils_primitives() -> None:
    common = importlib.import_module("unilab.algos.torch.common")
    assert "get_default_device" not in common.__all__
    assert "to_numpy" not in common.__all__
    assert "to_torch" not in common.__all__
