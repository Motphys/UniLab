"""Benchmark env.step performance for a given task and backend.

Usage:
    # Run all task-side benchmark specs over their benchmark backends:
    uv run benchmark/benchmark_env_step.py

    # Single task + backend:
    uv run benchmark/benchmark_env_step.py task=g1_walk_flat/motrix
    uv run benchmark/benchmark_env_step.py task=g1_walk_rough/mujoco

    # Override bench params:
    uv run benchmark/benchmark_env_step.py task=go1_joystick_flat/mujoco num_envs=4096 num_steps=500

    # Save to custom locations:
    uv run benchmark/benchmark_env_step.py --out-json tmp/env_step.json --plot-dir tmp/env_step_plots
"""

import importlib.util
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

plt: Any = None
mpatches: Any = None

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.patches as _mpatches
    import matplotlib.pyplot as _plt

    mpatches = _mpatches
    plt = _plt
except Exception:
    pass

ROOT_DIR = Path(__file__).parent.parent
CORE_DIR = ROOT_DIR / "benchmark" / "core"
DEFAULT_OUTPUT_JSON = ROOT_DIR / "benchmark" / "outputs" / "env_step" / "results.json"


def _load_helper_module(module_name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(module_name, CORE_DIR / relative_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load helper module {module_name} from {relative_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_DEVICE_INFO = _load_helper_module("bench_env_step_device_info", "device_info.py")
_OUTPUT = _load_helper_module("bench_env_step_output", "output.py")

get_device_info_dict = _DEVICE_INFO.get_device_info_dict
get_device_info_line = _DEVICE_INFO.get_device_info_line
save_json = _OUTPUT.save_json

BACKENDS = ["mujoco", "motrix"]


@dataclass(frozen=True)
class TaskConfig:
    task_id: str
    env_name: str
    cfg_factory: Callable[[str], Any]
    env_cls_factory: Callable[[], type]
    backends: tuple[str, ...] = ("mujoco", "motrix")
    aliases: tuple[str, ...] = ()
    cfg_finalizer: Callable[[Any, str], None] | None = None
    include_in_matrix: bool = True

    def build_cfg(self, backend: str) -> Any:
        return self.cfg_factory(backend)

    def finalize_cfg(self, cfg: Any, backend: str) -> None:
        if self.cfg_finalizer is not None:
            self.cfg_finalizer(cfg, backend)

    def matches_task_path(self, task_path: str) -> bool:
        return task_path in {self.task_id, self.env_name, *self.aliases}


def _quadruped_reward_config(reward_cls: type) -> Any:
    return reward_cls(
        scales={
            "tracking_lin_vel": 1.0,
            "tracking_ang_vel": 0.2,
            "lin_vel_z": -5.0,
            "ang_vel_xy": -0.1,
            "base_height": -100.0,
            "action_rate": -0.005,
            "similar_to_default": -0.1,
            "contact": 0.24,
            "swing_feet_z": 4.0,
        },
        tracking_sigma=0.25,
        base_height_target=0.3,
    )


def _g1_walk_reward_config() -> Any:
    from unilab.envs.locomotion.g1.joystick import G1WalkRewardConfig

    return G1WalkRewardConfig(
        scales={
            "tracking_lin_vel": 2.0,
            "tracking_ang_vel": 1.5,
            "penalty_ang_vel_xy": -1.0,
            "penalty_orientation": -10.0,
            "penalty_action_rate": -4.0,
            "pose": -0.5,
            "penalty_feet_ori": -20.0,
            "feet_phase": 5.0,
            "alive": 10.0,
        },
        tracking_sigma=0.25,
        base_height_target=0.754,
        min_base_height=0.3,
        max_tilt_deg=65.0,
        gait_frequency=1.5,
        feet_phase_swing_height=0.09,
        feet_phase_tracking_sigma=0.04,
    )


def _materialize_g1_rough_benchmark_scene() -> str:
    import shutil
    import xml.etree.ElementTree as ET

    source_dir = ROOT_DIR / "src" / "unilab" / "assets" / "robots" / "g1"
    output_dir = Path("/tmp/unilab_benchmark_g1_rough_scene")
    output_dir.mkdir(parents=True, exist_ok=True)

    for name in ("g1.xml",):
        shutil.copy2(source_dir / name, output_dir / name)
    for name in ("assets", "textures", "hfields"):
        shutil.copytree(source_dir / name, output_dir / name, dirs_exist_ok=True)

    tree = ET.parse(source_dir / "scene_rough.xml")
    hfield = tree.getroot().find("./asset/hfield[@name='hfield']")
    if hfield is not None:
        hfield.set("file", "hfields/hfield.png")
    output_path = output_dir / "scene_rough.xml"
    tree.write(output_path)
    return str(output_path)


def _materialize_go2w_rough_tiles_motrix_scene() -> str:
    import shutil
    import xml.etree.ElementTree as ET

    source_dir = ROOT_DIR / "src" / "unilab" / "assets" / "robots" / "go2w"
    assets_dir = ROOT_DIR / "src" / "unilab" / "assets"
    output_dir = Path("/tmp/unilab_benchmark_go2w_rough_tiles_scene")
    output_dir.mkdir(parents=True, exist_ok=True)

    robot_tree = ET.parse(source_dir / "go2w.xml")
    compiler = robot_tree.getroot().find("compiler")
    if compiler is not None:
        compiler.set("meshdir", str((source_dir.parent / "go2" / "assets").resolve()))
    robot_tree.write(output_dir / "go2w.xml")

    terrain_dst = output_dir / "go2w_tile_stairs_5x5.xml"
    shutil.copy2(assets_dir / "terrains" / "go2w_tile_stairs_5x5.xml", terrain_dst)

    scene_tree = ET.parse(source_dir / "scene_rough_tiles.xml")
    root = scene_tree.getroot()
    for include in root.findall("include"):
        include_file = include.get("file", "")
        if include_file == "go2w.xml":
            include.set("file", "go2w.xml")
        elif include_file.endswith("go2w_tile_stairs_5x5.xml"):
            include.set("file", terrain_dst.name)

    asset = root.find("asset")
    if asset is not None:
        for hfield in list(asset.findall("hfield")):
            if hfield.get("name") == "go2w_tile_stairs_5x5":
                asset.remove(hfield)

    worldbody = root.find("worldbody")
    if worldbody is not None:
        for geom in list(worldbody.findall("geom")):
            if geom.get("name") == "terrain_scan_probe":
                worldbody.remove(geom)

    output_path = output_dir / "scene_rough_tiles_motrix.xml"
    scene_tree.write(output_path)
    return str(output_path)


def _materialize_sharpa_motrix_scene() -> str:
    import xml.etree.ElementTree as ET

    source_dir = ROOT_DIR / "src" / "unilab" / "assets" / "robots" / "sharpa_wave"
    output_dir = Path("/tmp/unilab_benchmark_sharpa_scene")
    output_dir.mkdir(parents=True, exist_ok=True)

    robot_tree = ET.parse(source_dir / "right_sharpa_wave.xml")
    robot_root = robot_tree.getroot()
    compiler = robot_root.find("compiler")
    if compiler is not None:
        compiler.set("meshdir", str((source_dir / "meshes").resolve()))

    contact = robot_root.find("contact")
    if contact is not None:
        for exclude in list(contact.findall("exclude")):
            if exclude.get("body1") == exclude.get("body2"):
                contact.remove(exclude)
    robot_tree.write(output_dir / "right_sharpa_wave.xml")

    scene_tree = ET.parse(source_dir / "scene.xml")
    output_path = output_dir / "scene.xml"
    scene_tree.write(output_path)
    return str(output_path)


def _go1_cfg(_: str) -> Any:
    from unilab.envs.locomotion.go1.joystick import Go1JoystickCfg, RewardConfig

    cfg = Go1JoystickCfg()
    cfg.reward_config = _quadruped_reward_config(RewardConfig)
    return cfg


def _go1_env_cls() -> type:
    from unilab.envs.locomotion.go1.joystick import Go1WalkTask

    return Go1WalkTask


def _go2_cfg(backend: str) -> Any:
    from unilab.envs.locomotion.go2.joystick import Go2JoystickCfg, RewardConfig

    cfg = Go2JoystickCfg()
    cfg.reward_config = _quadruped_reward_config(RewardConfig)
    if backend == "motrix":
        cfg.domain_rand.randomize_kp = False
        cfg.domain_rand.randomize_kd = False
    return cfg


def _go2_env_cls() -> type:
    from unilab.envs.locomotion.go2.joystick import Go2WalkTask

    return Go2WalkTask


def _go2_rough_cfg(_: str) -> Any:
    from unilab.envs.locomotion.go2.rough import Go2JoystickRoughCfg, RoughRewardConfig

    cfg = Go2JoystickRoughCfg()
    cfg.scene.terrain.generator.seed = 42
    cfg.scene.terrain.generator.num_rows = 6
    cfg.scene.terrain.generator.num_cols = 6
    cfg.reward_config = RoughRewardConfig(
        scales={
            "lin_vel_z": -2.0,
            "ang_vel_xy": -0.05,
            "joint_torques_l2": -2.5e-5,
            "joint_acc_l2": -2.5e-7,
            "joint_pos_limits": -5.0,
            "joint_power": -2.0e-5,
            "stand_still": -2.0,
            "joint_pos_penalty": -1.0,
            "joint_mirror": -0.05,
            "action_rate": -0.01,
            "undesired_contacts": -1.0,
            "contact_forces": -1.5e-4,
            "tracking_lin_vel": 3.0,
            "tracking_ang_vel": 1.5,
            "feet_air_time": 0.5,
            "feet_air_time_variance": -1.0,
            "feet_contact_without_cmd": 0.1,
            "feet_slide": -0.1,
            "feet_height_body": -5.0,
            "feet_gait": 0.5,
            "upward": 1.0,
        },
        tracking_sigma=0.25,
        base_height_target=0.33,
    )
    return cfg


def _go2_rough_env_cls() -> type:
    from unilab.envs.locomotion.go2.rough import Go2JoystickRoughEnv

    return Go2JoystickRoughEnv


def _go2w_cfg(_: str) -> Any:
    from unilab.envs.locomotion.go2w.joystick import Go2WJoystickCfg, RewardConfig

    cfg = Go2WJoystickCfg()
    cfg.reward_config = _go2w_reward_config(RewardConfig)
    return cfg


def _go2w_rough_tiles_cfg(backend: str) -> Any:
    from unilab.envs.locomotion.go2w.joystick import RewardConfig
    from unilab.envs.locomotion.go2w.rough import Go2WJoystickRoughTilesCfg

    cfg = Go2WJoystickRoughTilesCfg()
    cfg.reward_config = _go2w_reward_config(RewardConfig)
    if backend == "motrix":
        cfg.model_file = _materialize_go2w_rough_tiles_motrix_scene()
        cfg.terrain_scan.enabled = False
    return cfg


def _go2w_reward_config(reward_cls: type) -> Any:
    return reward_cls(
        scales={
            "tracking_lin_vel": 1.0,
            "tracking_ang_vel": 0.2,
            "lin_vel_z": -5.0,
            "ang_vel_xy": -0.1,
            "base_height": -100.0,
            "action_rate": -0.005,
            "similar_to_default": -0.1,
            "torques": -0.0002,
            "wheel_vel": 0.0,
            "alive": 0.5,
        },
        tracking_sigma=0.25,
        base_height_target=0.3,
    )


def _go2w_env_cls() -> type:
    from unilab.envs.locomotion.go2w.joystick import Go2WJoystickEnv

    return Go2WJoystickEnv


def _go2w_rough_tiles_env_cls() -> type:
    from unilab.envs.locomotion.go2w.rough import Go2WJoystickRoughTilesEnv

    return Go2WJoystickRoughTilesEnv


def _g1_flat_cfg(backend: str) -> Any:
    from unilab.envs.locomotion.g1.joystick import G1WalkFlatCfg

    cfg = G1WalkFlatCfg()
    cfg.reward_config = _g1_walk_reward_config()
    if backend == "motrix":
        cfg.domain_rand.randomize_kp = False
        cfg.domain_rand.randomize_kd = False
    return cfg


def _g1_rough_cfg(backend: str) -> Any:
    from unilab.envs.locomotion.g1.joystick import G1WalkRoughCfg

    cfg = G1WalkRoughCfg()
    cfg.reward_config = _g1_walk_reward_config()
    if backend == "motrix":
        cfg.model_file = _materialize_g1_rough_benchmark_scene()
        cfg.domain_rand.randomize_kp = False
        cfg.domain_rand.randomize_kd = False
    return cfg


def _g1_motion_tracking_cfg(_: str) -> Any:
    from unilab.envs.motion_tracking.g1.tracking import G1MotionTrackingEnvCfg

    return G1MotionTrackingEnvCfg()


def _sharpa_inhand_cfg(backend: str) -> Any:
    from unilab.envs.manipulation.sharpa_inhand.rotation import (
        RewardConfig,
        SharpaInhandRotationCfg,
    )

    cfg = SharpaInhandRotationCfg()
    cfg.reward_config = RewardConfig()
    cfg.grasp_cache_path = "/tmp/unilab_benchmark_sharpa_grasp"
    cfg.sensor.tactile_force_sensor_names = [
        "contact_right_thumb_elastomer_force",
        "contact_right_index_elastomer_force",
        "contact_right_middle_elastomer_force",
        "contact_right_ring_elastomer_force",
        "contact_right_pinky_elastomer_force",
    ]
    if backend == "motrix":
        cfg.model_file = _materialize_sharpa_motrix_scene()
        cfg.obs.enable_tactile = False
        cfg.domain_rand.randomize_friction = False
        cfg.domain_rand.randomize_com = False
        cfg.domain_rand.randomize_mass = False
        cfg.domain_rand.force_scale = 0.0
    return cfg


def _ensure_sharpa_benchmark_grasp_cache(cfg: Any, _: str) -> None:
    from unilab.envs.manipulation.sharpa_inhand.base import (
        SOURCE_DEFAULT_HAND_JOINT_POS_DEG,
        resolve_grasp_cache_file,
    )

    if not str(cfg.grasp_cache_path).startswith("/tmp/unilab_benchmark_sharpa_grasp"):
        return

    hand_qpos = np.deg2rad(np.asarray(SOURCE_DEFAULT_HAND_JOINT_POS_DEG, dtype=np.float64))
    object_height = 0.5 * (float(cfg.reset_height_lower) + float(cfg.reset_height_upper))
    object_pose = np.asarray([0.0, 0.0, object_height, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    cache_row = np.concatenate([hand_qpos, object_pose], axis=0)

    for scale_value in np.asarray(cfg.domain_rand.scale_list, dtype=np.float64):
        cache_file = resolve_grasp_cache_file(cfg.grasp_cache_path, float(scale_value))
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_file, cache_row[None, :])


def _g1_walk_env_cls() -> type:
    from unilab.envs.locomotion.g1.joystick import G1WalkEnv

    return G1WalkEnv


def _g1_motion_tracking_env_cls() -> type:
    from unilab.envs.motion_tracking.g1.tracking import G1MotionTrackingEnv

    return G1MotionTrackingEnv


def _sharpa_inhand_env_cls() -> type:
    from unilab.envs.manipulation.sharpa_inhand.rotation import SharpaInhandRotationEnv

    return SharpaInhandRotationEnv


TASK_CONFIGS: dict[str, TaskConfig] = {
    "go1": TaskConfig(
        task_id="go1_joystick_flat",
        env_name="Go1JoystickFlat",
        cfg_factory=_go1_cfg,
        env_cls_factory=_go1_env_cls,
    ),
    "go2": TaskConfig(
        task_id="go2_joystick_flat",
        env_name="Go2JoystickFlat",
        cfg_factory=_go2_cfg,
        env_cls_factory=_go2_env_cls,
    ),
    "go2_rough": TaskConfig(
        task_id="go2_joystick_rough",
        env_name="Go2JoystickRough",
        cfg_factory=_go2_rough_cfg,
        env_cls_factory=_go2_rough_env_cls,
    ),
    "go2w": TaskConfig(
        task_id="go2w_joystick_flat",
        env_name="Go2WJoystickFlat",
        cfg_factory=_go2w_cfg,
        env_cls_factory=_go2w_env_cls,
    ),
    "go2w_rough": TaskConfig(
        task_id="go2w_joystick_rough_tiles",
        env_name="Go2WJoystickRoughTiles",
        cfg_factory=_go2w_rough_tiles_cfg,
        env_cls_factory=_go2w_rough_tiles_env_cls,
    ),
    "g1": TaskConfig(
        task_id="g1_walk_flat",
        env_name="G1WalkFlat",
        cfg_factory=_g1_flat_cfg,
        env_cls_factory=_g1_walk_env_cls,
    ),
    "g1_rough": TaskConfig(
        task_id="g1_walk_rough",
        env_name="G1WalkRough",
        cfg_factory=_g1_rough_cfg,
        env_cls_factory=_g1_walk_env_cls,
        aliases=("sac/g1_walk_rough",),
    ),
    "g1_mt": TaskConfig(
        task_id="g1_motion_tracking",
        env_name="G1MotionTracking",
        cfg_factory=_g1_motion_tracking_cfg,
        env_cls_factory=_g1_motion_tracking_env_cls,
    ),
    "sharpa_inhand": TaskConfig(
        task_id="sharpa_inhand",
        env_name="SharpaInhandRotation",
        cfg_factory=_sharpa_inhand_cfg,
        env_cls_factory=_sharpa_inhand_env_cls,
        cfg_finalizer=_ensure_sharpa_benchmark_grasp_cache,
        include_in_matrix=False,
    ),
}

# Default benchmark parameters
DEFAULT_NUM_ENVS = 2048
DEFAULT_NUM_STEPS = 20
DEFAULT_WARMUP_STEPS = 5

TASK_COLORS = {
    "go1": "#4C78A8",
    "go2": "#54A24B",
    "go2_rough": "#8CD17D",
    "g1": "#F58518",
    "g1_mt": "#B279A2",
    "g1_rough": "#E45756",
    "go2w": "#72B7B2",
    "go2w_rough": "#499894",
    "go2w_rough_tiles": "#499894",
    "sharpa_inhand": "#D37295",
}
BACKEND_STYLES = {
    "mujoco": {"marker": "o", "linestyle": "-", "hatch": "//"},
    "motrix": {"marker": "s", "linestyle": "--", "hatch": "xx"},
}
BACKEND_TICK_LABELS = {
    "mujoco": "mj",
    "motrix": "mx",
}
BREAKDOWN_SEGMENTS = [
    ("apply_action_ms", "apply_action", "#4C78A8"),
    ("backend_set_ctrl_ms", "set_ctrl", "#F58518"),
    ("backend_physics_ms", "physics", "#54A24B"),
    ("backend_refresh_cache_ms", "refresh_cache", "#E45756"),
    ("step_core_other_ms", "step_core_other", "#9D755D"),
    ("update_state_ms", "update_state", "#72B7B2"),
    ("reset_done_ms", "reset_done", "#B279A2"),
    ("env_step_other_ms", "env_step_other", "#BAB0AC"),
]


def _is_matrix_mode(argv: list[str]) -> bool:
    """Return True when user didn't specify task or backend explicitly."""
    for arg in argv:
        if arg.startswith("task=") or arg.startswith("training.sim_backend="):
            return False
    return True


def _parse_cli_args(args: list[str]) -> tuple[dict[str, str], dict[str, Any], list[str]]:
    """Parse benchmark/output args and return (bench_kwargs, output_kwargs, hydra_overrides).

    Benchmark args:
        num_envs=XXX, num_steps=XXX, warmup_steps=XXX
    Output args:
        --out-json PATH / --out-json=PATH / out_json=PATH
        --plot-dir PATH / --plot-dir=PATH / plot_dir=PATH
        --skip-plots / skip_plots=true|false

    Non-Hydra args are extracted before config composition.
    """
    bench_kwargs: dict[str, str] = {}
    output_kwargs: dict[str, Any] = {
        "out_json": None,
        "plot_dir": None,
        "skip_plots": False,
    }
    hydra_overrides: list[str] = []

    i = 0
    while i < len(args):
        arg = args[i]
        if arg.startswith("num_envs="):
            bench_kwargs["num_envs"] = arg.split("=", 1)[1]
        elif arg.startswith("num_steps="):
            bench_kwargs["num_steps"] = arg.split("=", 1)[1]
        elif arg.startswith("warmup_steps="):
            bench_kwargs["warmup_steps"] = arg.split("=", 1)[1]
        elif arg.startswith("out_json="):
            output_kwargs["out_json"] = arg.split("=", 1)[1]
        elif arg.startswith("plot_dir="):
            output_kwargs["plot_dir"] = arg.split("=", 1)[1]
        elif arg.startswith("skip_plots="):
            output_kwargs["skip_plots"] = arg.split("=", 1)[1].lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
        elif arg == "--out-json":
            i += 1
            if i >= len(args):
                raise ValueError("--out-json requires a path")
            output_kwargs["out_json"] = args[i]
        elif arg.startswith("--out-json="):
            output_kwargs["out_json"] = arg.split("=", 1)[1]
        elif arg == "--plot-dir":
            i += 1
            if i >= len(args):
                raise ValueError("--plot-dir requires a path")
            output_kwargs["plot_dir"] = args[i]
        elif arg.startswith("--plot-dir="):
            output_kwargs["plot_dir"] = arg.split("=", 1)[1]
        elif arg == "--skip-plots":
            output_kwargs["skip_plots"] = True
        else:
            hydra_overrides.append(arg)
        i += 1

    return bench_kwargs, output_kwargs, hydra_overrides


def _split_task_choice(task_choice: str) -> tuple[str, str | None]:
    """Return (task path without backend, backend if present)."""
    normalized = task_choice.strip()
    if normalized.startswith("task="):
        normalized = normalized.split("=", 1)[1]
    for backend in BACKENDS:
        suffix = f"/{backend}"
        if normalized.endswith(suffix):
            return normalized[: -len(suffix)], backend
    return normalized, None


def _matching_task_config(task_path: str) -> TaskConfig | None:
    for task_config in TASK_CONFIGS.values():
        if task_config.matches_task_path(task_path):
            return task_config
    return None


def _resolve_task_and_backend(hydra_overrides: list[str]) -> tuple[str, TaskConfig, str]:
    """Resolve benchmark task/backend without consulting training owner YAML."""
    task_key = "go1"
    task_config = TASK_CONFIGS[task_key]
    sim_backend: str | None = None

    for arg in hydra_overrides:
        if arg.startswith("task="):
            task_path, backend = _split_task_choice(arg)
            matched = _matching_task_config(task_path)
            if matched is None:
                available = ", ".join(task_config.task_id for task_config in TASK_CONFIGS.values())
                raise ValueError(f"Unknown benchmark task {task_path!r}. Available: {available}")
            task_config = matched
            task_key = next(
                key for key, candidate in TASK_CONFIGS.items() if candidate is task_config
            )
            sim_backend = backend or sim_backend
        elif arg.startswith("training.sim_backend="):
            sim_backend = arg.split("=", 1)[1]

    if sim_backend is None:
        sim_backend = task_config.backends[0]
    if sim_backend not in task_config.backends:
        supported = ", ".join(task_config.backends)
        raise ValueError(
            f"Benchmark task {task_config.task_id!r} supports backend(s) {supported}; "
            f"got {sim_backend!r}."
        )
    return task_key, task_config, sim_backend


def _dotlist_to_plain_dict(dotlist: list[str]) -> dict[str, Any]:
    if not dotlist:
        return {}
    from omegaconf import OmegaConf

    value = OmegaConf.to_container(OmegaConf.from_dotlist(dotlist), resolve=True)
    if not isinstance(value, dict):
        return {}
    return cast(dict[str, Any], value)


def _deep_merge_dict(target: dict[str, Any], updates: dict[str, Any]) -> None:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_merge_dict(cast(dict[str, Any], target[key]), value)
        else:
            target[key] = value


def _apply_plain_overrides(target: Any, overrides: dict[str, Any]) -> None:
    import dataclasses

    for key, value in overrides.items():
        if not hasattr(target, key):
            raise ValueError(f"Config class '{type(target).__name__}' has no attribute '{key}'")
        existing = getattr(target, key)
        if isinstance(value, dict) and isinstance(existing, dict):
            _deep_merge_dict(existing, value)
        elif isinstance(value, dict) and dataclasses.is_dataclass(existing):
            _apply_plain_overrides(existing, value)
        else:
            setattr(target, key, value)


def _apply_cli_cfg_overrides(cfg: Any, hydra_overrides: list[str]) -> None:
    env_dotlist: list[str] = []
    reward_dotlist: list[str] = []

    for arg in hydra_overrides:
        if arg.startswith("env."):
            env_dotlist.append(arg[len("env.") :])
        elif arg.startswith("+env."):
            env_dotlist.append(arg[len("+env.") :])
        elif arg.startswith("reward."):
            reward_dotlist.append(arg[len("reward.") :])
        elif arg.startswith("+reward."):
            reward_dotlist.append(arg[len("+reward.") :])

    env_overrides = _dotlist_to_plain_dict(env_dotlist)
    if env_overrides:
        _apply_plain_overrides(cfg, env_overrides)

    reward_overrides = _dotlist_to_plain_dict(reward_dotlist)
    if reward_overrides:
        reward_config = getattr(cfg, "reward_config", None)
        if reward_config is None:
            raise ValueError("Cannot apply reward.* overrides: task has no reward_config")
        _apply_plain_overrides(reward_config, reward_overrides)


def _run_single(extra_args: list[str]) -> dict[str, Any]:
    """Run a single task-side benchmark and return timing records."""
    bench_kwargs, _, hydra_overrides = _parse_cli_args(extra_args)
    _, task_config, sim_backend = _resolve_task_and_backend(hydra_overrides)

    num_envs = int(bench_kwargs.get("num_envs", DEFAULT_NUM_ENVS))
    num_steps = int(bench_kwargs.get("num_steps", DEFAULT_NUM_STEPS))
    warmup_steps = int(bench_kwargs.get("warmup_steps", DEFAULT_WARMUP_STEPS))

    cfg = task_config.build_cfg(sim_backend)
    _apply_cli_cfg_overrides(cfg, hydra_overrides)
    task_config.finalize_cfg(cfg, sim_backend)
    cfg.validate()

    env = None
    timing_records: dict[str, list[float]] = {}
    try:
        env_cls = task_config.env_cls_factory()
        env = env_cls(cfg, num_envs=num_envs, backend_type=sim_backend)

        nu = env._backend.num_actuators  # type: ignore[reportAttributeAccessIssue]
        env.init_state()

        for _ in range(warmup_steps):
            actions = np.random.uniform(-1, 1, size=(num_envs, nu)).astype(np.float32)
            env.step(actions)

        for _ in range(num_steps):
            actions = np.random.uniform(-1, 1, size=(num_envs, nu)).astype(np.float32)
            state = env.step(actions)
            timing = state.info.get("timing", {})
            for k, v in timing.items():
                timing_records.setdefault(k, []).append(float(v))
    finally:
        if env is not None:
            env.close()

    return {
        "task_name": task_config.env_name,
        "sim_backend": str(sim_backend),
        "num_envs": num_envs,
        "num_steps": num_steps,
        "warmup_steps": warmup_steps,
        "timing_records": timing_records,
    }


def _result_label(result: dict[str, Any]) -> str:
    return (
        f"{result.get('task_key', _short_task_label(result['task_name']))}/{result['sim_backend']}"
    )


def _compute_result_summary(result: dict[str, Any]) -> dict[str, Any]:
    tr = result["timing_records"]
    total_arr = np.array(tr.get("env_step_total_ms", []), dtype=np.float64)
    total_s = float(total_arr.sum() / 1000.0)
    steps_per_env = result["num_steps"] * result["num_envs"]
    throughput = float(steps_per_env / total_s) if total_s > 0 else 0.0

    summary: dict[str, Any] = {
        "label": _result_label(result),
        "total_time_s": total_s,
        "mean_step_ms": float(total_arr.mean()) if total_arr.size else 0.0,
        "median_step_ms": float(np.median(total_arr)) if total_arr.size else 0.0,
        "std_step_ms": float(total_arr.std()) if total_arr.size else 0.0,
        "min_step_ms": float(total_arr.min()) if total_arr.size else 0.0,
        "max_step_ms": float(total_arr.max()) if total_arr.size else 0.0,
        "throughput_env_steps_per_s": throughput,
        "timing_mean_ms": {},
        "timing_median_ms": {},
    }
    for key, values in tr.items():
        arr = np.array(values, dtype=np.float64)
        summary["timing_mean_ms"][key] = float(arr.mean()) if arr.size else 0.0
        summary["timing_median_ms"][key] = float(np.median(arr)) if arr.size else 0.0
    return summary


def _serialize_result(result: dict[str, Any]) -> dict[str, Any]:
    summary = _compute_result_summary(result)
    return {
        "task_name": result["task_name"],
        "task_key": result.get("task_key"),
        "sim_backend": result["sim_backend"],
        "num_envs": result["num_envs"],
        "num_steps": result["num_steps"],
        "warmup_steps": result["warmup_steps"],
        "plot_data": {
            "label": summary["label"],
            "median_step_ms": summary["median_step_ms"],
            "throughput_env_steps_per_s": summary["throughput_env_steps_per_s"],
            "breakdown_median_ms": {
                key: _median_timing_ms(result, key) for key, _, _ in BREAKDOWN_SEGMENTS
            },
        },
    }


def _print_single_report(result: dict[str, Any]) -> None:
    tr = result["timing_records"]
    summary = _compute_result_summary(result)

    print(f"\n{'=' * 60}")
    print(f"  Task:       {result['task_name']}")
    print(f"  Backend:    {result['sim_backend']}")
    print(f"  Num envs:   {result['num_envs']}")
    print(f"  Steps:      {result['num_steps']} (warmup: {result['warmup_steps']})")
    print(f"{'=' * 60}")
    print(f"  Total time:       {summary['total_time_s']:.3f}s")
    print(f"  Mean step time:   {summary['mean_step_ms']:.3f}ms")
    print(f"  Median step time: {summary['median_step_ms']:.3f}ms")
    print(f"  Std step time:    {summary['std_step_ms']:.3f}ms")
    print(f"  Min step time:    {summary['min_step_ms']:.3f}ms")
    print(f"  Max step time:    {summary['max_step_ms']:.3f}ms")
    print(f"  Throughput:       {summary['throughput_env_steps_per_s']:.0f} env-steps/s")
    if tr:
        print(f"{'- ' * 30}")
        print("  Breakdown:")
        for k, v in tr.items():
            if k == "env_step_total_ms":
                continue
            arr = np.array(v)
            print(f"    {k:25s}  mean={arr.mean():.3f}ms  median={np.median(arr):.3f}ms")
    print(f"{'=' * 60}")


def _short_task_label(task_name: str) -> str:
    """Shorten 'Go1JoystickFlat' → 'go1'."""
    name = task_name.lower()
    if "motiontracking" in name:
        return "g1_mt"
    if name.startswith("sharpainhand"):
        return "sharpa_inhand"
    if "rough" in name and name.startswith("g1"):
        return "g1_rough"
    if name.startswith("go2w") and "roughtiles" in name:
        return "go2w_rough_tiles"
    if name.startswith("go2w"):
        return "go2w"
    for prefix in ("go1", "go2", "g1"):
        if name.startswith(prefix):
            return prefix
    return task_name[:8]


def _print_comparison_table(results: list[dict[str, Any]]) -> None:
    rows_spec: list[tuple[str, str]] = [
        # (display_label, timing_key_or_special)
        ("total", "env_step_total_ms"),
        ("  apply_action", "apply_action_ms"),
        ("  set_ctrl", "backend_set_ctrl_ms"),
        ("  physics", "backend_physics_ms"),
        ("  refresh_cache", "backend_refresh_cache_ms"),
        ("  step_core_other", "step_core_other_ms"),
        ("  update_state", "update_state_ms"),
        ("  reset_done", "reset_done_ms"),
        ("  env_step_other", "env_step_other_ms"),
        ("throughput", "__throughput__"),
    ]

    # Build short column labels
    col_labels = [
        f"{r.get('task_key', _short_task_label(r['task_name']))}/{r['sim_backend']}"
        for r in results
    ]

    metric_w = 16
    col_w = max(12, max(len(c) for c in col_labels) + 2)

    def hline(left: str, mid: str, right: str, fill: str = "─") -> str:
        return left + fill * metric_w + mid + mid.join(fill * col_w for _ in results) + right

    # Header
    print()
    print(hline("┌", "┬", "┐"))
    header = "│" + "metric".center(metric_w) + "│"
    header += "│".join(c.center(col_w) for c in col_labels) + "│"
    print(header)

    unit_row = "│" + "(median ms)".center(metric_w) + "│"
    unit_row += "│".join(" " * col_w for _ in results) + "│"
    print(unit_row)
    print(hline("├", "┼", "┤"))

    # Data rows
    for label, key in rows_spec:
        cells: list[str] = []
        if key == "__throughput__":
            for r in results:
                total_arr = _timing_array(r, "env_step_total_ms")
                total_s = total_arr.sum() / 1000.0
                steps_per_env = r["num_steps"] * r["num_envs"]
                cells.append(f"{steps_per_env / total_s:,.0f}" if total_s > 0 else "-")
        else:
            for r in results:
                arr = _timing_array(r, key)
                cells.append(f"{np.median(arr):.3f}" if arr.size else "-")

        row = "│" + label.ljust(metric_w) + "│"
        row += "│".join(v.rjust(col_w - 1) + " " for v in cells) + "│"

        # Separator before throughput
        if key == "__throughput__":
            print(hline("├", "┼", "┤"))

        print(row)

    print(hline("└", "┴", "┘"))
    if results:
        r0 = results[0]
        print(f"  ({r0['num_envs']} envs, {r0['num_steps']} steps, throughput = env-steps/s)")


def _task_key(result: dict[str, Any]) -> str:
    return cast(str, result.get("task_key", _short_task_label(result["task_name"])))


def _task_color(task_key: str) -> str:
    return TASK_COLORS.get(task_key, "#4C78A8")


def _backend_style(backend: str) -> dict[str, str]:
    return BACKEND_STYLES.get(backend, {"marker": "o", "linestyle": "-", "hatch": ""})


def _ordered_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    task_order = {task_key: idx for idx, task_key in enumerate(TASK_CONFIGS)}
    backend_order = {backend: idx for idx, backend in enumerate(BACKENDS)}
    return sorted(
        results,
        key=lambda result: (
            task_order.get(_task_key(result), len(task_order)),
            backend_order.get(result["sim_backend"], len(backend_order)),
            result["task_name"],
        ),
    )


def _align_timing_array(arr: np.ndarray, target_size: int) -> np.ndarray:
    if arr.size == target_size:
        return arr
    if arr.size == 0:
        return np.zeros(target_size, dtype=np.float64)
    if arr.size > target_size:
        return arr[:target_size]
    padded = np.zeros(target_size, dtype=np.float64)
    padded[: arr.size] = arr
    return padded


def _timing_array(result: dict[str, Any], key: str) -> np.ndarray:
    timing_records = result["timing_records"]
    if key in timing_records:
        return np.asarray(timing_records[key], dtype=np.float64)

    if key == "step_core_other_ms":
        step_core = _timing_array(result, "step_core_ms")
        if step_core.size == 0:
            return step_core
        backend_total = (
            _align_timing_array(_timing_array(result, "backend_set_ctrl_ms"), step_core.size)
            + _align_timing_array(_timing_array(result, "backend_physics_ms"), step_core.size)
            + _align_timing_array(_timing_array(result, "backend_refresh_cache_ms"), step_core.size)
        )
        return cast(np.ndarray, np.clip(step_core - backend_total, a_min=0.0, a_max=None))

    if key == "env_step_other_ms":
        total = _timing_array(result, "env_step_total_ms")
        if total.size == 0:
            return total
        measured = (
            _align_timing_array(_timing_array(result, "apply_action_ms"), total.size)
            + _align_timing_array(_timing_array(result, "step_core_ms"), total.size)
            + _align_timing_array(_timing_array(result, "update_state_ms"), total.size)
            + _align_timing_array(_timing_array(result, "reset_done_ms"), total.size)
        )
        return cast(np.ndarray, np.clip(total - measured, a_min=0.0, a_max=None))

    return np.array([], dtype=np.float64)


def _median_timing_ms(result: dict[str, Any], key: str) -> float:
    arr = _timing_array(result, key)
    return float(np.median(arr)) if arr.size else 0.0


def _grouped_positions(
    results: list[dict[str, Any]],
    *,
    bar_spacing: float = 1.0,
    group_gap: float = 1.1,
) -> tuple[list[dict[str, Any]], np.ndarray, list[dict[str, Any]]]:
    ordered = _ordered_results(results)
    positions: list[float] = []
    groups: list[dict[str, Any]] = []
    cursor = 0.0
    group_start = 0

    for idx, result in enumerate(ordered):
        positions.append(cursor)
        task_key = _task_key(result)
        next_result = ordered[idx + 1] if idx + 1 < len(ordered) else None
        cursor += bar_spacing
        if next_result is None or _task_key(next_result) != task_key:
            group_positions = positions[group_start : idx + 1]
            groups.append(
                {
                    "task_key": task_key,
                    "label": task_key,
                    "start": group_positions[0],
                    "end": group_positions[-1],
                    "center": float(sum(group_positions) / len(group_positions)),
                }
            )
            group_start = idx + 1
            if next_result is not None:
                cursor += group_gap

    return ordered, np.array(positions, dtype=np.float64), groups


def _decorate_grouped_xaxis(
    ax,
    ordered_results: list[dict[str, Any]],
    positions: np.ndarray,
    groups: list[dict[str, Any]],
) -> None:
    if not ordered_results:
        return

    ax.set_xticks(positions)
    ax.set_xticklabels(
        [
            BACKEND_TICK_LABELS.get(result["sim_backend"], result["sim_backend"])
            for result in ordered_results
        ],
        rotation=0,
        ha="center",
        fontsize=9,
    )

    for idx, group in enumerate(groups):
        ax.axvspan(
            group["start"] - 0.48,
            group["end"] + 0.48,
            color=_task_color(group["task_key"]),
            alpha=0.05,
            zorder=0,
        )
        ax.text(
            group["center"],
            -0.18,
            group["label"],
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=9,
            fontweight="bold",
        )
        if idx < len(groups) - 1:
            boundary = (group["end"] + groups[idx + 1]["start"]) / 2.0
            ax.axvline(boundary, color="#9AA0A6", linewidth=1.0, alpha=0.7)

    ax.set_xlim(float(positions[0] - 0.7), float(positions[-1] + 0.7))


def _backend_legend_handles() -> list[Any]:
    if mpatches is None:
        return []
    return [
        mpatches.Patch(
            facecolor="#FFFFFF",
            edgecolor="#444444",
            hatch=_backend_style(backend)["hatch"],
            label=backend,
        )
        for backend in BACKENDS
    ]


def _breakdown_segments_for_results(results: list[dict[str, Any]]) -> list[tuple[str, str, str]]:
    visible_segments: list[tuple[str, str, str]] = []
    for key, label, color in BREAKDOWN_SEGMENTS:
        if key in {"step_core_other_ms", "env_step_other_ms"}:
            if max((_median_timing_ms(result, key) for result in results), default=0.0) <= 0.05:
                continue
        visible_segments.append((key, label, color))
    return visible_segments


def _save_summary_plot(results: list[dict[str, Any]], output_path: Path) -> bool:
    if plt is None or not results:
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ordered, x, groups = _grouped_positions(results)
    summaries = [_compute_result_summary(result) for result in ordered]
    labels = [summary["label"] for summary in summaries]
    width = 0.78

    fig, axes = plt.subplots(1, 2, figsize=(max(10, len(labels) * 1.6), 5.5))
    median_ms = [summary["median_step_ms"] for summary in summaries]
    throughput = [summary["throughput_env_steps_per_s"] for summary in summaries]

    for idx, result in enumerate(ordered):
        style = _backend_style(result["sim_backend"])
        color = _task_color(_task_key(result))
        axes[0].bar(
            x[idx],
            median_ms[idx],
            width=width,
            color=color,
            edgecolor="#444444",
            hatch=style["hatch"],
            alpha=0.92,
        )
        axes[1].bar(
            x[idx],
            throughput[idx],
            width=width,
            color=color,
            edgecolor="#444444",
            hatch=style["hatch"],
            alpha=0.92,
        )

    axes[0].set_ylabel("median env.step time (ms)")
    axes[0].set_title("Latency")
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].set_ylabel("throughput (env-steps/s)")
    axes[1].set_title("Throughput")
    axes[1].grid(axis="y", alpha=0.3)

    for ax in axes:
        _decorate_grouped_xaxis(ax, ordered, x, groups)

    fig.suptitle(f"Env step benchmark summary\n{get_device_info_line()}")
    handles = _backend_legend_handles()
    if handles:
        axes[1].legend(handles=handles, title="backend", loc="upper right")
    fig.subplots_adjust(bottom=0.24, top=0.82, wspace=0.22)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path.resolve()}")
    return True


def _save_breakdown_plot(results: list[dict[str, Any]], output_path: Path) -> bool:
    if plt is None or not results:
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ordered, x, groups = _grouped_positions(results)
    width = 0.78
    cumulative = np.zeros(len(ordered), dtype=np.float64)
    segments = _breakdown_segments_for_results(ordered)

    fig, ax = plt.subplots(figsize=(max(10, len(ordered) * 1.6), 5.8))
    for timing_key, display_name, color in segments:
        values_np = np.array(
            [_median_timing_ms(result, timing_key) for result in ordered], dtype=np.float64
        )
        for idx, result in enumerate(ordered):
            style = _backend_style(result["sim_backend"])
            ax.bar(
                x[idx],
                values_np[idx],
                width,
                bottom=cumulative[idx],
                label=display_name if idx == 0 else "_nolegend_",
                color=color,
                edgecolor="#444444",
                hatch=style["hatch"],
                alpha=0.94,
            )
        cumulative += values_np

    _decorate_grouped_xaxis(ax, ordered, x, groups)
    ax.set_ylabel("median step time (ms)")
    ax.set_title(f"Env step median breakdown\n{get_device_info_line()}")
    ax.grid(axis="y", alpha=0.3)
    handles, legend_labels = ax.get_legend_handles_labels()
    backend_handles = _backend_legend_handles()
    if backend_handles:
        handles.extend(backend_handles)
        legend_labels.extend([handle.get_label() for handle in backend_handles])
    ax.legend(handles, legend_labels, ncol=2, fontsize=8, loc="upper left")
    fig.subplots_adjust(bottom=0.24, top=0.86)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path.resolve()}")
    return True


def _persist_outputs(
    results: list[dict[str, Any]],
    *,
    mode: str,
    out_json: Path,
    plot_dir: Path | None,
    skip_plots: bool,
    failures: list[dict[str, str]] | None = None,
) -> None:
    plot_files: list[str] = []
    effective_plot_dir = plot_dir or out_json.parent

    if not skip_plots:
        summary_path = effective_plot_dir / "env_step_summary.png"
        breakdown_path = effective_plot_dir / "env_step_breakdown.png"

        if _save_summary_plot(results, summary_path):
            plot_files.append(str(summary_path.resolve()))
        elif plt is None and results:
            print("matplotlib unavailable; skipped env step summary plot.")

        if _save_breakdown_plot(results, breakdown_path):
            plot_files.append(str(breakdown_path.resolve()))
        elif plt is None and results:
            print("matplotlib unavailable; skipped env step breakdown plot.")

    meta: dict[str, Any] = {
        "mode": mode,
        "device_info": get_device_info_dict(),
        "matplotlib_available": plt is not None,
        "plot_files": plot_files,
    }
    if failures:
        meta["failures"] = failures

    save_json(out_json, [_serialize_result(result) for result in results], meta)


def _run_matrix(
    extra_args: list[str], *, out_json: Path, plot_dir: Path | None, skip_plots: bool
) -> None:
    """Run all task x backend combinations and print comparison."""
    from unilab.base.backend.motrix_backend import MOTRIX_AVAILABLE

    backends = BACKENDS if MOTRIX_AVAILABLE else ["mujoco"]
    if not MOTRIX_AVAILABLE:
        print("Note: motrixsim not available, running mujoco only\n")

    results: list[dict] = []
    failures: list[dict[str, str]] = []
    for task_key, task_config in TASK_CONFIGS.items():
        if not task_config.include_in_matrix:
            continue
        for backend in task_config.backends:
            if backend not in backends:
                continue
            label = f"{task_key}/{backend}"
            print(f"Running {label} ...", flush=True)
            try:
                args = [f"task={task_config.task_id}/{backend}"] + extra_args
                result = _run_single(args)
                result["task_key"] = task_key
                results.append(result)
                _print_single_report(result)
            except Exception as e:
                print(f"  FAILED: {e}\n")
                failures.append({"label": label, "error": str(e)})

    if len(results) > 1:
        _print_comparison_table(results)
    _persist_outputs(
        results,
        mode="matrix",
        out_json=out_json,
        plot_dir=plot_dir,
        skip_plots=skip_plots,
        failures=failures,
    )


def main() -> None:
    argv = sys.argv[1:]
    _, output_kwargs, _ = _parse_cli_args(argv)
    out_json = Path(output_kwargs["out_json"]) if output_kwargs["out_json"] else DEFAULT_OUTPUT_JSON
    plot_dir = Path(output_kwargs["plot_dir"]) if output_kwargs["plot_dir"] else None
    skip_plots = bool(output_kwargs["skip_plots"])

    if _is_matrix_mode(argv):
        _run_matrix(argv, out_json=out_json, plot_dir=plot_dir, skip_plots=skip_plots)
    else:
        result = _run_single(argv)
        _print_single_report(result)
        _persist_outputs(
            [result],
            mode="single",
            out_json=out_json,
            plot_dir=plot_dir,
            skip_plots=skip_plots,
        )


if __name__ == "__main__":
    main()
