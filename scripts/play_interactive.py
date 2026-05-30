"""MuJoCo-only interactive play script for trained RSL-RL policies.

This tool opens a live MuJoCo viewer for a trained RSL-RL policy. It is wired
directly to MuJoCo viewer/runtime APIs and is not available for Motrix tasks.

Usage:
    # Load the latest checkpoint for a task/backend owner config
    uv run scripts/play_interactive.py task=go2_joystick_flat/mujoco

    # Load a specific run
    uv run scripts/play_interactive.py task=go2_joystick_flat/mujoco algo.load_run=2024-02-04_12-00-00
    uv run scripts/play_interactive.py task=go2_joystick_rough/mujoco   interactive.action_mode=policy interactive.keyboard=true

    # Show target bodies / reward debug overlays
    uv run scripts/play_interactive.py task=g1_motion_tracking/mujoco \
      interactive.show_target_bodies=true \
      interactive.target_show_axes=true \
      interactive.show_reward_debug=true

Camera controls (MuJoCo viewer):
    Mouse drag     - rotate
    Scroll         - zoom
    Right-drag     - pan
"""

# pyright: reportAttributeAccessIssue=false, reportArgumentType=false, reportOptionalMemberAccess=false, reportOptionalSubscript=false

import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

ROOT_DIR = Path(__file__).parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from unilab.training import (
    ensure_registries,
    get_entrypoint_log_root,
    resolve_task_checkpoint_path,
)
from unilab.training.rsl_rl import (
    RslRlVecEnvWrapper,
    get_policy_obs_dims,
    normalize_ppo_train_cfg,
)
from unilab.visualization.interactive_playback import (
    KeyboardCommander,
    PlaybackControls,
    RslRlPlaybackConfig,
    create_rsl_rl_playback_session,
    prepare_motion_overlay_selection,
    select_torch_device,
)

_KEY_ENTER, _KEY_KP_ENTER = 257, 335
_KEY_BACKSPACE = 259
_KEY_RIGHT, _KEY_LEFT, _KEY_DOWN, _KEY_UP = 262, 263, 264, 265

ensure_registries()

from unilab.base import registry
from unilab.base.scene import SceneCfg
from unilab.structured_configs import PPOConfig as _StructuredPPOConfig

PPOConfig = _StructuredPPOConfig
_PLAYBACK_ENV_UNAVAILABLE = "playback_env_unavailable"

try:
    from rsl_rl.runners import OnPolicyRunner
except ImportError:
    print("Could not import rsl_rl. Please ensure it is installed.")
    sys.exit(1)

import mujoco
import mujoco.viewer


@dataclass
class PlayInteractiveArgs:
    task: str
    load_run: str
    checkpoint: str | None
    action_mode: str
    policy_obs_mode: str
    algo_log_name: str
    log_root: str | None
    show_target_bodies: bool
    show_reward_debug: bool
    target_show_axes: bool
    target_body_names: str
    target_max_bodies: int
    target_marker_radius: float
    target_axis_length: float
    target_marker_alpha: float
    reward_debug_show_velocity: bool
    reward_debug_lin_vel_scale: float
    reward_debug_ang_vel_scale: float
    reward_debug_show_connectors: bool
    reward_debug_show_global_anchor: bool
    camera_follow_body: bool
    camera_focus_body_name: str
    camera_height_offset: float
    camera_distance: float | None
    camera_elevation: float | None
    camera_azimuth: float | None
    use_env_visual_model: bool
    speed: float
    start_paused: bool
    keyboard: bool = False
    keyboard_step_lin: float = 0.1
    keyboard_step_ang: float = 0.2


def _infer_checkpoint_actor_input_dim(ckpt_path: str) -> int | None:
    loaded = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    state_dict = loaded.get("actor_state_dict")
    if not isinstance(state_dict, dict):
        return None

    # Common rsl-rl naming: "mlp.0.weight" or nested prefixes ending with ".0.weight".
    for key in ("mlp.0.weight", "actor.mlp.0.weight"):
        w = state_dict.get(key)
        if isinstance(w, torch.Tensor) and w.ndim == 2:
            return int(w.shape[1])

    for key, w in state_dict.items():
        if key.endswith(".0.weight") and isinstance(w, torch.Tensor) and w.ndim == 2:
            return int(w.shape[1])
    return None


def _backend_adapter(cfg: DictConfig):
    from unilab.base.backend.mujoco.xml import materialize_scene_visual_override
    from unilab.training import BackendAdapter

    return BackendAdapter(
        cfg,
        root_dir=ROOT_DIR,
        algo_name="ppo",
        scene_materializer=materialize_scene_visual_override,
    )


def _algo_config_dict(cfg: DictConfig | None) -> dict[str, Any]:
    """Return the composed PPO algo config as a plain dict.

    Args:
        cfg: Hydra config for the current playback run, or ``None`` when the
            script is driven through its legacy non-Hydra path.

    Returns:
        The resolved ``cfg.algo`` subtree as a mutable dict for rsl_rl.
    """
    if cfg is None:
        return cast(dict[str, Any], PPOConfig().to_dict())
    train_cfg_raw = OmegaConf.to_container(cfg.algo, resolve=True)
    if not isinstance(train_cfg_raw, dict):
        raise TypeError("cfg.algo must resolve to a dict")
    return cast(dict[str, Any], train_cfg_raw)


# ---------------------------------------------------------------------------
# Checkpoint resolution helpers
# ---------------------------------------------------------------------------


def resolve_checkpoint(
    task: str,
    load_run: str,
    checkpoint: str | None = None,
    algo_log_name: str = "rsl_rl_ppo",
    log_root: str | None = None,
) -> str | None:
    checkpoint_path, checkpoint_dir = resolve_task_checkpoint_path(
        ROOT_DIR,
        task_name=task,
        load_run=load_run,
        algo_log_name=algo_log_name,
        checkpoint=checkpoint,
        log_root=log_root,
    )
    if checkpoint_path is None:
        if checkpoint is not None and checkpoint_dir is not None:
            checkpoint_name = (
                f"model_{checkpoint}.pt" if str(checkpoint).isdigit() else str(checkpoint)
            )
            print(f"[play_interactive] Checkpoint not found: {checkpoint_dir / checkpoint_name}")
        elif checkpoint_dir is not None:
            print(f"[play_interactive] No model_*.pt files in {checkpoint_dir}")
        else:
            print(f"[play_interactive] Run not found for load_run={load_run}")
        return None

    print(f"[play_interactive] Loading checkpoint: {checkpoint_path}")
    return str(checkpoint_path)


# ---------------------------------------------------------------------------
# Interactive play
# ---------------------------------------------------------------------------


def _quat_to_rotmat_wxyz(quat: np.ndarray) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = q / n
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _add_sphere_marker(scene, pos: np.ndarray, radius: float, rgba: np.ndarray) -> bool:
    if scene.ngeom >= scene.maxgeom:
        return False
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([radius, 0.0, 0.0], dtype=np.float64),
        np.asarray(pos, dtype=np.float64),
        np.eye(3, dtype=np.float64).reshape(-1),
        rgba,
    )
    scene.ngeom += 1
    return True


def _add_axis_arrow(scene, p0: np.ndarray, p1: np.ndarray, width: float, rgba: np.ndarray) -> bool:
    if scene.ngeom >= scene.maxgeom:
        return False
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_ARROW,
        np.zeros((3,), dtype=np.float64),
        np.zeros((3,), dtype=np.float64),
        np.eye(3, dtype=np.float64).reshape(-1),
        rgba,
    )
    mujoco.mjv_connector(
        geom,
        mujoco.mjtGeom.mjGEOM_ARROW,
        width,
        np.asarray(p0, dtype=np.float64),
        np.asarray(p1, dtype=np.float64),
    )
    scene.ngeom += 1
    return True


def _add_vector_arrow(
    scene,
    origin: np.ndarray,
    vector: np.ndarray,
    scale: float,
    width: float,
    rgba: np.ndarray,
    min_len: float = 1e-6,
) -> bool:
    vec = np.asarray(vector, dtype=np.float64)
    length = float(np.linalg.norm(vec))
    if length < min_len:
        return True
    p0 = np.asarray(origin, dtype=np.float64)
    p1 = p0 + vec * scale
    return _add_axis_arrow(scene, p0, p1, width, rgba)


def _resolve_focus_body_id(mj_model, env, preferred_name: str) -> int:
    candidate_names: list[str] = []
    if preferred_name.strip():
        candidate_names.append(preferred_name.strip())

    cfg = getattr(env, "cfg", None)
    asset = getattr(cfg, "asset", None) if cfg is not None else None
    if asset is not None and getattr(asset, "base_name", None):
        candidate_names.append(str(asset.base_name))
    if cfg is not None and getattr(cfg, "base_name", None):
        candidate_names.append(str(cfg.base_name))

    candidate_names.extend(["base", "trunk", "pelvis", "torso", "torso_link"])

    for name in candidate_names:
        try:
            body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, name)
        except Exception:
            body_id = -1
        if body_id >= 0:
            return int(body_id)

    nbody = int(getattr(mj_model, "nbody", 1))
    return 1 if nbody > 1 else 0


def _available_backends_for_task(task_name: str) -> tuple[str, ...]:
    envs = registry.list_registered_envs()
    task_meta = envs.get(task_name, {})
    backends = task_meta.get("available_backends", ())
    if not isinstance(backends, list):
        return ()
    return tuple(str(backend) for backend in backends)


def _can_launch_glfw_viewer() -> bool:
    try:
        import glfw
    except Exception:
        return True

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ok = bool(glfw.init())
    if ok:
        glfw.terminate()
    return ok


def _uses_native_mujoco_viewer_launch() -> bool:
    launch_fn = getattr(mujoco.viewer, "launch_passive", None)
    module_name = str(getattr(launch_fn, "__module__", ""))
    return module_name.startswith("mujoco")


def _render_motion_targets(
    viewer,
    motion_data,
    selected_indices: np.ndarray,
    marker_radius: float,
    marker_alpha: float,
    show_axes: bool,
    axis_length: float,
) -> None:
    scene = viewer.user_scn
    scene.ngeom = 0

    if motion_data is None:
        return

    body_pos = motion_data.body_pos_w[0]
    body_quat = motion_data.body_quat_w[0]

    point_rgba = np.array([0.1, 0.85, 1.0, marker_alpha], dtype=np.float32)
    x_rgba = np.array([1.0, 0.35, 0.35, marker_alpha], dtype=np.float32)
    y_rgba = np.array([0.35, 1.0, 0.35, marker_alpha], dtype=np.float32)
    z_rgba = np.array([0.35, 0.55, 1.0, marker_alpha], dtype=np.float32)

    for idx in selected_indices:
        if idx < 0 or idx >= body_pos.shape[0]:
            continue

        p = body_pos[idx]
        if not _add_sphere_marker(scene, p, marker_radius, point_rgba):
            break

        if show_axes:
            rot = _quat_to_rotmat_wxyz(body_quat[idx])
            px = p + rot[:, 0] * axis_length
            py = p + rot[:, 1] * axis_length
            pz = p + rot[:, 2] * axis_length
            if not _add_axis_arrow(scene, p, px, marker_radius * 0.4, x_rgba):
                break
            if not _add_axis_arrow(scene, p, py, marker_radius * 0.4, y_rgba):
                break
            if not _add_axis_arrow(scene, p, pz, marker_radius * 0.4, z_rgba):
                break


def _render_reward_debug_targets(
    viewer,
    info: dict,
    selected_indices: np.ndarray,
    marker_radius: float,
    marker_alpha: float,
    show_axes: bool,
    axis_length: float,
    show_vel: bool,
    lin_vel_scale: float,
    ang_vel_scale: float,
    show_connectors: bool,
    show_global_anchor: bool,
) -> None:
    scene = viewer.user_scn
    scene.ngeom = 0

    if not info:
        return

    motion_data = info.get("motion_data", None)
    robot_body_pos_w = info.get("robot_body_pos_w", None)
    robot_body_quat_w = info.get("robot_body_quat_w", None)
    robot_body_lin_vel_w = info.get("robot_body_lin_vel_w", None)
    robot_body_ang_vel_w = info.get("robot_body_ang_vel_w", None)
    ref_body_pos_w = info.get("reward_ref_body_pos_w", None)
    ref_body_quat_w = info.get("reward_ref_body_quat_w", None)

    if motion_data is None or robot_body_pos_w is None or robot_body_quat_w is None:
        return
    if ref_body_pos_w is None or ref_body_quat_w is None:
        return

    robot_pos = robot_body_pos_w[0]
    robot_quat = robot_body_quat_w[0]
    ref_pos = ref_body_pos_w[0]
    ref_quat = ref_body_quat_w[0]
    motion_pos = motion_data.body_pos_w[0]
    motion_quat = motion_data.body_quat_w[0]
    motion_lin_vel = motion_data.body_lin_vel_w[0]
    motion_ang_vel = motion_data.body_ang_vel_w[0]
    robot_lin_vel = robot_body_lin_vel_w[0] if robot_body_lin_vel_w is not None else None
    robot_ang_vel = robot_body_ang_vel_w[0] if robot_body_ang_vel_w is not None else None

    ref_rgba = np.array([0.15, 0.95, 0.95, marker_alpha], dtype=np.float32)
    robot_rgba = np.array([1.0, 0.62, 0.15, marker_alpha], dtype=np.float32)
    motion_rgba = np.array([0.55, 0.55, 1.0, marker_alpha * 0.7], dtype=np.float32)
    connector_rgba = np.array([1.0, 1.0, 1.0, marker_alpha * 0.55], dtype=np.float32)

    ref_lin_vel_rgba = np.array([0.0, 1.0, 1.0, marker_alpha], dtype=np.float32)
    robot_lin_vel_rgba = np.array([1.0, 0.95, 0.1, marker_alpha], dtype=np.float32)
    ref_ang_vel_rgba = np.array([0.45, 0.75, 1.0, marker_alpha], dtype=np.float32)
    robot_ang_vel_rgba = np.array([1.0, 0.45, 0.1, marker_alpha], dtype=np.float32)

    x_rgba = np.array([1.0, 0.35, 0.35, marker_alpha], dtype=np.float32)
    y_rgba = np.array([0.35, 1.0, 0.35, marker_alpha], dtype=np.float32)
    z_rgba = np.array([0.35, 0.55, 1.0, marker_alpha], dtype=np.float32)

    for idx in selected_indices:
        if idx < 0 or idx >= robot_pos.shape[0]:
            continue

        p_ref = ref_pos[idx]
        p_robot = robot_pos[idx]
        p_motion = motion_pos[idx]

        if not _add_sphere_marker(scene, p_ref, marker_radius, ref_rgba):
            break
        if not _add_sphere_marker(scene, p_robot, marker_radius, robot_rgba):
            break
        if not _add_sphere_marker(scene, p_motion, marker_radius * 0.85, motion_rgba):
            break

        if show_connectors:
            if not _add_axis_arrow(scene, p_ref, p_robot, marker_radius * 0.2, connector_rgba):
                break

        if show_axes:
            for p, q in (
                (p_ref, ref_quat[idx]),
                (p_robot, robot_quat[idx]),
                (p_motion, motion_quat[idx]),
            ):
                rot = _quat_to_rotmat_wxyz(q)
                px = p + rot[:, 0] * axis_length
                py = p + rot[:, 1] * axis_length
                pz = p + rot[:, 2] * axis_length
                if not _add_axis_arrow(scene, p, px, marker_radius * 0.35, x_rgba):
                    break
                if not _add_axis_arrow(scene, p, py, marker_radius * 0.35, y_rgba):
                    break
                if not _add_axis_arrow(scene, p, pz, marker_radius * 0.35, z_rgba):
                    break

        if show_vel:
            if not _add_vector_arrow(
                scene,
                p_motion,
                motion_lin_vel[idx],
                lin_vel_scale,
                marker_radius * 0.28,
                ref_lin_vel_rgba,
            ):
                break
            if robot_lin_vel is not None and not _add_vector_arrow(
                scene,
                p_robot,
                robot_lin_vel[idx],
                lin_vel_scale,
                marker_radius * 0.28,
                robot_lin_vel_rgba,
            ):
                break
            if not _add_vector_arrow(
                scene,
                p_motion,
                motion_ang_vel[idx],
                ang_vel_scale,
                marker_radius * 0.24,
                ref_ang_vel_rgba,
            ):
                break
            if robot_ang_vel is not None and not _add_vector_arrow(
                scene,
                p_robot,
                robot_ang_vel[idx],
                ang_vel_scale,
                marker_radius * 0.24,
                robot_ang_vel_rgba,
            ):
                break

    if show_global_anchor:
        anchor_idx = int(info.get("anchor_body_idx", 0))
        if 0 <= anchor_idx < robot_pos.shape[0]:
            anchor_motion_p = motion_pos[anchor_idx]
            anchor_motion_q = motion_quat[anchor_idx]
            anchor_robot_p = robot_pos[anchor_idx]
            anchor_robot_q = robot_quat[anchor_idx]

            anchor_motion_rgba = np.array([0.95, 0.2, 0.95, marker_alpha], dtype=np.float32)
            anchor_robot_rgba = np.array([1.0, 0.2, 0.2, marker_alpha], dtype=np.float32)
            anchor_radius = marker_radius * 1.35

            _add_sphere_marker(scene, anchor_motion_p, anchor_radius, anchor_motion_rgba)
            _add_sphere_marker(scene, anchor_robot_p, anchor_radius, anchor_robot_rgba)
            _add_axis_arrow(
                scene,
                anchor_motion_p,
                anchor_robot_p,
                marker_radius * 0.3,
                connector_rgba,
            )

            if show_axes:
                for p, q in ((anchor_motion_p, anchor_motion_q), (anchor_robot_p, anchor_robot_q)):
                    rot = _quat_to_rotmat_wxyz(q)
                    px = p + rot[:, 0] * (axis_length * 1.25)
                    py = p + rot[:, 1] * (axis_length * 1.25)
                    pz = p + rot[:, 2] * (axis_length * 1.25)
                    _add_axis_arrow(scene, p, px, marker_radius * 0.45, x_rgba)
                    _add_axis_arrow(scene, p, py, marker_radius * 0.45, y_rgba)
                    _add_axis_arrow(scene, p, pz, marker_radius * 0.45, z_rgba)


def _load_viewer_model(env: Any, *, use_env_visual_model: bool):
    import mujoco

    backend = getattr(env, "_backend", None)
    backend_visual_model_file = getattr(backend, "scene_visual_model_file", None)
    if backend_visual_model_file:
        print(
            f"[play_interactive] Using backend visual model for viewer: {backend_visual_model_file}"
        )
        return mujoco.MjModel.from_xml_path(str(backend_visual_model_file))

    if use_env_visual_model:
        cfg_scene = getattr(getattr(env, "cfg", None), "scene", None)
        if cfg_scene is not None and not isinstance(cfg_scene, SceneCfg):
            raise TypeError("env.cfg.scene must be a SceneCfg")
        model_file = None if cfg_scene is None else cfg_scene.model_file
        if model_file:
            try:
                print(f"[play_interactive] Using configured visual model for viewer: {model_file}")
                return mujoco.MjModel.from_xml_path(str(model_file))
            except Exception as exc:
                print(
                    "[play_interactive] WARNING: failed to load configured visual model; "
                    f"falling back to playback model ({exc})."
                )

    try:
        playback_model = env.get_playback_model()
    except NotImplementedError as exc:
        raise AttributeError("Environment does not expose a playback model contract") from exc
    if isinstance(playback_model, str):
        print(f"[play_interactive] Using playback model for viewer: {playback_model}")
        return mujoco.MjModel.from_xml_path(playback_model)
    print("[play_interactive] Using backend playback model for viewer.")
    return playback_model


def _build_playback_config(args, *, num_envs: int = 1) -> RslRlPlaybackConfig:
    return RslRlPlaybackConfig(
        task=str(args.task),
        load_run=str(args.load_run),
        checkpoint=getattr(args, "checkpoint", None),
        action_mode=str(args.action_mode),
        policy_obs_mode=str(args.policy_obs_mode),
        algo_log_name=str(getattr(args, "algo_log_name", "rsl_rl_ppo")),
        log_root=getattr(args, "log_root", None),
        num_envs=num_envs,
        speed=float(getattr(args, "speed", 1.0)),
        start_paused=bool(getattr(args, "start_paused", False)),
    )


def _build_keyboard_commander(env: Any, args) -> KeyboardCommander | None:
    """Set up keyboard velocity teleop, or return None when unsupported/disabled."""
    if not bool(getattr(args, "keyboard", False)):
        return None

    state = getattr(env, "state", None)
    command_arr = state.info.get("commands") if state is not None else None
    cmds_cfg = getattr(getattr(env, "cfg", None), "commands", None)
    if not isinstance(command_arr, np.ndarray) or cmds_cfg is None:
        print("[play_interactive] interactive.keyboard ignored: task has no velocity 'commands'.")
        return None

    cmds_cfg.heading_command = False
    cmds_cfg.resampling_time = 0.0

    commander = KeyboardCommander.from_vel_limit(
        cmds_cfg.vel_limit,
        step_lin=float(getattr(args, "keyboard_step_lin", 0.1)),
        step_ang=float(getattr(args, "keyboard_step_ang", 0.2)),
    )
    env.state.info["commands"][:] = commander.command
    return commander


def _handle_command_key(commander: KeyboardCommander, keycode: int) -> None:
    if keycode == _KEY_UP:
        commander.nudge(commander.AXIS_VX, +1.0)
    elif keycode == _KEY_DOWN:
        commander.nudge(commander.AXIS_VX, -1.0)
    elif keycode == _KEY_LEFT:
        commander.nudge(commander.AXIS_VYAW, +1.0)
    elif keycode == _KEY_RIGHT:
        commander.nudge(commander.AXIS_VYAW, -1.0)
    elif keycode in (_KEY_ENTER, _KEY_KP_ENTER):
        commander.zero()
    else:
        return
    print(f"[play_interactive] {commander.describe()}")


def _print_keyboard_legend(args) -> None:
    print("[play_interactive] Keyboard teleop ENABLED (drive style):")
    print("  Up / Down    : forward / backward (vx)")
    print("  Left / Right : turn left / right  (vyaw)")
    print("  Enter        : full stop")
    if str(getattr(args, "action_mode", "")) != "policy":
        print("  NOTE: action_mode is not 'policy'; commands will not drive the robot.")


def play_interactive(args, cfg: DictConfig | None = None):
    device = select_torch_device()
    print(f"[play_interactive] Device: {device}")

    # Always use a single env for interactive view
    available_backends = _available_backends_for_task(args.task)
    if available_backends and "mujoco" not in available_backends:
        print(
            "[play_interactive] Task does not support MuJoCo backend: "
            f"{args.task}. Available backends: {available_backends or ('<none>',)}. "
            "This script only supports MuJoCo viewer mode."
        )
        return

    def _create_env(num_envs: int):
        if cfg is None:
            return registry.make(args.task, num_envs=num_envs, sim_backend="mujoco")
        from unilab.training import create_env

        env_cfg_override = _backend_adapter(cfg).build_task_env_cfg_override()
        try:
            return create_env(
                cfg,
                num_envs=num_envs,
                env_cfg_override=env_cfg_override,
                sim_backend="mujoco",
                task_name=args.task,
            )
        except ValueError as exc:
            if "does not support simulation backend 'mujoco'" in str(exc):
                print(
                    "[play_interactive] Task does not support MuJoCo backend: "
                    f"{args.task}. Available backends: {available_backends or ('<none>',)}. "
                    "This script only supports MuJoCo viewer mode."
                )
                raise RuntimeError(_PLAYBACK_ENV_UNAVAILABLE) from exc
            raise

    try:
        session = create_rsl_rl_playback_session(
            playback_cfg=_build_playback_config(args, num_envs=1),
            env_factory=_create_env,
            algo_config=_algo_config_dict(cfg),
            root_dir=ROOT_DIR,
            device=device,
            checkpoint_resolver=resolve_checkpoint,
            checkpoint_input_dim_reader=_infer_checkpoint_actor_input_dim,
            entrypoint_log_root=get_entrypoint_log_root,
            wrapper_cls=RslRlVecEnvWrapper,
            runner_cls=OnPolicyRunner,
            policy_obs_dims_getter=get_policy_obs_dims,
            train_cfg_normalizer=normalize_ppo_train_cfg,
            log=lambda message: print(f"[play_interactive] {message}"),
        )
    except RuntimeError as exc:
        if str(exc) == _PLAYBACK_ENV_UNAVAILABLE:
            return
        raise
    playback_session = session[0]
    env = playback_session.env

    if _uses_native_mujoco_viewer_launch() and not _can_launch_glfw_viewer():
        print(
            "[play_interactive] GLFW viewer initialization failed (no usable display). "
            "Set DISPLAY correctly, or run this command in a desktop session."
        )
        return
    overlay = prepare_motion_overlay_selection(
        env,
        show_target_bodies=bool(args.show_target_bodies),
        show_reward_debug=bool(args.show_reward_debug),
        target_body_names=str(args.target_body_names),
        target_max_bodies=int(args.target_max_bodies),
        log=lambda message: print(f"[play_interactive] {message}"),
    )

    if overlay.enabled:
        print(
            "[play_interactive] Target visualization enabled "
            f"({overlay.selected_indices.size} bodies, axes={args.target_show_axes})."
        )
    if args.show_reward_debug:
        print(
            "[play_interactive] Reward debug overlay enabled "
            f"(vel={args.reward_debug_show_velocity}, connectors={args.reward_debug_show_connectors}, "
            f"global_anchor={args.reward_debug_show_global_anchor})."
        )

    # Dedicated MjData for the viewer (never touches the rollout workers)
    use_env_visual_model = bool(getattr(args, "use_env_visual_model", True))
    mj_model = _load_viewer_model(env, use_env_visual_model=use_env_visual_model)

    viz_data = mujoco.MjData(mj_model)
    state_spec = mujoco.mjtState.mjSTATE_FULLPHYSICS
    ctrl_dt = env.cfg.ctrl_dt

    playback_session.reset()
    controls = PlaybackControls(
        paused=bool(getattr(args, "start_paused", False)),
        speed=float(getattr(args, "speed", 1.0)),
    )

    commander = _build_keyboard_commander(env, args)
    if commander is not None:
        env.set_autoreset(False)

    def _on_key(keycode: int) -> None:
        if keycode == ord(" "):
            paused = controls.toggle_pause()
            status = "paused" if paused else "resumed"
            print(f"[play_interactive] {status} (space)")
        elif keycode in (ord("N"), ord("n")):
            controls.request_single_step()
            if not controls.paused:
                controls.pause()
                print("[play_interactive] paused for single-step mode (n)")
            print("[play_interactive] single step requested (n)")
        elif keycode in (ord("+"), ord("=")):
            controls.set_speed(controls.speed * 1.25)
            print(f"[play_interactive] speed={controls.speed:.2f}x")
        elif keycode in (ord("-"), ord("_")):
            controls.set_speed(controls.speed / 1.25)
            print(f"[play_interactive] speed={controls.speed:.2f}x")
        elif commander is not None and keycode == _KEY_BACKSPACE:
            playback_session.reset()
            commander.zero()
            print("[play_interactive] reset (backspace)")
        elif commander is not None:
            _handle_command_key(commander, keycode)

    print("[play_interactive] Opening viewer — close the window or press Esc to quit.")
    print("[play_interactive] Controls: Space=pause/resume, N=single-step, +/-=speed")
    if commander is not None:
        _print_keyboard_legend(args)

    with mujoco.viewer.launch_passive(mj_model, viz_data, key_callback=_on_key) as viewer:
        focus_body_id = _resolve_focus_body_id(
            mj_model, env, getattr(args, "camera_focus_body_name", "")
        )

        # Initialize camera to a reasonable default and keep lookat on robot base.
        has_cam = hasattr(viewer, "cam")
        if has_cam:
            model_extent = float(getattr(getattr(mj_model, "stat", None), "extent", 1.0))
            default_distance = max(2.0, 2.5 * model_extent)
            if getattr(args, "camera_distance", None) is not None:
                viewer.cam.distance = float(args.camera_distance)
            else:
                viewer.cam.distance = default_distance
            if getattr(args, "camera_elevation", None) is not None:
                viewer.cam.elevation = float(args.camera_elevation)
            if getattr(args, "camera_azimuth", None) is not None:
                viewer.cam.azimuth = float(args.camera_azimuth)

        with torch.inference_mode():
            while viewer.is_running():
                t0 = time.perf_counter()

                # Write the command before stepping so this step's obs follow it.
                if commander is not None and env.state is not None:
                    env.state.info["commands"][:] = commander.command

                playback_session.advance(controls)

                # Push env state[0] into viz_data and refresh scene
                phys = playback_session.physics_state()[0].astype(np.float64)
                mujoco.mj_setState(mj_model, viz_data, phys, state_spec)
                mujoco.mj_forward(mj_model, viz_data)

                if has_cam and bool(getattr(args, "camera_follow_body", True)):
                    base_pos = viz_data.xpos[focus_body_id]
                    viewer.cam.lookat[0] = float(base_pos[0])
                    viewer.cam.lookat[1] = float(base_pos[1])
                    viewer.cam.lookat[2] = float(
                        base_pos[2] + float(getattr(args, "camera_height_offset", 0.15))
                    )

                if overlay.enabled:
                    if args.show_reward_debug:
                        _render_reward_debug_targets(
                            viewer,
                            playback_session.info,
                            overlay.selected_indices,
                            marker_radius=args.target_marker_radius,
                            marker_alpha=args.target_marker_alpha,
                            show_axes=args.target_show_axes,
                            axis_length=args.target_axis_length,
                            show_vel=args.reward_debug_show_velocity,
                            lin_vel_scale=args.reward_debug_lin_vel_scale,
                            ang_vel_scale=args.reward_debug_ang_vel_scale,
                            show_connectors=args.reward_debug_show_connectors,
                            show_global_anchor=args.reward_debug_show_global_anchor,
                        )
                    else:
                        motion_data = env.state.info.get("motion_data", None)
                        _render_motion_targets(
                            viewer,
                            motion_data,
                            overlay.selected_indices,
                            marker_radius=args.target_marker_radius,
                            marker_alpha=args.target_marker_alpha,
                            show_axes=args.target_show_axes,
                            axis_length=args.target_axis_length,
                        )
                else:
                    viewer.user_scn.ngeom = 0

                viewer.sync()

                # Real-time pacing
                elapsed = time.perf_counter() - t0
                target_dt = controls.target_dt(ctrl_dt)
                if target_dt - elapsed > 0:
                    time.sleep(target_dt - elapsed)

    print("[play_interactive] Done.")


def _normalize_checkpoint_value(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    return None if text in {"-1", "None", "null"} else text


def _build_play_args(cfg: DictConfig) -> PlayInteractiveArgs:
    return PlayInteractiveArgs(
        task=str(cfg.training.task_name),
        load_run=str(cfg.algo.load_run),
        checkpoint=_normalize_checkpoint_value(OmegaConf.select(cfg, "algo.checkpoint")),
        action_mode=str(cfg.interactive.action_mode),
        policy_obs_mode=str(cfg.interactive.policy_obs_mode),
        algo_log_name=str(cfg.algo.algo_log_name),
        log_root=(
            str(cfg.training.log_root)
            if OmegaConf.select(cfg, "training.log_root") is not None
            else None
        ),
        show_target_bodies=bool(cfg.interactive.show_target_bodies),
        show_reward_debug=bool(cfg.interactive.show_reward_debug),
        target_show_axes=bool(cfg.interactive.target_show_axes),
        target_body_names=str(cfg.interactive.target_body_names),
        target_max_bodies=int(cfg.interactive.target_max_bodies),
        target_marker_radius=float(cfg.interactive.target_marker_radius),
        target_axis_length=float(cfg.interactive.target_axis_length),
        target_marker_alpha=float(cfg.interactive.target_marker_alpha),
        reward_debug_show_velocity=bool(cfg.interactive.reward_debug_show_velocity),
        reward_debug_lin_vel_scale=float(cfg.interactive.reward_debug_lin_vel_scale),
        reward_debug_ang_vel_scale=float(cfg.interactive.reward_debug_ang_vel_scale),
        reward_debug_show_connectors=bool(cfg.interactive.reward_debug_show_connectors),
        reward_debug_show_global_anchor=bool(cfg.interactive.reward_debug_show_global_anchor),
        camera_follow_body=bool(cfg.interactive.camera_follow_body),
        camera_focus_body_name=str(cfg.interactive.camera_focus_body_name),
        camera_height_offset=float(cfg.interactive.camera_height_offset),
        camera_distance=(
            float(cfg.interactive.camera_distance)
            if OmegaConf.select(cfg, "interactive.camera_distance") is not None
            else None
        ),
        camera_elevation=(
            float(cfg.interactive.camera_elevation)
            if OmegaConf.select(cfg, "interactive.camera_elevation") is not None
            else None
        ),
        camera_azimuth=(
            float(cfg.interactive.camera_azimuth)
            if OmegaConf.select(cfg, "interactive.camera_azimuth") is not None
            else None
        ),
        use_env_visual_model=bool(cfg.interactive.use_env_visual_model),
        speed=float(OmegaConf.select(cfg, "interactive.speed", default=1.0)),
        start_paused=bool(OmegaConf.select(cfg, "interactive.start_paused", default=False)),
        keyboard=bool(OmegaConf.select(cfg, "interactive.keyboard", default=False)),
        keyboard_step_lin=float(
            OmegaConf.select(cfg, "interactive.keyboard_step_lin", default=0.1)
        ),
        keyboard_step_ang=float(
            OmegaConf.select(cfg, "interactive.keyboard_step_ang", default=0.2)
        ),
    )


@hydra.main(version_base="1.3", config_path="../conf/ppo", config_name="config")
def main(cfg: DictConfig) -> None:
    if str(cfg.training.sim_backend) != "mujoco":
        raise ValueError("play_interactive.py only supports MuJoCo viewer; use task=<task>/mujoco.")
    play_interactive(_build_play_args(cfg), cfg)


if __name__ == "__main__":
    main()
