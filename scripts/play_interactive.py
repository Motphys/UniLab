"""Interactive play script: opens a live MuJoCo viewer for a trained RSL-RL policy.

Usage:
    # Load the latest checkpoint for a task
    python scripts/play_interactive.py --task Go2JoystickFlatTerrain

    # Load a specific run
    python scripts/play_interactive.py --task Go2JoystickFlatTerrain --load_run 2024-02-04_12-00-00

Camera controls (MuJoCo viewer):
    Mouse drag     - rotate
    Scroll         - zoom
    Right-drag     - pan
"""

import argparse
import importlib
import os
import pkgutil
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import torch

ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR))


def ensure_registries():
    for pkg_name in (
        "unilab.envs.locomotion",
        "unilab.envs.manipulation",
        "unilab.envs.motion_tracking",
    ):
        try:
            package = importlib.import_module(pkg_name)
            if hasattr(package, "__path__"):
                for _, name, _ in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
                    try:
                        importlib.import_module(name)
                    except Exception:
                        pass
        except ImportError:
            pass


ensure_registries()

from unilab.base import registry
from unilab.config.structured_configs import PPOConfig
from unilab.utils.obs_utils import flatten_obs_dict
from unilab.utils.rsl_rl_compat import convert_config_v3_to_v4, is_rsl_rl_v4
from unilab.utils.run_utils import get_latest_run
from unilab.utils.torch_utils import to_torch

try:
    from rsl_rl.runners import OnPolicyRunner
except ImportError:
    print("Could not import rsl_rl. Please ensure it is installed.")
    sys.exit(1)

from tensordict import TensorDict

# ---------------------------------------------------------------------------
# Minimal env wrapper (mirrors RslRlVecEnvWrapper in train_rsl_rl.py)
# ---------------------------------------------------------------------------


class RslRlVecEnvWrapper:
    def __init__(self, env, device="cuda", policy_obs_mode: str = "flat"):
        self.env = env
        self.cfg = env.cfg
        self.device = device
        self.policy_obs_mode = policy_obs_mode
        self.num_envs = env.num_envs
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self._actor_obs_dim = int(env.obs_groups_spec.get("obs", sum(env.obs_groups_spec.values())))
        self._flat_obs_dim = int(sum(env.obs_groups_spec.values()))
        self.num_obs = self._flat_obs_dim if policy_obs_mode == "flat" else self._actor_obs_dim
        self.num_privileged_obs = self.num_obs
        self.num_actions = env.action_space.shape[0]
        self.episode_returns = torch.zeros(self.num_envs, device=device)
        self.episode_lengths = torch.zeros(self.num_envs, device=device)
        self.episode_length_buf = self.episode_lengths
        self.max_episode_length = int(env.cfg.max_episode_seconds / env.cfg.ctrl_dt)
        self.reset()

    def _obs_to_tensordict(self, obs: dict[str, np.ndarray]) -> TensorDict:
        actor = to_torch(obs["obs"], self.device)
        if self.policy_obs_mode == "actor":
            policy = actor
        else:
            policy = to_torch(flatten_obs_dict(obs), self.device)
        td = {"policy": policy, "actor": actor}
        if "privileged" in obs:
            td["privileged"] = to_torch(obs["privileged"], self.device)
        return TensorDict(td, batch_size=self.num_envs, device=self.device)

    def step(self, actions):
        actions_np = (
            actions.detach().cpu().numpy() if isinstance(actions, torch.Tensor) else actions
        )
        state = self.env.step(actions_np)
        rewards = to_torch(state.reward, self.device)
        dones = to_torch(state.done, self.device).bool()
        self.episode_returns += rewards
        self.episode_lengths += 1
        infos = {}
        done_idx = torch.nonzero(dones).flatten()
        if len(done_idx) > 0:
            if hasattr(state, "truncated"):
                infos["time_outs"] = to_torch(state.truncated, self.device).bool()
            self.episode_returns[done_idx] = 0
            self.episode_lengths[done_idx] = 0
        if hasattr(state, "info") and "log" in state.info:
            infos["log"] = state.info["log"]
        return self._obs_to_tensordict(state.obs), rewards, dones, infos

    def reset(self):
        if self.env.state is None:
            self.env.init_state()
        env_indices = np.arange(self.num_envs, dtype=np.int32)
        reset_out = self.env.reset(env_indices)
        if isinstance(reset_out, tuple) and len(reset_out) == 2:
            obs_out, _ = reset_out
        elif isinstance(reset_out, tuple) and len(reset_out) == 3:
            _, obs_out, _ = reset_out
        else:
            raise ValueError(
                "Unexpected env.reset return format. Expected (obs, info) or (_, obs, _)."
            )
        self.episode_returns[:] = 0
        self.episode_lengths[:] = 0
        return self._obs_to_tensordict(obs_out), {}

    def get_observations(self):
        return self._obs_to_tensordict(self.env.state.obs)

    def get_privileged_observations(self):
        if self.policy_obs_mode == "actor":
            return to_torch(self.env.state.obs["obs"], self.device)
        return to_torch(flatten_obs_dict(self.env.state.obs), self.device)


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


# ---------------------------------------------------------------------------
# Checkpoint resolution helpers
# ---------------------------------------------------------------------------


def resolve_checkpoint(task: str, load_run: str, checkpoint: str | None = None) -> str | None:
    base = ROOT_DIR / "logs" / "rsl_rl_train" / task
    if load_run == "-1":
        path = get_latest_run(str(base))
    elif os.path.exists(load_run):
        path = load_run
    else:
        path = str(base / load_run)

    if not path or not os.path.exists(path):
        print(f"[play_interactive] Run not found: {path}")
        return None

    if os.path.isdir(path):
        if checkpoint is not None:
            if str(checkpoint).isdigit():
                model_name = f"model_{checkpoint}.pt"
            else:
                model_name = checkpoint
            model_path = os.path.join(path, model_name)
            if os.path.exists(model_path):
                path = model_path
            else:
                print(f"[play_interactive] Checkpoint not found: {model_path}")
                return None
        else:
            model_files = sorted(
                [f for f in os.listdir(path) if f.startswith("model_") and f.endswith(".pt")],
                key=lambda f: int(f.split("_")[1].split(".")[0]),
            )
            if not model_files:
                print(f"[play_interactive] No model_*.pt files in {path}")
                return None
            path = os.path.join(path, model_files[-1])

    print(f"[play_interactive] Loading checkpoint: {path}")
    return path


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


def play_interactive(args):
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"[play_interactive] Device: {device}")

    # Always use a single env for interactive view
    env = registry.make(args.task, num_envs=1, sim_backend="mujoco")
    actor_obs_dim = int(env.obs_groups_spec.get("obs", sum(env.obs_groups_spec.values())))
    flat_obs_dim = int(sum(env.obs_groups_spec.values()))

    policy_obs_mode = args.policy_obs_mode
    ckpt = None
    if args.action_mode == "policy":
        ckpt = resolve_checkpoint(args.task, args.load_run, getattr(args, "checkpoint", None))
        if policy_obs_mode == "auto" and ckpt is not None:
            ckpt_dim = _infer_checkpoint_actor_input_dim(ckpt)
            if ckpt_dim == actor_obs_dim:
                policy_obs_mode = "actor"
            elif ckpt_dim == flat_obs_dim:
                policy_obs_mode = "flat"
            elif ckpt_dim is not None:
                raise RuntimeError(
                    "Checkpoint actor input dim mismatch: "
                    f"ckpt={ckpt_dim}, actor_obs={actor_obs_dim}, flat_obs={flat_obs_dim}. "
                    "Please pass --policy_obs_mode actor|flat explicitly if needed."
                )
            else:
                # Default fallback keeps current behavior.
                policy_obs_mode = "flat"

    wrapped_env = RslRlVecEnvWrapper(env, device=device, policy_obs_mode=policy_obs_mode)
    print(
        "[play_interactive] Policy obs mode: "
        f"{policy_obs_mode} (actor_obs={actor_obs_dim}, flat_obs={flat_obs_dim})"
    )

    cfg = PPOConfig()
    train_cfg = cfg.to_dict()
    if is_rsl_rl_v4():
        train_cfg = convert_config_v3_to_v4(train_cfg)

    policy = None
    if args.action_mode == "policy":
        if ckpt is None:
            print("[play_interactive] WARNING: no checkpoint found — falling back to zero actions.")
        else:
            log_dir = str(ROOT_DIR / "logs" / "rsl_rl_train" / args.task / "play_temp")
            runner = OnPolicyRunner(wrapped_env, train_cfg, log_dir=log_dir, device=device)
            runner.load(
                ckpt,
                load_cfg={
                    "actor": True,
                    "critic": False,
                    "optimizer": False,
                    "iteration": False,
                    "rnd": False,
                },
            )
            policy = runner.get_inference_policy(device=device)

    print(f"[play_interactive] Action mode: {args.action_mode}")

    target_vis_enabled = False
    selected_indices = np.zeros((0,), dtype=np.int32)
    if args.show_target_bodies or args.show_reward_debug:
        if hasattr(env, "motion_loader") and hasattr(env, "motion_sampler"):
            names = tuple(getattr(env.cfg, "body_names", ()))
            if len(names) > 0:
                name_to_idx = {name: i for i, name in enumerate(names)}
                if args.target_body_names.strip():
                    chosen = []
                    for name in [n.strip() for n in args.target_body_names.split(",") if n.strip()]:
                        if name in name_to_idx:
                            chosen.append(name_to_idx[name])
                        else:
                            print(
                                f"[play_interactive] WARNING: body name not found in task body list: {name}"
                            )
                    selected_indices = np.array(chosen, dtype=np.int32)
                else:
                    selected_indices = np.arange(len(names), dtype=np.int32)

                if args.target_max_bodies > 0:
                    selected_indices = selected_indices[: args.target_max_bodies]

                target_vis_enabled = selected_indices.size > 0
            else:
                print(
                    "[play_interactive] WARNING: task has no body_names; cannot visualize targets."
                )
        else:
            print(
                "[play_interactive] WARNING: target/reward visualization only works for motion-tracking tasks."
            )

    if target_vis_enabled:
        print(
            "[play_interactive] Target visualization enabled "
            f"({selected_indices.size} bodies, axes={args.target_show_axes})."
        )
    if args.show_reward_debug:
        print(
            "[play_interactive] Reward debug overlay enabled "
            f"(vel={args.reward_debug_show_velocity}, connectors={args.reward_debug_show_connectors}, "
            f"global_anchor={args.reward_debug_show_global_anchor})."
        )

    # Dedicated MjData for the viewer (never touches the rollout workers)
    if hasattr(env, "_backend") and hasattr(env._backend, "model"):
        mj_model = env._backend.model
    else:
        raise AttributeError(
            "Environment backend does not expose a MuJoCo model via env._backend.model"
        )
    viz_data = mujoco.MjData(mj_model)
    state_spec = mujoco.mjtState.mjSTATE_FULLPHYSICS
    ctrl_dt = env.cfg.ctrl_dt

    obs, _ = wrapped_env.reset()
    paused_state = {"paused": False}

    def _on_key(keycode: int) -> None:
        if keycode == ord(" "):
            paused_state["paused"] = not paused_state["paused"]
            status = "paused" if paused_state["paused"] else "resumed"
            print(f"[play_interactive] {status} (space)")

    print("[play_interactive] Opening viewer — close the window or press Esc to quit.")
    print("[play_interactive] Controls: Space = pause/resume")

    # Get action bounds for random mode
    action_low = env.action_space.low
    action_high = env.action_space.high

    with mujoco.viewer.launch_passive(mj_model, viz_data, key_callback=_on_key) as viewer:
        with torch.inference_mode():
            while viewer.is_running():
                t0 = time.perf_counter()

                if not paused_state["paused"]:
                    if args.action_mode == "policy" and policy is not None:
                        actions = policy(obs)
                    elif args.action_mode == "random":
                        actions = (
                            torch.from_numpy(
                                np.random.uniform(
                                    action_low, action_high, size=(1, env.action_space.shape[0])
                                )
                            )
                            .to(device)
                            .float()
                        )
                    else:  # zero
                        actions = torch.zeros(1, env.action_space.shape[0], device=device)

                    obs, _, _, _ = wrapped_env.step(actions)

                # Push env state[0] into viz_data and refresh scene
                phys = env._backend.get_physics_state()[0].astype(np.float64)
                mujoco.mj_setState(mj_model, viz_data, phys, state_spec)
                mujoco.mj_forward(mj_model, viz_data)

                if target_vis_enabled:
                    if args.show_reward_debug:
                        _render_reward_debug_targets(
                            viewer,
                            env.state.info,
                            selected_indices,
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
                            selected_indices,
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
                if ctrl_dt - elapsed > 0:
                    time.sleep(ctrl_dt - elapsed)

    print("[play_interactive] Done.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Interactive MuJoCo viewer for a trained RSL-RL policy"
    )
    parser.add_argument(
        "--task", type=str, required=True, help="Task name, e.g. Go2JoystickFlatTerrain"
    )
    parser.add_argument(
        "--load_run", type=str, default="-1", help="Run timestamp or path to load (-1 = latest)"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Specific model checkpoint number or file name to load (e.g., '1000' or 'model_1000.pt')",
    )
    parser.add_argument(
        "--action_mode",
        type=str,
        default="policy",
        choices=["policy", "zero", "random"],
        help="Action mode: policy (load ckpt), zero, or random",
    )
    parser.add_argument(
        "--policy_obs_mode",
        type=str,
        default="auto",
        choices=["auto", "flat", "actor"],
        help="Policy observation mode for checkpoint compatibility",
    )
    parser.add_argument(
        "--show_target_bodies",
        action="store_true",
        help="Visualize motion target body positions for motion-tracking tasks",
    )
    parser.add_argument(
        "--show_reward_debug",
        action="store_true",
        help="Visualize reward-used references: transformed body pose, raw motion pose/vel, robot pose/vel",
    )
    parser.add_argument(
        "--target_show_axes",
        action="store_true",
        help="Also draw orientation axes for target bodies",
    )
    parser.add_argument(
        "--target_body_names",
        type=str,
        default="",
        help="Comma-separated body names to visualize (default: all task body_names)",
    )
    parser.add_argument(
        "--target_max_bodies",
        type=int,
        default=0,
        help="Limit number of target bodies to draw (0 means no limit)",
    )
    parser.add_argument(
        "--target_marker_radius",
        type=float,
        default=0.02,
        help="Sphere marker radius in meters",
    )
    parser.add_argument(
        "--target_axis_length",
        type=float,
        default=0.08,
        help="Axis arrow length in meters when --target_show_axes is enabled",
    )
    parser.add_argument(
        "--target_marker_alpha",
        type=float,
        default=0.75,
        help="Target marker alpha in [0, 1]",
    )
    parser.add_argument(
        "--reward_debug_show_velocity",
        action="store_true",
        help="When --show_reward_debug is set, draw linear/angular velocity vectors",
    )
    parser.add_argument(
        "--reward_debug_lin_vel_scale",
        type=float,
        default=0.08,
        help="Meters per (m/s) for linear velocity arrows in reward debug mode",
    )
    parser.add_argument(
        "--reward_debug_ang_vel_scale",
        type=float,
        default=0.05,
        help="Meters per (rad/s) for angular velocity arrows in reward debug mode",
    )
    parser.add_argument(
        "--reward_debug_show_connectors",
        action="store_true",
        help="When --show_reward_debug is set, draw connector arrows from reward reference pose to robot pose",
    )
    parser.add_argument(
        "--reward_debug_show_global_anchor",
        action="store_true",
        help="When --show_reward_debug is set, emphasize global anchor (motion vs robot) used by root rewards",
    )
    args = parser.parse_args()
    play_interactive(args)


if __name__ == "__main__":
    main()
